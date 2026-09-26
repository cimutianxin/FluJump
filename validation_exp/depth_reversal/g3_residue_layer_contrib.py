#!/usr/bin/env python3
"""G3：残基×层贡献 → MAFFT 列坐标系聚合（H5 位点读出转移）

对全 31 层 × 双标签，逐残基贡献 c_i = ((h_i − m_l)/s_l)·w_l（w 来自 g1_probes.npz），
一次前向（output_hidden_states）在线计算并按 col_positions 聚合到
{label × layer × group × 1039 列} 的 sum/cnt（逐残基大图不落盘，RAM 占用仅聚合数组）。
组：H1H3± / H5± / H7±（全部主数据行，同 reversal_sites d_pos 约定）。

派生剖面：
  sep_l(col)  = mean_c(H1H3+) − mean_c(H1H3−)   —— w 在 train 任务上的逐列读出
  dposH7_l(col) = mean_c(H7+) − mean_c(H1H3+)   —— H7 推负逐列剖面（H5 对照）
读出指数：func_frac / diag_frac（功能位点集 = site_attribution MARKER+人适应位点；
诊断集 = g4_column_scores js_div 前 5% 列；valid 列掩膜同 g4）。
判定 H5：mid(13–22) vs late(28–30) 的 func_frac 差，10k 列标签置换双侧 p；
Fisher 富集（top-20 |sep| × 位点集，BH）；与 reversal_sites 驱动列交叉核对。

sanity：逐行 mean_i c_i ≈ z − b（mean-pool logit 对账，前 200 行断言）。

运行：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env1 python（GPU）
输出：output/g3_col_profiles.npz、output/g3_site_readout.json、figs/g3_heatmap_{label}.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

sys.path.insert(0, ".")
from validation_exp.depth_reversal.config import (
    SPLIT_CSV, ALIGNED_CSV, ALL_LAYERS_NPY, LABELS, COL_TO_H3_JSON,
    G4_COLUMN_SCORES, FUNCTIONAL_SITES, DIAG_JS_QUANTILE, MID_LAYERS,
    LATE_LAYERS, ESM_MODEL, EXTRACT_BATCH, ALN_LEN, N_LAYERS, RANDOM_SEED,
    LABEL_COLS, OUT_DIR,
)

N_PERM = 10000
TOPK = 20
SANITY_N = 200
DRIVER_SETS_JSON = "validation_exp/reversal_sites/output/g4_shared_driver_sets.json"


def col_positions(aln_seq):
    return np.fromiter((i for i, c in enumerate(aln_seq) if c != "-"),
                       dtype=np.int32, count=len(aln_seq) - aln_seq.count("-"))


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1))
        q[order[i]] = prev
    return q


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figs").mkdir(exist_ok=True)

    # ── 行框架（split 行序）──
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV,
                      usecols=["accession", "subtype", "ha_sequence", "aligned_ha_seq"])
    df = split.merge(ali, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    assert df["aligned_ha_seq"].notna().all() and df["ha_sequence"].notna().all()
    n = len(df)
    subtype = df["subtype"].to_numpy()
    seqs = df["ha_sequence"].tolist()
    col_maps = [col_positions(s) for s in df["aligned_ha_seq"]]
    m_h13 = np.isin(subtype, ["H1", "H3"])
    m_h5 = subtype == "H5"
    m_h7 = subtype == "H7"

    # ── probe 参数 → GPU 张量（label × layer × 640）──
    probes = np.load(OUT_DIR / "g1_probes.npz")
    device = torch.device("cuda")
    W = {}
    for label in LABEL_COLS:
        W[label] = {
            k: torch.tensor(
                np.stack([probes[f"{k}_L{li}_{label}"] for li in range(N_LAYERS)]),
                device=device)
            for k in ["w", "mean", "scale"]
        }
        # config 里键名是 w/mean/scale/intercept
    ys = {c: np.load(LABELS[c]) for c in LABEL_COLS}

    # ── 聚合数组 ──
    grp_defs = {  # label -> {grp: mask}
        c: {"H1H3+": m_h13 & (ys[c] == 1), "H1H3-": m_h13 & (ys[c] == 0),
            "H5+": m_h5 & (ys[c] == 1), "H5-": m_h5 & (ys[c] == 0),
            "H7+": m_h7 & (ys[c] == 1), "H7-": m_h7 & (ys[c] == 0)}
        for c in LABEL_COLS
    }
    grp_names = ["H1H3+", "H1H3-", "H5+", "H5-", "H7+", "H7-"]
    csum = {c: np.zeros((N_LAYERS, len(grp_names), ALN_LEN), dtype=np.float64)
            for c in LABEL_COLS}

    # ── sanity 缓冲（mean-pool 对账）──
    pooled = np.load(ALL_LAYERS_NPY, mmap_mode="r")
    sanity = {c: [] for c in LABEL_COLS}

    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True)
    model = model.to(device).eval()

    for i0 in tqdm(range(0, n, EXTRACT_BATCH), desc="残基×层贡献"):
        batch = seqs[i0:i0 + EXTRACT_BATCH]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        H = torch.stack(out.hidden_states)          # (31,B,Lmax,640)
        for label in LABEL_COLS:
            Hs = (H - W[label]["mean"].view(31, 1, 1, 640)) \
                 / W[label]["scale"].view(31, 1, 1, 640)   # (31,B,Lmax,640)
            c = torch.einsum("lbid,ld->lbi", Hs, W[label]["w"])
            c = c.cpu().numpy()                           # (31,B,Lmax)
            for j in range(len(batch)):
                r = i0 + j
                L = len(seqs[r])
                pos = col_maps[r]
                assert len(pos) == L
                for gi, g in enumerate(grp_names):
                    if not grp_defs[label][g][r]:
                        continue
                    for li in range(N_LAYERS):
                        np.add.at(csum[label][li, gi], pos, c[li, j, 1:1 + L])
                # sanity：(a) einsum 自洽（mean_i c_i vs mean-pool 同前向恒等式）
                #       (b) 与旧 npy logit 对账（防行错位；旧 npy EOS 包含不
                #           对称——reversal_sites G2 已记录的 off-by-one 类，
                #           故只报不准硬阈值）
                if r < SANITY_N:
                    mh = H[:, j, 1:1 + L].mean(dim=1).cpu().numpy()  # (31,640)
                    for li in [13, 17, 28, 30]:
                        m = probes[f"mean_L{li}_{label}"]
                        s = probes[f"scale_L{li}_{label}"]
                        w = probes[f"w_L{li}_{label}"]
                        z2 = float(((mh[li] - m) / s) @ w)
                        z_npy = float(((np.asarray(pooled[li, r]) - m) / s) @ w)
                        sanity[label].append(
                            (li, float(c[li, j, 1:1 + L].mean()), z2, z_npy))
    # 计数矩阵需逐 label（± 组不同）：用成员掩膜 × 列覆盖重算
    cnt = {}
    for label in LABEL_COLS:
        cnt[label] = {}
        for gi, g in enumerate(grp_names):
            msk = grp_defs[label][g]
            cg = np.zeros(ALN_LEN, dtype=np.int64)
            for r in np.where(msk)[0]:
                np.add.at(cg, col_maps[r], 1)
            cnt[label][g] = cg

    # sanity 断言
    for label in LABEL_COLS:
        arr_s = np.array(sanity[label])             # (SANITY_N×4, 4)
        for li in [13, 17, 28, 30]:
            sub = arr_s[arr_s[:, 0] == li]
            err_self = np.abs(sub[:, 1] - sub[:, 2]).max()
            assert err_self < 1e-2, \
                f"einsum 自洽失败 {label} L{li}: {err_self}"
            err_npy = np.abs(sub[:, 1] - sub[:, 3])
            med, mx = float(np.median(err_npy)), float(err_npy.max())
            # 行对齐以中位数判定（max 受 EOS 位贡献影响，L30 可达 ~1）
            assert med < 0.1, f"行错位疑似 {label} L{li}: med={med}"
            print(f"  sanity[{label} L{li}] 自洽 err={err_self:.2e} | "
                  f"npy 对账 med={med:.3f} max={mx:.3f}（EOS 不对称，记录用）")
    print("sanity 通过：einsum 自洽 + 无行错位")

    # ── 均值剖面（count>0 列）──
    mean_c = {c: np.full((N_LAYERS, len(grp_names), ALN_LEN), np.nan)
              for c in LABEL_COLS}
    for label in LABEL_COLS:
        for gi, g in enumerate(grp_names):
            cg = cnt[label][g]
            for li in range(N_LAYERS):
                mean_c[label][li, gi] = np.where(
                    cg > 0, csum[label][li, gi] / np.maximum(cg, 1), np.nan)

    # ── 位点集 ──
    mp = json.load(open(COL_TO_H3_JSON))
    inv_h3 = {}
    for col, num in mp["col_to_h3_ha1"].items():
        inv_h3.setdefault(int(num), []).append(int(col))
    for col, num in mp["col_to_h3_ha2"].items():
        inv_h3.setdefault(f"HA2_{num}", []).append(int(col))
    func_cols = set()
    for nums in FUNCTIONAL_SITES.values():
        for num in nums:
            func_cols.update(inv_h3.get(num, []))
    g4cs = pd.read_csv(G4_COLUMN_SCORES)
    valid_mask = g4cs["valid"].to_numpy(dtype=bool)
    js = g4cs["js_div"].to_numpy(dtype=float)
    js_valid = js[valid_mask & ~np.isnan(js)]
    thr = np.quantile(js_valid, DIAG_JS_QUANTILE)
    diag_cols = set(g4cs.index[(valid_mask & (js >= thr))].tolist())
    func_cols &= set(np.where(valid_mask)[0])
    print(f"功能列 {len(func_cols)}，诊断列 {len(diag_cols)}（js≥{thr:.4f}），"
          f"valid 列 {valid_mask.sum()}")

    # ── 读出指数 + 判定 ──
    rng = np.random.default_rng(RANDOM_SEED)
    results = {"site_sets": {"func_cols": sorted(func_cols),
                             "diag_cols": sorted(diag_cols),
                             "js_quantile_thr": float(thr)}}
    fisher_tests = []
    for label in LABEL_COLS:
        gi = {g: grp_names.index(g) for g in grp_names}
        sep = mean_c[label][:, gi["H1H3+"], :] - mean_c[label][:, gi["H1H3-"], :]
        dpos_h7 = mean_c[label][:, gi["H7+"], :] - mean_c[label][:, gi["H1H3+"], :]
        dpos_h5 = mean_c[label][:, gi["H5+"], :] - mean_c[label][:, gi["H1H3+"], :]
        # 亚型内 gap 逐列剖面（AUC 的决定量；reversal_sites 未分解过）
        dw_h7 = mean_c[label][:, gi["H7+"], :] - mean_c[label][:, gi["H7-"], :]
        dw_h5 = mean_c[label][:, gi["H5+"], :] - mean_c[label][:, gi["H5-"], :]
        np.savez_compressed(
            OUT_DIR / f"g3_dwithin_{label}.npz",
            dw_h7=dw_h7.astype(np.float32), dw_h5=dw_h5.astype(np.float32))

        vcols = np.where(valid_mask)[0]
        func_arr = np.array(sorted(func_cols))
        diag_arr = np.array(sorted(diag_cols))

        func_frac = np.array([
            np.nansum(np.abs(sep[li, func_arr]))
            / np.nansum(np.abs(sep[li, vcols])) for li in range(N_LAYERS)])
        diag_frac = np.array([
            np.nansum(np.abs(sep[li, diag_arr]))
            / np.nansum(np.abs(sep[li, vcols])) for li in range(N_LAYERS)])

        # 置换：列集合成员打乱（保持集合大小），统计 mid−late 的 func_frac 差
        delta_real = float(func_frac[MID_LAYERS].mean()
                           - func_frac[LATE_LAYERS].mean())
        delta_diag = float(diag_frac[MID_LAYERS].mean()
                           - diag_frac[LATE_LAYERS].mean())
        null_d, null_dd = [], []
        vset = np.array(vcols)
        for _ in range(N_PERM):
            perm = rng.permutation(vset)
            fset = perm[:len(func_arr)]
            dset = perm[:len(diag_arr)]
            ff = np.array([np.nansum(np.abs(sep[li, fset]))
                           / np.nansum(np.abs(sep[li, vset]))
                           for li in range(N_LAYERS)])
            dd = np.array([np.nansum(np.abs(sep[li, dset]))
                           / np.nansum(np.abs(sep[li, vset]))
                           for li in range(N_LAYERS)])
            null_d.append(ff[MID_LAYERS].mean() - ff[LATE_LAYERS].mean())
            null_dd.append(dd[MID_LAYERS].mean() - dd[LATE_LAYERS].mean())
        p_func = float((np.abs(null_d) >= abs(delta_real)).mean())
        p_diag = float((np.abs(null_dd) >= abs(delta_diag)).mean())

        # Fisher：分段 top-20 |sep| × 位点集
        for seg_name, seg in [("mid", MID_LAYERS), ("late", LATE_LAYERS)]:
            sep_seg = np.nanmean(np.abs(sep[seg]), axis=0)
            top = set(vcols[np.argsort(-sep_seg[vcols])[:TOPK]])
            for set_name, sset in [("func", func_cols), ("diag", diag_cols)]:
                a = len(top & sset)
                bcell = len(top) - a
                cc_ = len(sset) - a
                d_ = len(vcols) - a - bcell - cc_
                odds, p = fisher_exact([[a, bcell], [cc_, d_]])
                fisher_tests.append({"label": label, "segment": seg_name,
                                     "set": set_name, "overlap": a,
                                     "odds": float(odds) if odds else None,
                                     "p": float(p)})

        # reversal_sites 驱动列交叉核对（late 段 dpos_h7 最负 top-20）
        drivers = json.load(open(DRIVER_SETS_JSON))
        xcheck = {}
        for key in [f"{label}_L28", f"{label}_L17", f"{label}_L13", "core"]:
            if key not in drivers:
                continue
            dset = set(drivers[key])
            for seg_name, seg in [("mid", MID_LAYERS), ("late", LATE_LAYERS)]:
                d_seg = np.nanmean(dpos_h7[seg], axis=0)
                bot = set(vcols[np.argsort(d_seg[vcols])[:TOPK]])
                xcheck[f"{key}_vs_dposH7_{seg_name}_bottom20"] = {
                    "overlap": len(bot & dset), "driver_size": len(dset)}

        # ── 亚型内 gap 翻转驱动列：d_flip = late(28–30) − mid(13–22) 的 dw_h7 ──
        d_flip = np.nanmean(dw_h7[LATE_LAYERS], axis=0) \
            - np.nanmean(dw_h7[MID_LAYERS], axis=0)
        bot_flip = set(vcols[np.argsort(d_flip[vcols])[:TOPK]])   # 最负 = 翻负驱动
        for set_name, sset in [("func", func_cols), ("diag", diag_cols)]:
            a = len(bot_flip & sset)
            bcell = len(bot_flip) - a
            cc_ = len(sset) - a
            d_ = len(vcols) - a - bcell - cc_
            odds, p = fisher_exact([[a, bcell], [cc_, d_]])
            fisher_tests.append({"label": label, "segment": "flip_late_minus_mid",
                                 "set": set_name, "overlap": a,
                                 "odds": float(odds) if odds else None,
                                 "p": float(p)})
        for key in [f"{label}_L28", "core"]:
            if key in drivers:
                xcheck[f"{key}_vs_dflip_bottom20"] = {
                    "overlap": len(bot_flip & set(drivers[key])),
                    "driver_size": len(drivers[key])}
        # d_flip 最负 20 列的 canonical 编号（备查）
        mp_h3 = g4cs["h3"].tolist()
        flip_top_cols = sorted(bot_flip, key=lambda c: d_flip[c])
        flip_top_annot = [
            {"col": int(c), "h3": (mp_h3[c] if pd.notna(mp_h3[c]) else None),
             "d_flip": float(d_flip[c])} for c in flip_top_cols]
        qs = bh_adjust([t["p"] for t in fisher_tests])
        for t, q in zip(fisher_tests, qs):
            t["q_bh"] = float(q)

        results[label] = {
            "func_frac_per_layer": func_frac.tolist(),
            "diag_frac_per_layer": diag_frac.tolist(),
            "delta_func_mid_minus_late": delta_real,
            "perm_p_func": p_func,
            "delta_diag_mid_minus_late": delta_diag,
            "perm_p_diag": p_diag,
            "driver_xcheck": xcheck,
            "dflip_top20_cols": flip_top_annot,
        }

        # 热图（valid 列）
        fig, ax = plt.subplots(figsize=(11, 4))
        mat = sep[:, vcols]
        vmax = np.nanpercentile(np.abs(mat), 99)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        ax.set_xlabel("alignment column (valid)")
        ax.set_ylabel("layer")
        ax.set_title(f"G3 sep 剖面（train+ − train−）— {label}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"figs/g3_heatmap_{label}.png", dpi=300)
        plt.close(fig)

    results["fisher_tests"] = fisher_tests
    np.savez_compressed(
        OUT_DIR / "g3_col_profiles.npz",
        **{f"mean_c_{c}": mean_c[c].astype(np.float32) for c in LABEL_COLS},
        grp_names=np.array(grp_names), valid_mask=valid_mask)
    with open(OUT_DIR / "g3_site_readout.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"→ {OUT_DIR}/g3_site_readout.json + g3_col_profiles.npz + figs/")


if __name__ == "__main__":
    main()
