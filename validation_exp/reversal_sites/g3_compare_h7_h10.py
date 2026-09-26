#!/usr/bin/env python3
"""G3：H7 vs H10 反转驱动位点对比（H5/H4 对照）

三层（L28/L17/L13）× 双标签，统一定义：
  d_pos_ST[col] = mean(ST,+) − mean(H1H3,+)   —— 亚型特异的"推负"剖面
对比：
  Spearman(d_pos_H7, d_pos_H10) 跨 1039 列；top-20 最负列 Jaccard 重叠；
  H4 同法（Group 2 阴性对照，预期不相关）；H5（Group 1 对照）。
主数据逐位贡献：L28 复用 site_attribution npz probe；L17/L13 重训（同 G2 协议，
回归断言 H5/H7）。H10/H4 逐位贡献读 G2 npz（行序 concat(H10,H4)），
列映射用 group_boundary 的 {ST}_aligned.csv（mafft --add --keeplength 同坐标系）。

输出：output/g3_column_contrib_{label}_{layer}.csv、output/g3_h7_vs_h10_overlap.json
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.reversal_sites.config import (
    SPLIT_CSV, ALIGNED_CSV, ALL_LAYERS_NPY, RESIDUE_LENS, LABELS, PROBE_NPZ,
    COL_TO_H3_JSON, GB_PROC, GB_OUT, GB_SUBTYPES, RIDGE_C_VALUES, CV_FOLDS,
    RANDOM_SEED, LABEL_COLS, LAYER_IDX, ALN_LEN, OUT_DIR,
)

RESIDUE_EMB = {ln: f"ESM_tf_clf/output/residue_emb_{ln}.npy" for ln in LAYER_IDX}
MAIN_GROUPS = ["H1H3", "H5", "H7"]
TOPK = 20

# 回归校验基准（同 G2）：group_boundary 同层 AUC，容差含 09-19 标签修复位移
REPRO_REF = {
    (17, "label_is_jump"): (0.3918, 0.8937),
    (17, "label_is_jump_human"): (0.2646, 0.6277),
    (13, "label_is_jump"): (0.2483, 0.7571),
    (13, "label_is_jump_human"): (0.2523, 0.8632),
}
REPRO_TOL = 0.06


def col_positions(aln_seq):
    return np.fromiter((i for i, c in enumerate(aln_seq) if c != "-"),
                       dtype=np.int32, count=len(aln_seq) - aln_seq.count("-"))


def fit_probe(Xtr, ytr):
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    scl = StandardScaler().fit(Xtr)
    gs.fit(scl.transform(Xtr), ytr)
    return scl, gs.best_estimator_


def accumulate(h_res, lens, col_maps, groups, w, smean, sscale):
    """逐位贡献按组聚合到对齐列 → (sum, cnt) 字典"""
    keys = sorted(set(groups))
    csum = {k: np.zeros(ALN_LEN, dtype=np.float64) for k in keys}
    ccnt = {k: np.zeros(ALN_LEN, dtype=np.int64) for k in keys}
    n = len(lens)
    for i0 in range(0, n, 512):
        h = h_res[i0:i0 + 512].astype(np.float32)
        c = ((h - smean) / sscale) @ w
        for j, r in enumerate(range(i0, min(i0 + 512, n))):
            L = int(lens[r])
            np.add.at(csum[groups[r]], col_maps[r], c[j, :L])
            np.add.at(ccnt[groups[r]], col_maps[r], 1)
    return {k: np.where(ccnt[k] > 0, csum[k] / np.maximum(ccnt[k], 1), np.nan)
            for k in keys}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 主数据（H1H3/H5/H7）框架 ──
    split = pd.read_csv(SPLIT_CSV)
    ali = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype", "aligned_ha_seq"])
    df = split.merge(ali, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    assert df["aligned_ha_seq"].notna().all()
    n = len(df)
    subtype = df["subtype"].to_numpy()
    grp_main = np.where(np.isin(subtype, ["H1", "H3"]), "H1H3", subtype)
    lens_main = np.load(RESIDUE_LENS)
    col_maps_main = [col_positions(s) for s in df["aligned_ha_seq"]]
    split_arr = df["split"].to_numpy()

    m = json.load(open(COL_TO_H3_JSON))
    c2h1 = {int(k): v for k, v in m["col_to_h3_ha1"].items()}
    c2h2 = {int(k): v for k, v in m["col_to_h3_ha2"].items()}

    # ── H10/H4 框架 ──
    gb = np.load(OUT_DIR / "g2_h10h4_contrib.npz", allow_pickle=True)
    lens_gb = gb["lens"]
    gb_subtype = gb["subtype"].astype(str)
    gb_lab = {}
    for st in GB_SUBTYPES:
        d = pd.read_csv(GB_PROC / f"{st}_isolates.csv", dtype=str)
        gb_lab[st] = d
    gb_df = pd.concat(gb_lab.values(), ignore_index=True)
    assert len(gb_df) == len(lens_gb)
    assert (gb_df["accession"].values == gb["accession"].astype(str)).all()
    # 列映射（mafft --add 同坐标系）
    al_maps = {}
    for st in GB_SUBTYPES:
        a = pd.read_csv(GB_OUT / f"{st}_aligned.csv", dtype=str)
        al_maps.update(dict(zip(a["accession"], a["aligned_ha_seq"])))
    col_maps_gb = {}
    kept_rows = [
        r for r, a in enumerate(gb_df["accession"])
        if a in al_maps
        and al_maps[a].replace("-", "") == gb_df["ha_sequence"].iloc[r]
    ]
    n_drop = len(gb_df) - len(kept_rows)
    if n_drop:
        dropped = sorted(set(gb_df["accession"]) -
                         {gb_df["accession"][r] for r in kept_rows})
        print(f"对齐缺失/序列版本不一致剔除 {n_drop}/{len(gb_df)} 行"
              f"（{dropped}）", flush=True)
    for r in kept_rows:
        col_maps_gb[r] = col_positions(al_maps[gb_df["accession"][r]])
        # G2 lens 对非 batch 最长序列多计 EOS 一位（实测 lens-真长 ∈ {0,1}），以列映射真长为准
        assert lens_gb[r] - len(col_maps_gb[r]) in (0, 1), f"GB 行 {r} 长度异常"

    arr = np.load(ALL_LAYERS_NPY, mmap_mode="r")
    m_train = (split_arr == "train")
    m_h5 = split_arr == "h5_holdout"
    m_h7 = split_arr == "h7_holdout"

    results = {}
    for label in LABEL_COLS:
        y_main = np.load(LABELS[label])
        y_gb = gb_df[label].astype(int).to_numpy()
        grp_full_main = np.array([f"{g}_{v}" for g, v in
                                  zip(grp_main, y_main)])          # 如 H7_1
        grp_full_gb = np.array([f"{g}_{v}" for g, v in
                                zip(gb_subtype, y_gb)])            # 如 H10_1

        for lname, li in LAYER_IDX.items():
            # probe：L28 复用 npz，其余重训
            if lname == "L28":
                pz = np.load(PROBE_NPZ[label])
                w, smean, sscale = pz["w"], pz["scaler_mean"], pz["scaler_scale"]
            else:
                X = np.asarray(arr[li]).astype(np.float32)
                scl, clf = fit_probe(X[m_train], y_main[m_train])
                auc_h5 = float(roc_auc_score(
                    y_main[m_h5], clf.decision_function(scl.transform(X[m_h5]))))
                auc_h7 = float(roc_auc_score(
                    y_main[m_h7], clf.decision_function(scl.transform(X[m_h7]))))
                h5_ref, h7_ref = REPRO_REF[(li, label)]
                assert abs(auc_h5 - h5_ref) < REPRO_TOL, \
                    f"{lname} {label} H5 复现异常: {auc_h5} vs {h5_ref}"
                assert abs(auc_h7 - h7_ref) < REPRO_TOL, \
                    f"{lname} {label} H7 复现异常: {auc_h7} vs {h7_ref}"
                print(f"[{label}][{lname}] 重训 H5={auc_h5:.4f} "
                      f"H7raw={auc_h7:.4f}", flush=True)
                w, smean, sscale = (clf.coef_[0].astype(np.float32),
                                    scl.mean_.astype(np.float32),
                                    scl.scale_.astype(np.float32))

            # 主数据逐位贡献（L28 与 probe npz 一致时直接复用残差文件）
            res = np.load(RESIDUE_EMB[lname], mmap_mode="r")
            means_main = accumulate(res, lens_main, col_maps_main,
                                    grp_full_main, w, smean, sscale)
            # H10/H4 逐位贡献（G2 贡献矩阵 × 列映射）
            contrib = gb[f"contrib_{lname}_{label}"]
            keys_gb = sorted(set(grp_full_gb))
            csum = {k: np.zeros(ALN_LEN) for k in keys_gb}
            ccnt = {k: np.zeros(ALN_LEN, dtype=np.int64) for k in keys_gb}
            for r in kept_rows:
                L = len(col_maps_gb[r])
                np.add.at(csum[grp_full_gb[r]], col_maps_gb[r],
                          contrib[r, :L].astype(np.float64))
                np.add.at(ccnt[grp_full_gb[r]], col_maps_gb[r], 1)
            means_gb = {k: np.where(ccnt[k] > 0,
                                    csum[k] / np.maximum(ccnt[k], 1), np.nan)
                        for k in keys_gb}

            # ── d_pos 剖面与对比 ──
            tab = pd.DataFrame({"aln_col": np.arange(ALN_LEN)})
            tab["h3"] = [f"HA1_{c2h1[c]}" if c in c2h1 else
                         (f"HA2_{c2h2[c]}" if c in c2h2 else "")
                         for c in tab["aln_col"]]
            base = means_main["H1H3_1"]
            for gname, src in [("H5", means_main), ("H7", means_main),
                               ("H10", means_gb), ("H4", means_gb)]:
                key = f"{gname}_1"
                # H4 无 jump_human 阳性（全 0），该标签下 d_pos_H4 无定义 → NaN
                tab[f"d_pos_{gname}"] = (src[key] - base) if key in src else np.nan
            tab.to_csv(OUT_DIR / f"g3_column_contrib_{label}_{lname}.csv",
                       index=False)

            # 主对比 H7~H10 要求两向量完备；各对照两两 dropna（H5 部分列无阳性覆盖）
            valid = tab.dropna(subset=["d_pos_H7", "d_pos_H10"])
            rec = {"n_valid_cols": int(len(valid))}
            for gname in ["H10", "H4", "H5"]:
                pair = valid.dropna(subset=[f"d_pos_{gname}"])
                if len(pair) < 50:
                    rec[f"spearman_H7_vs_{gname}"] = None
                    rec[f"p_H7_vs_{gname}"] = None
                    rec[f"top{TOPK}_jaccard_H7_{gname}"] = None
                    continue
                rho, p = spearmanr(pair["d_pos_H7"], pair[f"d_pos_{gname}"])
                t7 = set(pair.nsmallest(TOPK, "d_pos_H7")["aln_col"])
                tg = set(pair.nsmallest(TOPK, f"d_pos_{gname}")["aln_col"])
                rec[f"spearman_H7_vs_{gname}"] = round(float(rho), 4)
                rec[f"p_H7_vs_{gname}"] = float(f"{p:.3g}")
                rec[f"top{TOPK}_jaccard_H7_{gname}"] = round(
                    len(t7 & tg) / len(t7 | tg), 3)
            results[f"{label}_{lname}"] = rec
            print(f"[{label}][{lname}] Spearman H7~H10 "
                  f"{rec['spearman_H7_vs_H10']}（p={rec['p_H7_vs_H10']}）"
                  f" | H7~H4 {rec['spearman_H7_vs_H4']} | H7~H5 "
                  f"{rec['spearman_H7_vs_H5']} | Jaccard "
                  f"H10 {rec[f'top{TOPK}_jaccard_H7_H10']} / "
                  f"H4 {rec[f'top{TOPK}_jaccard_H7_H4']}", flush=True)

    with open(OUT_DIR / "g3_h7_vs_h10_overlap.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"→ {OUT_DIR}/g3_h7_vs_h10_overlap.json")


if __name__ == "__main__":
    main()
