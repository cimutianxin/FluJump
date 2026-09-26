#!/usr/bin/env python3
"""G4：反转驱动位点的 (a) clade 指纹 vs (b) 功能机制 四项正交检验

输入：g3 产出的 6 张 g3_column_contrib_{label}_{layer}.csv（每格共享驱动列 =
top20 d_pos_H7 ∩ top20 d_pos_H10），外加 core 集（≥2 格共享的列）。

四项检验（预注册判定规则见计划/日志）：
  A. clade 标记：共享列的 JS 分歧度（pooled(H7,H10) vs pooled(H1,H3,H5)）百分位
     + 10k 置换 p；另记 H7/H10 consensus 一致率与保守度。
  B. 空间聚集：1HGG Cα（chain A=HA1, chain B=HA2，锚点 chain A 226 必须为 L），
     共享集 median pairwise 距离 vs 同 HA1/HA2 构成随机 null（10k 次）。
  C. 糖基化：N-X-S/T（X≠P）sequon 列级 fraction 差异 Δ=|H7H10 − H1H3H5|，
     共享列百分位 + 置换 p。
  D. 亚型身份对照 probe：L28 mean-pooled LR「H7 vs H1H3」与「H5 vs H1H3」，
     逐位贡献组差 d_id；判据 = Spearman(|d_id_H7|, −d_pos_H7) 与 top20 Jaccard
     （亚型诊断列是否就是 jump probe 推负 H7 的列）；对照 |d_id_H5|、d_pos_H5。

输出：output/g4_shared_driver_sets.json、output/g4_column_scores.csv、
      output/g4_orthogonal.json
运行：HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env1 python（CPU，<15 min）
"""

import json
import sys
import urllib.request
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from validation_exp.reversal_sites.config import (
    SPLIT_CSV, ALIGNED_CSV, ALL_LAYERS_NPY, RESIDUE_EMB_L28, RESIDUE_LENS,
    LABELS, COL_TO_H3_JSON, GB_OUT, RIDGE_C_VALUES, CV_FOLDS, RANDOM_SEED,
    LABEL_COLS, ALN_LEN, OUT_DIR,
)

DATA_DIR = OUT_DIR.parent / "data"
PDB_PATH = DATA_DIR / "1HGG.pdb"
PDB_URL = "https://files.rcsb.org/download/1HGG.pdb"
TOPK = 20
N_PERM = 10000
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
MIN_COUNT = 10          # 列有效所需每组最少非 gap 残基数
CELLS = [f"{lb}_{ln}" for lb in LABEL_COLS for ln in ["L28", "L17", "L13"]]

rng = np.random.default_rng(RANDOM_SEED)


# ─────────────────────────── 基础工具 ───────────────────────────

def col_positions(aln_seq):
    return np.fromiter((i for i, c in enumerate(aln_seq) if c != "-"),
                       dtype=np.int32, count=len(aln_seq) - aln_seq.count("-"))


def encode_group(seqs):
    """一组对齐序列 → uint8 矩阵（AA→0..19，其他→255）"""
    table = np.full(256, 255, dtype=np.uint8)
    for a, i in AA_IDX.items():
        table[ord(a)] = i
    m = np.empty((len(seqs), ALN_LEN), dtype=np.uint8)
    for r, s in enumerate(seqs):
        m[r] = table[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
    return m


def aa_counts(mat):
    """(n,1039) uint8 → (1039,20) 计数"""
    cnt = np.zeros((ALN_LEN, 20), dtype=np.int64)
    valid = mat != 255
    for a in range(20):
        cnt[:, a] = ((mat == a)).sum(axis=0)
    return cnt, valid.sum(axis=0)          # 每列非 gap 总数


def perm_p(percentiles, obs_med):
    """null：从全体有效列 percentile 中抽同大小集合取中位，10k 次"""
    n = len(percentiles)
    null = np.median(rng.choice(percentiles, size=(N_PERM, n), replace=True),
                     axis=1)
    return float((1 + (null >= obs_med).sum()) / (N_PERM + 1))


def accumulate(h_res, lens, col_maps, groups, w, smean, sscale):
    """逐位贡献按组聚合到对齐列（同 g3）→ {组: (ALN_LEN,) 均值}"""
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


# ─────────────────────── 共享驱动位点集 ───────────────────────

def build_shared_sets():
    sets, tabs = {}, {}
    for cell in CELLS:
        t = pd.read_csv(OUT_DIR / f"g3_column_contrib_{cell}.csv")
        tabs[cell] = t
        v = t.dropna(subset=["d_pos_H7", "d_pos_H10"])
        h7 = set(v.nsmallest(TOPK, "d_pos_H7")["aln_col"])
        h10 = set(v.nsmallest(TOPK, "d_pos_H10")["aln_col"])
        sets[cell] = sorted(h7 & h10)
    freq = {}
    for s in sets.values():
        for c in s:
            freq[c] = freq.get(c, 0) + 1
    sets["core"] = sorted(c for c, k in freq.items() if k >= 2)
    with open(OUT_DIR / "g4_shared_driver_sets.json", "w") as f:
        json.dump({k: [int(x) for x in v] for k, v in sets.items()}, f, indent=1)
    print("共享集大小:", {k: len(v) for k, v in sets.items()}, flush=True)
    return sets, tabs


# ─────────────────────── A. clade 标记 ───────────────────────

def section_A(group_mats, sets):
    # pooled 频率
    cnt_a = sum(aa_counts(group_mats[st])[0] for st in ["H7", "H10"])
    n_a = sum(aa_counts(group_mats[st])[1] for st in ["H7", "H10"])
    cnt_b = sum(aa_counts(group_mats[st])[0] for st in ["H1", "H3", "H5"])
    n_b = sum(aa_counts(group_mats[st])[1] for st in ["H1", "H3", "H5"])
    valid = (n_a >= MIN_COUNT) & (n_b >= MIN_COUNT)

    js = np.full(ALN_LEN, np.nan)
    for c in np.where(valid)[0]:
        js[c] = jensenshannon(cnt_a[c] / n_a[c], cnt_b[c] / n_b[c], base=2)

    # 全体有效列 percentile（JS 越大越 clade 特异）
    vcols = np.where(valid)[0]
    order = np.argsort(np.argsort(js[vcols]))
    pct = np.full(ALN_LEN, np.nan)
    pct[vcols] = order / (len(vcols) - 1) * 100

    # H7/H10 consensus 一致性与保守度
    cons = {}
    for st in ["H7", "H10"]:
        cnt, nn = aa_counts(group_mats[st])
        with np.errstate(invalid="ignore"):
            fr = cnt / np.maximum(nn, 1)[:, None]
        cons[st] = (fr.argmax(axis=1), fr.max(axis=1))

    out, scores = {}, {}
    for name, cols in sets.items():
        cols_v = [c for c in cols if valid[c]]
        if not cols_v:
            continue
        p_med = float(np.median(pct[cols_v]))
        same = float(np.mean([cons["H7"][0][c] == cons["H10"][0][c]
                              for c in cols_v]))
        c7 = float(np.mean([cons["H7"][1][c] for c in cols_v]))
        c10 = float(np.mean([cons["H10"][1][c] for c in cols_v]))
        out[name] = {"n_cols": len(cols_v),
                     "median_js_percentile": round(p_med, 1),
                     "perm_p": round(perm_p(pct[vcols], p_med), 5),
                     "h7_h10_consensus_same": round(same, 3),
                     "h7_conservation": round(c7, 3),
                     "h10_conservation": round(c10, 3)}
        scores[name] = {int(c): float(js[c]) for c in cols_v}
    bg = float(np.nanmedian(pct[vcols]))
    print(f"[A] 背景 JS 百分位中位={bg:.1f}；共享集:", flush=True)
    for k, v in out.items():
        print(f"  {k}: pct_med={v['median_js_percentile']} "
              f"p={v['perm_p']} consensus同={v['h7_h10_consensus_same']}", flush=True)
    return out, js, valid, pct


# ─────────────────────── B. 空间聚集 ───────────────────────

def load_1hgg():
    if not PDB_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"下载 {PDB_URL} ...", flush=True)
        urllib.request.urlretrieve(PDB_URL, PDB_PATH)
    ca = {"A": {}, "B": {}}
    resname = {}
    with open(PDB_PATH) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ch = line[21]
                if ch in ca:
                    rs = int(line[22:26])
                    ca[ch][rs] = np.array([float(line[30:38]),
                                           float(line[38:46]),
                                           float(line[46:54])])
                    resname[(ch, rs)] = line[17:20].strip()
    # 锚点验证：canonical 226 = L（同 build_h3_mapping）
    assert resname.get(("A", 226)) == "LEU", \
        f"1HGG chain A 226 = {resname.get(('A', 226))} ≠ LEU，编号验证失败"
    return ca


def section_B(sets, valid):
    m = json.load(open(COL_TO_H3_JSON))
    c2h1 = {int(k): v for k, v in m["col_to_h3_ha1"].items()}
    c2h2 = {int(k): v for k, v in m["col_to_h3_ha2"].items()}
    try:
        ca = load_1hgg()
    except Exception as e:
        print(f"[B] BLOCKED: {e}", flush=True)
        return {"status": "blocked", "reason": str(e)}, None

    # 可映射列池：有 canonical 编号 + 1HGG 有坐标 + A 段有效列
    pool1 = sorted(c for c in c2h1 if c2h1[c] in ca["A"] and valid[c])
    pool2 = sorted(c for c in c2h2 if c2h2[c] in ca["B"] and valid[c])
    coord = {}
    for c in pool1:
        coord[c] = ca["A"][c2h1[c]]
    for c in pool2:
        coord[c] = ca["B"][c2h2[c]]
    print(f"[B] 可映射列池 HA1={len(pool1)} HA2={len(pool2)}", flush=True)

    def med_dist(cols):
        pts = [coord[c] for c in cols]
        if len(pts) < 3:
            return np.nan
        d = [float(np.linalg.norm(pts[i] - pts[j]))
             for i, j in combinations(range(len(pts)), 2)]
        return float(np.median(d))

    out = {"status": "ok", "pdb": "1HGG", "pools": {"HA1": len(pool1),
                                                     "HA2": len(pool2)}}
    for name, cols in sets.items():
        cols_m = [c for c in cols if c in coord]
        if len(cols_m) < 3:
            out[name] = {"n_mapped": len(cols_m)}
            continue
        n1 = sum(1 for c in cols_m if c in c2h1)
        n2 = len(cols_m) - n1
        obs = med_dist(cols_m)
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            draw = list(rng.choice(pool1, size=min(n1, len(pool1)),
                                   replace=False))
            if n2 and len(pool2):
                draw += list(rng.choice(pool2, size=min(n2, len(pool2)),
                                        replace=False))
            null[b] = med_dist(draw)
        p = float((1 + (null <= obs).sum()) / (N_PERM + 1))  # 距离越小越聚集
        out[name] = {"n_mapped": len(cols_m), "n_ha1": n1, "n_ha2": n2,
                     "median_pairwise_A": round(obs, 1),
                     "null_median_A": round(float(np.median(null)), 1),
                     "perm_p": round(p, 5)}
        print(f"  {name}: n={len(cols_m)} d={obs:.1f}Å "
              f"null={np.median(null):.1f}Å p={p:.4f}", flush=True)
    return out, coord


# ─────────────────────── C. 糖基化 motif ───────────────────────

def sequon_columns(aln_seq):
    """gapless 序列中 N-X-S/T（X≠P）的 N 所在对齐列"""
    pos = col_positions(aln_seq)
    s = aln_seq.replace("-", "")
    return {int(pos[i]) for i in range(len(s) - 2)
            if s[i] == "N" and s[i + 1] != "P" and s[i + 2] in "ST"}


def section_C(group_seqs, sets, valid):
    # 每列每组：带 sequon 的序列占比
    frac = {}
    for st, seqs in group_seqs.items():
        cnt = np.zeros(ALN_LEN)
        for s in seqs:
            for c in sequon_columns(s):
                cnt[c] += 1
        frac[st] = cnt / len(seqs)
    f_a = (frac["H7"] * len(group_seqs["H7"]) +
           frac["H10"] * len(group_seqs["H10"])) / \
          (len(group_seqs["H7"]) + len(group_seqs["H10"]))
    f_b = (sum(frac[st] * len(group_seqs[st]) for st in ["H1", "H3", "H5"]) /
           sum(len(group_seqs[st]) for st in ["H1", "H3", "H5"]))
    delta = np.abs(f_a - f_b)
    delta[~valid] = np.nan

    vcols = np.where(valid)[0]
    order = np.argsort(np.argsort(delta[vcols]))
    pct = np.full(ALN_LEN, np.nan)
    pct[vcols] = order / (len(vcols) - 1) * 100

    out = {}
    for name, cols in sets.items():
        cols_v = [c for c in cols if valid[c]]
        if not cols_v:
            continue
        p_med = float(np.median(pct[cols_v]))
        out[name] = {"n_cols": len(cols_v),
                     "median_delta_percentile": round(p_med, 1),
                     "perm_p": round(perm_p(pct[vcols], p_med), 5),
                     "n_delta_gt_0.3": int((delta[cols_v] > 0.3).sum())}
        print(f"[C] {name}: Δpct_med={p_med:.1f} p={out[name]['perm_p']} "
              f"Δ>0.3: {out[name]['n_delta_gt_0.3']}/{len(cols_v)}", flush=True)
    return out, delta


# ─────────────────────── D. 亚型身份对照 probe ───────────────────────

def fit_probe(Xtr, ytr):
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": RIDGE_C_VALUES},
                      cv=StratifiedKFold(CV_FOLDS, shuffle=True,
                                         random_state=RANDOM_SEED),
                      scoring="roc_auc", n_jobs=-1)
    scl = StandardScaler().fit(Xtr)
    gs.fit(scl.transform(Xtr), ytr)
    return scl, gs.best_estimator_, float(gs.best_score_)


def section_D(df, col_maps_main, lens_main, grp_main):
    arr = np.asarray(np.load(ALL_LAYERS_NPY, mmap_mode="r")[28]).astype(np.float32)
    split_arr = df["split"].to_numpy()
    subtype = df["subtype"].to_numpy()
    is_h13 = np.isin(subtype, ["H1", "H3"])

    out = {}
    d_id = {}
    for target in ["H7", "H5"]:
        hold = split_arr == f"{target.lower()}_holdout"
        m = (is_h13 & (split_arr == "train")) | hold
        y = hold[m].astype(int)
        scl, clf, cv = fit_probe(arr[m], y)
        print(f"[D] {target}-identity probe: n={m.sum()} 阳性={y.sum()} "
              f"CV AUC={cv:.4f}", flush=True)
        means = accumulate(np.load(RESIDUE_EMB_L28, mmap_mode="r"), lens_main,
                           col_maps_main, grp_main,
                           clf.coef_[0].astype(np.float32),
                           scl.mean_.astype(np.float32),
                           scl.scale_.astype(np.float32))
        d_id[target] = means[target] - means["H1H3"]
        out[f"cv_auc_{target}"] = round(cv, 4)

    for label in LABEL_COLS:
        t = pd.read_csv(OUT_DIR / f"g3_column_contrib_{label}_L28.csv")
        for target in ["H7", "H5"]:
            d = pd.DataFrame({"aln_col": np.arange(ALN_LEN),
                              "d_id": d_id[target],
                              "d_pos_H7": t["d_pos_H7"],
                              "d_pos_H5": t["d_pos_H5"]}).dropna()
            neg7 = -d["d_pos_H7"].to_numpy()
            absid = np.abs(d["d_id"].to_numpy())
            rho, p = spearmanr(absid, neg7)
            # top20：|d_id| 最大列 vs −d_pos_H7 最大列
            a = set(d.iloc[np.argsort(-absid)[:TOPK]]["aln_col"])
            b7 = set(d.iloc[np.argsort(-neg7)[:TOPK]]["aln_col"])
            # 对照：−d_pos_H5
            neg5 = -d["d_pos_H5"].to_numpy()
            rho5, p5 = spearmanr(absid, neg5)
            b5 = set(d.iloc[np.argsort(-neg5)[:TOPK]]["aln_col"])
            rec = {"spearman_absid_vs_negdposH7": round(float(rho), 4),
                   "p": float(f"{p:.3g}"),
                   "top20_jaccard": round(len(a & b7) / len(a | b7), 3),
                   "ctrl_spearman_vs_negdposH5": round(float(rho5), 4),
                   "ctrl_top20_jaccard_H5": round(len(a & b5) / len(a | b5), 3)}
            out[f"{label}_id{target}"] = rec
            print(f"[D] {label} |d_id_{target}|: ρ(vs −d_pos_H7)={rec['spearman_absid_vs_negdposH7']} "
                  f"J={rec['top20_jaccard']} | ctrl H5 ρ={rec['ctrl_spearman_vs_negdposH5']}",
                  flush=True)
    return out


# ─────────────────────────── main ───────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets, _ = build_shared_sets()

    # 分组对齐序列（主数据 + H10/H4）
    ali = pd.read_csv(ALIGNED_CSV, usecols=["accession", "subtype",
                                            "aligned_ha_seq"])
    group_seqs = {st: g["aligned_ha_seq"].tolist()
                  for st, g in ali.groupby("subtype")}
    for st in ["H10", "H4"]:
        a = pd.read_csv(GB_OUT / f"{st}_aligned.csv", dtype=str)
        group_seqs[st] = a["aligned_ha_seq"].tolist()
    print("各组序列数:", {k: len(v) for k, v in group_seqs.items()}, flush=True)
    group_mats = {st: encode_group(s) for st, s in group_seqs.items()}

    res = {}
    res["A_clade_marker"], js, valid, _ = section_A(group_mats, sets)
    res["B_spatial"], _ = section_B(sets, valid)
    res["C_glyco"], delta = section_C(group_seqs, sets, valid)

    # 逐列评分表留痕
    m = json.load(open(COL_TO_H3_JSON))
    c2h1 = {int(k): v for k, v in m["col_to_h3_ha1"].items()}
    c2h2 = {int(k): v for k, v in m["col_to_h3_ha2"].items()}
    tab = pd.DataFrame({"aln_col": np.arange(ALN_LEN)})
    tab["h3"] = [f"HA1_{c2h1[c]}" if c in c2h1 else
                 (f"HA2_{c2h2[c]}" if c in c2h2 else "") for c in tab["aln_col"]]
    tab["js_div"] = js
    tab["delta_glyco"] = delta
    tab["valid"] = valid
    tab.to_csv(OUT_DIR / "g4_column_scores.csv", index=False)

    # D：主数据框架（与 g3 相同 merge，行序 = isolate_split.csv）
    split = pd.read_csv(SPLIT_CSV)
    df = split.merge(ali, on=["accession", "subtype"], how="left",
                     validate="one_to_one")
    assert df["aligned_ha_seq"].notna().all()
    subtype = df["subtype"].to_numpy()
    grp_main = np.where(np.isin(subtype, ["H1", "H3"]), "H1H3", subtype)
    lens_main = np.load(RESIDUE_LENS)
    col_maps_main = [col_positions(s) for s in df["aligned_ha_seq"]]
    res["D_identity_probe"] = section_D(df, col_maps_main, lens_main, grp_main)

    with open(OUT_DIR / "g4_orthogonal.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"→ {OUT_DIR}/g4_orthogonal.json", flush=True)


if __name__ == "__main__":
    main()
