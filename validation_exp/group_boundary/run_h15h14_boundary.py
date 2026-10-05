#!/usr/bin/env python3
"""H15/H14 零样本边界检验（S6，Major 5 彻底版）

完全仿照本目录 H10/H4 流程：
  download: NCBI Protein，union query（H15[all] OR H15N1..N9[all]）解决分词，
            post-2000（AND 2000:3000[pdat]），NOT pdb[filter]
  prep:     同 QC（PDB/长度 400–600/合成构建体/serotype 一致性 post-filter）
            + 09-19 词边界 host 推断（prep_clusters 已修复口径）
            + CD-HIT 99%（-c 0.99 -s 0.9 -g 1 -n 5）+ cluster 级 jump 标签
  embed:    ESM-2 150M 全 31 层 mean-pool（raw 序列，行序 = {ST}_isolates.csv）
  score:    H1+H3 train 重训 31 层 Ridge LR probe（同 run_direction_test.fit_probe），
            每层 isolate/cluster 原始 AUC + 24/31 式双侧符号检验
            + 预注册层几何签名（阳性/阴性投影均值）

注意：H15 post-2000 序列可能很少——阳性簇 <3 只报方向性、不做显著性表述；
      H14 若无阳性簇则只作"不反转"对照。

用法：python run_h15h14_boundary.py [download|prep|embed|score|all]
输出：data/raw/{ST}_raw.csv、data/processed/{ST}_*、
      output/h15_h14_emb_all_layers.npy、output/direction_test_h15_h14.json
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from validation_exp.group_boundary.config import (
    RAW_DIR, PROC_DIR, OUT_DIR, SPLIT_CSV, ALL_LAYERS_NPY, LABELS_DIR,
    LABEL_COLS, RANDOM_SEED, PREREG_LAYERS, ESM_MODEL,
)
from validation_exp.group_boundary.download_subtypes import download_subtype
from validation_exp.group_boundary.prep_clusters import (
    clean_subtype, run_cdhit, build_labels,
)
from validation_exp.group_boundary.run_direction_test import fit_probe
from validation_exp.group_boundary.supplement_analysis import binom_sign_p

NEW_SUBTYPES = ["H15", "H14"]

# H15/H14 的亚型一致性 post-filter 与 union query（config 只含 H10/H4，本地构造）
SUBTYPE_PATTERN_NEW = {st: re.compile(rf"\b{st}N\d\b|\({st}\)|\b{st}\b")
                       for st in NEW_SUBTYPES}


def _union_query_post2000(h: str) -> str:
    terms = " OR ".join([f"{h}[all fields]"] +
                        [f"{h}N{i}[all fields]" for i in range(1, 10)])
    return (f"hemagglutinin[protein name] AND Influenza A virus[organism] "
            f"AND ({terms}) NOT pdb[filter] AND 2000:3000[pdat]")


def stage_download():
    import validation_exp.group_boundary.prep_clusters as pc
    # prep_clusters.clean_subtype 内部用 config.SUBTYPE_PATTERN——注入 H15/H14
    pc.SUBTYPE_PATTERN.update(SUBTYPE_PATTERN_NEW)
    for st in NEW_SUBTYPES:
        download_subtype(st, _union_query_post2000(st))


def stage_prep():
    import validation_exp.group_boundary.prep_clusters as pc
    pc.SUBTYPE_PATTERN.update(SUBTYPE_PATTERN_NEW)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    for st in NEW_SUBTYPES:
        print(f"\n{'='*60}\n  {st}\n{'='*60}")
        rows = clean_subtype(st)
        acc2cid = run_cdhit(st)
        cl2label = build_labels(st, rows, acc2cid)
        # isolate 级评估表（行序 = embedding 行序），同 prep_clusters.main
        import csv as _csv
        out_path = PROC_DIR / f"{st}_isolates.csv"
        cols = ["accession", "subtype", "strain_name", "host_category",
                "collection_year", "country", "ha_sequence", "cluster_id",
                "label_is_jump", "label_is_jump_human"]
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                cid = acc2cid.get(r["accession"], "")
                lab = cl2label.get(cid, {})
                w.writerow({
                    "accession": r["accession"], "subtype": st,
                    "strain_name": r.get("strain_name", ""),
                    "host_category": r.get("host_category", ""),
                    "collection_year": r.get("collection_year", ""),
                    "country": r.get("country", ""),
                    "ha_sequence": r["ha_sequence"],
                    "cluster_id": cid,
                    "label_is_jump": lab.get("label_is_jump", ""),
                    "label_is_jump_human": lab.get("label_is_jump_human", ""),
                })
        print(f"  → {out_path}: {len(rows)} 行")


def stage_embed():
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, EsmModel

    df = pd.concat([pd.read_csv(PROC_DIR / f"{st}_isolates.csv", dtype=str)
                    for st in NEW_SUBTYPES], ignore_index=True)
    seqs = df["ha_sequence"].tolist()
    print(f"总 isolates: {len(seqs)}（{'+'.join(NEW_SUBTYPES)}）")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL, output_hidden_states=True
                                     ).to(device).eval()
    n_layers = model.config.num_hidden_layers + 1
    all_layers = [[] for _ in range(n_layers)]
    for i in tqdm(range(0, len(seqs), 8), desc="提取全层 embedding"):
        batch = seqs[i:i + 8]
        inp = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        for j in range(len(batch)):
            mask = inp["attention_mask"][j, 1:-1].bool()
            for li in range(n_layers):
                h = out.hidden_states[li][j, 1:-1][mask].mean(dim=0)
                all_layers[li].append(h.cpu().numpy())
    arr = np.stack([np.stack(layer) for layer in all_layers])
    np.save(OUT_DIR / "h15_h14_emb_all_layers.npy", arr)
    df[["accession", "subtype"]].to_csv(
        OUT_DIR / "emb_row_index_h15h14.csv", index=False)
    print(f"  → h15_h14_emb_all_layers.npy: {arr.shape}")


def stage_score():
    arr = np.load(ALL_LAYERS_NPY)
    split_df = pd.read_csv(SPLIT_CSV)
    m_train = (split_df["split"] == "train").values
    n_layers = arr.shape[0]

    ev_df = pd.concat([pd.read_csv(PROC_DIR / f"{st}_isolates.csv", dtype=str)
                       for st in NEW_SUBTYPES], ignore_index=True)
    ev_emb = np.load(OUT_DIR / "h15_h14_emb_all_layers.npy")
    assert ev_emb.shape[1] == len(ev_df)
    for col in LABEL_COLS:
        ev_df[col] = ev_df[col].astype(int)
    ev_mask = {st: (ev_df["subtype"] == st).values for st in NEW_SUBTYPES}

    results = {"eval_size": {}, "profile": {}, "geometry": {}, "sign_test": {}}
    for st in NEW_SUBTYPES:
        sub = ev_df[ev_mask[st]]
        results["eval_size"][st] = {
            "n_isolates": int(ev_mask[st].sum()),
            "n_clusters": int(sub["cluster_id"].nunique()),
            "n_pos_clusters_jump": int(
                sub.groupby("cluster_id")["label_is_jump"].max().sum()),
            "n_pos_clusters_jump_human": int(
                sub.groupby("cluster_id")["label_is_jump_human"].max().sum()),
        }
    print(f"评估集规模: {json.dumps(results['eval_size'])}")

    for label in LABEL_COLS:
        y = np.load(LABELS_DIR / f"labels_{label}.npy")
        yt = y[m_train]
        results["profile"][label] = {}
        results["geometry"][label] = {}
        for li in range(n_layers):
            scl, gs = fit_probe(arr[li][m_train], yt)
            sc_all = gs.decision_function(scl.transform(ev_emb[li]))
            for st in NEW_SUBTYPES:
                m = ev_mask[st]
                sub = ev_df[m]
                yv = sub[label].values
                sc = sc_all[m]
                iso_auc = float(roc_auc_score(yv, sc)) if yv.sum() > 0 else None
                cl_g = pd.DataFrame({"cl": sub["cluster_id"].values,
                                     "y": yv, "sc": sc}).groupby("cl").agg(
                    y=("y", "max"), sc=("sc", "mean"))
                cl_auc = float(roc_auc_score(cl_g["y"], cl_g["sc"])) \
                    if cl_g["y"].sum() > 0 else None
                results["profile"][label].setdefault(st, {})[li] = {
                    "isolate_auc": iso_auc, "cluster_auc": cl_auc}
                if li in PREREG_LAYERS[label]:
                    results["geometry"][label].setdefault(st, {})[li] = {
                        "proj_pos_mean": float(sc[yv == 1].mean()) if yv.sum() else None,
                        "proj_neg_mean": float(sc[yv == 0].mean()) if (yv == 0).sum() else None,
                        "frac_all_negative": float((sc < 0).mean()),
                        "frac_pos_negative": float((sc[yv == 1] < 0).mean()) if yv.sum() else None,
                    }
            if li % 10 == 0 or li == n_layers - 1:
                print(f"  {label} L{li} done", flush=True)

    # ── 24/31 式符号检验（cluster AUC < 0.5 层数，双侧二项）──
    print(f"\n{'='*76}\n符号检验（cluster AUC < 0.5 层数 / 31）\n{'='*76}")
    for st in NEW_SUBTYPES:
        for label in LABEL_COLS:
            au = [results["profile"][label][st][str(li) if str(li) in results["profile"][label][st] else li]["cluster_auc"]
                  for li in range(n_layers)]
            au = [a for a in au if a is not None]
            if not au:
                print(f"  {st} {label}: 无阳性簇，仅作不反转对照")
                results["sign_test"].setdefault(st, {})[label] = None
                continue
            k = sum(1 for a in au if a < 0.5)
            p = binom_sign_p(k, len(au))
            results["sign_test"].setdefault(st, {})[label] = {
                "n_layers_below_0.5": k, "n_layers": len(au),
                "p_two_sided": p,
                "cluster_aucs": [round(a, 4) for a in au]}
            print(f"  {st} {label}: {k}/{len(au)} 层 <0.5, p={p:.4f}")

    with open(OUT_DIR / "direction_test_h15_h14.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ {OUT_DIR}/direction_test_h15_h14.json")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if stage in ("download", "all"):
        stage_download()
    if stage in ("prep", "all"):
        stage_prep()
    if stage in ("embed", "all"):
        stage_embed()
    if stage in ("score", "all"):
        stage_score()
