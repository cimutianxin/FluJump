"""准备 VirHostPRED 冒烟测试的 20 条 FluJump 序列。

分组（每组 5 条，随机种子 42）：
  A: 人源 H3（all_isolates_clean.csv, host_category=human）—— 应判 human
  B: 人源 H1（同上）—— 应判 human
  C: 禽源 H5（同上, host_category=avian）—— 应判 avian（即 non-human）
  D: 2026 野鸟 H5N1（2026_isolates_clean.csv, (H5, cluster 70) jump 阳性簇）—— 宿主 avian 但为 jump 阳性

输出：
  output/smoke20.fasta      —— 头行仅 accession（服务端按首字段作 Sequence ID）
  output/smoke20_meta.csv   —— accession/subtype/cluster_id/host_category/分组/期望标签
"""
import sys
sys.path.insert(0, ".")

import pandas as pd

CLEAN_CSV = "data/processed_isolate/all_isolates_clean.csv"
Y2026_CSV = "data/processed_isolate/2026_isolates_clean.csv"
OUT_FASTA = "validation_exp/hostpred_baseline/output/smoke20.fasta"
OUT_META = "validation_exp/hostpred_baseline/output/smoke20_meta.csv"
RANDOM_SEED = 42
N_PER_GROUP = 5

EXPECTED = {"human_H3": "human", "human_H1": "human",
            "avian_H5": "avian", "avian_H5_2026_c70": "avian"}


def pick(df, group):
    sub = df.sample(n=N_PER_GROUP, random_state=RANDOM_SEED).copy()
    sub["group"] = group
    sub["expected_host"] = EXPECTED[group]
    return sub


def main():
    df = pd.read_csv(CLEAN_CSV)
    d26 = pd.read_csv(Y2026_CSV)

    # VirHostPRED 服务端拒绝非标准氨基酸（如 X；实测 HTTP 400
    # "contains unsupported residues"）——抽样池预先过滤为标准 20 字母表
    std = lambda s: s.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")
    df = df[std(df["ha_sequence"])]
    d26 = d26[std(d26["ha_sequence"])]

    parts = [
        pick(df[(df["subtype"] == "H3") & (df["host_category"] == "human")], "human_H3"),
        pick(df[(df["subtype"] == "H1") & (df["host_category"] == "human")], "human_H1"),
        pick(df[(df["subtype"] == "H5") & (df["host_category"] == "avian")], "avian_H5"),
    ]
    c70 = d26[(d26["subtype"] == "H5") & (d26["cluster_id"] == 70)]
    assert len(c70) >= N_PER_GROUP, f"(H5,70) 簇仅 {len(c70)} 条"
    d = c70.head(N_PER_GROUP).copy()
    d["group"] = "avian_H5_2026_c70"
    d["expected_host"] = "avian"
    parts.append(d)

    meta = pd.concat(parts, ignore_index=True)
    # 序列字符检查：VirHostPRED 会把非标准氨基酸替换为 X，这里先统计
    meta["seq_len"] = meta["ha_sequence"].str.len()
    meta["nonstd_chars"] = meta["ha_sequence"].apply(
        lambda s: "".join(sorted(set(s) - set("ACDEFGHIKLMNPQRSTVWY"))))
    print(meta.groupby("group")["seq_len"].describe()[["count", "min", "max"]])
    print("非标准字符:", meta[meta["nonstd_chars"] != ""][["accession", "nonstd_chars"]].to_dict("records"))

    with open(OUT_FASTA, "w") as f:
        for _, r in meta.iterrows():
            f.write(f">{r['accession']}\n{r['ha_sequence']}\n")

    meta[["accession", "subtype", "cluster_id", "host_category", "host_species",
          "collection_year", "group", "expected_host", "label_is_jump",
          "label_is_jump_human", "seq_len", "nonstd_chars"]].to_csv(OUT_META, index=False)
    print(f"写出 {len(meta)} 条 -> {OUT_FASTA} / {OUT_META}")


if __name__ == "__main__":
    main()
