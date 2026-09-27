"""向 VirHostPRED web server 提交冒烟 FASTA 并解析结果。

VirHostPRED 无公开代码仓库与权重文件（论文 Data availability：数据集见补充材料，
脚本"向通讯作者索取"），官方唯一可执行形态为 web server：
  POST https://www.biochemintelli.com/virhostpred/  (Django 表单, 字段 fasta_sequence,
  单次上限 100 条, 最短 10 aa)
服务端模型：ESM2-t48-15B mean-pool embedding + SVM-RBF（论文摘要/方法节）。

响应为同步 HTML，内含 resultsTable：Sequence ID + "HUMAN VIRUS: xx.x%" 或
"NON-HUMAN VIRUS: xx.x%"。本脚本将其统一折算为 p_human（HUMAN 时 = xx.x/100，
NON-HUMAN 时 = 1 - xx.x/100），原始文本保留在 raw_prediction 列。

按 (accession, subtype) 回 join 元数据（cluster_id 为亚型内编号，跨亚型复用）。

用法:
  python validation_exp/hostpred_baseline/run_virhostpred.py [--batch-size 20]
"""
import argparse
import re
import sys
import time

import pandas as pd
import requests

BASE = "https://www.biochemintelli.com/virhostpred/"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
OUT_CSV = "validation_exp/hostpred_baseline/output/smoke_scores.csv"
TIMING_CSV = "validation_exp/hostpred_baseline/output/smoke_timing.csv"


def read_fasta(path):
    """读 FASTA -> [(header_first_token, sequence)]，头行以 accession 为唯一标识。"""
    records, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf)))
            name, buf = line[1:].split()[0], []
        elif line:
            buf.append(line)
    if name is not None:
        records.append((name, "".join(buf)))
    return records


def post_batch(session, records, timeout=900):
    """提交一批序列，返回 (结果 dict[accession->raw], http_status, 耗时秒)。"""
    fasta_text = "\n".join(f">{n}\n{s}" for n, s in records)
    r = session.get(BASE, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    m = re.search(r'csrfmiddlewaretoken" value="([^"]+)"', r.text)
    if not m:
        raise RuntimeError("GET 响应中未找到 CSRF token")
    t0 = time.perf_counter()
    resp = session.post(
        BASE,
        headers={"User-Agent": UA, "Referer": BASE,
                 "Origin": "https://www.biochemintelli.com"},
        data={"csrfmiddlewaretoken": m.group(1), "fasta_sequence": fasta_text},
        timeout=timeout,
    )
    dt = time.perf_counter() - t0
    results = parse_results(resp.text)
    return results, resp.status_code, dt


def parse_results(html):
    """解析 resultsTable 行: Sequence ID + 预测文本。"""
    out = {}
    block = re.search(r'<table[^>]*id="resultsTable".*?</table>', html, re.S)
    if not block:
        return out
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", block.group(0), re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) >= 2:
            sid = re.sub(r"<[^>]+>", "", cells[0]).strip()
            pred = re.sub(r"<[^>]+>", "", cells[1]).strip()
            out[sid] = pred
    return out


def to_p_human(raw):
    """'HUMAN VIRUS: 75.6%' -> 0.756; 'NON-HUMAN VIRUS: 75.6%' -> 0.244。"""
    m = re.match(r"(NON-HUMAN|HUMAN)\s+VIRUS:\s*([\d.]+)%", raw or "", re.I)
    if not m:
        return None
    p = float(m.group(2)) / 100.0
    return p if m.group(1).upper() == "HUMAN" else 1.0 - p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="validation_exp/hostpred_baseline/output/smoke20.fasta")
    ap.add_argument("--meta", default="validation_exp/hostpred_baseline/output/smoke20_meta.csv")
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    records = read_fasta(args.fasta)
    meta = pd.read_csv(args.meta)
    print(f"共 {len(records)} 条, batch_size={args.batch_size}")

    session = requests.Session()
    rows, timings = [], []
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        try:
            res, status, dt = post_batch(session, batch)
        except Exception as e:
            print(f"batch {i // args.batch_size} 失败: {e}")
            res, status, dt = {}, -1, None
        timings.append({"batch_idx": i // args.batch_size, "n_seq": len(batch),
                        "http_status": status, "wall_seconds": dt})
        print(f"batch {i // args.batch_size}: {len(batch)} 条, HTTP {status}, "
              f"{dt:.1f}s, 解析到 {len(res)} 条")
        for name, seq in batch:
            raw = res.get(name)
            rows.append({"accession": name, "seq_len": len(seq),
                         "raw_prediction": raw, "p_human": to_p_human(raw)})
        time.sleep(2)  # 礼貌间隔

    out = pd.DataFrame(rows)
    # (accession, subtype) 回 join 元数据
    out = out.merge(meta, on="accession", how="left", validate="one_to_one")
    out = out[["accession", "subtype", "cluster_id", "host_category", "host_species",
               "collection_year", "group", "expected_host", "label_is_jump",
               "label_is_jump_human", "seq_len_x", "raw_prediction", "p_human"]]
    out = out.rename(columns={"seq_len_x": "seq_len"})
    out.to_csv(OUT_CSV, index=False)
    pd.DataFrame(timings).to_csv(TIMING_CSV, index=False)
    print(f"写出 {OUT_CSV} ({out['p_human'].notna().sum()}/{len(out)} 条解析成功)")


if __name__ == "__main__":
    main()
