"""VirHostPRED 全量批量打分（checkpoint 断点续跑）。

复用 run_virhostpred.py 的 POST/解析逻辑。设计：
  - batch=100，每批成功即追加落盘 output/scores_all.csv（含 (accession, subtype) 键）；
    重启时读已有结果跳过，可断点续跑。
  - 任何非 200（含 400/429/5xx/网络异常）先指数退避重试（服务端忙时 400 与 429
    都会出现，实测单条 AAA16879 在"400"后单独重试返回 200）；仍失败且为 400 时
    二分缩小批次隔离问题序列（记入 skipped_nonstd.csv，reason 列）；
    429 额外惩罚性等待。
  - 熔断：连续 CB_MAX 批彻底失败即中止（避免无人值守时空打服务端）。
  - 批间 sleep ≥3 s（礼貌间隔）。
  - --max-batches N：先导批验证模式。
  - --verify N：从已打分序列随机抽 N 条重新打分，核对 p_human 一致。

产出：output/scores_all.csv（accession, subtype, p_human, raw_prediction, scored_at）
      output/score_timing.csv（batch 计时流水）
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_virhostpred import post_batch, to_p_human  # noqa: E402

OUT = Path("validation_exp/hostpred_baseline/output")
POOL_CSV = OUT / "score_pool.csv"
SCORES_CSV = OUT / "scores_all.csv"
TIMING_CSV = OUT / "score_timing.csv"
SKIPPED_CSV = OUT / "skipped_nonstd.csv"
BATCH = 50      # 100 未验证过（首轮 400 为服务端忙瞬态；50 已实测通过）
SLEEP_S = 3.0
BACKOFF = [5, 15, 45, 90, 180]     # 非 200 的统一退避（秒）
CB_MAX = 3                          # 连续彻底失败批次的熔断上限


class Batch400(Exception):
    """重试后仍为 400：内容问题，走二分隔离。"""


def load_state():
    pool = pd.read_csv(POOL_CSV)
    done = pd.DataFrame()
    if SCORES_CSV.exists():
        done = pd.read_csv(SCORES_CSV)
    done_keys = set(zip(done["accession"], done["subtype"])) if len(done) else set()
    todo = pool[~pool.apply(lambda r: (r["accession"], r["subtype"]) in done_keys,
                            axis=1)]
    return pool, done, todo


def append_rows(path, df):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def score_batch(session, records):
    """带退避重试的单批打分；返回 (结果 dict, status, 耗时)。"""
    last_status, last_err = None, None
    for wait in [0] + BACKOFF:
        if wait:
            print(f"    {wait}s 后重试（上次 status={last_status} {last_err}）",
                  flush=True)
            time.sleep(wait)
        try:
            res, status, dt = post_batch(session, records)
            if status == 200 and len(res) == len(records):
                return res, status, dt
            last_status, last_err = status, f"parsed={len(res)}/{len(records)}"
            if status == 429:
                time.sleep(60)      # 限流惩罚性等待
        except requests.RequestException as e:
            last_status, last_err = None, f"{type(e).__name__}: {e}"
    if last_status == 400:
        raise Batch400()
    raise RuntimeError(f"批次重试耗尽仍失败（status={last_status} {last_err}）")


def score_recursive(session, records, skipped):
    """records: (accession, subtype, seq) 三元组；400 时二分隔离问题序列。

    fasta 头用 "{accession}|{subtype}" 复合 ID——同一 accession 可出现在多个
    亚型下（裸 accession 头会在解析 dict 里撞键丢行）。
    """
    pairs = [(f"{a}|{s}", seq) for a, s, seq in records]
    try:
        t0 = time.perf_counter()
        res, _, _ = score_batch(session, pairs)
        return res, time.perf_counter() - t0   # 键保持复合 "acc|sub"，调用方按此取
    except Batch400:
        if len(records) == 1:
            a, s, _ = records[0]
            print(f"    隔离问题序列 {a}|{s}（重试后仍 HTTP400）", flush=True)
            skipped.append({"accession": a, "subtype": s, "nonstd_chars": "",
                            "reason": "server_http400_singleton"})
            return {}, 0.0
        mid = len(records) // 2
        left, lt = score_recursive(session, records[:mid], skipped)
        time.sleep(SLEEP_S)
        right, rt = score_recursive(session, records[mid:], skipped)
        return {**left, **right}, lt + rt


def flush_skipped(skipped):
    if not skipped:
        return []
    add = pd.DataFrame(skipped)
    prev = pd.read_csv(SKIPPED_CSV) if SKIPPED_CSV.exists() else pd.DataFrame()
    pd.concat([prev, add], ignore_index=True).drop_duplicates(
        subset=["accession", "subtype"], keep="last").to_csv(SKIPPED_CSV, index=False)
    return []


def cmd_score(max_batches=None):
    pool, done, todo = load_state()
    print(f"池 {len(pool)} | 已完成 {len(done)} | 待打分 {len(todo)}", flush=True)
    if not len(todo):
        return
    session = requests.Session()
    skipped = []
    consec_fail = 0
    records = list(zip(todo["accession"], todo["subtype"], todo["ha_sequence"]))
    n_batches = (len(records) + BATCH - 1) // BATCH
    for bi in range(n_batches):
        if max_batches is not None and bi >= max_batches:
            print(f"达到 --max-batches {max_batches}，停止", flush=True)
            break
        batch = records[bi * BATCH:(bi + 1) * BATCH]
        t0 = time.perf_counter()
        try:
            res, dt = score_recursive(session, batch, skipped)
            consec_fail = 0
        except RuntimeError as e:
            consec_fail += 1
            print(f"batch {bi} 彻底失败: {e}（连续 {consec_fail}/{CB_MAX}）", flush=True)
            if consec_fail >= CB_MAX:
                flush_skipped(skipped)
                raise SystemExit(f"熔断：连续 {CB_MAX} 批失败，中止")
            continue
        rows = []
        for a, s, seq in batch:
            raw = res.get(f"{a}|{s}")
            rows.append({"accession": a, "subtype": s,
                         "p_human": to_p_human(raw), "raw_prediction": raw,
                         "scored_at": datetime.now(timezone.utc).isoformat()})
        append_rows(SCORES_CSV, pd.DataFrame(rows))
        append_rows(TIMING_CSV, pd.DataFrame([{
            "batch_idx": bi, "n_seq": len(batch), "n_parsed": len(res),
            "wall_seconds": round(time.perf_counter() - t0, 2)}]))
        print(f"batch {bi + 1}/{n_batches}: {len(batch)} 条, 解析 {len(res)}, "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        skipped = flush_skipped(skipped)
        time.sleep(SLEEP_S)
    print("done.", flush=True)


def cmd_verify(n):
    """随机抽 n 条已打分序列重新 POST，核对 p_human 完全一致。"""
    done = pd.read_csv(SCORES_CSV)
    pool = pd.read_csv(POOL_CSV)
    samp = done.sample(n=min(n, len(done)), random_state=42)
    m = samp.merge(pool, on=["accession", "subtype"], validate="one_to_one")
    session = requests.Session()
    recs = [(f"{r.accession}|{r.subtype}", r.ha_sequence) for r in m.itertuples()]
    res, status, dt = post_batch(session, recs)
    ok = True
    for _, r in m.iterrows():
        p_new = to_p_human(res.get(f"{r['accession']}|{r['subtype']}"))
        same = (p_new is not None) and abs(p_new - r["p_human"]) < 1e-9
        ok &= same
        print(f"  {r['accession']}|{r['subtype']}: old={r['p_human']} "
              f"new={p_new} {'OK' if same else 'MISMATCH'}")
    print(f"verify: status={status} {dt:.1f}s {'全部一致' if ok else '存在不一致!'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--verify", type=int, default=None, metavar="N")
    args = ap.parse_args()
    if args.verify:
        cmd_verify(args.verify)
    else:
        cmd_score(args.max_batches)
