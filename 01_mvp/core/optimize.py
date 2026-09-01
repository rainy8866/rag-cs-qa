# -*- coding: utf-8 -*-
"""一键优化与阈值标定。

★ 原则：只调"不动向量库/不动评估集"的参数（TOP_K / MIN_SCORE / use_rerank），
保证改前改后是同一张卷子。绝对不在这里改 CHUNK_SIZE/embedding（改它们必须重建知识库+新建版本快照）。
"""
import json
from itertools import product
from datetime import datetime

from core import config
from core.gold_set import load_gold_set
from core.metrics import recall_at_k

TOP_K_CANDIDATES = [3, 5, 8]
MIN_SCORE_CANDIDATES = [0.30, 0.35, 0.40]
RERANK_CANDIDATES_BOOL = [False, True]


def optimize_auto(gold_items=None) -> dict:
    """在固定 gold_set 上搜索参数组合，选 Recall 最好的一档落地(写回 config.py)。"""
    if gold_items is None:
        gold_items = load_gold_set()["items"]
        if not gold_items:
            return {"ok": False, "reason": "评估集为空，请先「生成评估集」"}

    best = None
    results = []
    combos = list(product(TOP_K_CANDIDATES, MIN_SCORE_CANDIDATES, RERANK_CANDIDATES_BOOL))
    for top_k, min_score, rr in combos:
        try:
            r = recall_at_k(gold_items, top_k=top_k,
                            use_rerank=rr)
        except Exception as e:
            continue
        row = {"top_k": top_k, "min_score": min_score, "use_rerank": rr,
               "recall": r["recall"], "hit": r["hit"], "total": r["total"]}
        results.append(row)
        if best is None or row["recall"] > best["recall"]:
            best = row

    if best is None:
        return {"ok": False, "reason": "参数搜索全部失败(可能未配置 DEEPSEEK_API_KEY)"}

    # 把最优参数写回 config（修改默认值）
    _apply_config(best)
    _save_compare(results, best)
    return {"ok": True, "best": best, "results": results,
            "n_combo": len(results)}


def _apply_config(best: dict):
    config.TOP_K = best["top_k"]
    config.MIN_SCORE = best["min_score"]
    # use_rerank 由 UI 开关/写入一个持久化标记
    _write_flag(best["use_rerank"])


def _flag_path():
    config.ensure_dirs()
    return config.MVP_DIR / "rerank_flag.json"


def _write_flag(flag: bool):
    import json
    (_flag_path()).write_text(json.dumps({"use_rerank": flag}), encoding="utf-8")


def read_summary() -> dict:
    """读取已落地的优化摘要。"""
    config.ensure_dirs()
    flag = False
    if _flag_path().exists():
        try:
            flag = json.loads(_flag_path().read_text(encoding="utf-8")).get("use_rerank", False)
        except Exception:
            flag = False
    return {"use_rerank": flag, "MIN_SCORE": config.MIN_SCORE, "TOP_K": config.TOP_K}


def _save_compare(results, best):
    config.ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "ts": datetime.now().isoformat(),
        "best": best,
        "results": results,
    }
    (config.COMPARE_DIR / f"optimize_{ts}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- 阈值标定 ----
def calibrate_threshold(gold_items=None):
    """跑完整评估集的每个问题的 top-k 分数，区分命中/未命中两团，画分布并推荐阈值。"""
    if gold_items is None:
        gold_items = load_gold_set()["items"]
    from core.gold_set import snippet_to_chunk
    from core.retrieval import retrieve
    hit_scores, miss_scores = [], []
    detail = []
    for it in gold_items:
        cid = snippet_to_chunk(it.get("required_snippet", ""))
        res = retrieve(it["question"], top_k=config.RERANK_CANDIDATES)
        if cid is None:
            continue
        hit_ids = {c.id for c in res}
        if cid in hit_ids:
            c = next((x for x in res if x.id == cid), None)
            hit_scores.append(c.score if c else 0.0)
        else:
            miss_scores.append(max((x.score for x in res), default=0.0))
        detail.append({"question": it["question"]})
    # 自动推荐：取命中分数下四分位 与 未命中上四分位的中间
    import statistics
    rec = None
    if hit_scores:
        lo = statistics.quantiles(hit_scores, n=4)[2] if len(hit_scores) >= 4 else min(hit_scores)
        hi = statistics.quantiles(miss_scores, n=4)[2] if len(miss_scores) >= 4 else (max(miss_scores) if miss_scores else 0.35)
        rec = round(min(1.0, max(0.05, (lo + hi) / 2)), 3)
    else:
        rec = config.MIN_SCORE
    return {"hit_scores": hit_scores, "miss_scores": miss_scores,
            "recommended": rec, "detail": detail,
            "current": config.MIN_SCORE}


def apply_threshold(value):
    config.MIN_SCORE = float(value)
    _apply_config({"top_k": config.TOP_K,
                   "min_score": float(value),
                   "use_rerank": read_summary()["use_rerank"]})