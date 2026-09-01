# -*- coding: utf-8 -*-
"""指标计算：Recall@k 与 引用准确率。"""
from core import config
from core.gold_set import snippet_to_chunk
from core.retrieval import retrieve


def _valid_items(gold_items):
    """过滤掉 snippet 映射不到的条目的总列表。返回 (valid, dropped)。"""
    valid, dropped = [], []
    for it in gold_items:
        cid = snippet_to_chunk(it.get("required_snippet", ""))
        if cid is None:
            it = dict(it); it["_drop_reason"] = "snippet 未命中任何 chunk"
            dropped.append(it)
        else:
            it = dict(it); it["_chunk_id"] = cid
            valid.append(it)
    return valid, dropped


def recall_at_k(gold_items, top_k: int = None, use_rerank: bool = False,
                use_rewrite: bool = False):
    """计算召回率。返回 {recall, hit, total, missed:[...], valid, dropped}。

    use_rewrite=True 且 config.REWRITE_ENABLED 时：查询改写拆分子查询多路召回（双口径）。
    """
    top_k = top_k or config.TOP_K
    valid, dropped = _valid_items(gold_items)
    hit, misses = 0, []
    for it in valid:
        if use_rewrite and config.REWRITE_ENABLED:
            from core.query_rewrite import rewrite
            from core.retrieval import retrieve_multi
            queries = rewrite(it["question"])
            res = retrieve_multi(queries, top_k=top_k, use_rerank=use_rerank)
        else:
            res = retrieve(it["question"], top_k=top_k, use_rerank=use_rerank)
        hit_ids = {c.id for c in res}
        if it["_chunk_id"] in hit_ids:
            hit += 1
        else:
            misses.append({"question": it["question"],
                           "snippet": it["required_snippet"],
                           "expected_chunk": it["_chunk_id"],
                           "got_chunks": [c.id for c in res]})
    total = len(valid)
    recall = (hit / total) if total else 0.0
    return {"recall": round(recall, 4), "hit": hit, "total": total,
            "missed": misses, "valid": valid, "dropped": dropped}


def citation_accuracy(gold_items, use_rerank: bool = False,
                      use_rewrite: bool = False, use_post_check: bool = False):
    """引用准确率：按 gold 的问答，对答案逐条判断引用是否指向支撑它的 chunk。
    简化规则：一引用若命中该问题的期望 chunk 即判正确。返回统计与明细。

    use_rewrite / use_post_check：双口径（查询改写 / 引用后校验），透传给 qa.answer。
    """
    from core import qa as qa_mod
    valid, dropped = _valid_items(gold_items)
    correct, total, details = 0, 0, []
    for it in valid:
        try:
            res = qa_mod.answer(it["question"], use_rerank=use_rerank,
                                use_rewrite=use_rewrite, use_post_check=use_post_check)
        except Exception as e:
            details.append({"question": it["question"], "error": str(e), "correct": False})
            continue
        refs = [c["chunk_id"] for c in res["citations"]]
        total += len(refs)
        # 期望 chunk 是否被引用
        exp = it["_chunk_id"]
        ok = exp in refs
        if ok:
            correct += 1
        details.append({"question": it["question"], "expected_chunk": exp,
                        "cited_chunks": refs, "correct": ok,
                        "answer": res["answer"][:180]})
    acc = (correct / len(valid)) if valid else 0.0
    return {"accuracy": round(acc, 4), "correct": correct, "total": len(valid),
            "details": details, "dropped": dropped,
            "citation_count": total}


def run_all(gold_items, use_rerank=False, use_rewrite=False, use_post_check=False):
    """一键跑全部三大指标(召回率/引用准确率/坏例)。"""
    rec = recall_at_k(gold_items, use_rerank=use_rerank, use_rewrite=use_rewrite)
    cite = citation_accuracy(gold_items, use_rerank=use_rerank,
                             use_rewrite=use_rewrite, use_post_check=use_post_check)
    from core.badcase import classify_badcases
    bc = classify_badcases(gold_items, rec, cite, use_rerank=use_rerank,
                           use_rewrite=use_rewrite, use_post_check=use_post_check)
    return {"recall": rec, "citation": cite, "badcase": bc}