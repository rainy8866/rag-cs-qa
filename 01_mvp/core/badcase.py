# -*- coding: utf-8 -*-
"""坏例分析：把评估集结果按错误类型归类。"""
CATEGORIES = ["recall_miss", "citation_error", "hallucination", "empty_answer"]


def classify_badcases(gold_items, rec_result=None, cite_result=None, use_rerank=False,
                      use_rewrite=False, use_post_check=False):
    """结合召回与引用的结果，给每条 gold 归类坏例。返回 {counts, items, per_cat}。

    use_rewrite / use_post_check：双口径，透传给兜底再跑的 qa.answer。
    """
    from core import qa as qa_mod

    # 建 问题->是召回命中 的映射
    rec_hit = {}
    if rec_result:
        for m in rec_result.get("missed", []):
            rec_hit[m["question"]] = False
        # 其余视为命中(在 valid 内)
    cite_hit = {}
    if cite_result:
        for d in cite_result.get("details", []):
            cite_hit[d["question"]] = d["correct"]

    counts = {c: 0 for c in CATEGORIES}
    items = {c: [] for c in CATEGORIES}
    for it in gold_items:
        q = it["question"]
        is_rec_miss = (q in rec_hit) and (rec_hit[q] is False)
        cite_ok = (q in cite_hit) and (cite_hit[q] is True)
        if is_rec_miss:
            _record(counts, items, "recall_miss", q, it)
        elif q in cite_hit and not cite_ok:
            _record(counts, items, "citation_error", q, it)
        else:
            # 需要实际跑一次来判断是否为空/兜底
            try:
                res = qa_mod.answer(q, use_rerank=use_rerank,
                                    use_rewrite=use_rewrite, use_post_check=use_post_check)
                if not res.get("success") or not res.get("answer"):
                    _record(counts, items, "empty_answer", q, it)
                elif not res.get("citations") and "资料不足" not in res.get("answer", ""):
                    _record(counts, items, "hallucination", q, it)
            except Exception:
                _record(counts, items, "empty_answer", q, it)

    return {"counts": counts, "items": items, "total": sum(counts.values())}


def _record(counts, items, cat, q, it):
    counts[cat] += 1
    items[cat].append({"question": q, "expected_answer": it.get("expected_answer", ""),
                       "required_snippet": it.get("required_snippet", "")})