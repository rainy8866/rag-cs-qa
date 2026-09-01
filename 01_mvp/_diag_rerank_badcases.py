# -*- coding: utf-8 -*-
"""Rerank 后新增 citation_error 坏例根因：检索顺序 / 引用块 / 答案三件套 + 排序验证。"""
from core import gold_set
from core.retrieval import retrieve_multi, retrieve
from core.query_rewrite import rewrite
from core import qa as qa_mod

QS = [
    "能否举个例子说明如何修改发票抬头？",
    "退货申请审核通过后，我把商品寄回，仓库多久能完成验收？",
    "为什么部分退款时优惠券只按比例返还，而不是全额返还？",
]


def main():
    gs = gold_set.load_gold_set()
    exp = {}
    for it in gs["items"]:
        cid = gold_set.snippet_to_chunk(it.get("required_snippet", ""))
        if cid is not None:
            exp[it["question"]] = cid

    for q in QS:
        queries = rewrite(q)
        # 返回顺序即 LLM prompt 里看到的顺序（retrieve_multi 的最终返回值）
        cands = retrieve_multi(queries, use_rerank=True)
        res = qa_mod.answer(q, use_rerank=True, use_rewrite=True,
                            use_post_check=True, queries=queries)
        print("=" * 100)
        print(f"Q: {q}")
        print(f"  期望块: {exp.get(q)}")
        print(f"  子查询: {queries}")
        print(f"  最终返回顺序: {[c.id for c in cands]}  (含 rerank_score)")
        for c in cands[:3]:
            rs = getattr(c, "rerank_score", None)
            print(f"    [{c.id}] bi={c.score} rerank={rs if rs is None else round(rs, 3)} :: {c.text[:100]}")
        print(f"  引用块: {[c['chunk_id'] for c in res['citations']]}")
        print(f"  答案: {res['answer'][:420]}")

    # 排序验证：rerank 后是否被 bi-encoder 分重新排序
    print("=" * 100)
    print("[排序验证] 黑卡会员有哪些专属权益？exp=15")
    q = "黑卡会员有哪些专属权益？"
    off = retrieve(q, use_rerank=False)
    on = retrieve(q, use_rerank=True)
    print("  关(纯bi):", [(c.id, c.score) for c in off[:5]])
    print("  开(rerank后):", [(c.id, getattr(c, "rerank_score", None)) for c in on[:5]])


if __name__ == "__main__":
    main()
