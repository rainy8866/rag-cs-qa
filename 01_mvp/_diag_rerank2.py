# -*- coding: utf-8 -*-
"""诊断2：rerank 是否真的改变排序？打印候选池 + 双打分。只读。"""
from core import config, gold_set
from core.gold_set import snippet_to_chunk
from core.vectorstore import query_collection
from core.embeddings import embed_query
from core.rerank import _get_reranker

TEST_QS = [
    "请问你们平台的退货政策是什么？",
    "黑卡会员有哪些专属权益？",
    "为什么退款时要原路退回而不是退到其他账户？",
    "我购买了一件预售商品，支付了定金，但尾款支付时发现商品降价了，我能申请价保吗？差价怎么退？",
]


def main():
    model = _get_reranker()
    gs = gold_set.load_gold_set()
    exp = {}
    for it in gs["items"]:
        cid = snippet_to_chunk(it.get("required_snippet", ""))
        if cid is not None:
            exp[it["question"]] = cid

    for q in TEST_QS:
        qvec = embed_query(config.QUERY_PREFIX + q)
        raw = query_collection(qvec, top_k=config.RERANK_CANDIDATES)
        cands = [(it["id"], round(1.0 - it["distance"], 4), it["text"]) for it in raw]
        cands = [c for c in cands if c[1] >= config.MIN_SCORE]
        # 内容去重（与 retrieval 一致）
        dedup = {}
        for cid, sc, txt in cands:
            key = "".join(txt.split())
            if key not in dedup or sc > dedup[key][0]:
                dedup[key] = (sc, cid)
        uniq = [(cid, sc) for key, (sc, cid) in dedup.items()]

        pairs = [[q, txt] for _, _, txt in cands]
        rs = model.compute_score(pairs)
        if isinstance(rs, float):
            rs = [rs]

        print("=" * 90)
        print(f"Q: {q[:34]}  exp={exp.get(q)}  候选(去重后)={len(uniq)}")
        rows = [(cid, sc, rr) for (cid, sc, txt), rr in zip(cands, rs)]
        rows.sort(key=lambda r: r[1], reverse=True)
        print("  按 bi-encoder 前5:", [(r[0], r[1], round(r[2], 3)) for r in rows[:5]])
        rows.sort(key=lambda r: r[2], reverse=True)
        print("  按 rerank   前5:", [(r[0], round(r[1], 3), round(r[2], 3)) for r in rows[:5]])
        rerank_top3 = [r[0] for r in rows[:3]]
        print(f"  rerank top3={rerank_top3}  exp命中={exp.get(q) in rerank_top3}")


if __name__ == "__main__":
    main()
