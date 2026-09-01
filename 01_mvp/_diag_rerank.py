# -*- coding: utf-8 -*-
"""诊断：rerank 是否真的在跑？对比开/关 rerank 的 top3 集合。只读，不写文件。"""
from core import config, gold_set
from core.gold_set import snippet_to_chunk
from core.retrieval import retrieve

TEST_QS = [
    "请问你们平台的退货政策是什么？",
    "黑卡会员有哪些专属权益？",
    "你们金卡会员都有啥权益呀？",
    "为什么退款时要原路退回而不是退到其他账户？",
]


def main():
    # 1) 尝试加载 reranker，暴露真实异常
    from core.rerank import _get_reranker
    try:
        model = _get_reranker()
        print(f"[reranker 加载成功] {type(model).__name__}")
    except Exception as e:
        print(f"[reranker 加载失败] {type(e).__name__}: {str(e)[:300]}")
        return

    # 2) 对比开/关 rerank 的 top3（集合是否一致）
    gs = gold_set.load_gold_set()
    exp = {}
    for it in gs["items"]:
        cid = snippet_to_chunk(it.get("required_snippet", ""))
        if cid is not None:
            exp[it["question"]] = cid

    for q in TEST_QS:
        off = retrieve(q, use_rerank=False)
        on = retrieve(q, use_rerank=True)
        off_ids = [c.id for c in off]
        on_ids = [c.id for c in on]
        print(f"Q: {q[:28]} exp={exp.get(q)}")
        print(f"  关: {off_ids}  开: {on_ids}  集合相同={set(off_ids)==set(on_ids)}")


if __name__ == "__main__":
    main()
