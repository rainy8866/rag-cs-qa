# -*- coding: utf-8 -*-
"""根因诊断：改写/多路召回为何只修好部分 recall_miss。

对仍坏案例打印：
- 改写出的子查询
- 期望 chunk 在原单路召回里的得分（是否过 MIN_SCORE 闸门）
- 多路召回 top-5 及其得分
- 期望 chunk 在多路召回中的最高得分与名次
只读推理，不写文件。
"""
from core import config, gold_set
from core.gold_set import snippet_to_chunk
from core.retrieval import retrieve, retrieve_multi
from core.query_rewrite import rewrite

STILL_BAD = [
    "请问你们平台的退货政策是什么？",
    "黑卡会员有哪些专属权益？",
    "你们金卡会员都有啥权益呀？",
    "为什么退款时要原路退回而不是退到其他账户？",
    "为什么退款到账时间有时需要1-3个工作日，最长甚至7个工作日？",
    "能否举个例子说明一下，如果我在签收后发现商品有质量问题，退货流程具体是怎样的？",
    "能否举个例子说明一下，什么情况下退货申请会被系统审核不通过？",
    "能否举个例子说明一下，什么情况下可以申请全额退款？",
    "我是黑卡会员，想用免费上门取件服务退一件大件家具，可以用吗？",
    "我是金卡会员，如果我的订单使用了优惠券和积分抵扣，积分是按实付金额还是原价计算？另外，如果这个订单退款了，积分怎么扣？",
    "我购买的跨境商品在运输途中丢失了，客服核实后怎么赔付？退款时限是多久？",
    "我购买了一件预售商品，支付了定金，但尾款支付时发现商品降价了，我能申请价保吗？差价怎么退？",
    "我们公司想开通账期结算，需要满足什么条件？额度是怎么定的？还款逾期了会怎么样？",
]


def main():
    gs = gold_set.load_gold_set()
    exp = {}
    for it in gs["items"]:
        cid = snippet_to_chunk(it.get("required_snippet", ""))
        if cid is not None:
            exp[it["question"]] = cid

    config.REWRITE_ENABLED = True
    for q in STILL_BAD:
        cid = exp.get(q)
        # 1) 原单路召回里期望块的得分
        single = retrieve(q, top_k=config.TOP_K, use_rerank=False)
        single_ids = [c.id for c in single]
        # 2) 改写 + 多路召回
        queries = rewrite(q)
        multi = retrieve_multi(queries, top_k=config.TOP_K, use_rerank=False)
        multi_ids = [c.id for c in multi]
        # 期望块在多路中的最高得分
        from core.vectorstore import query_collection
        from core.embeddings import embed_query
        best_score = None
        for sq in queries:
            qvec = embed_query(config.QUERY_PREFIX + sq)
            raw = query_collection(qvec, top_k=config.TOP_K)
            for item in raw:
                if item["id"] == cid:
                    sim = round(1.0 - item["distance"], 4)
                    best_score = sim if best_score is None else max(best_score, sim)

        print("=" * 80)
        print(f"Q: {q}")
        print(f"  exp={cid}  单路top3={single_ids}  多路top3={multi_ids}")
        print(f"  改写子查询: {queries}")
        print(f"  期望块跨子查询最高分: {best_score}  (MIN_SCORE={config.MIN_SCORE}, TOP_K={config.TOP_K})")
        print(f"  多路top5: {[ (c.id, round(c.score, 3)) for c in retrieve_multi(queries, top_k=5, use_rerank=False) ]}")


if __name__ == "__main__":
    main()
