# -*- coding: utf-8 -*-
"""只读预验证：18 条坏例（v20260829_043818 · Rerank 关）看「查询改写 + 引用后校验」能否修正，
并输出引用后校验逐句相似度用于标定 CITATION_CHECK_THRESHOLD。
只读推理，不写任何评估/快照文件。"""
import json
import re

from core import config, gold_set
from core.gold_set import snippet_to_chunk
from core.retrieval import retrieve_multi
from core.query_rewrite import rewrite
from core.qa import answer
from core.embeddings import embed_query

# 18 条坏例问题（来自 badcases_20260829_050057.md）
BAD_CASES = [
    "请问你们平台的退货政策是什么？",
    "黑卡会员有哪些专属权益？",
    "你们金卡会员都有啥权益呀？",
    "我昨天下的单怎么到现在还没发货呀？",
    "为什么退款时要原路退回而不是退到其他账户？",
    "为什么退款到账时间有时需要1-3个工作日，最长甚至7个工作日？",
    "能否举个例子说明一下，如果我在签收后发现商品有质量问题，退货流程具体是怎样的？",
    "能否举个例子说明一下，什么情况下退货申请会被系统审核不通过？",
    "能否举个例子说明一下，什么情况下可以申请全额退款？",
    "我在大促期间买了一件商品，使用了满减和优惠券，后来申请了部分退款，那优惠券和满减的金额怎么算？",
    "我是黑卡会员，想用免费上门取件服务退一件大件家具，可以用吗？",
    "我是金卡会员，如果我的订单使用了优惠券和积分抵扣，积分是按实付金额还是原价计算？另外，如果这个订单退款了，积分怎么扣？",
    "我购买的跨境商品在运输途中丢失了，客服核实后怎么赔付？退款时限是多久？",
    "我是黑卡会员，想用会员日的三倍积分买一件商品，同时我还有一张平台优惠券，这两者能叠加吗？",
    "我购买了一件预售商品，支付了定金，但尾款支付时发现商品降价了，我能申请价保吗？差价怎么退？",
    "我们公司想开通账期结算，需要满足什么条件？额度是怎么定的？还款逾期了会怎么样？",
    "我买了一件羽绒服，签收后第8天发现内衬有破损，想退货，但已经超过7天了，能退吗？",
    "我买了件衣服，签收后第8天发现尺码不合适，想退货，但已经超过7天了，还能退吗？",
]


def inspect_post_check(ans, chunks):
    """逐句返回 (句文本, 引用编号, 最优块, 最优相似度)。用于阈值标定。"""
    chunk_embs = [embed_query(c.text) for c in chunks]
    rows = []
    for part in re.split(r"(?<=[。！？!?；;])", ans):
        refs = re.findall(r"\[\d+\]", part)
        if not refs:
            continue
        body = re.sub(r"\[\d+\]", "", part).strip()
        if not body:
            continue
        be = embed_query(body)
        sims = [sum(x * y for x, y in zip(be, e)) for e in chunk_embs]
        best_sim = max(sims)
        rows.append({"sentence": body[:40], "refs": [int(r[1:-1]) for r in refs],
                     "best_chunk": sims.index(best_sim) + 1,
                     "best_sim": round(float(best_sim), 4)})
    return rows


def main():
    gs = gold_set.load_gold_set()
    items = gs["items"]
    exp = {}
    for it in items:
        cid = snippet_to_chunk(it.get("required_snippet", ""))
        if cid is not None:
            exp[it["question"]] = cid

    config.REWRITE_ENABLED = True  # 预验证口径：开启查询改写
    fixed_by_rewrite, fixed_by_postcheck = [], []
    still_bad = []
    post_rows = []

    for q in BAD_CASES:
        cid = exp.get(q)
        # 1) 查询改写 + 多路召回
        queries = rewrite(q)
        chunks = retrieve_multi(queries, use_rerank=False, top_k=config.TOP_K)
        hit_ids = [c.id for c in chunks]
        rec_fixed = cid in hit_ids
        # 2) 问答 + 引用后校验
        res = answer(q, use_rewrite=True, use_post_check=True)
        refs = [c["chunk_id"] for c in res["citations"]]
        cite_fixed = cid in refs
        # 3) 引用后校验逐句明细（阈值标定）
        rows = inspect_post_check(res["answer"], res["chunks"])
        post_rows.append({"q": q, "expected": cid, "rows": rows})

        if rec_fixed and cite_fixed:
            fixed_by_postcheck.append(q)
        elif rec_fixed:
            fixed_by_rewrite.append(q)
        else:
            still_bad.append({"q": q, "expected": cid, "got": hit_ids})

    print("=== 预验证汇总（18 条坏例，Rerank 关 + 改写/校验开） ===")
    print(f"改写已修召回+引用全对：{len(fixed_by_postcheck)}")
    print(f"仅改写修好召回：{len(fixed_by_rewrite)}")
    print(f"仍坏：{len(still_bad)}")
    for s in still_bad:
        print(f"  仍坏: {s['q'][:30]} | exp={s['expected']} got={s['got']}")

    print("\n=== 引用后校验逐句明细（阈值标定） ===")
    for item in post_rows:
        for r in item["rows"]:
            flag = "KEEP" if (r["refs"] and r["refs"][0] == r["best_chunk"]) else "FIX/REMOVE"
            print(f"[{flag}] exp={item['expected']} refs={r['refs']} best={r['best_chunk']} "
                  f"sim={r['best_sim']} | {r['sentence']}")


if __name__ == "__main__":
    main()
