# 二次优化日志：每次改动记一行，作为作品集"坏例→优化→对比"的证据。

## 2026-08-29 查询改写 + 引用后校验（坏例驱动优化）
- 改动：`core/query_rewrite.py`（LLM 拆子查询多路召回）+ `retrieval.retrieve_multi` + `qa._post_check_citations`（引用语义后校验，阈值 0.5）；生成/改写模型统一切 `deepseek-chat`；`_run_metrics.py` 并发 4 线程双口径重跑。
- 全量双口径（126 条 · 255 块）：关 87.20%/85.60%/坏例18 → 开 88.00%/88.00%/坏例15（召回 +1、引用 +3、坏例 -3，2 条 citation_error 全消除）。
- 结论（如实）：收益有限。仍坏 15 条期望块在任何子查询 top3 都不出现，属 bi-encoder 排序短板，改写救不了；Rerank 为针对性解法（模型未缓存，二期启用）。报告：`compare/compare_rewrite_20260829_065149.md`（md+json）。
- 预验证：`01_mvp/_validate_rewrite.py` 只读预验证 18 条坏例，仅 5/18 修正；阈值标定 0.5 保留有效引用无误删。

## 2026-08-29 Rerank 落地（bge-reranker-v2-m3 交叉编码器精排）
- 改动：下载 `BAAI/bge-reranker-v2-m3`（HF 镜像 + `HF_HUB_DISABLE_XET=1` 规避 401）；新增 `core/rerank.py`（transformers 直调，规避 FlagEmbedding 旧 API 崩溃被 except 吞掉导致 rerank 静默失效——"rerank 前后没变化"根因）；`retrieval` 支持 `use_rerank`（候选放大 20 再精排）；`_run_metrics.py` 开口径接入 Rerank。
- 排序缺口修复：rerank 精排后曾按 bi-encoder `sort(key=c.score)` 重排、顺序被丢弃，已改为 `use_rerank=True` 时直出精排顺序。
- 全量双口径（126 条 · 255 块，开=改写+Rerank+引用后校验）：关 87.20%/85.60%/坏例18 → 开 **92.00%/90.40%/坏例12**（召回 +6、引用 +6、坏例 -6）；相对无 Rerank 开口径再 +5 召回、+3 引用、-3 坏例。报告：`compare/compare_rewrite_20260829_110735.md`（md+json，含上一轮无 Rerank 对照）。
- 结论（如实）：Rerank 确认为 bi-encoder 排序短板的针对性解法，修好 9 条召回、回归 4 条召回；新增 citation_error ×2 为"答案正确、引用更直接相关块但 ≠ 评估期望块"的边界 case（`_diag_rerank_badcases.py` 根因）；引用/坏例含 ±1 量级 LLM 非确定性波动（召回为确定性指标）。

## 2026-xx-xx 一键优化
- 组合搜索范围: TOP_K=[3,5,8] MIN_SCORE=[0.30,0.35,0.40] use_rerank=[关,开]
- 最优参数: 待填 (recall 最高的一档)
- Recall 变化: 基线 xx% -> 优化后 xx%
- 说明: 只调不动知识库/评估集的参数，保证同一张卷子对比。

## 阈值标定
- 标定推荐 MIN_SCORE: 待填
- 数据依据: 命中 vs 未命中分数分布