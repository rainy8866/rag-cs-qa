# RAG 客服问答平台

一句话：**一个带可量化评测体系的 RAG 客服问答网页** —— 上传文档建知识库 → 对话回答带引用出处 → 用召回率 / 引用准确率 / 坏例三类指标自我证明效果，并支持一键二次优化。

## 架构

```
04_文档源(md)
   │  ingest: 结构感知切块 → bge-m3 向量化
   ▼
ChromaDB 向量库 ──retrieval(cosine, MIN_SCORE)──> top-k 候选
   │                                                   │ Rerank(可开)
   │                                                   ▼
用户问题 ───────────────────────────────→ qa: DeepSeek 生成带引用答案
                                                        │
                                                    评测: Recall@k / 引用准确率 / 坏例
```

## 快速开始
```bash
cd 01_mvp
pip install -r requirements.txt
copy .env.example .env     # 填入 DEEPSEEK_API_KEY；若报模型不存在，把 LLM_MODEL 改 deepseek-chat
# 如果 04_文档源 还没有文档，先确认 offline_*.md 存在；或运行 fetch_docs.py 下载 LangChain 文档
streamlit run app.py       # 访问 http://localhost:8501
```

在「评测与优化」页依次点：重建知识库 → 生成评估集 → 检测召回率 / 引用准确率 / 坏例 → 一键优化 → 导出指标报告。

## 三大指标（项目招牌）
- **Recall@k**：检索层是否把答案需要的原文找出来。
- **引用准确率**：答案真的引用了支撑它的原文，而非编造。
- **坏例分析**：按 `recall_miss / citation_error / hallucination / empty_answer` 归类，驱动二次优化。

最新评测（2026-08-29 · 126 条评估集 · 255 块）：开（查询改写 + Rerank 精排 + 引用后校验）
召回率 **92.00%**、引用准确率 **90.40%**、坏例 12；相对扩容后基线（87.20% / 85.60% / 18）
召回 +5 个百分点、坏例 -6。详见 `../02_优化/compare/compare_rewrite_20260829_110735.md`。

## 文档源
当前知识库 = 客服 FAQ 系列文档（`04_文档源/` 下 16 份 md，覆盖促销价保/会员积分/订单/跨境/运费/
支付/售后等场景）。换成自己的手册只需替换 `04_文档源` 下的文件后重建知识库；docx/pdf/xlsx/pptx
走「🔄 转码文档源」一键转 md 再入库。

## 版本快照
生成评估集时会给当时的文档源做快照（`03_测评集/snapshots/`），保证"改前 vs 改后"是同一份试卷对比。

## 作品集材料
- 指标基线报告：`03_测评集/reports/metrics_*.md`
- 二次优化前后对比：`02_优化/compare/`
- 优化日志：`02_优化/optimize_log.md`
- 演示录屏：见 `05_运行与交付/演示录屏说明.md`
- 二期路线：`二期优化路线.md`