# RAG 客服问答平台 — MVP 实现规格

> 本文档是**给 Code 模式读取并照做**的完整实现规格。请严格按本文档生成工程代码。
> 目标：让一个人能在 **2 天内**把一个带"可引用答案 + 评测体系（召回率/引用准确率/坏例分析）"的网页问答项目跑通。

---

## 0. 一次性结论（先读这个）

- 项目名：`rag-cs-qa`
- 定位：个人练手项目，但要求"成熟、可落地、MVP 后能迭代、别人也能用"。
- 本次范围（MVP）：本地网页问答 + 答案带引用出处 + 一份可重复跑的评测报告。
- 大模型：**DeepSeek**（走 API）。默认模型名 `deepseek-v4-flash`，**可在 `.env` 用 `LLM_MODEL` 覆盖**（若平台未提供该型号，改回 `deepseek-chat`）；启动/调用时**须自检模型名**并在不存在时报清晰错误，不得静默失败。Embedding 用本地 **bge-m3**（免费、中英都好）。
- 技术栈：Python 3.11 + **Streamlit(前端，带聊天 + 评测/优化按钮)** + ChromaDB(向量库) + sentence-transformers/FlagEmbedding(bge-m3) + OpenAI 客户端(调 DeepSeek) + 脚本评测。
- 文档源（示例知识库）：**LangChain 官方文档**，用脚本自动下载，另附离线兜底样例。
- 项目位置：**桌面 `Desktop\rag-cs-qa\`**，内分子文件夹：`01_mvp`(应用代码)、`02_优化`、`03_测评集`、`04_文档源`、`05_运行与交付`。

---

## 1. 需求与验收线

### 1.1 这个项目要解决的问题
把散落的公开产品文档，变成一个"输入问题 → 得到带引用来源的稳定答案"的知识问答，并**用量化指标证明自己答得准**。

### 1.2 MVP 验收线（两天做完算达标）
1. 网页能提问，回答中带引用（`[1][2]…`）且前端能折叠展示出处。
2. 能上传示例文档建知识库（至少走通"读文档→切块→向量化→入库"脚本）。
3. 内置评测：跑通 20–30 条评估集，输出**召回率 + 引用准确率 + 坏例清单**三样东西。

### 1.3 二期（本次不做，仅留扩展点）
Rerank 提召回、上传分享页与多人登录、坏例驱动的自动优化 + 改前/改后对比图。

---

## 2. 技术栈与依赖

`requirements.txt`：
```
streamlit
chromadb
sentence-transformers
FlagEmbedding
openai
httpx
beautifulsoup4
markdownify
python-dotenv
```

模型与关键配置：
- LLM：DeepSeek。默认 `LLM_MODEL="deepseek-v4-flash"`，可由 `.env` 覆盖（不存在时报错并提示改用 `deepseek-chat`）；OpenAI 兼容接口：`base_url = https://api.deepseek.com`。
- Embedding：本地模型 `BAAI/bge-m3`（向量维度 1024），首次运行会从 HuggingFace 下载（国内需设 `HF_ENDPOINT=https://hf-mirror.com`）。
- Rerank（二期）：`BAAI/bge-reranker-v2-m3`，依赖 `FlagEmbedding`。
- 向量库：ChromaDB，统一持久化到 **`01_mvp/.vector_store/`**（全文档唯一路径，勿沿用旧的 `data/vector_store`）。
- 密钥：放进 `.env`（`DEEPSEEK_API_KEY=xxx`），代码用 `python-dotenv` 读取。

---

## 3. 目录结构（严格按照这个生成，位于桌面 `Desktop\rag-cs-qa\`）

```
Desktop\rag-cs-qa\
├── 01_mvp\                      # ★ 主应用(Streamlit)
│   ├── app.py                   # 入口：侧边栏导航「智能问答 / 评测与优化」
│   ├── core\
│   │   ├── config.py            # 读取模型/向量库常量 + 各子文件夹路径
│   │   ├── ingest.py            # 读文档 → 切块 → embedding → 写 Chroma
│   │   ├── retrieval.py         # 检索 top-k（返回 chunk 及来源）
│   │   ├── qa.py                # 检索 + 拼 prompt + 生成带引用答案
│   │   ├── gold_set.py          # ★ 评估集自动生成(无需手写)
│   │   ├── metrics.py           # 召回率/引用准确率计算
│   │   └── badcase.py           # 坏例归类
│   ├── ui\
│   │   ├── chat.py              # 「智能问答」页(聊天+引用折叠)
│   │   ├── eval.py              # 「评测与优化」页(按钮+指标仪表盘+对比图)
│   │   └── style.py             # 页面美化(自定义CSS/图标/配色)
│   ├── requirements.txt
│   ├── .env.example             # DEEPSEEK_API_KEY / HF_ENDPOINT 示例
│   └── README.md
├── 02_优化\                     # 二次优化
│   ├── rerank.py                # bge-reranker 重排(提召回)
│   ├── optimize.py              # 一键优化：针对坏例调参/切换检索策略
│   ├── optimize_log.md          # 坏例→优化→改动记录
│   └── compare\                 # 优化前后指标对比图/报告(自动生成)
├── 03_测评集\                   # 评估集 & 报告
│   ├── gold_set.json            # 自动生成的评估集(问题/答案/应命中片段)
│   ├── gold_set_manual.json     # (可选)手工覆盖/补充
│   └── reports\                 # 指标报告(自动生成)
├── 04_文档源\                   # 知识库原始文档
│   ├── fetch_docs.py            # 下载 LangChain 文档
│   └── offline_01.md …          # 离线兜底样例
└── 05_运行与交付\               # 运行说明 + 作品集材料
    ├── 一键运行.bat             # 双击启动(装依赖+起 Streamlit)
    └── 演示录屏说明.md
```

> 路径约定：代码统一从 `01_mvp` 运行；文档源读 `../04_文档源`、评估集读写 `../03_测评集`、优化产物写 `../02_优化`，统一在 `config.py` 用相对项目的常量定义，避免硬编码。

---

## 4. 文档源（示例知识库）怎么拿

### 4.1 `04_文档源/fetch_docs.py`（推荐）
逻辑：
1. GET `https://python.langchain.com/sitemap.xml`。
2. 解析出包含 `/docs/` 的 URL，**取前 30 条**。
3. 逐条下载 HTML → 用 `markdownify` 转成 Markdown → 存为 `04_文档源/doc_<序号>.md`。
4. 容错：单个页面失败就跳过，不影响整体。

### 4.2 离线兜底（文档下载失败时用）
在 `04_文档源/` 里内置至少 3 份手写 Markdown 问答样例，命名 `offline_01.md` 等，主题自拟（如"如何安装/常见配置/API 参数"），每份 5–8 条 Q&A，内容要真实、有可命中的知识点（用于验证检索与引用）。

> 说明：默认用 LangChain 官方文档做示例是因为它公开、免费、可下载。请勿整站爬取无授权站点。

---

## 5. 各模块实现要求（关键契约）

### 5.1 `01_mvp/core/config.py`
- 从 `.env` 读 `DEEPSEEK_API_KEY`、`LLM_MODEL`（默认 `deepseek-v4-flash`，可覆盖为 `deepseek-chat`）、可选的 `HF_ENDPOINT`。
- 常量：`EMBED_MODEL="BAAI/bge-m3"`、`CHUNK_SIZE=500`、`CHUNK_OVERLAP=50`、`TOP_K=5`、`RERANK_CANDIDATES=20`、`COLLECTION_NAME="docs"`、`SIMILARITY_METRIC="cosine"`（统一用 cosine，勿用 Chroma 默认 L2 距离）、`MIN_SCORE=0.35`（**可调参数，默认值仅作起点，需用评估集标定**，见 5.8）。
- 提供 `verify_llm() -> bool`：调一次 DeepSeek 列表/最小请求，若模型名不存在则返回 False，并在日志里给出"请把 `.env` 的 `LLM_MODEL` 改为 `deepseek-chat`"的提示（用于启动自检，不得静默失败）。
- 路径常量（基于项目根 `Desktop\rag-cs-qa\`，用相对 `01_mvp` 的 `..\` 计算）：
  `PROJECT_ROOT`、`DOC_DIR=../04_文档源`、`VECTOR_DIR=01_mvp/.vector_store`、`GOLD_DIR=../03_测评集`、`REPORT_DIR=../03_测评集/reports`、`OPT_DIR=../02_优化`、`COMPARE_DIR=../02_优化/compare`，全部以 `pathlib.Path` 表示。

### 5.2 `01_mvp/core/ingest.py`
- 遍历 `DOC_DIR/*.md`。
- **切块策略（结构感知，禁止固定字符硬切）**：
  1. 先按 Markdown 结构逐级下钻：`## / # 标题 → 段落(空行分隔) → 句子`，优先让每个块对应一个完整章节/段落；
  2. 只有超过 `CHUNK_SIZE` 的长段落，才按**句子边界**切分，并保留 `CHUNK_OVERLAP` 的重叠，避免从一句话中间截断；
  3. 目标：**「一个块 = 一个完整知识点/论点」**，每块约 300–500 字。优先级：语义完整 > 字数达标。
- 每块附带元数据：`{source: 文件名, chunk_id: 序号, heading: 所属标题}`（heading 用于引用时展示"来自哪一节"）。
- > `chunk_id` 是切块时**自动生成的序号**（第 0、1、2…块），不是账号、无需用户注册或填写，仅供程序内部定位引用段落。
- 用 bge-m3 向量化后写入 Chroma collection（持久化到 `VECTOR_DIR`）。
- 可重复执行（先 `collection.delete` 再重建）；在 UI 上提供"重新建档"触发入口。
- 说明：`required_chunk_id` 不再需要用户手工查填，改为运行时由"原文片段"(snippet) 自动映射，见第 6 节。
- 实现提示：可用 `RecursiveCharacterTextSplitter`（分隔符顺序 `["\n\n", "\n", "。", "！", "？", ". ", "! ", "? "]`）或自写同等效果的递归分割器；不要用 `CharacterTextSplitter` 的固定长度方式。

### 5.3 `01_mvp/core/retrieval.py`
- 函数：`retrieve(query: str, top_k: int = TOP_K, use_rerank: bool = False) -> list[Chunk]`
- `Chunk = {id, text, source, heading, score}`。
- **召回方式 = 向量相似度召回**：bge-m3 把 query 和每个文本块编码为向量，全库按 **cosine** 计算并取最相似 top-k。
- **★ 距离↔相似度换算（关键，勿错）**：Chroma 返回的是 **`distance`（cosine 距离，越小越相似，≈ 1 − 相似度）**，不是相似度。代码**必须**做 `score = 1.0 - distance`，后续所有基于 `score` 的逻辑（排序、`MIN_SCORE` 过滤、阈值标定、召回率/引用判定）都使用换算后的相似度。过滤条件统一为 **`score >= MIN_SCORE`**。若用 Chroma 默认 `hnsw:space="cosine"`，其返回即余弦距离，务必记得上转成相似度再判断。
- bge 检索类模型给 query 加统一前缀（如"为这个句子生成表示以用于检索相关文章："）以提高稳定性。
- `use_rerank=True`：先把候选放大到 top `RERANK_CANDIDATES`(20)（向量召回），再调 `02_优化/rerank.py` 精排取最终 top-k。
- 可选升级位（二期）：**混合检索** = 向量召回 + BM25 关键词召回 分数融合（`Reciprocal Rank Fusion`），用于纯向量召回不佳的领域术语场景；本期可暂不做，仅预留接口。

### 5.4 `01_mvp/core/qa.py`
- 函数：`answer(question: str, use_rerank: bool = False) -> {answer: str, citations: list[{chunk_id, source, text}], success: bool}`
- 流程：
  1. `retrieve(question, use_rerank)` 得 top-k chunk。
  2. 拼 prompt（见下 5.5），让模型输出答案并在句末用 `[n]` 标注引用来源。
  3. 解析 `[n]` → 映射回对应 chunk → 组成 `citations`。
  4. 若 top-k 为空或模型不配合，`success=False` 并给出兜底话术（"抱歉，知识库中没有找到相关资料，建议改述或联系人工。"）。
- **引用契约**：告诉模型"只能引用我提供的编号资料，不准编造来源"，并把资料编号与原文一起放入 prompt。

### 5.5 生成答案的 Prompt 模板（照此实现）
```
你是客服问答助手。只依据下面提供的资料回答，严禁编造。
若资料不足，直接回答"资料不足无法确认"。

资料：
[1] (来源: {source}) {text}
[2] ...

问题：{question}

要求：先给结论；用到哪条资料就在对应句末标 [编号]；最后换行列出"参考资料"编号与文件名。
```

### 5.6 「智能问答」页 `ui/chat.py`（Streamlit）
- 用 `st.chat_message` 做聊天气泡；输入用 `st.chat_input`。
- 回答渲染：正文 Markdown + 引用角标 `[1]`，每条用 `st.expander("参考资料 [1] · 文件名")` 折叠原文。
- 侧边栏放开关：`是否用 Rerank`（即"二次优化"开关，默认关，开启后走 `retrieve(use_rerank=True)`）。
- 首用若检测到没有向量库，给出"先去建档"引导按钮。

### 5.7 页面美化 `ui/style.py`
- 用 `st.set_page_config(page_title, page_icon, layout="wide")`。
- 注入自定义 CSS：柔和主色(如 #4F6DF5)、圆角卡片、消息气泡配色、按钮 hover 效果、标题字体层级。
- 侧边栏分组（`st.sidebar.markdown` 分区 + 图标）：导航项「💬 智能问答」「📊 评测与优化」。
- 目标：一眼能看出是"成熟产品界面"，而非白底 demo。

### 5.8 「评测与优化」页 `ui/eval.py`（★ 用户明确要求的按钮区）
按功能区分为若干卡片（`st.container`/`st.expander` 分组），每个按钮行为如下：

**A. 评估集区**
- 按钮「🔄 生成评估集」→ 调 `core/gold_set.py` 生成 `03_测评集/gold_set.json`，完成后显示条数。

**B. 指标检测区（三个按钮，各自独立触发并展示结果）**
- 按钮「🎯 检测召回率」→ `metrics.py` 跑召回率，展示 `st.metric("Recall@5", 0.xx)` + 「未命中明细」表格。
- 按钮「🎯 检测引用准确率」→ `metrics.py` 跑引用准确率，展示 `st.metric("引用准确率", 0.xx)` + 明细表。
- 按钮「🧪 检测坏例」→ `badcase.py` 对评估集归类，左侧显示坏例分类统计（柱状图），下方用 `st.dataframe`/`st.expander` 列出每类 3–5 个代表性坏例。
- 按钮「⚖️ 标定相似度阈值」→ 遍历评估集，把每个问题的 top-k cosine 分数捞出来，画出**"真实命中 vs 真实未命中"的分数分布直方图**（两团分布），在空隙处**自动推荐一个 `MIN_SCORE`**，并把建议值写回 `config` 的配置（可让用户确认后应用）。这是把 `MIN_SCORE=0.35` 这个默认值变成"贴合你的数据"的关键一步。
- 全部跑完显示「📄 导出指标报告」按钮 → 写出 `03_测评集/reports/metrics_<时间戳>.md`。

**C. 二次优化区（呼应你的"二次优化"诉求）**
- 概览：优先展示当前坏例 Top 分类，提示可优化点。
- 按钮「🚀 一键优化」→ 跑 `02_优化/optimize.py`。**严格限定只调"不动向量库/不动评估集"的参数**，保证改前改后是同一张卷子：
  - 只搜索：`TOP_K`（如 3/5/8）、`MIN_SCORE`（如 0.30/0.35/0.40）、`use_rerank`（开/关）；
  - **不得**改切块参数（`CHUNK_SIZE`/`CHUNK_OVERLAP`/embedding 模型）——改它们会 rebuild 全库、`chunk_id` 与评估集映射全失效，与版本快照原则冲突。
- 一键优化自动尝试候选组合、在**同一份** `gold_set` 上评估并选指标最好的一档落地，完成后刷新指标。
- 对比展示：用 Streamlit 图表（`st.bar_chart` 或 plotly）画出**优化前后** 召回率/引用准确率/坏例数对比，并把优化结果写 `02_优化/compare/`。
- 手动开关：`st.toggle("启用 Rerank")` 也会同步到"智能问答"页。
- **改切块的正确入口**（单独、显式）：若用户确实要动切块/embedding，必须走"重建知识库"动作 → 重新 `ingest` → **新建一个版本快照**（新 `gold_set`），而不是走一键优化——这条在 5.8 和 6.0 都要写明，避免操作冲突。

> 单个按钮内用 `st.spinner` + `st.status` 显示执行进度；所有长任务结果写入对应子文件夹，避免页面重跑时丢失。

---

## 6. 评测体系（项目招牌，务必实现）

### 6.0 评估集全自动建立（用户无需手写，无需查 chunk_id）
**核心思路**：`gold_set.py` 用 DeepSeek 根据知识库里**真实存在的原文**自动生成 20–30 条评估样本，每条用 `required_snippet`（一段"必须被检索命中的原文片段"）来标定答案，运行时再把 snippet 映射成 chunk。这样：
- 用户**不用手写**任何一条评估样本；
- 用户**不用填 chunk_id**（代码自己比对 snippet 出现在哪个 chunk）。

**评估集版本快照（★ 二期的基石，一期就要做好）**
- `gold_set.json` 必须带元数据字段：`{"version": "v1", "created_at": "<时间戳>", "doc_set": "<当前文档源指纹>", "params": {"CHUNK_SIZE":..., "EMBED_MODEL":..., "MIN_SCORE":..., "SIMILARITY_METRIC":"cosine"}}`。
- 生成评估集时，**同时把当时的文档源做一份快照**：复制到 `03_测评集/snapshots/v1_<日期>/docs_snapshot/`，并保存基线指标与配置。
- 评测报告必须记录它所用的 `version`，保证"改前 vs 改后"是同一套卷子对比。
- 任何改变向量库的改动（换文档源/切块/embedding）都必须**新建一个版本快照**，绝不在旧版本上覆盖。
- 这样二期做优化时，才有一条可信的"同一卷子不同方案"的对比链。

实现要求 `01_mvp/core/gold_set.py`：
1. 读取 `04_文档源/*.md` 全文。
2. 调用 DeepSeek(`deepseek-v4-flash`)，让它**只根据给定原文**生成 20–30 条记录，每记录：
   ```json
   {"question": "…", "expected_answer": "…", "required_snippet": "必须原样出现在原文中的句子片段"}
   ```
3. 要求模型：`required_snippet` 必须是从原文里**逐字复制**的一段，禁止改写（否则映射会失败）。
4. 结果写 `03_测评集/gold_set.json`（带第 6.0 节版本元数据）；已有文件则不重复生成（可强制重跑）。可选的手写补充放 `03_测评集/gold_set_manual.json`，运行时与自动集合并。
5. 生成时同步做文档源快照到 `03_测评集/snapshots/<version>/docs_snapshot/`。
6. 提供一个函数 `snippet_to_chunk(snippet) -> chunk_id`：遍历入库 chunks，找到 `text` 包含该 snippet 的 chunk，返回其 id；找不到则标 `DROP`（该条从指标里剔除并在报告注明）。

### 6.1 `metrics.py`：召回率 `Recall@k`
- 对每条 gold：跑 `retrieve(question, top_k=TOP_K)`，若 `snippet_to_chunk(required_snippet)` 命中在结果中 → 命中。
- 输出：`Recall@k = 命中数 / 有效条数`，并列"未命中明细表"（问题 + 应命中片段 + 实际前几块）。

### 6.2 `metrics.py`：引用准确率
- 对每条 gold：跑 `answer(question)`。
- 规则化判定（2 天版本就够）：
  - 一条引用的 chunk **在本次 top-k 检索结果中** 且 **与答案句有词级重叠(Jaccard 相似度 ≥ 0.2)** → 判正确；
  - 其余 → 判错误。
- 输出：`CitationAccuracy = 正确引用数 / 总引用数`，并列明细。

### 6.3 `badcase.py`：坏例分析
- 对每条 gold 归类（一经发现即记一类）：
  - `recall_miss`：required chunk 没被检索到。
  - `citation_error`：引用标注错/乱/编造（引用准确率为 0 或引用了未检索到的 chunk）。
  - `hallucination`：答案含镀金内容但无有效引用且自圆其说。
  - `empty_answer`：`success=False` 或空回答。
- 输出：分类统计 + 每类 3–5 个代表性的**原始 问题/答案/引用/判断**（供人工复盘截图）。

### 6.4 汇总报告
输出 `03_测评集/reports/metrics_<时间戳>.md`，内容：三项指标数值 + 坏例分类表 + 未命中明细。**这是作品集里最有说服力的第一张"基线报告"。**

---

## 7. 运行顺序（照做即可复现）
```bash
# 0) 桌面项目根：C:\Users\qzh-0\Desktop\rag-cs-qa
#    在 01_mvp 目录下执行；先把 .env.example 复制为 .env 并填入 DEEPSEEK_API_KEY

cd 01_mvp
pip install -r requirements.txt
# (可选，国内网络可先 set HF_ENDPOINT=https://hf-mirror.com) 首次会下载 bge-m3
# 首次运行前：复制 .env.example → .env，填 DEEPSEEK_API_KEY；若调 /ask 或评测报 model not found，
#            把 .env 的 LLM_MODEL 改成 deepseek-chat（默认 deepseek-v4-flash 不存在时会自检并提示）。

python ../04_文档源/fetch_docs.py    # 拿知识库；失败则用 offline_*.md（可跳过）
python -m core.ingest               # 建向量库（会读取 ../04_文档源）

# 启动 Streamlit（浏览器自动打开 http://localhost:8501）
streamlit run app.py

# 在「评测与优化」页点按钮即可完成：生成评估集 / 召回率 / 引用准确率 / 坏例 / 一键优化。
# 纯命令行等价操作（也可手动复现）：
python -m core.gold_set             # 生成 03_测评集/gold_set.json（需 DEEPSEEK_API_KEY）
python -m core.ingest               # 在建档后确认 chunk 映射
# 指标与优化产物分别落在 03_测评集/reports/ 与 02_优化/compare/
```

---

## 8. 需要我补充/确认后才可继续执行的项（如果缺，代码先生成好占位，跑起来再填）
1. `.env` 里真实 **DEEPSEEK_API_KEY**（必填，问答和 gold_set 自动生成都依赖它；先留空不阻塞搭代码，但跑问答/评测会提示未配置）。
2. 若 `fetch_docs.py` 网络失败，确认使用 `04_文档源/offline_*.md` 兜底（直接可用，无需额外操作）。
3. 首次运行 bge-m3 会从 HuggingFace 下载模型（国内网络可能慢，可设 `HF_ENDPOINT=https://hf-mirror.com`）；与评估集生成无关，正常等待即可。

> 已取消的旧步骤：不再需要用户手写评估集、也不需要手工查填 `required_chunk_id`（已改为 snippet 自动映射）。

---

## 9. 作品集沉淀（`01_mvp/README.md` 要写这些）
- 项目一句话定位 + 架构图（用文字/ASCII 描述：文档→入库→检索→生成→引用）。
- 三大指标基线 + 坏例分类截图（取自 `03_测评集/reports/`）。
- 二次优化前后对比图（取自 `02_优化/compare/`）+ 优化日志。
- 演示录屏链接（3 分钟；说明见 `05_运行与交付/演示录屏说明.md`）。
- 二期路线图（接入更多数据源/多租户/正式前端）。