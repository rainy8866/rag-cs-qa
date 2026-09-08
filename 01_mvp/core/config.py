# -*- coding: utf-8 -*-
"""RAG 客服问答 - 全局配置与路径约定。"""
from pathlib import Path
import os
from dotenv import load_dotenv

# 项目根 = Desktop\rag-cs-qa
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 各子文件夹
DOC_DIR = PROJECT_ROOT / "04_文档源"          # 原始文档
GOLD_DIR = PROJECT_ROOT / "03_测评集"         # 评估集 & 报告
REPORT_DIR = GOLD_DIR / "reports"
SNAPSHOT_DIR = GOLD_DIR / "snapshots"
OPT_DIR = PROJECT_ROOT / "02_优化"            # 二次优化
COMPARE_DIR = OPT_DIR / "compare"

# MVP 内部目录（在 01_mvp 下）
MVP_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = MVP_DIR / ".vector_store"
ENV_PATH = MVP_DIR / ".env"

# 读取 .env（缺省安静）
load_dotenv(ENV_PATH, override=False)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "")
if HF_ENDPOINT:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
if os.getenv("HF_HUB_OFFLINE"):
    os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 模型与超参
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")  # 可覆盖为 deepseek-chat
LLM_BASE_URL = "https://api.deepseek.com"
EMBED_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 500          # 长段落超过此长度后按句子切
CHUNK_MIN_SIZE = 150      # 段落超过此长度且含多个句子时，按句子切分成更聚焦的块
CHUNK_OVERLAP = 50
TOP_K = 3
RERANK_CANDIDATES = 20    # Rerank 前先放大候选
MIN_SCORE = 0.35          # 相似度阈值闸门（默认，待标定）。注意：一律用 score=1-distance 换算后的相似度
SIMILARITY_METRIC = "cosine"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
COLLECTION_NAME = "docs"

# 评测集语义去重（见 gold_set_enhance._semantic_dedup）
SEMANTIC_DEDUP_THRESHOLD = 0.90   # question 两两余弦相似度 ≥ 此值视为语义重复（经验初值，待实测标定）
GOLD_TARGET = 115                 # 评测集扩容目标条数（参考值）
HARD_CASE_TARGET = 25             # 难例/边界例目标条数（参考值）

# 查询改写与引用后校验（2026-08-29 优化，见 core/query_rewrite.py / qa._post_check_citations）
REWRITE_ENABLED = False           # 是否启用查询改写（拆分子查询多路召回）
REWRITE_MAX_QUERIES = 3           # 最大子查询数（含原问题）
REWRITE_MODEL = "deepseek-chat"   # 改写用 LLM（非推理模型，避免 reasoning 占满 token）
CITATION_CHECK_THRESHOLD = 0.5    # 引用后校验：句子-块余弦相似度阈值（≥此值视为有效引用）

# 文档源多格式 -> md 转换引擎（见 core/convert_docs.py）
# 可选: "markitdown" | "anydoc"
DOC_CONVERT_ENGINE = "markitdown"
# 支持的办公文档后缀（命中才转换，未命中直接忽略不计入统计）
DOC_CONVERT_SUFFIXES = (".docx", ".pdf", ".xlsx", ".pptx")


def ensure_dirs():
    for d in (DOC_DIR, GOLD_DIR, REPORT_DIR, SNAPSHOT_DIR, OPT_DIR,
              COMPARE_DIR, VECTOR_DIR):
        d.mkdir(parents=True, exist_ok=True)


def gold_set_path():
    """评估集 JSON 路径。"""
    return GOLD_DIR / "gold_set.json"


def gold_set_manual_path():
    return GOLD_DIR / "gold_set_manual.json"


def verify_llm() -> bool:
    """发一次最小聊天请求，确认当前 LLM_MODEL 可调用。
    注意：DeepSeek 不支持 models.retrieve() 端点（返回 404），故改用 chat.completions 实测。"""
    if not DEEPSEEK_API_KEY:
        return False
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=LLM_BASE_URL)
    try:
        client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        return True
    except Exception:
        return False