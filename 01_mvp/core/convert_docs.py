# -*- coding: utf-8 -*-
"""文档源多格式 -> md 转换封装（PRD 方案 A）。

把 04_文档源 下的 .docx/.pdf/.xlsx/.pptx 转成同名 .md，产出可见的中间文件，
供 ingest.py 的 read_all_docs() 直接消费。本模块不改动 ingest.py 的任何逻辑。
"""
from pathlib import Path

from core import config


SUPPORTED_SUFFIXES = config.DOC_CONVERT_SUFFIXES


def _convert_with_markitdown(f: Path) -> str:
    """markitdown 引擎：返回含 # / ## 标题的 Markdown 文本。"""
    from markitdown import MarkItDown
    md = MarkItDown()
    res = md.convert(str(f))
    return res.text_content or ""


def _convert_with_anydoc(f: Path) -> str:
    """firecrawl-anydoc 引擎：返回含 # / ## 标题的 Markdown 文本。"""
    from firecrawl_anydoc import convert_document
    return convert_document(str(f)) or ""


_ENGINES = {
    "markitdown": _convert_with_markitdown,
    "anydoc": _convert_with_anydoc,
}


def _ensure_heading(text: str, fallback_title: str) -> str:
    """防御性兜底：若引擎未吐出任何 # 标题，补一个一级标题以保证切块兼容。
    markitdown/anydoc 默认都会带标题，这里只在异常情况下兜底。"""
    if not text:
        return f"# {fallback_title}\n"
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            return text
    return f"# {fallback_title}\n\n{text}"


def convert_docs(doc_dir: Path, engine: str | None = None) -> dict:
    """扫描 doc_dir 下办公文档，转成同名 .md。

    - engine 缺省取 config.DOC_CONVERT_ENGINE。
    - 同名 .md 已存在则跳过（不覆盖），计入 skipped_existing。
    - 单文件异常不中断整批，记入 errors 并计入 failed。
    - 未命中 SUPPORTED_SUFFIXES 的文件直接忽略，不进任何计数。

    返回 {"converted", "skipped_existing", "failed", "errors"}。
    """
    engine = engine or config.DOC_CONVERT_ENGINE
    if engine not in _ENGINES:
        return {
            "converted": 0,
            "skipped_existing": 0,
            "failed": 0,
            "errors": [f"未知引擎: {engine}（可选: {list(_ENGINES)}）"],
        }

    doc_dir = Path(doc_dir)
    if not doc_dir.exists():
        return {
            "converted": 0,
            "skipped_existing": 0,
            "failed": 0,
            "errors": [f"目录不存在: {doc_dir}"],
        }

    convert_fn = _ENGINES[engine]
    converted = 0
    skipped_existing = 0
    failed = 0
    errors = []

    for f in sorted(doc_dir.iterdir()):
        if not f.is_file():
            continue
        if f.name.startswith("."):
            continue
        if f.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        out_md = f.with_suffix(".md")
        if out_md.exists():
            skipped_existing += 1
            continue

        try:
            text = convert_fn(f)
            text = _ensure_heading(text or "", f.stem)
            out_md.write_text(text, encoding="utf-8")
            converted += 1
        except Exception as e:  # noqa: BLE001 - 单文件失败不影响整批
            failed += 1
            errors.append(f"{f.name}: {e}")

    return {
        "converted": converted,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "errors": errors,
    }


if __name__ == "__main__":
    config.ensure_dirs()
    r = convert_docs(config.DOC_DIR)
    print(r)
