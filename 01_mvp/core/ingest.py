# -*- coding: utf-8 -*-
"""文档入库：读取 md -> 结构感知切块(非固定字符) -> bge-m3 向量化 -> 写 Chroma。"""
import re
from pathlib import Path

from core import config
from core.embeddings import get_embedder


class Chunk:
    def __init__(self, text: str, source: str, chunk_id: int, heading: str = ""):
        self.text = text
        self.source = source
        self.chunk_id = chunk_id
        self.heading = heading

    def to_dict(self):
        return {"id": self.chunk_id, "text": self.text,
                "source": self.source, "heading": self.heading}

    def __repr__(self):
        return f"<Chunk {self.chunk_id} {self.source} {self.heading}>"


# ---- 切块（结构感知，递归下钻，禁止固定字符硬切） ----
_SENT_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_by_sentences(text: str) -> list:
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def _merge_with_overlap(blocks: list, overlap: int, target_size: int = None) -> list:
    """按句子边界切分后的结果仍可能超长，此处用于单个段落内部的句子聚合。
    target_size 为聚合目标长度：默认 config.CHUNK_SIZE（超长兜底切分时用）；
    传入较小值（如 CHUNK_MIN_SIZE）时按更小目标聚合，产生多个更聚焦的小块（细粒度切块）。"""
    target = target_size or config.CHUNK_SIZE
    merged = []
    for b in blocks:
        if not merged:
            merged.append(b)
            continue
        if len(merged[-1]) + len(b) <= target:
            merged[-1] = merged[-1] + b
        else:
            merged.append(merged[-1][-overlap:] + b)
    return merged


def _split_doc_with_headings(doc_path: Path):
    """把标题当作分级 heading 元数据，正文块继承所属标题；标题本身不再单独成块。
    这样避免产生大量"只有标题"的空块导致召回结果全是标题且重复。"""
    content = doc_path.read_text(encoding="utf-8", errors="ignore")
    out = []
    lines = content.splitlines()
    heading_hint = None  # 当前最近标题行文本，供拼接给后代块作上下文
    cur_heading = ""
    para = []

    def flush_para():
        nonlocal para, cur_heading
        body = " ".join(p.strip() for p in para).strip()
        para = []
        if not body:
            return
        if heading_hint:
            body = f"{heading_hint}\n{body}"
        sentences = _split_by_sentences(body)
        # 短段落(≤CHUNK_MIN_SIZE 或单句)整段成块保留上下文；
        # 超过 CHUNK_MIN_SIZE 且含多个句子的段落，即使未超 CHUNK_SIZE 也按句子切分，
        # 并按 CHUNK_MIN_SIZE 聚合为多个聚焦小块，让知识点不被稀释
        if len(body) <= config.CHUNK_SIZE and not (
                len(body) > config.CHUNK_MIN_SIZE and len(sentences) > 1):
            out.append((body, cur_heading))
            return
        merged = _merge_with_overlap(sentences, config.CHUNK_OVERLAP,
                                     target_size=config.CHUNK_MIN_SIZE)
        for mm in merged:
            out.append((mm, cur_heading))

    for line in lines:
        s = line.strip()
        m = _HEADING_RE.match(s)
        if m:
            # 新标题开始：先 flush 上一段正文
            flush_para()
            cur_heading = m.group(2).strip()
            heading_hint = s
            continue
        if not s:
            flush_para()
            continue
        para.append(line)
    flush_para()
    return out


def read_all_docs() -> list:
    """读取 DOC_DIR 下所有 .md，返回 Chunk 列表(带全局递增 id)。"""
    chunks = []
    gid = 0
    files = sorted(config.DOC_DIR.glob("*.md"))
    if not files:
        return chunks
    for f in files:
        if f.name.startswith("."):
            continue
        for (text, heading) in _split_doc_with_headings(f):
            chunks.append(Chunk(text=text, source=f.name,
                                chunk_id=gid, heading=heading))
            gid += 1
    return chunks


def build_index(progress=None) -> dict:
    """重新建档：读取文档 -> 切块 -> embedding -> 写 Chroma。
    progress: 可选回调 fn(done, total) 用于前端进度显示。
    """
    from core.vectorstore import get_collection, reset_collection
    chunks = read_all_docs()
    if not chunks:
        return {"ok": False, "n_chunks": 0, "reason": "04_文档源 下没有 .md 文档"}

    embedder = get_embedder()
    col = reset_collection()

    total = len(chunks)
    texts = [c.text for c in chunks]
    # 分批编码，避免一次吃太多内存
    batch = 16
    ids, sources, headings, embs = [], [], [], []
    for i in range(0, total, batch):
        tb = texts[i:i + batch]
        vecs = embedder.encode(tb, normalize_embeddings=True).tolist()
        for j, c in enumerate(chunks[i:i + batch]):
            ids.append(str(c.chunk_id))
            sources.append(c.source)
            headings.append(c.heading or "")
            embs.append(vecs[j])
        if progress:
            progress(min(i + batch, total), total)

    col.add(
        ids=ids,
        embeddings=embs,
        documents=texts,
        metadatas=[{"source": s, "heading": h}
                   for s, h in zip(sources, headings)],
    )
    return {"ok": True, "n_chunks": total}


if __name__ == "__main__":
    config.ensure_dirs()
    r = build_index()
    print(r)