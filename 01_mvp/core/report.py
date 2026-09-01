# -*- coding: utf-8 -*-
"""指标报告落盘。"""
from datetime import datetime

from core import config


def write_metrics_report(metrics, version: str, extra: str = "") -> str:
    config.ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = metrics["recall"]
    cite = metrics["citation"]
    bc = metrics["badcase"]

    lines = [
        "# RAG 客服问答 — 指标报告",
        f"- 生成时间: {datetime.now().isoformat()}",
        f"- 评估集版本: {version}",
        "",
        "## 一、核心指标",
        f"- Recall@{config.TOP_K}: **{rec['recall']:.2%}** ({rec['hit']}/{rec['total']})",
        f"- 引用准确率: **{cite['accuracy']:.2%}** ({cite['correct']}/{cite['total']})",
        f"- 坏例总数: {bc['total']}",
        "",
    ]
    if rec.get("dropped"):
        lines.append(f"> 注: {len(rec['dropped'])} 条 snippet 未命中任何 chunk 已剔除。")
        lines.append("")
    lines.append("## 二、坏例分类")
    for cat, cnt in bc["counts"].items():
        lines.append(f"- {cat}: {cnt}")
    lines.append("")
    lines.append("## 三、召回未命中明细")
    if rec.get("missed"):
        for m in rec["missed"]:
            lines.append(f"- 问: {m['question']}")
            lines.append(f"  - 应命中: {m['expected_chunk']} | 实际: {m['got_chunks']}")
    else:
        lines.append("- 无")
    lines.append("")
    if extra:
        lines.append(extra)
    body = "\n".join(lines)
    path = config.REPORT_DIR / f"metrics_{ts}.md"
    path.write_text(body, encoding="utf-8")
    return str(path)