# -*- coding: utf-8 -*-
"""02_优化/rerank.py — 重排器封装(二期)。
实际实现位于 01_mvp/core/rerank.py；本案列说明用法。"""
import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent.parent / "01_mvp"
sys.path.insert(0, str(CORE_DIR))

from core.rerank import rerank  # noqa: E402,F401


def demo():
    print("Rerank 已就绪。在 retrieval.retrieve(query, use_rerank=True) 时启用。")


if __name__ == "__main__":
    demo()