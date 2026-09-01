# -*- coding: utf-8 -*-
"""评测集扩容驱动：按评测集PRD §4.4 增量模式执行。"""
import json
from core.gold_set_enhance import enhance

if __name__ == "__main__":
    r = enhance(n_auto=40, n_colloquial=15, n_hard=35, do_semantic_dedup=True)
    print(json.dumps(r, ensure_ascii=False, indent=2))
