# -*- coding: utf-8 -*-
"""02_优化 模块入口。实际逻辑在 01_mvp/core/optimize.py，这里提供可直接运行的
命令行入口与说明，保证项目结构清晰、Code 模式可识别。"""
import sys
from pathlib import Path

# 让本目录可直接运行并能 import core
CORE_DIR = Path(__file__).resolve().parent.parent / "01_mvp"
sys.path.insert(0, str(CORE_DIR))

from core import config  # noqa: E402
from core.optimize import optimize_auto, calibrate_threshold  # noqa: E402


if __name__ == "__main__":
    config.ensure_dirs()
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    if mode == "calibrate":
        r = calibrate_threshold()
        print("推荐阈值:", r["recommended"], "| 当前:", r["current"])
    else:
        r = optimize_auto()
        if r.get("ok"):
            print("最优先(按 Recall):", r["best"])
        else:
            print("失败:", r)