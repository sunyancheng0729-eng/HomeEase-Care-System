# -*- coding: utf-8 -*-
"""启动脚本。

    cd systems
    E:/anaconda3/envs/animedev/python.exe run.py                 # 默认 127.0.0.1:8000
    E:/anaconda3/envs/animedev/python.exe run.py --port 8080

等价于：
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description="加速度单通道跌倒检测系统")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="开发模式热重载")
    a = ap.parse_args()

    import uvicorn
    print(f"\n  加速度单通道跌倒检测系统\n  → http://{a.host}:{a.port}\n")
    uvicorn.run("app.main:app", host=a.host, port=a.port, reload=a.reload, log_level="info")


if __name__ == "__main__":
    main()
