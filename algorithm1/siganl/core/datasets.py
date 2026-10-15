# -*- coding: utf-8 -*-
"""数据源：只读引用父工程 `Fall_Acc_Single/data/urfd/csv` 的 URFD 序列。

**不复制任何数据**进 systems/。系统启动后：
  · 模拟数据模式 —— 从本模块列出的序列中任选一条回放/整段检测；
  · 也支持用户上传自己的 CSV（见 preprocess.read_acc_csv）。

若脱离父工程单独部署，可设置环境变量 `FALL_SYSTEM_DATA_DIR` 指向一份
`{seq}-acc.csv` 目录。
"""
import os
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEMS_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PARENT_ROOT = os.path.abspath(os.path.join(SYSTEMS_ROOT, ".."))

ENV_DIR = "FALL_SYSTEM_DATA_DIR"


def acc_csv_dir():
    d = os.environ.get(ENV_DIR)
    if d and os.path.isdir(d):
        return d
    return os.path.join(PARENT_ROOT, "data", "urfd", "csv")


def is_available():
    return os.path.isdir(acc_csv_dir())


def list_sequences():
    """返回 {'fall': [...], 'adl': [...]}，按序号排序。序列自带类别线索。"""
    d = acc_csv_dir()
    if not os.path.isdir(d):
        return {"fall": [], "adl": []}
    falls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(d, "fall-*-acc.csv")))
    adls = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(d, "adl-*-acc.csv")))
    return {"fall": falls, "adl": adls}


def acc_csv(seq):
    p = os.path.join(acc_csv_dir(), f"{seq}-acc.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"序列不存在: {seq}")
    return p


def sequence_meta(seq):
    """序列元信息（不含任何标注文件，仅由命名推断类别用于演示提示）。"""
    kind = "fall" if str(seq).startswith("fall") else ("adl" if str(seq).startswith("adl") else "unknown")
    return {"seq": seq, "kind": kind,
            "hint": "URFD 跌倒序列" if kind == "fall" else ("URFD 日常动作序列" if kind == "adl" else "自定义序列")}
