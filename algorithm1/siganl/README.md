# 加速度单通道跌倒检测系统（systems/）

> 把 `Fall_Acc_Single` 算法工程判定的**最优识别流 A2** 落地成一个可直接运行的检测系统：
> 持续输入的加速度传感器流（或模拟数据）→ 实时跌倒检测 → SQLite 落库 → 生成可查看 / 可下载的跌倒报告。
>
> 技术栈：**FastAPI + WebSocket + SQLite + 原生 HTML/Canvas**（无前端框架、无 CDN，完全离线可用）。
> 运行环境：`E:/anaconda3/envs/animedev`（Python 3.12），无需安装任何新依赖。

---

## 目录

1. [系统概览](#1-系统概览)
2. [算法：唯一的最优组合 A2](#2-算法唯一的最优组合-a2)
3. [目录结构](#3-目录结构)
4. [快速开始](#4-快速开始)
5. [三种工作模式](#5-三种工作模式)
6. [HTTP 接口文档](#6-http-接口文档)
7. [WebSocket 实时流协议](#7-websocket-实时流协议)
8. [数据库结构](#8-数据库结构)
9. [跌倒报告的内容](#9-跌倒报告的内容)
10. [关键工程实现与取舍](#10-关键工程实现与取舍)
11. [联调与自测脚本](#11-联调与自测脚本)
12. [环境变量](#12-环境变量)
13. [常见问题](#13-常见问题)
14. [与算法工程的一致性验证](#14-与算法工程的一致性验证)

---

## 1. 系统概览

| 编号 | 能力 | 入口 | 说明 |
| --- | --- | --- | --- |
| ① | **传感器实时流** | UI 标签页「① 传感器实时流」/ `WS /ws/stream` | 逐样本持续输入，2 s/0.25 s 滑窗评估；**命中即刻报警**，事件闭合后落库 |
| ② | **模拟数据输入** | UI 标签页「② 模拟数据输入」/ `POST /api/simulate`、`POST /api/simulate/upload` | 对整段数据一次跑完（内置 URFD 序列或上传 CSV），口径与算法工程 exp03 一致 |
| ③ | **记录与报告** | UI 标签页「③ 跌倒记录与报告」/ `GET /api/falls`、`GET /report/{id}` | 列表 / 详情 / 删除 / 清空；报告含**系统报警时间 + 报警前后 3 s 加速度曲线 + 报警标记** |
| ④ | **模型与阈值管理** | `GET /api/model`、`POST /api/threshold` | 查看模型元信息与 39 维特征重要性；运行时改报警阈值（不重训） |

**核心指标（来自算法工程实验三，3 seed × 分层 5 折 CV）**：序列级 AUROC **0.9853 ± 0.0010**，事件准确率 0.826，事件召回 0.967。

> ⚠️ 本系统**只内置最优组合**，不含实验期的其他候选算法/滤波器/特征子集——部署不需要消融开关。
> 所有实验依据、对比数据、鲁棒性分析见父工程 `../report/研究报告.md`。

---

## 2. 算法：唯一的最优组合 A2

### 2.1 识别流（与算法工程逐行一致）

```
三轴加速度（原始 ~62.5 Hz，时间戳不规则）
  → 线性插值重采样到 60 Hz
  → Butterworth 4 阶低通 5 Hz（零相位 filtfilt，保重力基线）
  → 2.0 s 窗 / 0.25 s 步 滑窗
  → 39 维统计特征（motion 13 / impact 8 / orient 8 / freq 7 / calm 3）
  → RF(400) 与 XGB(400) 的 predict_proba 算术平均 = 窗口跌倒概率 p
  → p ≥ 阈值(默认 0.5) 记为命中窗
  → 事件聚合：相邻命中间隔 ≤ 2 窗（0.5 s）合并为一个事件
  → 报警时刻 = 该事件中**最早命中窗口的结束时刻**
```

### 2.2 固化参数（`core/config.py` 是唯一真源）

| 环节 | 参数 |
| --- | --- |
| 重采样 | `RESAMPLE_FS = 60.0` Hz |
| 滤波 | `FILTER_METHOD = butter5`（4 阶低通 5 Hz，零相位、保重力） |
| 滑窗 | `WIN_S = 2.0` s，`STEP_S = 0.25` s |
| 事件合并 | `MERGE_GAP_WIN = 2`（0.5 s） |
| 特征 | 5 族 39 维：`motion / impact / orient / freq / calm` |
| RF | `n_estimators=400, max_depth=10, min_samples_leaf=2, class_weight=balanced` |
| XGB | `n_estimators=400, max_depth=5, lr=0.06, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0` |
| 类别权重 | `scale_pos_weight = 4.9794`（训练集正样本比 16.72%） |
| 报警阈值 | `ALARM_THRESHOLD = 0.5`（固定阈值，零校准） |
| 训练种子 | `TRAIN_SEED = 2026` |

训练产物：`model/best_model.pkl`（joblib 序列化的 RF+XGB）+ `model/meta.json`（参数、特征名、参考指标）。

### 2.3 为什么是 A2（实验三对比）

| 算法 | 序列级 AUROC | 事件准确率 | 说明 |
| --- | --- | --- | --- |
| A1 阈值规则（无学习） | 0.6496 | 0.067 | 权重与阈值耦合 + 滤波削峰，严重失效（`raw` 下可回到 0.9825） |
| **A2 特征 + RF/XGB 概率平均** | **0.9853** | **0.826** | **本系统采用** |
| A3 时序网络（LSTM/GRU/BiLSTM/TCN 软投票） | 0.9411 | 0.777 | 样本量小（1160 窗）喂不饱深度模型 |
| A4 7 基学习器 OOF Stacking | 0.9622 | 0.823 | **集成负收益**：低于其最强基学习器 rf OOF 0.9620 |

A2 同时是**计算量最小**的（单窗推理 ≈23 ms，见 §10.3），因此也是唯一适合实时流的方案。

---

## 3. 目录结构

```
systems/
├── run.py                     # 启动入口（uvicorn 封装）
├── start.bat                  # Windows 一键启动（默认端口 8000）
├── requirements.txt           # 依赖清单（animedev 环境已全部具备）
├── README.md                  # 本文档
│
├── core/                      # ★ 算法核心（从算法工程抽取的最优组合，零实验代码）
│   ├── config.py              #   超参唯一真源
│   ├── preprocess.py          #   重采样 + butter5 零相位滤波 + CSV 读取
│   ├── windowize.py           #   滑窗、窗口标签、命中窗事件合并
│   ├── features.py            #   39 维特征提取（与训练端同一份实现）
│   ├── model.py               #   BestFallModel：RF+XGB 概率平均、推理线程固定、持久化
│   ├── detector.py            #   detect_offline() 整段检测 / StreamingDetector 流式检测
│   └── datasets.py            #   只读引用父工程 URFD 数据目录（不复制数据）
│
├── app/                       # FastAPI 后端
│   ├── main.py                #   路由 + WebSocket 流处理
│   ├── db.py                  #   SQLite 读写（fall_events / fall_samples）
│   ├── schemas.py             #   Pydantic 请求模型
│   └── report.py              #   单文件离线报告 HTML 生成（内联 CSS/JS/Canvas）
│
├── web/                       # 前端（原生，无框架无 CDN）
│   ├── index.html             #   三标签页控制台
│   ├── report.html            #   在线报告页（读取 /api/falls/{id}）
│   ├── style.css
│   ├── chart.js               #   自写 Canvas 绘图（波形 / 概率柱 / 概率曲线）
│   └── app.js                 #   交互逻辑 + WebSocket 客户端
│
├── model/                     # 训练产物
│   ├── best_model.pkl
│   └── meta.json
│
├── scripts/
│   ├── train_best_model.py    # 用父工程 URFD 数据训练并保存最优模型
│   └── test_stream.py         # WS 端到端联调（推流 + 校验）
│
└── runtime/                   # 运行时目录（自动创建，可随时删除）
    ├── fall.db                #   SQLite 数据库
    ├── server.log             #   服务日志
    └── uploads/               #   上传的 CSV 存档
```

---

## 4. 快速开始

### 4.1 环境

无需安装任何包——`animedev` 环境已具备 fastapi / uvicorn / numpy / scipy / pandas / scikit-learn / xgboost / joblib（`sqlite3` 是标准库）。

```bash
# 校准环境（可选）
E:/anaconda3/envs/animedev/python.exe -c "import fastapi, uvicorn, sklearn, xgboost; print('ok')"
```

### 4.2 训练模型（仅首次，或想换数据/种子时）

```bash
cd C:/Users/sprin/Desktop/Fall_Acc_Single/systems
E:/anaconda3/envs/animedev/python.exe scripts/train_best_model.py
```

预期输出：`训练窗数 1160（正样本 16.72%）` → 写出 `model/best_model.pkl` + `model/meta.json`。
数据来源是父工程只读目录 `../data/urfd/csv`（70 序列 = 30 跌倒 + 40 日常），**不会修改任何算法工程文件**。

### 4.3 启动服务

```bash
# 方式一：python 入口（推荐，可指定端口）
E:/anaconda3/envs/animedev/python.exe run.py --port 8011

# 方式二：Windows 双击
start.bat                     # 默认 127.0.0.1:8000
```

打开浏览器访问 **http://127.0.0.1:8011**（或 8000）。

### 4.4 端口与热重载

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--host` | `127.0.0.1` | 监听地址；改成 `0.0.0.0` 可供局域网/手机访问 |
| `--port` | `8000` | 监听端口 |
| `--reload` | 关 | 开发模式热重载（改代码自动重启） |

---

## 5. 三种工作模式

### 5.1 ① 传感器实时流（对应「持续输入的加速度传感器模式」）

界面：选择内置序列 → 选推流倍速（1× 实时 / 2× / 5× / 10× / 30×）→ 「开始推流」。

界面上能看到三条实时视图：

- **实时加速度波形**：滚动显示最近 ~6–20 s 的 ax/ay/az 与模长；
- **窗口概率曲线**：每个被评估窗口的概率柱，红柱 = 命中（≥阈值），橙色虚线 = 阈值；
- **本次会话检出的跌倒**：命中即刻出现「⚠ 报警中…」占位，事件闭合后替换为可点击的记录卡片（点击 → 报告）。

**真实传感器接入**：连 `ws://<host>:<port>/ws/stream`，持续发送

```json
{"type":"sample","t":12.345,"ax":0.12,"ay":0.93,"az":0.21}
```

单位 g，`t` 为秒（单调递增，起点任意）。采样率不限（内部按 60 Hz 插值），建议 ≥50 Hz。
用内置序列推流只是为了**在没有硬件时复现真实时间轴**。
详细协议见 [§7](#7-websocket-实时流协议)。

### 5.2 ② 模拟数据输入（对应「选择模拟数据输入」）

两种方式：

- **内置 URFD 序列**：下拉选择 `fall-01…fall-30` / `adl-01…adl-40` → 「运行整段检测」（默认把检出结果写入数据库）；
- **上传 CSV**：选择本地 CSV → 「上传并检测」。支持两种列序，自动判别：

| 列序 | 含义 | 说明 |
| --- | --- | --- |
| `t_ms, \|a\|, ax, ay, az` | URFD 原生 5 列 | 第 1 列毫秒；`\|a\|` 列会被忽略并按三轴重算模长 |
| `t, ax, ay, az` 或 `t_ms, ax, ay, az` | 通用 4 列 | 第 1 列是秒还是毫秒按相邻间隔自动判断 |

结果面板显示：数据时长 / 窗口数 / 命中窗数 / 检出事件数，以及**逐窗概率图**（x = 窗口结束时刻），
下方列出检出事件，点击任意事件 → 打开报告。

### 5.3 ③ 跌倒记录与报告

列表显示 `ID / 报警时间（系统时间）/ 来源 / 序列或文件 / 报警概率 / 阈值 / 命中窗数`，
每行三个操作：**报告**（在线页）、**下载**（单文件离线 HTML）、**删除**；
顶部可「刷新」「按来源筛选」「清空全部记录」。

---

## 6. HTTP 接口文档

基础地址：`http://127.0.0.1:8011`。所有响应均为 JSON（报告 HTML 除外）。
交互式文档：启动后访问 **`/docs`**（Swagger UI）与 **`/redoc`**。

### 6.1 系统与模型

#### `GET /api/health` — 健康检查与运行配置

**响应**

```json
{
  "status": "ok",
  "error": null,
  "algorithm": "A2_RF+XGB_avg",
  "threshold": 0.5,
  "config": {
    "resample_fs": 60.0, "filter": "butter5", "butter_order": 4, "butter_cutoff": 5.0,
    "win_s": 2.0, "step_s": 0.25, "merge_gap_win": 2,
    "rf":  {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2, "class_weight": "balanced"},
    "xgb": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.06,
            "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0}
  },
  "data_available": true,
  "data_dir": "C:\\Users\\sprin\\Desktop\\Fall_Acc_Single\\data\\urfd\\csv",
  "db": {"total": 3, "sensor": 1, "sim": 2,
         "last_alarm_time": "2026-10-09 20:56:22.487",
         "db_path": "...\\systems\\runtime\\fall.db"}
}
```

`status` 为 `model_missing` 表示模型未训练/未找到（`error` 给出原因），此时流式与模拟接口不可用。

#### `GET /api/model` — 模型元信息与特征重要性

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `top` | int，1–39 | 20 | 返回前 N 个特征的重要性 |

**响应**

```json
{
  "meta": { "algorithm": "A2_RF+XGB_avg", "seed": 2026, "n_features": 39,
            "n_train_windows": 1160, "pos_ratio": 0.167, "trained_at": "...",
            "reference_metrics": {"auroc_seq_mean": 0.98528, "auroc_seq_std": 0.00104,
                                  "event_accuracy": 0.8262, "event_recall": 0.9667}, "...": "..." },
  "top_features": [ {"feature": "jerk_max", "importance": 0.25229},
                    {"feature": "mag_ptp",  "importance": 0.08888}, "..." ]
}
```

#### `GET /api/sequences` — 可用内置序列

```json
{"fall": [{"seq": "fall-01", "kind": "fall", "hint": "URFD 跌倒序列"}, "..."],
 "adl":  [{"seq": "adl-01",  "kind": "adl",  "hint": "URFD 日常动作序列"}, "..."],
 "available": true, "dir": "...\\data\\urfd\\csv"}
```

#### `POST /api/threshold` — 设置运行时报警阈值

```json
// 请求
{"threshold": 0.6}
// 响应
{"threshold": 0.6}
```

阈值是**进程级**状态：影响后续所有检测（含已连接的 WebSocket）；已入库的历史记录保留各自的阈值。

### 6.2 模拟数据输入

#### `POST /api/simulate` — 对内置序列做整段检测

```json
// 请求
{"seq": "fall-01", "threshold": 0.5, "save": true, "include_curve": true}
```

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `seq` | string | — | 内置序列名，必填 |
| `threshold` | float 0–1 | 系统当前阈值 | 本次检测的阈值 |
| `save` | bool | `true` | 是否把检出的事件写入数据库 |
| `include_curve` | bool | `true` | 是否返回逐窗概率曲线 |

**响应**

```json
{
  "duration_s": 5.33, "n_windows": 14, "n_hits": 10, "n_events": 1,
  "threshold": 0.5, "source": "sim", "seq": "fall-01", "label": null,
  "saved_ids": [6],
  "events": [{
    "id": 6, "t_alarm": 2.983, "stream_alarm": 2.983,
    "score": 0.922, "score_max": 1.0, "n_windows": 10,
    "snippet": {"t": [-2.0, "…", 1.0], "ax": [], "ay": [], "az": [], "mag": []}
  }],
  "curve": {"t": [1.983, 2.233, "…"], "p": [0.0133, 0.0523, "…"]}
}
```

- `events[].snippet` 是**已滤波**的报警前后 3 s 加速度（`t` 相对报警时刻，共 181 点 = 3 s × 60 Hz + 1）；
- `curve` 的 `t` 是窗口结束时刻（秒，相对数据起点），与 `p` 等长。

#### `POST /api/simulate/upload` — 上传 CSV 做整段检测

`multipart/form-data`：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `file` | file | — | CSV/TXT，列序见 §5.2 |
| `threshold` | float | 系统当前阈值 | |
| `save` | bool | `true` | |
| `include_curve` | bool | `true` | |

响应结构同 `/api/simulate`，`seq`/`label` 为上传文件名。上传文件会存档到 `runtime/uploads/`。

### 6.3 记录与报告

| 方法 | 路径 | 说明 | 关键参数 |
| --- | --- | --- | --- |
| `GET` | `/api/stats` | 记录统计 | — |
| `GET` | `/api/falls` | 记录列表（按报警时间倒序） | `limit` 1–2000（默认 200）、`source` = `sensor`/`sim` |
| `GET` | `/api/falls/{id}` | 记录详情（含 3 s 切片） | — |
| `DELETE` | `/api/falls/{id}` | 删除一条 | — |
| `DELETE` | `/api/falls` | 清空全部 | — |
| `GET` | `/api/falls/{id}/report.html` | **单文件离线报告**（浏览器内联打开） | — |
| `GET` | `/api/falls/{id}/export.json` | 导出记录 JSON（附件下载） | — |
| `GET` | `/report/{id}` | 在线报告页（前端渲染） | — |
| `GET` | `/` | 控制台 | — |

`GET /api/falls/{id}` 响应示例：

```json
{
  "id": 6, "alarm_time": "2026-10-09 20:56:18.548", "alarm_ts": 1791636978.548,
  "source": "sim", "seq": "fall-01", "stream_alarm": 2.983,
  "score": 0.922, "score_max": 1.0, "score_mean": 0.97, "threshold": 0.5,
  "n_windows": 10, "t_win_start": 0.9833, "t_win_end": 3.4833,
  "duration_s": 5.33, "model": "A2_RF+XGB_avg", "note": null,
  "created_at": "2026-10-09 20:56:18.548",
  "snippet": {"t": [-2.0, "…", 1.0], "ax": ["…"], "ay": ["…"], "az": ["…"], "mag": ["…"]}
}
```

> 字段口径：`alarm_time` 是**报警确认时刻的系统时间**（不是落库时刻，见 §10.4）；
> `stream_alarm` 是同一时刻在**数据流内的相对秒数**，两者关系为 `报警时刻 ≈ 数据流起点 + stream_alarm`。
> 传感器流里前者是真实时钟，后者是对齐到数据的时间轴。

---

## 7. WebSocket 实时流协议

端点：`ws://<host>:<port>/ws/stream`（页面用 `ws://` / `wss://` 按当前协议自动选择）。

### 7.1 客户端 → 服务端

| `type` | 载荷 | 说明 |
| --- | --- | --- |
| `config` | `{"threshold": 0.5}` | 设置本次连接的报警阈值（同时更新进程默认阈值） |
| `meta` | `{"seq": "fall-01"}` | 给后续落库的记录打来源标签 |
| `sample` | `{"t": 1.234, "ax": .., "ay": .., "az": ..}` | **逐个样本推入**；`t` 单位秒，单调递增（起点任意） |
| `flush` | `{}` | 数据流结束：补评尾部窗口 + 输出未闭合事件（落库） |
| `reset` | `{}` | 清空检测状态（缓冲区、当前事件、计数） |

### 7.2 服务端 → 客户端

| `type` | 载荷 | 何时发 |
| --- | --- | --- |
| `ready` | `{threshold, algorithm, config}` | 连接建立后 |
| `config_ok` | `{threshold}` | 收到 `config` 后 |
| `reset_ok` | `{}` | 收到 `reset` 后 |
| `status` | `{t_stream, last_prob, alarming, n_eval, n_samples, n_fall, threshold, curve}` | 推流期间**节流**下发（约每 0.12 s 一次）；`curve` 是**增量**的窗口概率 `[{t,p}, …]` |
| `alarm` | `{t_alarm, t_stream, score, threshold, t_win_start, wall_alarm}` | **命中即刻**（事件尚未闭合）——用于实时报警提示 |
| `fall` | `{id, n_fall, event:{...}, snippet:{...}}` | 事件闭合、已落库；`id` 为数据库记录号 |
| `flushed` | `{ids: [...], n_fall, curve: [...]}` | 收到 `flush` 后；`ids` 为本次收尾落库的记录号 |
| `error` | `{message}` | 字段非法 / 模型未加载 / 未知消息类型 |

### 7.3 两段式报警（重要）

**命中即报**、**闭合才落库**是刻意设计：

1. 任意窗口概率 ≥ 阈值 → 立刻下发 `alarm`（前端出现「⚠ 报警中…」），
2. 之后若 1.0 s 内仍有命中，事件继续；超过 1.0 s 无命中 → 事件闭合 → 下发 `fall`（带 `id`，已落库）。

原因：URFD 序列只有 3–5 s，若等到事件闭合才提示，用户会感觉"推流都结束了才报警"。
真实连续流下两者几乎同时发生（闭合延迟仅 1 s）。`alarm.t_stream` 与 `fall.event.stream_alarm` 一致，前端用时间邻近把占位替换成正式记录。

### 7.4 最小客户端示例（Python）

```python
import json, asyncio, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8011/ws/stream") as ws:
        await ws.send(json.dumps({"type": "config", "threshold": 0.5}))
        await ws.send(json.dumps({"type": "meta", "seq": "sensor-A"}))
        for t, ax, ay, az in sensor_stream:          # 你的传感器循环
            await ws.send(json.dumps({"type": "sample", "t": t,
                                      "ax": ax, "ay": ay, "az": az}))
            msg = json.loads(await ws.recv())        # 实际使用建议单独开协程收
            if msg["type"] == "alarm":
                print("报警！流内 %.2fs 概率 %.3f" % (msg["t_stream"], msg["score"]))
        await ws.send(json.dumps({"type": "flush"}))

asyncio.run(main())
```

---

## 8. 数据库结构

SQLite 文件：`runtime/fall.db`（可用 `FALL_SYSTEM_DB` 覆盖）。两张表：

### 8.1 `fall_events`（每次跌倒一行）

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `id` | INTEGER PK | 记录号 |
| `alarm_time` | TEXT | **报警系统时间**（本地，毫秒精度，`YYYY-MM-DD HH:MM:SS.fff`） |
| `alarm_ts` | REAL | 同上，epoch 秒（便于排序/筛选） |
| `source` | TEXT | `sensor`（实时流）/ `sim`（模拟数据） |
| `seq` | TEXT | 来源序列名或上传文件名 |
| `stream_alarm` | REAL | 报警在数据流内的相对秒数 |
| `score` | REAL | 报警窗口概率（最早命中窗） |
| `score_max` / `score_mean` | REAL | 命中窗的最大 / 平均概率 |
| `threshold` | REAL | 当时的报警阈值 |
| `n_windows` | INTEGER | 命中窗口数 |
| `t_win_start` / `t_win_end` | REAL | 命中窗口区间（数据流内秒数） |
| `duration_s` | REAL | 本次数据总时长 |
| `model` | TEXT | 模型版本标识（如 `A2_RF+XGB_avg`） |
| `note` | TEXT | 备注 |
| `created_at` | TEXT | **落库时刻**（与 `alarm_time` 分开，便于审计延迟） |

索引：`idx_events_time(alarm_ts DESC)`。

### 8.2 `fall_samples`（3 s 切片，每采样点一行）

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `id` | INTEGER PK | |
| `fall_id` | INTEGER FK → `fall_events.id` | 级联删除 |
| `idx` | INTEGER | 切片内序号（0 起） |
| `t_rel` | REAL | 相对报警时刻（秒），范围 `-2.0 … +1.0` |
| `ax` / `ay` / `az` / `mag` | REAL | **已滤波**的三轴与模长（单位 g） |

索引：`idx_samples_fall(fall_id, idx)`。典型一条记录 = 181 行（3 s × 60 Hz + 1）。

### 8.3 直接查库示例

```bash
sqlite3 C:/Users/sprin/Desktop/Fall_Acc_Single/systems/runtime/fall.db \
  "SELECT id, alarm_time, source, seq, ROUND(score,3), n_windows FROM fall_events ORDER BY id DESC LIMIT 10;"
```

---

## 9. 跌倒报告的内容

两种形态，内容一致：

- **在线报告** `GET /report/{id}`：前端 `web/report.html` 读取 `/api/falls/{id}` 渲染；
- **单文件离线报告** `GET /api/falls/{id}/report.html`：服务端 `app/report.py` 生成**完全自包含**的 HTML（CSS/JS 内联，数据内嵌，无任何外部请求），可下载、邮件转发、长期存档，**双击即可看**。

报告包含四部分：

1. **报警摘要**：报警系统时间（毫秒）、报警概率 / 阈值、数据来源、命中窗口数；
2. **报警时刻前后 3 s 加速度曲线**（Canvas 绘制）：
   - 三条轴曲线（ax 蓝 / ay 青 / az 橙）+ 模长 `|a|`（紫，加粗）；
   - **红色虚线标出报警时刻（t = 0）**；
   - **红色底纹标出命中窗口区间**；
   - 下方附加**窗口概率条**：红柱 = 命中，并画出阈值参考线；
3. **记录明细表**：记录 ID、来源、序列/文件、流内报警时刻、命中窗数、三个概率、阈值、数据时长、模型版本、落库时间；
4. **算法上下文**：完整识别流说明 + 模型版本/种子/训练窗数 + 参考性能（AUROC）。

> 报告里的"报警时刻"红线对应**最早命中窗口的结束时刻**，即系统最早可确认跌倒的时刻——这是零校准（固定 0.5 阈值）口径下的定义，与算法工程事件级主口径一致。

---

## 10. 关键工程实现与取舍

### 10.1 零相位滤波的实时化：0.25 s 固定延迟

最优组合用 `filtfilt`（**非因果**零相位滤波）保证离线无相位滞后，但流式无法无延迟使用。本系统的处理：

- 每次评估窗口时，额外向前向后各取 `FILTER_PAD_S = 0.25 s` 上下文，
- 并把评估推迟到"窗口结束时刻 + 0.25 s"才执行。

结果：**流式与整段离线的窗口数值高度一致**（见 §14），代价是 **0.25 s 固定检测延迟**（离线分析无此延迟）。
报告 §7 已把这一部署代价记录在案；若要做真正的低延迟在线部署，需换成因果滤波器（会改变算法数值行为，需重新评测）。

### 10.2 事件闭合延迟：`close_delay = 1.0 s`

事件要等「最后一次命中 + max(合并间隔 0.5 s, 报告后置窗口 1.0 s) = 1.0 s」才闭合，
这样闭合时报告所需的"报警后 1 s 数据"已经到齐，落库即可立刻出报告。`flush` 会强制收尾未闭合事件。

### 10.3 推理线程固定：单窗 100 ms → 23 ms

流式检测每次只评估**一个**窗口，此时 `n_jobs=-1` 只会为单样本创建/同步线程池：

| 环节 | `n_jobs=-1` | `n_jobs=1`（本系统） |
| --- | --- | --- |
| RF(400 树) + XGB(400 树) 单窗 | ≈100 ms | **≈23 ms** |

实测拆分：RF ≈22 ms、XGB ≈0.4 ms、特征提取 ≈0.5 ms。预测值是**确定性的**，
实测多线程与单线程输出**逐位相同**（差 0.00e+00），因此这只是去掉调度开销，不改变算法行为。
训练仍用满核（`n_jobs=-1` 在 `fit` 时生效），只有推理收敛到 `INFER_JOBS = 1`。

> 23 ms ≪ 250 ms（一个滑窗步长），因此只要采样率不显著高于 60 Hz，实时性有充足余量。

### 10.4 报警系统时间的口径

传感器流的事件要等闭合才落库（晚 1 s 以上）。若用落库时刻当"报警时间"会系统性偏晚。
因此 `StreamingDetector` 在**首次命中**时记录 `wall_alarm`（命中窗口被评估的系统时刻），
落库时优先用它作为 `alarm_time`，`created_at` 单独记录落库时刻。

实测：传感器记录 `alarm_time → created_at` 相差 **1.041 s**（≈ `close_delay`），
而模拟数据路径没有 `wall_alarm`，退化为写入时刻（两者相同）。

### 10.5 流结束时补评尾部窗口

正常流式路径只在 `t_eval + 0.25 s` 已到达时才评估窗口，因此**流最后一个 0.25 s 内的窗口**
会缺后向上下文。`flush(complete_tail=True)` 会用"够多少用多少"的上下文把它们补评，
否则恰好在录制末尾发生的跌倒会漏报。实测：补评后流式窗口数与离线**完全相等**（如 fall-01 14/14）。

### 10.6 其他

- **每次调用新开 SQLite 连接** + 线程锁：避免多线程共享句柄；`PRAGMA foreign_keys=ON` 保证级联删除。
- **前端零依赖**：`chart.js` 自写 Canvas 绘图（波形 / 概率柱 / 概率曲线），不引任何 CDN，完全离线可用。
- **数据只读引用**：`core/datasets.py` 只从父工程 `../data/urfd/csv` 读，不复制任何数据进 `systems/`；
  也支持用户上传 CSV。脱离父工程部署时用 `FALL_SYSTEM_DATA_DIR` 指定目录。

---

## 11. 联调与自测脚本

### 11.1 `scripts/test_stream.py` — WebSocket 端到端联调

```bash
E:/anaconda3/envs/animedev/python.exe scripts/test_stream.py fall-01 30   # 30× 倍速（功能联调）
E:/anaconda3/envs/animedev/python.exe scripts/test_stream.py fall-20 1    # 1× 真实时间轴
E:/anaconda3/envs/animedev/python.exe scripts/test_stream.py fall-10 0    # 不限速（尽快推完）
```

脚本会把序列逐样本推进 `/ws/stream`，实时打印 `alarm` / `fall` / `flushed` 三类消息，
最后汇总"实时报警 N 次 / 落库 M 条 / 收到 K 个窗口概率点"。

实测（不限速）：

```
===== fall-01 =====
[推流] fall-01  n=193  dur=5.343s  speed=0.0x
  [推流完毕] 发送 flush，等待服务端收尾…
  ⚠ 实时报警！流内 2.983s  score=0.905
  [flush] 新增记录 ids=[9] 曲线点累计=14
[结果] 实时报警 1 次 / 落库 0 条；收到 3 条状态、14 个窗口概率点

===== adl-01 =====
[推流] adl-01  n=249  dur=5.21s  speed=0.0x
  [推流完毕] 发送 flush，等待服务端收尾…
  [flush] 新增记录 ids=[] 曲线点累计=13
[结果] 实时报警 0 次 / 落库 0 条；收到 3 条状态、13 个窗口概率点
```

> 不限速推流时，服务端要跑完所有窗口评估（每个 ≈23 ms）才会返回 `flushed`，
> 所以"实时报警"通常会打印在"推流完毕"之后——这是客户端的异步时序，不是服务端延迟；
> 用 `1×` 真实时间轴推流时，报警会在推流过程中如期出现（见 §14.3）。

> 注：内置序列只有 3–5 s，事件来不及在流内闭合，记录通常由 `flush` 收尾落库（所以"落库 0 条"但 `flushed.ids` 非空）；
> 真实连续流中会在流内闭合并即时给出 `fall`。

### 11.2 `scripts/train_best_model.py` — 训练与自检

```bash
E:/anaconda3/envs/animedev/python.exe scripts/train_best_model.py
```

从父工程 URFD 数据构建窗口 → 训练 RF+XGB → 打印训练集窗口数/正样本比 → 保存模型与元信息。

---

## 12. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `FALL_SYSTEM_DB` | `systems/runtime/fall.db` | SQLite 数据库路径 |
| `FALL_SYSTEM_DATA_DIR` | `../data/urfd/csv` | 内置序列所在目录（含 `{seq}-acc.csv`） |

`core/config.py` 中的常量（采样率、滤波、窗长、阈值、线程数、报告切片长度等）
是**算法行为的唯一真源**，改它们等于改算法，需同步重训并重新评测。

---

## 13. 常见问题

**Q1：页面打开提示"模型未加载"。**
先运行 `scripts/train_best_model.py`，确认 `model/best_model.pkl` 与 `model/meta.json` 存在，再重启服务。

**Q2：`/api/sequences` 返回空、模拟数据模式不可用。**
`data_available=false` 表示找不到 URFD 数据目录。要么把 `systems/` 放回 `Fall_Acc_Single/` 下，
要么设置 `FALL_SYSTEM_DATA_DIR` 指向含 `{seq}-acc.csv` 的目录。上传 CSV 功能不受影响。

**Q3：端口被占用 / 想换端口。**
`run.py --port 8011`（或改 `start.bat`）。被旧进程占住时：
`Get-NetTCPConnection -LocalPort 8011 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }`。

**Q4：推流很快时，`flushed` 要等一会儿才回来。**
正常。服务端每个窗口要跑一次 RF+XGB（≈23 ms），序列有 6–14 个窗口；脚本会等到 `flushed` 或超时 20 s。

**Q5：能接真实传感器吗？采样率要多少？**
能。连 `/ws/stream` 持续发 `sample` 即可，`t` 为秒。内部按 60 Hz 线性插值，**建议 ≥50 Hz**；
采样率过低会显著改变高频特征（`e_10_25Hz` 等），性能不再保证。

**Q6：报警太灵敏 / 太迟钝。**
用 `POST /api/threshold` 或界面上的"报警阈值"调（0.5 是零校准默认值）。调高→误报少、漏报多。
注意：改阈值只影响判定，不改变模型概率。

**Q7：能拿到 3 s 切片自己画图吗？**
可以。`GET /api/falls/{id}` 的 `snippet`，或直接查 `fall_samples` 表；导出的 JSON（`/export.json`）也可以。

**Q8：报告里的红线为什么不是数据流的第 0 秒？**
红线是**报警时刻**（最早命中窗口的结束时刻），不是数据起点。横轴是"相对报警时刻"的秒数，范围 −2.0 … +1.0。

**Q9：数据库能换 MySQL / PostgreSQL 吗？**
`app/db.py` 是薄封装（两条 SQL + 一个 `conn()` 上下文），替换成本主要在 SQLite 方言（`AUTOINCREMENT`、`?` 占位符）。
练习/演示场景 SQLite 足够（单机、零运维、可随目录拷走）。

---

## 14. 与算法工程的一致性验证

系统必须证明"部署的算法"就是"实验里的算法"。为此做了三层验证（全部实测）：

### 14.1 训练口径一致

| 指标 | 算法工程（exp03） | 本系统 `meta.json` |
| --- | --- | --- |
| 训练窗数 | 1160 | 1160 |
| 正样本比例 | 16.72% | 16.724% |
| `scale_pos_weight` | 4.9794 | 4.9794 |
| 39 维特征名与顺序 | 一致 | 一致 |
| 序列级 AUROC（参考） | 0.9853 ± 0.0010 | 0.98528 ± 0.00104（引用） |

### 14.2 流式 vs 离线（全 70 序列逐窗比对）

| 指标 | 实测值 |
| --- | --- |
| 事件级结果不一致的序列 | **0 / 70** |
| 窗口总数 | 离线 1160 vs 流式 1152（差异来自流起始/结束的上下文边界，`flush` 补评后逐序列相等） |
| 逐窗概率比对点 | 1143 |
| `\|Δp\| ≤ 0.01` | 98.1% |
| `\|Δp\| ≤ 0.05` | 99.7% |
| 最大单窗差 | 0.1475（fall-07 第 5 窗，占 0.3%） |

差异来源明确：离线对整段做零相位滤波，流式在滑动缓冲内做同样运算，
**滤波边界反射**导致个别窗口概率有偏差。事件级结论完全不受影响。

### 14.3 端到端实测样例

| 序列 | 离线 / 流式报警时刻 | 事件数（离线/流式） | 记录 system 时间偏差 |
| --- | --- | --- | --- |
| fall-01 | 2.98 s / 2.98 s | 1 / 1 | 传感器流：`created_at − alarm_time = 1.041 s`（= close_delay） |
| fall-10 | 1.98 s / 1.98 s | 1 / 1 | 模拟数据：0.000 s |
| fall-20 | 1.98 s / 1.98 s | 1 / 1 | — |
| adl-01 | 无 / 无 | 0 / 0 | 无记录（正确不报警） |

---

## 附：本系统的边界

- **只做加速度单通道**。3.1 节的视觉路径（Pose CNN / 视觉运动学特征）是独立工程 `Fall_Video_Single`，
  多模态融合层（3.1 节模块⑤）**尚未实现**，报告 §9.3 已给出设计依据（建议决策级融合）。
- **阈值零校准**：默认 0.5。算法工程里训练折校准过阈值，但事件级主口径采用固定 0.5 以便公平比较；
  不同部署场景建议按误报/漏报偏好调整。
- **实时性代价**：0.25 s 零相位滤波延迟（§10.1）。真正低延迟部署需改因果滤波并重新验证。
- **未做跨域泛化**：所有指标来自 URFD 内部折；换设备/换佩戴位置需重新采集评测（报告 §10.4 列为后续工作）。
