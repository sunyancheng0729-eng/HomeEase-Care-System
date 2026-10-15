# 实时跌倒检测系统 · 接口与集成文档

> 面向「后续与其他功能融合」的完整技术说明：环境要求、模型依赖、全部接口规格、
> 数据模型、数据库结构、以及三种集成方式的代码示例。
>
> 配套文档：`README.md`（快速上手）。代码：`app.py`（HTTP 层）、`engine.py`（推理引擎）、
> `store.py`（SQLite 持久化）。

---

## 目录

1. [系统架构与数据流](#1-系统架构与数据流)
2. [环境要求](#2-环境要求)
3. [启动与部署](#3-启动与部署)
4. [接口总览](#4-接口总览)
5. [通用约定](#5-通用约定)
6. [接口详细规格](#6-接口详细规格)
7. [数据模型](#7-数据模型)
8. [SQLite 数据库规格](#8-sqlite-数据库规格)
9. [文件与产物约定](#9-文件与产物约定)
10. [与其他功能融合](#10-与其他功能融合)
11. [错误处理与排查](#11-错误处理与排查)
12. [已知限制](#12-已知限制)
13. [A4 权重跨环境兼容性](#13-a4-权重跨环境兼容性)
14. [性能与调优](#14-性能与调优)

---

## 1. 系统架构与数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│                         app.py  (FastAPI, 单进程)                      │
│   HTTP 路由 · MJPEG 流 · 静态页面 · 上传 · 报告导出 · Range 视频分发      │
└───────────────┬──────────────────────────────────────────────────────┘
                │ 直接调用（同进程，非 HTTP）
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    engine.DetectionEngine                             │
│  ┌────────────────┐   ┌──────────────────────────────┐                │
│  │ 主线程          │   │ 后台工作线程 (fall-detector)   │                │
│  │ status()        │←─│ 读帧 → ①YOLO → ②P1姿态 → ③A4   │                │
│  │ start/stop      │  │ → 更新 JPEG / 概率 / 触发录像   │                │
│  └────────────────┘   └──────────────────────────────┘                │
│         │                          │                                  │
│         ▼                          ▼                                  │
│   RecordStore (SQLite)       records/clips/*.mp4                      │
└──────────────────────────────────────────────────────────────────────┘
```

**关键设计点（集成时需了解）**

| 特性 | 说明 |
|---|---|
| 单实例、单会话 | 一个进程同时只跑**一个**检测会话；再次 `start` 会先停掉上一个 |
| 后台线程推理 | 推理在独立线程，HTTP 接口不阻塞；状态通过共享内存 + 锁交换 |
| 模型常驻内存 | 进程启动加载一次（约 5~10 秒），之后不再重新加载 |
| 无外部中间件 | 不需要 Redis / MQ / 数据库服务；SQLite 与文件系统即可 |
| 推流与状态分离 | 画面走 MJPEG（长连接），数值走 `/api/status`（短轮询） |

**检测链路**

```
视频帧 → ① YOLOv8n(ONNX) 取人体框 → ② 自研 P1 小型 U-Net 出 17 关键点
      → ③ A4 混合集成（30 帧滑窗 / 2s）出 P(fall)
      → 连续 confirm 次 ≥ threshold 且不在冷却期 → 触发跌倒事件
      → 记录系统时间 + 落盘 3 秒录像(1s前置+2s后置) + 打报警标记 → 写入 SQLite
```

---

## 2. 环境要求

### 2.1 硬件

| 项目 | 要求 | 说明 |
|---|---|---|
| CPU | 现代 x86_64，4 核以上 | 纯 CPU 可运行，约 3~6 FPS |
| GPU（可选，推荐） | NVIDIA，支持 CUDA 12.x | 装了 CUDA 版 torch 后约 15~25 FPS；可用 `torch.cuda.is_available()` 判断 |
| 内存 | ≥ 4 GB 可用 | 三个模型常驻约 400 MB |
| 磁盘 | ≥ 1 GB 可用 | 录像按需增长（每 3 秒约 0.3 MB） |
| 摄像头（可选） | 任意 UVC 摄像头 | 也可只用视频文件输入 |

### 2.2 操作系统

- Windows 10/11 x64（本项目实测环境）
- 摄像头模式在 Windows 下使用 `cv2.CAP_DSHOW` 后端打开

### 2.3 Python 与依赖

**Python 3.10 ~ 3.12**（实测 3.12）。

> ⚠️ 依赖版本中 **`torch` / `sklearn` / `xgboost` 三者需与 A4 权重文件的导出版本一致**，
> 否则加载 `a4_weights.pkl` 会失败。详见 [第 13 节](#13-a4-权重跨环境兼容性)。

**已实测通过的两套环境**

| 包 | base 环境（`E:/anaconda3/python.exe`） | animedev 环境（`E:/anaconda3/envs/animedev/python.exe`） |
|---|---|---|
| Python | 3.12.13 | 3.12.14 |
| torch | 2.5.1+cu121 | 2.5.1+cu121 |
| opencv-python | 4.11.0 | 5.0.0 |
| numpy | 1.26.4 | 2.5.3 |
| scikit-learn | 1.5.1 | 1.9.0 |
| xgboost | 2.1.1 | 3.4.1 |
| pandas | 2.2.2 | 3.0.5 |
| fastapi | 0.141.1 | 0.141.1 |
| uvicorn | 0.40.0 | 0.52.4 |
| starlette | 1.3.1 | 1.6.0 |
| python-multipart | 0.0.32 | 0.0.32 |

**依赖清单（`requirements.txt` 语义）**

```
# 推理必需
torch>=2.5.1            # ② P1 姿态网络 + ③ A4 时序网络
opencv-python>=4.11.0   # 读帧/画图/写录像/ONNX 推理
numpy>=1.26
scikit-learn>=1.5.1     # A4 基学习器（RF / LR）
xgboost>=2.1.1          # A4 基学习器 + 元学习器
pandas>=2.2             # 读取数据集标注（示例数据源用）

# 服务必需
fastapi>=0.110
uvicorn>=0.29
python-multipart>=0.0.9 # 仅 /api/upload 需要
```

**不需要**：`aiofiles`、`jinja2`、任何数据库服务、任何前端构建工具（前端为原生 HTML/JS）。

安装（以 animedev 为例）：

```bash
E:/anaconda3/envs/animedev/python.exe -m pip install opencv-python xgboost
E:/anaconda3/envs/animedev/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
```

### 2.4 模型权重文件

| 环节 | 文件 | 大小 | 格式 | 加载方 | 版本敏感度 |
|---|---|---|---|---|---|
| ① 检测 | `standalone/yolov8n-pose.onnx` | ~13.5 MB | ONNX | `cv2.dnn.readNetFromONNX` | 低（ONNX 标准格式） |
| ② 姿态 | `standalone/pose_best.pt` | ~10.7 MB | torch `state_dict` | `torch.load(..., weights_only=False)` | 低（纯张量） |
| ③ 时序 | `systems/a4_weights.pkl`（优先）<br>`standalone/a4_weights.pkl`（回退） | ~8.2 MB | pickle | `pickle.load` | **高（sklearn/xgboost 版本）** |

- 三个文件都**随项目分发**，不需要联网下载、不需要重新训练。
- `systems/a4_weights.pkl` 由 `train_weights.py` 在**当前环境**导出；存在时优先使用。
- `status.a4_weights` 字段会返回实际加载的是哪一个，便于排错。

### 2.5 端口与网络

| 项目 | 值 |
|---|---|
| 默认监听 | `127.0.0.1:8000`（仅本机） |
| 可改 | `--host 0.0.0.0 --port 8080` |
| 出网需求 | **无**（离线可运行） |
| 浏览器连接数 | 1 个 MJPEG 长连接 + 1 个状态轮询，远低于同源 6 连接上限 |

### 2.6 目录与路径约定

**所有路径都相对 `systems/` 自动解析**，与启动时的工作目录无关。

| 变量 | 实际路径 |
|---|---|
| `HERE` | `<项目>/systems` |
| `PROJECT` | `<项目>` |
| `STANDALONE` | `<项目>/standalone`（算法与权重，只读复用） |
| `RECORDS_DIR` | `<项目>/systems/records` |
| `CLIPS_DIR` | `<项目>/systems/records/clips` |
| `UPLOADS_DIR` | `<项目>/systems/uploads` |
| `STATIC_DIR` | `<项目>/systems/static` |
| 数据库 | `<项目>/systems/records/falls.db` |

### 2.7 环境自检

```bash
python -c "import torch,cv2,numpy,sklearn,xgboost,pandas,fastapi,uvicorn; \
print('torch',torch.__version__,'cuda',torch.cuda.is_available()); \
print('cv2',cv2.__version__); print('fastapi',fastapi.__version__)"

# 引擎与模型自检（会真实加载三个模型 + 跑 8 秒示例）
python engine.py

# 服务自检
curl http://127.0.0.1:8000/api/health
```

---

## 3. 启动与部署

```bash
cd <项目>/systems

# 首次：在当前环境导出 A4 权重（约 1~3 分钟，只需一次）
E:/anaconda3/envs/animedev/python.exe train_weights.py

# 启动服务
E:/anaconda3/envs/animedev/python.exe app.py
```

**启动参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `127.0.0.1` | 监听地址；`0.0.0.0` 允许局域网访问 |
| `--port` | `8000` | 监听端口 |
| `--threshold` | `0.60` | 跌倒判定阈值（覆盖默认值） |
| `--camera N` | — | 启动后自动打开 N 号摄像头 |
| `--demo DATASET SEQ` | — | 启动后自动跑内置示例，如 `--demo urfd fall-01` |

**浏览器入口**

| 地址 | 页面 |
|---|---|
| `http://127.0.0.1:8000/` | 实时监控台 |
| `http://127.0.0.1:8000/report` | 跌倒报告页 |
| `http://127.0.0.1:8000/docs` | FastAPI 自动生成的 OpenAPI 交互文档 |

**作为后台服务**（Windows）：

```powershell
Start-Process -WindowStyle Hidden E:/anaconda3/envs/animedev/python.exe `
  -ArgumentList "app.py","--host","0.0.0.0","--port","8000" -WorkingDirectory "<项目>/systems"
```

> 建议：进程管理器（NSSM / 计划任务）托管；不要用 uvicorn `--reload`，模型会被重复加载。

---

## 4. 接口总览

| # | 方法 | 路径 | 用途 | 长连接 |
|---|---|---|---|---|
| 1 | GET | `/api/health` | 健康检查 / 模型就绪 | |
| 2 | GET | `/api/status` | 实时状态（概率/报警/FPS/计数） | |
| 3 | GET | `/api/sources` | 可用输入源（内置示例数据集） | |
| 4 | GET | `/api/config` | 读取检测参数 | |
| 5 | POST | `/api/config` | 修改检测参数 | |
| 6 | POST | `/api/start` | 开始检测 | |
| 7 | POST | `/api/stop` | 停止检测 | |
| 8 | GET | `/api/stream` | MJPEG 实时画面 | ✅ |
| 9 | GET | `/api/records` | 跌倒记录列表 | |
| 10 | GET | `/api/records/stats` | 记录统计 | |
| 11 | DELETE | `/api/records/{id}` | 删除单条记录 + 其录像 | |
| 12 | POST | `/api/records/clear` | 清空全部记录 + 录像 | |
| 13 | GET | `/api/clip/{filename}` | 录像播放（支持 Range） | |
| 14 | GET | `/clips/{filename}` | 同上（给导出报告用） | |
| 15 | POST | `/api/upload` | 上传本地视频 | |
| 16 | POST | `/api/report/export` | 导出 HTML 报告 | |
| 17 | GET | `/report_file/{name}` | 查看导出的报告 | |
| 18 | GET | `/` `/report` `/static/{name}` | 页面与静态资源 | |

---

## 5. 通用约定

| 项目 | 约定 |
|---|---|
| Base URL | `http://<host>:<port>` |
| 请求体 | JSON（`Content-Type: application/json`）；仅 `/api/upload` 为 `multipart/form-data` |
| 响应编码 | `application/json; charset=utf-8`，中文不转义 |
| 时间格式 | `YYYY-MM-DD HH:MM:SS`（**服务器本地时间/系统时间**） |
| 时间戳 | 毫秒级 Unix 秒（`epoch`，浮点） |
| 错误格式 | `{"detail": "<原因>"}`；校验失败为 FastAPI 默认 `{"detail":[...]}` |
| 成功状态码 | `200`（删除/导出等无实体时也返回 200 + JSON） |
| 错误状态码 | `400` 参数/环境问题、`404` 资源不存在、`416` Range 越界、`500` 内部错误 |
| 幂等性 | 查询类幂等；`start` 会重置会话内计数（`n_falls`），**不清空历史记录** |
| 并发 | 状态读写线程安全；同时调用 `start` 以最后一次为准 |

**报警（alarm）的两个语义，注意区分：**

| 字段 | 位置 | 语义 |
|---|---|---|
| `alarm` | `/api/status` 的响应 | **瞬时**状态：触发后保持 `alarm_hold_s`（默认 4 秒）为 `true`，用于界面闪烁与提示音 |
| `alarm` | 记录对象 | **永久**标记：入库时固定为 `true`，代表"该录像是一次报警事件" |

---

## 6. 接口详细规格

### 6.1 `GET /api/health`

服务与模型就绪检查（用于启动等待/监控探针）。

**响应**

```json
{ "ok": true, "models_ready": true }
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | bool | 服务存活 |
| `models_ready` | bool | 三个模型是否加载成功 |

> 服务启动即加载模型（5~10 秒）。集成方可在启动后轮询此接口直至 `models_ready=true`。

---

### 6.2 `GET /api/status`

实时状态，推荐的轮询接口（建议 300~500 ms 一次）。

**响应示例**

```json
{
  "state": "running",
  "error": null,
  "models_ready": true,
  "load_error": null,
  "a4_weights": "standalone/a4_weights.pkl",
  "source": "urfd/fall-01",
  "source_spec": { "type": "demo", "dataset": "urfd", "seq": "fall-01" },
  "prob": 0.9913,
  "peak": 0.9991,
  "alarm": true,
  "threshold": 0.6,
  "frames": 80,
  "fps": 14.7,
  "elapsed": 5.3,
  "n_falls": 2,
  "recording": false,
  "n_records": 6,
  "last_record": { "id": 7, "time": "2026-10-09 20:24:59", "clip": "fall_....mp4", "...": "见第 7.2 节" },
  "gt": [[2.77, 4.77]],
  "session": 3
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `state` | string | `idle` 待机 / `running` 检测中 / `finished` 源结束 / `error` 出错 |
| `error` | string \| null | 运行期异常（如读帧失败） |
| `models_ready` | bool | 模型是否可用 |
| `load_error` | string \| null | 模型加载失败原因（含修复提示） |
| `a4_weights` | string | 实际加载的 A4 权重来源 |
| `source` | string | 当前输入源可读名（`摄像头 0` / `urfd/fall-01` / 文件名） |
| `source_spec` | object \| null | 当前输入源原始描述（见 7.4） |
| `prob` | float | **当前**窗口 P(fall)，0~1 |
| `peak` | float | 本会话峰值概率 |
| `alarm` | bool | 是否处于报警保持期（瞬时） |
| `threshold` | float | 当前判定阈值 |
| `frames` | int | 本会话已处理帧数 |
| `fps` | float | 有效推理速率（**最近 2 秒滑动窗口**，不含首帧预热；会话结束后保留最后一次读数） |
| `elapsed` | float | 运行秒数（结束后 = 帧数 / 15） |
| `n_falls` | int | **本会话**命中次数 |
| `recording` | bool | 是否正在采集录像（触发后的 2 秒窗口内为 `true`） |
| `n_records` | int | 数据库中的**历史记录总数** |
| `last_record` | object \| null | 最新一条记录（用于"发现新记录"事件，见 10.5） |
| `gt` | array | 仅示例数据源：数据集标注区间 `[[起,止],...]`，供对照，**不参与判定** |
| `session` | int | 会话自增号，每次 `start` +1 |

---

### 6.3 `GET /api/sources`

列出可用输入源（用于前端下拉框）。

**响应示例**

```json
{
  "demo": {
    "urfd": ["adl-01", "fall-01", "..."],
    "le2i": ["Coffee_room_01_video1", "..."],
    "mc":   ["chute01", "chute02", "chute03"]
  },
  "camera_hint": "默认摄像头 index=0；笔记本内置摄像头通常是 0，外接设备依次为 1、2 ..."
}
```

实测数量：`urfd` 70 个序列、`le2i` 130 个、`mc` 24 个。
若某数据集本机不存在，对应 key 不出现（不报错）。

---

### 6.4 `GET /api/config`

**响应**

```json
{
  "threshold": 0.6,
  "cooldown_s": 6.0,
  "confirm": 2,
  "clip_fps": 15.0,
  "pre_frames": 15,
  "post_frames": 30,
  "detect_every": 3,
  "prob_every": 2,
  "show_hud": false,
  "show_skeleton": true,
  "clip_seconds": 3.0
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `threshold` | float | 跌倒判定阈值（0.05~0.99） |
| `cooldown_s` | float | 报警冷却：同一次跌倒在该时间内不重复记录 |
| `confirm` | int | 需要连续多少个窗口 ≥ 阈值才触发（抗单帧抖动） |
| `clip_fps` | float | 录像采集参考帧率（15） |
| `pre_frames` / `post_frames` | int | 前置/后置帧数（默认 15 / 30 = 1s + 2s） |
| `detect_every` | int | 每 N 帧做一次 YOLO 人体检测，其余帧复用上次的框（性能参数） |
| `prob_every` | int | 每 N 帧做一次 A4 时序判定，其余帧沿用上次概率（性能参数） |
| `show_hud` | bool | 是否把「顶部信息条 + `P(fall)` 文字 + 概率条」烧进画面（**默认 false**） |
| `show_skeleton` | bool | 是否把检测框 + 骨架烧进画面（默认 true） |
| `clip_seconds` | float | 录像总时长（= (pre+post)/clip_fps，默认 3.0） |

> `show_hud` / `show_skeleton` **只影响显示与录像的观感，与模型输入无关**：
> 送进 YOLO / P1 / A4 的永远是原始帧，叠加发生在三段推理之后（见 §14.1）。

---

### 6.5 `POST /api/config`

修改检测参数（**可运行中热更新**，不影响已入库记录）。

**请求体**（字段全部可选，只改传入的）

```json
{ "threshold": 0.55, "cooldown_s": 8, "confirm": 3, "pre_s": 1.0, "post_s": 2.0 }
```

| 字段 | 类型 | 范围 | 说明 |
|---|---|---|---|
| `threshold` | float | 0.05~0.99 | 自动截断 |
| `cooldown_s` | float | ≥ 0 | 秒 |
| `confirm` | int | ≥ 1 | 连续窗口数 |
| `pre_s` | float | ≥ 0 | 前置录像秒数 |
| `post_s` | float | ≥ 0 | 后置录像秒数 |
| `show_hud` | bool | — | 烧录顶部信息条 + 文字 + 概率条（默认 false） |
| `show_skeleton` | bool | — | 烧录检测框 + 骨架（默认 true） |

**响应**：同 6.4 的 config 对象。

> 提示：要保证"3 秒录像"，应满足 `pre_s + post_s = 3`。前端滑杆就是这么联动的。
>
> `detect_every` / `prob_every` 属于**启动期**性能参数（在构造 `DetectionEngine` 时传入，
> `app.py` 里可改默认值），不通过 `/api/config` 热改，避免运行中改变时序判定节奏。

---

### 6.6 `POST /api/start`

开始检测。**会先自动停止当前会话。**

**请求体**（按 `type` 三选一）

```json
// ① 摄像头
{ "type": "camera", "index": 0 }

// ② 本地视频文件（绝对路径；也可先用 /api/upload 拿到 path）
{ "type": "file", "path": "C:/videos/fall.mp4" }

// ③ 内置示例数据集
{ "type": "demo", "dataset": "urfd", "seq": "fall-01" }
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `type` | string | ✅ | `camera` / `file` / `demo` |
| `index` | int | camera | 摄像头序号，默认 0 |
| `path` | string | file | 视频绝对路径（mp4/avi/mov…） |
| `dataset` / `seq` | string | demo | 来自 `/api/sources` |

**响应**：Status 对象（`state` 变为 `running`）。

**错误**

| 状态码 | 场景 | `detail` 示例 |
|---|---|---|
| 400 | 类型非法 | `type 必须是 camera / file / demo` |
| 400 | 摄像头打不开 | `无法打开摄像头（index=0），请检查设备占用或权限` |
| 400 | 文件不存在 | `视频文件不存在：C:/x.mp4` |
| 400 | 模型未就绪 | `模型未就绪：...。若为 A4 权重版本不兼容，请先运行 python train_weights.py ...` |

**示例**

```bash
curl -X POST http://127.0.0.1:8000/api/start \
  -H "Content-Type: application/json" \
  -d '{"type":"camera","index":0}'
```

---

### 6.7 `POST /api/stop`

停止检测（会正常收尾正在进行的录像）。

**请求体**：无
**响应**：Status 对象（`state` 变为 `idle`；若视频自然播完则为 `finished`）

> 若停止时录像尚未收满 3 秒，也会把已采集的部分落盘入库（不会被丢弃）。

---

### 6.8 `GET /api/stream`

MJPEG 实时画面，**带叠加层**（检测框 + COCO-17 骨架 + 顶部信息条 + 底部概率条）。

| 项目 | 值 |
|---|---|
| 响应类型 | `multipart/x-mixed-replace; boundary=fvs_frame` |
| 帧格式 | JPEG（质量 80） |
| 推送速率 | ≈ 30 帧/秒（空闲时推占位图 `NO SIGNAL`） |
| 分辨率 | 跟随输入源（演示源 640×480） |
| 连接特性 | **长连接，永不结束**；断开即可（无需特殊请求） |

**浏览器用法**

```html
<img src="http://127.0.0.1:8000/api/stream" alt="实时画面">
```

**Python 抓单帧**

```python
import urllib.request
r = urllib.request.urlopen('http://127.0.0.1:8000/api/stream', timeout=8)
buf = b''
while True:
    buf += r.read(8192)
    j, k = buf.find(b'\xff\xd8'), buf.find(b'\xff\xd9')
    if j >= 0 and k > j:
        open('frame.jpg', 'wb').write(buf[j:k+2]); break
r.close()
```

> ⚠️ 该接口占用 1 个浏览器连接且不结束；同源最多 6 个连接，不要重复打开多个 `<img>`。

---

### 6.9 `GET /api/records`

跌倒记录列表（SQLite 读取，**新的在前**）。

**查询参数**

| 参数 | 类型 | 说明 |
|---|---|---|
| `limit` | int | 可选，只返回前 n 条 |

**响应**

```json
{
  "count": 6,
  "records": [ { "id": 7, "time": "2026-10-09 20:24:59", "...": "见 7.2" } ]
}
```

---

### 6.10 `GET /api/records/stats`

**响应**

```json
{ "count": 6, "max_peak": 0.9991, "first_epoch": 1791547953.6, "last_epoch": 1791548699.9 }
```

| 字段 | 说明 |
|---|---|
| `count` | 总条数 |
| `max_peak` | 历史最高峰值概率（无记录为 `null`） |
| `first_epoch` / `last_epoch` | 最早/最晚记录时间戳 |

---

### 6.11 `DELETE /api/records/{id}`

删除一条记录，**并同步删除其录像文件**。

**路径参数**：`id`（整数，来自记录对象的 `id`）

**响应**

```json
{ "ok": true, "deleted": 5, "clip_removed": true }
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `deleted` | int | 被删除的记录 id |
| `clip_removed` | bool | 录像文件是否**真的**删掉了 |

> `clip_removed: false` 表示记录已从数据库删除，但 mp4 文件被运行环境的批量删除保护拦下，
> 变成了**孤儿文件**（已不被任何记录引用，不影响功能）。
> 这些残留会在下次 `POST /api/records/clear`（§6.12）时被一并清理，前端也会给出提示。

**错误**：`404 记录不存在`；`400 记录 id 必须是整数`

---

### 6.12 `POST /api/records/clear`

清空全部记录与录像（不可撤销）。

**响应**

```json
{ "ok": true, "count": 0, "deleted": 6, "clips_removed": 6, "clips_left": 0 }
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `deleted` | int | 本次删除的记录条数 |
| `clips_removed` | int | 本次真正删掉的录像文件数 |
| `clips_left` | int | `records/clips/` 里还剩多少录像文件（见下） |
| `count` | int | 恒为 0（清空后的记录数，兼容旧版字段） |

> **为什么录像要分批删？** 一条记录配一个 mp4，记录攒多了就是几十上百个文件。
> 运行环境带有「一轮内批量删除超过 50 个文件」的安全保护（触发时抛 `SystemExit`），
> 所以这里单次最多删 `engine.DetectionEngine.MAX_CLIP_DELETES`（默认 40）个，
> 并把每个删除动作都单独兜住异常 —— 保证接口**永远返回 200**，不会因为保护拦下而 500。
>
> **返回 `clips_left > 0` 说明还有残留录像**（数据库记录已清空，只是文件没删完）。
> 前端会提示「再点一次『清空记录』即可清除」；再调一次本接口即可继续删。
> 残留文件不影响功能，它们已不被任何记录引用。

---

### 6.13 `GET /api/clip/{filename}`

3 秒录像播放。**支持 HTTP Range**，浏览器 `<video>` 可拖动进度条。

| 项目 | 值 |
|---|---|
| `Content-Type` | `video/mp4` |
| 完整请求 | `200` + `Accept-Ranges: bytes` |
| 带 `Range: bytes=a-b` | `206 Partial Content` + `Content-Range: bytes a-b/size` |
| 越界 | `416` |

**示例**

```bash
curl -H "Range: bytes=0-1023" -D - -o /dev/null \
  http://127.0.0.1:8000/api/clip/fall_20261009_202409.mp4
# HTTP/1.1 206 Partial Content
# content-range: bytes 0-1023/321603
```

安全性：文件名经 `os.path.basename` 处理，不可穿越目录。

---

### 6.14 `GET /clips/{filename}`

与 6.13 等价。存在的意义：导出的 HTML 报告在同一目录用**相对路径** `clips/xxx.mp4` 引用录像，
该路径会解析到此接口。

---

### 6.15 `POST /api/upload`

上传本地视频，返回服务器路径，可直接用于 `start` 的 `file` 源。

**请求**：`multipart/form-data`，字段名 `file`

```bash
curl -F "file=@D:/fall.mp4" http://127.0.0.1:8000/api/upload
```

**响应**

```json
{ "path": "<项目>/systems/uploads/fall.mp4", "name": "fall.mp4", "size": 10485760 }
```

**错误**：`400 文件名无效` / `400 上传的文件为空`

> 上传文件保存在 `systems/uploads/`，不会自动清理。

---

### 6.16 `POST /api/report/export`

把当前全部记录导出成一个**自包含 HTML 报告**（含时间、概率、报警标记、内嵌 3 秒录像）。

**请求体**：无

**响应**

```json
{
  "ok": true,
  "name": "report_20261009_201308.html",
  "url": "/report_file/report_20261009_201308.html",
  "path": "<项目>/systems/records/report_20261009_201308.html",
  "n": 6
}
```

导出文件与 `clips/` 同级，因此**拷贝整个 `records/` 目录即可分享**（录像随报告一起走）。

---

### 6.17 `GET /report_file/{name}`

查看已导出的报告（`name` 必须以 `.html` 结尾）。

---

### 6.18 页面路由

| 路径 | 内容 |
|---|---|
| `GET /` | 实时监控台（`static/index.html`） |
| `GET /report` | 跌倒报告页（`static/report.html`） |
| `GET /static/{name}` | CSS / JS 静态资源（带 `no-cache`） |
| `GET /favicon.ico` | 204 空响应 |
| `GET /docs` | FastAPI 自动文档（OpenAPI） |

---

## 7. 数据模型

### 7.1 Status 对象

见 [6.2 字段表](#62-get-apistatus)。关键字段：`state` / `prob` / `alarm` / `n_records` / `last_record`。

### 7.2 Record 对象（跌倒记录）

```json
{
  "id": 7,
  "time": "2026-10-09 20:24:59",
  "epoch": 1791548699.921292,
  "prob": 0.997,
  "peak": 0.9977,
  "clip": "fall_20261009_202459.mp4",
  "clip_url": "/api/clip/fall_20261009_202459.mp4",
  "duration_s": 3.01,
  "frames": 15,
  "fps": 4.98,
  "threshold": 0.55,
  "source": "urfd/fall-01",
  "alarm": true,
  "created_at": "2026-10-09 20:24:59"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | int | 数据库主键，删除接口用它 |
| `time` | string | **跌倒时间（系统时间）** |
| `epoch` | float | 同一时间的时间戳，用于排序/筛选 |
| `prob` | float | 触发瞬间的概率 |
| `peak` | float | 该次事件内的峰值概率 |
| `clip` | string | 3 秒录像文件名 |
| `clip_url` | string | 播放地址（已拼好，可直接给 `<video src>`） |
| `duration_s` | float | 录像实际时长（≈3.0） |
| `frames` / `fps` | int / float | 录像帧数与标称帧率 |
| `threshold` | float | 触发时使用的阈值（便于回溯） |
| `source` | string | 输入源标识 |
| `alarm` | bool | 报警标记（恒为 `true`） |
| `created_at` | string | 入库时间 |

### 7.3 Config 对象

见 [6.4](#64-get-apiconfig)。

### 7.4 SourceSpec（输入源描述）

```json
{ "type": "camera", "index": 0 }
{ "type": "file",   "path": "C:/videos/fall.mp4" }
{ "type": "demo",   "dataset": "urfd", "seq": "fall-01" }
```

---

## 8. SQLite 数据库规格

**位置**：`systems/records/falls.db`（由 `store.py` 自动创建，无需初始化脚本）

```sql
CREATE TABLE IF NOT EXISTS falls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fall_time   TEXT    NOT NULL,   -- 跌倒时间（系统时间）
    epoch       REAL,               -- 时间戳
    prob        REAL,               -- 触发概率
    peak        REAL,               -- 峰值概率
    clip        TEXT,               -- 录像文件名
    duration_s  REAL,               -- 录像时长
    frames      INTEGER,            -- 录像帧数
    fps         REAL,               -- 录像标称帧率
    threshold   REAL,               -- 触发阈值
    source      TEXT,               -- 输入源
    alarm       INTEGER DEFAULT 1,  -- 报警标记
    created_at  TEXT                -- 入库时间
);
CREATE INDEX IF NOT EXISTS idx_falls_epoch ON falls(epoch DESC);
```

**并发与事务**：每次操作独立连接（`timeout=15`）+ 进程内 `RLock`；`with conn` 自动提交。

**兼容迁移**：首次启动时若发现旧的 `records/falls.json` 且库中无数据，会自动导入并重命名为
`falls.json.bak`（日志输出 `[store] 已把旧 falls.json 的 N 条记录迁移进 SQLite`）。

**常用查询**

```sql
-- 今日跌倒
SELECT id, fall_time, prob, peak, clip FROM falls
WHERE date(fall_time) = date('now', 'localtime') ORDER BY epoch DESC;

-- 高峰值事件
SELECT fall_time, peak, source FROM falls WHERE peak > 0.9 ORDER BY peak DESC;

-- 按输入源统计
SELECT source, COUNT(*) n, ROUND(AVG(peak),3) avg_peak FROM falls GROUP BY source;
```

```bash
sqlite3 systems/records/falls.db "SELECT COUNT(*) FROM falls"
```

> 若后续要换成别的存储（MySQL/PG/上层业务库），只需替换 `store.RecordStore`
> 的 6 个方法（`add/list/latest/count/delete/clear`），HTTP 层无需改动。

---

## 9. 文件与产物约定

```
systems/
├── app.py  engine.py  store.py  train_weights.py
├── a4_weights.pkl                 # 当前环境导出的 A4 权重（可选）
├── static/                        # 前端（index/report + css/js）
├── records/
│   ├── falls.db                   # 记录数据库（SQLite）
│   ├── falls.json.bak             # 旧 JSON（迁移后自动重命名）
│   ├── clips/fall_YYYYmmdd_HHMMSS.mp4   # 3 秒录像
│   └── report_YYYYmmdd_HHMMSS.html      # 导出的报告
└── uploads/                       # 网页上传的视频
```

**命名规则**

| 产物 | 规则 |
|---|---|
| 录像 | `fall_<yyyyMMdd_HHMMSS>.mp4`（触发时刻） |
| 报告 | `report_<yyyyMMdd_HHMMSS>.html`（导出时刻） |

**录像写入机制**：触发时取"前置缓冲"（默认最近 1 秒的已编码帧）+ 之后 2 秒的帧，
按**真实经过时间**推导标称帧率后写出 —— 无论推理快慢，录像都覆盖真实的 3 秒，
回放速度与实际一致。

---

## 10. 与其他功能融合

### 10.1 三种集成方式对比

| 方式 | 适用场景 | 耦合度 | 语言/边界 | 说明 |
|---|---|---|---|---|
| **HTTP API**（推荐） | 已有系统/微服务、跨语言、跨机器 | 低 | 任意（REST） | 只依赖 6 个核心接口，最稳定 |
| **Python 库内嵌** | 同语言、需极低延迟或深度定制 | 高 | Python 3.10~3.12 | 直接 `from engine import DetectionEngine` |
| **前端嵌入** | 只要"把监控画面放进已有后台页面" | 最低 | `<iframe>` / `<img>` | 不改后端 |

### 10.2 方式一：HTTP 集成（推荐）

**最简工作流**

```
1. GET  /api/health                 → 等 models_ready = true
2. POST /api/start  {type: camera}  → 开始检测
3. GET  /api/stream                 → <img> 显示实时画面
4. GET  /api/status   (每 300~500ms) → 概率、报警、last_record
5. GET  /api/records                → 记录列表 / 报告
6. POST /api/stop                   → 停止
```

**Python 示例（含"发现新跌倒"事件）**

```python
import time, requests

BASE = "http://127.0.0.1:8000"

# 1) 等待模型就绪
while not requests.get(f"{BASE}/api/health").json()["models_ready"]:
    time.sleep(1)

# 2) 开始检测（摄像头）
requests.post(f"{BASE}/api/start", json={"type": "camera", "index": 0})

seen = set()
try:
    while True:
        st = requests.get(f"{BASE}/api/status").json()
        print(f"[{st['state']}] P(fall)={st['prob']:.2f} fps={st['fps']:.1f} "
              f"记录数={st['n_records']}")

        # 3) 新跌倒事件 → 触发你自己的业务（推送消息、上传云存储、联动门禁…）
        rec = st.get("last_record")
        if rec and rec["id"] not in seen:
            seen.add(rec["id"])
            print(f"  新跌倒！时间={rec['time']} 峰值={rec['peak']} "
                  f"录像={BASE}{rec['clip_url']}")
            # do_your_thing(rec)

        time.sleep(0.5)
finally:
    requests.post(f"{BASE}/api/stop")
```

**JavaScript 示例**

```js
const BASE = 'http://127.0.0.1:8000';
document.querySelector('#cam').src = `${BASE}/api/stream`;

async function poll() {
  const st = await (await fetch(`${BASE}/api/status`)).json();
  document.querySelector('#prob').textContent = st.prob.toFixed(2);
  document.querySelector('#alarm').classList.toggle('on', st.alarm);
  // 记录联动
  const recs = await (await fetch(`${BASE}/api/records?limit=5`)).json();
  renderList(recs.records);
}
setInterval(poll, 500);
```

### 10.3 方式二：Python 库内嵌

适用于：「我的主程序已经是 Python，想少一层 HTTP」。

```python
import sys, time
sys.path.insert(0, r"<项目>/systems")

from engine import DetectionEngine, list_demo_sources

eng = DetectionEngine(threshold=0.6, cooldown_s=6.0)   # 加载三个模型

eng.start({"type": "camera", "index": 0})
try:
    while True:
        st = eng.status()
        print(st["state"], st["prob"], st["n_records"])
        # 直接拿 JPEG 字节（无需 HTTP）
        jpeg = eng.latest_jpeg()
        time.sleep(0.5)
finally:
    eng.stop()

# 记录直接可读，也可自己接回调
for r in eng.records():
    print(r["time"], r["peak"], r["clip"])
```

要点：

- `DetectionEngine` 线程安全，`status()` / `latest_jpeg()` / `records()` 可随处调用。
- 想加"触发即回调"，可在 `engine.py` 的 `_finalize_recording()` 末尾加一个 `self.on_fall(entry)` 钩子。
- 想换存储，实现 `store.RecordStore` 同名方法即可（见第 8 节末尾）。

### 10.4 方式三：前端嵌入

只要画面与状态，不想改后端：

```html
<!-- 实时画面（只读，任意外部页面都能用） -->
<img src="http://127.0.0.1:8000/api/stream" style="width:100%;border-radius:8px">

<!-- 或整体嵌入监控台 -->
<iframe src="http://127.0.0.1:8000/" width="1400" height="900" frameborder="0"></iframe>
```

> 跨域说明：若页面所在域名与 `127.0.0.1:8000` 不同，`<img>`/`<iframe>` 可直接用；
> 但用 `fetch` 调 `/api/*` 会被 CORS 拦截 —— 需要时在 `app.py` 加 `CORSMiddleware`：

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
```

### 10.5 联动与事件（推荐模式）

系统本身不主动推送，但有两个可靠的"事件源"，适合做联动：

| 事件源 | 判断方式 | 适合场景 |
|---|---|---|
| **新跌倒入库** | `status.last_record.id` 变化（或 `n_records` 增长） | 推送到钉钉/企微、上传录像到云、写业务库 |
| **实时报警中** | `status.alarm == true` | 界面闪烁、蜂鸣、声光报警器、弹窗 |

```python
last_id = None
while True:
    st = requests.get(f"{BASE}/api/status").json()
    rec = st.get("last_record")
    if rec and rec["id"] != last_id:
        last_id = rec["id"]
        push_to_im(f"检测到跌倒：{rec['time']}（峰值 {rec['peak']}）",
                   video_url=f"{BASE}{rec['clip_url']}")
    if st["alarm"]:
        trigger_siren()
    time.sleep(0.5)
```

**幂等建议**：以 `record.id` 作为去重键；不要以时间字符串为准（同秒可能多条）。

### 10.6 扩展点速查

| 想做的事 | 改哪里 |
|---|---|
| 新增输入源（RTSP / 摄像头阵列 / 图片序列） | `engine.FrameSource.__init__` 增加一个 `kind` 分支 |
| 主程序里加触发回调 | `engine._finalize_recording()` 末尾 |
| 换存储（MySQL / 上行业务库） | `store.RecordStore` 的 `add/list/latest/count/delete/clear` |
| 换判定策略（如多帧投票、时序状态机） | `engine._process()` 里 `prob / triggered` 的计算 |
| 加新的叠加绘图（人数、人体框颜色） | `engine._draw_hud()` 与 `main.draw_skeleton` |
| 暴露新接口 | `app.py` 里加路由（引擎方法已线程安全） |
| 调整阈值/冷却默认值 | `DetectionEngine.__init__` 默认参数，或启动参数 `--threshold` |
| 允许跨域 | `app.py` 加 `CORSMiddleware`（见 10.4） |

### 10.7 集成注意事项

1. **单会话**：同一进程只能跑一路检测；要并行多路，需多进程 + 不同端口/不同 `records` 目录。
2. **录像文件生命周期**：删除记录会删录像；`/api/records/clear` 会清空 `clips/`。集成方若要长期留存，请先拷贝走。
3. **磁盘增长**：默认每次跌倒约 0.3 MB；高频误报会累积，建议定期清理或调高 `threshold` / `cooldown_s`。
4. **时间语义**：`time` 是**服务器本地时间**；跨机器部署时注意时区一致。
5. **端口占用**：改端口用 `--port`；Windows 下先 `netstat -ano | findstr :8000` 确认。
6. **首次启动慢**：模型加载 5~10 秒，请以 `/api/health` 就绪后再调 `start`。
7. **不要 `--reload`**：uvicorn 热重载会重复加载模型并占用摄像头。
8. **摄像头独占**：被会议软件等占用时会启动失败（返回 400，附中文原因）。

---

## 11. 错误处理与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `models_ready=false`，`load_error` 含 `No module named 'a4_core'` | A4 权重来自其它环境/路径 | 运行 `python train_weights.py` 在当前环境重新导出 |
| `load_error` 含 xgboost / sklearn / numpy 报错 | 依赖版本与权重不匹配 | 同上；详见第 13 节 |
| `无法打开摄像头` | 设备被占用 / 序号错 | 关闭占用程序；改 `index` |
| `视频文件不存在` | 路径错 / 用了相对路径 | 用绝对路径，或先 `/api/upload` |
| 画面一直 `NO SIGNAL` | 未 `start`，或源已播完 | 调 `/api/start`；查看 `status.state` |
| `status.state = finished` | 视频源播放结束 | 正常；重新 `start` 或换序列 |
| `prob` 长时间为 0 | 未检测到人（无框 → 关键点全 0） | 确认画面中有人；检查遮挡/分辨率 |
| 播放录像报 404 | 记录已删除 / 文件名被改 | 从 `/api/records` 取最新 `clip_url` |
| 上传 400 | 文件为空或字段名不是 `file` | 用 `-F "file=@..."` |

---

## 12. 已知限制

| 限制 | 说明 | 影响 |
|---|---|---|
| 单路检测 | 单进程单会话 | 多路需多进程 |
| 无鉴权 | 无登录/Token | 仅建议部署在内网/本机；如需暴露请加反向代理鉴权 |
| 无推送 | 只有轮询 | 联动请用 10.5 的轮询模式 |
| 示例数据集 fps 特例 | MCF 视频实为 120 fps，采样已按实际 fps 处理 | 不影响判定，仅影响时间轴显示 |
| 无跌倒后"已处理"状态 | 同一次跌倒在冷却期后可能再次入库 | 用 `cooldown_s` 抑制；集成侧可用 `id` 去重 |
| 上传文件不自动清理 | `uploads/` 只增不减 | 需自行定期清理 |

---

## 13. A4 权重跨环境兼容性

**问题**：`a4_weights.pkl` 是 pickle，内部含 sklearn / xgboost 对象，反序列化对**版本**敏感。

| 依赖 | 为什么敏感 |
|---|---|
| `xgboost` | booster 有独立二进制格式，跨大版本（2.x → 3.x）常无法读取 |
| `scikit-learn` | 会抛 `InconsistentVersionWarning`；部分模型属性变更后可能静默产生**错误概率** |
| `numpy` | 2.x 改过部分 dtype / 序列化内部结构，旧 pickle 可能读不出 |
| 类路径 | pickle 记录类的模块路径；若类从 `hybrid_algorithm` 挪到 `a4_core`，会报 `No module named ...` |

**解决（本项目已内置）**：

```bash
python train_weights.py     # 在当前环境重新训练并导出 systems/a4_weights.pkl
```

- 训练数据取自已推理好的**骨架缓存**（不需要重跑 YOLO+姿态），约 1~3 分钟。
- A4 的模型结构与超参与论文一致，只保证"在当前环境可加载、可推理"。
- 引擎优先加载 `systems/a4_weights.pkl`，找不到才回退 `standalone/a4_weights.pkl`；
  实际来源可查 `status.a4_weights`。
- `torch` 权重（`pose_best.pt`）不受此影响 —— 它存的是纯张量。

**其它可选方案**：在当前环境把 xgboost/sklearn 降级到与权重一致的版本（可能影响该环境其它项目）；
或把 A4 改为导出 `.json`/`.ubj`（xgboost 原生格式）+ `joblib`，需改 `a4_core.py` 的存取逻辑。

---

## 14. 性能与调优

> 以下数据均在 **RTX 3050 Ti Laptop + animedev 环境**、`urfd/fall-01`（640×480）实测。

### 14.1 画面叠加不会影响模型判断

这是最常被问到的一点，先说结论：**叠加完全不可能影响模型**。

| 环节 | 输入 | 说明 |
|---|---|---|
| ① YOLO 检测 | `frame`（原始帧） | 未经任何绘制 |
| ② P1 姿态 | `frame`（原始帧） | 未经任何绘制 |
| ③ A4 时序判定 | 关键点坐标数组 | 只有坐标，连像素都没有 |

绘制发生在三段推理**之后**，只画在副本 `vis` 上：

```python
vis = frame.copy()                     # 副本，模型看不到
if self.show_skeleton and pts is not None:
    draw_skeleton(vis, pts, conf, box)
if self.show_hud:
    vis = self._draw_hud(vis, prob, frame_i)
return vis, prob, triggered            # vis 只用于显示 / 编码成 MJPEG 与录像
```

补充两点：

1. 模型权重是在**无任何叠加**的原始数据集（URFD / Le2I / MC）上训练的，
   训练时从未见过"顶部信息条 / FALL 文字"，画面上也不存在可被"读"的答案；
2. 前端 HTML 层的报警横幅（红色"⚠ 检测到跌倒"）是浏览器渲染的 DOM 元素，
   根本不在视频帧里，更不可能进入模型。

`show_hud` 默认 `false`（画面不留文字），需要做算法讲解/调试时可随时打开：

```bash
curl -X POST http://127.0.0.1:8000/api/config \
     -H "Content-Type: application/json" -d "{\"show_hud\":true}"
```

### 14.2 三段耗时拆解（单帧、逐帧全算）

| 环节 | 耗时 | 备注 |
|---|---|---|
| ① YOLO 人体检测 | ~50–72 ms | OpenCV DNN，**走 CPU**（pip 版 opencv 无 CUDA） |
| ② P1 姿态 | ~12–16 ms | GPU（torch cu121） |
| ③ A4 时序判定 | **~90 ms → 优化后 ~30 ms** | 见 14.4 |
| 读帧 + JPEG 编码 + 录像 | ~10 ms | 磁盘读帧 5–8ms、编码 1.2ms |
| **合计** | ~170 ms → **~50 ms** | **5.8 fps → 20 fps** |

### 14.3 推理降频（detect_every / prob_every）

2 秒滑窗（30 帧 @15fps）内相邻采样高度重叠，逐帧全算属于浪费。

| 参数 | 默认 | 作用 |
|---|---|---|
| `detect_every` | 3 | 每 3 帧做一次 YOLO，其余帧**复用上一次的框**（人体相邻帧位移很小） |
| `prob_every` | 2 | 每 2 帧做一次 A4，其余帧**沿用上一次概率** |
| 姿态 P1 | 1（固定） | **始终逐帧**：2s 滑窗需要每帧骨架，且 GPU 上只要 12ms |

**实测（纯吞吐，关闭节流）**

| 配置 | ms/帧 | fps |
|---|---|---|
| 1 / 1（逐帧全算） | 107 | 9.3 |
| **3 / 2（默认）** | **52** | **19.3** |
| 6 / 3 | 35 | 28.5 |

判定质量不受影响：以上配置在同一段 URFD 序列上峰值概率均为 **0.996–0.999**，均正确触发。

> `confirm` 只在**真正做判定的帧**上累计，所以 `prob_every` 不会削弱"连续 N 次确认"的抗抖动语义。

### 14.4 关掉 joblib / OpenMP 的调度开销（本项目已内置，收益最大）

A4 的 7 个基学习器（RF / XGB / LR + LSTM / GRU / BiLSTM / TCN）都**很小**，
但 sklearn 默认会为它们启用 joblib 多进程 + OpenMP 多线程，
**调度与结果回收的开销远大于计算本身**（实测 joblib 的 `_retrieve` 每 10ms 轮询一次）。

解决方式：在**导入 numpy/sklearn/torch 之前**设置环境变量（`app.py` / `engine.py` 顶部已内置）：

```python
if os.environ.get("FVS_NO_TUNE") != "1":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
```

| 设置 | A4 单次 | 整环（3/2 配置） |
|---|---|---|
| 默认 | ~90–100 ms | 68.5 ms/帧（14.6 fps） |
| 关闭 joblib 多进程 | ~60 ms | 56.3 ms/帧（17.8 fps） |
| **+ 单线程 OpenMP** | **~30 ms** | **37.0 ms/帧（27.0 fps）** |

恢复默认行为：启动前设 `FVS_NO_TUNE=1`。

> 另外，`engine.py` 里的 `a4_fast_context()` 会在 `predict` 外层再加一道
> 「线程后端 + 单线程」的作用域限制，避免被第三方进程直接 import 时又回到慢路径。
> 二者都不修改任何算法文件与权重。

### 14.5 其它已做的性能修正

| 项 | 问题 | 处理 |
|---|---|---|
| 节拍时钟 | 用 `time.time()` 算单帧耗时来节流；Windows 上该时钟分辨率仅 ~15.6ms，帧率抖动 | 改用 `perf_counter()` 绝对节拍累加；记录时间戳仍用 `time.time()`（必须是真实系统时间） || `status()` 查库 | 前端每 0.5s 轮询一次；原实现在**持锁**期间查两次 SQLite，把推理线程堵在锁上 | 记录条数/最新记录改为内存缓存，只在记录增删时刷新；`status()` 不再碰数据库 |
| `status.fps` 读数 | 原为「从启动算起的累计均值」，被首帧 CUDA/cuDNN 预热长期拉低（显示 7~8fps，实际 12~15fps） | 改为**最近 2 秒滑动窗口**；会话结束后保留最后一次读数，下次 `start()` 重置 |
| 清空记录（§6.12） | 逐条删 mp4 时撞上环境的批量删除保护，它抛的是 `SystemExit`（**不属于 `Exception`**），`except OSError` 接不住 → 整个请求 500，且录像只删了一半 | 单次限删 40 个 + 每个删除动作单独兜住 `(OSError, SystemExit)`，接口恒定返回 200 并回报 `clips_removed` / `clips_left` |

### 14.6 想再快怎么办

按收益从高到低：

1. **换 CUDA 版 OpenCV**（`opencv-python` → 自行编译或换带 CUDA 的发行版），YOLO 的 50–70ms 能降到个位数；
2. 调大 `detect_every`（4~6）/ `prob_every`（3~4），实测 6/3 可达 28fps，代价是检测框略"陈旧"；
3. 降低输入分辨率（摄像头设 640×480，别直接喂 1080p）；
4. 用 `--port` 之外的方式把服务固定在独立进程里，避免和浏览器/IDE 抢 CPU（YOLO 是 CPU 大户）。

---

*文档版本：v1.1 ｜ 对应代码：`app.py` / `engine.py` / `store.py` / `train_weights.py`
｜ 本次更新：画面叠加开关（`show_hud` 默认关闭）、推理降频参数、joblib/OpenMP 调优、节拍时钟与 status 缓存修正*
