# 实时跌倒检测系统（systems/）

把本项目训练好的三段式跌倒检测算法，封装成一个**可持续运行、带界面、能自动留证**的系统：
支持摄像头实时输入或本地视频输入，检测到跌倒后自动记录**系统时间 + 3 秒录像 + 报警标记**，
并生成可查看 / 可导出的**跌倒报告**。

技术栈：**FastAPI（后端）+ 原生 HTML/CSS/JS（前端）+ OpenCV（取帧与录像）+ SQLite（记录持久化）**。

> 📄 需要**接口规格 / 环境依赖 / 与其它功能融合**的详细说明，请见 **[`API.md`](./API.md)**
> （含全部接口的请求响应字段、数据模型、数据库结构、三种集成方式代码示例）。

---

## 1. 系统做了什么

```
                 ┌──────────────────────────────────────────────────────┐
   摄像头 / 视频 →│  ① YOLOv8n 人体检测  →  ② 自研 P1 姿态 CNN  →  ③ A4 混合集成  │→ P(fall)
                 └──────────────────────────────────────────────────────┘
                                        │
                        P(fall) 连续确认超过阈值
                                        ▼
                      ┌─────────────────────────────────┐
                      │ 触发一次跌倒事件：              │
                      │  · 记录系统时间（本地时间）      │
                      │  · 落盘 3 秒录像（1s前置+2s后置）│
                      │  · 打上报警标记                  │
                      └─────────────────────────────────┘
                                        ▼
                        records/falls.db（SQLite） + records/clips/*.mp4
                                        ▼
                                 报告页 / 导出 HTML
```

三个环节用的都是本项目已训练好的权重，**不重新训练、不改算法**：

| 环节 | 权重文件 | 来源 |
|---|---|---|
| ① 人体检测 | `standalone/yolov8n-pose.onnx` | YOLOv8n 预训练，只用 box head |
| ② 姿态估计 | `standalone/pose_best.pt` | 自研 P1 小型 U-Net（从零训练，PCK@0.2=43.1%） |
| ③ 时序判定 | `systems/a4_weights.pkl` 或 `standalone/a4_weights.pkl` | A4 混合集成（7 基学习器 OOF Stacking + XGBoost） |

---

## 2. 目录结构

```
systems/
├── app.py                 FastAPI 后端：路由 / MJPEG 实时流 / 记录 / 报告 / 上传
├── engine.py              检测引擎：后台线程逐帧推理、3 秒录像、记录写库
├── store.py               记录持久化（SQLite，标准库 sqlite3，无额外依赖）
├── train_weights.py       在【当前环境】重新导出 A4 权重（版本兼容用，通常只需跑一次）
├── a4_weights.pkl         A4 权重（由 train_weights.py 生成；没有则回退 standalone 的）
├── static/                前端页面
│   ├── index.html         实时监控台
│   ├── report.html        跌倒报告页
│   ├── style.css
│   ├── app.js
│   └── report.js
├── records/               运行产物
│   ├── falls.db           跌倒记录数据库（SQLite）
│   ├── clips/*.mp4        3 秒跌倒录像
│   └── report_*.html      导出的 HTML 报告
└── uploads/               通过网页上传的本地视频
```

### 记录表结构（SQLite：`records/falls.db`）

| 字段 | 说明 |
|---|---|
| `id` | 主键，自增 |
| `fall_time` | 跌倒时间（系统本地时间） |
| `epoch` | 同一时间的时间戳，便于排序 |
| `prob` / `peak` | 触发时概率 / 该次事件峰值概率 |
| `clip` | 3 秒录像文件名 |
| `duration_s` / `frames` / `fps` | 录像时长 / 帧数 / 标称帧率 |
| `threshold` | 触发时使用的判定阈值 |
| `source` | 输入源（摄像头 0 / urfd/fall-01 …） |
| `alarm` | 报警标记（1 = 报警） |
| `created_at` | 入库时间 |

可直接用命令行查看：

```bash
sqlite3 systems/records/falls.db "SELECT id, fall_time, prob, peak, clip FROM falls ORDER BY epoch DESC"
```

---

## 3. 环境与安装

依赖：`torch`、`opencv-python`、`numpy`、`sklearn`、`xgboost`、`pandas`、`fastapi`、`uvicorn`。

```bash
# animedev 环境补齐算法依赖（只装一次）
E:/anaconda3/envs/animedev/python.exe -m pip install opencv-python xgboost
E:/anaconda3/envs/animedev/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
```

> 有 NVIDIA 显卡时装 CUDA 版 torch 可获得实时帧率；纯 CPU 也能跑，只是每秒处理帧数较低
> （录像时长按真实时间采集，不受推理速度影响）。

---

## 4. 启动

```bash
# 1) 首次使用：在当前环境重新导出 A4 权重（约 1~3 分钟）
E:/anaconda3/envs/animedev/python.exe train_weights.py

# 2) 启动服务
E:/anaconda3/envs/animedev/python.exe app.py
```

启动后终端会打印地址，浏览器打开：

| 地址 | 页面 |
|---|---|
| http://127.0.0.1:8000 | 实时监控台 |
| http://127.0.0.1:8000/report | 跌倒报告 |

常用启动参数：

```bash
python app.py --port 8080                     # 换端口
python app.py --camera 0                      # 启动即打开 0 号摄像头
python app.py --demo urfd fall-01             # 启动即跑内置示例
python app.py --threshold 0.5                 # 自定义判定阈值
```

---

## 5. 使用步骤（监控台）

1. **选输入源**：摄像头 / 本地视频 / 内置示例数据
   - *摄像头*：填设备序号（内置摄像头一般为 `0`）
   - *本地视频*：直接填路径，或点「选择并上传」从硬盘选文件
   - *内置示例*：选数据集（urfd / le2i / mc）+ 序列，**没有摄像头也能验证系统**
2. 点 **开始检测**，画面区出现实时视频（含检测框 + 骨架 + 概率条）
3. 右侧「实时判定」实时显示 `P(FALL)`、判定结果；命中阈值且连续确认后：
   - 画面 **红框闪烁 + 报警标记**，同时**响两声提示音**
   - 自动保存 3 秒录像并写入记录
4. 点 **停止** 结束本次会话

### 参数说明

| 参数 | 含义 | 建议 |
|---|---|---|
| 跌倒判定阈值 | `P(fall)` 超过该值视为跌倒 | 0.6（越低越灵敏、误报可能变多） |
| 报警冷却（秒） | 同一次跌倒在该时间内不重复记录 | 6 |
| 前置录像（秒） | 录像从触发前多少秒开始 | 1.0（后置自动 = 3 − 前置） |

### 画面叠加说明（不会影响模型判断）

画面上的检测框、骨架、顶部信息条都是**画在副本上的**，送进三段模型的**永远是原始帧**，
且叠加发生在三段推理之后——所以叠加不可能让模型"作弊"。
（模型权重还是在无叠加的原始数据集上训练的，画面里也没有可被读到的答案。）

- 默认 `show_hud = false`：**不烧录顶部信息条 / `P(fall)` 文字 / 概率条**，画面干净；
  判定结果直接看右侧「实时判定」环。
- `show_skeleton = true`：保留检测框 + 骨架，方便一眼看出模型在看哪里。
- 监控台右侧有 **「画面叠加」** 卡片，两个复选框可直接勾选（也可以随时关掉骨架）：

```bash
# 等价于勾选/取消复选框
curl -X POST http://127.0.0.1:8000/api/config -H "Content-Type: application/json" -d "{\"show_hud\":true}"
curl -X POST http://127.0.0.1:8000/api/config -H "Content-Type: application/json" -d "{\"show_skeleton\":false}"
```

### 性能与调优

三段推理里 **YOLO 走 CPU（~50–70ms）**、姿态走 GPU（~12ms）、A4 混合集成原本要 ~90ms。
默认已经做了三件事，让系统在 640×480 上跑到 **~20fps**（已高于 15fps 采样率）：

| 优化 | 默认值 | 效果 |
|---|---|---|
| 推理降频 `detect_every` | 3 | YOLO 每 3 帧一次，其余帧复用上次的框 |
| 推理降频 `prob_every` | 2 | A4 每 2 帧一次，其余帧沿用上次概率 |
| 关闭 joblib / OpenMP 调度开销 | 已内置 | A4 单次 ~90ms → **~30ms**（3 倍） |

实测（RTX 3050 Ti + animedev）：逐帧全算 9.3fps → 默认配置 **19.3fps** → 6/3 配置 28.5fps，
判定质量不变（同一序列峰值概率 0.996~0.999）。

想改降频档位：改 `engine.DetectionEngine(detect_every=..., prob_every=...)`（见 `API.md` §14）。
想恢复默认线程行为（不关闭 joblib/OpenMP）：启动前设 `FVS_NO_TUNE=1`。

---

## 6. 报告

- **监控台**右下角显示最近的跌倒记录，点「播放」可直接看 3 秒录像。
- **报告页**（`/report`）以表格列出每条记录：**序号 / 跌倒时间（系统时间）/ 触发概率 /
  峰值概率 / 录像时长 / 输入源 / 报警标记 / 3 秒录像播放 / 删除**。
- 点 **导出 HTML 报告** 生成 `records/report_YYYYmmdd_HHMMSS.html`，该文件可独立分享
  （录像以相对路径引用同目录下的 `clips/`）。
- 点 **清空记录** 会删除全部记录与录像文件（不可撤销）。

---

## 7. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 实时监控页 |
| GET | `/report` | 跌倒报告页 |
| GET | `/api/stream` | MJPEG 实时画面（检测框 + 骨架 + 概率） |
| GET | `/api/status` | 状态：概率 / 报警 / fps / 帧数 / 命中数 / 记录数 |
| GET | `/api/sources` | 可用输入源（内置示例数据集列表） |
| POST | `/api/start` | 开始检测 `{type: camera\|file\|demo, ...}` |
| POST | `/api/stop` | 停止检测 |
| GET/POST | `/api/config` | 读取 / 修改阈值、冷却、前置后置秒数 |
| GET | `/api/records` | 跌倒记录列表（SQLite，新的在前；支持 `?limit=n`） |
| GET | `/api/records/stats` | 统计（总条数 / 最高峰值 / 首末时间） |
| DELETE | `/api/records/{id}` | 删除单条记录及其录像 |
| POST | `/api/records/clear` | 清空全部记录与录像 |
| GET | `/api/clip/{file}` | 录像播放（支持 HTTP Range，可拖动进度条） |
| POST | `/api/upload` | 上传本地视频文件 |
| POST | `/api/report/export` | 导出 HTML 报告 |
| GET | `/api/health` | 健康检查（含模型是否就绪） |

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/start -H "Content-Type: application/json" \
     -d "{\"type\":\"demo\",\"dataset\":\"urfd\",\"seq\":\"fall-01\"}"
curl http://127.0.0.1:8000/api/status
curl -X POST http://127.0.0.1:8000/api/stop
```

---

## 8. 常见问题

**Q：启动时报 A4 权重反序列化失败（`No module named 'a4_core'` 或 xgboost/sklearn 报错）**
A：说明 `standalone/a4_weights.pkl` 是用别的环境导出的。运行
`python train_weights.py` 在当前环境重新导出一份 `systems/a4_weights.pkl` 即可，
引擎会优先加载它。

**Q：摄像头打不开**
A：确认没有被其它程序（会议软件等）占用；换 `--camera 1` / `2` 试其它设备；
也可先用「内置示例数据」验证系统本身是否正常。

**Q：视频源播完就停了？**
A：正常行为，状态会变为「已结束」，重新点「开始检测」或换序列即可。

**Q：录像为什么只有 3 秒？**
A：这是需求规定的留证时长（跌倒前 1 秒 + 跌倒后 2 秒），前置秒数可在监控台调整。
录像按**真实时间**采集，因此无论推理快慢，录像都覆盖真实的这 3 秒。

**Q：点「清空记录」提示「还有 N 个录像文件」，或者删单条后录像还在？**
A：一条记录配一个 mp4，清空时要删几十上百个文件。运行环境带有「一轮内批量删除超过
50 个文件」的安全保护，所以删除是**分批**的：单次最多删 40 个，剩下的下次继续。
看到 `clips_left > 0`（前端提示「还有 N 个」）时**再点一次「清空记录」**即可清完。
残留文件已不被任何记录引用，不影响使用。接口本身**永远是 200**，不会因此报错。

---

## 9. 与算法文件夹的关系

`systems/` 只做「工程化封装」，不修改任何算法代码与权重：

- 检测/姿态/时序三个模型的实现，直接从不改动的 `standalone/` 复用；
- `engine.py` 用 `importlib` 按绝对路径加载 `standalone/main.py`，
  避免与项目根目录下同名的旧 `main.py` 冲突；
- `standalone/` 里的 `main.py`（单文件演示）依然可独立运行，两者互不影响。
