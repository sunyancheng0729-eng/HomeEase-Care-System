# HomeEase 居家照护系统

FastAPI + 原生 HTML/CSS/JavaScript + SQLite 电脑端系统。面向照护者和长者，提供独立的视频、加速度跌倒检测，四种运动统计，环境声音标签、语音求救和跌倒问询，以及事件处置与照护报告。

## 从 GitHub 获取

仓库包含整体系统、`algorithm1/`、`algorithm2/`、`algorithm3/` 的原始代码、模型权重、样例及随附训练资料。大体积模型与数据使用 Git LFS，克隆后需取得实际文件，不能直接使用 GitHub ZIP 下载里的 LFS 指针。

```powershell
git lfs install
git clone https://github.com/sunyancheng0729-eng/HomeEase-Care-System.git
cd HomeEase-Care-System
git lfs pull
```

虚拟环境、依赖缓存、运行数据库、历史告警录像和本机测试截图不上传；在新电脑按下方依赖说明创建环境。三个原始算法目录的文件保留原有字节内容，未做换行转换。

## 本机启动

本机的 Python 环境与模型已经配置好。双击 `start.bat`，打开终端打印的地址，通常为 <http://127.0.0.1:8000>。端口被占用时自动选择后续端口。

```powershell
cd E:\gc2
.\.venv\Scripts\python.exe run.py
```

停止：双击 `stop.bat`，或执行 `.\.venv\Scripts\python.exe run.py --stop`。等待服务退出后再启动，可让视频、音频、模型和后台任务正常释放。启动时各模块独立加载；在设置页查看状态及重新初始化。

电脑浏览器入口：照护工作台 `/`，简化长者操作台 `/elder`，外部加速度数据接入 `/sensor`，交互 API 文档 `/docs`。

## 首次使用

1. 在设置页填写长者资料与紧急联系人，按优先顺序排列联系人，并测试通知。
2. 取得长者同意后记录音频授权。默认关闭授权；授权不等于浏览器的麦克风权限，两者均需开启。
3. 跌倒监测页同时展示视频与加速度通道。视频选择本机摄像头、上传视频或 URFD 样例；实时摄像头由系统统一采集，跌倒监测与运动训练可共用同一序号的设备。
4. 加速度分支可上传 CSV，或通过外部传感器的数据接口输入。电脑自身无需内置加速度计；摄像头和传感器需要分别开启，模型就绪不代表已经监测。
5. 运动页选择坐站、深蹲、站立抬腿或跨步，设置目标，启动、暂停、继续或结束训练。画面不可用、多人和目标变化按原算法停止有效计数，界面展示原因。
6. 声音页开启联合监听后，同时进行全部环境声音分类和语音求助监听，无需选择声音类别。录音与样例回放也同时分析两者。长者页可主动求助、表示安全和回答语音问询，也可选择接收自动问询。
7. 事件页接手处理、联系、升级响应，记录核实结论后关闭。报告页按时间范围分别查看实时与回放数据，导出 CSV，或打印为 PDF；训练可下载 JSON，事件有独立处置报告。

声音标签为辅助信息。语音回答或“我没事”不会自行关闭、降级事件；需要照护者明确记录处置结论。未播放问询、未授权和识别不可用均有独立状态。

## 原算法映射

| 模块 | 唯一使用的已选方案 | 原实现入口 |
| --- | --- | --- |
| 算法一：视频 | YOLOv8n + P1 + A4，原默认参数 | `algorithm1/video/engine.py` 的 `DetectionEngine` |
| 算法一：加速度 | A2 RF + XGB 平均模型 | `algorithm1/siganl/core/model.py`、`detector.py` |
| 算法二：运动 | `opt_tcn_b4_w48`，原选定 `yolo11n-pose.pt` | `algorithm2/homeease/api.py`、`engine.py`、`pose.py` |
| 算法三：声音与回复 | Silero VAD、YAMNet + LR v2、Whisper tiny、rule_v2、原 reply policy | `algorithm3/algorithm3_handover/src/algorithm3` 与原音频流水线 |

算法一的两个分支独立运行，因此共有四个工作进程，仍然只有三个交付算法模块。系统不新增融合分类器、不重训模型、不修改算法目录中的核心代码、阈值或预处理。Proposal 中与实际交付算法不同的技术方案未作为新算法实现。

## 输入与设备

- 原生 CSV：5 列 `t_ms, magnitude, ax, ay, az`，时间毫秒，加速度 g。兼容 4 列带表头 `t,ax,ay,az` 或 `t_ms,ax,ay,az`；前者以秒计。适配层仅补齐原读取器要求的格式。
- 实时传感器：每批 `[t_seconds, ax_g, ay_g, az_g]`，严格递增时间戳。数据包含重力、单位 g。外部采集程序通过 `/ws/sensor` 提交数据；电脑端 `/sensor` 可连接并提交设备批次。原交付文件未包含具体 USB/串口设备驱动，接入硬件时由采集程序提供此格式，不要求手机。
- 视频支持本机摄像头和可解码的 MP4/AVI/MOV/MKV/WebM，上传上限 512 MB。音频上限 20 MB / 120 秒。运动的骨架输入遵循原算法2接口。
- 摄像头由服务所在电脑统一打开，原始 BGR 像素通过共享内存交给两种原算法，不重复打开设备、不用预览 JPEG 作推理输入。运动页与跌倒页需选择同一序号才能共享同一台相机。浏览器仅需麦克风权限；其他电脑通过局域网访问时需设置密码并使用 HTTPS。

```powershell
$env:HOMEEASE_PASSWORD = '自行设置的访问密码'
.\.venv\Scripts\python.exe run.py --host 0.0.0.0 --port 8000
```

登录用户名 `caregiver`。局域网电脑通过可信 HTTPS 反向代理访问；代理须支持 WebSocket，并保留 Host 与 Origin 的一致性。本地回环地址默认无需密码。

## 数据、通知与配置

SQLite 默认 `runtime/homeease.sqlite3`，日志在 `runtime/logs/`，上传视频、事件片段和播放派生文件在 `runtime/`。原始音频与识别文字不写入数据库；音频临时数组在工作进程中清零。视频事件片段和上传视频会保留，应按家庭隐私约定管理磁盘与备份。

站内通知、浏览器通知和联系人拨号链接已实现。站内送达表示工作台取得了通知，查看、接手、处置时间分别记录。配置 Webhook 后实时事件可发送到接收端，附幂等键，失败最多尝试 5 次；回放不会发送外部告警。系统没有预置短信、自动拨号或急救服务。

| 配置 | 用途 |
| --- | --- |
| `HOMEEASE_SYSTEM_DATA` | 独立数据、日志和桥接目录，默认 `E:\gc2\runtime` |
| `HOMEEASE_PASSWORD` | HTTP Basic 访问密码，绑定非回环地址时必填 |
| `HOMEEASE_VIDEO_CV_THREADS` | 视频工作进程 OpenCV 线程数，默认 4；不改变模型与检测参数 |
| `HOMEEASE_VIDEO_DNN` | 默认 `auto`，优先用 CUDA FP32 执行原 ONNX；不可用时保留原 OpenCV 执行器。`opencv` 可显式使用原执行器 |
| `HOMEEASE_EXERCISE_DEVICE` | 默认 `auto`，与原摄像头脚本一致自动选择 CUDA / CPU；可显式设为 `cpu` 或 `cuda` |
| `run.py --host / --port` | 监听地址与首选端口 |
| 设置页 | 资料、联系人、音频授权、通知、训练目标、联系人升级间隔 |

仅运行一个 FastAPI worker；同一数据目录重复启动时返回已运行的地址，避免重复占用数据库与设备。多个实例必须使用不同的数据目录和端口。停止命令需使用与启动时相同的 `HOMEEASE_SYSTEM_DATA`。

## 依赖与验证

本机 `.venv` 基于 `E:\conda_envs\dl` 的 Python 3.11.15，并复用该环境已安装的 PyTorch。迁移到新电脑时创建独立 Python 3.11 环境并安装 `requirements.txt`，保留三个算法文件夹、权重和样例。`artifacts/environment.json` 与 `requirements-lock.txt` 记录本次实测环境；复现该锁定环境时，CUDA 12.8 版 PyTorch 需要对应 wheel 源。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\snapshot_algorithms.py
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=runtime\pytest-check
.\.venv\Scripts\python.exe scripts\verify_integrations.py
.\.venv\Scripts\python.exe scripts\acceptance.py
```

浏览器测试：`npm ci`、`npx playwright install chromium`，启动独立测试数据目录的服务器后设置 `HOMEEASE_TEST_URL`，执行 `npm run test:browser`。测试会创建事件与训练记录，应使用隔离数据目录。

详见 [架构与集成](docs/architecture.md)、[API 与数据格式](docs/api.md)、[验证记录](docs/verification.md) 和 [样例来源](samples/SOURCES.md)。目前完成了原算法一致性、API 工作流和浏览器测试；真实家庭设备、外部通知送达、七天持续运行与真实误报漏报率仍需现场验收。

2026-10-15 的卡顿、零计数和实时声音接入排查见 [本次排查记录](docs/audit-20261015.md)。停止服务后，可用 `python scripts/clear_records.py` 清除视频/加速度跌倒与训练记录及关联通知、录像；资料、联系人、声音历史不受影响。验证脚本应始终使用独立数据目录。
