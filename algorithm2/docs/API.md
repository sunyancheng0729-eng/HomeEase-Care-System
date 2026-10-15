# HomeEase 算法 2 系统接口与交接文档

## 1. 接入结论

推荐系统端把摄像头图像在本地转成 COCO 17 关键点，再调用本服务的骨架流接口。这样算法 1 和算法 2 可以共用同一套 Pose CNN。已有图像入口 `homeease.pose.PoseEstimator` 和本地视频程序 `scripts/run_video.py`，无需把原始视频传到服务器。

HTTP API v1，默认地址 `http://127.0.0.1:8002`。接口完全由当前代码导出到 `docs/openapi.json`。启动时加载实际权重，权重缺失会使启动失败，不返回假预测。系统集成可先用 `examples/*_replay.json` 做四动作端到端验收。

模型替换由 `artifacts/selected.json` 控制。系统端直接使用 HTTP 或 `Predictor`；窗口长度、stable80 特征、训练 mean/std 和分层概率合成都由 checkpoint 与 SDK 内部加载，系统端继续发送相机平面 COCO 17 点。不要在客户端额外做骨架归一化或写死旧版网络结构。服务重启后重新创建会话，并检查 `/health` 的 `model_id`。

### 当前优化后默认算法

**`opt_tcn_b4_w48`**：分层残差因果 TCN，4 个 block、宽度 48、62,453 参数；输入为 stable80，部署窗口 64 帧／30 Hz。动作头输出五类动作概率，每种目标动作有四类条件阶段概率。实际权重为 `artifacts/opt_tcn_b4_w48.pt`，mean/std 随权重保存。Pose 前端仍是 YOLO11n-pose@640。

默认 decoder 为 `ordered_transition`，阶段证据阈值 0.25，消抖 0.033 秒，运动阶段目标动作支持门槛仍为 0.15。它在已经观察到向外与最高点后，可用稳定回到起点的后验概率推断短暂返回阶段；必须再次确认基线才计数，缺少最高点或被中断的周期不补齐。

| 指标 | 原默认 GRU | 优化后默认 TCN |
|---|---:|---:|
| 验证计数 MAE | 1.00 次 | 0.10 次 |
| 测试动作 Macro-F1 | 0.8213 | 0.9612 |
| 测试联合阶段 Macro-F1 | 0.7007 | 0.8203 |
| 测试计数 MAE | 1.80 次 | 0.10 次 |
| 测试非零会话 MAPE | 23.47% | 1.25% |
| 测试周期 recall／precision | 详见原报告 | 0.9872／1.0000 |
| 同轮 CPU 时序推理 P50 | 8.926 ms | 5.619 ms |
| 参数量 | 55,185 | 62,453 |

测试共十个重建会话、78 个参考周期；TCN 匹配 77 个周期、漏一次抬腿、没有多计。这里的阶段标签是运动学弱标签，计数来自公共 mocap 原分段重复的重建回放，**不是老人真实视频精度**。原测试集前一轮已经查看，本轮根据验证集锁定方案后复用测试集比较，不称为全新盲测。

优化 GRU 在这份测试回放计数 MAPE=0%，但验证 MAE=0.80 次且计算成本较高；不能因为看到了测试结果就替换已锁定的 TCN。部署选择兼顾验证性能与计算成本。训练过程、CPU 实测、三个候选和逐会话结果见 [OPTIMIZATION.md](OPTIMIZATION.md) 与 [RESULTS.md](RESULTS.md)。

计时是本机 CPU 单次时序前向，不含 Pose CNN 与 HTTP 开销。固定模型在 15°／75°投影压力测试的 MAPE 均为 15%，45°加原始单位标准差 1 的坐标噪声时为 24.58%；常规投影结果不能覆盖极端视角与真实关键点噪声，系统验收需同时测试这些输入条件。

## 2. 启动与部署边界

```powershell
Set-Location E:\gc2-project-algorithm\algorithm2
.\.venv\Scripts\python.exe -m uvicorn homeease.api:app --host 127.0.0.1 --port 8002 --workers 1
```

- 只能使用一个 worker。会话和幂等缓存保存在进程内，多 worker 不能共享状态。
- 默认只监听 localhost。部署到家庭局域网时由系统负责鉴权、TLS、设备与用户权限、请求路由。
- 服务重启会清空会话。系统负责保存终态报告、开始新会话并在界面标记中断，不能恢复半个周期。
- 最多 100 个保留会话，结束后调用 DELETE 释放。状态读取不修改模型；每个会话有自己的特征历史和计数器。
- 单个请求体最多 2 MB，一批最多 300 帧。通常每 5 至 15 帧发送一批，使界面及时更新。
- 请求在 CPU 上执行时序模型。视频 Pose CNN 可以单独使用 GPU。`Predictor(device="cuda")` 可在 Python 集成中使用 CUDA。
- 原始图像和音频不在本服务中保存，HTTP 接口仅接收关键点。会话报告也不附原始关键点。

## 3. 动作与完整周期

| exercise 枚举 | 含义 | 一个完整周期 |
|---|---|---|
| `chair_sit_to_stand` | 椅子坐站 | sitting → rising → standing → lowering → sitting |
| `squat` | 深蹲动作 | standing → lowering → squatting → rising → standing |
| `standing_leg_raise` | 站立直腿抬腿 | standing → lifting → raised → lowering → standing |
| `hurdle_step` | 跨步抬腿 | standing → stepping → raised → returning → standing |

选择动作由 exercise 指定，识别模型同时返回 `recognized_exercise`。静止基线在动作之间可能相同，计数阶段按选定动作条件化；运动阶段还要求选定动作概率 ≥0.15，否则输出 other 并停止该周期。一侧完整抬起并回到基线计一周期，接口不提供左右腿分别统计。目标完成不会自动结束，额外完整周期会继续增加计数。

动作名称对应公开数据集的实际动作。数据集不是为虚弱老人定义的训练处方；家庭训练的动作幅度、扶持条件与安排由产品或专业人员决定。当前模型不能评判是否适合某一位老人，也不输出疗效、平衡评分或跌倒风险。

## 4. 关键点与时间协议

`schema_name="coco17"`，`coordinate_space="camera_xy"`。每帧 17×3，顺序如下：

| 索引 | 关节 | 索引 | 关节 |
|---|---|---|---|
| 0 | nose | 1 | left_eye |
| 2 | right_eye | 3 | left_ear |
| 4 | right_ear | 5 | left_shoulder |
| 6 | right_shoulder | 7 | left_elbow |
| 8 | right_elbow | 9 | left_wrist |
| 10 | right_wrist | 11 | left_hip |
| 12 | right_hip | 13 | left_knee |
| 14 | right_knee | 15 | left_ankle |
| 16 | right_ankle | | |

每个点是 `[x, y, confidence]`。x 向右、y 向下，confidence 范围 0 至 1。可以使用像素坐标或 x/y 同比例的相机平面单位。不要把 x 分别除以图像宽、y 除以高，非正方形图像会因此扭曲角度。镜像时交换左右关节语义，训练与部署要一致。拒绝 NaN、Inf、错误长度、越界 confidence 和未声明字段；x/y 绝对值最多 10,000,000，避免异常大数在 float32 中溢出。

`timestamp_ms` 是当前设备单调递增的观测时间，单位毫秒，范围 0 至 1e15。可以从 0 开始，也可使用持续的单调时钟。时间戳在一批内和批次之间都必须严格递增。不要使用 HTTP 请求到达时间来替代视频帧采集时间，不要重复末帧填充网络间隔。

输入建议 30 Hz，可接收其他采样率，服务以因果线性插值重采样到 30 Hz，等待当前观测后处理它之前的插值时刻，不预测未来。超过 250 ms 的间隔会重置半周期和特征历史。低采样率不能保证保留短暂阶段，30 Hz 的离线结果不能用于推断 5 Hz 的效果。

两肩、两髋、两膝、两踝 confidence 均需 ≥0.3。腿部长度或躯干长度退化为零时进入 `degenerate_pose`，即使 confidence 很高也不计数。缺人、严重遮挡时仍发送当前时间戳和 confidence=0 的 17 个点，使系统及时进入 `insufficient_pose`。稳定 `person_id` 标识当前人，变化时中断计数。自带视频适配器对零人或多人画面拒绝计数；已有多人系统应先跟踪并选择目标人。

## 5. HTTP 路由

| 方法和路径 | 用途 | 成功状态 |
|---|---|---|
| GET `/health` | 确认服务和模型已加载 | 200 |
| POST `/v1/sessions` | 创建并开始一个训练会话 | 201 |
| POST `/v1/sessions/{id}/frames` | 发送一批骨架，获得当前状态 | 200 |
| GET `/v1/sessions/{id}` | 读取当前状态或终态报告 | 200 |
| POST `/v1/sessions/{id}/control` | pause、resume、end | 200 |
| DELETE `/v1/sessions/{id}` | 释放会话和缓存 | 204 |

### 创建会话

```json
{"exercise": "squat", "target_reps": 10}
```

`exercise` 默认 chair_sit_to_stand。`target_reps` 默认 10，范围 1 至 10000。服务返回 UUID `session_id` 和完整状态，`status="running"`、`quality="waiting_for_input"`、count=0。

### 发送骨架

请求结构如下，`keypoints` 必须是完整 17 点，真实合法示例在 `examples/squat_frames.json`，不要把下方说明性字符串直接发送。

```json
{
  "request_id": "device-a-batch-00001",
  "schema_name": "coco17",
  "coordinate_space": "camera_xy",
  "frames": [
    {"timestamp_ms": 0, "person_id": "person-1", "keypoints": "17 个 [x,y,confidence]"}
  ]
}
```

`request_id` 必填，1 至 128 字符。同一会话保留最近 128 个请求的结果，完全相同请求的重试返回原始结果，不再运行模型也不加次数。相同 request_id 的不同内容返回 409。超过缓存窗口的重试通常因时间戳过期返回 409。系统必须按会话串行发送，先收到第 N 批确认再发送第 N+1 批。

每个批次先验证全部帧的结构、有限数值和时间顺序，再开始处理，非法后半批不会使前半批悄悄生效。遇到服务运行时异常应读取会话当前时间戳确认接收状态，无法确认时结束会话并重新开始，不自行假设计数。

### 会话控制

```json
{"action": "pause"}
```

pause 清除未完成周期，已完成 count 保留。resume 同样从新周期开始。暂停期间发送帧返回 409。end 固定会话终态，重复 end 仍返回终态，结束后不再接收帧。DELETE 后任何路由均返回 404。控制命令顺序由系统管理，end 后不能 resume。

## 6. 输出字段

| 字段 | 类型 | 语义 |
|---|---|---|
| api_version | string | 固定 v1 |
| session_id | string | 会话 UUID |
| status | string | running、paused、ended |
| exercise | string | 系统指定的训练动作 |
| recognized_exercise | string | 当前最可能动作或 other，静止初段可能不确定 |
| exercise_probabilities | object | 四动作和 other 的类别概率和，合计约 1 |
| phase | string | 当前动作阶段或 other，枚举见动作表 |
| phase_confidence | number | 联合动作／阶段分类器的最大类别概率，未经可靠校准 |
| phase_probabilities | object | 选定动作内的条件阶段概率；运动阶段动作支持不足时 other=1 |
| count | integer | 已完成有效周期，不是四舍五入的回归值 |
| target_reps | integer | 系统设置的目标 |
| progress | number | min(count/target_reps,1) |
| target_completed | boolean | count ≥ target_reps |
| elapsed_s | number | 首帧至末帧的观测时间，包含暂停跨度 |
| active_duration_s | number | 运行中实际观测覆盖时间，不包含暂停恢复跨度或 >250ms 缺帧间隔 |
| valid_duration_s | number | 相邻有效可见观测覆盖时间，且没有失效中断 |
| quality | string | waiting_for_input、good、insufficient_pose、paused 等 |
| last_timestamp_ms | number/null | 最后接受的观测时间 |
| interruptions | array | reason 与 timestamp_ms，中断会重置半周期 |
| cycles | array | 完成的重复事件，含 rep_index、start_s、end_s、duration_s |
| model_id | string | 实际选定时序模型标识 |
| model_validation | string | 当前模型的实验验证范围标记 |
| request_id | string | 帧接口专有，当前请求 ID |
| accepted_frames | integer | 帧接口专有，本批接受数量 |
| new_cycles | array | 帧接口专有，仅本批新增周期 |

`cycles` 时间是相对会话第一帧的秒数。`start_s` 是起始基线阶段经消抖确认的时间，`end_s` 是回到基线阶段经消抖确认的时间，不是医学动作起止的人工精确边界。quality 恢复后可以立即显示 good，但必须重新经历完整周期才能新增 count。`interruptions` 保存 timestamp_gap、person_changed、insufficient_pose、paused 等原因。

## 7. 错误与系统处理

业务错误格式：`{"error":{"code":"..."},"api_version":"v1"}`。参数验证错误使用 FastAPI 标准 `{"detail":[...]}`，含 loc、msg、type，客户端不要依赖英文错误文本。

| HTTP | code／来源 | 处理 |
|---|---|---|
| 404 | SESSION_NOT_FOUND | 创建新会话，不复用旧 ID |
| 409 | SESSION_NOT_RUNNING | 停止发送，检查 paused／ended 状态 |
| 409 | SESSION_ENDED | 保存终态，再创建新会话 |
| 409 | TIMESTAMP_NOT_INCREASING | 修复队列顺序；不要偷偷改变原帧时间戳 |
| 409 | REQUEST_ID_CONFLICT | 修复幂等 ID 生成，内容不同不能复用 ID |
| 413 | BODY_TOO_LARGE | 减小批次 |
| 422 | 参数／帧校验 | 修复 schema、枚举、坐标、confidence 或数量 |
| 429 | SESSION_LIMIT_REACHED | 释放旧会话 |

网络超时重发同一 request_id 和完全相同 JSON 字段值。quality 失效是 200 下的正常业务状态，界面显示计数暂不可用，禁止用最后可信阶段填补缺失帧。

## 8. 可运行 Python HTTP 客户端

```python
import json
from pathlib import Path
import requests

base = "http://127.0.0.1:8002"
record = json.loads(Path("examples/squat_replay.json").read_text())
created = requests.post(base + "/v1/sessions", json={"exercise": record["exercise"], "target_reps": 10})
created.raise_for_status()
sid = created.json()["session_id"]
for start in range(0, len(record["frames"]), 10):
    body = {"request_id": f"batch-{start}", "frames": record["frames"][start:start+10]}
    response = requests.post(f"{base}/v1/sessions/{sid}/frames", json=body, timeout=30)
    response.raise_for_status()
    print(response.json()["count"])
ended = requests.post(f"{base}/v1/sessions/{sid}/control", json={"action": "end"})
ended.raise_for_status()
Path("session-report.json").write_text(json.dumps(ended.json(), indent=2))
requests.delete(f"{base}/v1/sessions/{sid}").raise_for_status()
```

## 9. SDK 与 Pose CNN 复用

```python
from homeease.engine import Predictor, ExerciseSession
from homeease.pose import PoseEstimator

predictor = Predictor(device="cpu")  # 进程共用权重，不共用会话历史
pose = PoseEstimator(device=0)  # 或 cpu，CNN 参数由 selected_pose.json 决定
session = ExerciseSession(predictor, exercise="standing_leg_raise", target_reps=10)
points, person_id = pose.estimate(frame_bgr)
state = session.update(capture_timestamp_ms, points, person_id)
session.control("pause")
session.control("resume")
final_report = session.control("end")
```

已有 Pose CNN 可直接绕过 PoseEstimator，使用同一 COCO 格式调用 update。SDK 也接受 12×3，顺序是 COCO 的索引 5 至 16；HTTP 为减少歧义只接受 17×3。SDK 时间戳同样严格递增；抛出 ValueError 的稳定 code 与 HTTP 表一致。摄像头多人跟踪由系统拥有，自带适配器只处理单人画面。

## 10. 给系统同学 AI 的集成提示词

```text
把 HomeEase 算法2接入现有系统。先读取 algorithm2/docs/API.md、
algorithm2/docs/openapi.json 和 algorithm2/README.md，然后检查现有项目的摄像头、
训练会话、报告存储及UI结构。直接复用 algorithm2/artifacts 的已训练权重，
不得重新写一个计数启发式来替代算法。

实现四个动作选项 chair_sit_to_stand、squat、standing_leg_raise、hurdle_step。
开始训练时创建 /v1/sessions，摄像头使用已有Pose CNN或homeease.pose.PoseEstimator，
按COCO17 [x,y,confidence]协议串行发送骨架批次。帧采集时间戳单调递增，
person_id稳定，网络重试使用同一request_id和同一内容。

显示动作、计数、目标进度及quality。计数以服务count为准，禁止客户端自行加一。
暂停/继续/结束分别调用control，结束后保存完整报告，再DELETE释放会话。
失去可见性时发送零confidence，提示暂不可计数。服务重启后新建会话，
保留已经保存的报告并标记中断，不恢复半周期。

先用 examples/*_replay.json 对四动作验收，检查SDK与HTTP得到相同结果，
再用连续摄像头视频人工记录真实次数验证。当前公共骨架结果不等于家庭视频准确率。
坚持单worker服务，局域网部署由系统负责认证和用户权限。集成修改完成后
运行算法tests以及系统自身测试，并交付启动方法、改动文件和验收结果。
```

## 11. 交接验收清单

1. GET health 返回 ready，model_id 与 artifacts/selected.json 一致。
2. 四个示例可完整回放；计数与 reports/replay_sessions.json 一致。
3. 重发同一批次不会增加计数，乱序帧返回 409。
4. pause、遮挡、换人和 >250 ms 缺帧不会跨中断拼接重复。
5. 原始视频保持在本地，系统只保存必要报告。
6. 服务重启能正确处理会话丢失，结束会话及时删除。
7. 连续摄像头数据有独立人工计数，报告模型错误、遮挡和视角条件。
