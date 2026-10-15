# API 与调用格式

启动后可在 `/docs` 查看请求 schema。HTTP 与 WebSocket 使用相同访问控制；非同源写请求拒绝。错误响应为 `error` 或 FastAPI `detail`，未找到 404、状态冲突 409、输入错误 422、模块不可用 503。

| 功能 | 主要接口 |
| --- | --- |
| 状态、配置 | `GET /api/health`、`GET /api/status`、`GET/PUT /api/settings` |
| 模块恢复 | `POST /api/modules/{name}/initialize` |
| 视频 | `POST /api/uploads/video`、`POST /api/video/start`、`POST /api/video/stop`、`GET /api/video/frame` |
| 连续画面 | `GET /api/video/stream`、`GET /api/sessions/{sid}/camera/stream`，MJPEG |
| 证据片段 | `GET /api/video/clip/{event_id}` |
| 信号回放 | `POST /api/signal/sample`、`POST /api/signal/analyze`，后者 multipart `file` |
| 实时信号 | `WS /ws/sensor` |
| 训练 | `POST /api/sessions`、`GET /api/sessions/{sid}`、`POST /api/sessions/{sid}/control` |
| 训练输入 | `POST /api/sessions/{sid}/frames`、`POST /api/sessions/{sid}/image` |
| 音频 | `POST /api/audio/sample`、`POST /api/audio/analyze`、`WS /ws/audio` |
| 事件 | `GET /api/events`、`GET /api/events/{id}`、`POST /api/events/{id}/action` |
| 问询 | `POST /api/events/{id}/inquiry`、`/inquiry/processing`、`/inquiry/unavailable` |
| 主动求助 | `POST /api/help` |
| 通知 | `GET /api/notifications`、`POST /api/notifications/{id}/seen`、`POST /api/notifications/test` |
| 报告 | `GET /api/reports`、`/api/reports/export.csv`、`/reports/print`、`/reports/events/{id}` |
| 训练导出 | `GET /api/sessions/{sid}/export.json` |

视频开始请求 `{"source":"sample"}` 或 `{"source":"camera","camera_index":0}`；上传后使用 `{"source":"upload","upload_id":"上传接口返回的值"}`。

训练开始请求：

```json
{"exercise":"chair_sit_to_stand","target_reps":10,"source":"camera"}
```

摄像头默认 `capture:"shared"`、`camera_index:0`，由服务统一采集，可与视频监测并行。外部图片／骨架采集客户端使用 `capture:"client"`，自行提交原输入契约。动作另有 `squat`、`standing_leg_raise`、`hurdle_step`，来源另有 `replay`、`upload`。控制为 `{"action":"pause"}`、`resume`、`end`。骨架批次遵循原算法2规范：

```json
{"request_id":"batch-0","frames":[{"timestamp_ms":0,"person_id":"no_person","keypoints":[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]}]}
```

上例表示一帧无人体观测，不计数；实际人体输入需 17 个原姿态模型的 `[x,y,confidence]` 点。图片输入为 multipart `file`，query 包含严格递增 `timestamp_ms` 和唯一 `request_id`。同一编号的不同内容返回冲突。

音频分析为 multipart `file`、`task`、`mode`；页面使用 `task:"monitor"`，返回 `sounds` 与 `help_result` 两个原算法结果。保留 `sounds`、`help` 和 `reply` 独立调用；reply 必须带关联 `event_id` 且已启动问询。样例默认联合分析，例如 JSON `{"sample":"reply_help_en"}`，固定作为回放。连续监听自动处理全部声音类别与语音求助，没有类别选择参数。

实时信号连接成功返回 `{"state":"connected"}`，发送：

```json
{"samples":[[0.0,0.0,0.0,1.0],[0.02,0.0,0.0,1.0]]}
```

每批 1 至 200 点，时间秒、加速度 g。返回原窗口/告警结果。结束发送 `{"type":"flush"}`，等待 `{"state":"stopped"}` 后断开；最终事件已入库。

连续音频连接成功返回 `{"state":"listening"}`。发送 `{"pcm":"base64 little-endian float32 mono PCM","timestamp_ms":0}`，16 kHz、每块不超过 32000 点、时间递增；浏览器每块 8000 点。处理后返回原 pipeline/listener 状态及声音/求助结果。停止发送 `{"type":"stop"}` 并等待 stopped 确认。

事件操作有 `acknowledge`、`contact`、`escalate`、`safe`、`resolve`。照护者更新附 `revision` 可避免覆盖并发处理。resolve 需要 `outcome`：`confirmed_fall`、`false_alarm`、`safe`、`unresolved`，可附 `note`。报告参数 `days=1..90` 与 `mode=live|replay`。
