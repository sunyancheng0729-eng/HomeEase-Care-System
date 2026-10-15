# 最终验证记录

日期：2026-10-08，Asia/Shanghai。

## 验证结果

| 验证项 | 结果 | 证据 |
|---|---|---|
| 原始来源与参与者隔离 | 2,000 文件哈希通过；s01–06／s07–08／s09–10 无交叉 | `reports/source_integrity.json`、`records.json` |
| 优化候选与测试前锁 | 3 个 checkpoint 重加载；`opt_tcn_b4_w48`、decoder、阈值和 SHA 已锁定 | `reports/optimization/results.json`、`selection_lock.json` |
| 模型与特征自检 | 原模型及优化 TCN／GRU 因果；stable80 不变性；按需窗口与预展开窗口一致 | `tests/test_contract.py` |
| 周期、HTTP、真实权重 | 四动作与中断、HTTP 幂等／生命周期、批／流推理一致 | `pytest`、`docs/openapi.json` |
| 实际 localhost 服务 | ready，`model_id=opt_tcn_b4_w48`，Swagger HTTP 200；四动作完整 HTTP／SDK count 一致 | `reports/replay_sessions.json`、`http://127.0.0.1:8002/docs` |
| 优化后 Python 测试 | **28 passed**，1 条 Starlette/httpx 弃用提示 | `pytest -q` |
| 负样本和静止输入 | 两个 m07 回放对四目标均 0；静止 300 帧对四目标均 0 | `reports/optimization/deployment_audit.json` |
| 同轮 CPU 审计 | 原 GRU 8.926 ms P50；优化 TCN 5.619 ms P50 | `deployment_audit.json` |
| 视频／CNN 入口 | 静止 COCO 图片重复视频跑通，count=0；不是动作精度测试 | `reports/smoke_video_session.json` |
| 视角／噪声压力测试 | 15°、45°、75°及坐标噪声已运行，未据此改模型 | `reports/robustness.json` |

## 优化后固定测试回放

部署模型：`opt_tcn_b4_w48`。测试动作 Macro-F1 **0.9612**，联合动作／阶段 Macro-F1 **0.8203**；10 个重建会话 MAE **0.10 次**，八个非零会话 MAPE **1.25%**，周期 recall／precision **0.9872／1.0000**。78 个参考周期匹配 77 个，漏 1 个直腿抬腿周期，没有多计。

| 动作 | 来源参考 | SDK 预测 | HTTP 预测 |
|---|---:|---:|---:|
| chair_sit_to_stand，s09 | 9 | 9 | 9 |
| squat，s09 | 10 | 10 | 10 |
| standing_leg_raise，s09 | 10 | 10 | 10 |
| hurdle_step，s09 | 10 | 10 | 10 |

原默认 GRU 的测试动作 F1=0.8213、阶段 F1=0.7007、计数 MAE=1.80、MAPE=23.47%。优化 GRU 在测试回放中 MAPE=0%，但验证 MAE=0.80 且 CPU P50=13.706 ms；根据测试前验证锁选择 TCN，不能用测试结果改选模型。

## 服务复验

旧后台进程已经关闭，新服务成功加载 `opt_tcn_b4_w48`。已执行以下命令，四动作 SDK／HTTP 数量一致，所有会话结束并释放：

```powershell
Set-Location E:\gc2-project-algorithm\algorithm2
.\.venv\Scripts\python.exe scripts/replay_demo.py --url http://127.0.0.1:8002
```

health 返回 ready、`model_id=opt_tcn_b4_w48`，Swagger 返回 200。HTTP 服务以一个 worker 运行，地址 `http://127.0.0.1:8002/docs`。训练、校准、评估、压力测试、接口回放和视频冒烟命令都已完成，只保留本地 API 服务供系统接入。

## 尚未完成的验证

没有实测老人连续摄像头输入，没有独立人工逐帧阶段金标准，没有自然家庭干扰／多人覆盖验收，也未测长期运行。二维投影骨架、弱阶段标签和两个测试参与者不能代表老人群体泛化。有限回放 MAPE 达到 proposal 的 ≤5% 只说明这套公开重建回放，不等于家庭视频验收完成，也不证明预防跌倒临床效果。
