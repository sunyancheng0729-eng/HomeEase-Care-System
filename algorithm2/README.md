# HomeEase Algorithm 2

四动作骨架时序识别与完整周期计数。包含真实数据下载、训练、结构／特征消融、姿态 CNN 对比、可加载权重、Python SDK、本地视频／摄像头入口、HTTP API 和交接文档。

支持 `chair_sit_to_stand`、`squat`、`standing_leg_raise`、`hurdle_step`。计数必须经历完整的四阶段顺序，失去可见性、暂停或换人后重新开始一个完整周期。

当前默认算法为优化后的 **分层因果 TCN `opt_tcn_b4_w48`**：62,453 参数，64 帧窗口。固定测试重建回放中，动作 Macro-F1 0.9612、联合阶段 Macro-F1 0.8203、计数 MAE 0.10 次、MAPE 1.25%；首轮默认 MAPE 为 23.47%。模型与 decoder 由验证集选择并在本轮测试前锁定，接口 v1 可直接使用新权重。完整过程见 [优化记录](docs/OPTIMIZATION.md)。这些结果不代表老人真实摄像头精度。

## 本机启动

```powershell
Set-Location E:\gc2-project-algorithm\algorithm2
.\.venv\Scripts\python.exe -m uvicorn homeease.api:app --host 127.0.0.1 --port 8002 --workers 1
```

接口浏览器：<http://127.0.0.1:8002/docs>，健康检查：<http://127.0.0.1:8002/health>。已有训练权重，可直接推理，不需要重新训练。项目虚拟环境复用本机 `E:\conda_envs\dl` 的 CUDA PyTorch。

## 使用算法

```powershell
# 离线真实骨架回放，同时检查实际 HTTP 集成
.\.venv\Scripts\python.exe scripts/replay_demo.py --url http://127.0.0.1:8002
# 摄像头，q 结束，p 暂停／继续
.\.venv\Scripts\python.exe scripts/run_video.py --source 0 --exercise chair_sit_to_stand
# 本地视频，不弹出窗口
.\.venv\Scripts\python.exe scripts/run_video.py --source path\exercise.mp4 --exercise squat --no-display
```

```python
from homeease.engine import Predictor, ExerciseSession

predictor = Predictor(device="cpu")
session = ExerciseSession(predictor, exercise="squat", target_reps=10)
state = session.update(timestamp_ms=0, keypoints=coco17_points, person_id="person-1")
report = session.control("end")
```

`coco17_points` 为 17 个 `[x, y, confidence]`，横坐标向右、纵坐标向下，x/y 使用相同单位。实时流会因果重采样到 30 Hz。

## 复现实验

新机器先创建 Python 3.11 环境，按设备从 PyTorch 官方安装合适的 CPU 或 CUDA 版本，然后安装 `requirements.txt`。本项目不包含虚拟环境在其他机器上的可移植保证。

```powershell
python -m pip install -r requirements.txt
python scripts/download_data.py --movements m01 m02 m05 m06 m07
python scripts/prepare_data.py
python scripts/train_compare.py --epochs 25 --seed 42
python scripts/evaluate_select.py
python scripts/benchmark_pose.py --images 100
python scripts/optimize_top.py --epochs 80 --seed 2026
python scripts/optimize_top.py --calibrate
python scripts/optimize_top.py --finalize
python scripts/audit_optimized.py
python scripts/robustness.py
python scripts/replay_demo.py
python -m pytest -q
python scripts/build_optimization_report.py
python scripts/build_report.py
```

## 交付导航

- [详细开发记录](docs/DEVELOPMENT.md)：需求、数据、标签、模型、实验、选择理由、局限和复现。
- [优化与自检记录](docs/OPTIMIZATION.md)：三个强候选的优化、原模型对照、验证集选择锁及实际训练日志。
- [实际实验结果](docs/RESULTS.md)：默认配置、首轮与优化轮对比、分动作误差、计算成本及压力测试。
- [系统接口与 AI 接入指南](docs/API.md)：完整 HTTP／SDK 契约、字段、错误、示例及可直接交接的集成提示词。
- [机器可读 OpenAPI](docs/openapi.json)：可生成系统端客户端。
- `artifacts/selected.json` 与 `selected_pose.json`：验证集选择的默认模型。
- `reports/comparison.csv`、`experiments.json`、`pose_comparison.json`：实际结果。
- `examples/*_replay.json`：四种真实测试参与者的重建骨架回放。
- [最终验证记录](reports/verification.md)：自动测试、实际服务回放、视频入口验证与尚未验证的范围。

## 结果适用范围

四动作实验使用 UI-PRMD 真实 3D 骨架投影，阶段标签为明确记录的运动学弱标注。计数评估使用原始分段重复拼成的回放，不是自然连续家庭视频。COCO 子集用于比较通用姿态估计，不代表老年人摄像头计数精度。应在系统同学采集的连续视频上人工标记阶段和计数，再验证摄像头链路，才能宣称满足 proposal 的相对误差 ≤5% 目标。
