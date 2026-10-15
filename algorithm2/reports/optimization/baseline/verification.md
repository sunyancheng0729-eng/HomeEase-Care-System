# 最终验证记录

日期：2026-10-08，Asia/Shanghai。

## 验证结果

| 验证项 | 结果 | 证据 |
|---|---|---|
| 原始文本来源与哈希 | 2,000 文件通过 SHA-256／Git blob SHA 校验 | source_integrity.json |
| 参与者隔离 | s01–06／s07–08／s09–10 无交叉 | records.json、pytest |
| 所有训练 checkpoint | 11 个实际权重均重加载并重新评估 | experiments.json |
| 模型因果性 | TCN／GRU／Transformer 不读取未来帧 | pytest |
| 完整周期与中断 | 四动作、暂停、换人、遮挡、缺帧、退化骨架通过 | pytest |
| HTTP schema／幂等／乱序／生命周期 | 通过 | pytest、openapi.json |
| 真权重批推理／流推理 | 数值在指定容差内一致 | test_trained_pipeline.py |
| 真权重四动作 HTTP／SDK | 完整实际服务回放 count 一致 | replay_sessions.json |
| Python 测试 | 最终 24 passed，36.30 秒，一条依赖弃用提示 | pytest -q 输出 |
| 视频／CNN 入口 | 真实 COCO 图片生成的 60 帧静止视频跑通，count=0 | smoke_video_session.json |
| 固定模型视角／噪声 | 四组完成，未据此修改模型 | robustness.json |
| 图表 | 已检查并修复比较图文字重叠 | reports/figures |
| localhost 服务 | health ready，model_id=gru_b2_w64，Swagger HTTP 200 | http://127.0.0.1:8002/docs |

## 实际完整 HTTP 回放

| 动作 | 来源参考 | SDK | HTTP |
|---|---:|---:|---:|
| chair_sit_to_stand，s09 | 9 | 2 | 2 |
| squat，s09 | 10 | 10 | 10 |
| standing_leg_raise，s09 | 10 | 10 | 10 |
| hurdle_step，s09 | 10 | 10 | 10 |

一致性通过不意味着预测全部正确。坐站这一会话有明显漏计，默认模型在完整测试上的总体 MAPE 为 23.47%，未达到 proposal 的 ≤5% 目标。

## 尚未完成的验证

没有实测老人进行四动作的连续摄像头输入，没有独立人工逐帧阶段金标准，没有自然家庭干扰／多人覆盖验收，也未测长期运行。静止视频仅验证读取、Pose CNN、时序模型和 JSON 报告链路，不证明摄像头 exercise counting 准确率。

主训练长时工具命令返回非零状态但没有 Python 异常记录，输出包含全部 checkpoint。后续独立重加载评估、实际服务回放和测试均完成且返回 0，模型有效性以这些验证为依据。
