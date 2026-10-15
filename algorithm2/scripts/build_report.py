"""Generate auditable Markdown and scientific figures from actual JSON results."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.exercises import EXERCISE_NAMES


def main():
    results = json.loads((ROOT / "reports/experiments.json").read_text())
    poses = json.loads((ROOT / "reports/pose_comparison.json").read_text())
    selected = json.loads((ROOT / "artifacts/selected.json").read_text())
    row = next(r for r in results if r["model"] == selected["model"])
    test_best = min(results, key=lambda r: (r["test"]["count_mae"], -r["test"]["phase_macro_f1"]))
    baseline_path = ROOT / "reports/optimization/baseline/selected.json"
    baseline_name = json.loads(baseline_path.read_text())["model"] if baseline_path.exists() else None
    baseline = next((r for r in results if r["model"] == baseline_name), None)
    selected_pose = json.loads((ROOT / "artifacts/selected_pose.json").read_text())
    figs = ROOT / "reports/figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    handles, legend_labels = [], []
    for number, r in enumerate(results, 1):
        marker = "*" if r["model"] == row["model"] else "o"
        handle = axes[0].scatter(r["cpu_p50_ms"], r["validation"]["count_mae"], marker=marker, s=120 if marker == "*" else 40)
        handles.append(handle)
        legend_labels.append(f"{number}. {r['model']}")
        if r["model"] == row["model"]:
            axes[0].annotate(f"Recommended: {r['model']}", (r["cpu_p50_ms"], r["validation"]["count_mae"]),
                             fontsize=10, fontweight="bold", color="darkgreen", xytext=(0, 20),
                             textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "darkgreen"})
        else:
            axes[0].annotate(str(number), (r["cpu_p50_ms"], r["validation"]["count_mae"]), fontsize=9,
                             xytext=(5, 6 if number % 2 or r["validation"]["count_mae"] <= 1.05 else -14), textcoords="offset points")
    axes[0].set(xlabel="CPU temporal inference p50 (ms)", ylabel="Validation count MAE (reps)", title="Validation accuracy vs compute")
    axes[0].grid(alpha=.2)
    for number, p in enumerate(poses):
        axes[1].scatter(p["latency_p50_ms"], p["pck_bbox_02"] * 100)
        axes[1].annotate(f"{p['model'].replace('-pose.pt','')}@{p['imgsz']}", (p["latency_p50_ms"], p["pck_bbox_02"] * 100),
                         fontsize=8, xytext=(3, -14 if number == 1 else 6), textcoords="offset points")
    axes[1].set(xlabel="GPU full pose call p50 (ms)", ylabel="PCK@0.2bbox (%)", title="COCO 100-image pose subset")
    axes[1].grid(alpha=.2)
    axes[1].margins(x=.2)
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .15, 1, 1))
    fig.savefig(figs / "accuracy_compute.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix = np.asarray(row["test"]["action_confusion_matrix"])
    plot = ax.imshow(matrix, cmap="Blues")
    labels = ["Chair STS", "Squat", "Leg raise", "Hurdle step", "Other"]
    ax.set_xticks(range(5), labels, rotation=35, ha="right")
    ax.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * .5 else "black")
    ax.set(xlabel="Predicted action", ylabel="Source action", title=f"Held-out action confusion: {row['model']}")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(figs / "action_confusion.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    sessions = row["test"]["sessions"]
    x = np.arange(len(sessions))
    ax.bar(x - .2, [s["expected"] for s in sessions], width=.4, label="Source reference")
    ax.bar(x + .2, [s["predicted"] for s in sessions], width=.4, label="Predicted")
    ax.set_xticks(x, [s["session"].replace("_reconstructed", "") for s in sessions], rotation=40, ha="right")
    ax.set(ylabel="Complete cycles", title="Test reconstructed sessions, including negative actions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "test_session_counts.png", dpi=160)
    plt.close(fig)
    lines = ["# HomeEase 四动作算法 2 实验结果", "",
             "## 最佳算法结果速览", "",
             f"> **综合最佳、当前部署推荐：分层因果 TCN `{row['model']}`**\n> 验证计数 MAE **{row['validation']['count_mae']:.2f} 次**；测试计数 MAPE **{row['test']['count_mape_pct']:.2f}%**；测试动作 Macro-F1 **{row['test']['action_macro_f1']:.4f}**；CPU P50 **{row['cpu_p50_ms']:.3f} ms**。", "",
             f"> **测试回放指标最高：优化 GRU `{test_best['model']}`**\n> 测试计数 MAPE **{test_best['test']['count_mape_pct']:.2f}%**；测试动作 Macro-F1 **{test_best['test']['action_macro_f1']:.4f}**；测试联合阶段 Macro-F1 **{test_best['test']['phase_macro_f1']:.4f}**。", "",
             f"| 核心指标 | **综合最佳 TCN（推荐部署）** | **测试指标最高 GRU** |", "|---|---:|---:|",
             f"| 模型标识 | **`{row['model']}`** | **`{test_best['model']}`** |"]
    for label, split, key, fmt, suffix in [
        ("验证动作 Macro-F1 ↑", "validation", "action_macro_f1", ".4f", ""),
        ("验证联合阶段 Macro-F1 ↑", "validation", "phase_macro_f1", ".4f", ""),
        ("验证计数 MAE ↓", "validation", "count_mae", ".2f", " 次"),
        ("测试动作 Macro-F1 ↑", "test", "action_macro_f1", ".4f", ""),
        ("测试联合阶段 Macro-F1 ↑", "test", "phase_macro_f1", ".4f", ""),
        ("测试计数 MAE ↓", "test", "count_mae", ".2f", " 次"),
        ("测试计数 MAPE ↓", "test", "count_mape_pct", ".2f", "%"),
        ("测试相对计数 accuracy ↑", "test", "count_accuracy_pct", ".2f", "%"),
        ("测试会话完全正确率 ↑", "test", "exact_session_pct", ".1f", "%"),
        ("测试周期 recall ↑", "test", "cycle_recall", ".4f", ""),
        ("测试周期 precision ↑", "test", "cycle_precision", ".4f", ""),
    ]:
        values = [f"{r[split][key]:{fmt}}{suffix}" for r in [row, test_best]]
        lines.append(f"| {label} | **{values[0]}** | **{values[1]}** |")
    lines += [f"| CPU 时序前向 P50 ↓ | **{row['cpu_p50_ms']:.3f} ms** | {test_best['cpu_p50_ms']:.3f} ms |",
              f"| CPU 时序前向 P95 ↓ | **{row['cpu_p95_ms']:.3f} ms** | {test_best['cpu_p95_ms']:.3f} ms |",
              f"| 参数量 | {row['parameters']:,} | {test_best['parameters']:,} |", "",
              f"**为什么推荐 TCN：**它的验证计数 MAE 为 **{row['validation']['count_mae']:.2f} 次**，低于 GRU 的 {test_best['validation']['count_mae']:.2f} 次；GRU 的 CPU P50 是 TCN 的 **{test_best['cpu_p50_ms']/row['cpu_p50_ms']:.2f} 倍**。因此根据验证表现和计算成本选择 TCN，保留 GRU 作为测试回放精度最高的候选。模型与 decoder 已在本轮测试前锁定。", "",
              "以上 F1 是 Macro-F1；阶段标签为运动学弱标签。MAE 包含十个会话（八正、两负），MAPE 排除两个零参考会话；相对计数 accuracy=100%−MAPE，不是分类正确率。CPU 是本机单次时序前向，未包含 Pose CNN、HTTP、特征提取和排队开销。", ""]
    if baseline:
        lines += ["## 优化前后：原默认 GRU → 综合最佳 TCN", "",
                  f"| 指标 | 原默认 `{baseline['model']}` | **优化后 `{row['model']}`** |", "|---|---:|---:|"]
        for label, key, fmt, suffix in [("测试动作 Macro-F1", "action_macro_f1", ".4f", ""),
                                        ("测试联合阶段 Macro-F1", "phase_macro_f1", ".4f", ""),
                                        ("测试计数 MAE", "count_mae", ".2f", " 次"),
                                        ("测试计数 MAPE", "count_mape_pct", ".2f", "%"),
                                        ("测试周期 recall", "cycle_recall", ".4f", "")]:
            lines.append(f"| {label} | {baseline['test'][key]:{fmt}}{suffix} | **{row['test'][key]:{fmt}}{suffix}** |")
        lines += ["", f"测试动作 F1 提升 **{100*(row['test']['action_macro_f1']-baseline['test']['action_macro_f1']):.2f} 个百分点**；计数 MAE 降低 **{100*(1-row['test']['count_mae']/baseline['test']['count_mae']):.2f}%**。这是同一标签与划分下的描述性对照。", ""]
    matched, spurious, missed = (row["test"][k] for k in ["matched_cycles", "spurious_cycles", "missed_cycles"])
    lines += ["## 综合最佳 TCN 的四动作效果", "",
              f"**{matched}/{matched+missed} 个参考周期匹配成功，漏计 {missed} 个，多计 {spurious} 个。** 两个肩部动作负会话均为 0 次。", "",
              "| 动作 | 参考周期总数 | 预测周期总数 | MAE 次 | MAPE | 会话完全正确率 | 周期 recall | 周期 precision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    action_names = {"chair_sit_to_stand": "椅子坐站", "squat": "深蹲", "standing_leg_raise": "站立直腿抬腿", "hurdle_step": "跨步抬腿"}
    for exercise in EXERCISE_NAMES:
        m = row["test"]["per_exercise"][exercise]
        expected = sum(s["expected"] for s in m["sessions"])
        predicted = sum(s["predicted"] for s in m["sessions"])
        lines.append(f"| {action_names[exercise]} `{exercise}` | {expected} | **{predicted}** | {m['count_mae']:.2f} | **{m['count_mape_pct']:.2f}%** | {m['exact_session_pct']:.1f}% | {m['cycle_recall']:.3f} | {m['cycle_precision']:.3f} |")
    lines += ["", "每动作仅两名测试参与者。周期时间匹配使用弱边界 ±0.5 秒；数量参考来自原始分段重复。", "",
              "## 最佳算法部署配置", "",
              f"实际加载权重：`{', '.join(selected['members'])}`，见 `artifacts/selected.json`。Decoder 为 `{selected['decoder_strategy']}`，阶段证据阈值 {selected['phase_evidence_threshold']}，消抖时长 {selected['min_dwell_s']} 秒，部署窗口 {row.get('config', {}).get('context', 64)} 帧。默认姿态前端：**{selected_pose['model']} @ {selected_pose['imgsz']}**。系统接入见 [API.md](API.md)，完整优化过程见 [OPTIMIZATION.md](OPTIMIZATION.md)。", "",
              f"部署选择规则：`{selected['selection_rule']}`。原测试集在前一轮已经查看，本轮先锁定验证集选择，再复用测试集作描述性比较，不视为全新盲测。公开骨架投影、弱标签和重建回放结果不等于老人家庭视频性能。", "",
              "## 完整模型对比（最佳候选置顶）", "",
              "| 模型与定位 | 参数 | CPU P50 ms | 验证 MAE | 测试动作 F1 | 测试联合阶段 F1 | 测试 MAE | 测试 MAPE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    ordered = [row] + ([test_best] if test_best["model"] != row["model"] else [])
    ordered += [r for r in results if r["model"] not in {row["model"], test_best["model"]}]
    for r in ordered:
        label = r["model"]
        if r["model"] == row["model"]:
            label += "（综合最佳／推荐部署）"
        elif r["model"] == test_best["model"]:
            label += "（测试指标最高）"
        cells = [label, f"{r['parameters']:,}", f"{r['cpu_p50_ms']:.3f}", f"{r['validation']['count_mae']:.2f}",
                 f"{r['test']['action_macro_f1']:.4f}", f"{r['test']['phase_macro_f1']:.4f}",
                 f"{r['test']['count_mae']:.2f}", f"{r['test']['count_mape_pct']:.2f}%"]
        if r["model"] in {row["model"], test_best["model"]}:
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "集成行延迟为各成员独立测量之和，尤其 P95 相加不是实测联合 P95。首轮 seed=42，优化轮 seed=2026；未做多种 seed 的统计重复。CPU 是本机测量，旧轮与新轮计时条件可能不同。", ""]
    lines += ["## 逐测试会话", "", "| 重建会话 | 来源参考 | 预测 |", "|---|---:|---:|"]
    for s in sessions:
        lines.append(f"| {s['session']} | {s['expected']} | {s['predicted']} |")
    lines += ["", "## 姿态 CNN 对比", "", "100 张官方 COCO val 图像，198 个标注人体，2,482 个可见关键点。此处不是全量 COCO AP，PCK 阈值为 bbox 最长边的 20%，漏检计错。", "", "| Pose CNN | 输入 | PCK@0.2 | 平均 OKS | 匹配人体 | GPU P50 ms | GPU P95 ms | 参数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in poses:
        lines.append(f"| {p['model']} | {p['imgsz']} | {p['pck_bbox_02']*100:.2f}% | {p['mean_oks_including_misses']:.4f} | {p['matched_persons']}/{p['persons']} | {p['latency_p50_ms']:.2f} | {p['latency_p95_ms']:.2f} | {p['parameters']:,} |")
    lines += ["", "计时包含预处理、前向与后处理，并做 CUDA 同步。P50 达到 30 FPS 的预算不代表 P95 能达到，也不包含所有系统开销。", "", "## 图表", "", "![验证精度与计算比较](../reports/figures/accuracy_compute.png)", "", "![动作混淆矩阵](../reports/figures/action_confusion.png)", "", "![测试会话计数](../reports/figures/test_session_counts.png)", "", "## 是否达到 proposal 的目标", ""]
    lines += [f"当前默认模型重建回放总体 MAPE 为 {row['test']['count_mape_pct']:.2f}%。" + ("在该有限回放上达到 ≤5%，但尚不能宣布家庭视频验收完成。" if row["test"]["count_mape_pct"] <= 5 else "未达到 ≤5% 的目标，不能把实现完成写成精度目标已经完成。"), "", "代码、训练权重、四动作识别与计数流程、接口和离线实验已实现。下一步应使用独立人工标记的自然连续摄像头视频，包含老人、不同视角、遮挡及日常干扰，然后验证真实前端的关键点误差和完整周期性能。", ""]
    robustness_path = ROOT / "reports/robustness.json"
    if robustness_path.exists() and all(r.get("model") == selected["model"] for r in json.loads(robustness_path.read_text())):
        robustness = json.loads(robustness_path.read_text())
        lines += ["## 固定模型的视角与噪声测试", "", "下表只包含四动作的八个正会话，因此 MAE 的分母与含两个负会话的主表不同；模型与阈值没有根据这些测试结果修改。噪声单位是 UI-PRMD 原始长度单位，不是特定摄像头的像素。", "", "| 投影视角 | 坐标噪声标准差 | 正会话 MAE | MAPE |", "|---|---:|---:|---:|"]
        for r in robustness:
            lines.append(f"| {r['angle_degrees']}° | {r['noise_source_units_std']} | {r['mae']:.3f} | {r['mape_pct']:.2f}% |")
        lines += ["", "未见训练参与者的 15°／75°视角是投影鲁棒性压力测试，不能替代实际摄像头角度验收。", ""]
    lines += ["本文件由 `scripts/build_report.py` 从实际实验 JSON 生成。开发方法和评估限制见 [DEVELOPMENT.md](DEVELOPMENT.md)。", ""]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    environment = {"python": sys.version, "platform": platform.platform(), "packages": {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "matplotlib", "opencv-python", "ultralytics", "fastapi", "uvicorn", "pytest"]}}
    (ROOT / "reports/environment.json").write_text(json.dumps(environment, indent=2))
    manifest = [{"file": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((ROOT / "artifacts").glob("*")) if p.is_file()]
    (ROOT / "reports/artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Generated docs/RESULTS.md, figures, environment and artifact hashes")


if __name__ == "__main__":
    main()
