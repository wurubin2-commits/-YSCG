"""Export a readable report, a per-round table and a chart from matched receipts."""

import csv
from dataclasses import asdict


TASKS = {"A": "Macro-F1", "B": "RMSE (minutes)", "C": "RMSE (cycles)"}


def draw_chart(comparison, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    figure.suptitle("Ten-round public validation replay", fontsize=16)
    for (task, rounds), (scores, gains) in zip(comparison.items(), axes, strict=True):
        numbers = [row.number for row in rounds]
        scores.plot(numbers, [row.parent_score for row in rounds], "--", label="Parent", color="#65758b")
        scores.plot(numbers, [row.candidate_score for row in rounds], "o-",
                    label="Candidate", color="#d9762a", markersize=4)
        scores.plot(numbers, [row.selected_score for row in rounds], "s-",
                    label="Selected", color="#184b96", markersize=3, alpha=0.8)
        scores.set_ylabel(f"{task} | {TASKS[task]}")
        gains.bar(numbers, [row.gain for row in rounds],
                  color=["#28785e" if row.adopted else "#be6754" for row in rounds])
        gains.axhline(0, color="#4c5666", linewidth=0.8)
        gains.set_ylabel("Candidate gain")
        for axis in (scores, gains):
            axis.set_xticks(numbers)
            axis.grid(axis="y", alpha=0.2)
            axis.set_xlim(0.5, 10.5)
    axes[0, 0].set_title("Parent, candidate and selected model")
    axes[0, 1].set_title("Same-round change (positive is better)")
    axes[0, 0].legend(fontsize="small", ncol=3)
    for axis in axes[-1]:
        axis.set_xlabel("Round")
    figure.savefig(path, dpi=160)


def markdown(comparison, reference_name, current_name):
    lines = [
        "# 第一阶段｜十轮公开验证回顾", "",
        f"输入是旧版 `{reference_name}` 和整合版 `{current_name}` 的本地回执。",
        "A、B、C 各十轮，合计逐轮核对两版 60 条记录。原演练使用真实 CPU 续训和模拟模型建议，评分取公开验证集。",
        "同轮父检查点、训练与验证数据摘要、种子、训练参数一致，每轮候选检查点和采用决定也一致。下表耗时取整合版回执。", "",
        "![三项任务十轮曲线和每轮候选收益](rounds.png)", "",
        "[逐轮明细 CSV](rounds.csv)", "",
        "| 任务 | 公开主指标 | 起点 | 第十轮选中 | 采用新模型 | 验证投影数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task, rounds in comparison.items():
        lines.append(f"| {task} | {TASKS[task]} | {rounds[0].parent_score:.4f} | "
                     f"{rounds[-1].selected_score:.4f} | {sum(row.adopted for row in rounds)}/10 | "
                     f"{len({row.validation_sha for row in rounds})} |")

    bearing = comparison["B"]
    kept = [row for row in bearing if not row.adopted]
    lines.extend(("", "## B 轴承任务逐轮结果", "",
                  "RMSE 与 MAE 的单位均为分钟。收益为父模型 RMSE 减去候选 RMSE。",
                  "轮次耗时为本地 Harness 单次记录，包含模型模拟请求与训练相关开销。", "",
                  "| 轮次 | 父 RMSE | 候选 RMSE | 收益 | 候选 MAE | 决策 | 更新步数 | 轮次耗时（秒） |",
                  "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"))
    for row in bearing:
        decision = "采用候选" if row.adopted else "保留父模型"
        lines.append(f"| {row.number} | {row.parent_score:.2f} | {row.candidate_score:.2f} | "
                     f"{row.gain:+.2f} | {row.candidate_mae:.2f} | {decision} | {row.updates} | {row.elapsed:.3f} |")
    if kept:
        worst = min(kept, key=lambda row: row.gain)
        worse = sum(row.gain < 0 for row in kept)
        lines.extend(("", f"B 的 {len(kept)} 次保留来自训练后比较，其中 {worse} 次候选 RMSE 升高。"
                      f"第 {worst.number} 轮候选收益最低，RMSE 从 {worst.parent_score:.2f} 变为 "
                      f"{worst.candidate_score:.2f} 分钟。"))
    if (len({row.validation_sha for row in bearing}) == 1
            and all(row.selected_score <= row.parent_score for row in bearing)):
        lines.append("B 十轮共用同一公开验证投影，选中模型的 RMSE 未出现回升，因此这组记录没有可度量的恢复轮数。")
    lines.extend(("", "## 下一阶段的对照依据", "",
                  f"B 每轮实际更新 {min(row.updates for row in bearing)} 至 {max(row.updates for row in bearing)} 步。"
                  "L2-SP 的父模型约束从后续梯度更新起可能产生作用，这批日志支持开展对照实验。",
                  "各轮使用不同的训练批次，父模型和学习率也可能变化。只凭这次回放无法判定某个因素导致了退步。",
                  "保留现行方案作为对照。固定初始检查点、顺序、随机种子、学习率和训练预算，先验零强度能复现现行训练，"
                  "再按预定强度测 B 的整体 RMSE、MAE、低寿命误差和资源开销。", "",
                  "本次回顾仅分析已有回执。实际 Token 用量和正式平台效果仍待验证。", ""))
    return "\n".join(lines)


def write_review(comparison, reference_name, current_name, output):
    output.mkdir(parents=True, exist_ok=True)
    draw_chart(comparison, output / "rounds.png")
    rows = [{**asdict(row), "gain": row.gain} for rounds in comparison.values() for row in rounds]
    with (output / "rounds.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.md").write_text(markdown(comparison, reference_name, current_name), encoding="utf-8")
