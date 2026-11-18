"""Evaluate the three supplied CPU checkpoints on their public validation sets."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    ("A", "气体识别", "a-气体识别", "分类"),
    ("B", "轴承剩余寿命", "b-轴承剩余寿命预测", "分钟"),
    ("C", "发动机剩余寿命", "c-发动机寿命预测", "循环"),
]


def evaluate(code, label, directory, unit):
    folder = ROOT / directory / "cpu工业模型"
    sys.path.insert(0, str(folder))
    try:
        from baselines.data import read_dataset
        from baselines.model import Model

        checkpoint = folder / "checkpoint.npz"
        model = Model.load(checkpoint)
        data = read_dataset(
            folder / "public" / "validation.jsonl",
            model.meta["task"],
            model.meta["classes"],
            model.meta["paths"],
        )
        predictions = model.predict(data["x"])
        if model.meta["task"] == "classification":
            predicted = predictions.argmax(axis=1)
            actual = data["y"]
            f1_by_class = {}
            for index, class_name in enumerate(model.meta["classes"]):
                tp = int(np.sum((actual == index) & (predicted == index)))
                fp = int(np.sum((actual != index) & (predicted == index)))
                fn = int(np.sum((actual == index) & (predicted != index)))
                f1_by_class[class_name] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            scores = {
                "accuracy": float(np.mean(predicted == actual)),
                "macro_f1": float(np.mean(list(f1_by_class.values()))),
                "f1_by_class": f1_by_class,
            }
            example = {"actual": model.meta["classes"][int(actual[0])], "predicted": model.meta["classes"][int(predicted[0])]}
        else:
            errors = predictions - data["y"]
            scores = {
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
            }
            example = {"actual": float(data["y"][0]), "predicted": float(predictions[0])}
        return {
            "task": code,
            "name": label,
            "unit": unit,
            "validation_rows": len(data["y"]),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "epochs": model.meta["epochs"],
            "scores": scores,
            "first_example": example,
        }
    finally:
        sys.path.remove(str(folder))
        for name in list(sys.modules):
            if name == "baselines" or name.startswith("baselines."):
                del sys.modules[name]


def main():
    results = [evaluate(*task) for task in TASKS]
    output = ROOT / "初始模型运行结果.md"
    lines = [
        "# 三个初始工业模型：公开验证集运行结果",
        "",
        "本报告由工作区中的 checkpoint.npz 重新加载并预测 public/validation.jsonl 得到。",
        "这是现有模型的验证结果，没有续训，也不是比赛隐藏测试成绩。",
        "",
        "| 任务 | 验证样本 | 指标 | 结果 |",
        "|---|---:|---|---:|",
    ]
    for result in results:
        score = result["scores"]
        if result["task"] == "A":
            metric = "Macro-F1 / Accuracy"
            value = f"{score['macro_f1']:.4f} / {score['accuracy']:.2%}"
        else:
            metric = f"RMSE / MAE（{result['unit']}）"
            value = f"{score['rmse']:.2f} / {score['mae']:.2f}"
        lines.append(f"| {result['task']} · {result['name']} | {result['validation_rows']:,} | {metric} | {value} |")
    lines += ["", "## 逐任务说明", ""]
    for result in results:
        lines += [
            f"### {result['task']} · {result['name']}",
            "",
            f"- 已加载检查点；原模型训练轮数：{result['epochs']}。",
            f"- 首条验证样本：实际 `{result['first_example']['actual']}`，预测 `{result['first_example']['predicted']}`。",
            f"- 检查点 SHA-256：`{result['checkpoint_sha256']}`。",
            "",
        ]
        if result["task"] == "A":
            lines += ["各类别 F1：", ""]
            for name, value in result["scores"]["f1_by_class"].items():
                lines.append(f"- {name}: {value:.3f}")
            lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:11]))
    print(f"报告：{output}")


if __name__ == "__main__":
    main()
