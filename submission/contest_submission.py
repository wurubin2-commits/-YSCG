"""YSCG 工业模型自进化系统 · contest.harness.workspace.v2 选手模块。

每轮闭环（无人干预）：
  读平台历史 → 父检查点校验加载 → 公开投影漂移/残差诊断
  → 规则提案 + PyTRiO 顾问建议（白名单夹取）→ 续训实验
  → 验证对照准入闸门 → submit_candidate / 保留父模型 → save_state → finish

安全护栏：
- 只从 rnd.parent.path() 声明并核对摘要的父检查点续训，不改结构/任务/种子；
- LLM 建议只落在白名单超参网格内，异常一律降级为规则提案；
- 任何实验失败降级为保守续训或保留父模型，绝不崩溃退出；
- 密钥/端点全部由平台持有，本模块只经 rnd.model() 访问大模型。
"""
import json
import os
import sys
import time
from pathlib import Path

from contest_harness import Round

from baselines.data import read_dataset
from baselines.model import Model
from baselines.train import train_run, metrics
from diagnostics import diagnose
from policy import clamp_config, rule_proposal, should_keep_parent

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
ROOT = Path(__file__).resolve().parent


def _log(message):
    print(message, file=sys.stderr, flush=True)


def _work_root():
    return Path(os.environ.get("CONTEST_WORK", "/workspace"))


def _load_state(rnd):
    """读取上一轮保存的原生状态（首轮可能不存在）。"""
    if not rnd.state_path:
        return {}
    path = Path(rnd.state_path) / "memory.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _advisor_context(rnd, diagnostics, history, proposal, state):
    """给 PyTRiO 顾问的紧凑上下文：只放本轮数据、代码状态和问题。"""
    recent = []
    for record in history[-5:]:
        recent.append({k: record.get(k) for k in ("id", "round_index", "hypothesis", "change_summary") if k in record})
    return {
        "dataset_key": rnd.dataset_key,
        "round_index": rnd.round_index,
        "task": rnd.task.get("task"),
        "metric": rnd.task.get("metric"),
        "diagnostics": diagnostics,
        "rule_proposal": proposal,
        "recent_history": recent,
        "last_round_memory": {k: state.get(k) for k in ("last_decision", "last_metrics") if k in state},
        "question": (
            "你是工业模型续训策略顾问。基于以上诊断与历史，在白名单内给出本轮续训超参。"
            '只输出 JSON：{"learning_rate": 数值, "batch_size": 整数, "epochs": 整数, "reason": "一句话理由"}。'
            "learning_rate ∈ {0.0005,0.001,0.002,0.003,0.005}，batch_size ∈ {64,128,256}，epochs ∈ [1,30]。"
            "模型结构、任务与随机种子不可更改。若认为不应继续训练，epochs 填 0。"
        ),
    }


def _consult_model(rnd, context):
    """真实 PyTRiO 调用（每轮至少一次）。失败返回 None，由规则兜底。"""
    try:
        receipt = rnd.model(
            [
                {"role": "system", "content": "你是受控自进化系统的策略顾问，只输出 JSON，不输出其他文本。"},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            max_tokens=min(512, rnd.max_output_tokens),
            temperature=0,
        )
    except Exception as error:  # 平台调用失败：降级规则提案，但本轮已有调用尝试记录
        _log(f"advisor call failed: {error}")
        return None, None
    text = receipt.text.strip()
    suggestion = None
    try:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            suggestion = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        suggestion = None
    return receipt, suggestion


def train(rnd: Round):
    work = _work_root()
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = _load_state(rnd)
    history = rnd.history()

    # 1) 父检查点：只经 SDK 声明路径打开并核对摘要
    parent_ckpt = rnd.parent.path("checkpoint.npz")
    parent_model = Model.load(parent_ckpt)
    task = rnd.task["task"]
    if parent_model.meta["task"] != task:
        raise ValueError("parent checkpoint task differs from frozen contract")
    classes = parent_model.meta["classes"]

    # 2) 公开投影（只读授权数据）；B 的 RUL 配置带 target_unit，需显式 JSON 路径
    train_data = read_dataset(rnd.data.train_path, task, classes)
    val_data = read_dataset(rnd.data.validation_path, task, classes, train_data["paths"])

    # 3) 诊断：协变量漂移 + 父模型残差
    diag = diagnose(parent_model, train_data, val_data)
    _log(f"round {rnd.round_index} diagnostics: {json.dumps(diag, ensure_ascii=False)}")

    # 4) 规则提案 → LLM 顾问 → 白名单夹取
    proposal = rule_proposal(rnd.dataset_key, diag, history, rnd.round_index)
    receipt, suggestion = _consult_model(rnd, _advisor_context(rnd, diag, history, proposal, state))
    advisor_reason = ""
    config = proposal
    if isinstance(suggestion, dict):
        clamped = clamp_config(suggestion, rnd.dataset_key)
        if clamped is not None:
            config = clamped
            advisor_reason = str(suggestion.get("reason", ""))[:500]
    keep_parent_only = isinstance(suggestion, dict) and suggestion.get("epochs") == 0

    # 保证每轮至少一次真实模型调用（顾问失败时补一次决策确认调用）
    if receipt is None:
        receipt = rnd.model(
            [{"role": "user", "content": json.dumps({
                "dataset_key": rnd.dataset_key, "round_index": rnd.round_index,
                "diagnostics": diag, "plan": config,
                "question": "顾问调用失败，规则引擎将按上述保守计划续训。请用一句话确认或提醒风险。",
            }, ensure_ascii=False)}],
            max_tokens=min(128, rnd.max_output_tokens),
            temperature=0,
        )

    hypothesis = (
        f"第{rnd.round_index}轮：诊断 psi_mean={diag['drift']['psi_mean']:.3f}，"
        f"父模型验证 {json.dumps(diag['parent_on_validation'], ensure_ascii=False)}；"
        f"假设按 lr={config['learning_rate']}, bs={config['batch_size']}, epochs={config['epochs']} "
        "续训可在不破坏父状态的前提下改善主指标。"
    )

    submitted = None
    train_report = None
    selected_id = rnd.parent.artifact_id
    decision = "keep_parent"
    if not keep_parent_only:
        # 5) 续训实验：官方 train_run，从父检查点恢复权重与 Adam 状态
        candidate = work / f"candidate-r{rnd.round_index}"
        try:
            train_report = train_run(
                str(ROOT / "baselines" / "configs" / f"{rnd.dataset_key}.json"),
                rnd.data.train_path, rnd.data.validation_path, candidate,
                resume=str(parent_ckpt),
                epochs=config["epochs"], learning_rate=config["learning_rate"],
                batch_size=config["batch_size"], stage_id=f"round-{rnd.round_index}",
            )
        except Exception as error:
            _log(f"training failed, keep parent: {error}")
            train_report = None
        if train_report is not None:
            before = train_report["validation_before"]
            after = train_report["validation_after"]
            if should_keep_parent(rnd.dataset_key, before, after):
                decision = "keep_parent_gate"
                hypothesis += " 实测候选劣于父模型，准入闸门拒绝上线。"
            else:
                change = (
                    f"从父检查点 {rnd.parent.checkpoint_sha256[:19]}… 续训 {config['epochs']} epochs"
                    f"（lr={config['learning_rate']}, batch={config['batch_size']}），"
                    f"步数 {train_report['initial_steps']}→{train_report['final_steps']}，"
                    f"验证 {json.dumps(before, ensure_ascii=False)} → {json.dumps(after, ensure_ascii=False)}。"
                )
                submitted = rnd.submit_candidate(
                    candidate,
                    hypothesis=hypothesis,
                    change_summary=change,
                    configuration={**config, "advisor_reason": advisor_reason,
                                   "diagnostics": diag, "training_seconds": train_report["training_seconds"]},
                    seed=parent_model.meta["seed"],
                    checkpoint_paths=["checkpoint.npz"],
                    code_paths=["training.json", "history.jsonl"],
                    predict_argv=PREDICT_ARGV,
                )
                selected_id = submitted["artifact_id"]
                decision = "promote_candidate"

    # 6) 轮间记忆（策略层状态跨轮传递）
    memory = {
        "round_index": rnd.round_index,
        "last_decision": decision,
        "last_config": config,
        "last_metrics": (train_report or {}).get("validation_after") or diag["parent_on_validation"],
        "advisor_reason": advisor_reason,
        "decisions": (state.get("decisions") or [])[-19:] + [{"round": rnd.round_index, "decision": decision}],
    }
    state_dir = rnd.state_workspace(str(work / f"state-r{rnd.round_index}"))
    (state_dir / "memory.json").write_text(json.dumps(memory, ensure_ascii=False), encoding="utf-8")
    rnd.save_state(state_dir)

    # 7) 结束本轮：引用真实读过的历史与实际模型调用号
    metric_summary = json.dumps(memory["last_metrics"], ensure_ascii=False)
    reason = (
        f"决策 {decision}：依据本轮诊断与验证对照（{metric_summary}），"
        f"模型调用 {receipt.call_id} 的建议{'已采纳' if advisor_reason else '未采纳/无有效建议'}。"
        f"{'候选已通过准入闸门并提交。' if decision == 'promote_candidate' else '保留父模型以防劣化。'}"
    )
    change_summary = (
        f"续训 {config['epochs']} epochs 并上线候选。" if decision == "promote_candidate"
        else "本轮未上线新权重：保留父检查点（闸门拒绝或顾问建议暂缓）。"
    )
    rnd.finish(
        selected_artifact_id=selected_id,
        hypothesis=hypothesis,
        change_summary=change_summary,
        reason=reason,
        configuration={**config, "decision": decision, "wall_seconds_used": round(time.monotonic() - started, 2)},
        seed=parent_model.meta["seed"],
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )


_MODEL_CACHE = {}


def predict(sample, model_dir):
    """行式预测：模型只加载一次；检查点路径来自平台环境变量。"""
    ckpt = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
    model = _MODEL_CACHE.get(ckpt)
    if model is None:
        model = Model.load(ckpt)
        _MODEL_CACHE[ckpt] = model
    from baselines.data import flatten
    paths, values = flatten(sample.inputs)
    if paths != model.meta["paths"]:
        raise ValueError("prediction feature schema differs from checkpoint")
    import numpy as np
    output = model.predict(np.asarray([values], dtype=np.float64))[0]
    if sample.task == "regression":
        return {"value": float(output)}
    scores = [float(s) for s in output]
    top = max(range(len(scores)), key=lambda i: scores[i])
    # 保证最高分唯一（协议硬性要求）：先把与最高分并列的其它类压低，再裁剪到 [0,1]
    for i in range(len(scores)):
        if i != top and scores[i] >= scores[top]:
            scores[i] = scores[top] - 1e-6
    scores = [max(0.0, min(1.0, s)) for s in scores]
    if sum(s == scores[top] for s in scores) != 1:
        scores = [0.0] * len(scores)
        scores[top] = 1.0
    return [{"label": label, "score": s} for label, s in zip(model.meta["classes"], scores)]