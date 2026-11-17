# Harness 提交骨架（contest_harness 1.0.2 / `contest.harness.workspace.v2`）

本目录是**符合官方协议的最小可提交骨架**：先保证能过 `describe`、能训练一轮、能提交候选、
能保存状态、能 `finish`、能逐行预测；YSCG 的漂移自适应逻辑后续替换即可。

## 文件

| 文件 | 作用 |
|---|---|
| `contest_submission.py` | 选手模块：`TASKS`、`describe()`、`train(rnd)`、`predict(sample, model_dir)` |
| `README.md` | 本说明 |

官方 `template/harness.py`（需从 SDK 包获取）放到镜像内 `/app/harness.py`。

## 镜像内放置

```dockerfile
# 示例（实际以官方 SDK 包为准）
COPY submission/contest_submission.py /app/contest_submission.py
# 官方 harness.py 来自 SDK 的 template/harness.py
COPY sdk/template/harness.py /app/harness.py
```

若不用默认模块名，设置环境变量 `CONTEST_SUBMISSION`。

## 自检

```sh
python3 /app/harness.py describe   # 必须输出一行合法 JSON，tasks 与实际支持一致
python3 /app/harness.py train      # 平台会写入 start 帧
python3 /app/harness.py predict    # 预测进程
```

## 协议硬约束 checklist

- [ ] `describe` 输出合法 JSON，`tasks` 只含镜像真实支持的任务
- [ ] 每轮至少一次真实模型调用，`finish` 的 `model_call_ids` 非空
- [ ] 候选目录 ≤ 64 个普通文件、合计 ≤ 64MiB
- [ ] `hypothesis` / `change_summary` 非空且 ≤ 4000 字节；`configuration` 序列化 ≤ 64KiB
- [ ] stdout 只写协议 JSONL；日志一律走 stderr
- [ ] `predict` 一行进一行出、每行 flush；分类 label 唯一 / 最高分唯一 / score∈[0,1]；回归 value 有限非负
- [ ] 不写 `/app`、不改 `/previous`；状态只写 `state_workspace` 返回的目录

## 本地演练

```sh
python -m pytest tests/test_submission_skeleton.py -q
```

用 `FakeRnd` 模拟平台，覆盖 `train`（模型调用 / 候选提交 / 状态保存 / finish）与 `predict`（分类+回归）。

## 相对参考骨架的 3 处协议修正

1. **`predict` 返回内层 payload**：分类返回 `[{"label":...,"score":...}]`、回归返回 `{"value":...}`；
   由 harness 包成 `{"outputs": <返回值>}`。若返回完整 `{"outputs": ...}` 会变成双层嵌套。
   （依据官方文档的 `predict` 示例；如你手上的 `harness.py` 逻辑不同，以 harness 源码为准。）
2. **`save_state` 传 `state_workspace()` 返回的对象**，而不是 `str(...)`。
3. **新增 `_ensure_one_model_call`**：`call_ids` 为空时自动重试一次，避免 `finish` 因空 `model_call_ids` 被判无效。

## YSCG 替换点

| 位置 | 替换成 |
|---|---|
| `_load_parent_state` | 从父产物恢复锚点 floor、保护性记忆、策略参数 |
| `_train_simple` | YSCG 增量续训 / 漂移自适应 |
| `_call_llm_once` | 真实假设生成（必须保留至少一次 `rnd.model()`） |
| `predict` | YSCG 推理；预测期只能用无监督信号，不依赖隐藏标签 |
| `finish` 前决策 | 基于 `submit_candidate` 返回测量 + `rnd.history()` 选新候选/父模型/历史产物 |
| `state_payload` | 写全 `/state` 跨轮状态（锚点/记忆/策略） |

## 本地演练用环境变量（仅本机，容器内不需要）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CONTEST_WORKSPACE` | `/workspace` | 替换候选目录/状态目录根 |
| `CONTEST_DATA_DIR` | `/data` | 替换 JSONL 数据目录 |
| `CONTEST_MODEL_DIR` | `/model` | 替换预测时的产物挂载目录 |

## 官方 CPU 基线模型（A / B / C）支持

`submission/` 内已 vendor 官方基线的加载器：

| 文件 | 作用 |
|---|---|
| `baseline_model.py` | 官方 NumPy MLP 的 `Model`（load / predict / validate） |
| `baseline_data.py` | 官方 `flatten` / `read_dataset`（特征路径展开） |

`predict` 会**先检测官方基线 checkpoint**（npz 含 `metadata` + `w1`）：
- 是 → 用官方 `Model.load()` + `flatten(inputs)` 推理，按任务返回 `[{"label","score"}]` 或 `{"value"}`；
- 否 → 回退到本骨架自己的 `checkpoint.npz` 格式。

三个官方基线（已解压在 `baselines/`，仅本机，不入库）：

| 数据集 | 任务 | 基线指标（public validation） |
|---|---|---|
| A（气体 6 类） | classification | accuracy 0.8786 / macro-F1 0.7140（n=1244） |
| B（轴承 RUL，分钟） | regression | RMSE 681.16 / MAE 563.11（n=1679） |
| C（CMAPSS RUL，循环） | regression | RMSE 52.87 / MAE 38.90（n=23049） |

复现：

```sh
python baselines/eval_baselines.py          # 直接调官方 Model.predict
python baselines/eval_harness_baselines.py  # 走 harness 的 predict（逐行）
python -m pytest tests/test_harness_baseline.py -q
```

### 官方基线包自带的训练/续训（可选）

官方 `TRAINING_GUIDE.md` / `START.md` 里的相对路径以**单个基线包根目录**为基准
（`baselines/a`、`baselines/b`、`baselines/c`）。在仓库根目录直接执行
`pip install -r baselines/requirements.txt` 会报 `No such file or directory`，
实际文件在 `baselines/<a|b|c>/baselines/requirements.txt`。正确用法是先 `cd` 进包根目录：

```powershell
cd D:\-YSCG\baselines\a        # b / c 同理，--profile 换成 B / C
D:\-YSCG\.venv\Scripts\python.exe -m baselines.train --profile A `
  --train public/train.jsonl --validation public/validation.jsonl `
  --from-baseline checkpoint.npz --epochs 5 --output round-1 --stage-id round-1
```

官方 `requirements.txt` 只固定 `numpy==1.26.4`，而主环境是 numpy 2.x（pandas 3.1 / sklearn 1.9
的二进制扩展按 numpy 2 ABI 编译），**不要把 numpy 降级装进主 venv**，否则会破坏主项目依赖。
实测官方训练代码在 numpy 2.4.6 下可正常运行：`tests/test_harness_baseline.py` 通过，
A/C 各做了 1 epoch 续训 smoke（macro-F1 0.7140→0.7257、RMSE 52.87→52.35）。若确实要复现
官方锁定版本，另建独立 venv：`python -m venv baselines\venv-baseline`。

> 注：`train(rnd)` 的 `rnd.parent` 若为官方基线格式，当前 `_load_parent_state` 仍只读本骨架格式；
> **从官方基线续训**是下一步（可复用官方 `baselines/train.py --from-baseline`）。
