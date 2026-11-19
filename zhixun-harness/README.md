# 智训引擎 · 参赛 Harness 源码交接

本目录保留 `hxx`（`49b32ac`）的官方入口和构建方式，并吸收 `ZMC`（`1eb762a`）的实验记忆、受控续训和准入检查设计。整合改动集中在 `zhixun-harness/`。下列命令均从该目录运行。

| 路径 | 用途 |
| --- | --- |
| `stage2_official/contest_submission.py` | 赛方 SDK 的训练和预测回调。读取当轮数据、历史和父检查点，通过平台网关取得建议，比较候选与父模型后决定保留或采用。 |
| `stage2_official/trainer.py`、`continuation.py` | 在独立 CPU 子进程从指定父检查点续训。默认训练行为沿用赛方代码；新增可选的类别加权、Huber 损失、L2 正则。 |
| `stage2_official/policy.py`、`drift.py` | 压缩历史反馈，按父检查点、训练投影、种子和实际参数去重。`drift.py` 提供研究模式的分布变化摘要。 |
| `stage2_official/rehearse.py`、`verify_linux.sh` | 模拟平台逐轮发放数据与评估候选，验证 Docker 中的训练、状态交接和逐行预测。 |
| `stage2_official/build_context.py`、`Dockerfile` | 按清单收集镜像所需源码和依赖。支持从已校验的 Linux wheels 离线安装。 |
| `stage2_official/stage_linux_workspace.py` | 为 WSL 2 在 Ubuntu 自己的磁盘上准备验收目录，供 SDK 使用标准 Linux 文件权限。 |
| `stage2_official/export_swanlab.py` | 将本地模拟实验记录导出到 SwanLab 离线文件并回读核验。 |
| `main.py`、`round_review/` | 本地十轮复盘入口及分析模块，读取已有回执，验证同条件复跑并导出图表。 |
| `stage1_local/`、`stage2_local/`、`tools/run_initial_models.py` | 公开起点验证、早期训练与评估工具、策略约束和本地故障测试。 |
| `stage2_official/tests/` | 本包单元测试；模块职责与依赖方向见 `stage2_official/README.md`。 |

## 一轮怎样作出决定

1. 通过 SDK 读取当轮公开数据、历史和父检查点。将平台历史和保存的实测记录按轮合并，优先选同父模型、同一批次的结果，并保留有代表性的成功和失败结果，最多三条。
2. 一次模型请求发送批次摘要、父模型分数、有效参数、相关结果和剩余预算。回复为简短 JSON，最多 512 个输出 Token（自设上限，实际取宿主 `rnd.max_output_tokens` 与它的较小值），保留父模型的轮次也按 SDK 要求登记有效回执。
3. 提议训练时，用父检查点、训练投影摘要、种子和实际参数在训练前检查重复。曾完成的训练在参数相同但理由变动时直接跳过。换批次、换父模型或实际参数变化时可以再试。没有生成有效检查点的失败可重试。每轮至多一个候选，最多两个 epoch，每任务策略预算为 12 epoch。
4. 执行赛方标准续训并比较当轮公开主指标。A 使用 Macro-F1，B/C 使用各自单位的 RMSE。失败、分数不可比、打平或退步时保留父模型。完整决策、回执编号和用量保存在轮次状态中，验证标签只用于测量。

通过 `--scenario recipes` 可以离线试验漂移分析、类别加权、Huber 损失、L2 正则和公开验证子集保护。子集保护在 A 比较类别召回率，在 B/C 比较低寿命四分位的 MAE。先前对照中，这组功能降低了 B 的低寿命误差，也提高了总体 RMSE。默认流程采用主指标比较。这些扩展设计参考了 [ZMC 的续训方法](https://github.com/wurubin2-commits/-YSCG/blob/1eb762ada375b013631f342b2b2c1dc819c90183/baselines/diverse_trainer.py) 和 [准入对照](https://github.com/wurubin2-commits/-YSCG/blob/1eb762ada375b013631f342b2b2c1dc819c90183/docs/准入决策与对照试验设计.md)。

## 从赛方下载的运行输入

使用赛方 SDK 1.0.2（`contest.harness.workspace.v2`）。把解压后的 `contest_harness/`、`template/harness.py`、`pyproject.toml` 和 `README.md` 按原结构放到 `stage2_official/sdk/`。队友可直接使用各自从赛方取得的副本。

将三套赛方 CPU 工业模型包放到以下位置，保留包内的 `baselines/`、`training.json`、`checkpoint.npz` 和 `public/`。B 使用标签和评估单位为分钟的轴承寿命回归包。

```text
a-气体识别/cpu工业模型/
b-轴承剩余寿命预测/cpu工业模型/
c-发动机寿命预测/cpu工业模型/
```

赛方 SDK、原始训练包、数据、检查点、实验记录和镜像归档在本目录的 `.gitignore` 中保持为本地输入或输出。调用真实模型所需的密钥由平台设置。

## 十轮回执复盘

从 `zhixun-harness/` 运行以下命令。绘图使用本机的 Matplotlib，属于本地分析依赖。两份回执目录都需要 `summary.json` 及三个任务的 `rounds.json`，来自已有的同条件十轮正常分批复跑。

```bash
python -B main.py review-rounds --reference stage2_official/runs/rerun-nov17-verify01-hxx-normal --current stage2_official/runs/rerun-nov17-verify01-compact-normal --output evidence/phase1-round-review
```

输出 `report.md`、`rounds.csv` 与 `rounds.png`，都存放在本地证据目录。入口检查两次复跑的同轮输入与输出是否一致，供核验当前性能基准。后续候选策略可能产生不同结果，需要按它自己的对照计划评估。正式平台继续从官方 SDK 入口启动。

## B 轴承任务的 L2-SP 对照

L2-SP 参考 Transfer Learning Library 的起点正则化方法。用 NumPy 对两层权重偏离当轮父模型的幅度加惩罚，继续使用赛方检查点的 Adam 状态、标准化和预测接口。B 支持实验开关 `ZHIXUN_ENABLE_L2SP=1`，并记录建议的强度，供验证与训练去重。正式默认策略为标准续训。 [参考实现](https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/regularization/delta.py) · [原论文](https://arxiv.org/abs/1802.01483)

本地对照使用官方 SDK 模拟宿主、同一公开验证投影和相同训练预算。第二条训练命令采用预先设定的强度 `1.0`。每次使用新的演练名称。

```bash
python -B -m unittest stage2_official.test_policy stage2_official.test_l2sp -v
python -B -m stage2_official.rehearse --name phase2-b-standard --task B --rounds 10
python -B -m stage2_official.rehearse --name phase2-b-l2sp-strength1 --task B --rounds 10 --scenario l2sp
```

本机 Windows CPU 的单次配对回放如下。低剩余寿命切片为公开验证集真实寿命最低四分位，共 420 条。

| B 公开验证 | 标准续训 | L2-SP |
| --- | ---: | ---: |
| 最终 RMSE（分钟） | 570.76 | 572.26 |
| 最终 MAE（分钟） | 463.95 | 466.51 |
| 低剩余寿命 MAE（分钟） | 435.95 | 424.92 |
| 最差候选相对当轮父模型的 RMSE 增量（分钟） | 123.49 | 117.14 |
| 采用候选 | 3/10 | 3/10 |
| 墙钟时间（秒） | 26.17 | 26.23 |

两组从同一检查点出发，每轮从各自上轮选中的模型续训。L2-SP 降低了低寿命子集误差和最差候选退步幅度，最终整体 RMSE 与 MAE 均上升，因此保留为 B 的实验选项。本次只测了一个强度和一个公开批次顺序。

## C 发动机任务的学习率对照

实验开关 `ZHIXUN_C_HALF_RATE=1` 会将 C 的标准续训建议学习率减半，最低保留 `0.00005`。保持大模型给出的训练或保留动作、训练轮数和批大小，按实际生效学习率去重，并在回执中保留发生调整的原建议值。可在本地执行两次十轮演练，分别比较标准流程和开启开关的结果。

```bash
python -B -m stage2_official.rehearse --name c-reference --task C --rounds 10
python -B -m stage2_official.rehearse --name c-half-rate --task C --rounds 10 --scenario c_half_rate
```

官方 SDK 本地模拟宿主的十轮对照中，正常批次顺序的 RMSE 从 51.8281 降至 51.7725 循环，MAE 从 38.02 降至 37.73 循环，两组均训练和登记十个候选。逐轮核对了数据摘要、随机种子和训练预算。将整台发动机保留到各折结束才评分时，三折 RMSE 均下降。更换公开训练批次顺序后，倒序回放的 RMSE 下降，轮转回放的 RMSE 上升。开关默认关闭，适用于本地试验。

## 运行和验收

用 Python 3.12 安装 `stage2_local/requirements.txt`，然后执行以下命令。单轮演练使用赛方 A 任务的公开数据和检查点。

```bash
python -m pip install -r stage2_local/requirements.txt
python -B -m unittest stage2_local.test_harness stage2_official.test_policy -v
python -B -m stage2_official.rehearse --name team-local-a --task A --rounds 1
python -B -m stage2_official.rehearse --name team-abc10 --rounds 10
python -B -m stage2_official.export_swanlab stage2_official/runs/team-abc10 stage2_official/runs/team-swanlab
```

完整验收使用 Linux/amd64、本机 Docker 服务及支持 NumPy 1.26.4 的宿主 Python。WSL 2 下将项目和运行输入放在 Ubuntu 的原生目录中。可选的 `ZHIXUN_WHEELHOUSE` 指定已校验的 Linux wheels，`ZHIXUN_BASE_IMAGE` 指定固定摘要的 Python 基础镜像。

```bash
bash stage2_official/verify_linux.sh team-linux-01 zhixun-harness:team-linux-01
```

每次演练写出各任务的公开分数、调用次数、输入正文 UTF-8 字节数、单次请求的输出上限、训练次数、候选提交次数、拦截的重复次数和端到端耗时。输入字节数只计 Harness 发出的消息正文。真实网关使用的输入和输出 Token 以逐轮回执的 `usage` 字段统计。

普通演练每任务使用十个不同训练批次。`--scenario repeat_pairs --rounds 6` 使用每任务三个批次，每批重复一次。重复场景将候选分数模拟为平分以固定父模型，便于检查前置去重带来的成本差异。普通演练中的公开验证投影也用于每轮选择模型，平台实际效果和 Token 成本需要用真实网关验收。

普通分批在 4 核、8 GiB、无 GPU、断外网的 Linux 容器中完成三任务各十轮和六条预测。旧 `hxx` 和精简版的公开主指标一致，A Macro-F1 为 0.7368，B RMSE 为 570.76 分钟，C RMSE 为 51.83 循环。旧版与精简版均完成 30 次训练和 30 次模型调用，Harness 发送的正文分别为 56,687 和 27,365 字节，单次输出上限分别为 512 和 192 Token。

| 同资源对照 | 旧 `hxx` | 旧 `hxx` 只加去重 | 去重与精简历史 |
| --- | ---: | ---: | ---: |
| 普通分批正文（字节） | 56,687 | 56,687 | 27,365 |
| 普通分批端到端耗时（秒） | 121.82 | 116.32 | 122.07 |
| 重复批次训练次数 | 18 | 9 | 9 |
| 重复批次候选提交次数 | 18 | 9 | 9 |
| 重复批次模型回执次数 | 18 | 18 | 18 |
| 重复批次端到端耗时（秒） | 88.10 | 64.77 | 68.09 |

三组对照使用一致的起始检查点、逐批训练数据、公开验证集和模拟建议。耗时为各任务从读取数据到完成最终预测的本机单次测量之和。原始记录是本地实验产物，未随仓库发布；核对后的汇总数值见项目说明文档“成本基准”一节。SDK、Harness 和新版去重与历史筛选测试合计 25 项通过。

队友审阅时以本目录内的源码和本说明为交接范围。赛方 SDK、公开数据、模型检查点、实验目录和镜像都保存在本地。真实网关与平台试运行仍需团队确认后通过赛方平台验收。
