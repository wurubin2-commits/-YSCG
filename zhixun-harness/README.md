# 智训引擎 · 参赛 Harness 源码交接

本目录保留 `hxx`（`49b32ac`）的官方入口和构建方式，并吸收 `ZMC`（`1eb762a`）的实验记忆、受控续训和准入检查设计。整合改动集中在 `zhixun-harness/`；所有下列命令均从该目录运行。

| 路径 | 用途 |
| --- | --- |
| `stage2_official/contest_submission.py` | 赛方 SDK 的训练和预测回调。读取当轮数据、历史和父检查点，通过平台网关取得建议，比较候选与父模型后决定保留或采用。 |
| `stage2_official/trainer.py`、`continuation.py` | 在独立 CPU 子进程从指定父检查点续训。默认训练行为沿用赛方代码；新增可选的类别加权、Huber 损失、L2 正则。 |
| `stage2_official/policy.py`、`drift.py` | 压缩历史反馈，按父检查点、训练投影、种子和实际参数去重。`drift.py` 提供研究模式的分布变化摘要。 |
| `stage2_official/rehearse.py`、`verify_linux.sh` | 模拟平台逐轮发放数据与评估候选，验证 Docker 中的训练、状态交接和逐行预测。 |
| `stage2_official/build_context.py`、`Dockerfile` | 按清单收集镜像所需源码和依赖。支持从已校验的 Linux wheels 离线安装。 |
| `stage2_official/stage_linux_workspace.py` | 为 WSL 2 在 Ubuntu 自己的磁盘上准备验收目录，供 SDK 使用标准 Linux 文件权限。 |
| `stage2_official/export_swanlab.py` | 将本地模拟实验记录导出到 SwanLab 离线文件并回读核验。 |
| `stage1_local/`、`stage2_local/`、`run_initial_models.py` | 公开起点验证、早期训练与评估工具、策略约束和本地故障测试。 |

## 一轮怎样作出决定

1. 通过 SDK 读取当轮公开数据、历史和父检查点。将平台历史和保存的实测记录按轮合并，优先选同父模型、同一批次的结果，并保留有代表性的成功和失败结果，最多三条。
2. 一次模型请求发送批次摘要、父模型分数、有效参数、相关结果和剩余预算。回复为简短 JSON，最多 192 个输出 Token，并遵守宿主提供的更低上限。保留父模型的轮次也按 SDK 要求登记有效回执。
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

三组对照使用一致的起始检查点、逐批训练数据、公开验证集和模拟建议。耗时为各任务从读取数据到完成最终预测的本机单次测量之和。原始记录位于 `stage2_official/runs/cost-*`，核对后的汇总位于 `evidence/cost-benchmark.json`。SDK、Harness 和新版去重与历史筛选测试合计 25 项通过。

队友审阅时以本目录内的源码和本说明为交接范围。赛方 SDK、公开数据、模型检查点、实验目录和镜像都保存在本地。真实网关与平台试运行仍需团队确认后通过赛方平台验收。
