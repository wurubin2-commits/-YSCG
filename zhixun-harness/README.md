# 智训引擎 · 参赛 Harness 源码交接

本目录是 `hxx` 分支上参赛 Harness 的独立交接。所有命令从 `zhixun-harness/` 运行。现有 YSCG 自进化原型仍位于仓库的 `src/yscg/`，可在后续合并时独立评审。

| 路径 | 用途 |
| --- | --- |
| `stage2_official/contest_submission.py` | 赛方 SDK 的训练和预测回调。读取当轮数据、历史和父检查点，通过平台网关取得建议，比较候选与父模型后决定保留或采用。 |
| `stage2_official/trainer.py` | 在独立 CPU 子进程中调用相应的赛方基线续训。 |
| `stage2_official/rehearse.py`、`verify_linux.sh` | 模拟平台逐轮发放数据与评估候选，验证 Docker 中的训练、状态交接和逐行预测。 |
| `stage2_official/build_context.py`、`Dockerfile` | 按清单收集镜像所需源码和依赖。支持从已校验的 Linux wheels 离线安装。 |
| `stage2_official/stage_linux_workspace.py` | 为 WSL 2 在 Ubuntu 自己的磁盘上准备验收目录，供 SDK 使用标准 Linux 文件权限。 |
| `stage2_official/export_swanlab.py` | 将本地模拟实验记录导出到 SwanLab 离线文件并回读核验。 |
| `stage1_local/`、`stage2_local/`、`run_initial_models.py` | 公开起点验证、早期训练与评估工具、策略约束和本地故障测试。 |

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
python -B -m unittest stage2_local.test_harness -v
python -B -m stage2_official.rehearse --name team-local-a --task A --rounds 1
```

完整验收使用 Linux/amd64、本机 Docker 服务及支持 NumPy 1.26.4 的宿主 Python。WSL 2 下将项目和运行输入放在 Ubuntu 的原生目录中。可选的 `ZHIXUN_WHEELHOUSE` 指定已校验的 Linux wheels，`ZHIXUN_BASE_IMAGE` 指定固定摘要的 Python 基础镜像。

```bash
bash stage2_official/verify_linux.sh team-linux-01 zhixun-harness:team-linux-01
```

已在 Ubuntu 24.04 的 Linux/amd64 Docker 上按 4 核、8 GiB、无 GPU、断网及只读根文件系统验证。三任务各 10 轮共完成 30 个候选、30 个模拟网关回执和 6 条最终预测。SDK 自带的 15 项离线测试通过。

| 公开验证指标 | 赛方起点 | 本地选中的模型 |
| --- | ---: | ---: |
| A 气体识别 Macro-F1（越高越好） | 0.7140 | 0.7368 |
| B 轴承寿命 RMSE（分钟，越低越好） | 681.16 | 570.76 |
| C 发动机寿命 RMSE（循环，越低越好） | 52.87 | 51.83 |

真实模型网关、赛方试运行和隐藏评测由赛方平台验收。
