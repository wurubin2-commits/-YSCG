# 工业模型自进化系统 YSCG

第二届北京市大学生「人工智能+」创新大赛「智训引擎」赛题参赛项目。

当前主线是可提交的 `zhixun-harness/`：在官方 SDK 的轮次协议下，逐轮读取公开数据、续训候选、
按公开主指标决定采用或保留，并把每轮证据写进 `rounds.json`。

## 目录

| 路径 | 用途 |
| --- | --- |
| `zhixun-harness/stage2_official/` | 平台提交入口与核心模块，模块地图见该目录的 [README](zhixun-harness/stage2_official/README.md) |
| `zhixun-harness/stage2_official/tests/` | 该包单元测试 |
| `zhixun-harness/stage1_local/`、`stage2_local/` | V1/V2 历史工具：本地训练、规划、追踪 |
| `zhixun-harness/round_review/` | 十轮演练对比报告 |
| `zhixun-harness/tools/` | 独立小工具 |
| `zhixun-harness/docs/` | 交接文档、调研与实验记录 |
| `models/` | 官方 CPU 模型包（本地输入，不入库） |
| `data/`、`artifacts/`、`swanlog/` | 数据、模型产物、SwanLab 本地记录，均不入库 |

## 快速开始

```powershell
$env:PYTHONPATH = "$PWD\zhixun-harness;$PWD"
Set-Location zhixun-harness
python -m pytest stage2_official/tests stage2_local/test_harness.py -q
python -B -m stage2_official.rehearse --name demo --task all --rounds 10
```

演练依赖两份本地输入：`stage2_official/sdk/`（官方 SDK 1.0.2）与
`{a-气体识别,b-轴承剩余寿命预测,c-发动机寿命预测}/cpu工业模型/`。恢复方法见
`zhixun-harness/docs/README.md`。

## 分支

| 分支 | 内容 |
| --- | --- |
| `main` | 当前主线：V3 harness + 准入护栏 + 归一化兜底 + 协变量诊断 |
| `WRB` | M0 闭环快照，仅存档，不参与开发 |
| `archive/main-m0` / `archive/wrb-implementation` | 迁移前的备份标签 |

## 提交镜像

```bash
bash zhixun-harness/stage2_official/verify_linux.sh <run-name> <image-tag>
```
