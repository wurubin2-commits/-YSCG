# 工业模型自进化系统 YSCG

> 第二届北京市大学生「人工智能+」创新大赛 ·「智训引擎」赛题参赛项目

赛题要求在官方 SDK 的轮次协议下，用有限轮次（默认 10 轮）让三套给定的工业基线模型持续改进，而平台只给公开主指标、每轮只允许一次训练机会。本仓库实现这条闭环：**读数据 → 诊断 → 出候选 → 续训 → 按公开分裁决 → 采纳或回滚**，并把每一轮的决策、证据与开销写入 `rounds.json`，可离线复盘。

设计的出发点是一个实测反例：在一个已接近收敛的父模型上按正常学习率再训一轮，B 的验证 RMSE 反而从 598.24 升到 626.22 分钟。所以这里的关键不是「更小的学习率」，而是**方向**——把测量证据压缩成一组有依据的候选（步长退火、结构配方、精英回放），让模型在候选之间裁决，而不是凭空发明超参。

## 一轮怎么决策

1. **读数据**：取当轮公开训练数据、历史与父检查点，算批次摘要；研究模式下额外算特征分布漂移与归因。
2. **出候选**：确定性变异算子按历史实测增益、失败切片、已试配方排序候选，并附上「为什么」。
3. **模型裁决**：一次请求发送摘要、父分、剩余预算和候选列表；模型返回候选编号（或完整训练对象），输出上限 512 token（与宿主上限取小），保留父模型同样登记有效回执。
4. **去重与续训**：按父检查点、训练投影、种子和实参在训练前查重；在独立 CPU 子进程从父检查点续训，每轮至多一个候选、默认 1 epoch、每任务 12 epoch 预算。
5. **裁决与护栏**：用当轮公开主指标比较；增益门（必须优于父分）、锚点门（不跌破已验证的最优）、回滚预演（父检查点必须可完整重载）三关通过才采用，否则保留父模型。

可选配方（类别加权、Huber、L2 正则、L2-SP、传感器遮蔽、B 残差目标、C 工况分组损失）只有在证据或归因触发时才进入候选；默认流程只做标准续训比较，保证可审计。

## 三个任务

| 任务 | 类型 | 主指标 | 单位 |
| --- | --- | --- | --- |
| A 气体识别 | 分类 | Macro-F1（越高越好） | — |
| B 轴承剩余寿命预测 | 回归 | RMSE（越低越好） | 分钟 |
| C 发动机寿命预测 | 回归 | RMSE（越低越好） | 循环 |

## 本地十轮验证（官方 SDK 模拟宿主）

三任务各十轮、每轮一次真实训练，`quota` 切分策略（`runs/evo10-10`）的公开主指标：

| 任务 | 基线 | 十轮后 | 绝对变化 | 相对增益 |
| --- | ---: | ---: | ---: | ---: |
| A Macro-F1 | 0.7140 | 0.7347 | +0.0207 | +2.89% |
| B RMSE（分钟） | 681.16 | 573.51 | −107.65 | +15.80% |
| C RMSE（循环） | 52.87 | 51.71 | −1.16 | +2.19% |
| 平均 | — | — | — | **+6.96%** |

真实平台 4B 网关驱动同构对照（`runs/live-4b-*`）为 A +3.04%、B −15.19%、C −0.88%，三任务平均 6.37%，与模拟宿主量级一致。以上均为本地单机测量；真实平台效果与 token 成本需以赛方验收为准。

## 目录

```
zhixun-harness/
├─ stage2_official/          平台提交入口与核心模块（模块地图见该目录 README）
│  ├─ contest_submission.py  赛方 SDK 的训练 / 预测回调
│  ├─ evolution.py           确定性变异算子（候选生成与排序）
│  ├─ continuation.py        续训与可选正则 / 结构配方
│  ├─ admission.py           增益 / 锚点 / 回滚三关准入护栏
│  ├─ policy.py normalize.py 历史压缩、去重、提案归一化
│  ├─ drift.py diagnostics.py attribution.py  漂移 / 诊断 / 归因（研究模式）
│  ├─ trainer.py tool_layer.py rehearse.py    训练子进程、读写工具、演练器
│  └─ tests/                 本包单元测试
├─ stage1_local/ stage2_local/  V1/V2 本地工具（被主线少量复用）
├─ round_review/             十轮复盘分析
├─ docs/                     交接文档、调研与实验记录
├─ a-气体识别/ b-轴承剩余寿命预测/ c-发动机寿命预测/   官方 CPU 模型包（本地输入，不入库）
└─ main.py                   本地命令入口
```

## 快速开始

```powershell
$env:PYTHONPATH = "$PWD\zhixun-harness;$PWD"
python -m pip install -r zhixun-harness/stage2_local/requirements.txt
Set-Location zhixun-harness
python -m pytest stage2_official/tests stage2_local/test_harness.py -q
python -B -m stage2_official.rehearse --name demo --task all --rounds 10
```

演练依赖两份**本地输入**：`stage2_official/sdk/`（赛方 SDK `contest.harness.workspace.v2`）与 `{a,b,c}*/cpu工业模型/`。恢复方法见 `zhixun-harness/docs/README.md`。

## 技术栈

- Python 3.12、NumPy 1.26.4
- Pydantic / pydantic-ai-slim（提案的 schema 校验）
- 赛方 SDK `contest.harness.workspace.v2`（1.0.x）
- Docker `linux/amd64`，基础镜像 `python:3.12-slim`
- SwanLab 0.10.3（本地离线记录，可选导出与回读核验）
- 平台模型网关（Qwen 27B 主验证，4B 同构对照）

## 仓库边界

赛方 SDK、三套模型包、公开数据、检查点、实验目录与镜像构建产物都在 `.gitignore` 中，保持为**本地输入 / 输出，不入库**。仓库只包含源码、文档与单元测试；复跑前需按 `zhixun-harness/docs/README.md` 放好 SDK 与模型包。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [`系统运行说明.md`](系统运行说明.md) | 构建、单轮执行链、环境变量、复现命令、安全边界 |
| [`项目说明文档.md`](项目说明文档.md) | 逐轮改动、验证结果与取舍 |
| [`zhixun-harness/README.md`](zhixun-harness/README.md) | 源码交接与实验对照（L2-SP、C 学习率等） |
| [`zhixun-harness/stage2_official/README.md`](zhixun-harness/stage2_official/README.md) | 模块职责与依赖方向 |
| [`交付材料/技术报告.pdf`](交付材料/技术报告.pdf) | 参赛技术报告 |

## 分支

`main` 为现状主线；`hxx` 为赛方官方起点；`ZMC`、`WRB`、`codex/harness-v3-handoff`、`harness-integration-compact` 为设计参考与历史交接分支，不参与当前开发。
