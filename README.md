# 工业模型自进化系统 YSCG

「智训引擎」赛题参赛项目 — 工业制造场景下的模型训练与超参数优化。

目标：构建一套在漂移数据流上**无人值守**运行的模型自进化系统，闭环完成
`漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚`，实现
「模型层自迭代 + 策略层自改进」的受控 RSI（Recursive Self-Improvement）。

## 系统架构

```
 数据流 ──▶ ① 漂移监测(PSI/置信度) ──▶ ② 诊断归因(逐特征贡献) ──▶ ③ 策略决策
 (隐藏测试流)                                                   规则引擎兜底 + LLM 顾问
                                                                    │
          ⑤ champion/challenger 对比 ◀── ④ 实验执行(SwanLab 记录) ◀─┘
          胜→上线 / 负→回滚              轻量适配 / 全量重训 / HPO
```

全部变更受护栏约束：标注配额、重训次数、HPO 试验数上限、策略白名单。

## 快速开始

```bash
pip install -e .
python experiments/run_simulation.py --model lgbm --batches 90
# 或深度学习路线
python experiments/run_simulation.py --model mlp --batches 90
```

仿真场景与对照臂：

```bash
# persistent：漂移持续到流结束（考验自适应）；transient：漂移后恢复（考验回滚保护）
python experiments/run_simulation.py --scenario persistent
# 静态基线对照（部署后不更新的模型），用于计算自进化的性能增益
python experiments/run_simulation.py --static
# 标注预算、SwanLab 模式等可调
python experiments/run_simulation.py --budget 600 --mode local
```

实验记录写入 `swanlog/`（本地模式）；接入 SwanLab 云端账号后自动同步。

```bash
pytest   # 冒烟测试
```

## 目录结构

```
src/yscg/
├── config.py       # 系统配置与策略参数
├── data/           # 数据流接口 + 漂移仿真器（官方数据包发布前用于开发闭环）
├── drift/          # 漂移监测（PSI/置信度）+ 归因
├── models/         # 模型工厂（LightGBM / Keras MLP）+ 版本注册表
├── strategy/       # 规则策略引擎 + 可插拔 LLM 顾问
├── pipeline/       # 闭环编排 SelfEvolutionLoop
└── guardrails/     # 预算管理与策略白名单（受控 RSI）
experiments/        # SwanLab 实验入口脚本
tests/              # 冒烟测试
```

## 里程碑

- M0（进行中）：仓库骨架 + 漂移仿真器 + 闭环 v0.1 + SwanLab 规范接入
- M1：官方数据包发布后对齐数据格式与评测脚本，公开集跑通全流程
- M2：PyTRiO LLM 接入、Docker 镜像打包、离线自测与成本调优
- M3：实验报告与答辩材料

## 环境

Python ≥ 3.10；依赖见 `requirements.txt`（与开发机实测版本锁定一致）。
官方评测在统一离线环境运行提交的镜像，与本地环境无关。
