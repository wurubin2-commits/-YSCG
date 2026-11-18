# 提分 Harness 源码调研

2026-11-18 源码核查。建议先用 **autoresearch 的固定预算对照**筛续训参数，再用 **OpenEvolve 的分级评估**控制回放成本，用 **AI Scientist-v2 的复验**决定是否把配方加入正式 Harness。下文是实验设计，尚无新增训练结果。

## 上游实际行为

| 上游 | 可核对的机制 | 用于本项目 |
| --- | --- | --- |
| **karpathy/autoresearch**，README 声明 **MIT** | `program.md` 要求先跑基线，固定 `prepare.py` 的数据与评分，逐项改 `train.py` 并用 `results.tsv` 记录保留、丢弃或出错。训练循环以 300 秒为预算，按耗用比例控制学习率的预热、平台段和衰减段，结束时调用固定的 `evaluate_bpb`。 | 锁定父检查点、训练/验证投影、样本顺序、种子及计算预算。先比一个生效参数，例如学习率倍率，按原任务指标定胜负。若加入训练内调度，用计划更新步数作进度，并把实际步数和耗时写进日志。CPU 短续训的有效性需单独验证。 [试验规程](https://github.com/karpathy/autoresearch/blob/master/program.md) · [预算与固定验证数据](https://github.com/karpathy/autoresearch/blob/master/prepare.py#L24-L40) · [学习率调度](https://github.com/karpathy/autoresearch/blob/master/train.py#L479-L523) · [许可声明](https://github.com/karpathy/autoresearch/blob/master/README.md#license) |
| **OpenEvolve**，**Apache-2.0** | `Evaluator._cascade_evaluate` 依次调用评估器实际提供的 `evaluate_stage1/2/3`，前段未过阈值即返回。阈值优先使用 `combined_score`，缺省时取非布尔数值的均值（排除 `error`）。后段超时或异常时带回已得到的前段结果及失败信息。 | 先检查续训与加载有效性，再对固定若干公开轮次做配对初筛，通过后才做完整十轮。每项任务用自己的主指标方向，切片退化与超支各设独立门槛。首批可抽查一条初筛落选配方是否在十轮反超，再决定是否使用该初筛。 [分级评估](https://github.com/algorithmicsuperintelligence/openevolve/blob/main/openevolve/evaluator.py#L420-L547) · [阈值实现](https://github.com/algorithmicsuperintelligence/openevolve/blob/main/openevolve/evaluator.py#L723-L758) · [许可证](https://github.com/algorithmicsuperintelligence/openevolve/blob/main/LICENSE) |
| **Sakana AI Scientist-v2**，**The AI Scientist Source Code License v1.0** | `AgentManager.run` 把上一子阶段最佳节点复制到新日志，在大阶段结束时对最佳节点做多种子复跑。 `Journal.get_best_node` 支持按验证指标挑选，多节点时默认调用模型综合选优。许可证衍生自 Responsible AI Source Code License，含用途及研究输出披露条款。 | 借用「入围后复验」的顺序。把入围配方和基线用留出的公开批次或独立回放顺序复验，记录变差的任务切片。若本地演练支持独立种子，应用同种子配对。正式环节由现有规则和宿主公开评分作决定。 [阶段与复跑](https://github.com/SakanaAI/AI-Scientist-v2/blob/96bd516/ai_scientist/treesearch/agent_manager.py#L647-L707) · [节点选择](https://github.com/SakanaAI/AI-Scientist-v2/blob/main/ai_scientist/treesearch/journal.py#L388-L461) · [许可证](https://github.com/SakanaAI/AI-Scientist-v2/blob/main/LICENSE) |

## 第一组有判定力的对照

1. **先按任务固定一个父模型和同批基线。** 从既有公开演练里预先选代表轮次，同批比较当前配置与一个有明确假设的改动，例如固定 batch 和 epoch 后只改学习率倍率。再比较两种 12 epoch 分配，同为每轮最多用 1 个常规 epoch，剩下 2 个额外 epoch 分别按预定轮次或预定条件分配。把每轮最多 1 epoch 的版本作为低成本参照，实际发生保留或失败时照实记账。报告尝试 epoch、更新步数、候选数和耗时，通过实际用量判断收益中有多少可能来自增加计算。
2. **先看相同轮次的成对差值，再复验整条父子链。** A 看 Macro-F1 的候选减父值，B/C 看父减候选的 RMSE，单位分别为分钟和循环。整条轨迹与基线用同一公开回放顺序，允许各自按现行规则演化父检查点，同时报告末轮、最差切片、采用次数及成本。不同轮次的原始分数受数据和父模型变化影响。根据预留公开复验决定是否开启配方，父摘要、验证摘要或指标口径不同的记录留作背景。
3. **上线只带经复验的参数或调度边界。** 正式 SDK 读取当轮平台指定的 NumPy 父检查点及公开投影，十轮各通过宿主网关取得至少一条真实模型调用回执。现有参赛入口每轮最多登记一个候选，12 epoch 是项目上限。`rnd.submit_candidate` 与 `rnd.finish` 使用现有宿主评估、父模型保留和回执流程。赛前试验日志可复查，离线生成的检查点只用于本地评估。 [当前入口](../../stage2_official/contest_submission.py#L101) · [官方 SDK 说明](https://rsi-index.115.zone/#rules)
