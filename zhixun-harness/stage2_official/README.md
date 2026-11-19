# stage2_official 模块地图

平台挂载的提交入口与全部官方 SDK 逻辑。`contest_submission.py` 是平台回调入口，其余模块按职责分组。

## 入口

| 文件 | 职责 |
| --- | --- |
| `contest_submission.py` | `describe` / `train` / `predict` 回调；读当轮数据与历史，决策并写 `rounds.json` |
| `local_entry.py` | 无 Docker 时的本地路径映射，仅供测试 |
| `Dockerfile` | linux/amd64 提交镜像 |

## 决策层

| 文件 | 职责 |
| --- | --- |
| `policy.py` | `Experiment` 模型、实验去重键、历史反馈筛选 |
| `normalize.py` | 模型建议的数值夹取与规则兜底 |
| `admission.py` | 可选准入护栏：增益门、锚点 floor、回滚预演 |
| `continuation.py` | 续训配方：standard、类平衡（A）、huber（B/C）、l2、L2-SP、传感器遮蔽、B 残差损失、C 工况分组 MSE |
| `operating_conditions.py` | C 任务的工况分组与分组 MSE 权重 |
| `trainer.py` | 候选训练子进程入口，产出 `training.json` 与 `checkpoint.npz` |

## 观测层

| 文件 | 职责 |
| --- | --- |
| `drift.py` | 相邻轮输入直方图的 TV 距离（research 模式） |
| `diagnostics.py` | 训练投影与验证投影的 PSI（`ZHIXUN_DIAGNOSE=1`） |
| `export_swanlab.py` | 把 `rounds.json` 转成 SwanLab 离线记录，附单轮效率比 |

## 演练与打包

| 文件 | 职责 |
| --- | --- |
| `rehearse.py` | 本地模拟宿主：逐轮发数据、评测候选、检查协议与产物 |
| `build_context.py` | 按白名单收集镜像上下文，校验依赖锁与基线源码哈希 |
| `stage_linux_workspace.py` | 为 WSL 2 在原生磁盘上准备验收目录 |
| `verify_linux.sh` | 构建镜像、跑十轮、校验 cgroup 限制与归档大小 |
| `tests/` | 本包单元测试，运行 `pytest stage2_official/tests` |

## 依赖方向

```
contest_submission → policy, normalize, admission, continuation, operating_conditions,
                     trainer, diagnostics, drift, rehearse
policy             → continuation
normalize          → policy, continuation
diagnostics        → trainer
export_swanlab     → rehearse
```

`continuation`、`operating_conditions`、`drift` 不依赖决策层；发现反向引用即为循环依赖，应改在
`contest_submission` 里装配。

## 环境开关

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `ZHIXUN_ADMISSION` | 开启（设 0 关闭） | 启用准入护栏 |
| `ZHIXUN_DIAGNOSE` | 关闭 | 计算训练/验证投影 PSI |
| `ZHIXUN_RESEARCH_MODE` | 关闭 | 开放多配方、计算 TV 漂移与稳定性切片 |
| `ZHIXUN_ENABLE_L2SP` / `ZHIXUN_ENABLE_B_RESIDUAL` / `ZHIXUN_ENABLE_SENSOR_MASK` / `ZHIXUN_ENABLE_C_GROUP_MSE` | 关闭 | 按任务开放对应配方 |
| `ZHIXUN_C_HALF_RATE` | 关闭 | C 标准续训学习率减半 |
