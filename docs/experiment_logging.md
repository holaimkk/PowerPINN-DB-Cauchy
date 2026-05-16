# 实验记录规范

每一次实验都必须能够通过保存下来的配置文件复现，并能追溯到外部飞书实验记录表。

## 实验编号

使用连续编号：

```text
EXP-001
EXP-002
EXP-003
```

输出目录应为：

```text
experiments/EXP-001/
```

## 必需目录内容

```text
experiments/EXP-001/
├── config.yaml
├── metrics.csv
├── loss_history.csv
├── train.log
├── predictions.csv
├── model.pt
├── figures/
│   ├── loss_curve.png
│   └── prediction_vs_truth.png
└── README.md
```

## 必需指标

至少记录：

- `l2_relative_error`
- `mse`
- `mae`
- `max_absolute_error`
- `physics_residual_mse`
- `training_time_sec`
- `epoch`
- `seed`
- `device`

## 飞书表格字段

推荐字段：

| 字段 | 含义 |
|---|---|
| 实验编号 | `EXP-001`、`EXP-002` 等 |
| 日期 | 实验运行日期 |
| 负责人 | 本次实验的主要执行人 |
| 模型 | ToyPINN、PowerPINN、Cauchy-PowerPINN 等 |
| 配置路径 | 保存后的配置文件路径 |
| 网络结构 | 层数、宽度、激活函数 |
| 损失项 | 物理残差、初始条件、数据损失及其权重 |
| 优化器 | 优化器名称和学习率 |
| 训练轮数 | epoch 数 |
| 采样点数量 | collocation、data、IC 等采样点数量 |
| 随机种子 | 本次实验使用的 seed |
| 运行设备 | CPU、MPS、CUDA |
| 运行时间 | wall-clock time |
| 指标 | L2RE、MSE、MAE、物理残差等 |
| 输出路径 | `experiments/EXP-XXX/` |
| 备注 | 是否成功、问题、下一步动作 |

## 记录原则

- 不要覆盖重要的已完成实验，除非明确是在重复同一实验编号。
- 如果对应输出目录不存在，不要在论文、汇报或阶段总结中引用该实验结果。
- 必要时在备注中标记 `未运行`、`失败` 或 `需要重跑`。
- 消融实验中每次只改变一个变量。
