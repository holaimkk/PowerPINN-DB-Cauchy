# AGENTS.md

本文件用于约束所有在本仓库中工作的 AI 编码代理和协作者。执行任务前请先阅读本文件，并优先遵守这里的项目约定。

## 项目背景

本项目研究面向电力系统动态仿真的 PINN 改进方法，核心路线为：

```text
PowerPINN 基准复现 -> Cauchy 激活函数改进 -> DB-PINN 动态损失平衡 -> DB-Cauchy-PowerPINN 融合模型 -> 对比实验与论文整理
```

当前仓库处于“文献拆解 + 最小 Toy PINN 示例验证”阶段。Toy PINN 用于验证配置系统、训练循环、日志、指标和图表输出；PowerPINN 的 9 阶同步发电机 baseline 尚未正式实现。

## 语言与文档要求

- 仓库内所有说明性文档必须使用中文撰写，包括 `README.md`、`docs/` 下的阶段计划和实验规范、每次实验输出目录中的 `README.md`、阶段总结、复现实验报告和论文阅读卡片。
- 代码标识符、配置键、命令、指标字段和必要的英文论文术语可以保留英文，但解释文字必须使用中文。
- 如果发现已有说明文档为英文，应在相关任务中同步改为中文，不要继续扩散英文说明。
- 每完成一次与用户的对话后，必须在飞书中新建一篇中文文档，记录本次对话完成了什么、修改了哪些文件、是否运行了测试或实验、后续还需要做什么。若当前环境无法直接访问飞书，应在最终回复中明确说明无法代建飞书文档，并给出可直接粘贴到飞书的新文档内容。

## 代码结构

```text
PowerPINN-DB-Cauchy/
├── configs/         # 实验配置文件
├── docs/            # 阶段计划、实验规范、总结和报告
├── papers/          # 核心论文与阅读模板
├── scripts/         # 命令行入口
├── src/             # 模型、系统、损失、训练和工具代码
│   ├── losses/
│   ├── models/
│   ├── systems/
│   ├── training/
│   └── utils/
├── tests/           # 单元测试
├── README.md
└── AGENTS.md
```

当前最小实验入口：

```bash
python3 scripts/run_toy_pinn.py --config configs/toy_pinn.yaml
```

## 开发原则

- 先复现基准，再加入改进模块。PowerPINN baseline 的变量、方程、损失项、指标和数据来源确认前，不要直接接入 Cauchy、DB-PINN、ConFIG 或 PI-KAN。
- 每次只改变一个关键变量。进行消融实验时，除被比较模块外，网络结构、采样点、学习率、训练轮数、随机种子和设备应尽量保持一致。
- 保持配置驱动。新增实验优先通过 `configs/` 中的 YAML 配置控制，不要把实验参数硬编码进训练脚本。
- 保持输出可追溯。任何论文、汇报或总结中的结论，都必须能追溯到真实存在的 `experiments/EXP-XXX/` 目录和飞书实验记录。
- 不要覆盖重要实验结果。除非明确是在重复同一编号并且配置中允许覆盖，否则不要删除或覆盖已完成实验。
- 保持 `main` 分支可运行。新功能建议使用独立分支，例如 `feature-baseline`、`feature-cauchy`、`feature-db`、`feature-config`。

## 实验记录要求

每一次实验都应生成并保留以下文件：

```text
experiments/EXP-XXX/
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

每次有效实验还必须登记到飞书实验记录表，至少包含：

- 实验编号；
- 日期；
- 负责人；
- 模型版本；
- 配置路径；
- 网络结构；
- 损失项与权重；
- 优化器与学习率；
- 训练轮数；
- 采样点数量；
- 随机种子；
- 运行设备；
- 运行时间；
- 核心指标；
- 输出路径；
- 问题、结论与下一步。

## 测试与验证

常用验证命令：

```bash
python3 -m pytest
python3 scripts/run_toy_pinn.py --config configs/toy_pinn.yaml
```

如果修改了训练流程、损失函数、系统方程或模型结构，应至少运行相关单元测试。若运行完整训练成本过高，应说明没有运行的原因，并给出替代检查结果。

## 文件修改注意事项

- 不要回滚用户已有改动。开始修改前先查看 `git status --short`，只编辑与任务相关的文件。
- 新增说明文档使用中文，新增代码注释保持简洁，只有在公式或逻辑不直观时才添加。
- 新增模型或损失模块时，应同步补充配置示例、实验记录字段和最小测试。
- 论文公式、变量维度和物理含义必须人工核对，不要只依赖代码生成结果。

## 阶段推进顺序

1. 完成核心论文阅读卡片。
2. 明确 PowerPINN 9 阶同步发电机任务的变量、方程、损失项和评价指标。
3. 跑通并记录 Toy PINN 最小实验。
4. 实现 PowerPINN baseline 并形成复现报告。
5. 在固定 baseline 上比较 Cauchy 激活函数。
6. 在固定 baseline 上比较 DB-PINN 动态损失权重。
7. 完成四组消融实验：PowerPINN、Cauchy-PowerPINN、DB-PowerPINN、DB-Cauchy-PowerPINN。
8. 选择 ConFIG 或 PI-KAN 等方法做扩展对比，并整理论文和软著材料。
