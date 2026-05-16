# PowerPINN-DB-Cauchy ⚡

本项目面向电力系统动态仿真中的高阶非线性微分方程求解问题，拟以 **PowerPINN 中的 9 阶同步发电机动态模型** 为主要研究对象，探索将 **Cauchy 激活函数** 与 **DB-PINN 动态损失平衡策略** 引入电力系统 PINN 建模任务的可行性与有效性。

项目核心路线：

> **PowerPINN 基准复现 → Cauchy 激活函数架构改进 → DB-PINN 损失平衡优化 → DB-Cauchy-PowerPINN 融合模型 → 对比实验与论文投稿**

---

## 1. 项目定位

本项目不是简单复现某一篇论文，而是围绕电力系统动态模型构建一个可复现、可对比、可分析的 PINN 改进实验框架。

主要研究问题：

1. 在 PowerPINN 的 9 阶同步发电机任务中，普通 PINN 是否存在收敛慢、物理残差下降困难或多损失项不平衡问题？
2. 引入 Cauchy 激活函数后，网络对复杂动态轨迹的拟合能力是否提升？
3. 引入 DB-PINN 动态损失平衡后，多损失项训练过程是否更加稳定？
4. 二者结合后的 DB-Cauchy-PowerPINN 是否优于原始 PowerPINN 及相关对比方法？

---

## 2. 团队组织与分工

本项目采用 **“负责人主导推进 + 成员模块协助”** 的组织方式。项目负责人参与并把控所有核心环节，其他成员围绕文献整理、模块实现、实验记录和结果分析提供支持。

### 杨泽坤｜项目负责人 / 全流程统筹

负责项目整体推进与关键技术把关，参与并统筹所有核心阶段。

主要职责：

- 制定项目技术路线、阶段目标和实验规范；
- 主导 PowerPINN 基准模型复现；
- 统筹 Cauchy 激活函数、DB-PINN 损失平衡模块的接入；
- 维护主分支代码、实验记录表和阶段总结；
- 负责论文整体框架、主要实验结果分析与最终统稿；
- 负责与导师沟通阶段性进展和下一步方向。

### 孙修驰｜文献整理 / Cauchy 模块协助 / 代码实现支持

主要协助完成网络架构相关内容，重点支持 compleX-PINN 与 Cauchy 激活函数部分。

主要职责：

- 精读 compleX-PINN 及相关架构改进论文，完成阅读卡片；
- 整理 Cauchy 激活函数的公式、参数含义、初始化方式和 PyTorch 实现要点；
- 协助实现和测试 Cauchy activation 模块；
- 参与 Cauchy-PowerPINN 相关实验记录和结果分析；
- 协助撰写论文中的相关工作与架构改进部分。

### 汤震雨｜DB-PINN 模块协助 / 对比实验与结果整理

主要协助完成损失平衡和对比实验相关内容，重点支持 DB-PINN、ConFIG、PI-KAN 的整理与实验记录。

主要职责：

- 精读 DB-PINN、ConFIG、PI-KAN 等论文，完成阅读卡片；
- 整理 DB-PINN 的 inter-balancing、intra-balancing 权重更新逻辑；
- 协助实现和测试动态损失权重模块；
- 负责实验结果表格、训练日志、图表文件的整理；
- 协助完成对比实验、消融实验与论文图表制作。

---

## 3. 阶段路线

### 阶段 0：项目协作框架搭建

目标：建立统一代码仓库、实验记录表和文献阅读模板。

验收成果：

- GitHub 仓库建立完成；
- 飞书实验记录表建立完成；
- README、论文阅读模板、目录结构完成；
- 明确近期任务分工。

### 阶段 1：文献精读与方法拆解

目标：明确每篇论文中可以复用的内容和暂不开展的内容。

核心论文：

- PowerPINN；
- DB-PINN；
- compleX-PINN；
- ConFIG；
- PI-KAN。

验收成果：

- 每篇核心论文形成一份阅读卡片；
- 明确 PowerPINN 的输入、输出、损失项和评价指标；
- 明确 Cauchy 激活函数与 DB-PINN 损失平衡的接入位置。

### 阶段 2：最小 PINN 示例跑通

目标：先在简单 ODE/PDE 上跑通 PINN 基本流程，降低后续 PowerPINN 复现风险。

验收成果：

- 完成一个最小 PINN 示例；
- 输出预测曲线、loss 曲线和误差指标；
- 三名成员理解自动微分、物理残差和初始条件损失的基本写法。

### 阶段 3：PowerPINN 基准模型复现

目标：跑通 9 阶同步发电机模型，得到可重复 baseline。

验收成果：

- PowerPINN baseline 能稳定运行；
- 记录预测误差、物理残差、训练时间和 loss 曲线；
- 完成《PowerPINN 复现报告》。

### 阶段 4：Cauchy 激活函数改进

目标：在 PowerPINN 框架中引入 Cauchy activation，评估架构改进效果。

验收成果：

- 完成 Cauchy activation 模块；
- 完成 tanh / Cauchy 等激活函数对比；
- 得到误差对比表和 loss 曲线；
- 完成《Cauchy 激活改进实验报告》。

### 阶段 5：DB-PINN 动态损失平衡改进

目标：将 DB-PINN 损失平衡策略迁移到 PowerPINN 多损失项训练中。

验收成果：

- 完成动态损失权重模块；
- 记录各损失项权重变化曲线；
- 完成固定权重与 DB 权重对比；
- 完成《DB-PINN 损失平衡实验报告》。

### 阶段 6：融合模型与消融实验

目标：实现 DB-Cauchy-PowerPINN，并验证两个模块的独立贡献与组合效果。

核心实验组：

| 模型 | Cauchy 激活 | DB 权重 | 作用 |
|---|---|---|---|
| PowerPINN | × | × | 原始基准 |
| Cauchy-PowerPINN | √ | × | 检验架构改进 |
| DB-PowerPINN | × | √ | 检验损失平衡 |
| DB-Cauchy-PowerPINN | √ | √ | 检验融合效果 |

验收成果：

- 完成融合模型训练；
- 完成核心消融实验表；
- 明确最终方法相较 baseline 的优势与局限。

### 阶段 7：对比实验与论文整理

目标：与 ConFIG、PI-KAN 等方法进行对比，形成论文级实验结果。

验收成果：

- 完成至少一种前沿方法对比；
- 完成稳定性和效率分析；
- 完成论文初稿、投稿准备和软著材料整理。

---

## 4. 目录结构说明

```text
PowerPINN-DB-Cauchy/
├── papers/          # 核心参考论文与阅读笔记
├── src/             # 核心代码：模型、损失函数、训练流程
├── configs/         # 实验配置文件，如 yaml/json
├── data/            # 数据文件，默认不上传仓库
├── experiments/     # 每次实验的日志、权重和输出结果
├── figures/         # 论文和汇报使用的图表
├── docs/            # 周报、阶段总结、答辩材料
├── README.md
└── .gitignore
```

---

## 5. 实验记录规范

每一次实验都必须在飞书实验记录表中登记，建议字段如下：

| 字段 | 说明 |
|---|---|
| 实验编号 | 例如 E0001 |
| 日期 | 实验运行日期 |
| 负责人 | 主要执行人 |
| 模型版本 | PowerPINN / Cauchy / DB / DB-Cauchy |
| 网络结构 | 层数、宽度、激活函数 |
| 损失设置 | 数据损失、物理残差、初值损失等 |
| 学习率 | optimizer 与 lr |
| 训练轮数 | epoch 数 |
| 采样点数量 | collocation points / data points |
| 随机种子 | seed |
| 运行设备 | Mac M2 / 联想拯救者 GPU / 云端 GPU |
| 运行时间 | wall-clock time |
| 最终误差 | L2RE / MSE / MAE 等 |
| 物理残差 | 最终 residual |
| 结果路径 | experiments/ 下的目录 |
| 备注 | 是否成功、问题与下一步 |

---

## 6. 协作规范

1. **主分支保持稳定**：`main` 分支只保留可运行版本，不直接提交未验证代码。
2. **功能分支开发**：新功能建议使用独立分支，例如：
   - `feature-baseline`
   - `feature-cauchy`
   - `feature-db`
   - `feature-config`
3. **实验先记录再结论**：所有结论必须对应实验编号和结果文件路径。
4. **代码辅助工具使用原则**：可以使用 Codex、Claude Code 等工具辅助写代码，但必须人工检查公式、变量维度和实验结果。
5. **每周复盘**：每周至少更新一次 `docs/` 中的阶段进展，记录当前已完成内容、问题和下一步计划。

---

## 7. 当前最小可运行实验：Toy PINN

阶段 2 的最小 PINN 示例用于验证配置系统、训练循环、日志、指标和图表输出是否可用。当前 toy 问题为：

```text
u'(t) = -u(t), u(0) = 1
精确解：u(t) = exp(-t)
```

### 7.1 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

也可以使用 Conda：

```bash
conda env create -f environment.yml
conda activate powerpinn-db-cauchy
```

### 7.2 运行 Toy PINN

```bash
python3 scripts/run_toy_pinn.py --config configs/toy_pinn.yaml
```

运行后会生成：

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

说明：`experiments/` 默认不提交到 Git。实验结果需要手动登记到飞书实验记录表中。

### 7.3 后续 PowerPINN Baseline 入口规划

PowerPINN baseline 尚未实现。建议后续保持同样的配置驱动方式，例如：

```bash
python3 scripts/run_powerpinn_baseline.py --config configs/powerpinn_baseline.yaml
```

在完成 PowerPINN 论文公式拆解、变量定义和数据来源确认前，不应直接加入 Cauchy、DB-PINN、ConFIG 或 PI-KAN 模块。

---

## 8. 当前状态

当前处于 **阶段 1 文献拆解 + 阶段 2 最小 PINN 示例验证**。

已完成：

- GitHub 仓库创建；
- 基础目录结构创建；
- 飞书实验记录表创建；
- README 与阅读模板初步整理。
- Toy PINN 最小示例框架已建立；
- `EXP-001` 已在本机 CPU 上真实运行并生成完整实验输出。

下一步：

1. 完成 5 篇核心论文阅读卡片；
2. 确认 PowerPINN 代码入口与 9 阶同步发电机模型脚本；
3. 制定 PowerPINN baseline 复现清单；
4. 在不引入 Cauchy/DB 的前提下实现 PowerPINN baseline。
