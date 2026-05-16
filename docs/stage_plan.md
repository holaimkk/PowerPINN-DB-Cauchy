# 项目阶段计划

本文档用于保持研究流程清晰、可复现。项目应先从经过验证的小规模示例开始，再推进到完整 PowerPINN 复现，随后每次只加入一个改进模块。

## 阶段 0：协作框架搭建

状态：基本完成。

交付物：

- GitHub 仓库；
- 基础目录结构；
- README；
- 论文阅读模板；
- 外部飞书实验记录表。

## 阶段 1：论文精读与方法拆解

目标：在正式实现前，明确每篇论文对本项目的可复用贡献。

必需产出：

- `papers/` 中每篇核心论文对应一份阅读卡片；
- PowerPINN 9 阶同步发电机任务的变量、方程、损失项和评价指标；
- Cauchy 激活函数公式与 PyTorch 实现注意事项；
- DB-PINN 的 inter-balancing 与 intra-balancing 权重更新逻辑；
- ConFIG 和 PI-KAN 中适合作为公平对比对象的组件说明。

退出标准：

- baseline 任务已经完成数学定义；
- 团队能够明确指出 Cauchy 激活函数和 DB 权重将接入代码的具体位置；
- 在 baseline 路径理解清楚前，不添加最终融合模型代码。

## 阶段 2：最小 PINN 示例

目标：用一个简单 ODE 验证项目代码框架：

```text
u'(t) = -u(t), u(0) = 1, 精确解 u(t) = exp(-t)
```

必需产出：

- 从 `configs/toy_pinn.yaml` 驱动运行；
- `experiments/EXP-XXX/` 下生成实验目录；
- `config.yaml`；
- `metrics.csv`；
- `loss_history.csv`；
- `train.log`；
- `predictions.csv`；
- `model.pt`；
- `figures/loss_curve.png`；
- `figures/prediction_vs_truth.png`；
- 每次实验目录中的 `README.md`。

退出标准：

- Toy PINN 命令可以在 CPU、MPS 或 CUDA 上运行；
- 输出文件已保存，并能与外部飞书实验记录对应；
- 团队理解自动微分、物理残差和初始条件损失的基本写法。

## 阶段 3：PowerPINN Baseline 复现

目标：在加入新方法前，先复现 9 阶同步发电机 baseline。

必需产出：

- `src/systems/` 下清晰的系统定义；
- baseline 配置文件；
- baseline 实验日志与指标；
- `docs/` 下的简短复现报告。

## 阶段 4：Cauchy 激活函数

目标：在保持 baseline 其他设置不变的情况下，只比较激活函数选择的影响。

必需产出：

- Cauchy 激活函数模块；
- baseline 与 Cauchy 版本的配套配置；
- 受控对比表格与图表。

## 阶段 5：DB-PINN 动态损失平衡

目标：比较固定损失权重与 DB-PINN 风格动态权重的差异。

必需产出：

- 损失权重模块；
- 权重历史日志；
- 受控对比表格与图表。

## 阶段 6：融合模型与消融实验

目标：运行四组核心消融实验。

必需实验组：

| 模型 | Cauchy 激活 | DB 权重 |
|---|---|---|
| PowerPINN | 否 | 否 |
| Cauchy-PowerPINN | 是 | 否 |
| DB-PowerPINN | 否 | 是 |
| DB-Cauchy-PowerPINN | 是 | 是 |

## 阶段 7：扩展对比与论文撰写

目标：加入选定的 ConFIG 或 PI-KAN 对比，并准备论文与软著材料。

重要规则：报告中的每一个结论都必须能追溯到真实存在的实验目录。
