# 论文阅读卡片：PowerPINN

> 目的：不是泛泛总结论文，而是判断这篇论文对本项目有什么可复用价值、代码怎么接入、实验怎么对标。

---

## 0. 基本信息

**论文题目：** Toolbox for Developing Physics Informed Neural Networks for Power Systems Components  
**论文方向：** PowerPINN  
**发表年份：** 2025  
**发表期刊/会议/平台：** arXiv / 电力系统 PINN 工具箱论文  
**阅读人：** 杨泽坤  
**阅读日期：** 2026-05-16  
**代码链接：** https://github.com/radiakos/PowerPINN  
**论文文件路径：** `papers/PowerPINN.pdf`

---

## 1. 这篇论文解决了什么问题？

PowerPINN 主要解决的是电力系统动态组件 PINN 建模缺乏标准化工具链的问题。传统电力系统动态仿真依赖 Runge-Kutta 等数值 ODE 求解器，但在高阶、非线性、需要大量轨迹推理的场景下计算成本较高。论文提出一个 Python-based toolbox，用于定义电力系统组件 ODE、生成数据集、训练 PINN，并评价其对组件动态行为的逼近能力。

它针对 PINN 在电力系统领域的两个痛点：一是复现流程不统一，二是复杂动态组件训练成本高。对本项目关系极强，因为我们项目的研究对象就暂定为该论文中的 9 阶同步发电机 + AVR + Governor 模型。

它解决的是“电力系统建模问题”和“PowerPINN baseline 问题”，不是本文主打的激活函数或损失平衡方法。

**简要总结：**  
PowerPINN 是本项目最重要的 baseline 论文，提供了 9 阶同步发电机动态模型、数据生成流程、PINN 输入输出形式、损失函数结构和评价指标。

---

## 2. 方法核心逻辑

### 2.1 模型输入与输出

**输入变量：**  
- 初始状态向量 `x0`
- 时间 `t`
- 论文中通用 ODE 表达包含系统输入 `u` 和参数 `μ`，但实际神经网络输入主要围绕初始状态和时间构造。

**输出变量：**  
- 系统在时间 `t` 的状态向量 `x(t)`

**是否适合迁移到 9 阶同步发电机模型：** 是  

说明：  
论文明确说明，对于包含 `K` 个动态状态的系统，网络输入层具有 `K + 1` 个输入，输出层具有 `K` 个输出，其中额外的 1 个输入表示时间。对于本项目的 9 阶系统，初步可理解为输入维度为 10，输出维度为 9。后续必须从原代码中确认 9 个状态变量的具体顺序。

---

### 2.2 网络结构

- 使用的网络类型：Feed-forward Neural Network / PINN
- 激活函数：论文正文未在当前阅读卡片中完全确认，需结合原仓库配置进一步核实；项目后续重点检查是否为 `tanh`
- 网络层数与宽度：需从 `setup_dataset_nn.yaml` 或训练脚本确认
- 是否有特殊结构：标准 PINN，结合数据损失、物理残差损失、初始条件损失
- 参数量是否需要对齐：需要。后续做 Cauchy-PowerPINN 时应优先保持网络层数、宽度、采样点数、优化器等不变，只替换激活函数，以保证对比公平。

说明：  
PowerPINN 的重点不是提出新网络架构，而是提供组件级 PINN 训练流程。论文强调网络复杂度与被建模组件的动态状态数有关，复杂系统需要更多神经元或层数。

---

### 2.3 损失函数设计

| 损失项 | 含义 | 是否和本项目相关 |
|---|---|---|
| `L_data` | PINN 预测状态与 ODE solver 轨迹之间的数据拟合损失 | 是 |
| `L_data_physics` | 在 labeled data 点上计算的物理残差损失 | 是 |
| `L_col_physics` | 在 collocation points 上计算的无标签物理残差损失 | 是 |
| `L_ic` / `L_ic_col` | 初始条件约束损失 | 是 |
| 其他 | 系统输入、参数、采样配置等辅助项 | 是，但不是主 loss |

**总损失形式：**

```text
L_total = λd * L_data
        + λdp * L_data_physics
        + λcp * L_col_physics
        + λic * L_ic_col
```

说明：  
这组损失项非常适合后续接入 DB-PINN。第一阶段可以先简化为三项：`L_data`、`L_phy`、`L_ic`；等 baseline 跑通后，再拆成四项权重进行细粒度动态平衡。

---

## 3. 关键算法步骤

```text
Step 1: 定义电力系统组件 ODE，包括状态变量、输入、参数和右端函数 f(t, x, u, μ)。
Step 2: 在输入域内采样初始条件，论文推荐 LHS 以覆盖更广状态空间。
Step 3: 使用 RK45 等数值 ODE solver 生成 labeled trajectories。
Step 4: 将轨迹处理为 (x0, t) -> x(t) 的监督学习格式，同时构造 collocation points。
Step 5: 训练 PINN，使其同时最小化数据拟合损失、物理残差损失和初始条件损失。
Step 6: 用测试轨迹对比 ODE solver 输出，计算 MSE、MAE、Max AE 和推理时间。
```

论文中存在伪代码、公式推导或关键流程图时，在这里记录位置：

**伪代码 / 关键公式 / 图表位置：**

- ODE 初值问题：`dx/dt = f(t, x, u, μ), x(t0)=x0`
- 数据损失：论文公式 (3)
- 数据点物理损失：论文公式 (4)
- 初始条件损失：论文公式 (5)
- collocation 物理损失：论文公式 (6)
- 总损失：论文公式 (7)
- PINN 输入输出结构：Fig. 1
- 9 阶系统输入域、采样和数据集规模：实验部分
- 结果图：Fig. 2 / Fig. 3
- 推理时间：Table II

---

## 4. 实验设置与评价指标

### 4.1 实验对象

- 方程类型：常微分方程 ODE
- 应用场景：电力系统动态组件建模
- 数据来源：数值 ODE solver 生成的模拟轨迹
- 是否有 ground truth：有，以 ODE solver 结果作为参考解
- 是否和 PowerPINN 9 阶模型相近：完全一致，是本项目 baseline

论文实验对象为：

```text
4 阶同步机 + 3 阶 AVR + 2 阶 Governor = 9 阶系统
```

输入域中包含 `δ, ω, E'_d, E'_q, RF, Vr, Efd, Psv, Pm` 等状态变量或相关量。论文使用 500 组 LHS 初始条件，1 ms 时间步长，1 s 仿真周期，形成约 500,000 个 labeled data points 和 500,000 个 collocation points。

### 4.2 Baseline 方法

| Baseline | 作用 |
|---|---|
| ODE solver / RK45 | 作为参考解和数据生成器 |
| 原始 PowerPINN | 本项目 baseline |
| 后续 Cauchy-PowerPINN | 与原始激活函数对比 |
| 后续 DB-PowerPINN | 与原始静态权重对比 |

### 4.3 评价指标

| 指标 | 含义 | 我们是否采用 |
|---|---|---|
| L2 Relative Error | 相对误差，论文未必主用，但本项目建议加入 | 是 |
| MSE / MAE | 预测误差 | 是 |
| Max Error / Max AE | 最大绝对误差 | 是 |
| Physics Residual | 物理残差 | 是 |
| Epoch | 收敛轮数 | 是 |
| Wall-clock Time | 实际训练时间 / 推理时间 | 是 |

---

## 5. 论文中最值得我们复用的部分

### 5.1 可直接复用

- 9 阶同步发电机 + AVR + Governor 作为核心实验对象。
- `(x0, t) -> x(t)` 的输入输出格式。
- LHS 初始条件采样 + RK45 轨迹生成的数据生成流程。
- `L_data + L_data_physics + L_col_physics + L_ic` 的损失函数拆分方式。
- MSE、MAE、Max AE、推理时间等评价指标。

### 5.2 需要改造后复用

- 原始网络结构：先保持不变用于 baseline，再替换激活函数为 Cauchy。
- 原始损失权重：先复现 static weight，再改为 DB-PINN 动态权重。
- 原始数据规模：本科复现不建议一开始就使用 500,000 点，应先使用小规模数据跑通。

### 5.3 暂时不做

- 不立即扩展到整个电力系统仿真器。
- 不立即开发完整组件库。
- 不立即研究 IBR 等新型组件。

---

## 6. 和本项目的对应关系

这篇论文在我们项目中的角色：

- [x] 研究对象基准：PowerPINN
- [ ] 架构改进来源：Cauchy 激活函数 / compleX-PINN
- [ ] 损失优化来源：DB-PINN
- [ ] 优化策略对比：ConFIG
- [ ] 网络架构对比：PI-KAN
- [x] 背景与相关工作
- [ ] 其他：

请用一句话说明它在项目中的用途：

> PowerPINN 提供本项目的 9 阶电力系统动态模型、baseline 训练流程、数据生成方式和评价指标，是所有后续 Cauchy 与 DB-PINN 改进的基础。

---

## 7. 代码复现信息

**是否有开源代码：** 是  
**代码是否已下载：** 是，项目组已建立 `PowerPINN-DB-Cauchy` 仓库；原仓库是否完整拉取需确认  
**是否已运行：** 否  
**运行设备：** Mac M2 / 联想拯救者 GPU  
**环境依赖：** Python、PyTorch、SciPy、可能包含 WandB、YAML 配置等  
**主入口文件：** 待从原仓库确认，优先查 `create_dataset_d.py`、`test_sweep.py`  
**最重要的脚本：**

```text
create_dataset_d.py
test_sweep.py
src/ode/sm_models_d.py
setup_dataset.yaml
setup_dataset_nn.yaml
modellings_guide.yaml
src/conf/initial_conditions/
src/conf/params/
```

**运行命令：**

```bash
# 数据生成
python create_dataset_d.py

# 训练
python test_sweep.py
```

**目前遇到的问题：**

- 尚未完成 9 个状态变量顺序定位。
- 尚未确认原始网络层数、宽度、激活函数和优化器配置。
- 尚未确认 Mac M2 是否能稳定跑通完整数据规模。
- 需要判断 WandB 是否必须启用，或能否关闭。

---

## 8. 对论文写作的潜在贡献

这篇论文可以放在我们论文的哪个部分？

- [x] Introduction
- [x] Related Work
- [x] Method
- [x] Experiment
- [x] Discussion
- [ ] Conclusion

可引用观点：

1. PINN 可作为电力系统动态组件的神经网络代理模型。
2. 组件级 PINN 比直接对整个电力系统建模更适合当前阶段。
3. 9 阶 SM + AVR + Governor 可作为复杂电力系统组件 PINN benchmark。
4. PINN 的准确性受训练数据质量、loss weighting 和超参数选择影响。
5. 推理阶段 PINN 相比 ODE solver 具有并行与速度优势。

---

## 9. 阅读结论

### 9.1 一句话总结

> PowerPINN 是本项目必须首先复现的 baseline，它定义了 9 阶电力系统动态 PINN 建模任务，并给出了后续激活函数与损失权重改进的接口。

### 9.2 对本项目的优先级

- [x] 极高：必须精读并复现
- [ ] 较高：需要理解并部分复用
- [ ] 中等：作为对比或相关工作
- [ ] 较低：仅作为背景参考

### 9.3 下一步行动

- [x] 继续精读
- [x] 运行代码
- [x] 抽取核心模块
- [x] 写入 related work
- [ ] 暂时搁置

具体下一步：

```text
1. 打开 PowerPINN 原仓库，定位 src/ode/sm_models_d.py。
2. 从 modellings_guide.yaml 和 ODE 文件中确认 9 个状态变量顺序。
3. 记录 setup_dataset.yaml 和 setup_dataset_nn.yaml 中的 baseline 配置。
4. 形成 docs/powerpinn_code_map.md 和 docs/powerpinn_model_decomposition.md。
5. 先用小规模数据复现 baseline，不直接跑 500,000 点完整规模。
```
