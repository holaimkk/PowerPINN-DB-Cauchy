# 论文阅读卡片：PI-KAN

> 目的：不是泛泛总结论文，而是判断这篇论文对本项目有什么可复用价值、代码怎么接入、实验怎么对标。

---

## 0. 基本信息

**论文题目：** Physics-informed Kolmogorov-Arnold networks for multi-material elasticity problems in electronic packaging  
**论文方向：** PI-KAN  
**发表年份：** 2026  
**发表期刊/会议/平台：** Applied Mathematical Modelling  
**阅读人：** 孙修驰  
**阅读日期：** 2026-05-16  
**代码链接：** https://github.com/yanpeng-gong/PIKAN-MultiMaterial  
**论文文件路径：** `papers/PI-KAN .pdf`

---

## 1. 这篇论文解决了什么问题？

这篇论文解决的是多材料电子封装结构中的弹性力学分析问题，不是电力系统动态建模问题。它将 Kolmogorov-Arnold Networks（KAN）引入 physics-informed learning 框架，用 KAN 替代传统 MLP，并结合 Deep Energy Method（DEM）求解多材料结构中的位移场和应力场。

论文强调 KAN 中基于 B-spline 的可学习边激活函数具有分段函数特性，适合处理多材料界面、材料参数突变和局部不连续问题。它避免了传统多材料 PINN 中常见的区域分解和界面连续性惩罚项。

它主要解决的是“网络架构问题”，但应用对象与本项目不同。因此它对本项目的作用是 KAN/PI-KAN 架构参考，而不是当前阶段必须复现的电力系统对比方法。

**简要总结：**  
这篇 PI-KAN 论文可以作为“网络架构对比 / 相关工作”使用，但由于研究对象是多材料弹性问题，不建议作为阶段 2 必做复现实验。

---

## 2. 方法核心逻辑

### 2.1 模型输入与输出

**输入变量：**  
- 结构域坐标，例如二维空间点 `(x, y)`
- 多材料问题中还涉及不同材料域、边界条件、材料参数等

**输出变量：**  
- 位移场，例如 `u_x, u_y`
- 后续通过自动微分得到应变、应力、能量等物理量

**是否适合迁移到 9 阶同步发电机模型：** 不确定  

说明：  
KAN 理论上可以替代 PowerPINN 中的 MLP，用于输入 `(x0, t)` 到输出 `x(t)` 的函数逼近。但这篇论文使用的是能量形式 DEM 和弹性力学问题，不是 ODE residual PINN。因此本项目不应直接照搬其 loss 和实验设置，只能参考其 KAN 替代 MLP 的架构思想。

---

### 2.2 网络结构

- 使用的网络类型：Physics-Informed Kolmogorov-Arnold Network / PIKAN
- 激活函数：KAN 中边上的可学习一元 B-spline 函数；中间层还使用 `tanh` 限制输出范围
- 网络层数与宽度：根据 KAN architecture 设置，例如 `[2,5,5,2]` 等；具体依实验调整
- 是否有特殊结构：激活函数从 MLP 的“节点”移动到 KAN 的“边”，每条边是可学习函数
- 参数量是否需要对齐：如果作为 PowerPINN 对比，必须对齐参数量或至少报告参数量差异

说明：  
传统 MLP 每层执行线性变换 + 固定激活函数；KAN 则在边上设置可学习的一元函数，节点主要做求和。论文认为这有助于提高复杂函数表达能力，尤其适合多材料界面中的分段特征。

---

### 2.3 损失函数设计

请写清楚损失函数由哪些部分组成。

| 损失项 | 含义 | 是否和本项目相关 |
|---|---|---|
| `L_data` | 本文主要不是数据拟合型 PINN，而是能量泛函优化 | 否 |
| `L_phy` | 通过势能泛函体现物理约束，不是强形式 residual | 部分相关 |
| `L_ic` | 本文为静态弹性问题，无时间初值 | 否 |
| DEM energy loss | 内部应变能减外力势能 | 方法参考 |
| admissible displacement constraint | 构造自动满足本质边界条件的位移场 | 可作为相关工作参考 |

**总损失形式：**

论文中 DEM / PIKAN 的核心损失可理解为：

```text
L_PIKAN = Σ_i Ψ_in^{m_i} - Ψ_ex
```

其中：

```text
Ψ_in^{m_i}：第 i 个材料域的内部应变能
Ψ_ex：外力势能 / 外部做功
```

对本项目不建议直接使用该 loss。PowerPINN 中仍应使用 ODE residual loss：

```text
L_total = λdata * L_data + λphy * L_phy + λic * L_ic
```

---

## 3. 关键算法步骤

```text
Step 1: 定义多材料物理域、边界条件、材料参数和采样点。
Step 2: 对坐标进行归一化，使其适配 KAN 的 B-spline 定义域。
Step 3: 用单个 KAN 网络预测所有采样点的位移场。
Step 4: 构造 admissible displacement field，使本质边界条件自动满足。
Step 5: 用自动微分计算位移梯度、应变和应力。
Step 6: 对每个材料域计算内部应变能。
Step 7: 计算外力势能，形成 DEM energy loss。
Step 8: 反向传播更新 KAN 参数，直至能量损失收敛。
Step 9: 与 FEM reference solution 比较位移误差和应力误差。
```

论文中存在伪代码、公式推导或关键流程图时，在这里记录位置：

**伪代码 / 关键公式 / 图表位置：**

- DEM 能量损失：Section 2.1，公式 (1)-(2)
- admissible displacement field：Section 2.2，公式 (3)-(5)
- KAN 基本结构：Section 3.1，公式 (6)-(16)
- PIKAN 框架：Section 3.2，Figure 2
- 采样和积分策略：Section 3.3
- PIKAN flowchart：Figure 5
- PIKAN loss：公式 (20)
- 误差指标：公式 (21)-(22)
- PIKAN algorithm：Appendix A Algorithm 1
- 超参数敏感性：Appendix B

---

## 4. 实验设置与评价指标

### 4.1 实验对象

- 方程类型：静态弹性力学 / 多材料结构
- 应用场景：电子封装结构可靠性分析、多材料界面问题
- 数据来源：物理能量泛函训练；FEM 作为参考解
- 是否有 ground truth：无解析解时使用 FEM reference
- 是否和 PowerPINN 9 阶模型相近：不相近。它是静态固体力学问题，不是电力系统 ODE 动态系统。

### 4.2 Baseline 方法

| Baseline | 作用 |
|---|---|
| FEM | reference solution |
| CENN | 多材料/区域分解神经网络方法对比 |
| MLP-based DEM / PINN | 架构对比参考 |
| PIKAN | 本文方法 |

### 4.3 评价指标

| 指标 | 含义 | 我们是否采用 |
|---|---|---|
| L2 Relative Error | 相对误差 | 是，可迁移 |
| MSE / MAE | 预测误差 | 是，PowerPINN 采用 |
| Max Error | 最大误差 | 是 |
| Physics Residual | 本文用能量损失，不是 residual；本项目仍采用 ODE residual | 是，但定义不同 |
| Epoch | 收敛轮数 | 是 |
| Wall-clock Time | 实际训练时间 | 是 |

---

## 5. 论文中最值得我们复用的部分

### 5.1 可直接复用

- KAN 替代 MLP 的架构描述。
- KAN 将可学习激活函数放在边上，而不是节点上的解释。
- B-spline 可学习函数具有局部支撑和分段表达能力的观点。
- 报告参数量、训练时间、L2 error 的实验习惯。
- 作为 Related Work 中“PINN 架构改进”的代表。

### 5.2 需要改造后复用

- KAN 网络模块：可尝试迁移到 PowerPINN 中替代 MLP，但需重新适配 ODE residual。
- 输入归一化：PowerPINN 的 `x0` 和 `t` 也需要归一化，KAN 对输入范围更敏感。
- KAN 超参数：grid size、spline order、depth、width 需要单独调参。

### 5.3 暂时不做

- 不复现电子封装多材料弹性实验。
- 不采用 DEM energy loss。
- 不把 PI-KAN 作为阶段 2 主线。
- 不与 Cauchy 和 DB-PINN 同时融合，避免项目过载。

---

## 6. 和本项目的对应关系

这篇论文在我们项目中的角色：

- [ ] 研究对象基准：PowerPINN
- [ ] 架构改进来源：Cauchy 激活函数 / compleX-PINN
- [ ] 损失优化来源：DB-PINN
- [ ] 优化策略对比：ConFIG
- [x] 网络架构对比：PI-KAN
- [x] 背景与相关工作
- [ ] 其他：

请用一句话说明它在项目中的用途：

> PI-KAN 论文为本项目提供 KAN 架构类相关工作和潜在对比方向，但由于其研究对象不是电力系统动态 ODE，本项目阶段 1 只做架构参考，不作为优先复现对象。

---

## 7. 代码复现信息

**是否有开源代码：** 是  
**代码是否已下载：** 否  
**是否已运行：** 否  
**运行设备：** 联想拯救者 GPU 优先  
**环境依赖：** PyTorch、KAN implementation、数值积分、自动微分  
**主入口文件：** 待下载代码后确认  
**最重要的脚本：** 待确认，重点关注 KAN layer 和 PIKAN loss  
**运行命令：**

```bash
# 待下载原代码后确认
# 本项目如果后续尝试 KAN 对比，可考虑：
# src/models/kan_model.py
# src/models/pikan_baseline.py
```

**目前遇到的问题：**

- 这篇论文的实验对象与本项目差异较大。
- KAN 的 grid size、spline order、归一化等超参数较多。
- KAN 训练计算复杂度可能较高。
- 如果要用于 PowerPINN，必须重写为 ODE residual PINN，而不是 DEM energy loss。

---

## 8. 对论文写作的潜在贡献

这篇论文可以放在我们论文的哪个部分？

- [x] Introduction
- [x] Related Work
- [ ] Method
- [x] Experiment
- [x] Discussion
- [ ] Conclusion

可引用观点：

1. PINN 的网络结构并不局限于 MLP，可以扩展到 KAN 等新型架构。
2. KAN 通过边上的可学习一元函数增强复杂函数表达能力。
3. KAN 可能缓解 MLP 的 spectral bias 或提高对局部复杂特征的表达能力。
4. KAN 的优势伴随更复杂的梯度计算和超参数敏感性。
5. 对本项目而言，KAN 可作为未来架构对比，而 Cauchy activation 是当前更轻量的主线。

---

## 9. 阅读结论

### 9.1 一句话总结

> 这篇 PI-KAN 论文适合作为网络架构类相关工作，但不适合作为本项目阶段 2 的主线复现对象。

### 9.2 对本项目的优先级

- [ ] 极高：必须精读并复现
- [ ] 较高：需要理解并部分复用
- [x] 中等：作为对比或相关工作
- [ ] 较低：仅作为背景参考

### 9.3 下一步行动

- [x] 继续精读
- [ ] 运行代码
- [ ] 抽取核心模块
- [x] 写入 related work
- [ ] 暂时搁置

具体下一步：

```text
1. 在 comparison_methods_plan.md 中将 PI-KAN 列为“架构类高级对比方法”。
2. 在论文 Related Work 中介绍 KAN-based PINN 的思想。
3. 暂不复现电子封装实验。
4. 等 PowerPINN、Cauchy、DB-PINN 主线完成后，再考虑是否实现 KAN-PowerPINN。
```
