# PowerPINN Baseline 规格说明

本文档用于在正式实现 PowerPINN baseline 前，固定 9 阶同步发电机任务的数学定义、数据生成方式、网络输入输出、损失函数、评价指标和实现边界。后续 `src/systems/`、`src/training/`、`configs/` 和 `scripts/` 的代码应以本文档为准。

## 1. 是否需要参考原代码仓库

需要。

PowerPINN 论文正文篇幅较短，部分实现细节没有完整写出，例如：

- 9 个状态变量的精确顺序；
- 具体机器参数、AVR 参数、Governor 参数；
- 初始条件采样范围；
- 数据集生成脚本；
- 网络层数、宽度、激活函数、优化器；
- 损失项权重；
- collocation points 的构造方式；
- 训练/验证/测试划分方式。

因此 baseline 复现不能只依赖论文正文，应同时参考官方仓库：

```text
https://github.com/radiakos/PowerPINN
```

本次已将官方仓库浅克隆到临时目录用于查阅：

```text
/private/tmp/powerpinn-upstream
```

注意：官方仓库只作为实现参考，不应直接整体复制到本项目中提交。我们应抽取必要模型、配置和实验逻辑，改写为本项目已有的配置驱动风格。

## 2. Baseline 目标

目标是在不加入 Cauchy 激活函数、不加入 DB-PINN 动态损失平衡的前提下，复现 PowerPINN 中的 9 阶同步发电机 + AVR + Governor 动态模型。

baseline 应回答：

1. 原始 PowerPINN 在 9 阶系统上的预测误差是多少；
2. 原始静态损失权重下，各损失项是否存在明显不平衡；
3. 训练是否稳定，物理残差是否能有效下降；
4. 后续 Cauchy 和 DB-PINN 改进是否有公平对比基础。

## 3. 参考来源

| 来源 | 用途 |
|---|---|
| `papers/PowerPINN.pdf` | 论文方法、损失函数结构和实验目标 |
| `papers/reading_cards/01_powerpinn_reading_card.md` | 已整理的论文阅读结论 |
| 官方仓库 `src/ode/sm_models_d.py` | 9 阶 ODE 方程和代数电流/电压计算 |
| 官方仓库 `src/conf/setup_dataset.yaml` | 数据生成配置 |
| 官方仓库 `src/conf/setup_dataset_nn.yaml` | 神经网络训练配置 |
| 官方仓库 `src/conf/initial_conditions/` | 初始条件采样范围 |
| 官方仓库 `src/conf/params/` | 机器、AVR、Governor 和系统参数 |
| 官方仓库 `src/nn/nn_actions.py` | 训练损失项、自动微分和优化流程 |
| 官方仓库 `src/nn/nn_dataset.py` | `(x0, t) -> x(t)` 数据格式和 collocation 构造 |

## 4. 动态系统定义

### 4.1 模型类型

采用官方配置中的：

```text
model_flag: SM_AVR_GOV
machine_num: 7
```

该模型由三部分组成：

- 4 阶同步发电机；
- 3 阶 AVR / Exciter；
- 2 阶 Governor；
- 总计 9 个动态状态。

### 4.2 状态变量

官方 ODE 代码 `SynchronousMachineModels.odequations` 中对 `SM_AVR_GOV` 的解包顺序为：

```text
[theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv]
```

其中：

| 序号 | 变量 | 含义 |
|---:|---|---|
| 1 | `theta` | 转子角 / 功角 |
| 2 | `omega` | 转速偏差或角速度相关状态 |
| 3 | `E_d_dash` | d 轴暂态电势 |
| 4 | `E_q_dash` | q 轴暂态电势 |
| 5 | `R_F` | AVR 稳定反馈状态 |
| 6 | `V_r` | AVR 调节器输出 |
| 7 | `E_fd` | 励磁电压 |
| 8 | `P_m` | 机械功率 |
| 9 | `P_sv` | Governor 阀门/伺服状态 |

重要待确认：官方 `modellings_guide.yaml` 和初始条件文件中 `SM_AVR_GOV` 的顺序写为：

```text
[theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_sv, P_m]
```

这与 ODE 函数解包顺序不一致。由于官方默认初值中 `P_sv = P_m = 0.7048`，该问题在默认配置下不会暴露。我们实现 baseline 时应优先采用 ODE 方程顺序：

```text
[theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv]
```

同时在数据加载或初始条件生成中显式检查并记录顺序，避免后续扩展采样范围时把 `P_m` 和 `P_sv` 对调。

## 5. 代数变量

### 5.1 电流计算

官方代码通过代数方程计算 `I_d` 和 `I_q`：

```text
alpha = [[Rs + Re, -(X_q_dash + Xep)],
         [X_d_dash + Xep, Rs + Re]]

beta = [[E_d_dash - Vs * sin(theta - theta_vs)],
        [E_q_dash - Vs * cos(theta - theta_vs)]]

[I_d, I_q]^T = inv(alpha) * beta
```

官方实现中固定：

```text
Rs = 0
Re = 0
Xep = 0.1
```

系统参数来自 `system_bus.yaml`：

```text
Vs = 1
theta_vs = 0
omega_B = 314.1592653589793
```

### 5.2 端电压计算

AVR 模型使用端电压幅值 `V_t`：

```text
V_d = Re * I_d - Xep * I_q + Vs * sin(theta - theta_vs)
V_q = Re * I_q + Xep * I_d + Vs * cos(theta - theta_vs)
V_t = sqrt(V_d^2 + V_q^2)
```

## 6. 9 阶 ODE 方程

以下方程按官方 `odequations` 和 `modelling_method=True` 记录。

### 6.1 同步机方程

```text
dtheta/dt = omega
```

```text
domega/dt = omega_B / (2H) *
            (P_m
             - E_d_dash * I_d
             - E_q_dash * I_q
             - (X_q_dash - X_d_dash) * I_q * I_d
             - D * omega)
```

```text
dE_q_dash/dt = 1 / T_d_dash *
               (-E_q_dash - I_d * (X_d - X_d_dash) + E_fd)
```

```text
dE_d_dash/dt = 1 / T_q_dash *
               (-E_d_dash + I_q * (X_q - X_q_dash))
```

注意：状态顺序中 `E_d_dash` 在 `E_q_dash` 前，但方程常以 `dE_q_dash`、`dE_d_dash` 的物理含义出现。代码返回时必须严格匹配状态顺序：

```text
[dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, ...]
```

### 6.2 AVR / Exciter 方程

```text
dR_F/dt = 1 / T_F * (-R_F + (K_F / T_F) * E_fd)
```

```text
dV_r/dt = 1 / T_A *
          (-V_r
           + K_A * R_F
           - K_A * K_F / T_F * E_fd
           + K_A * (V_ref - V_t))
```

```text
dE_fd/dt = 1 / T_E *
           (-(K_E + 0.098 * exp(0.55 * E_fd)) * E_fd + V_r)
```

### 6.3 Governor 方程

```text
dP_m/dt = 1 / T_ch * (-P_m + P_sv)
```

```text
dP_sv/dt = 1 / T_sv *
           (-P_sv + P_c - (1 / R_d) * (omega / omega_B))
```

## 7. 参数设置

### 7.1 机器参数

官方 baseline 配置使用 `machine_num: 7`，对应 `machine7.yaml`：

| 参数 | 数值 |
|---|---:|
| `D` | 2 |
| `E_fd` | 1 |
| `H` | 5.06 |
| `P_m` | 0.7 |
| `Rs` | 0 |
| `T_d_dash` | 4.75 |
| `T_q_dash` | 1.6 |
| `X_d` | 1.25 |
| `X_d_dash` | 0.232 |
| `X_q` | 1.22 |
| `X_q_dash` | 0.715 |
| `X_d_dash_dash` | 0.2 |
| `T_d_dash_dash` | 0.05 |
| `X_q_dash_dash` | 0.25 |
| `T_q_dash_dash` | 0.04 |

9 阶 `SM_AVR_GOV` 不使用次暂态状态，但文件中仍保留 6 阶模型参数。

### 7.2 AVR 参数

| 参数 | 数值 |
|---|---:|
| `K_A` | 20 |
| `T_A` | 0.2 |
| `K_E` | 1.0 |
| `T_E` | 0.314 |
| `K_F` | 0.063 |
| `T_F` | 0.35 |
| `V_ref` | 1.095 |

### 7.3 Governor 参数

| 参数 | 数值 |
|---|---:|
| `P_c` | 0.7 |
| `R_d` | 0.05 |
| `T_ch` | 0.4 |
| `T_sv` | 0.2 |

### 7.4 系统参数

| 参数 | 数值 |
|---|---:|
| `Vs` | 1 |
| `theta_vs` | 0 |
| `omega_B` | 314.1592653589793 |
| `Rs` | 0 |
| `Re` | 0 |
| `Xep` | 0.1 |

## 8. 数据生成规格

### 8.1 时间设置

官方配置：

```text
time = 1
num_of_points = 1000
```

含义：仿真区间为 `t in [0, 1]`，每条轨迹包含约 1000 个时间点。

论文阅读卡片中记录为 1 ms 时间步长、1 s 仿真周期，约 1000 点/轨迹。

### 8.2 初始条件采样

官方数据生成配置 `setup_dataset.yaml`：

```text
model_flag = SM_AVR_GOV
machine_num = 7
init_condition_bounds = 1
sampling = Lhs
torch = False
```

用于数据集生成的 `init_cond1.yaml`：

| 变量 | 范围 | iterations |
|---|---|---:|
| `theta` | `[-2, 2]` | 10 |
| `omega` | `[-1, 1]` | 10 |
| `E_d_dash` | `[0]` | 1 |
| `E_q_dash` | `[0.9, 1.1]` | 5 |
| `R_F` | `[1]` | 1 |
| `V_r` | `[1.105]` | 1 |
| `E_fd` | `[1.08]` | 1 |
| `P_sv` | `[0.7048]` | 1 |
| `P_m` | `[0.7048]` | 1 |

理论轨迹数：

```text
10 * 10 * 1 * 5 * 1 * 1 * 1 * 1 * 1 = 500
```

因此完整数据集约为：

```text
500 trajectories * 1000 points = 500,000 labeled data points
```

### 8.3 参考解生成

官方数据生成脚本使用 SciPy：

```text
solve_ivp(modelling_full.odequations, t_span, x0, t_eval=t_eval)
```

本项目 baseline 初版建议使用 `scipy.integrate.solve_ivp` 生成参考轨迹，并保存为可复现实验目录。

### 8.4 数据格式

官方训练数据格式为：

```text
input  x = [t, x0_1, x0_2, ..., x0_9]
target y = [x_1(t), x_2(t), ..., x_9(t)]
```

因此：

```text
input_dim = 10
output_dim = 9
```

后续代码中建议把状态变量顺序作为常量保存，例如：

```python
STATE_NAMES = [
    "theta",
    "omega",
    "E_d_dash",
    "E_q_dash",
    "R_F",
    "V_r",
    "E_fd",
    "P_m",
    "P_sv",
]
```

## 9. Collocation Points

官方训练配置 `setup_dataset_nn.yaml`：

```text
new_coll_points_flag = True
num_of_points = 1000
model.init_condition_bounds = 1
model.sampling = Lhs
model.torch = True
```

用于 collocation 的 `nn_init_cond1.yaml` 与数据生成初值类似，但 `E_d_dash` 的 `iterations` 写为 4。不过官方创建初值时如果某个变量范围长度为 1，会强制把 `iterations` 置为 1，因此当前配置下 `E_d_dash=[0]` 仍只生成 1 个取值。

collocation 点格式同样为：

```text
[t, x0]
```

训练时还会取 `t=0` 的 collocation 点构造初始条件损失：

```text
x_train_col_ic = x_train_col[x_train_col[:, 0] == 0]
y_train_col_ic = x_train_col_ic[:, 1:]
```

## 10. 网络结构

官方 baseline 训练配置：

```text
nn.type = DynamicNN
hidden_layers = 4
hidden_dim = 64
activation = tanh
weight_init = xavier_normal
```

官方 `Network` 实现中每层使用 `tanh`：

```text
Linear(input_dim, hidden_dim)
4 个隐藏层配置参数对应动态 hidden ModuleList
Linear(hidden_dim, output_dim)
```

本项目初版应实现为：

```text
MLP(input_dim=10, output_dim=9, hidden_dim=64, num_hidden_layers=4, activation=tanh)
```

注意：官方 `Network.forward` 中循环范围为 `range(self.num_layers)`，而 `hidden` 列表实际包含 `num_layers + 1` 个 Linear，其中最后一个 hidden 可能未使用。为避免复现歧义，本项目先按配置语义实现 4 个隐藏层，并在复现实验报告中记录差异。

## 11. 损失函数

PowerPINN baseline 至少包含四类损失：

### 11.1 数据拟合损失

```text
L_data = MSE(y_pred(t, x0), y_true(t))
```

### 11.2 数据点物理导数损失

在 labeled data points 上计算：

```text
dy_pred/dt = autograd(y_pred, t)
f_true = f(y_true)
L_dt = mean_i MSE(dy_pred_i/dt, f_true_i)
```

官方代码中该项名为 `loss_dt`。

### 11.3 Collocation 物理残差损失

在无标签 collocation points 上计算：

```text
dy_pred_col/dt = autograd(y_pred_col, t)
f_pred_col = f(y_pred_col)
L_pinn = mean_i MSE(dy_pred_col_i/dt, f_pred_col_i)
```

官方代码中该项名为 `loss_pinn`。

### 11.4 初始条件损失

对 `t = 0` 的 collocation points：

```text
L_pinn_ic = MSE(y_pred(0, x0), x0)
```

### 11.5 总损失

官方静态权重配置：

```text
weights = [1, 1e-3, 1e-4, 1e-3]
```

对应：

```text
L_total = 1      * L_data
        + 1e-3   * L_dt
        + 1e-4   * L_pinn
        + 1e-3   * L_pinn_ic
```

baseline 阶段应使用该静态权重，不引入 DB-PINN。

## 12. 训练配置

官方配置摘要：

| 配置项 | 数值 |
|---|---|
| seed | 37 |
| optimizer | LBFGS |
| lr | 0.001 |
| lr_scheduler | No_scheduler |
| loss_criterion | MSELoss |
| num_epochs | 15000 |
| batch_size | None |
| early_stopping | True |
| early_stopping_patience | 2500 |
| early_stopping_min_delta | 1e-6 |
| transform_input | None |
| transform_output | None |
| split_ratio | 0.8 |
| validation_flag | True |
| perc_of_data_points | 1 |
| perc_of_col_points | 1 |

官方 `test_sweep.py` 中若使用 LBFGS，会将：

```text
num_epochs = num_epochs / 10
early_stopping_patience = early_stopping_patience / 10
update_weights_freq = update_weights_freq * 4
```

并设置抽样跳步：

```text
num_of_skip_data_points = 23
num_of_skip_col_points = 19
num_of_skip_val_points = 4
```

本项目建议分两步复现：

1. 小规模调试版：减少初始条件数量、时间点数和 epoch，先验证方程、梯度和日志输出。
2. baseline 对齐版：尽量对齐官方 500 条轨迹、1000 时间点、静态权重和 LBFGS 设置。

## 13. 评价指标

每次 PowerPINN baseline 实验至少记录：

| 指标 | 说明 |
|---|---|
| `l2_relative_error` | 全状态整体相对 L2 误差 |
| `mse` | 全状态 MSE |
| `mae` | 全状态 MAE |
| `max_absolute_error` | 全状态最大绝对误差 |
| `physics_residual_mse` | collocation 或测试点物理残差 MSE |
| `loss_data` | 数据拟合损失 |
| `loss_dt` | 数据点物理导数损失 |
| `loss_pinn` | collocation 物理残差损失 |
| `loss_pinn_ic` | 初始条件损失 |
| `training_time_sec` | 训练耗时 |
| `inference_time_sec` | 推理耗时 |
| `epoch` | 训练轮数 |
| `seed` | 随机种子 |
| `device` | CPU / MPS / CUDA |

## 14. 计划新增文件

建议后续实现时新增：

```text
src/systems/power_generator_9th_order.py
src/training/powerpinn_trainer.py
configs/powerpinn_baseline.yaml
scripts/run_powerpinn_baseline.py
tests/test_power_generator_system.py
docs/powerpinn_code_map.md
docs/powerpinn_model_decomposition.md
```

其中：

- `power_generator_9th_order.py`：只负责 9 阶 ODE、代数电流/电压、采样和参考解生成；
- `powerpinn_trainer.py`：负责数据集加载、PINN 损失、训练、评估和输出；
- `powerpinn_baseline.yaml`：固定所有 baseline 配置；
- `test_power_generator_system.py`：验证状态维度、初始条件维度、ODE 输出维度、无 NaN、简单参考解生成可运行。

## 15. 实现前确认结论

本节记录 baseline 实现前已经确认的决策。后续代码和实验报告应按这些结论执行。

### 15.1 状态顺序

采用官方 ODE 函数中的状态顺序：

```text
[theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv]
```

实验报告中必须说明：官方 `modellings_guide.yaml` 和初始条件文件中最后两个状态写为 `[P_sv, P_m]`，与 ODE 解包顺序 `[P_m, P_sv]` 不一致。由于官方默认初值中 `P_sv = P_m = 0.7048`，该差异在默认配置下不会影响数值结果，但本项目代码必须显式固定顺序，避免后续扩展采样范围时出错。

### 15.2 网络层数

先按 4 个隐藏层实现：

```text
MLP(input_dim=10, output_dim=9, hidden_dim=64, num_hidden_layers=4, activation=tanh)
```

实验报告中记录与官方代码的差异：官方 `Network` 中 `hidden_layers=4` 与实际 `forward` 循环存在一层未使用的可能。本项目先按配置语义实现 4 个隐藏层，保证结构清晰、可解释，后续如需严格对齐官方 bug/行为，可单独做一次复现实验对照。

### 15.3 日志与 WandB

默认关闭 WandB，不把 WandB 作为 baseline 的必要条件。

本项目统一使用本地文件记录：

```text
metrics.csv
loss_history.csv
weight_history.csv
train.log
predictions.csv
model.pt
figures/loss_curve.png
figures/prediction_vs_truth.png
```

这样更适合团队复现和飞书实验记录表同步。后续如果需要 WandB，可作为可选开关加入配置，但默认不启用。

### 15.4 GPU 运行与仓库迁移

正式 PowerPINN baseline 建议放到联想拯救者 Y9000P 的 NVIDIA GPU 上运行。Mac M2 可以继续用于文档、代码开发、小规模调试和单元测试。

迁移原则：

1. 仓库以 GitHub `main` 分支为唯一代码同步源；
2. Codex 对话内容不需要完整迁移到 Y9000P，关键结论已经写入仓库文档和飞书知识库；
3. Y9000P 上只需要拉取 GitHub 仓库、配置 Python/CUDA 环境，然后运行脚本；
4. 每次在 Y9000P 上跑出的实验结果仍按 `experiments/EXP-XXX/` 保存，并同步写入飞书实验记录表。

Y9000P 推荐操作：

```bash
git clone https://github.com/holaimkk/PowerPINN-DB-Cauchy.git
cd PowerPINN-DB-Cauchy
conda env create -f environment.yml
conda activate powerpinn-db-cauchy
python -m pytest
```

如果 Y9000P 上需要 CUDA 版 PyTorch，可能需要单独安装与显卡驱动匹配的 PyTorch。安装后先确认：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

只要飞书知识库和仓库文档保持更新，Codex 对话本身不必迁移；之后在 Y9000P 上继续开发时，可以把当前仓库作为上下文重新打开，并让 Codex 读取 `AGENTS.md`、`README.md`、`docs/powerpinn_baseline_spec.md` 和飞书总结文档。

### 15.5 KAN 依赖策略

官方 PowerPINN 仓库中导入了 `kan`，因为它同时包含 KAN/PI-KAN 相关实验入口。但 PowerPINN baseline 使用的是 `DynamicNN`，不是 KAN。

因此本项目 baseline 阶段不应依赖 `kan` 包：

- 不在 baseline 必需依赖中加入 `kan`；
- 不因为 `kan` 安装失败而阻塞 PowerPINN baseline；
- 后续做 PI-KAN 对比实验时，再单独新增 KAN 依赖和配置；
- 如果复用官方代码片段，应避免在 baseline 模块顶层导入 `kan`，防止未使用的可选依赖导致脚本启动失败。

换句话说，第 5 条的意思是：**PowerPINN baseline 只复现原始 MLP-PINN，不做 KAN；KAN 是后续对比方法，不是当前 baseline 的前置条件。**

## 16. 下一步行动

1. 根据本文档实现 `src/systems/power_generator_9th_order.py`。
2. 添加 `tests/test_power_generator_system.py`，先验证 ODE 维度和数值有限性。
3. 创建 `configs/powerpinn_baseline.yaml`，先设置小规模调试参数。
4. 创建 `scripts/run_powerpinn_baseline.py` 和训练器。
5. 跑通 `EXP-002-debug`，确认日志、指标和图表输出。
6. 扩大到正式 baseline 数据规模，生成 `EXP-002` 并写入飞书实验记录表。
