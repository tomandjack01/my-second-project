# CLAUDE.md

Last updated: 2026-04-05

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

基于 **Directional Diffusion Models (DDM)** (NeurIPS 2023) 的 fMRI 脑连接因果结构学习。原论文 DDM 是图表示学习框架（图/节点分类），本项目将其改造用于从时间序列数据学习**有向因果邻接矩阵**。

**核心任务三元组：**

1. **骨架发现** — 哪些脑区对之间存在连接
2. **方向辨识** — 因果方向 A→B 还是 B→A
3. **检查点选择** — 训练中哪个 epoch 的邻接矩阵最优

## Build & Run Commands

### Environment Setup

```shell
conda create -n ddm python=3.8
conda activate ddm
pip install -r requirements.txt
```

Note: PyTorch, torchvision, and DGL with CUDA 11.3 must be installed separately (see commented lines in requirements.txt).

### Running Experiments

**Graph Classification** (e.g., MUTAG dataset):

```shell
cd GraphExp
python main_graph.py --yaml_dir ./yamls/MUTAG.yaml
```

**Node Classification** (e.g., Photo dataset):

```shell
cd NodeExp
python main_node.py --yaml_dir ./yamls/photo.yaml
```

**Brain Connectivity Structure Learning** (`GraphExp/main_structure_learning.py`):

```shell
# 当前推荐配置（support_direction + causal-lag + composite selector）
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --device cuda --epochs 100 \
    --pretrain_checkpoint ./results/run_20260310_185625/pretrained_encoder.pt \
    --top_k_edges 61 \
    --structure_parameterization support_direction \
    --fixed_support_mask_mode maxgap_kappa \
    --direction_init_mode random \
    --structure_init_mode random \
    --adj_activation sigmoid \
    --directional_prior_mode patel \
    --directional_schedule plateau \
    --directional_kappa_gate \
    --directional_kappa_gate_quantile 0.5 \
    --directional_target_ratio 0.01 \
    --lambda_l1 0.02 \
    --freeze_direction_after_epoch 30 \
    --gradient_routing_mode warmup_then_orthogonal \
    --detach_direction_from_main_after_epoch 23 \
    --causal_lag_main_weight 0.25 \
    --causal_lag_main_lags 1,2 \
    --causal_lag_main_aggregation mean \
    --selection_score_mode causal_lag_composite

# 禁用时序编码器（直接在原始时间序列上扩散）
python main_structure_learning.py --epochs 100 --disable_temporal_encoder
```

## Architecture

### Directory Structure

- **GraphExp/**: 图分类实验 + **fMRI 结构学习主程序**
- **NodeExp/**: 节点分类实验
- **nni_search/**: NNI 超参搜索
- **fMRI_dataset/**: 仿真数据集（fMRI/sim2/sim3/sim4）
- **patel/**: Patel tau/kappa 参考 MATLAB 实现

### Core Model Components

**DDM Model** (`models/DDM.py`):

- `DDM` class: 主扩散模型，支持 `coupled` 和 `support_direction` 两种结构参数化
- `build_noise()`: 构建邻居引导的各向异性噪声（基于固定 noise_guide_adj）
- `sample_q()`: 前向扩散 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon'$
- `node_denoising()`: 去噪损失计算（smooth_l1 + cosine hybrid）
- `Denoising_Unet`: GraphConv/GCN-based U-Net 去噪网络（encoder-decoder with skip connections）
- `CausalConv1d`: 因果一维卷积（left-padding + truncation）
- `NodeSpecificTemporalEncoder`: 因果膨胀时序编码器（可选，默认开启）

**结构参数化模式：**

1. **`coupled` 模式**（原始）：单一 `sender @ receiver.T + bias` → sigmoid → 邻接矩阵
2. **`support_direction` 模式**（当前主力）：
   - **支持分支**（对称）：`support_logits = 0.5 * (sender @ receiver.T + bias + transpose)`
   - **方向分支**（非对称）：`direction_logits = dir_sender @ dir_receiver.T`
   - **组合**：`adj[i,j] = sigmoid(support_logits[i,j]) × sigmoid(D[i,j] - D[j,i])`
   - 方向门控 `sigmoid(D - D^T)` 保证 `gate[i,j] + gate[j,i] = 1`

### Noise Construction（前向扩散噪声）

`build_noise()` 构建邻居引导的各向异性噪声：

- `noise_guide_adj`：固定的对称先验（来自 Patel kappa 或 Pearson），`register_buffer` 无梯度
- 噪声统计量：$\mu_i = \sum_j A_{\text{guide}}[i,j] \cdot x[j]$，$\sigma_i = \sqrt{\text{Var}_{\text{neighbors}}(x)}$
- 默认 `noise_zero_mean=True`：丢弃邻居均值偏置
- 可选 `preserve_noise_sign`：保持噪声与数据同象限

**关键限制**：noise_guide_adj 是静态 buffer，去噪损失对其无梯度。扩散主损失对 $A[i,j]$ vs $A[j,i]$ 的偏序**完全无感**——方向学习完全依赖外部辅助损失。

### Training Pipeline

#### 1. 编码器预训练（可选）

- 自回归预训练：预测 $t+1$ 时刻的值
- 预训练后冻结编码器（requires_grad=False, eval mode）
- 可用 `--pretrain_checkpoint` 加载已有权重

#### 2. 结构学习主训练循环

**主损失**：去噪重建损失（smooth_l1 + cosine）+ L1 稀疏正则

**辅助损失栈（全部可选，通过 CLI 开关控制）：**

| 损失                   | 参数                             | 作用                           |
| ---------------------- | -------------------------------- | ------------------------------ |
| Patel 方向 margin loss | `--directional_prior_mode patel` | 从 Patel tau 先验学方向        |
| Kappa 门控             | `--directional_kappa_gate`       | 仅对高置信度节点对施加方向监督 |
| Parent entropy         | `--parent_entropy_lambda`        | 精度导向剪枝（降低入度熵）     |
| Parent cap (hinge)     | `--parent_cap_lambda`            | 限制有效父节点数上限           |
| Ungated symmetry reg   | `--ungated_symmetry_lambda`      | 抑制门控外残余假阳性           |
| **Causal-lag main**    | `--causal_lag_main_weight`       | 滞后重建损失（方向敏感去噪）   |

**Causal-lag main loss**（最新机制，2026-03-29 引入）：

- 每个节点的未来由其候选父节点的过去预测
- `adj_causal[cause, effect]` 决定聚合权重
- 正确方向的预测误差 < 反向 → 方向梯度信号
- 支持多滞后步 `--causal_lag_main_lags 1,2`
- 聚合模式：`mean`（当前最优）或 `softmax`

#### 3. 梯度路由系统

`--gradient_routing_mode` 控制不同损失更新哪些参数分支：

| 模式                     | 主损失/正则                                  | Causal-lag                                   |
| ------------------------ | -------------------------------------------- | -------------------------------------------- |
| `legacy`                 | 支持+方向（detach epoch 前）                 | 支持+方向                                    |
| `orthogonal`             | 仅支持                                       | 仅方向                                       |
| `warmup_then_orthogonal` | epoch < detach: 联合; epoch ≥ detach: 仅支持 | epoch < detach: 联合; epoch ≥ detach: 仅方向 |

**当前推荐**：`warmup_then_orthogonal` + `detach_direction_from_main_after_epoch=23`

#### 4. 方向分支冻结

`--freeze_direction_after_epoch N`：第 N epoch 后冻结 direction_emb_sender/receiver，方向不再更新，仅支持分支继续学习。

#### 5. Best-Epoch 选择

`compute_epoch_quality()` 支持 4 种评分模式：

| 模式                           | 主信号                                | 特点         |
| ------------------------------ | ------------------------------------- | ------------ |
| `legacy`                       | 骨架重叠 + Patel 一致 + 密度 + margin | 原始启发式   |
| `causal_lag_composite`         | causal-lag forward/reverse 差异       | **当前推荐** |
| `causal_lag_entropy_composite` | + 跨被试稳定性                        | 扩展版       |
| `causal_lag_primary`           | causal-lag 主导 + 弱 tiebreak         | 激进版       |

所有模式均通过 `evaluate_selection_guardrails()` 进行保守筛选（骨架保留、密度比等）。

### Raw vs Causal Convention

- **Raw**（内部）：`A[effect, cause]` — 去噪时 GNN 消息流方向
- **Causal**（外部）：`A[cause, effect]` — 可解释的因果方向
- 转换：`causal = raw.T`
- 选择器和 GT 评估均使用 causal convention

### Evaluation

**Directional Edge Evaluation** (`GraphExp/test_eval.py`):

- 对每个无向对 (i,j)，比较 adj[i,j] vs adj[j,i] 确定方向
- 按权重排序截断到 top-k
- 报告 directed Precision/Recall/F1
- 支持 margin deadzone 过滤（`margin_eps`）
- Usage: `python test_eval.py --gt ../fMRI_dataset/h4.txt --top_k 61`

**Selector Audit**（训练时可选）：

- `--selector_audit_gt_path ../fMRI_dataset/h4.txt`
- 每 epoch 输出 strict_f1、margin stats、failure mode 等 GT 对照信息

### Datasets

| 文件     | 节点数 | GT 边数 | GT 文件 | 真实密度 |
| -------- | ------ | ------- | ------- | -------- |
| fMRI.csv | 5      | 5       | h1.txt  | 50%      |
| sim2.csv | 10     | 11      | h2.txt  | ~24%     |
| sim3.csv | 15     | 18      | h3.txt  | ~17%     |
| sim4.csv | 50     | 61      | h4.txt  | ~5%      |

所有数据集：50 subjects × 200 time points。

### Key Dependencies

- DGL (Deep Graph Library) for graph neural networks
- PyTorch for deep learning
- scikit-learn for SVM evaluation
- OGB for ogbn-arxiv dataset

---

## 核心实验结论（截至 2026-04-05）

### 扩散模型与方向学习的根本关系

**结论：扩散主损失（去噪重建）对因果方向完全无感。**

证据链：

1. random init 下所有 margin → 0，扩散损失不提供翻转方向的梯度
2. Phase 0 多步滞后诊断：离线预测不对称性太弱（sim4 方向准确率仅 0.52，阈值 0.70）
3. Phase 0B 梯度探针：去噪损失在各向同性噪声下不产生方向偏好
4. Phase 0C 边消融探测：去噪器不含可用的方向信息

原因分析：

- `noise_guide_adj` 是固定 buffer，无梯度
- 去噪损失是标量重建误差，翻转 adj[i,j] 与 adj[j,i] 不改变重建质量
- 原论文 DDM 场景是**已知固定图结构**上的表示学习，方向噪声用于保留 SNR；我们的场景是**学习图结构本身**，存在循环依赖

### 已证伪的方向（不要再试）

| 方向                                      | 结论                             | 关键证据                                         |
| ----------------------------------------- | -------------------------------- | ------------------------------------------------ |
| 纯扩散主损失自主发现方向                  | ❌ 不可行                        | random init margin 全趋近 0                      |
| cross-pred v1（邻接加权聚合预测）         | ❌ 信号被稀释                    | pred_cos ≈ 0.85-0.99，方向梯度消失               |
| 更强 L1 做稀疏化                          | ❌ sigmoid 下只均匀缩放          | effective parents 几乎不变                       |
| lag-1 交叉相关替代 Patel                  | ❌ 信号太弱                      | 55% vs 随机 50%                                  |
| cosine anneal 调度                        | ❌ final epoch 权重退火到 0      | 改用 plateau                                     |
| Residual Patel fusion（持久先验偏置）     | ❌ 中性到负面                    | sim4 负迁移                                      |
| Time-supervised anti-collapse signed_gate | ⚠️ sim3 内部成功但 sim4 导出折叠 | 内部方向解未能通过 support × direction_gate 保留 |
| patel_score 非对称初始化                  | ❌ 冗余                          | random init + 训练期监督已超过                   |

### 当前有效机制

| 机制                                     | 效果                                                    |
| ---------------------------------------- | ------------------------------------------------------- |
| `support_direction` 参数化分解           | 解耦支持（对称）和方向（非对称）学习                    |
| `fixed_support_mask_mode = maxgap_kappa` | **不可去除** — 去掉后支持选择性崩溃                     |
| Patel tau margin loss + kappa gate       | 从对称初始化学到方向，gate 防止幻觉边                   |
| Parent cap (hinge)                       | 限制有效父节点数，90%+ 删的是幻觉边                     |
| Ungated symmetry reg                     | 压制门控外残余假阳性                                    |
| Plateau 调度（不退火）                   | 辅助损失在整个训练过程中保持活跃                        |
| **Causal-lag main loss**                 | **最新最强信号** — 滞后重建差异提供方向敏感梯度         |
| **Gradient routing (warmup→orthogonal)** | 前期联合学习，后期正交分离支持与方向                    |
| **Causal-lag composite selector**        | **修复选择器对齐** — proxy vs GT 相关性从 -0.09 → +0.71 |

### 当前最优配置（sim4, 40-epoch pilot, seed=11）

```yaml
# 结构参数化
structure_parameterization: support_direction
fixed_support_mask_mode: maxgap_kappa
structure_init_mode: random # Patel init 已证明冗余
direction_init_mode: random
adj_activation: sigmoid

# 方向监督
directional_prior_mode: patel
directional_schedule: plateau
directional_kappa_gate: true
directional_kappa_gate_quantile: 0.50

# 梯度路由
gradient_routing_mode: warmup_then_orthogonal
detach_direction_from_main_after_epoch: 23
freeze_direction_after_epoch: 30

# Causal-lag（新机制）
causal_lag_main_weight: 0.25
causal_lag_main_lags: 1,2
causal_lag_main_aggregation: mean

# 正则化
lambda_l1: 0.02
directional_target_ratio: 0.01
main_loss_weight: 1.0

# 选择器
selection_score_mode: causal_lag_composite
selection_soft_agreement_weight: 0.20
selection_causal_lag_weight: 1.0
selection_margin_penalty_weight: 0.05
```

### 最新实验结果（2026-03-30 ~ 2026-04-05）

#### Causal-lag main branch（sim4, seed=11）

| 指标                     | 40-epoch Baseline | 40-epoch Causal-Lag (mean) |
| ------------------------ | ----------------- | -------------------------- |
| Best GT strict-F1        | 0.8197            | **0.8689** (+0.049)        |
| Final strict-F1          | 0.7705            | **0.8033** (+0.033)        |
| Forward > Reverse epochs | N/A               | **35/40**                  |

#### Composite selector 修复效果

| 对比               | Legacy Selector    | Composite Selector  |
| ------------------ | ------------------ | ------------------- |
| sim4 导出 epoch    | epoch 9 (F1=0.803) | epoch 26 (F1=0.869) |
| Proxy vs GT 相关性 | -0.088             | **+0.709**          |

#### Support ablation 关键结论（2026-04-03 ~ 2026-04-05）

| 消融项                                             | 结论                                        |
| -------------------------------------------------- | ------------------------------------------- |
| `structure_init_mode = patel_kappa` → `random`     | ✅ 冗余，random init 不损失甚至略优         |
| `kappa_logit_bias_scale` 持久先验                  | ✅ 冗余，近中性                             |
| `fixed_support_mask_mode = maxgap_kappa` → `none`  | ❌ **不可去除**，去掉后支持选择性崩溃       |
| `support_prior_mode = patel_kappa` → `pearson_abs` | ⚠️ 在 maxgap 算子下产生相同骨架，差异未测到 |

### 核心瓶颈（截至 2026-04-05）

1. 🔴 **导出空间方向保留** — 内部方向解正确但 `support × direction_gate` 导出近似对称（sim4）
2. 🟠 **晚期漂移** — 完美 epoch 出现后质量回退（fMRI.csv: best=1.0 → final=0.8）
3. 🟢 **选择器对齐** — 已通过 causal-lag composite 基本修复

### 已关闭的实验线

详细日志见对应文件：

- Cross-Pred V1 全系列 → `GraphExp/CROSS_PRED_V1_TRACKER.md`
- Option B（扩散侧自主方向学习） → `plan.md`
- Residual Patel fusion → `plan.md`
- Time-supervised anti-collapse → `experiment_synthesis.md`

### 当前活跃实验线（遵循 `constrict.md` 约束）

**机制线**：

- 目标：导出空间方向保留
- 候选：exported-gate floor、exported-margin hinge、support-preservation constraint
- 规则：不向主训练循环添加新辅助损失，与冻结主线对比

**选择线**：

- 目标：选择器权重跨数据集稳定性
- 候选：sim3/sim4 多种子确认、composite 权重微调

### 关键代码路径

- 结构参数化初始化：`DDM.py:300-330`（SVD + support/direction 分支创建）
- 支持 logits（对称强制）：`DDM.py:366-372`（`get_structure_logits`）
- 方向 logits（非对称）：`DDM.py:374-382`（`get_direction_logits`）
- 最终邻接矩阵组合：`DDM.py:384-415`（`get_structure_adj`，support × direction_gate）
- 噪声构建：`DDM.py:581-651`（`build_noise`）
- raw/causal 转换：`main_structure_learning.py`（`to_causal_matrix_torch` = transpose）
- 梯度路由：`main_structure_learning.py:720-786`（`build_epoch_gradient_routing`）
- Causal-lag 主损失：`main_structure_learning.py:1754`（`compute_causal_lag_main_loss`）
- 方向 margin loss：`main_structure_learning.py:1075`（`compute_directional_margin_loss`）
- Best-epoch 评分：`main_structure_learning.py:1947`（`compute_epoch_quality`，4 种模式）
- 选择守卫规则：`main_structure_learning.py:2254`（`evaluate_selection_guardrails`）
- 实验 runner：`GraphExp/run_cross_pred_v1_final_only_compare.py`

### 文档索引

| 文件                                | 内容                                             |
| ----------------------------------- | ------------------------------------------------ |
| `experiment_synthesis.md`           | 全实验历程综合整理与后续方向                     |
| `constrict.md`                      | 主线冻结策略 + 机制线/选择线拆分日志             |
| `plan.md`                           | Causal-lag 扩散计划（含 Phase 0/0B/0C 诊断结果） |
| `ablation.md`                       | Support learning 消融实验记录                    |
| `GraphExp/CROSS_PRED_V1_TRACKER.md` | Cross-prediction v1 全系列追踪                   |
| `Structure learning change log.md`  | 代码变更日志                                     |

### 分析时的自省清单

1. 追踪损失项的完整生命周期（初始值 → 调度 → 评估时刻的实际权重）
2. 区分 "信号弱" vs "信号被调度关掉了"
3. 不要从单一实验推导过强结论
4. 矩阵运算语义必须用 2×2 例子手动验证（raw/causal convention 容易搞混）
5. 审视自己的方案要像审视别人的一样严格
6. 设计多辅助损失方案时，逐一检查每个损失的调度路径
7. smoke 结论不能直接外推到 formal（λ 与训练长度有交互）
8. 内部方向解正确 ≠ 导出邻接矩阵方向正确（导出空间保留是当前主瓶颈）
