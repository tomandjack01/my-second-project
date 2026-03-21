# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official implementation of **Directional Diffusion Models (DDM)** from NeurIPS 2023. DDM introduces data-dependent, anisotropic, and directional noises in the forward diffusion process for graph representation learning. The codebase supports both graph classification and node classification tasks.

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

### Hyperparameter Search with NNI
```shell
cd nni_search
python run_search.py
```
The NNI web UI runs on port 6006. Hyperparameter search is recommended for best results.

## Architecture

### Directory Structure
- **GraphExp/**: Graph classification experiments (IMDB-B, IMDB-M, COLLAB, REDDIT-B, PROTEINS, MUTAG)
- **NodeExp/**: Node classification experiments (Cora, Citeseer, PubMed, ogbn-arxiv, Amazon-Computer, Amazon-Photo)
- **nni_search/**: NNI-based hyperparameter search configuration

### Core Model Components

**DDM Model** (`models/DDM.py`):
- `DDM` class: Main diffusion model with configurable beta schedules (linear, quad, const, jsd, sigmoid)
- `sample_q()`: Forward diffusion process with directional noise (noise aligned with data distribution via mean/std matching and sign preservation)
- `embed()`: Extract representations at specified timestep T for downstream evaluation
- `Denoising_Unet`: U-Net style denoising network using GAT layers with skip connections
- `CausalConv1d`: Causal 1D convolution with left-padding + truncation, ensures no future information leakage
- `NodeSpecificTemporalEncoder`: **Causal dilated temporal encoder** with autoregressive pretrain head (can be disabled with `use_temporal_encoder=False`)
  - Architecture: 3-layer causal dilated conv (dilation 1-2-4) → projector (Conv1d H→1) → LayerNorm
  - `forward()`: `[N, T]` → `[N, T]` (same dimension, no compression)
  - `pretrain_forward()`: Autoregressive self-supervised loss (predict t+1 from t), uses unnormalized features + `pred_head` to avoid LayerNorm scale destruction
  - When disabled: Model works directly on raw time series without encoding

**Encoder Pretraining** (`GraphExp/pretrain_temporal_encoder.py`):
- Legacy three-objective pretrain module (reconstruction + forecasting + VICReg)
- No longer imported by `main_structure_learning.py` — replaced by native autoregressive pretrain
- Kept for reference / standalone use

**Denoising Network** (`models/mlp_gat.py`):
- `Denoising_Unet`: Encoder-decoder architecture with down/up GAT layers
- `MlpBlock`: Residual MLP blocks with configurable normalization and activation
- Time embeddings are added at each layer

### Training Pipeline

1. Forward diffusion: Add directional noise at random timestep t
2. Denoising: Predict original features from noisy features using U-Net
3. Loss: Cosine similarity loss with configurable alpha power

#### Auxiliary Losses (Directional Prior & Feature Decoupling)

`main_structure_learning.py` adds two auxiliary losses on top of the base diffusion loss:

**`compute_directional_margin_loss(logits, patel_matrix)`**:
- Encourages learned edge directions to agree with Patel's Tau prior
- Both `q_threshold` and `margin` are fully adaptive (no hardcoded constants):
  - `q_threshold` = median of nonzero `|delta_P|` → ~50% of edges participate
  - `margin` = 25th percentile of `sign(delta_P) * D` on active edges → strictly 25% violation rate
- Weighted by Patel confidence: `w = (|delta_P| > q_threshold) * |delta_P|`

**`compute_feature_ortho_loss(S, R)`**:
- Decouples sender/receiver embedding spaces via cross-covariance Frobenius norm
- Prevents sender ≈ receiver collapse that would produce symmetric adjacency

**Ratio-adaptive lambda with stability controls (no warmup):**
- No warmup delay: Patel SVD init already provides directional structure, auxiliary losses protect it from epoch 0
- Target ratios: Dir loss = 1% of main loss, Ortho loss = 0.5%
- EMA smoothing (alpha=0.1) + ±10% step-change cap on lambda values to prevent ratio-adaptive jitter

#### Structure Learning Initialization

- `init_features` uses **Patel matrix** (asymmetric) instead of Pearson (symmetric) for SVD-based sender/receiver embedding init. This gives directional bias from the start (U ≠ V).
- `adj_bias_init` = `logit(target_density)` with safety clamp: `min(61/N(N-1), 0.95)` to prevent math domain error on small graphs (N < 8).

#### Training/Export Consistency

Log printing and final adjacency export both call `model._get_structure_graph(device)`, ensuring `adj_bias`, `clamp(-6, 6)`, and diagonal masking are applied identically to training. Previously, export used raw `sigmoid(sender @ receiver.T)` without bias/clamp, causing systematic weight mismatch.

#### Best-Epoch Selection

Training no longer blindly returns the last epoch's adjacency. Instead:

**`compute_epoch_quality(adj_np, patel_matrix_cpu, top_k)`** scores each checkpoint:
- `score = agreement × dir_margin × density_factor × skeleton_overlap`
- `agreement`: direction match rate on HIGH-CONFIDENCE Patel edges only (excludes ties)
- `dir_margin`: mean `|adj[i,j] - adj[j,i]|` of top-k edges (asymmetry strength)
- `density_factor`: Gaussian penalty `exp(-0.5 * log(ratio)²)` on density deviation
- `skeleton_overlap`: fraction of model's top-k edges that also appear in Patel's top-k
- `top_k` auto-computed: `max(10, int(N*(N-1)*0.05/2))`

The best-scoring epoch's adjacency is returned. `train_brain_connectivity` returns 5 values: `(model, adj_matrix, loss_history, collapse_history, best_epoch)`.

### Encoder Pretraining Pipeline (Autoregressive Causal)

The `NodeSpecificTemporalEncoder` collapses when trained end-to-end with diffusion (cosine sim→1.0, Diff Loss→0.0). Solution: pretrain encoder with autoregressive objective, then freeze.

**Autoregressive pretrain loss:**
- Uses `pretrain_forward()`: predict time step t+1 from t via MSE on unnormalized features
- Bypasses LayerNorm to preserve signal scale for the prediction target
- `pred_head` (Linear) maps encoder output to prediction space

**Integration flow (`main_structure_learning.py`):**
1. Create DDM model with Patel matrix as `init_features` (asymmetric SVD init for directed edges)
2. Autoregressive pretrain: iterate all subjects, call `pretrain_forward()`, accumulate gradients
3. Freeze `model.temporal_encoder` (requires_grad=False, eval mode)
4. Rebuild optimizer with only unfrozen parameters
5. Diffusion training with auxiliary losses (directional margin + feature orthogonality)
6. Best-epoch selection via `compute_epoch_quality()` (Patel-based proxy, no GT needed)

**Brain Connectivity Structure Learning** (`GraphExp/main_structure_learning.py`):
```shell
# Full pipeline (pretrain + freeze + diffusion)
python main_structure_learning.py --epochs 100 --pretrain_epochs 50

# Skip pretrain (original behavior)
python main_structure_learning.py --epochs 100 --skip_pretrain

# Load existing pretrained weights
python main_structure_learning.py --pretrain_checkpoint ./results/run_xxx/pretrained_encoder.pt

# Disable temporal encoder (work directly on raw time series)
python main_structure_learning.py --epochs 100 --disable_temporal_encoder

# Standalone pretrain
python pretrain_temporal_encoder.py --epochs 50 --save_path ./pretrained_encoder.pt
```

**Collapse diagnostics** (`diagnose_encoder_collapse()`): Healthy indicators after pretrain:
- `effective_rank` > 5 (ideally > 10)
- `mean_cosine_sim` < 0.5 (ideally < 0.3)
- `dead_dims_ratio` = 0%
- `feature_std_mean` > 0.1

### Temporal Encoder Control

The DDM model supports **optional temporal encoding** via causal dilated convolutions.

**Two Operating Modes:**

1. **With Temporal Encoder (Default):**
   - Raw data `[N, T]` → `temporal_encoder` → Causal features `[N, T]` → Diffusion
   - Output dimension = input dimension (no compression, preserves full temporal info)
   - Requires autoregressive pretraining to avoid encoder collapse
   - Causal convolutions enforce physical time ordering (no future leakage)

2. **Without Temporal Encoder:**
   - Raw data `[N, T]` → Directly to Diffusion → Output `[N, T]`
   - No pretraining needed
   - No causal inductive bias

**Usage:**

```shell
# Default: causal encoder with autoregressive pretrain + freeze + diffusion
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --epochs 100 \
    --pretrain_epochs 50

# Disable temporal encoder (direct diffusion on raw time series)
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --epochs 100 \
    --disable_temporal_encoder
```

**Implementation Details:**
- `DDM.__init__()`: `use_temporal_encoder` parameter (default: `True`)
- When disabled: `temporal_encoder = None`, denoising network input/output dim = `in_dim`
- When enabled: `temporal_encoder` active, denoising network input/output dim = `in_dim` (same — no dimension reduction)
- Diffusion always operates in original time dimension space (`denoising_in_dim = in_dim`)
- Pretraining, freezing, and collapse diagnostics only run when encoder is enabled

### Evaluation

**Directional Edge Evaluation** (`GraphExp/test_eval.py`):
- Loads learned adjacency and transposes it (GNN denoising convention: `adj[effect, cause]` → causal convention: `adj[cause, effect]`)
- For each undirected pair (i,j), picks direction by comparing `adj[i,j]` vs `adj[j,i]`
- Ranks edges by absolute weight (stronger direction), truncates to top-k or sparsity %
- Reports directed Precision/Recall/F1 against ground truth DAG
- Usage: `python test_eval.py --gt ../fMRI_dataset/h4.txt --top_k 61`
- Output fields: `weight` = `max(adj[i,j], adj[j,i])`, `margin` = `|adj[i,j] - adj[j,i]|`

**Graph Classification** (`GraphExp/evaluator.py`):
- Extract embeddings at multiple timesteps (eval_T)
- Pool graph representations (mean/sum/max pooling)
- Train SVM classifier with GridSearchCV
- 10-fold stratified cross-validation

**Node Classification** (`NodeExp/evaluator.py`):
- Extract embeddings at multiple timesteps
- Train linear probe (LogisticRegression) for classification
- Ensemble predictions across timesteps using mode voting

### Configuration (YAML files)

Key hyperparameters in yaml configs:
- `MODEL.T`: Number of diffusion timesteps
- `MODEL.beta_schedule`: Noise schedule type
- `MODEL.beta_1`, `MODEL.beta_T`: Beta bounds
- `MODEL.num_hidden`, `MODEL.num_layers`, `MODEL.nhead`: Network architecture
- `eval_T`: List of timesteps for evaluation embedding extraction
- `seeds`: Random seeds for multiple runs

Pretrain CLI parameters (`main_structure_learning.py`):
- `--pretrain_epochs`: Number of autoregressive pretrain epochs (default: 50)
- `--pretrain_lr`: Pretrain learning rate (default: 1e-3)
- `--skip_pretrain`: Skip pretraining entirely (equivalent to `--pretrain_epochs 0`)
- `--pretrain_checkpoint`: Path to load existing pretrained encoder weights
- `--disable_temporal_encoder`: Disable temporal encoder and work directly on raw time series (skips all pretraining)
- `--lambda_l1`: L1 sparsity coefficient (default: 0.02, normalized by N²)
- ~~`--pretrain_split_ratio`~~: Deprecated (autoregressive pretrain does not need split)

### Ablation Script (`GraphExp/run_temporal_encoder_ablation.py`)

Compares temporal encoder variants (full / no_temporal_stack / reduced_receptive_field) across multiple seeds.
- `build_noise_guide_adj()` returns `(noise_guide_adj, patel_matrix)` tuple
- Calls `train_brain_connectivity` with `patel_matrix` (required param) and unpacks 5 return values
- Outputs: `raw_runs.csv`, `summary.csv`, `meta.json`

### Key Dependencies
- DGL (Deep Graph Library) for graph neural networks
- PyTorch for deep learning
- scikit-learn for SVM evaluation
- OGB for ogbn-arxiv dataset

---

## 方向约束实验记录（2026-03-11 ~ 2026-03-14，branch: direction_constraint）

详细实验日志见 `GraphExp/CROSS_PRED_V1_TRACKER.md`。以下是关键结论和后续方向的摘要。

### 核心问题诊断

**问题：Patel score 通过初始化主导了模型的方向学习，训练过程本身不改变方向。**

证据链：
1. `init_scale` 消融（fMRI 5节点）：patel_score 初始化下，训练前后邻接矩阵 Spearman > 0.98，方向完全由初始化决定
2. `patel_score_t`（方向镜像）实验：镜像初始化也保持 Spearman > 0.98，证明训练不纠正方向
3. random init + final-only 实验：所有 margin → 0，扩散主损失对方向完全无感

根本原因（三个叠加）：
- `init_features`（patel_score）直接进入 `DDM.py:224` 的 SVD，`emb_dim=N` 时是全秩初始化，`sender @ receiver.T` 精确重构初始矩阵
- `DDM.py:232-236` 的 rescale 把初始 logit std 拉到 ~1.0，放大了初始化的方向信号
- 扩散主损失（diffusion + L1 + hub）全部是方向无关的，不提供翻转方向的梯度

### 已证伪的方向（不要再试）

| 方向 | 结论 | 关键证据 |
|------|------|---------|
| 纯扩散主损失自主发现方向 | 不可行 | random init final epoch margin 全趋近 0 |
| cross-pred v1（邻接加权聚合预测） | 信号被稠密均匀图稀释 | pred_cos ≈ 0.85-0.99，方向梯度消失 |
| cross-pred + softmax 聚合 | 机制有效但结果不改善 | 尾部 margin 上升但 F1 不变 |
| 更强 L1 做稀疏化 | sigmoid 参数化下只均匀缩放 | adj_eff_parents 从 3.99 到 3.97，几乎不变 |
| 更大图自然救活 v1 | 平均化更严重 | sim4 50节点 agg_eff_par ≈ 48.77 |
| lag-1 交叉相关替代 Patel | 信号太弱（55% vs 随机50%） | 同一 margin loss 下明显弱于 Patel |
| cosine anneal 调度下评估辅助损失 | final epoch 权重已退火到 0 | weighted_cross epoch 100 = 0.0000 |

### 已确认有效的机制

| 机制 | 效果 | 证据 |
|------|------|------|
| Patel tau margin loss（方向先验） | 强方向信号，能从对称初始化学到方向 | sim3 formal: margin_median=0.858（从 patel_kappa 对称 init） |
| kappa-gated margin（方向监督范围控制） | 阻止 margin loss 在非 GT 对上制造幻觉边 | FP margin 从 0.995 → 0.024，GT margin 不受影响 |
| parent entropy（精度导向剪枝） | 砍掉低置信度预测边，提升 precision | sim3 formal: strict_prec 从 0.14 → 0.20 |
| parent cap (hinge)（有效父节点数约束） | 限制预测边总数，主要砍幻觉边 | 删 39 边中 37 条幻觉 + 2 条真边（94.9% 精度） |
| ungated symmetry reg（残余 FP 抑制） | 把 gate 外残余 FP 的 margin 压向 0 | FP margin 从 0.024 → 0.010，GT margin 维持 0.995 |
| deadzone (margin_eps)（评估截断） | 利用 margin 双峰分布过滤噪声预测 | eps=0.05 时 strict_f1 从 ~0.28 → ~0.48（smoke） |
| plateau 调度（不退火） | 消除 final-only 评估的调度混淆 | plateau 下辅助损失在 epoch 100 保持非零 |
| patel_kappa 对称初始化 | 解耦初始化与方向学习 | 对称 init + 训练期监督 > 非对称 init 硬编码 |

### 已验证的最佳方案（2026-03-16，sim3 15节点，5 seeds 100 epochs formal）

**里程碑结论：**
1. patel_score 非对称初始化不是不可替代的。对称初始化 + 训练期监督已超过 legacy 方案。
2. 假阳性的主要来源是"非 GT 节点对的高置信幻觉边"（96% FP），不是互逆边或方向翻转。
3. 通过 kappa-gated margin + cap + symmetry reg，训练端已成功将真边和幻觉边在 margin 维度上分离。
4. sym=0.5 在 100 epoch 下会把 GT margin 也压塌（gt_margin_median 从 0.80 → 0.06），smoke 结论不能直接外推到 formal。
5. 当前 formal 最优：gated+cap+sym=0.5 配 eps=3e-4，strict_f1=0.3864，5/5 seeds 全胜。

#### 当前 Best Setting（formal 已验证）

```
# 初始化与方向
structure_init_mode = patel_kappa         # 对称骨架初始化
directional_prior_mode = patel            # Patel tau margin loss
directional_schedule = plateau            # 不退火
directional_kappa_gate = True             # 只在高 kappa 对上施加方向监督
directional_kappa_gate_quantile = 0.50    # kappa 中位数以上才 gate

# 结构剪枝
parent_entropy_lambda = 0.3              # 精度导向剪枝（已验证 0.25-0.35 平台）
parent_entropy_warmup_epochs = 10
parent_entropy_ramp_epochs = 10
parent_cap_lambda = 0.50                 # hinge cap，限制有效父节点数
parent_cap_target = 2.5                  # 目标有效父节点数

# 残余 FP 抑制
ungated_symmetry_lambda = 0.5            # 对 gate 外节点对施加对称正则

# 评估（保守 deadzone，零额外 TP 损失）
margin_eps = 3e-4                        # 保守 deadzone，不丢任何当前已恢复的真边
```

#### 机制链：每个组件解决什么问题

| 机制 | 解决的问题 | 关键证据 |
|------|-----------|---------|
| Patel margin loss | 从对称 init 学方向 | margin_median 从 ~0 → ~0.99 |
| kappa gate | 阻止 margin loss 在非 GT 对上制造幻觉边 | FP margin 从 0.995 → 0.024 |
| parent cap (hinge) | 限制预测边总数 | pred_count 从 105 → ~66，94.9% 删的是幻觉边 |
| ungated symmetry reg | 把残余 FP 的 margin 压向 0 | FP margin 从 0.024 → 0.010 |
| deadzone (eps=0.05) | 评估端利用 margin 分离信号 | strict_f1 从 0.27 → 0.48 |

机制分工：Patel margin 管方向，kappa gate 管监督范围，cap 管总量，symmetry reg 管残余噪声，deadzone 管评估截断。

#### 假阳性诊断（cap=0.25@2.5 smoke, seed=11）

- 57.4 条 strict 预测边中：11.4 TP + 46.0 FP
- FP 构成：44.2 条非 GT 对幻觉（96.1%）+ 1.8 条方向翻转（3.9%）
- 幻觉边特征：median margin = 0.995，极其自信 → 被 margin loss 主动推高
- kappa gate 后：幻觉边 margin 降到 ~0.01，真边 margin 维持 ~0.99

#### Formal 对照结果（sim3, 5 seeds, 100 epochs）

| 配置 | eps | strict_f1 | strict_prec | strict_recall |
|------|-----|-----------|-------------|---------------|
| patel_score init + dir_off（legacy baseline） | 0 | 0.2033 | 0.1192 | 68.89% |
| patel_score init + dir_patel（legacy + 方向监督） | 0 | 0.2309 | 0.1352 | 78.89% |
| patel_kappa + dir_patel + entropy=0.3 | 0 | 0.3009 | 0.1974 | 63.33% |
| gated+cap（sym=0.0） | 0 | 0.3079 | 0.2006 | 66.67% |
| gated+cap（sym=0.0） | 3e-4 | 0.3153 | 0.2068 | 66.67% |
| gated+cap（sym=0.0） | 0.1 | 0.3568 | 0.2604 | 56.67% |
| **gated+cap+sym=0.5（best）** | **0** | **0.3590** | **0.2527** | 62.22% |
| **gated+cap+sym=0.5（best）** | **3e-4** | **0.3864** | **0.2807** | **62.22%** |
| gated+cap+sym=0.5 | 0.1 | 0.3748 | 0.4099 | 35.56% ⚠️ |

⚠️ sym=0.5 + eps=0.1：100 epoch 下 GT margin 被压塌（gt_margin_median=0.06），recall 崩到 35.56%，不可用。smoke 结论不能外推。

#### Entropy λ 扫描结果（patel_kappa + dir_patel, 5 seeds, eps=0）

| entropy λ | strict_f1 | strict_prec | strict_recall |
|-----------|-----------|-------------|---------------|
| 0.0 | ~0.23 | ~0.14 | ~79% |
| 0.1 | ~0.26 | ~0.17 | ~70% |
| **0.3** | **0.30** | **0.20** | **63%** |
| 0.4 | ~0.29 | ~0.20 | ~60% |

plain entropy 在 λ=0.25-0.35 区间已平台化，继续调 λ 无法突破。

#### Cap λ 在 gated vs ungated 下的行为差异

ungated 条件下 cap λ>0.35 会伤真边（GT margin 从 0.99 暴跌到 0.03），因为 cap 和 margin loss 正面对抗。gated 条件下 cap λ=0.50 仍保持 GT margin=0.99，因为 gate 撤掉了非 GT 对上的 margin loss 保护，cap 可以干净地压幻觉边。

#### 评估注意事项

- **必须用 tie-aware strict 评估**：entropy-only 条件下大量 0-margin ties 被 legacy 评估按节点编号打破，产生虚假高 F1
- **deadzone 是正当的评估组件**：模型在 margin 维度上形成真边和噪声边的双峰分布，eps=3e-4 落在天然 gap 中，不是后处理调参
- **smoke 结论不能直接外推到 formal**：sym=0.5 在 30 epoch 下 GT margin 维持 0.99，100 epoch 下被压到 0.06。λ 需要和训练长度一起调
- 报告时给出 eps=0（conservative baseline）和 eps=3e-4（primary）两个点；eps=0.1 仅在 sym=0.0 分支下有效

#### 已停止的分支

| 分支 | 停止原因 |
|------|---------|
| plain entropy timing sweep（warmup=20/30） | warmup=20 比 10 更差或持平 |
| plain entropy 窄 λ sweep（0.25-0.35） | 三个点 strict_f1 在 0.297-0.302，噪声范围 |
| cap-only 加大 λ（>0.35 ungated） | GT margin 暴跌，开始伤真边 |
| reciprocal-edge penalty | 诊断证明互逆边只占 FP 的 ~4%，不对症 |
| sym=0.5 + eps=0.1 | 100 epoch 下 GT margin 被压塌，recall 崩到 35.56% |

#### 后续方向

**损失函数层面的优化已接近天花板**，继续堆 loss 的边际收益很低。下一阶段应从模型架构和数据流层面找突破口。

**最高优先级（架构层面改动）：**

1. **降低 emb_dim（结构参数化约束）**
   - 当前 emb_dim=N（全秩），sender @ receiver.T 可以表达任意 N×N 矩阵
   - sim3 有 210 个可能的有向边但只有 18 条真边，参数化表达能力远超需要
   - 这是幻觉边的根本原因之一 — 模型有足够自由度在任意节点对上创造强边
   - 降低到 emb_dim=N/2 或更低，可能从根源上替代 entropy+cap+sym 的软约束
   - 实现：只需改一个参数，零代码改动

2. **验证 GraphConv 消息传递方向（零成本诊断）**
   - 去噪网络用 GraphConv 做消息传递，但方向和因果方向是否一致从未验证
   - DGL GraphConv 默认 h_dst = sum(edge_weight * h_src)
   - 如果 adj[i,j] 表示 i→j（因果），去噪应该是"用原因 i 的信息恢复结果 j"
   - 如果方向搞反了，去噪在强化错误的信息流

3. **多被试梯度累积**
   - 当前每个被试独立 optimizer.step()，梯度信号极其嘈杂
   - 改成遍历所有被试后再 step，或打包成 batch [B, N, T]

4. **用学到的图替代固定 noise_guide_adj**
   - 加噪用固定先验图（Patel/Pearson），去噪用学到的结构图，两者完全脱节
   - 如果加噪也用学到的图（detach 后），加噪过程本身就参与结构学习
   - 正确的图 → 更有针对性的噪声 → 更容易去噪 → 更低的损失 → 正反馈

5. **编码器 fine-tune（替代完全冻结）**
   - 冻结后编码器不再接收结构学习的梯度反馈
   - 用更小学习率做 fine-tune，让编码器适应结构学习需求

**中等优先级（继续验证）：**

6. 在 sim4（50节点）上验证 best setting 是否 scale
7. 如果想推 eps=0.1 的大 deadzone，需要先把 sym λ 降到 0.1-0.2
8. recall 恢复：当前 62% vs legacy 79%

**不应再回到的分支：** patel_score init、cross-pred v1、L1 加大、lag-corr 替代 Patel

### 关键代码路径

- 结构初始化：`DDM.py:217-252`（SVD 分解 init_features → sender/receiver）
- 邻接矩阵：`DDM.py:254-267`（get_structure_logits → sigmoid → diag_mask）
- raw/causal 转换：`main_structure_learning.py:361-370`（`to_causal_matrix_torch` = transpose）
- 方向 margin loss（含 kappa gate）：`main_structure_learning.py:574-611`（`directional_kappa_gate` 参数控制 gate 开关）
- parent entropy loss：`main_structure_learning.py:632-654`
- parent cap loss (hinge)：`main_structure_learning.py:702`（`parent_cap_lambda/target` 参数）
- ungated symmetry reg：`main_structure_learning.py:753`（`ungated_symmetry_lambda` 参数）
- cross-pred loss：`main_structure_learning.py:664-685`（v1，已证明不够强）
- 辅助 lambda 调度：`main_structure_learning.py:481-566`（compute_single_aux_lambda，注意 cosine anneal 问题）
- 训练循环损失组装：`main_structure_learning.py:1396-1488`（含 cap 接线 ~1524，sym 接线 ~2198）
- 实验 runner：`GraphExp/run_cross_pred_v1_final_only_compare.py`（支持 parent_cap/kappa_gate/ungated_sym sweep）

### 数据集

| 文件 | 节点数 | GT 边数 | GT 文件 | 真实密度 |
|------|--------|---------|---------|---------|
| fMRI.csv | 5 | 5 | h1.txt | 50% |
| sim2.csv | 10 | 11 | h2.txt | ~24% |
| sim3.csv | 15 | 18 | h3.txt | ~17% |
| sim4.csv | 50 | 61 | h4.txt | ~5% |

所有数据集：50 subjects × 200 time points。

### 分析时的自省清单

每次做实验诊断前先过一遍（详见 `memory/analysis_pitfalls.md`）：
1. 追踪损失项的完整生命周期（初始值 → 调度 → 评估时刻的实际权重）
2. 区分 "信号弱" vs "信号被调度关掉了"
3. 不要从单一实验推导过强结论
4. 矩阵运算语义必须用 2×2 例子手动验证（raw/causal convention 容易搞混）
5. 审视自己的方案要像审视别人的一样严格
6. 设计多辅助损失方案时，逐一检查每个损失的调度路径
