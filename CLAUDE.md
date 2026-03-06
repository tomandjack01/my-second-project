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

### Encoder Pretraining Pipeline (Autoregressive Causal)

The `NodeSpecificTemporalEncoder` collapses when trained end-to-end with diffusion (cosine sim→1.0, Diff Loss→0.0). Solution: pretrain encoder with autoregressive objective, then freeze.

**Autoregressive pretrain loss:**
- Uses `pretrain_forward()`: predict time step t+1 from t via MSE on unnormalized features
- Bypasses LayerNorm to preserve signal scale for the prediction target
- `pred_head` (Linear) maps encoder output to prediction space

**Integration flow (`main_structure_learning.py`):**
1. Create DDM model (encoder output dim = `in_dim`, i.e. `time_points`)
2. Autoregressive pretrain: iterate all subjects, call `pretrain_forward()`, accumulate gradients
3. Freeze `model.temporal_encoder` (requires_grad=False, eval mode)
4. Rebuild optimizer with only unfrozen parameters
5. Normal diffusion training (diffusion operates in original time dimension space)

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
- ~~`--pretrain_split_ratio`~~: Deprecated (autoregressive pretrain does not need split)

### Key Dependencies
- DGL (Deep Graph Library) for graph neural networks
- PyTorch for deep learning
- scikit-learn for SVM evaluation
- OGB for ogbn-arxiv dataset
