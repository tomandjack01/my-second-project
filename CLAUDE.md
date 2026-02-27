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
- `NodeSpecificTemporalEncoder`: Causal dilated temporal encoder for fMRI time series
  - `forward()`: Returns `[N, output_dim]` encoding
  - `encode_features()`: Returns both pre-GAP `[B*N, C, T]` feature map and post-GAP `[N, H]` encoding (used by pretrain decoder)

**Encoder Pretraining** (`GraphExp/pretrain_temporal_encoder.py`):
- `TemporalDecoder`: Lightweight decoder (2 blocks) for reconstruction loss
- `ForecastingHead`: 3-layer MLP for future prediction loss
- `vicreg_loss()`: Variance + Covariance regularization (anti-collapse)
- `pretrain_temporal_encoder()`: Main pretrain loop with three-objective loss
- Supports standalone execution via `__main__`

**Denoising Network** (`models/mlp_gat.py`):
- `Denoising_Unet`: Encoder-decoder architecture with down/up GAT layers
- `MlpBlock`: Residual MLP blocks with configurable normalization and activation
- Time embeddings are added at each layer

### Training Pipeline

1. Forward diffusion: Add directional noise at random timestep t
2. Denoising: Predict original features from noisy features using U-Net
3. Loss: Cosine similarity loss with configurable alpha power

### Encoder Pretraining Pipeline (Anti-Collapse)

The `NodeSpecificTemporalEncoder` collapses when trained end-to-end with diffusion (cosine sim→1.0, Diff Loss→0.0). Solution: pretrain encoder independently, then freeze.

**Three-objective pretrain loss:**
1. Reconstruction: `MSE(decoder(z), x_original)` — faithfulness
2. Forecasting: `MSE(mlp(encoder(x[:,:P])), x[:,P:])` — future information (warmup 10 epochs)
3. VICReg: variance term (std≥1.0 per dim) + covariance term (decorrelate dims) — anti-collapse

**Loss weights:** `total = 1.0*recon + 0.5*forecast + 1.0*variance + 0.04*covariance`

**Integration flow (`main_structure_learning.py`):**
1. Create DDM model
2. Pretrain or load encoder (skip with `--skip_pretrain`)
3. Freeze `model.temporal_encoder` (requires_grad=False)
4. Rebuild optimizer with only unfrozen parameters
5. Normal diffusion training

**Brain Connectivity Structure Learning** (`GraphExp/main_structure_learning.py`):
```shell
# Full pipeline (pretrain + freeze + diffusion)
python main_structure_learning.py --epochs 100 --pretrain_epochs 50

# Skip pretrain (original behavior)
python main_structure_learning.py --epochs 100 --skip_pretrain

# Load existing pretrained weights
python main_structure_learning.py --pretrain_checkpoint ./results/run_xxx/pretrained_encoder.pt

# Standalone pretrain
python pretrain_temporal_encoder.py --epochs 50 --save_path ./pretrained_encoder.pt
```

**Collapse diagnostics** (`diagnose_encoder_collapse()`): Healthy indicators after pretrain:
- `effective_rank` > 5 (ideally > 10)
- `mean_cosine_sim` < 0.5 (ideally < 0.3)
- `dead_dims_ratio` = 0%
- `feature_std_mean` > 0.1

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
- `--pretrain_epochs`: Number of encoder pretrain epochs (default: 50)
- `--pretrain_lr`: Pretrain learning rate (default: 1e-3)
- `--pretrain_split_ratio`: Input/forecast split ratio (default: 0.75, i.e. 150/50 for T=200)
- `--skip_pretrain`: Skip pretraining entirely (end-to-end training, original behavior)
- `--pretrain_checkpoint`: Path to load existing pretrained encoder weights

### Key Dependencies
- DGL (Deep Graph Library) for graph neural networks
- PyTorch for deep learning
- scikit-learn for SVM evaluation
- OGB for ogbn-arxiv dataset
