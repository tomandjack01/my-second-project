#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Graph Structure Learning for fMRI Brain Connectivity



Learns a shared brain connectivity matrix from fMRI time-series data

using DDM (Directional Diffusion Models) with L1 sparsity regularization.

"""



import argparse
import math
import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models import DDM
from utils.patel_util import compute_patel_components


# ============================================================================
# ENCODER COLLAPSE DIAGNOSTICS
# ============================================================================

@torch.no_grad()
def diagnose_encoder_collapse(model, data_3d, device, max_subjects=8):
    """
    Diagnose whether the temporal encoder is collapsing.

    Returns a dict of scalar metrics:
      - effective_rank: Effective rank of the encoding matrix (ideal: close to num_hidden)
      - mean_cosine_sim: Mean pairwise cosine similarity between node encodings (collapse → 1.0)
      - dead_dims_ratio: Fraction of feature dimensions with std < 1e-3 (collapse → 1.0)
      - feature_std_mean: Mean per-dimension std across nodes (collapse → 0.0)
      - inter_subject_var: Variance of encodings across subjects for same node (collapse → 0.0)
      - encoder_weight_norm: L2 norm of encoder input projection weights
    """
    model.eval()
    n_subj = min(max_subjects, data_3d.shape[0])
    encodings = []

    for s in range(n_subj):
        x = data_3d[s].to(device)  # [N, T]
        x_enc = model.temporal_encoder(x)  # [N, H]
        x_enc = F.layer_norm(x_enc, (x_enc.shape[-1],))
        encodings.append(x_enc)

    # Stack: [n_subj, N, H]
    enc_stack = torch.stack(encodings, dim=0)
    N, H = enc_stack.shape[1], enc_stack.shape[2]

    # --- Metric 1: Effective rank (on pooled encodings) ---
    # Pool all subjects: [n_subj * N, H]
    pooled = enc_stack.reshape(-1, H)
    # Center
    pooled_centered = pooled - pooled.mean(dim=0, keepdim=True)
    # SVD
    singular_values = torch.linalg.svdvals(pooled_centered)
    # Normalized singular values → probability distribution → entropy → effective rank
    sv_norm = singular_values / (singular_values.sum() + 1e-12)
    sv_norm = sv_norm[sv_norm > 1e-12]  # filter zeros
    entropy = -(sv_norm * torch.log(sv_norm)).sum()
    effective_rank = torch.exp(entropy).item()

    # --- Metric 2: Mean pairwise cosine similarity (per subject, then average) ---
    cos_sims = []
    for s in range(n_subj):
        enc_normed = F.normalize(enc_stack[s], p=2, dim=-1)  # [N, H]
        sim_matrix = enc_normed @ enc_normed.T  # [N, N]
        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
        cos_sims.append(sim_matrix[mask].mean().item())
    mean_cosine_sim = sum(cos_sims) / len(cos_sims)

    # --- Metric 3: Dead dimensions ratio ---
    # Per-dimension std across all nodes and subjects
    dim_std = pooled.std(dim=0)  # [H]
    dead_dims_ratio = (dim_std < 1e-3).float().mean().item()

    # --- Metric 4: Feature std mean ---
    feature_std_mean = dim_std.mean().item()

    # --- Metric 5: Inter-subject variance ---
    # For each node, how much does its encoding vary across subjects?
    # enc_stack: [n_subj, N, H]
    inter_subj_var = enc_stack.var(dim=0).mean().item()  # mean over [N, H]

    # --- Metric 6: Encoder weight norm ---
    encoder_weight_norm = model.temporal_encoder.input_proj.weight.data.norm().item()

    model.train()

    return {
        "effective_rank": effective_rank,
        "mean_cosine_sim": mean_cosine_sim,
        "dead_dims_ratio": dead_dims_ratio,
        "feature_std_mean": feature_std_mean,
        "inter_subject_var": inter_subj_var,
        "encoder_weight_norm": encoder_weight_norm,
    }


def print_collapse_diagnostics(metrics, epoch, num_epochs):
    """Pretty-print collapse diagnostic metrics with warning flags."""
    rank = metrics["effective_rank"]
    cos = metrics["mean_cosine_sim"]
    dead = metrics["dead_dims_ratio"]
    std = metrics["feature_std_mean"]
    isv = metrics["inter_subject_var"]
    wnorm = metrics["encoder_weight_norm"]

    # Warning thresholds
    rank_warn = " ⚠ LOW RANK" if rank < 5 else ""
    cos_warn = " ⚠ HIGH SIM" if cos > 0.8 else ""
    dead_warn = " ⚠ DEAD DIMS" if dead > 0.3 else ""
    std_warn = " ⚠ LOW STD" if std < 0.01 else ""
    isv_warn = " ⚠ NO SUBJECT VARIATION" if isv < 1e-4 else ""

    print(f"  [Collapse Diag] Epoch [{epoch+1:3d}/{num_epochs}]")
    print(f"    Effective Rank:    {rank:8.2f}{rank_warn}")
    print(f"    Mean Cosine Sim:   {cos:8.4f}{cos_warn}")
    print(f"    Dead Dims Ratio:   {dead:8.2%}{dead_warn}")
    print(f"    Feature Std Mean:  {std:8.6f}{std_warn}")
    print(f"    Inter-Subject Var: {isv:8.6f}{isv_warn}")
    print(f"    Encoder W Norm:    {wnorm:8.4f}")



# ============================================================================

# CONFIGURATION

# ============================================================================

TIME_POINTS_PER_SUBJECT = 200  # Number of time points per subject





def set_seed(seed: int):

    """Set random seeds for reproducibility."""

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

    # Force all PyTorch ops (including CUDA scatter/gather used by DGL) to be deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)





def load_fmri_data(csv_path: str, time_points_per_subject: int = TIME_POINTS_PER_SUBJECT):

    """

    Load and reshape fMRI data from CSV.

    

    Args:

        csv_path: Path to fMRI.csv file (NO HEADER)

        time_points_per_subject: Number of time points per subject

    

    Returns:

        data_3d: torch.Tensor of shape [Num_Subjects, N, TIME_POINTS]

        data_2d: torch.Tensor of shape [Total_Rows, N] for Pearson computation

        num_subjects: Number of subjects

        num_nodes: Number of brain regions (N)

    """

    # Load CSV with NO HEADER

    df = pd.read_csv(csv_path, header=None)

    data = df.values.astype(np.float32)

    

    total_rows, num_nodes = data.shape

    print(f"Loaded data with {total_rows} rows and {num_nodes} columns.")

    

    # Validate row count

    if total_rows % time_points_per_subject != 0:

        raise ValueError(

            f"Total rows ({total_rows}) is not divisible by TIME_POINTS_PER_SUBJECT "

            f"({time_points_per_subject}). Please check data or adjust configuration."

        )

    

    num_subjects = total_rows // time_points_per_subject

    print(f"Detected {num_subjects} subjects with {time_points_per_subject} time points each.")

    

    # Keep 2D for Pearson computation [Total_Rows, N]

    data_2d = torch.from_numpy(data).float()

    

    # Reshape to 3D: [Num_Subjects, TIME_POINTS, N]

    data_3d = data.reshape(num_subjects, time_points_per_subject, num_nodes)

    # Transpose to [Num_Subjects, N, TIME_POINTS] for model input

    data_3d = np.transpose(data_3d, (0, 2, 1))

    data_3d = torch.from_numpy(data_3d).float()

    

    print(f"Reshaped data to: {data_3d.shape} [Num_Subjects, N, TIME_POINTS]")

    

    return data_3d, data_2d, num_subjects, num_nodes





def compute_global_pearson(data_2d: torch.Tensor):

    """

    Compute Pearson correlation matrix from 2D data.

    

    Args:

        data_2d: torch.Tensor of shape [Total_Rows, N]

    

    Returns:

        pearson_matrix: torch.Tensor of shape [N, N]

    """

    # Transpose to [N, Total_Rows] for correlation computation

    data_t = data_2d.T  # [N, Total_Rows]

    

    # Normalize: subtract mean and divide by std

    data_norm = (data_t - data_t.mean(dim=1, keepdim=True)) / (data_t.std(dim=1, keepdim=True) + 1e-8)

    

    # Compute Pearson correlation: [N, N]

    pearson_matrix = data_norm @ data_norm.T / data_norm.shape[1]

    

    print(f"Computed global Pearson matrix: {pearson_matrix.shape}")

    print(f"Pearson range: [{pearson_matrix.min().item():.4f}, {pearson_matrix.max().item():.4f}]")

    

    return pearson_matrix


def compute_target_density(num_nodes: int, target_edge_count: int, eps: float = 1e-4) -> float:
    """Map a target edge count to a safe directed-edge density for logit init."""
    max_directed_edges = max(num_nodes * (num_nodes - 1), 1)
    raw_density = float(target_edge_count) / float(max_directed_edges)
    return min(max(raw_density, eps), 0.95)


def build_noise_guide_adjacency(patel_strength_matrix: torch.Tensor, top_k_pairs: int):
    """
    Build a symmetric row-normalized adjacency for neighbor-based noise.

    `patel_strength_matrix` should encode undirected skeleton strength. In practice
    we pass positive Patel kappa so skeleton selection is decoupled from direction.
    """
    num_nodes = patel_strength_matrix.shape[0]
    device = patel_strength_matrix.device
    dtype = patel_strength_matrix.dtype

    eye = torch.eye(num_nodes, device=device, dtype=dtype)
    pair_strength = torch.maximum(patel_strength_matrix, patel_strength_matrix.t())
    pair_strength = torch.clamp(pair_strength, min=0.0) * (1.0 - eye)

    triu_i, triu_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    flat_strength = pair_strength[triu_i, triu_j]
    num_pairs = flat_strength.numel()
    k_pairs = min(max(int(top_k_pairs), 0), num_pairs)

    adj_binary = torch.zeros_like(pair_strength)
    threshold = 0.0
    if k_pairs > 0 and num_pairs > 0:
        top_idx = torch.topk(flat_strength, k_pairs).indices
        src = triu_i[top_idx]
        dst = triu_j[top_idx]
        adj_binary[src, dst] = 1.0
        adj_binary[dst, src] = 1.0
        threshold = float(flat_strength[top_idx].min().item())

    adj_with_self = adj_binary + eye
    degree = adj_with_self.sum(dim=1, keepdim=True)
    noise_guide_adj = adj_with_self / (degree + 1e-9)
    return noise_guide_adj, adj_binary, k_pairs, threshold


@torch.no_grad()
def get_current_structure_adj(model: DDM) -> torch.Tensor:
    """Fetch the current adjacency using the exact export-time structure logic."""
    return model.get_structure_adj().detach()


def compute_auxiliary_lambdas(
    epoch: int,
    num_epochs: int,
    loss_ddm_main: torch.Tensor,
    raw_loss_dir: torch.Tensor,
    raw_loss_ortho: torch.Tensor,
    prev_lambda_dir: float,
    prev_lambda_ortho: float,
    warmup_epochs: int = 5,
):
    """Warmup + ratio-adaptive scaling + cosine anneal + EMA smoothing."""
    if epoch < warmup_epochs:
        return 0.0, 0.0

    post_warmup_epochs = max(num_epochs - warmup_epochs, 1)
    ramp_epochs = max(1, min(10, post_warmup_epochs))
    ramp = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    anneal_progress = float(epoch - warmup_epochs) / float(max(post_warmup_epochs - 1, 1))
    anneal = 0.5 * (1.0 + math.cos(math.pi * anneal_progress))
    epoch_factor = ramp * anneal

    scale_dir = (loss_ddm_main.detach() * 0.01) / (raw_loss_dir.detach() + 1e-6)
    scale_ortho = (loss_ddm_main.detach() * 0.005) / (raw_loss_ortho.detach() + 1e-6)

    lambda_dir_raw = min(scale_dir.item() * epoch_factor, 0.5)
    lambda_ortho_raw = min(scale_ortho.item() * epoch_factor, 0.5)

    ema_alpha = 0.1
    lambda_dir = ema_alpha * lambda_dir_raw + (1 - ema_alpha) * prev_lambda_dir
    lambda_ortho = ema_alpha * lambda_ortho_raw + (1 - ema_alpha) * prev_lambda_ortho

    max_change = 0.1
    if prev_lambda_dir > 0:
        lambda_dir = max(prev_lambda_dir * (1 - max_change),
                         min(prev_lambda_dir * (1 + max_change), lambda_dir))
    if prev_lambda_ortho > 0:
        lambda_ortho = max(prev_lambda_ortho * (1 - max_change),
                           min(prev_lambda_ortho * (1 + max_change), lambda_ortho))

    return lambda_dir, lambda_ortho



# ============================================================================
# AUXILIARY LOSSES: Directional Prior & Feature Decoupling
# ============================================================================

def compute_directional_margin_loss(logits, direction_prior_matrix, margin=1.0):
    """
    基于 Patel 算法的高置信度先验，在 Logit 空间计算带 Margin 的方向引导损失。

    q_threshold 自适应：取 |delta_P| 非零值的中位数，确保约 50% 的边参与约束。
    margin 自适应：在有效边（w>0）上对 sign(delta_P)*D 取 25 分位数，
                  严格保证约 25% 的有效边违反约束、持续提供梯度。
    """
    delta_prior = direction_prior_matrix - direction_prior_matrix.t()
    abs_delta_prior = torch.abs(delta_prior)

    # 自适应 q_threshold：非零 |delta_P| 的中位数
    nonzero_vals = abs_delta_prior[abs_delta_prior > 0]
    if nonzero_vals.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    q_threshold = nonzero_vals.median().item()

    # 只对高置信度（差值大于自适应阈值）的边施加先验
    active_mask = abs_delta_prior > q_threshold
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    w = active_mask.float() * abs_delta_prior

    D = logits - logits.t()
    signed_D = torch.sign(delta_prior) * D  # 正值=方向正确，负值=方向错误

    # 自适应 margin：有效边上 signed_D 的 25 分位数（detach，不参与梯度）
    # quantile(0.25) 意味着 25% 的有效边 signed_D ≤ margin，即 25% 违反约束
    active_signed_D = signed_D[active_mask].detach()
    if active_signed_D.numel() > 0:
        adaptive_margin = active_signed_D.quantile(0.25).item()
    else:
        adaptive_margin = margin

    # Margin Loss
    wrong_dir_penalty = F.relu(adaptive_margin - signed_D)
    loss_dir = torch.sum(w * wrong_dir_penalty) / (torch.sum(w) + 1e-8)
    return loss_dir


def compute_feature_ortho_loss(S, R):
    """
    对发送端和接收端的特征空间进行解耦，计算互协方差矩阵的 Frobenius 范数。
    先 L2 归一化，防止嵌入范数膨胀导致 loss 爆炸。
    """
    N = S.shape[0]
    # L2 归一化：消除范数量级影响，只关注方向相关性
    S_n = F.normalize(S, p=2, dim=1)
    R_n = F.normalize(R, p=2, dim=1)
    # 沿节点维度去中心化
    S_c = S_n - S_n.mean(dim=0, keepdim=True)
    R_c = R_n - R_n.mean(dim=0, keepdim=True)

    # 计算特征维度的互协方差矩阵 [H, H]
    C = torch.mm(S_c.t(), R_c) / N
    return torch.sum(C ** 2)


@torch.no_grad()
def compute_epoch_quality(adj_np, patel_direction_cpu, patel_strength_cpu, top_k=61):
    """
    不依赖 GT 的 epoch 质量评分，用于 best-epoch 选择。

    评分 = agreement_strict * dir_margin * density_factor

    - agreement_strict: top-k 边中，仅在 Patel 高置信方向边上计算方向一致率
      （排除 Patel 平局，避免乐观偏差）
    - dir_margin: top-k 边的平均 |adj[i,j] - adj[j,i]|（衡量方向强度，
      不受饱和高分假边误导）
    - density_factor: 惩罚实际密度偏离目标密度过远的情况（抑制过稀疏/过饱和）

    Args:
        adj_np: 邻接矩阵 [N, N] numpy array (sigmoid 后)
        patel_direction_cpu: Patel tau 方向先验 [N, N] numpy array
        patel_strength_cpu: Patel kappa/score 强度先验 [N, N] numpy array
        top_k: 取 top-k 条边评估（应按数据集配置）
    Returns:
        score: float, 越高越好
        details: dict with sub-metrics
    """
    n = adj_np.shape[0]
    patel_direction = patel_direction_cpu
    patel_strength = np.maximum(patel_strength_cpu, 0.0)

    # --- Patel 高置信阈值：|p_ij - p_ji| 的中位数 ---
    patel_delta = np.abs(patel_direction - patel_direction.T)
    off_diag_mask = ~np.eye(n, dtype=bool)
    patel_nonzero = patel_delta[off_diag_mask]
    patel_nonzero = patel_nonzero[patel_nonzero > 0]
    patel_thresh = float(np.median(patel_nonzero)) if len(patel_nonzero) > 0 else 0.0

    # --- 收集所有无向边对 ---
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            w_ij, w_ji = float(adj_np[i, j]), float(adj_np[j, i])
            margin = abs(w_ij - w_ji)
            if w_ij > w_ji:
                src, dst = i, j
            elif w_ji > w_ij:
                src, dst = j, i
            else:
                src, dst = i, j  # tie: 方向不确定
            max_w = max(w_ij, w_ji)
            candidates.append((src, dst, max_w, margin))

    # 按最大方向权重降序，取 top-k
    candidates.sort(key=lambda x: x[2], reverse=True)
    k = min(top_k, len(candidates))
    if k == 0:
        return 0.0, {"agreement": 0.0, "dir_margin": 0.0,
                      "density_factor": 0.0, "skeleton_overlap": 0.0,
                      "high_conf_edges": 0, "k": 0}

    top_edges = candidates[:k]
    top_edge_set = {(e[0], e[1]) for e in top_edges}
    # Also store as undirected for skeleton comparison
    top_edge_undirected = {(min(e[0], e[1]), max(e[0], e[1])) for e in top_edges}

    # --- 0) skeleton_overlap: Patel top-k 骨架一致性 ---
    patel_candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_strength = max(float(patel_strength[i, j]), float(patel_strength[j, i]))
            patel_candidates.append((i, j, pair_strength))
    patel_candidates.sort(key=lambda x: x[2], reverse=True)
    patel_top_k_set = {(e[0], e[1]) for e in patel_candidates[:k]}
    skeleton_overlap = len(top_edge_undirected & patel_top_k_set) / k

    # --- 1) agreement_strict: 仅在 Patel 高置信边上算 ---
    agree_count = 0
    high_conf_count = 0
    for src, dst, _, _ in top_edges:
        p_delta = abs(patel_direction[src, dst] - patel_direction[dst, src])
        if p_delta <= patel_thresh:
            continue  # Patel 低置信 / 平局，跳过
        high_conf_count += 1
        # Patel 认为 src→dst 当 patel[src,dst] > patel[dst,src]
        if patel_direction[src, dst] > patel_direction[dst, src]:
            agree_count += 1

    agreement = agree_count / high_conf_count if high_conf_count > 0 else 0.0

    # --- 2) dir_margin: 平均方向强度 |adj[i,j] - adj[j,i]| ---
    dir_margin = float(np.mean([e[3] for e in top_edges]))

    # --- 3) density_factor: 惩罚密度偏离 ---
    total_pairs = n * (n - 1) // 2
    target_density = k / max(total_pairs, 1)
    actual_positive_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if max(float(adj_np[i, j]), float(adj_np[j, i])) > 0.5:
                actual_positive_pairs += 1
    actual_density = actual_positive_pairs / max(total_pairs, 1)
    density_ratio = actual_density / (target_density + 1e-8)
    # Gaussian-style penalty: ratio=1 → factor=1, 宽容到 ~10x 偏离仍有 ~0.1 分
    density_factor = float(np.exp(-0.5 * (np.log(density_ratio + 1e-8)) ** 2))

    score = agreement * dir_margin * density_factor * skeleton_overlap
    return score, {
        "agreement": agreement,
        "dir_margin": dir_margin,
        "density_factor": density_factor,
        "skeleton_overlap": skeleton_overlap,
        "actual_pair_density": actual_density,
        "target_pair_density": target_density,
        "high_conf_edges": high_conf_count,
        "k": k,
    }


def train_brain_connectivity(

    data_3d: torch.Tensor,

    pearson_matrix: torch.Tensor,

    num_nodes: int,

    time_points: int,

    patel_matrix: torch.Tensor,
    patel_direction_matrix: Optional[torch.Tensor] = None,
    patel_strength_matrix: Optional[torch.Tensor] = None,
    noise_guide_adj: Optional[torch.Tensor] = None,

    num_epochs: int = 100,

    learning_rate: float = 1e-3,

    lambda_l1: float = 0.01,

    device: str = 'cuda',

    log_interval: int = 10,

    num_hidden: int = 64,

    num_layers: int = 2,

    batch_size: int = 4,

    debug_checks: bool = False,

    ddm_kwargs: Optional[Dict[str, Any]] = None,

    # Pretrain parameters
    skip_pretrain: bool = False,
    pretrain_checkpoint: Optional[str] = None,
    pretrain_epochs: int = 50,
    pretrain_lr: float = 1e-3,
    result_dir: Optional[str] = None,
    target_edge_count: int = 61,

):

    """

    Train DDM to learn brain connectivity from fMRI data.

    

    Args:

        data_3d: Subject data [Num_Subjects, N, TIME_POINTS]

        pearson_matrix: Global Pearson correlation [N, N]

        num_nodes: Number of brain regions

        time_points: Number of time points per subject

        num_epochs: Number of training epochs

        learning_rate: Learning rate

        lambda_l1: L1 regularization coefficient

        device: Device to train on

        log_interval: Epochs between log messages

        num_hidden: Hidden dimension

        num_layers: Number of GNN layers

        batch_size: Batch size for subjects

        patel_matrix: Patel score matrix (-kappa * tau), used for asymmetric init

        patel_direction_matrix: Patel tau matrix used as weak directional prior

        patel_strength_matrix: Patel kappa-like skeleton strength used for proxy scoring

        noise_guide_adj: Row-normalized adjacency matrix for neighbor-based noise

        ddm_kwargs: Optional extra keyword arguments forwarded to DDM

    

    Returns:

        model: Trained DDM model

        adj_matrix: Learned adjacency matrix [N, N]

    """

    num_subjects = data_3d.shape[0]

    data_3d = data_3d.to(device)

    patel_score_matrix = patel_matrix.to(device)
    if patel_direction_matrix is None:
        patel_direction_matrix = patel_score_matrix
    else:
        patel_direction_matrix = patel_direction_matrix.to(device)
    if patel_strength_matrix is None:
        patel_strength_matrix = torch.clamp(0.5 * (patel_score_matrix + patel_score_matrix.t()), min=0.0)
    else:
        patel_strength_matrix = patel_strength_matrix.to(device)

    ddm_kwargs = {} if ddm_kwargs is None else dict(ddm_kwargs)

    # Extract use_temporal_encoder from ddm_kwargs to avoid duplicate argument
    use_temporal_encoder = ddm_kwargs.pop('use_temporal_encoder', True)



    # Initialize DDM with Patel score matrix for structure learning.
    # The score matrix carries asymmetric strength for SVD init, while tau is
    # reserved for the weak directional guidance loss.
    # in_dim = TIME_POINTS (features per node)

    # Compute sparsity bias: logit(target_density) so initial sigmoid mean ≈ target_density
    target_density = compute_target_density(num_nodes, target_edge_count)
    adj_bias_init = math.log(target_density / (1.0 - target_density))

    model = DDM(

        in_dim=time_points,

        num_hidden=num_hidden,

        num_layers=num_layers,

        nhead=4,

        activation='prelu',

        feat_drop=0.1,

        attn_drop=0.1,

        norm='layernorm',

        alpha_l=2,

        beta_schedule='linear',

        beta_1=0.0001,

        beta_T=0.02,

        T=1000,

        init_features=patel_score_matrix,  # [N, N] asymmetric Patel score -> directional SVD init

        noise_guide_adj=noise_guide_adj,  # Row-normalized adj for neighbor-based noise

        adj_bias_init=adj_bias_init,

        use_temporal_encoder=use_temporal_encoder,

        **ddm_kwargs,

    )

    model = model.to(device)

    # ---- Autoregressive Causal Pretraining ----
    if model.use_temporal_encoder and not skip_pretrain and pretrain_epochs > 0:
        if pretrain_checkpoint and os.path.exists(pretrain_checkpoint):
            print(f"\n[Pretrain] Loading encoder weights from: {pretrain_checkpoint}")
            state = torch.load(pretrain_checkpoint, map_location=device)
            model.temporal_encoder.load_state_dict(state)
        else:
            print(f"\n=== 开始时间因果编码器的自回归预训练 ({pretrain_epochs} Epochs) ===")
            enc_optimizer = torch.optim.Adam(model.temporal_encoder.parameters(), lr=pretrain_lr)
            model.temporal_encoder.train()

            for pre_epoch in range(pretrain_epochs):
                enc_optimizer.zero_grad()
                # 遍历所有被试，累积梯度后更新
                total_pre_loss = 0.0
                for s_idx in range(num_subjects):
                    x_subj = data_3d[s_idx]  # [N, T]
                    pre_loss = model.temporal_encoder.pretrain_forward(x_subj)
                    pre_loss.backward()
                    total_pre_loss += pre_loss.item()
                enc_optimizer.step()
                avg_pre_loss = total_pre_loss / num_subjects

                if (pre_epoch + 1) % 10 == 0 or pre_epoch == 0:
                    print(f"Pretrain Epoch [{pre_epoch+1}/{pretrain_epochs}] | "
                          f"Autoregressive MSE Loss: {avg_pre_loss:.4f}")

            # Save pretrained weights
            if result_dir:
                save_path = os.path.join(result_dir, 'pretrained_encoder.pt')
                torch.save(model.temporal_encoder.state_dict(), save_path)
                print(f"[Pretrain] Saved encoder weights to: {save_path}")

        # Post-pretrain collapse diagnostics
        print("\n[Pretrain] Post-pretrain collapse diagnostics:")
        pt_metrics = diagnose_encoder_collapse(model, data_3d, device)
        print_collapse_diagnostics(pt_metrics, 0, 1)

        # [极其关键] 冻结参数，防止扩散过程导致表征坍缩
        print("\n=== 预训练完成！开始冻结编码器参数 ===")
        for param in model.temporal_encoder.parameters():
            param.requires_grad = False
        model.temporal_encoder.eval()
        print("=== 进入正式的扩散图学习阶段 ===")

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model initialized: {trainable_params:,} trainable / {total_params:,} total parameters")

    print(f"Learning adjacency matrix of shape [{num_nodes}, {num_nodes}]")



    # Rebuild optimizer with only unfrozen parameters
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    

    # Track loss history for plotting

    loss_history = []

    collapse_history = []
    quality_history = []

    # Best-epoch tracking (Patel-based proxy, no GT needed)
    best_adj = None
    best_score = -1.0
    best_epoch = -1
    best_quality_details = None
    patel_direction_cpu = patel_direction_matrix.detach().cpu().numpy()
    patel_strength_cpu = patel_strength_matrix.detach().cpu().numpy()
    quality_top_k = min(max(int(target_edge_count), 1), max(num_nodes * (num_nodes - 1) // 2, 1))

    # Lambda smoothing state (EMA + step-change cap)
    prev_lambda_dir = 0.0
    prev_lambda_ortho = 0.0

    

    # Training loop

    for epoch in range(num_epochs):

        model.train()

        epoch_loss = 0.0

        epoch_sparsity = 0.0

        epoch_dir_loss = 0.0

        epoch_ortho_loss = 0.0

        num_batches = 0

        

        # Shuffle subjects

        perm = torch.randperm(num_subjects)

        

        for i in range(0, num_subjects, batch_size):

            batch_idx = perm[i:i+batch_size]

            # batch_data: [batch_size, N, TIME_POINTS]

            batch_data = data_3d[batch_idx]

            

            # Process each subject in batch

            for subj_idx in range(batch_data.shape[0]):

                optimizer.zero_grad()

                

                # Get subject data: [N, TIME_POINTS]

                x = batch_data[subj_idx]  # [N, TIME_POINTS]

                if debug_checks and epoch == 0 and i == 0 and subj_idx == 0:

                    with torch.no_grad():

                        if model.use_temporal_encoder:
                            x_encoded = model.temporal_encoder(x)
                            x_encoded = F.layer_norm(x_encoded, (x_encoded.shape[-1],))
                        else:
                            x_encoded = F.layer_norm(x, (x.shape[-1],))

                        t = torch.randint(model.T, size=(x_encoded.shape[0],), device=x_encoded.device)

                        g_dbg = None

                        if getattr(model, 'structure_learning_mode', False):

                            g_dbg, _ = model._get_structure_graph(x_encoded.device)

                        x_t, _, _ = model.sample_q(t, x_encoded, g_dbg)
                        cos_xt_x = F.cosine_similarity(x_t, x_encoded, dim=-1).mean().item()
                        abs_delta = (x_t - x_encoded).abs()
                        abs_delta_mean = abs_delta.mean().item()
                        x_mean_abs = x_encoded.abs().mean().item()
                        delta_ratio = abs_delta_mean / (x_mean_abs + 1e-8)
                        same_storage = x_t.data_ptr() == x_encoded.data_ptr()
                        allclose = torch.allclose(x_t, x_encoded)
                        print(
                            f"[Debug] cosine(x_t, x_encoded) mean={cos_xt_x:.4f} | "
                            f"t[min,max,mean]=min={int(t.min().item())} max={int(t.max().item())} mean={t.float().mean().item():.1f} | "
                            f"noise_abs_mean={abs_delta_mean:.4e} (ratio={delta_ratio:.4e}) | "
                            f"alias={same_storage} allclose={allclose}"
                        )

                

                # Forward pass - model uses learned structure internally

                loss, loss_dict = model(g=None, x=x)

                

                # L1 sparsity regularization on learned adjacency.
                # Reuse the exact same clamped logits/sigmoid path used for export.
                adj_weights = model.get_structure_adj()  # [N, N], diag already zeroed

                n_off_diag = num_nodes * num_nodes - num_nodes
                l1_norm = torch.norm(adj_weights, p=1)

                if n_off_diag > 0:
                    sparsity_loss = lambda_l1 * (l1_norm / n_off_diag)
                else:
                    sparsity_loss = torch.tensor(0.0, device=device)

                # Hub regularization: penalize variance of embedding norms
                # to prevent a few nodes from dominating all edges
                sender_norms = torch.norm(model.node_emb_sender, dim=1)
                receiver_norms = torch.norm(model.node_emb_receiver, dim=1)
                hub_loss = 0.01 * (sender_norms.var() + receiver_norms.var())

                # DDM main loss (diffusion + sparsity + hub)
                loss_ddm_main = loss + sparsity_loss + hub_loss

                # --- Directional margin loss & feature orthogonality loss ---
                logits = model.get_structure_logits()
                raw_loss_dir = compute_directional_margin_loss(logits, patel_direction_matrix)
                raw_loss_ortho = compute_feature_ortho_loss(
                    model.node_emb_sender, model.node_emb_receiver,
                )

                lambda_dir, lambda_ortho = compute_auxiliary_lambdas(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    loss_ddm_main=loss_ddm_main,
                    raw_loss_dir=raw_loss_dir,
                    raw_loss_ortho=raw_loss_ortho,
                    prev_lambda_dir=prev_lambda_dir,
                    prev_lambda_ortho=prev_lambda_ortho,
                )

                prev_lambda_dir = lambda_dir
                prev_lambda_ortho = lambda_ortho

                weighted_dir = lambda_dir * raw_loss_dir
                weighted_ortho = lambda_ortho * raw_loss_ortho

                total_loss = loss_ddm_main + weighted_dir + weighted_ortho

                

                # Backward pass

                total_loss.backward()

                if debug_checks and epoch == 0 and i == 0 and subj_idx == 0:

                    if model.use_temporal_encoder:
                        grad = model.temporal_encoder.input_proj.weight.grad
                        if grad is None:
                            print("[Debug] temporal_encoder grad is None")
                        else:
                            print(f"[Debug] temporal_encoder grad norm: {grad.norm().item():.6e}")
                    else:
                        print("[Debug] Temporal encoder disabled - skipping grad check")

                optimizer.step()

                

                epoch_loss += loss.item()

                epoch_sparsity += sparsity_loss.item()

                epoch_dir_loss += weighted_dir.item()

                epoch_ortho_loss += weighted_ortho.item()

                num_batches += 1

        

        avg_loss = epoch_loss / num_batches
        avg_sparsity = epoch_sparsity / num_batches
        avg_dir_loss = epoch_dir_loss / num_batches
        avg_ortho_loss = epoch_ortho_loss / num_batches

        with torch.no_grad():
            diag_logits = model.get_structure_logits()
            raw_dir_snap = compute_directional_margin_loss(diag_logits, patel_direction_matrix).item()
            raw_ortho_snap = compute_feature_ortho_loss(
                model.node_emb_sender, model.node_emb_receiver,
            ).item()
            adj_sigmoid = get_current_structure_adj(model)
            adj_mean = adj_sigmoid.mean().item()
            sparsity_ratio = (adj_sigmoid < 0.5).float().mean().item()

        curr_adj = adj_sigmoid.cpu().numpy()
        epoch_score, epoch_details = compute_epoch_quality(
            curr_adj,
            patel_direction_cpu,
            patel_strength_cpu,
            top_k=quality_top_k,
        )
        quality_history.append({
            "epoch": epoch + 1,
            "score": epoch_score,
            **epoch_details,
        })
        if epoch_score > best_score:
            best_score = epoch_score
            best_adj = curr_adj.copy()
            best_epoch = epoch + 1
            best_quality_details = dict(epoch_details)
            marker = " ★ NEW BEST"
        else:
            marker = ""

        # Log progress
        if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Diff Loss: {avg_loss:.4f} | "
                  f"Sparsity Loss: {avg_sparsity:.4f} | "
                  f"Dir Loss(raw/w): {raw_dir_snap:.4f}/{avg_dir_loss:.4f} | "
                  f"Ortho Loss(raw/w): {raw_ortho_snap:.4f}/{avg_ortho_loss:.4f} | "
                  f"Adj Mean: {adj_mean:.3f} | "
                  f"Sparsity: {sparsity_ratio:.2%}")

            if model.use_temporal_encoder:
                collapse_metrics = diagnose_encoder_collapse(model, data_3d, device)
                print_collapse_diagnostics(collapse_metrics, epoch, num_epochs)
                collapse_history.append({"epoch": epoch + 1, **collapse_metrics})

            print(f"  [Quality] score={epoch_score:.4f} "
                  f"(agree={epoch_details['agreement']:.2%}[{epoch_details['high_conf_edges']}], "
                  f"margin={epoch_details['dir_margin']:.4f}, "
                  f"skel={epoch_details['skeleton_overlap']:.2%}, "
                  f"dens={epoch_details['density_factor']:.3f}, "
                  f"pair_dens={epoch_details['actual_pair_density']:.2%}/{epoch_details['target_pair_density']:.2%}) | "
                  f"Best: epoch {best_epoch} score={best_score:.4f}{marker}")

        # Record loss for every epoch

        loss_history.append(epoch_loss / num_batches)

    

    # Extract final adjacency matrix
    with torch.no_grad():
        last_adj = get_current_structure_adj(model).cpu().numpy()

    # Use best-epoch adjacency if available, otherwise fall back to last epoch
    if best_adj is not None:
        adj_matrix = best_adj
        print(f"\n[Best-Epoch] Using epoch {best_epoch} (score={best_score:.4f}) "
              f"instead of final epoch {num_epochs}")
    else:
        adj_matrix = last_adj
        best_epoch = num_epochs

    model.last_epoch_adj_matrix = last_adj
    model.best_epoch_adj_matrix = adj_matrix
    model.best_epoch_score = best_score
    model.best_epoch = best_epoch
    model.best_epoch_quality = best_quality_details
    model.quality_history = quality_history

    return model, adj_matrix, loss_history, collapse_history, best_epoch





def main():

    parser = argparse.ArgumentParser(description='Brain Connectivity Learning with DDM')

    parser.add_argument('--csv_path', type=str, default='../fMRI_dataset/sim4.csv',

                        help='Path to fMRI.csv file')

    parser.add_argument('--time_points', type=int, default=TIME_POINTS_PER_SUBJECT,

                        help='Number of time points per subject')

    parser.add_argument('--epochs', type=int, default=100,

                        help='Number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-3,

                        help='Learning rate')

    # NOTE: lambda_l1 is normalized by N*N, making it scale-invariant across graph sizes.

    # The actual L1 penalty = lambda_l1 * mean(|adj|). Typical range: 0.05 - 0.5

    parser.add_argument('--lambda_l1', type=float, default=0.02,

                        help='L1 regularization coefficient for sparsity (normalized by N^2)')

    parser.add_argument('--num_hidden', type=int, default=64,

                        help='Hidden dimension for DDM')

    parser.add_argument('--num_layers', type=int, default=2,

                        help='Number of GNN layers')

    parser.add_argument('--batch_size', type=int, default=4,

                        help='Batch size (number of subjects)')

    parser.add_argument('--save_path', type=str, default='./learned_brain_network.npy',

                        help='Path to save learned adjacency matrix')

    parser.add_argument('--device', type=str, default='cuda',

                        help='Device to use (cuda or cpu)')

    parser.add_argument('--seed', type=int, default=42,

                        help='Random seed')

    parser.add_argument('--log_interval', type=int, default=10,

                        help='Epochs between log messages')

    parser.add_argument('--top_k_edges', type=int, default=50,

                        help='Number of top undirected pairs for Patel skeleton/noise guidance')

    parser.add_argument('--debug_checks', action='store_true', default=False,

                        help='Run one-step debug checks (cos(x_t,x_encoded) and temporal encoder grad)')

    # Pretrain arguments
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                        help='Number of encoder pretrain epochs')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3,
                        help='Learning rate for encoder pretraining')
    # --pretrain_split_ratio 已废弃（新因果编码器使用自回归预训练，不需要 split）
    # parser.add_argument('--pretrain_split_ratio', type=float, default=0.75)
    # --skip_pretrain 已废弃（使用 --pretrain_epochs 0 代替）
    # parser.add_argument('--skip_pretrain', action='store_true', default=False)
    parser.add_argument('--skip_pretrain', action='store_true', default=False,
                        help='Skip encoder pretraining entirely (equivalent to --pretrain_epochs 0)')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='Path to existing pretrained encoder weights to load')

    parser.add_argument('--disable_temporal_encoder', action='store_true', default=False,
                        help='Disable temporal encoder and work directly on raw time series')



    args = parser.parse_args()

    

    # Set device

    if args.device == 'cuda' and not torch.cuda.is_available():

        print("CUDA not available, falling back to CPU")

        args.device = 'cpu'

    

    print("=" * 60)

    print("Brain Connectivity Learning with DDM")

    print("=" * 60)

    print(f"Device: {args.device}")

    print(f"Time points per subject: {args.time_points}")

    print(f"L1 regularization (lambda): {args.lambda_l1}")

    print("=" * 60)

    

    set_seed(args.seed)

    

    # Load and reshape fMRI data

    data_3d, data_2d, num_subjects, num_nodes = load_fmri_data(

        csv_path=args.csv_path,

        time_points_per_subject=args.time_points

    )

    

    # Step 1: Compute global Pearson correlation matrix

    # This will be used as init_features for the learnable adjacency matrix

    pearson_matrix = compute_global_pearson(data_2d)

    

    # Step 2: Compute Patel score / kappa / tau with separated semantics

    print("\nComputing Patel score/kappa/tau matrices...")

    patel_score_np, patel_kappa_np, patel_tau_np = compute_patel_components(data_2d.numpy())
    patel_score_matrix = torch.from_numpy(patel_score_np).float()
    patel_kappa_matrix = torch.from_numpy(patel_kappa_np).float()
    patel_tau_matrix = torch.from_numpy(patel_tau_np).float()

    print(f"Patel score range: [{patel_score_matrix.min():.4f}, {patel_score_matrix.max():.4f}]")
    print(f"Patel kappa range: [{patel_kappa_matrix.min():.4f}, {patel_kappa_matrix.max():.4f}]")
    print(f"Patel tau range:   [{patel_tau_matrix.min():.4f}, {patel_tau_matrix.max():.4f}]")

    # Step 3: Build an undirected noise-guide skeleton from positive Patel kappa
    noise_guide_adj, adj_binary, k_pairs, threshold = build_noise_guide_adjacency(
        patel_strength_matrix=torch.clamp(patel_kappa_matrix, min=0.0),
        top_k_pairs=args.top_k_edges,
    )

    print(f"Keeping top {k_pairs} undirected pairs (threshold: {threshold:.4f})")
    print(f"Noise guide adj: {adj_binary.sum().item() / 2:.0f} undirected pairs + {num_nodes} self-loops")

    

    # Create results folder with timestamp

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    result_dir = f'./results/run_{timestamp}'

    os.makedirs(result_dir, exist_ok=True)

    print(f"\nResults will be saved to: {result_dir}")

    

    # Step 6: Train model

    # - patel_score_matrix: asymmetric init for sender/receiver embeddings
    # - patel_tau_matrix: weak directional prior only
    # - patel_kappa_matrix: skeleton prior for proxy scoring / noise guidance

    model, adj_matrix, loss_history, collapse_history, best_epoch = train_brain_connectivity(

        data_3d=data_3d,

        pearson_matrix=pearson_matrix,  # Pearson for reference/saving

        num_nodes=num_nodes,

        time_points=args.time_points,

        noise_guide_adj=noise_guide_adj,  # For neighbor-based noise

        patel_matrix=patel_score_matrix,  # Asymmetric Patel score for SVD init
        patel_direction_matrix=patel_tau_matrix,  # Pure direction prior for margin loss
        patel_strength_matrix=torch.clamp(patel_kappa_matrix, min=0.0),  # Skeleton prior
        target_edge_count=k_pairs,

        num_epochs=args.epochs,

        learning_rate=args.lr,

        lambda_l1=args.lambda_l1,

        device=args.device,

        log_interval=args.log_interval,

        num_hidden=args.num_hidden,

        num_layers=args.num_layers,

        batch_size=args.batch_size,

        debug_checks=args.debug_checks,

        skip_pretrain=args.skip_pretrain,
        pretrain_checkpoint=args.pretrain_checkpoint,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        result_dir=result_dir,
        ddm_kwargs={'use_temporal_encoder': not args.disable_temporal_encoder},

    )

    

    # Plot and save loss curve

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)

    plt.ylabel('Diffusion Loss', fontsize=12)

    plt.title('Training Convergence', fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)

    

    # Add convergence annotation

    final_loss = loss_history[-1]

    plt.axhline(y=final_loss, color='r', linestyle='--', alpha=0.5, label=f'Final: {final_loss:.4f}')

    plt.legend()

    

    loss_plot_path = os.path.join(result_dir, 'loss_curve.png')

    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')

    plt.close()

    print(f"\nSaved loss curve to: {loss_plot_path}")

    # Plot encoder collapse diagnostics
    if collapse_history:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Encoder Collapse Diagnostics', fontsize=16, fontweight='bold')
        log_epochs = [m['epoch'] for m in collapse_history]

        metric_configs = [
            ("effective_rank", "Effective Rank", "tab:blue", None, "higher = healthier"),
            ("mean_cosine_sim", "Mean Cosine Similarity", "tab:red", 0.8, "< 0.8 healthy"),
            ("dead_dims_ratio", "Dead Dimensions Ratio", "tab:orange", 0.3, "< 0.3 healthy"),
            ("feature_std_mean", "Feature Std Mean", "tab:green", None, "higher = healthier"),
            ("inter_subject_var", "Inter-Subject Variance", "tab:purple", None, "higher = healthier"),
            ("encoder_weight_norm", "Encoder Weight Norm", "tab:brown", None, "reference"),
        ]

        for ax, (key, title, color, threshold, note) in zip(axes.flat, metric_configs):
            values = [m[key] for m in collapse_history]
            ax.plot(log_epochs, values, '-o', color=color, markersize=3, linewidth=2)
            if threshold is not None:
                ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label=f'warn={threshold}')
                ax.legend(fontsize=8)
            ax.set_title(f'{title}\n({note})', fontsize=11)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        collapse_plot_path = os.path.join(result_dir, 'collapse_diagnostics.png')
        plt.savefig(collapse_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved collapse diagnostics to: {collapse_plot_path}")

        # Save raw metrics as CSV
        collapse_df = pd.DataFrame(collapse_history)
        collapse_csv_path = os.path.join(result_dir, 'collapse_diagnostics.csv')
        collapse_df.to_csv(collapse_csv_path, index=False, float_format='%.6f')
        print(f"Saved collapse metrics to: {collapse_csv_path}")

    

    quality_history = getattr(model, 'quality_history', [])
    final_epoch_adj = getattr(model, 'last_epoch_adj_matrix', None)

    # Save adjacency matrix to results folder (both npy and csv)

    adj_save_path = os.path.join(result_dir, 'learned_adjacency.npy')

    np.save(adj_save_path, adj_matrix)

    

    # Save as CSV for easy viewing

    adj_csv_path = os.path.join(result_dir, 'learned_adjacency.csv')

    pd.DataFrame(adj_matrix).to_csv(adj_csv_path, index=False, header=False, float_format='%.4f')

    if final_epoch_adj is not None:
        final_adj_save_path = os.path.join(result_dir, 'final_epoch_adjacency.npy')
        final_adj_csv_path = os.path.join(result_dir, 'final_epoch_adjacency.csv')
        np.save(final_adj_save_path, final_epoch_adj)
        pd.DataFrame(final_epoch_adj).to_csv(
            final_adj_csv_path, index=False, header=False, float_format='%.4f'
        )

    

    # Save loss history (both npy and csv)

    np.save(os.path.join(result_dir, 'loss_history.npy'), np.array(loss_history))

    pd.DataFrame({'epoch': range(1, len(loss_history)+1), 'loss': loss_history}).to_csv(

        os.path.join(result_dir, 'loss_history.csv'), index=False

    )

    

    # Save config

    config = dict(vars(args))

    config['num_neighbors_avg'] = float(adj_binary.sum() / num_nodes) if 'adj_binary' in dir() else 0

    config['num_subjects'] = int(data_3d.shape[0])

    config['num_nodes'] = int(num_nodes)
    config['noise_guide_pairs'] = int(k_pairs)
    config['exported_epoch'] = int(best_epoch)
    config['best_proxy_score'] = float(getattr(model, 'best_epoch_score', -1.0))

    np.save(os.path.join(result_dir, 'config.npy'), config, allow_pickle=True)

    

    # Save Pearson matrix for reference (both npy and csv)

    np.save(os.path.join(result_dir, 'pearson_matrix.npy'), pearson_matrix.numpy())

    pd.DataFrame(pearson_matrix.numpy()).to_csv(

        os.path.join(result_dir, 'pearson_matrix.csv'), index=False, header=False, float_format='%.4f'

    )

    if quality_history:
        quality_df = pd.DataFrame(quality_history)
        quality_csv_path = os.path.join(result_dir, 'quality_history.csv')
        quality_df.to_csv(quality_csv_path, index=False, float_format='%.6f')
        print(f"Saved quality history to: {quality_csv_path}")

    # Save Patel matrices for reference (both npy and csv)
    np.save(os.path.join(result_dir, 'patel_score.npy'), patel_score_matrix.numpy())
    pd.DataFrame(patel_score_matrix.numpy()).to_csv(
        os.path.join(result_dir, 'patel_score.csv'),
        index=False,
        header=False,
        float_format='%.6f'
    )
    np.save(os.path.join(result_dir, 'patel_kappa.npy'), patel_kappa_matrix.numpy())
    pd.DataFrame(patel_kappa_matrix.numpy()).to_csv(
        os.path.join(result_dir, 'patel_kappa.csv'),
        index=False,
        header=False,
        float_format='%.6f'
    )
    np.save(os.path.join(result_dir, 'patel_tau.npy'), patel_tau_matrix.numpy())
    pd.DataFrame(patel_tau_matrix.numpy()).to_csv(
        os.path.join(result_dir, 'patel_tau.csv'),
        index=False,
        header=False,
        float_format='%.6f'
    )

    # Backward-compatible legacy name: Patel score matrix
    np.save(os.path.join(result_dir, 'patel_weights.npy'), patel_score_matrix.numpy())
    pd.DataFrame(patel_score_matrix.numpy()).to_csv(
        os.path.join(result_dir, 'patel_weights.csv'),
        index=False,
        header=False,
        float_format='%.6f'
    )

    

    print("=" * 60)

    print("Training Complete!")

    print("=" * 60)

    print(f"Results saved to: {result_dir}")

    print(f"  [Best Epoch: {best_epoch}/{args.epochs}]")

    print(f"  - loss_curve.png          <- 查看此图判断收敛")

    print(f"  - learned_adjacency.csv   <- best-epoch 导出的邻接矩阵")

    print(f"  - final_epoch_adjacency.csv <- 最后一轮邻接矩阵（用于对照）")

    print(f"  - loss_history.csv")

    print(f"  - quality_history.csv")

    print(f"  - pearson_matrix.csv")

    print(f"  - patel_score.csv / patel_kappa.csv / patel_tau.csv")

    print(f"  - config.npy")

    print(f"\nMatrix shape: {adj_matrix.shape}")

    print(f"Intensity stats:")

    print(f"  - Min:  {adj_matrix.min():.4f}")

    print(f"  - Max:  {adj_matrix.max():.4f}")

    print(f"  - Mean: {adj_matrix.mean():.4f}")

    print(f"  - Std:  {adj_matrix.std():.4f}")

    



if __name__ == '__main__':

    main()

