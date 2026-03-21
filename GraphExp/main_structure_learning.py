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
from typing import Any, Dict, List, Optional, Tuple

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
RAW_ADJ_CONVENTION = "effect_to_cause"
CAUSAL_ADJ_CONVENTION = "cause_to_effect"





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





def load_fmri_data(
    csv_path: str,
    time_points_per_subject: int = TIME_POINTS_PER_SUBJECT,
    subject_limit: int = -1,
    time_limit: int = -1,
):

    """

    Load and reshape fMRI data from CSV.

    

    Args:

        csv_path: Path to fMRI.csv file (NO HEADER)

        time_points_per_subject: Number of time points per subject in the CSV
        subject_limit: Optional cap on the number of subjects to keep; -1 keeps all
        time_limit: Optional cap on time points per subject after reshape; -1 keeps all

    

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

    

    num_subjects_full = total_rows // time_points_per_subject

    print(f"Detected {num_subjects_full} subjects with {time_points_per_subject} time points each.")

    if subject_limit == 0 or subject_limit < -1:
        raise ValueError(f"subject_limit must be -1 or a positive integer, got {subject_limit}")
    if time_limit == 0 or time_limit < -1:
        raise ValueError(f"time_limit must be -1 or a positive integer, got {time_limit}")

    effective_num_subjects = (
        num_subjects_full if subject_limit < 0 else min(num_subjects_full, subject_limit)
    )
    effective_time_points = (
        time_points_per_subject if time_limit < 0 else min(time_points_per_subject, time_limit)
    )

    if effective_num_subjects != num_subjects_full:
        print(f"Applying subject_limit={subject_limit}: keeping {effective_num_subjects} subjects.")
    if effective_time_points != time_points_per_subject:
        print(
            f"Applying time_limit={time_limit}: keeping "
            f"{effective_time_points}/{time_points_per_subject} time points per subject."
        )

    data_3d_np = data.reshape(num_subjects_full, time_points_per_subject, num_nodes)
    data_3d_np = data_3d_np[:effective_num_subjects, :effective_time_points, :]

    # Keep 2D for Pearson / Patel computation after any requested truncation.
    data_2d = torch.from_numpy(data_3d_np.reshape(-1, num_nodes).astype(np.float32)).float()

    # Transpose to [Num_Subjects, N, TIME_POINTS] for model input
    data_3d = np.transpose(data_3d_np, (0, 2, 1))
    data_3d = torch.from_numpy(data_3d).float()

    print(f"Reshaped data to: {data_3d.shape} [Num_Subjects, N, TIME_POINTS]")

    return data_3d, data_2d, effective_num_subjects, num_nodes





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


def build_structure_init_matrix(
    mode: str,
    patel_score_matrix: torch.Tensor,
    patel_kappa_matrix: torch.Tensor,
    pearson_matrix: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Build the matrix used only for structure-embedding initialization."""
    if mode == 'patel_score':
        init_matrix = patel_score_matrix.clone()
    elif mode == 'patel_score_t':
        init_matrix = patel_score_matrix.t().clone()
    elif mode == 'neg_patel_score':
        init_matrix = (-patel_score_matrix).clone()
    elif mode == 'neg_patel_score_t':
        init_matrix = (-patel_score_matrix).t().clone()
    elif mode == 'patel_kappa':
        init_matrix = torch.clamp(patel_kappa_matrix, min=0.0).clone()
    elif mode == 'pearson':
        init_matrix = pearson_matrix.clone()
    elif mode == 'random':
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
        init_matrix = torch.randn(
            pearson_matrix.shape,
            generator=generator,
            dtype=pearson_matrix.dtype,
        )
    else:
        raise ValueError(f"Unsupported structure init mode: {mode}")

    init_matrix.fill_diagonal_(0.0)
    return init_matrix


def to_causal_matrix_torch(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert model-internal adjacency/logit convention to causal convention.

    Internal/raw convention:
        A_raw[effect, cause] is high when the source node helps denoise the target.
    Causal convention:
        A_causal[cause, effect] is high for cause -> effect.
    """
    return matrix.transpose(-1, -2)


def to_causal_matrix_np(matrix: np.ndarray) -> np.ndarray:
    """NumPy version of `to_causal_matrix_torch`."""
    return np.asarray(matrix).T.copy()


def build_undirected_kappa_skeleton(
    patel_strength_matrix: torch.Tensor,
    top_k_pairs: int,
    selection_mode: str = 'topk',
):
    """Select an undirected kappa skeleton by either fixed top-k or max-gap cutoff."""
    if selection_mode not in {'topk', 'maxgap'}:
        raise ValueError(f"selection_mode must be 'topk' or 'maxgap', got {selection_mode}")

    num_nodes = patel_strength_matrix.shape[0]
    device = patel_strength_matrix.device
    dtype = patel_strength_matrix.dtype

    eye = torch.eye(num_nodes, device=device, dtype=dtype)
    pair_strength = torch.maximum(patel_strength_matrix, patel_strength_matrix.t())
    pair_strength = torch.clamp(pair_strength, min=0.0) * (1.0 - eye)

    triu_i, triu_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    flat_strength = pair_strength[triu_i, triu_j]
    num_pairs = flat_strength.numel()

    selected_idx = flat_strength.new_zeros((0,), dtype=torch.long)
    threshold = 0.0
    selection_gap = 0.0

    if selection_mode == 'topk':
        k_pairs = min(max(int(top_k_pairs), 0), num_pairs)
        if k_pairs > 0 and num_pairs > 0:
            selected_idx = torch.topk(flat_strength, k_pairs).indices
            threshold = float(flat_strength[selected_idx].min().item())
    else:
        positive_idx = torch.nonzero(flat_strength > 0, as_tuple=False).flatten()
        positive_count = int(positive_idx.numel())
        if positive_count == 0:
            k_pairs = 0
        elif positive_count == 1:
            k_pairs = 1
            selected_idx = positive_idx
            threshold = float(flat_strength[selected_idx[0]].item())
        else:
            positive_strength = flat_strength[positive_idx]
            sorted_strength, sort_order = torch.sort(positive_strength, descending=True)
            gaps = sorted_strength[:-1] - sorted_strength[1:]
            gap_pos = int(torch.argmax(gaps).item())
            k_pairs = gap_pos + 1
            selected_idx = positive_idx[sort_order[:k_pairs]]
            threshold = float(sorted_strength[k_pairs - 1].item())
            selection_gap = float(gaps[gap_pos].item())

    adj_binary = torch.zeros_like(pair_strength)
    if selected_idx.numel() > 0:
        src = triu_i[selected_idx]
        dst = triu_j[selected_idx]
        adj_binary[src, dst] = 1.0
        adj_binary[dst, src] = 1.0

    return adj_binary, k_pairs, threshold, selection_gap


def build_noise_guide_adjacency(
    patel_strength_matrix: torch.Tensor,
    top_k_pairs: int,
    selection_mode: str = 'topk',
):
    """
    Build a symmetric row-normalized adjacency for neighbor-based noise.

    `patel_strength_matrix` should encode undirected skeleton strength. In practice
    we pass positive Patel kappa so skeleton selection is decoupled from direction.
    """
    num_nodes = patel_strength_matrix.shape[0]
    device = patel_strength_matrix.device
    dtype = patel_strength_matrix.dtype
    eye = torch.eye(num_nodes, device=device, dtype=dtype)
    adj_binary, k_pairs, threshold, selection_gap = build_undirected_kappa_skeleton(
        patel_strength_matrix,
        top_k_pairs=top_k_pairs,
        selection_mode=selection_mode,
    )
    adj_with_self = adj_binary + eye
    degree = adj_with_self.sum(dim=1, keepdim=True)
    noise_guide_adj = adj_with_self / (degree + 1e-9)
    return noise_guide_adj, adj_binary, k_pairs, threshold, selection_gap


def build_directed_noise_guide_adjacency(
    patel_kappa: torch.Tensor,
    patel_tau: torch.Tensor,
    top_k_pairs: int,
    direction_alpha: float = 0.5,
):
    """
    Build a direction-biased row-normalized adjacency for neighbor-based noise.

    Unlike `build_noise_guide_adjacency` which symmetrizes, this version uses
    Patel tau to assign asymmetric weights: the causal direction gets higher weight.

    Args:
        patel_kappa: Symmetric association strength [N, N]
        patel_tau: Directional prior [N, N] (Pate.m convention: -tau returned)
        top_k_pairs: Number of undirected pairs to keep
        direction_alpha: Bias strength (0=symmetric, 1=max asymmetry)
    """
    num_nodes = patel_kappa.shape[0]
    device = patel_kappa.device
    dtype = patel_kappa.dtype

    eye = torch.eye(num_nodes, device=device, dtype=dtype)
    # Skeleton selection: symmetric kappa, same as undirected version
    sym_kappa = torch.maximum(patel_kappa, patel_kappa.t())
    sym_kappa = torch.clamp(sym_kappa, min=0.0) * (1.0 - eye)

    triu_i, triu_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    flat_strength = sym_kappa[triu_i, triu_j]
    num_pairs = flat_strength.numel()
    k_pairs = min(max(int(top_k_pairs), 0), num_pairs)

    adj = eye.clone()  # Start with self-loops
    if k_pairs > 0 and num_pairs > 0:
        top_idx = torch.topk(flat_strength, k_pairs).indices
        for idx_t in top_idx:
            idx_val = idx_t.item()
            i = triu_i[idx_val].item()
            j = triu_j[idx_val].item()
            tau_val = patel_tau[i, j].item()
            bias = direction_alpha * abs(tau_val)
            # tau_val > 0 → i→j stronger; tau_val < 0 → j→i stronger
            adj[i, j] += 1.0 + bias * (1.0 if tau_val > 0 else -1.0)
            adj[j, i] += 1.0 + bias * (1.0 if tau_val < 0 else -1.0)

    # Clamp to non-negative before row normalization
    adj = torch.clamp(adj, min=0.0)
    row_sum = adj.sum(dim=1, keepdim=True)
    noise_guide_adj = adj / (row_sum + 1e-9)
    return noise_guide_adj


def row_normalize_noise_guide_adjacency(
    adjacency: torch.Tensor,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """Clamp to non-negative and row-normalize into a valid noise-guide matrix."""
    if adjacency.dim() != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected square adjacency, got shape {tuple(adjacency.shape)}")

    adj = torch.clamp(adjacency, min=0.0)
    if add_self_loops:
        eye = torch.eye(adj.shape[0], dtype=adj.dtype, device=adj.device)
        adj = adj + eye
    row_sum = adj.sum(dim=1, keepdim=True)
    return adj / (row_sum + 1e-9)


@torch.no_grad()
def build_detached_learned_noise_guide_adjacency(model: DDM) -> Optional[torch.Tensor]:
    """
    Convert the current learned raw adjacency into a detached row-stochastic noise guide.

    We keep the model's internal/raw convention here: row `i` defines which source
    nodes contribute to the noise statistics of target/effect node `i`.
    """
    if not getattr(model, "structure_learning_mode", False):
        return None
    learned_adj_raw = model.get_structure_adj().detach()
    return row_normalize_noise_guide_adjacency(learned_adj_raw, add_self_loops=True)


@torch.no_grad()
def compute_noise_guide_probe_diagnostics(model: DDM, x: torch.Tensor) -> Dict[str, float]:
    """Compare Patel-vs-learned noise guides using the same probe timestep and Gaussian eps."""
    default_stats = {
        "noise_probe_available": 0.0,
        "noise_probe_timestep": 0.0,
        "noise_probe_patel_loss": 0.0,
        "noise_probe_blend50_loss": 0.0,
        "noise_probe_learned_loss": 0.0,
        "noise_probe_delta_blend50_minus_patel": 0.0,
        "noise_probe_delta_learned_minus_patel": 0.0,
        "noise_probe_ratio_blend50_over_patel": 0.0,
        "noise_probe_ratio_learned_over_patel": 0.0,
        "noise_probe_guide_l1_mean": 0.0,
    }
    if not getattr(model, "structure_learning_mode", False):
        return default_stats
    if getattr(model, "noise_guide_adj", None) is None:
        return default_stats

    learned_noise_guide_adj = build_detached_learned_noise_guide_adjacency(model)
    if learned_noise_guide_adj is None:
        return default_stats

    clean = model.prepare_clean_target(x)
    if clean.shape[-1] <= 1:
        return default_stats

    patel_noise_guide_adj = model.noise_guide_adj.detach()
    blend50_noise_guide_adj = 0.5 * patel_noise_guide_adj + 0.5 * learned_noise_guide_adj
    probe_timestep = max(int(model.T // 2), 0)
    probe_t = torch.full((clean.shape[0],), probe_timestep, dtype=torch.long, device=clean.device)
    probe_eps_shape = (1, *clean.shape) if clean.dim() == 2 else tuple(clean.shape)
    probe_eps = torch.from_numpy(
        np.random.default_rng(0).standard_normal(size=probe_eps_shape).astype(np.float32)
    ).to(clean.device)

    if getattr(model, "structure_learning_mode", False):
        g, edge_weight = model._get_structure_graph(clean.device)
    else:
        g, edge_weight = None, None

    def compute_probe_loss(noise_guide_adj: torch.Tensor) -> float:
        x_t, time_embed, _ = model.sample_q(
            probe_t,
            clean,
            g,
            eps_override=probe_eps,
            noise_guide_adj_override=noise_guide_adj,
        )
        return float(model.node_denoising(clean, x_t, time_embed, g, edge_weight=edge_weight).item())

    patel_loss = compute_probe_loss(patel_noise_guide_adj)
    blend50_loss = compute_probe_loss(blend50_noise_guide_adj)
    learned_loss = compute_probe_loss(learned_noise_guide_adj)

    return {
        "noise_probe_available": 1.0,
        "noise_probe_timestep": float(probe_timestep),
        "noise_probe_patel_loss": patel_loss,
        "noise_probe_blend50_loss": blend50_loss,
        "noise_probe_learned_loss": learned_loss,
        "noise_probe_delta_blend50_minus_patel": blend50_loss - patel_loss,
        "noise_probe_delta_learned_minus_patel": learned_loss - patel_loss,
        "noise_probe_ratio_blend50_over_patel": blend50_loss / (patel_loss + 1e-8),
        "noise_probe_ratio_learned_over_patel": learned_loss / (patel_loss + 1e-8),
        "noise_probe_guide_l1_mean": float(
            torch.abs(learned_noise_guide_adj - patel_noise_guide_adj).mean().item()
        ),
    }


def get_current_structure_logits(model: DDM, causal: bool = False, detach: bool = False) -> torch.Tensor:
    """Fetch the current structure logits under raw or causal convention."""
    logits = model.get_structure_logits()
    if causal:
        logits = to_causal_matrix_torch(logits)
    return logits.detach() if detach else logits


def get_current_directional_logits(model: DDM, causal: bool = False, detach: bool = False) -> torch.Tensor:
    """Fetch the logits that should carry directional supervision."""
    if getattr(model, "structure_parameterization", "coupled") == "support_direction":
        logits = model.get_direction_logits()
    else:
        logits = model.get_structure_logits()
    if causal:
        logits = to_causal_matrix_torch(logits)
    return logits.detach() if detach else logits


def get_direction_branch_parameters(model: DDM) -> List[torch.nn.Parameter]:
    """Return the separate direction-branch parameters when factorization is enabled."""
    if getattr(model, "structure_parameterization", "coupled") != "support_direction":
        return []
    params: List[torch.nn.Parameter] = []
    for name in ("direction_emb_sender", "direction_emb_receiver"):
        param = getattr(model, name, None)
        if isinstance(param, torch.nn.Parameter):
            params.append(param)
    return params


def flatten_gradient_parts(grad_parts: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Concatenate gradient tensors into one flat vector, skipping missing grads."""
    flat_parts = [grad.reshape(-1) for grad in grad_parts if grad is not None]
    if not flat_parts:
        return None
    return torch.cat(flat_parts, dim=0)


def build_training_optimizer(
    model: DDM,
    learning_rate: float,
    direction_lr_multiplier: float,
):
    """Create optimizer groups so the direction branch can use a separate LR."""
    direction_params = [
        param for param in get_direction_branch_parameters(model) if param.requires_grad
    ]
    direction_param_ids = {id(param) for param in direction_params}
    base_params = [
        param for param in model.parameters()
        if param.requires_grad and id(param) not in direction_param_ids
    ]

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": learning_rate})
    if direction_params:
        param_groups.append(
            {
                "params": direction_params,
                "lr": learning_rate * direction_lr_multiplier,
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters found when building the optimizer.")

    optimizer = torch.optim.Adam(param_groups)
    optimizer_stats = {
        "base_param_count": sum(param.numel() for param in base_params),
        "direction_param_count": sum(param.numel() for param in direction_params),
        "base_lr": float(learning_rate),
        "direction_lr": (
            float(learning_rate * direction_lr_multiplier) if direction_params else 0.0
        ),
        "has_direction_group": int(bool(direction_params)),
    }
    return optimizer, optimizer_stats


def freeze_direction_branch(model: DDM) -> int:
    """Freeze the separate direction branch in-place and clear stale grads."""
    frozen_param_count = 0
    for param in get_direction_branch_parameters(model):
        if param.requires_grad:
            param.requires_grad = False
            param.grad = None
            frozen_param_count += param.numel()
    return frozen_param_count


def compute_direction_grad_alignment_diagnostics(
    model: DDM,
    x: torch.Tensor,
    num_nodes: int,
    lambda_l1: float,
    directional_prior_mode: str,
    lag_direction_source: str,
    patel_direction_matrix: torch.Tensor,
    directional_pair_gate_matrix: Optional[torch.Tensor],
    lambda_dir_effective: float,
    seed: int = 0,
) -> Dict[str, float]:
    """Compare diffusion-vs-directional gradients on the separate direction branch."""
    default_stats = {
        "grad_probe_available": 0.0,
        "grad_probe_lambda_dir": float(lambda_dir_effective),
        "grad_probe_diff_norm": 0.0,
        "grad_probe_dir_norm_raw": 0.0,
        "grad_probe_dir_norm_weighted": 0.0,
        "grad_probe_dir_to_diff_norm_ratio": 0.0,
        "grad_probe_cosine": 0.0,
        "grad_probe_cosine_negative": 0.0,
    }
    direction_params = [param for param in get_direction_branch_parameters(model) if param.requires_grad]
    if not direction_params:
        return default_stats

    was_training = model.training
    cuda_devices: List[int] = []
    if x.is_cuda:
        cuda_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
        cuda_devices = [cuda_index]

    try:
        model.train()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)

            loss, _ = model(g=None, x=x)
            adj_weights = model.get_structure_adj()
            n_off_diag = num_nodes * num_nodes - num_nodes
            l1_norm = torch.norm(adj_weights, p=1)
            if n_off_diag > 0:
                sparsity_loss = lambda_l1 * (l1_norm / n_off_diag)
            else:
                sparsity_loss = torch.tensor(0.0, device=x.device)
            sender_norms = torch.norm(model.node_emb_sender, dim=1)
            receiver_norms = torch.norm(model.node_emb_receiver, dim=1)
            hub_loss = 0.01 * (sender_norms.var() + receiver_norms.var())
            loss_ddm_main = loss + sparsity_loss + hub_loss

            direction_prior_matrix = compute_online_direction_prior_matrix(
                model=model,
                x=x,
                mode=directional_prior_mode,
                patel_direction_matrix=patel_direction_matrix,
                lag_direction_source=lag_direction_source,
            )
            causal_logits = get_current_directional_logits(model, causal=True)
            raw_loss_dir = compute_directional_margin_loss(
                causal_logits,
                direction_prior_matrix,
                pair_gate_matrix=directional_pair_gate_matrix,
            )
            weighted_loss_dir = raw_loss_dir * float(lambda_dir_effective)

            grad_diff_parts = torch.autograd.grad(
                loss_ddm_main,
                direction_params,
                retain_graph=True,
                allow_unused=True,
            )
            if weighted_loss_dir.requires_grad and abs(float(lambda_dir_effective)) > 0.0:
                grad_dir_parts = torch.autograd.grad(
                    weighted_loss_dir,
                    direction_params,
                    allow_unused=True,
                )
            else:
                grad_dir_parts = [None for _ in direction_params]
    finally:
        if not was_training:
            model.eval()

    grad_diff_flat = flatten_gradient_parts(list(grad_diff_parts))
    grad_dir_flat = flatten_gradient_parts(list(grad_dir_parts))
    if grad_diff_flat is None:
        return default_stats

    diff_norm = float(grad_diff_flat.norm().item())
    dir_weighted_norm = float(grad_dir_flat.norm().item()) if grad_dir_flat is not None else 0.0
    dir_raw_norm = (
        dir_weighted_norm / max(abs(float(lambda_dir_effective)), 1e-12)
        if abs(float(lambda_dir_effective)) > 0.0
        else 0.0
    )
    if grad_dir_flat is not None and diff_norm > 0.0 and dir_weighted_norm > 0.0:
        cosine = float(F.cosine_similarity(grad_diff_flat, grad_dir_flat, dim=0).item())
    else:
        cosine = 0.0

    return {
        "grad_probe_available": 1.0,
        "grad_probe_lambda_dir": float(lambda_dir_effective),
        "grad_probe_diff_norm": diff_norm,
        "grad_probe_dir_norm_raw": dir_raw_norm,
        "grad_probe_dir_norm_weighted": dir_weighted_norm,
        "grad_probe_dir_to_diff_norm_ratio": dir_weighted_norm / (diff_norm + 1e-12),
        "grad_probe_cosine": cosine,
        "grad_probe_cosine_negative": float(cosine < 0.0),
    }


@torch.no_grad()
def get_current_structure_adj(model: DDM, causal: bool = False) -> torch.Tensor:
    """Fetch the current adjacency under raw or causal convention."""
    adj = model.get_structure_adj().detach()
    return to_causal_matrix_torch(adj) if causal else adj


def compute_single_aux_lambda(
    epoch: int,
    num_epochs: int,
    loss_ddm_main: torch.Tensor,
    raw_loss: torch.Tensor,
    prev_lambda: float,
    target_ratio: float,
    warmup_epochs: int = 5,
    schedule: str = "cosine_anneal",
):
    """Warmup + ratio-adaptive scaling + optional anneal + EMA smoothing for one auxiliary loss."""
    if target_ratio <= 0.0 or epoch < warmup_epochs:
        return 0.0

    post_warmup_epochs = max(num_epochs - warmup_epochs, 1)
    ramp_epochs = max(1, min(10, post_warmup_epochs))
    ramp = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
    if schedule == "plateau":
        anneal = 1.0
    elif schedule == "cosine_anneal":
        anneal_progress = float(epoch - warmup_epochs) / float(max(post_warmup_epochs - 1, 1))
        anneal = 0.5 * (1.0 + math.cos(math.pi * anneal_progress))
    else:
        raise ValueError(f"Unsupported auxiliary schedule: {schedule}")
    epoch_factor = ramp * anneal

    scale = (loss_ddm_main.detach() * target_ratio) / (raw_loss.detach() + 1e-6)
    lambda_raw = min(scale.item() * epoch_factor, 0.5)

    ema_alpha = 0.1
    lambda_val = ema_alpha * lambda_raw + (1 - ema_alpha) * prev_lambda

    max_change = 0.1
    if prev_lambda > 0:
        lambda_val = max(prev_lambda * (1 - max_change),
                         min(prev_lambda * (1 + max_change), lambda_val))
    return lambda_val


def compute_fixed_aux_weight(
    epoch: int,
    target_weight: float,
    warmup_epochs: int = 0,
    ramp_epochs: int = 1,
) -> float:
    """Warmup + linear ramp for fixed-weight auxiliary losses."""
    if target_weight <= 0.0 or epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 1:
        return float(target_weight)
    progress = float(epoch - warmup_epochs + 1) / float(max(ramp_epochs, 1))
    return float(target_weight) * min(max(progress, 0.0), 1.0)


def compute_auxiliary_lambdas(
    epoch: int,
    num_epochs: int,
    loss_ddm_main: torch.Tensor,
    raw_loss_dir: torch.Tensor,
    raw_loss_ortho: torch.Tensor,
    raw_loss_cross: torch.Tensor,
    prev_lambda_dir: float,
    prev_lambda_ortho: float,
    prev_lambda_cross: float,
    dir_target_ratio: float = 0.01,
    ortho_target_ratio: float = 0.005,
    cross_target_ratio: float = 0.02,
    warmup_epochs: int = 5,
    dir_schedule: str = "cosine_anneal",
    cross_warmup_epochs: int = 3,
    cross_schedule: str = "cosine_anneal",
):
    """Compute adaptive weights for directional, orthogonality, and cross-prediction losses."""
    lambda_dir = compute_single_aux_lambda(
        epoch=epoch,
        num_epochs=num_epochs,
        loss_ddm_main=loss_ddm_main,
        raw_loss=raw_loss_dir,
        prev_lambda=prev_lambda_dir,
        target_ratio=dir_target_ratio,
        warmup_epochs=warmup_epochs,
        schedule=dir_schedule,
    )
    lambda_ortho = compute_single_aux_lambda(
        epoch=epoch,
        num_epochs=num_epochs,
        loss_ddm_main=loss_ddm_main,
        raw_loss=raw_loss_ortho,
        prev_lambda=prev_lambda_ortho,
        target_ratio=ortho_target_ratio,
        warmup_epochs=warmup_epochs,
    )
    lambda_cross = compute_single_aux_lambda(
        epoch=epoch,
        num_epochs=num_epochs,
        loss_ddm_main=loss_ddm_main,
        raw_loss=raw_loss_cross,
        prev_lambda=prev_lambda_cross,
        target_ratio=cross_target_ratio,
        warmup_epochs=cross_warmup_epochs,
        schedule=cross_schedule,
    )
    return lambda_dir, lambda_ortho, lambda_cross



# ============================================================================
# AUXILIARY LOSSES: Directional Prior & Feature Decoupling
# ============================================================================

def build_kappa_gate_matrix(
    patel_strength_matrix: torch.Tensor,
    quantile: float = 0.5,
) -> Tuple[torch.Tensor, float, float]:
    """Build a symmetric boolean gate that keeps only high-kappa unordered pairs."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"directional_kappa_gate_quantile must be in [0, 1], got {quantile}")

    num_nodes = patel_strength_matrix.shape[0]
    gate = torch.zeros_like(patel_strength_matrix, dtype=torch.bool)
    if num_nodes <= 1:
        return gate, 0.0, 0.0

    offdiag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=patel_strength_matrix.device)
    offdiag_vals = patel_strength_matrix.masked_select(offdiag_mask)
    positive_vals = offdiag_vals[offdiag_vals > 0]
    if positive_vals.numel() == 0:
        return gate, 0.0, 0.0

    threshold = float(torch.quantile(positive_vals, quantile).item())
    gate = (patel_strength_matrix >= threshold) & offdiag_mask
    pair_frac = float(gate.float().mean().item())
    return gate, threshold, pair_frac


def compute_directional_margin_loss(
    logits,
    direction_prior_matrix,
    margin=1.0,
    pair_gate_matrix: Optional[torch.Tensor] = None,
):
    """
    基于 Patel 算法的高置信度先验，在 causal Logit 空间计算带 Margin 的方向引导损失。

    q_threshold 自适应：取 |delta_P| 非零值的中位数，确保约 50% 的边参与约束。
    margin 自适应：在有效边（w>0）上对 sign(delta_P)*D 取 25 分位数，
                  并以下界 margin 保持正的监督压力，避免全 tie 时梯度归零。
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
    if pair_gate_matrix is not None:
        active_mask = active_mask & pair_gate_matrix.bool()
    if active_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    w = active_mask.float() * abs_delta_prior

    D = logits - logits.t()
    signed_D = torch.sign(delta_prior) * D  # 正值=方向正确，负值=方向错误

    # 自适应 margin：有效边上 signed_D 的 25 分位数（detach，不参与梯度）
    # quantile(0.25) 意味着 25% 的有效边 signed_D ≤ margin，即 25% 违反约束。
    # 但当所有方向 logit 都 tie（例如 factorized + zero init）时，该分位数会退化到 0，
    # 使得损失直接失活。用 base margin 作为下界，保持方向监督可学习。
    active_signed_D = signed_D[active_mask].detach()
    if active_signed_D.numel() > 0:
        adaptive_margin = max(float(margin), float(active_signed_D.quantile(0.25).item()))
    else:
        adaptive_margin = float(margin)

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


def compute_incoming_parent_profile(
    adj_causal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized entropy, effective parents, and active-target mask."""
    if adj_causal.dim() != 2 or adj_causal.shape[0] != adj_causal.shape[1]:
        raise ValueError(f"Expected square causal adjacency, got shape {tuple(adj_causal.shape)}")

    num_nodes = adj_causal.shape[0]
    if num_nodes <= 1:
        empty = adj_causal.new_zeros((num_nodes,))
        return empty, empty, torch.zeros((num_nodes,), dtype=torch.bool, device=adj_causal.device)

    incoming_mass = adj_causal.sum(dim=0, keepdim=True)
    probs = adj_causal / incoming_mass.clamp_min(1e-8)
    safe_probs = probs.clamp_min(1e-8)
    entropy = -(probs * safe_probs.log()).sum(dim=0)
    effective_parents = torch.exp(entropy)

    max_entropy = math.log(max(num_nodes - 1, 1))
    if max_entropy <= 0.0:
        normalized_entropy = adj_causal.new_zeros((num_nodes,))
    else:
        normalized_entropy = entropy / max_entropy

    active_targets = incoming_mass.squeeze(0) > 1e-8
    return normalized_entropy, effective_parents, active_targets


def compute_incoming_entropy_loss(adj_causal: torch.Tensor) -> torch.Tensor:
    """Penalize diffuse incoming-parent distributions without shrinking total edge mass."""
    normalized_entropy, _, active_targets = compute_incoming_parent_profile(adj_causal)
    if not active_targets.any():
        return adj_causal.new_tensor(0.0)
    return normalized_entropy[active_targets].mean()


def compute_excess_effective_parents_loss(
    adj_causal: torch.Tensor,
    target_effective_parents: float,
) -> torch.Tensor:
    """Penalize only the effective-parent mass above a target value."""
    if target_effective_parents <= 0.0:
        return adj_causal.new_tensor(0.0)

    _, effective_parents, active_targets = compute_incoming_parent_profile(adj_causal)
    if not active_targets.any():
        return adj_causal.new_tensor(0.0)

    excess = F.relu(effective_parents[active_targets] - target_effective_parents)
    return excess.mean()


@torch.no_grad()
def compute_incoming_parent_diagnostics(adj_causal: torch.Tensor) -> Dict[str, float]:
    """Summarize incoming-parent concentration under causal convention."""
    normalized_entropy, effective_parents, active_targets = compute_incoming_parent_profile(adj_causal)
    if not active_targets.any():
        return {
            "adj_parent_entropy_mean": 0.0,
            "adj_eff_parents_mean": 0.0,
            "adj_eff_parents_p90": 0.0,
        }

    active_entropy = normalized_entropy[active_targets]
    active_effective_parents = effective_parents[active_targets]
    return {
        "adj_parent_entropy_mean": float(active_entropy.mean().item()),
        "adj_eff_parents_mean": float(active_effective_parents.mean().item()),
        "adj_eff_parents_p90": float(torch.quantile(active_effective_parents, 0.90).item()),
    }


def compute_ungated_symmetry_loss(
    adj_causal: torch.Tensor,
    pair_gate_matrix: Optional[torch.Tensor],
) -> torch.Tensor:
    """Encourage low-kappa / ungated pairs to stay close to symmetric ties."""
    if pair_gate_matrix is None:
        return adj_causal.new_tensor(0.0)

    num_nodes = adj_causal.shape[0]
    offdiag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=adj_causal.device)
    ungated_mask = offdiag_mask & (~pair_gate_matrix.bool())
    if not ungated_mask.any():
        return adj_causal.new_tensor(0.0)

    asymmetry = torch.abs(adj_causal - adj_causal.t())
    return asymmetry.masked_select(ungated_mask).mean()


@torch.no_grad()
def compute_ungated_asymmetry_diagnostics(
    adj_causal: torch.Tensor,
    pair_gate_matrix: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Summarize asymmetry magnitude on pairs outside the directional kappa gate."""
    if pair_gate_matrix is None:
        return {
            "adj_ungated_asym_mean": 0.0,
            "adj_ungated_asym_median": 0.0,
            "adj_ungated_asym_p90": 0.0,
        }

    num_nodes = adj_causal.shape[0]
    offdiag_mask = ~torch.eye(num_nodes, dtype=torch.bool, device=adj_causal.device)
    ungated_mask = offdiag_mask & (~pair_gate_matrix.bool())
    vals = torch.abs(adj_causal - adj_causal.t()).masked_select(ungated_mask)
    if vals.numel() == 0:
        return {
            "adj_ungated_asym_mean": 0.0,
            "adj_ungated_asym_median": 0.0,
            "adj_ungated_asym_p90": 0.0,
        }
    return {
        "adj_ungated_asym_mean": float(vals.mean().item()),
        "adj_ungated_asym_median": float(vals.median().item()),
        "adj_ungated_asym_p90": float(torch.quantile(vals, 0.90).item()),
    }


def zscore_per_node_time(x: torch.Tensor) -> torch.Tensor:
    """Per-node z-score along the time dimension."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-6)


@torch.no_grad()
def compute_pairwise_lagged_score_matrix(
    source_node_time: torch.Tensor,
    target_node_time: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise lagged similarity scores for all ordered node pairs."""
    source_past_z = zscore_per_node_time(source_node_time[:, :-1])
    target_future_z = zscore_per_node_time(target_node_time[:, 1:])
    num_steps = max(source_past_z.shape[-1], 1)
    return torch.einsum('nt,mt->nm', source_past_z, target_future_z) / float(num_steps)


@torch.no_grad()
def compute_online_direction_prior_matrix(
    model: DDM,
    x: torch.Tensor,
    mode: str,
    patel_direction_matrix: torch.Tensor,
    lag_direction_source: str = "raw",
) -> torch.Tensor:
    """Build the directional-prior matrix used by the margin loss."""
    if mode == "patel":
        return patel_direction_matrix
    if mode != "lag_corr":
        raise ValueError(f"Unsupported directional_prior_mode: {mode}")

    if lag_direction_source == "raw":
        source = x
    elif lag_direction_source == "encoder":
        source = model.prepare_clean_target(x)
    else:
        raise ValueError(f"Unsupported lag_direction_source: {lag_direction_source}")
    return compute_pairwise_lagged_score_matrix(source, x)


def build_cross_prediction_aggregation_weights(
    model: DDM,
    aggregation: str = "mean",
    softmax_temp: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build causal adjacency and cross-pred aggregation weights."""
    adj_causal = to_causal_matrix_torch(model.get_structure_adj())
    if aggregation == "mean":
        in_degree = adj_causal.sum(dim=0, keepdim=True).clamp_min(1e-8)
        agg_weights = adj_causal / in_degree
        return adj_causal, agg_weights
    if aggregation == "softmax":
        if softmax_temp <= 0.0:
            raise ValueError(f"softmax_temp must be positive, got {softmax_temp}")
        causal_logits = get_current_structure_logits(model, causal=True)
        diag_mask = torch.eye(
            causal_logits.shape[0], dtype=torch.bool, device=causal_logits.device,
        )
        masked_logits = causal_logits.masked_fill(diag_mask, float("-inf"))
        agg_weights = torch.softmax(masked_logits / softmax_temp, dim=0)
        agg_weights = agg_weights.masked_fill(diag_mask, 0.0)
        return adj_causal, agg_weights
    raise ValueError(f"Unsupported cross-pred aggregation: {aggregation}")


def get_cross_prediction_tensors(
    model: DDM,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build frozen source features, prediction, target, and causal adjacency for cross-prediction."""
    with torch.no_grad():
        h = model.prepare_clean_target(x)

    h_past = h[:, :-1]
    x_future = x[:, 1:]

    adj_causal, agg_weights = build_cross_prediction_aggregation_weights(
        model,
        aggregation=getattr(model, "cross_pred_aggregation", "mean"),
        softmax_temp=getattr(model, "cross_pred_softmax_temp", 1.0),
    )
    pred = torch.einsum('ce,ct->et', agg_weights, h_past)
    return h_past, pred, x_future, adj_causal, agg_weights


def offdiag_values(matrix: torch.Tensor) -> torch.Tensor:
    """Return off-diagonal values from a square matrix."""
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix.masked_select(mask)


def compute_pairwise_node_cosine_stats(node_time: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Summarize off-diagonal node-to-node cosine similarities for [N, T] features."""
    if node_time.dim() != 2:
        raise ValueError(f"Expected [N, T] tensor, got shape {tuple(node_time.shape)}")

    if node_time.shape[0] <= 1:
        return {
            f"{prefix}_mean_cos": 0.0,
            f"{prefix}_p90_cos": 0.0,
            f"{prefix}_gt_0p7_frac": 0.0,
        }

    normalized = F.normalize(node_time, p=2, dim=-1, eps=1e-8)
    sim = normalized @ normalized.transpose(0, 1)
    vals = offdiag_values(sim)
    return {
        f"{prefix}_mean_cos": float(vals.mean().item()),
        f"{prefix}_p90_cos": float(torch.quantile(vals, 0.90).item()),
        f"{prefix}_gt_0p7_frac": float((vals > 0.7).float().mean().item()),
    }


@torch.no_grad()
def compute_cross_prediction_diagnostics(model: DDM, x: torch.Tensor) -> Dict[str, float]:
    """Summarize shared-signal and prediction-target alignment diagnostics for cross-prediction."""
    h_past, pred, x_future, _, agg_weights = get_cross_prediction_tensors(model, x)
    h_past_z = zscore_per_node_time(h_past)
    pred_z = zscore_per_node_time(pred)
    target_z = zscore_per_node_time(x_future)

    pred_norm = F.normalize(pred_z, p=2, dim=-1, eps=1e-8)
    target_norm = F.normalize(target_z, p=2, dim=-1, eps=1e-8)
    pred_target_sim = pred_norm @ target_norm.transpose(0, 1)
    diag_vals = torch.diagonal(pred_target_sim)
    offdiag_sim = offdiag_values(pred_target_sim)

    diag_mean = float(diag_vals.mean().item())
    offdiag_mean = float(offdiag_sim.mean().item()) if offdiag_sim.numel() > 0 else 0.0

    stats = {}
    stats.update(compute_pairwise_node_cosine_stats(h_past_z, "cross_source"))
    stats.update(compute_pairwise_node_cosine_stats(pred_z, "cross_pred"))
    stats.update(compute_pairwise_node_cosine_stats(target_z, "cross_target"))
    per_target_max = agg_weights.max(dim=0).values
    effective_parents = 1.0 / agg_weights.pow(2).sum(dim=0).clamp_min(1e-8)
    stats.update({
        "cross_pred_target_diag_mean": diag_mean,
        "cross_pred_target_offdiag_mean": offdiag_mean,
        "cross_pred_target_diag_gap": diag_mean - offdiag_mean,
        "cross_agg_max_mean": float(per_target_max.mean().item()),
        "cross_agg_max_p90": float(torch.quantile(per_target_max, 0.90).item()),
        "cross_agg_eff_parents_mean": float(effective_parents.mean().item()),
    })
    return stats


@torch.no_grad()
def normalize_message_graph_weights(message_adj: torch.Tensor) -> torch.Tensor:
    """Column-normalize message weights so each target aggregates a convex combination of sources."""
    if message_adj.dim() != 2 or message_adj.shape[0] != message_adj.shape[1]:
        raise ValueError(f"Expected square [N, N] message adjacency, got shape {tuple(message_adj.shape)}")
    diag_mask = torch.eye(message_adj.shape[0], dtype=torch.bool, device=message_adj.device)
    weights = message_adj.masked_fill(diag_mask, 0.0)
    incoming = weights.sum(dim=0, keepdim=True)
    weights = weights / incoming.clamp_min(1e-8)
    zero_incoming = (incoming <= 1e-8).squeeze(0)
    if zero_incoming.any():
        weights[:, zero_incoming] = 0.0
    return weights


@torch.no_grad()
def compute_message_graph_direction_diagnostics(model: DDM, x: torch.Tensor) -> Dict[str, float]:
    """
    Compare raw-vs-causal message orientation using a lightweight future-prediction proxy.

    We freeze the current structure, build message aggregation weights under:
    - raw internal convention `A_raw[effect, cause]`
    - transposed causal convention `A_causal[cause, effect]`

    Then we ask which orientation better predicts each node's next-step clean target
    from the previous-step clean targets of all nodes.
    """
    default_stats = {
        "msg_dir_mode_is_causal": 0.0,
        "msg_dir_raw_diag_mean": 0.0,
        "msg_dir_raw_offdiag_mean": 0.0,
        "msg_dir_raw_diag_gap": 0.0,
        "msg_dir_causal_diag_mean": 0.0,
        "msg_dir_causal_offdiag_mean": 0.0,
        "msg_dir_causal_diag_gap": 0.0,
        "msg_dir_active_diag_gap": 0.0,
        "msg_dir_gap_delta_causal_minus_raw": 0.0,
        "msg_dir_prefers_causal": 0.0,
    }
    if not getattr(model, "structure_learning_mode", False) or x.shape[-1] <= 1:
        return default_stats

    clean = model.prepare_clean_target(x)
    clean_past = clean[:, :-1]
    clean_future = clean[:, 1:]
    target_z = zscore_per_node_time(clean_future)
    target_norm = F.normalize(target_z, p=2, dim=-1, eps=1e-8)

    raw_adj = model.get_structure_adj()
    raw_message = normalize_message_graph_weights(raw_adj)
    causal_message = normalize_message_graph_weights(to_causal_matrix_torch(raw_adj))

    stats = {
        "msg_dir_mode_is_causal": float(getattr(model, "structure_message_graph_mode", "raw") == "causal"),
    }
    for mode_name, weights in (("raw", raw_message), ("causal", causal_message)):
        pred = torch.einsum("se,st->et", weights, clean_past)
        pred_z = zscore_per_node_time(pred)
        pred_norm = F.normalize(pred_z, p=2, dim=-1, eps=1e-8)
        sim = pred_norm @ target_norm.transpose(0, 1)
        diag_vals = torch.diagonal(sim)
        offdiag_sim = offdiag_values(sim)
        diag_mean = float(diag_vals.mean().item())
        offdiag_mean = float(offdiag_sim.mean().item()) if offdiag_sim.numel() > 0 else 0.0
        stats[f"msg_dir_{mode_name}_diag_mean"] = diag_mean
        stats[f"msg_dir_{mode_name}_offdiag_mean"] = offdiag_mean
        stats[f"msg_dir_{mode_name}_diag_gap"] = diag_mean - offdiag_mean

    active_mode = getattr(model, "structure_message_graph_mode", "raw")
    stats["msg_dir_active_diag_gap"] = float(stats[f"msg_dir_{active_mode}_diag_gap"])
    stats["msg_dir_gap_delta_causal_minus_raw"] = (
        float(stats["msg_dir_causal_diag_gap"]) - float(stats["msg_dir_raw_diag_gap"])
    )
    stats["msg_dir_prefers_causal"] = float(stats["msg_dir_gap_delta_causal_minus_raw"] > 0.0)
    return stats


@torch.no_grad()
def compute_adjacency_uniformity_diagnostics(adj_causal: torch.Tensor) -> Dict[str, float]:
    """Summarize how uniform and dense the current causal adjacency is."""
    vals = offdiag_values(adj_causal)
    if vals.numel() == 0:
        return {
            "adj_offdiag_mean": 0.0,
            "adj_offdiag_std": 0.0,
            "adj_offdiag_cv": 0.0,
            "adj_offdiag_min": 0.0,
            "adj_offdiag_max": 0.0,
            "adj_in_degree_mean": 0.0,
            "adj_in_degree_std": 0.0,
        }

    in_degree = adj_causal.sum(dim=0)
    offdiag_mean = float(vals.mean().item())
    offdiag_std = float(vals.std(unbiased=False).item())
    return {
        "adj_offdiag_mean": offdiag_mean,
        "adj_offdiag_std": offdiag_std,
        "adj_offdiag_cv": offdiag_std / (offdiag_mean + 1e-8),
        "adj_offdiag_min": float(vals.min().item()),
        "adj_offdiag_max": float(vals.max().item()),
        "adj_in_degree_mean": float(in_degree.mean().item()),
        "adj_in_degree_std": float(in_degree.std(unbiased=False).item()),
    }


def compute_cross_prediction_loss(model: DDM, x: torch.Tensor) -> torch.Tensor:
    """
    Online directionality loss from cross prediction.

    We use explicit causal adjacency semantics:
    adj_causal[cause, effect] predicts how source-node history contributes to
    the target node's future. For each target node, aggregate over causes.
    """
    _, pred, x_future, _, _ = get_cross_prediction_tensors(model, x)
    pred_z = zscore_per_node_time(pred)
    target_z = zscore_per_node_time(x_future)
    return F.smooth_l1_loss(pred_z, target_z)


@torch.no_grad()
def compute_epoch_quality(
    adj_np,
    patel_direction_cpu,
    patel_strength_cpu,
    top_k=61,
    agreement_weight: float = 0.25,
):
    """
    不依赖 GT 的 epoch 质量评分，用于 best-epoch 选择。

    评分使用稳定加权和，而不是脆弱的纯乘法。

    - agreement_score: Patel 高置信方向边上的一致率，但在覆盖不足时回缩到中性分 0.5，
      避免短训或语义不稳时整轮清零
    - dir_margin: top-k 边的平均 |adj[i,j] - adj[j,i]|（衡量方向强度，
      不受饱和高分假边误导）
    - density_factor: 惩罚实际密度偏离目标密度过远的情况（抑制过稀疏/过饱和）
    - skeleton_overlap: 骨架与 Patel 强度先验的重叠度

    Args:
        adj_np: causal 邻接矩阵 [N, N] numpy array (sigmoid 后)
        patel_direction_cpu: Patel tau 方向先验 [N, N] numpy array，按 causal convention 解读
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
        return 0.0, {
            "agreement": 0.0,
            "agreement_score": 0.5,
            "agreement_coverage": 0.0,
            "dir_margin": 0.0,
            "margin_score": 0.0,
            "density_factor": 0.0,
            "skeleton_overlap": 0.0,
            "global_asymmetry": 0.0,
            "asymmetry_score": 0.0,
            "high_conf_edges": 0,
            "k": 0,
        }

    top_edges = candidates[:k]
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
    agreement_coverage = high_conf_count / k if k > 0 else 0.0
    agreement_score = 0.5 + (agreement - 0.5) * agreement_coverage

    # --- 2) dir_margin: 平均方向强度 |adj[i,j] - adj[j,i]| ---
    dir_margin = float(np.mean([e[3] for e in top_edges]))
    margin_score = float(np.tanh(dir_margin / 0.25))

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

    global_asymmetry = float(np.mean(np.abs(adj_np - adj_np.T)[off_diag_mask]))
    asymmetry_score = float(np.tanh(global_asymmetry / 0.15))

    score = (
        0.35 * skeleton_overlap +
        agreement_weight * agreement_score +
        0.20 * density_factor +
        0.15 * margin_score +
        0.05 * asymmetry_score
    )
    return score, {
        "agreement": agreement,
        "agreement_score": agreement_score,
        "agreement_weight": agreement_weight,
        "agreement_coverage": agreement_coverage,
        "dir_margin": dir_margin,
        "margin_score": margin_score,
        "density_factor": density_factor,
        "skeleton_overlap": skeleton_overlap,
        "actual_pair_density": actual_density,
        "target_pair_density": target_density,
        "global_asymmetry": global_asymmetry,
        "asymmetry_score": asymmetry_score,
        "high_conf_edges": high_conf_count,
        "k": k,
    }


def evaluate_selection_guardrails(
    epoch_details: Dict[str, float],
    peak_skeleton_overlap: float,
    min_skeleton_overlap: float = 0.50,
    min_skeleton_retention: float = 0.85,
    min_density_factor: float = 0.65,
    max_density_ratio: float = 2.50,
) -> Dict[str, Any]:
    """Conservative best-epoch eligibility checks on top of the proxy score."""
    target_density = max(float(epoch_details.get("target_pair_density", 0.0)), 1e-8)
    actual_density = max(float(epoch_details.get("actual_pair_density", 0.0)), 0.0)
    density_ratio = max(actual_density / target_density, 1e-8)

    required_skeleton_overlap = max(
        float(min_skeleton_overlap),
        float(peak_skeleton_overlap) * float(min_skeleton_retention),
    )

    reasons = []
    if float(epoch_details.get("skeleton_overlap", 0.0)) < required_skeleton_overlap:
        reasons.append("low_skeleton")
    if float(epoch_details.get("density_factor", 0.0)) < float(min_density_factor):
        reasons.append("low_density_factor")
    if density_ratio > float(max_density_ratio) or density_ratio < (1.0 / float(max_density_ratio)):
        reasons.append("density_ratio_out_of_range")

    return {
        "guardrail_pass": int(len(reasons) == 0),
        "guardrail_reason": "pass" if not reasons else "|".join(reasons),
        "guardrail_density_ratio": density_ratio,
        "guardrail_required_skeleton_overlap": required_skeleton_overlap,
        "guardrail_peak_skeleton_overlap": float(peak_skeleton_overlap),
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
    optimizer_step_mode: str = "subject",

    debug_checks: bool = False,

    ddm_kwargs: Optional[Dict[str, Any]] = None,

    # Pretrain parameters
    skip_pretrain: bool = False,
    pretrain_checkpoint: Optional[str] = None,
    pretrain_epochs: int = 50,
    pretrain_lr: float = 1e-3,
    result_dir: Optional[str] = None,
    target_edge_count: int = 61,
    selection_top_k: Optional[int] = None,
    selection_start_epoch: int = 6,
    selection_min_skeleton_overlap: float = 0.50,
    selection_min_skeleton_retention: float = 0.85,
    selection_min_density_factor: float = 0.65,
    selection_max_density_ratio: float = 2.50,
    enable_directional_loss: bool = True,
    directional_prior_mode: str = "patel",
    directional_schedule: str = "cosine_anneal",
    lag_direction_source: str = "raw",
    directional_kappa_gate: bool = False,
    directional_kappa_gate_quantile: float = 0.5,
    directional_target_ratio: float = 0.01,
    main_loss_weight: float = 1.0,
    directional_loss_end_epoch: int = -1,
    direction_lr_multiplier: float = 1.0,
    freeze_direction_after_epoch: int = -1,
    enable_gradient_alignment_probe: bool = False,
    gradient_alignment_probe_seed: int = 0,
    enable_cross_prediction: bool = False,
    cross_pred_target_ratio: float = 0.02,
    cross_pred_schedule: str = "cosine_anneal",
    cross_pred_aggregation: str = "mean",
    cross_pred_softmax_temp: float = 1.0,
    parent_entropy_lambda: float = 0.0,
    parent_entropy_warmup_epochs: int = 0,
    parent_entropy_ramp_epochs: int = 1,
    parent_cap_lambda: float = 0.0,
    parent_cap_target: float = 0.0,
    parent_cap_warmup_epochs: int = 0,
    parent_cap_ramp_epochs: int = 1,
    ungated_symmetry_lambda: float = 0.0,
    ungated_symmetry_warmup_epochs: int = 0,
    ungated_symmetry_ramp_epochs: int = 1,
    selection_agreement_weight: float = 0.25,

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
    if optimizer_step_mode not in {"subject", "batch_mean"}:
        raise ValueError(
            f"optimizer_step_mode must be 'subject' or 'batch_mean', got {optimizer_step_mode}"
        )
    if main_loss_weight < 0.0:
        raise ValueError(f"main_loss_weight must be >= 0, got {main_loss_weight}")
    if direction_lr_multiplier <= 0.0:
        raise ValueError(
            f"direction_lr_multiplier must be > 0, got {direction_lr_multiplier}"
        )
    if freeze_direction_after_epoch < -1:
        raise ValueError(
            "freeze_direction_after_epoch must be >= -1, "
            f"got {freeze_direction_after_epoch}"
        )
    if directional_loss_end_epoch < -1:
        raise ValueError(
            "directional_loss_end_epoch must be >= -1, "
            f"got {directional_loss_end_epoch}"
        )

    data_3d = data_3d.to(device)
    probe_x = data_3d[0]

    patel_score_matrix = patel_matrix.to(device)
    if patel_direction_matrix is None:
        patel_direction_matrix = patel_score_matrix
    else:
        patel_direction_matrix = patel_direction_matrix.to(device)
    if patel_strength_matrix is None:
        patel_strength_matrix = torch.clamp(0.5 * (patel_score_matrix + patel_score_matrix.t()), min=0.0)
    else:
        patel_strength_matrix = patel_strength_matrix.to(device)
    if directional_kappa_gate or ungated_symmetry_lambda > 0.0:
        directional_pair_gate_matrix, directional_kappa_threshold, directional_kappa_gate_pair_frac = (
            build_kappa_gate_matrix(
                patel_strength_matrix,
                quantile=directional_kappa_gate_quantile,
            )
        )
    else:
        directional_pair_gate_matrix = None
        directional_kappa_threshold = 0.0
        directional_kappa_gate_pair_frac = 0.0

    ddm_kwargs = {} if ddm_kwargs is None else dict(ddm_kwargs)
    fixed_support_mask = ddm_kwargs.get('fixed_support_mask', None)
    if fixed_support_mask is not None:
        fixed_support_mask = fixed_support_mask.to(device)
        ddm_kwargs['fixed_support_mask'] = fixed_support_mask
        if directional_pair_gate_matrix is not None:
            directional_pair_gate_matrix = directional_pair_gate_matrix & fixed_support_mask.bool()

    # Extract use_temporal_encoder from ddm_kwargs to avoid duplicate argument
    use_temporal_encoder = ddm_kwargs.pop('use_temporal_encoder', True)



    # Initialize DDM with Patel score matrix for structure learning.
    # The score matrix carries asymmetric strength for SVD init, while tau is
    # reserved for the weak directional guidance loss.
    # in_dim = TIME_POINTS (features per node)

    # Compute sparsity bias: logit(target_density) so initial sigmoid mean ≈ target_density
    target_density = compute_target_density(num_nodes, target_edge_count)
    adj_bias_init = math.log(target_density / (1.0 - target_density))
    kappa_logit_bias_prior = torch.maximum(patel_strength_matrix, patel_strength_matrix.t()).clone()
    kappa_logit_bias_prior.fill_diagonal_(0.0)

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
        kappa_logit_bias_prior=kappa_logit_bias_prior,

        adj_bias_init=adj_bias_init,

        use_temporal_encoder=use_temporal_encoder,

        **ddm_kwargs,

    )

    model = model.to(device)
    requested_emb_dim = ddm_kwargs.get('emb_dim', None)
    print(
        "Structure parameterization: "
        f"emb_dim={getattr(model, 'emb_dim', num_nodes)} "
        f"(requested={'full' if requested_emb_dim is None else requested_emb_dim}) | "
        f"message_graph_mode={getattr(model, 'structure_message_graph_mode', 'raw')} | "
        f"kappa_logit_bias_scale={getattr(model, 'kappa_logit_bias_scale', 0.0):g}"
    )
    model.cross_pred_aggregation = cross_pred_aggregation
    model.cross_pred_softmax_temp = cross_pred_softmax_temp
    if directional_kappa_gate:
        print(f"Directional kappa gate: enabled | quantile={directional_kappa_gate_quantile:.2f} | "
              f"threshold={directional_kappa_threshold:.4f} | "
              f"pair_frac={directional_kappa_gate_pair_frac:.2%}")

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
    print(f"Main DDM loss weight: {main_loss_weight:g}")

    optimizer, optimizer_stats = build_training_optimizer(
        model,
        learning_rate=learning_rate,
        direction_lr_multiplier=direction_lr_multiplier,
    )
    direction_branch_has_separate_params = bool(optimizer_stats["has_direction_group"])
    direction_branch_frozen = False
    direction_branch_frozen_from_epoch = -1
    directional_loss_deactivated_from_epoch = -1
    if direction_branch_has_separate_params:
        print(
            "Optimizer groups: "
            f"base_lr={optimizer_stats['base_lr']:.4e} "
            f"({optimizer_stats['base_param_count']:,} params) | "
            f"direction_lr={optimizer_stats['direction_lr']:.4e} "
            f"(x{direction_lr_multiplier:g}, {optimizer_stats['direction_param_count']:,} params) | "
            f"freeze_after_epoch={freeze_direction_after_epoch}"
        )
    else:
        print(
            f"Optimizer groups: single_lr={optimizer_stats['base_lr']:.4e} "
            f"({optimizer_stats['base_param_count']:,} params)"
        )

    

    # Track loss history for plotting

    loss_history = []

    collapse_history = []
    quality_history = []

    # Best-epoch tracking (Patel-based proxy, no GT needed)
    best_guarded_adj = None
    best_guarded_adj_causal = None
    best_guarded_score = -1.0
    best_guarded_epoch = -1
    best_guarded_quality_details = None
    fallback_best_adj = None
    fallback_best_adj_causal = None
    fallback_best_score = -1.0
    fallback_best_epoch = -1
    fallback_best_quality_details = None
    patel_direction_cpu = patel_direction_matrix.detach().cpu().numpy()
    patel_strength_cpu = patel_strength_matrix.detach().cpu().numpy()
    selection_top_k = target_edge_count if selection_top_k is None else int(selection_top_k)
    quality_top_k = min(max(selection_top_k, 1), max(num_nodes * (num_nodes - 1) // 2, 1))
    peak_eligible_skeleton_overlap = 0.0

    # Lambda smoothing state (EMA + step-change cap)
    prev_lambda_dir = 0.0
    prev_lambda_ortho = 0.0
    prev_lambda_cross = 0.0

    

    # Training loop

    for epoch in range(num_epochs):
        directional_loss_active = enable_directional_loss and (
            directional_loss_end_epoch < 0 or (epoch + 1) <= directional_loss_end_epoch
        )
        if (
            enable_directional_loss and
            directional_loss_end_epoch >= 0 and
            not directional_loss_active and
            directional_loss_deactivated_from_epoch < 0
        ):
            directional_loss_deactivated_from_epoch = epoch + 1
            print(
                "[DirectionalLoss] Disabling directional supervision "
                f"from epoch {epoch + 1} onward "
                f"(after epoch {directional_loss_end_epoch})"
            )
        if (
            direction_branch_has_separate_params and
            (not direction_branch_frozen) and
            freeze_direction_after_epoch >= 0 and
            (epoch + 1) > freeze_direction_after_epoch
        ):
            frozen_param_count = freeze_direction_branch(model)
            direction_branch_frozen = True
            direction_branch_frozen_from_epoch = epoch + 1
            print(
                "[DirectionRetention] Freezing direction branch "
                f"from epoch {epoch + 1} onward "
                f"(after epoch {freeze_direction_after_epoch}, "
                f"params={frozen_param_count:,})"
            )

        model.train()

        epoch_loss = 0.0

        epoch_sparsity = 0.0

        epoch_dir_loss = 0.0

        epoch_ortho_loss = 0.0

        epoch_cross_loss = 0.0

        epoch_parent_entropy_loss = 0.0

        epoch_parent_cap_loss = 0.0

        epoch_ungated_symmetry_loss = 0.0

        current_parent_entropy_weight = 0.0

        current_parent_cap_weight = 0.0

        current_ungated_symmetry_weight = 0.0

        num_batches = 0

        

        # Shuffle subjects

        perm = torch.randperm(num_subjects)

        

        for i in range(0, num_subjects, batch_size):

            batch_idx = perm[i:i+batch_size]

            # batch_data: [batch_size, N, TIME_POINTS]

            batch_data = data_3d[batch_idx]
            batch_subject_count = int(batch_data.shape[0])
            if batch_subject_count == 0:
                continue
            if optimizer_step_mode == "batch_mean":
                optimizer.zero_grad()

            

            # Process each subject in batch

            for subj_idx in range(batch_data.shape[0]):

                if optimizer_step_mode == "subject":
                    optimizer.zero_grad()

                

                # Get subject data: [N, TIME_POINTS]

                x = batch_data[subj_idx]  # [N, TIME_POINTS]

                if debug_checks and epoch == 0 and i == 0 and subj_idx == 0:

                    with torch.no_grad():

                        x_encoded = model.prepare_clean_target(x)

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
                adj_weights_causal = to_causal_matrix_torch(adj_weights)

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
                causal_logits = get_current_directional_logits(model, causal=True)
                if directional_loss_active:
                    direction_prior_matrix = compute_online_direction_prior_matrix(
                        model=model,
                        x=x,
                        mode=directional_prior_mode,
                        patel_direction_matrix=patel_direction_matrix,
                        lag_direction_source=lag_direction_source,
                    )
                    raw_loss_dir = compute_directional_margin_loss(
                        causal_logits,
                        direction_prior_matrix,
                        pair_gate_matrix=directional_pair_gate_matrix,
                    )
                else:
                    raw_loss_dir = torch.tensor(0.0, device=device)
                if enable_cross_prediction:
                    raw_loss_cross = compute_cross_prediction_loss(model, x)
                else:
                    raw_loss_cross = torch.tensor(0.0, device=device)
                if parent_entropy_lambda > 0.0:
                    raw_loss_parent_entropy = compute_incoming_entropy_loss(adj_weights_causal)
                else:
                    raw_loss_parent_entropy = torch.tensor(0.0, device=device)
                if parent_cap_lambda > 0.0 and parent_cap_target > 0.0:
                    raw_loss_parent_cap = compute_excess_effective_parents_loss(
                        adj_weights_causal,
                        target_effective_parents=parent_cap_target,
                    )
                else:
                    raw_loss_parent_cap = torch.tensor(0.0, device=device)
                if ungated_symmetry_lambda > 0.0:
                    raw_loss_ungated_symmetry = compute_ungated_symmetry_loss(
                        adj_weights_causal,
                        pair_gate_matrix=directional_pair_gate_matrix,
                    )
                else:
                    raw_loss_ungated_symmetry = torch.tensor(0.0, device=device)
                raw_loss_ortho = compute_feature_ortho_loss(
                    model.node_emb_sender, model.node_emb_receiver,
                )

                lambda_dir, lambda_ortho, lambda_cross = compute_auxiliary_lambdas(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    loss_ddm_main=loss_ddm_main,
                    raw_loss_dir=raw_loss_dir,
                    raw_loss_ortho=raw_loss_ortho,
                    raw_loss_cross=raw_loss_cross,
                    prev_lambda_dir=prev_lambda_dir,
                    prev_lambda_ortho=prev_lambda_ortho,
                    prev_lambda_cross=prev_lambda_cross,
                    dir_target_ratio=directional_target_ratio,
                    dir_schedule=directional_schedule,
                    cross_target_ratio=cross_pred_target_ratio,
                    cross_schedule=cross_pred_schedule,
                )

                if directional_loss_active:
                    prev_lambda_dir = lambda_dir
                else:
                    lambda_dir = 0.0
                    prev_lambda_dir = 0.0
                prev_lambda_ortho = lambda_ortho
                if enable_cross_prediction:
                    prev_lambda_cross = lambda_cross
                else:
                    lambda_cross = 0.0
                    prev_lambda_cross = 0.0

                weighted_dir = lambda_dir * raw_loss_dir
                weighted_ortho = lambda_ortho * raw_loss_ortho
                weighted_cross = lambda_cross * raw_loss_cross
                current_parent_entropy_weight = compute_fixed_aux_weight(
                    epoch=epoch,
                    target_weight=parent_entropy_lambda,
                    warmup_epochs=parent_entropy_warmup_epochs,
                    ramp_epochs=parent_entropy_ramp_epochs,
                )
                weighted_parent_entropy = current_parent_entropy_weight * raw_loss_parent_entropy
                current_parent_cap_weight = compute_fixed_aux_weight(
                    epoch=epoch,
                    target_weight=parent_cap_lambda,
                    warmup_epochs=parent_cap_warmup_epochs,
                    ramp_epochs=parent_cap_ramp_epochs,
                )
                weighted_parent_cap = current_parent_cap_weight * raw_loss_parent_cap
                current_ungated_symmetry_weight = compute_fixed_aux_weight(
                    epoch=epoch,
                    target_weight=ungated_symmetry_lambda,
                    warmup_epochs=ungated_symmetry_warmup_epochs,
                    ramp_epochs=ungated_symmetry_ramp_epochs,
                )
                weighted_ungated_symmetry = (
                    current_ungated_symmetry_weight * raw_loss_ungated_symmetry
                )

                weighted_ddm_main = main_loss_weight * loss_ddm_main

                total_loss = (
                    weighted_ddm_main +
                    weighted_dir +
                    weighted_ortho +
                    weighted_cross +
                    weighted_parent_entropy +
                    weighted_parent_cap +
                    weighted_ungated_symmetry
                )

                

                # Backward pass

                loss_scale = 1.0 / batch_subject_count if optimizer_step_mode == "batch_mean" else 1.0
                (total_loss * loss_scale).backward()

                if debug_checks and epoch == 0 and i == 0 and subj_idx == 0:

                    if model.use_temporal_encoder:
                        grad = model.temporal_encoder.input_proj.weight.grad
                        if grad is None:
                            print("[Debug] temporal_encoder grad is None")
                        else:
                            print(f"[Debug] temporal_encoder grad norm: {grad.norm().item():.6e}")
                    else:
                        print("[Debug] Temporal encoder disabled - skipping grad check")

                if optimizer_step_mode == "subject":
                    optimizer.step()

                

                epoch_loss += loss.item()

                epoch_sparsity += sparsity_loss.item()

                epoch_dir_loss += weighted_dir.item()

                epoch_ortho_loss += weighted_ortho.item()

                epoch_cross_loss += weighted_cross.item()

                epoch_parent_entropy_loss += weighted_parent_entropy.item()

                epoch_parent_cap_loss += weighted_parent_cap.item()

                epoch_ungated_symmetry_loss += weighted_ungated_symmetry.item()

                num_batches += 1

            if optimizer_step_mode == "batch_mean":
                optimizer.step()

        

        avg_loss = epoch_loss / num_batches
        avg_sparsity = epoch_sparsity / num_batches
        avg_dir_loss = epoch_dir_loss / num_batches
        avg_ortho_loss = epoch_ortho_loss / num_batches
        avg_cross_loss = epoch_cross_loss / num_batches
        avg_parent_entropy_loss = epoch_parent_entropy_loss / num_batches
        avg_parent_cap_loss = epoch_parent_cap_loss / num_batches
        avg_ungated_symmetry_loss = epoch_ungated_symmetry_loss / num_batches

        grad_probe_lambda_dir = float(prev_lambda_dir) if directional_loss_active else 0.0
        if enable_gradient_alignment_probe:
            grad_probe_stats = compute_direction_grad_alignment_diagnostics(
                model=model,
                x=probe_x,
                num_nodes=num_nodes,
                lambda_l1=lambda_l1,
                directional_prior_mode=directional_prior_mode,
                lag_direction_source=lag_direction_source,
                patel_direction_matrix=patel_direction_matrix,
                directional_pair_gate_matrix=directional_pair_gate_matrix,
                lambda_dir_effective=grad_probe_lambda_dir,
                seed=gradient_alignment_probe_seed,
            )
        else:
            grad_probe_stats = {
                "grad_probe_available": 0.0,
                "grad_probe_lambda_dir": grad_probe_lambda_dir,
                "grad_probe_diff_norm": 0.0,
                "grad_probe_dir_norm_raw": 0.0,
                "grad_probe_dir_norm_weighted": 0.0,
                "grad_probe_dir_to_diff_norm_ratio": 0.0,
                "grad_probe_cosine": 0.0,
                "grad_probe_cosine_negative": 0.0,
            }

        with torch.no_grad():
            causal_logits = get_current_directional_logits(model, causal=True)
            if directional_loss_active:
                probe_direction_prior = compute_online_direction_prior_matrix(
                    model=model,
                    x=probe_x,
                    mode=directional_prior_mode,
                    patel_direction_matrix=patel_direction_matrix,
                    lag_direction_source=lag_direction_source,
                )
                raw_dir_snap = compute_directional_margin_loss(
                    causal_logits,
                    probe_direction_prior,
                    pair_gate_matrix=directional_pair_gate_matrix,
                ).item()
            else:
                raw_dir_snap = 0.0
            if enable_cross_prediction:
                raw_cross_snap = compute_cross_prediction_loss(model, probe_x).item()
                cross_diag_stats = compute_cross_prediction_diagnostics(model, probe_x)
            else:
                raw_cross_snap = 0.0
                cross_diag_stats = {}
            noise_guide_probe_stats = compute_noise_guide_probe_diagnostics(model, probe_x)
            message_dir_stats = compute_message_graph_direction_diagnostics(model, probe_x)
            raw_ortho_snap = compute_feature_ortho_loss(
                model.node_emb_sender, model.node_emb_receiver,
            ).item()
            adj_sigmoid_raw = get_current_structure_adj(model, causal=False)
            adj_sigmoid_causal = get_current_structure_adj(model, causal=True)
            raw_parent_entropy_snap = compute_incoming_entropy_loss(adj_sigmoid_causal).item()
            raw_parent_cap_snap = compute_excess_effective_parents_loss(
                adj_sigmoid_causal,
                target_effective_parents=parent_cap_target,
            ).item()
            adj_diag_stats = compute_adjacency_uniformity_diagnostics(adj_sigmoid_causal)
            parent_profile_stats = compute_incoming_parent_diagnostics(adj_sigmoid_causal)
            raw_ungated_symmetry_snap = compute_ungated_symmetry_loss(
                adj_sigmoid_causal,
                pair_gate_matrix=directional_pair_gate_matrix,
            ).item()
            ungated_asym_stats = compute_ungated_asymmetry_diagnostics(
                adj_sigmoid_causal,
                pair_gate_matrix=directional_pair_gate_matrix,
            )
            adj_mean = adj_sigmoid_raw.mean().item()
            sparsity_ratio = (adj_sigmoid_raw < 0.5).float().mean().item()

        curr_adj_raw = adj_sigmoid_raw.cpu().numpy()
        curr_adj_causal = adj_sigmoid_causal.cpu().numpy()
        epoch_score, epoch_details = compute_epoch_quality(
            curr_adj_causal,
            patel_direction_cpu,
            patel_strength_cpu,
            top_k=quality_top_k,
            agreement_weight=selection_agreement_weight,
        )
        selection_eligible = int((epoch + 1) >= selection_start_epoch)
        if selection_eligible:
            guardrail_details = evaluate_selection_guardrails(
                epoch_details,
                peak_skeleton_overlap=peak_eligible_skeleton_overlap,
                min_skeleton_overlap=selection_min_skeleton_overlap,
                min_skeleton_retention=selection_min_skeleton_retention,
                min_density_factor=selection_min_density_factor,
                max_density_ratio=selection_max_density_ratio,
            )
            peak_eligible_skeleton_overlap = max(
                peak_eligible_skeleton_overlap,
                float(epoch_details["skeleton_overlap"]),
            )
        else:
            guardrail_details = {
                "guardrail_pass": 0,
                "guardrail_reason": "before_selection_start",
                "guardrail_density_ratio": (
                    float(epoch_details["actual_pair_density"]) /
                    max(float(epoch_details["target_pair_density"]), 1e-8)
                ),
                "guardrail_required_skeleton_overlap": float(selection_min_skeleton_overlap),
                "guardrail_peak_skeleton_overlap": peak_eligible_skeleton_overlap,
            }
        quality_history.append({
            "epoch": epoch + 1,
            "score": epoch_score,
            "selection_eligible": selection_eligible,
            **epoch_details,
            **guardrail_details,
            **adj_diag_stats,
            **parent_profile_stats,
            **ungated_asym_stats,
            **cross_diag_stats,
            **noise_guide_probe_stats,
            **message_dir_stats,
            "directional_kappa_gate_enabled": int(directional_kappa_gate),
            "directional_kappa_gate_quantile": float(directional_kappa_gate_quantile),
            "directional_kappa_gate_threshold": float(directional_kappa_threshold),
            "directional_kappa_gate_pair_frac": float(directional_kappa_gate_pair_frac),
            "main_loss_weight": float(main_loss_weight),
            "directional_loss_active": int(directional_loss_active),
            "directional_loss_end_epoch": int(directional_loss_end_epoch),
            "directional_loss_deactivated_from_epoch": int(directional_loss_deactivated_from_epoch),
            "dir_loss_raw": float(raw_dir_snap),
            "dir_loss_weighted": float(avg_dir_loss),
            "dir_lambda_current": float(grad_probe_lambda_dir),
            "direction_lr_multiplier": float(direction_lr_multiplier),
            "direction_lr_current": (
                0.0
                if direction_branch_frozen or not direction_branch_has_separate_params
                else float(learning_rate * direction_lr_multiplier)
            ),
            "direction_branch_frozen": int(direction_branch_frozen),
            "freeze_direction_after_epoch": int(freeze_direction_after_epoch),
            "direction_branch_frozen_from_epoch": int(direction_branch_frozen_from_epoch),
            "cross_loss_raw": float(raw_cross_snap),
            "cross_loss_weighted": float(avg_cross_loss),
            "ortho_loss_raw": float(raw_ortho_snap),
            "ortho_loss_weighted": float(avg_ortho_loss),
            "parent_entropy_raw": raw_parent_entropy_snap,
            "parent_entropy_weighted": avg_parent_entropy_loss,
            "parent_entropy_lambda_current": current_parent_entropy_weight,
            "parent_cap_raw": raw_parent_cap_snap,
            "parent_cap_weighted": avg_parent_cap_loss,
            "parent_cap_lambda_current": current_parent_cap_weight,
            "parent_cap_target": float(parent_cap_target),
            "ungated_symmetry_raw": raw_ungated_symmetry_snap,
            "ungated_symmetry_weighted": avg_ungated_symmetry_loss,
            "ungated_symmetry_lambda_current": current_ungated_symmetry_weight,
            "gradient_alignment_probe_enabled": int(enable_gradient_alignment_probe),
            "gradient_alignment_probe_seed": int(gradient_alignment_probe_seed),
            **grad_probe_stats,
        })
        marker_parts = []
        if selection_eligible and epoch_score > fallback_best_score:
            fallback_best_score = epoch_score
            fallback_best_adj = curr_adj_raw.copy()
            fallback_best_adj_causal = curr_adj_causal.copy()
            fallback_best_epoch = epoch + 1
            fallback_best_quality_details = dict(epoch_details)
            marker_parts.append("score-best")
        if (
            selection_eligible and
            guardrail_details["guardrail_pass"] and
            epoch_score > best_guarded_score
        ):
            best_guarded_score = epoch_score
            best_guarded_adj = curr_adj_raw.copy()
            best_guarded_adj_causal = curr_adj_causal.copy()
            best_guarded_epoch = epoch + 1
            best_guarded_quality_details = dict(epoch_details)
            marker_parts.append("guarded-best")
        marker = "" if not marker_parts else f" ★ {'/'.join(marker_parts)}"

        # Log progress
        if (epoch + 1) % log_interval == 0 or epoch == num_epochs - 1:
            current_best_epoch = best_guarded_epoch if best_guarded_epoch >= 0 else fallback_best_epoch
            current_best_score = best_guarded_score if best_guarded_epoch >= 0 else fallback_best_score
            current_best_mode = "guarded" if best_guarded_epoch >= 0 else "score-only"
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Diff Loss: {avg_loss:.4f} | "
                  f"Sparsity Loss: {avg_sparsity:.4f} | "
                  f"DirSup(active): {int(directional_loss_active)} | "
                  f"DirBranch(lr/frozen): "
                  f"{0.0 if direction_branch_frozen or not direction_branch_has_separate_params else learning_rate * direction_lr_multiplier:.4e}/"
                  f"{int(direction_branch_frozen)} | "
                  f"Dir Loss(raw/w): {raw_dir_snap:.4f}/{avg_dir_loss:.4f} | "
                  f"Cross Loss(raw/w): {raw_cross_snap:.4f}/{avg_cross_loss:.4f} | "
                  f"Parent Ent(raw/w): {raw_parent_entropy_snap:.4f}/{avg_parent_entropy_loss:.4f} "
                  f"(lambda={current_parent_entropy_weight:.4f}) | "
                  f"Parent Cap(raw/w): {raw_parent_cap_snap:.4f}/{avg_parent_cap_loss:.4f} "
                  f"(lambda={current_parent_cap_weight:.4f}, target={parent_cap_target:.2f}) | "
                  f"Ungated Sym(raw/w): {raw_ungated_symmetry_snap:.4f}/{avg_ungated_symmetry_loss:.4f} "
                  f"(lambda={current_ungated_symmetry_weight:.4f}) | "
                  f"Ortho Loss(raw/w): {raw_ortho_snap:.4f}/{avg_ortho_loss:.4f} | "
                  f"Adj Mean: {adj_mean:.3f} | "
                  f"Sparsity: {sparsity_ratio:.2%}")

            if model.use_temporal_encoder:
                collapse_metrics = diagnose_encoder_collapse(model, data_3d, device)
                print_collapse_diagnostics(collapse_metrics, epoch, num_epochs)
                collapse_history.append({"epoch": epoch + 1, **collapse_metrics})

            print(f"  [Quality] score={epoch_score:.4f} "
                  f"(agree={epoch_details['agreement']:.2%}/{epoch_details['agreement_score']:.2%}"
                  f"[{epoch_details['high_conf_edges']}], "
                  f"margin={epoch_details['dir_margin']:.4f}/{epoch_details['margin_score']:.3f}, "
                  f"skel={epoch_details['skeleton_overlap']:.2%}, "
                  f"dens={epoch_details['density_factor']:.3f}, "
                  f"asym={epoch_details['global_asymmetry']:.4f}, "
                  f"pair_dens={epoch_details['actual_pair_density']:.2%}/{epoch_details['target_pair_density']:.2%}) | "
                  f"Guard={guardrail_details['guardrail_reason']} "
                  f"(req_skel={guardrail_details['guardrail_required_skeleton_overlap']:.2%}, "
                  f"ratio={guardrail_details['guardrail_density_ratio']:.2f}) | "
                  f"Best[{current_best_mode}]: epoch {current_best_epoch} score={current_best_score:.4f} | "
                  f"Eligible from epoch {selection_start_epoch}{marker}")
            print(f"  [AdjDiag] offdiag={adj_diag_stats['adj_offdiag_mean']:.4f}±"
                  f"{adj_diag_stats['adj_offdiag_std']:.4f} "
                  f"(cv={adj_diag_stats['adj_offdiag_cv']:.3f}, "
                  f"min/max={adj_diag_stats['adj_offdiag_min']:.4f}/"
                  f"{adj_diag_stats['adj_offdiag_max']:.4f}) | "
                  f"in_deg={adj_diag_stats['adj_in_degree_mean']:.4f}±"
                  f"{adj_diag_stats['adj_in_degree_std']:.4f} | "
                  f"eff_par={parent_profile_stats['adj_eff_parents_mean']:.3f}"
                  f"/p90={parent_profile_stats['adj_eff_parents_p90']:.3f} | "
                  f"parent_ent={parent_profile_stats['adj_parent_entropy_mean']:.3f} | "
                  f"ungated_asym={ungated_asym_stats['adj_ungated_asym_mean']:.3f}"
                  f"/med={ungated_asym_stats['adj_ungated_asym_median']:.3f}"
                  f"/p90={ungated_asym_stats['adj_ungated_asym_p90']:.3f}")
            if enable_cross_prediction:
                print(f"  [CrossDiag] h_cos={cross_diag_stats['cross_source_mean_cos']:.3f}"
                      f"/p90={cross_diag_stats['cross_source_p90_cos']:.3f}"
                      f"/>0.7={cross_diag_stats['cross_source_gt_0p7_frac']:.2%} | "
                      f"pred_cos={cross_diag_stats['cross_pred_mean_cos']:.3f} | "
                      f"target_cos={cross_diag_stats['cross_target_mean_cos']:.3f} | "
                      f"agg_max={cross_diag_stats['cross_agg_max_mean']:.3f}"
                      f"/p90={cross_diag_stats['cross_agg_max_p90']:.3f} | "
                      f"agg_eff_par={cross_diag_stats['cross_agg_eff_parents_mean']:.3f} | "
                      f"pred->target diag/off/gap="
                      f"{cross_diag_stats['cross_pred_target_diag_mean']:.3f}/"
                      f"{cross_diag_stats['cross_pred_target_offdiag_mean']:.3f}/"
                      f"{cross_diag_stats['cross_pred_target_diag_gap']:.3f}")
            if noise_guide_probe_stats.get("noise_probe_available", 0.0) > 0.5:
                print(f"  [NoiseProbe] t={int(noise_guide_probe_stats['noise_probe_timestep'])} | "
                      f"patel={noise_guide_probe_stats['noise_probe_patel_loss']:.4f} | "
                      f"blend50={noise_guide_probe_stats['noise_probe_blend50_loss']:.4f} "
                      f"(d={noise_guide_probe_stats['noise_probe_delta_blend50_minus_patel']:+.4f}, "
                      f"r={noise_guide_probe_stats['noise_probe_ratio_blend50_over_patel']:.3f}) | "
                      f"learned={noise_guide_probe_stats['noise_probe_learned_loss']:.4f} "
                      f"(d={noise_guide_probe_stats['noise_probe_delta_learned_minus_patel']:+.4f}, "
                      f"r={noise_guide_probe_stats['noise_probe_ratio_learned_over_patel']:.3f}) | "
                      f"guide_l1={noise_guide_probe_stats['noise_probe_guide_l1_mean']:.4f}")
            active_msg_mode = "causal" if message_dir_stats["msg_dir_mode_is_causal"] > 0.5 else "raw"
            preferred_msg_mode = "causal" if message_dir_stats["msg_dir_prefers_causal"] > 0.5 else "raw"
            print(f"  [MsgDir] active={active_msg_mode} | "
                  f"raw_gap={message_dir_stats['msg_dir_raw_diag_gap']:.3f} | "
                  f"causal_gap={message_dir_stats['msg_dir_causal_diag_gap']:.3f} | "
                  f"delta(causal-raw)={message_dir_stats['msg_dir_gap_delta_causal_minus_raw']:.3f} | "
                  f"prefers={preferred_msg_mode}")
            if enable_gradient_alignment_probe:
                print(f"  [GradDiag] avail={int(grad_probe_stats['grad_probe_available'])} | "
                      f"lambda_dir={grad_probe_stats['grad_probe_lambda_dir']:.4e} | "
                      f"diff_norm={grad_probe_stats['grad_probe_diff_norm']:.4e} | "
                      f"dir_norm_w={grad_probe_stats['grad_probe_dir_norm_weighted']:.4e} | "
                      f"ratio={grad_probe_stats['grad_probe_dir_to_diff_norm_ratio']:.4e} | "
                      f"cos={grad_probe_stats['grad_probe_cosine']:+.4f}")

        # Record loss for every epoch

        loss_history.append(epoch_loss / num_batches)

    

    # Extract final adjacency matrix
    with torch.no_grad():
        last_adj_raw = get_current_structure_adj(model, causal=False)
        last_adj_causal = to_causal_matrix_torch(last_adj_raw)
        last_adj = last_adj_raw.cpu().numpy()
        last_adj_diag = compute_adjacency_uniformity_diagnostics(last_adj_causal)

    # Prefer guarded best-epoch selection, but fall back to score-only selection if
    # all epochs fail the guardrails.
    if best_guarded_adj is not None:
        adj_matrix = best_guarded_adj
        best_epoch = best_guarded_epoch
        best_score = best_guarded_score
        best_quality_details = best_guarded_quality_details
        best_adj_causal = best_guarded_adj_causal
        best_selection_mode = "guarded"
        print(f"\n[Best-Epoch] Using guarded epoch {best_epoch} (score={best_score:.4f}) "
              f"instead of final epoch {num_epochs}")
    elif fallback_best_adj is not None:
        adj_matrix = fallback_best_adj
        best_epoch = fallback_best_epoch
        best_score = fallback_best_score
        best_quality_details = fallback_best_quality_details
        best_adj_causal = fallback_best_adj_causal
        best_selection_mode = "score_only_fallback"
        print(f"\n[Best-Epoch] No epoch passed guardrails; falling back to score-only epoch "
              f"{best_epoch} (score={best_score:.4f}) instead of final epoch {num_epochs}")
    else:
        adj_matrix = last_adj
        best_score = -1.0
        best_epoch = num_epochs
        best_quality_details = None
        best_adj_causal = to_causal_matrix_np(adj_matrix)
        best_selection_mode = "final_epoch_fallback"

    model.last_epoch_adj_matrix = last_adj
    model.best_epoch_adj_matrix = adj_matrix
    model.last_epoch_adj_matrix_causal = last_adj_causal.cpu().numpy()
    model.best_epoch_adj_matrix_causal = (
        best_adj_causal if best_adj_causal is not None else to_causal_matrix_np(adj_matrix)
    )
    model.last_epoch_adj_diagnostics = last_adj_diag
    model.best_epoch_score = best_score
    model.best_epoch = best_epoch
    model.best_epoch_quality = best_quality_details
    model.best_epoch_selection_mode = best_selection_mode
    model.best_epoch_guarded = best_guarded_epoch
    model.best_epoch_guarded_score = best_guarded_score
    model.best_epoch_score_only = fallback_best_epoch
    model.best_epoch_score_only_score = fallback_best_score
    model.quality_history = quality_history
    model.directional_loss_end_epoch = int(directional_loss_end_epoch)
    model.directional_loss_deactivated_from_epoch = int(directional_loss_deactivated_from_epoch)
    model.main_loss_weight = float(main_loss_weight)
    model.direction_lr_multiplier = float(direction_lr_multiplier)
    model.freeze_direction_after_epoch = int(freeze_direction_after_epoch)
    model.direction_branch_frozen_from_epoch = int(direction_branch_frozen_from_epoch)
    model.gradient_alignment_probe_enabled = int(enable_gradient_alignment_probe)
    model.gradient_alignment_probe_seed = int(gradient_alignment_probe_seed)

    return model, adj_matrix, loss_history, collapse_history, best_epoch





def main():

    parser = argparse.ArgumentParser(description='Brain Connectivity Learning with DDM')

    parser.add_argument('--csv_path', type=str, default='../fMRI_dataset/sim4.csv',

                        help='Path to fMRI.csv file')

    parser.add_argument('--time_points', type=int, default=TIME_POINTS_PER_SUBJECT,

                        help='Number of time points per subject')
    parser.add_argument('--subject_limit', type=int, default=-1,
                        help='Optional cap on the number of subjects to load; -1 keeps all')
    parser.add_argument('--time_limit', type=int, default=-1,
                        help='Optional cap on time points per subject after reshape; -1 keeps all')

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
    parser.add_argument('--optimizer_step_mode', type=str, default='subject',
                        choices=['subject', 'batch_mean'],
                        help='subject: one optimizer step per subject; batch_mean: average gradients across the subject minibatch before stepping')

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

    parser.add_argument('--structure_init_mode', type=str, default='patel_score',
                        choices=[
                            'patel_score',
                            'patel_score_t',
                            'neg_patel_score',
                            'neg_patel_score_t',
                            'patel_kappa',
                            'pearson',
                            'random',
                        ],
                        help='Matrix used only for structure embedding initialization')
    parser.add_argument('--structure_init_scale', type=float, default=1.0,
                        help='Target std of initial structure logits after SVD-based rescaling; 0 disables directional init strength')
    parser.add_argument('--emb_dim', type=int, default=0,
                        help='Low-rank factor dimension for structure parameterization; <=0 uses full rank N')
    parser.add_argument('--structure_parameterization', type=str, default='coupled',
                        choices=['coupled', 'support_direction'],
                        help='coupled = one adjacency parameterization for existence+direction; support_direction = symmetric support with pairwise direction split')
    parser.add_argument('--structure_message_graph_mode', type=str, default='raw',
                        choices=['raw', 'causal'],
                        help='Adjacency convention used for GraphConv message passing: raw keeps internal A[effect,cause], causal uses transpose A[cause,effect]')
    parser.add_argument('--adj_activation', type=str, default='sigmoid',
                        choices=['sigmoid', 'sparsemax', 'entmax15'],
                        help='Adjacency activation: sigmoid = independent edges; sparsemax/entmax15 = competing parents per target')
    parser.add_argument('--kappa_logit_bias_scale', type=float, default=0.0,
                        help='Persistent symmetric Patel-kappa bias added to structure logits during training')
    parser.add_argument('--fixed_support_mask_mode', type=str, default='none',
                        choices=['none', 'topk_kappa', 'maxgap_kappa'],
                        help='Optional fixed undirected support mask injected into support/direction factorization')
    parser.add_argument('--direction_init_mode', type=str, default='patel_tau',
                        choices=['patel_tau', 'zeros', 'random'],
                        help='Initialization for the directional branch in support/direction factorization')

    parser.add_argument('--selection_start_epoch', type=int, default=6,
                        help='First epoch eligible for best-epoch export selection')

    parser.add_argument('--selection_top_k', type=int, default=None,
                        help='Top-k undirected pairs used only for best-epoch proxy selection')

    parser.add_argument('--selection_min_skeleton_overlap', type=float, default=0.50,
                        help='Absolute minimum skeleton overlap required by guarded selection')

    parser.add_argument('--selection_min_skeleton_retention', type=float, default=0.85,
                        help='Keep at least this fraction of the best eligible skeleton overlap so far')

    parser.add_argument('--selection_min_density_factor', type=float, default=0.65,
                        help='Minimum density_factor required by guarded selection')

    parser.add_argument('--selection_max_density_ratio', type=float, default=2.50,
                        help='Maximum allowed actual/target pair-density ratio for guarded selection')

    parser.add_argument('--selection_agreement_weight', type=float, default=0.25,
                        help='Weight of Patel tau agreement term in best-epoch proxy score; set 0 to disable')

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

    # === Fix 4: Uniform timestep ===
    parser.add_argument('--uniform_timestep', action='store_true', default=True,
                        help='All nodes share the same timestep per forward pass (default)')
    parser.add_argument('--per_node_timestep', action='store_true', default=False,
                        help='Each node samples an independent timestep (legacy behavior)')

    # === Fix 2: Noise normalization mode ===
    parser.add_argument('--noise_norm_mode', type=str, default='global',
                        choices=['global', 'layernorm', 'none'],
                        help='Noise normalization: global (preserve node differences), '
                             'layernorm (legacy), none')

    # === Fix 3: Zero-mean noise ===
    parser.add_argument('--noise_zero_mean', action='store_true', default=True,
                        help='Drop neighbor mean from noise (reduce signal correlation, default)')
    parser.add_argument('--noise_with_mean', action='store_true', default=False,
                        help='Include neighbor mean in noise (legacy behavior)')

    # === Fix 1: Directed noise guidance ===
    parser.add_argument('--directed_noise', action='store_true', default=False,
                        help='Use direction-biased noise guide adjacency from Patel tau')
    parser.add_argument('--direction_alpha', type=float, default=0.5,
                        help='Direction bias strength for directed noise (0=symmetric, 1=max)')

    parser.add_argument('--disable_directional_loss', action='store_true', default=False,
                        help='Disable Patel tau directional margin loss during training')
    parser.add_argument('--directional_prior_mode', type=str, default='patel',
                        choices=['patel', 'lag_corr'],
                        help='Directional prior used by the margin loss')
    parser.add_argument('--directional_schedule', type=str, default='cosine_anneal',
                        choices=['cosine_anneal', 'plateau'],
                        help='Directional auxiliary schedule after warmup')
    parser.add_argument('--lag_direction_source', type=str, default='raw',
                        choices=['raw', 'encoder'],
                        help='Signal source for lag_corr directional prior')
    parser.add_argument('--directional_kappa_gate', action='store_true', default=False,
                        help='Gate directional margin loss to high-kappa pairs only')
    parser.add_argument('--directional_kappa_gate_quantile', type=float, default=0.5,
                        help='Quantile over positive Patel kappa used to define the directional gate')
    parser.add_argument('--directional_target_ratio', type=float, default=0.01,
                        help='Target ratio of directional margin loss relative to main loss')
    parser.add_argument('--main_loss_weight', type=float, default=1.0,
                        help='Weight applied to the main DDM loss (diffusion + sparsity + hub)')
    parser.add_argument('--directional_loss_end_epoch', type=int, default=-1,
                        help='Keep directional supervision through this epoch inclusive; -1 keeps it on for the full run')
    parser.add_argument('--direction_lr_multiplier', type=float, default=1.0,
                        help='LR multiplier applied only to the separate direction branch in support_direction mode')
    parser.add_argument('--freeze_direction_after_epoch', type=int, default=-1,
                        help='Freeze the separate direction branch after this many full epochs; -1 disables')
    parser.add_argument('--enable_gradient_alignment_probe', action='store_true', default=False,
                        help='Record end-of-epoch gradient alignment diagnostics on the direction branch')
    parser.add_argument('--gradient_alignment_probe_seed', type=int, default=0,
                        help='RNG seed used by the fixed gradient-alignment probe forward pass')
    parser.add_argument('--enable_cross_prediction', action='store_true', default=False,
                        help='Enable online cross-prediction auxiliary loss for directionality')
    parser.add_argument('--cross_pred_target_ratio', type=float, default=0.02,
                        help='Target ratio of cross-prediction loss relative to main loss')
    parser.add_argument('--cross_pred_schedule', type=str, default='cosine_anneal',
                        choices=['cosine_anneal', 'plateau'],
                        help='Cross-pred auxiliary schedule after warmup: cosine_anneal or plateau')
    parser.add_argument('--cross_pred_aggregation', type=str, default='mean',
                        choices=['mean', 'softmax'],
                        help='Cross-pred aggregation inside the auxiliary loss')
    parser.add_argument('--cross_pred_softmax_temp', type=float, default=1.0,
                        help='Temperature for softmax cross-pred aggregation')
    parser.add_argument('--parent_entropy_lambda', type=float, default=0.0,
                        help='Weight for incoming-parent entropy regularization on the main graph')
    parser.add_argument('--parent_entropy_warmup_epochs', type=int, default=0,
                        help='Warmup epochs before parent-entropy regularization becomes active')
    parser.add_argument('--parent_entropy_ramp_epochs', type=int, default=1,
                        help='Linear ramp epochs for parent-entropy regularization')
    parser.add_argument('--parent_cap_lambda', type=float, default=0.0,
                        help='Weight for hinge-style effective-parent cap regularization')
    parser.add_argument('--parent_cap_target', type=float, default=0.0,
                        help='Target effective parents; only excess above this value is penalized')
    parser.add_argument('--parent_cap_warmup_epochs', type=int, default=0,
                        help='Warmup epochs before parent-cap regularization becomes active')
    parser.add_argument('--parent_cap_ramp_epochs', type=int, default=1,
                        help='Linear ramp epochs for parent-cap regularization')
    parser.add_argument('--ungated_symmetry_lambda', type=float, default=0.0,
                        help='Weight for asymmetry suppression on pairs outside the directional kappa gate')
    parser.add_argument('--ungated_symmetry_warmup_epochs', type=int, default=0,
                        help='Warmup epochs before ungated-pair symmetry regularization activates')
    parser.add_argument('--ungated_symmetry_ramp_epochs', type=int, default=1,
                        help='Linear ramp epochs for ungated-pair symmetry regularization')

    # === Fix 6: Loss function ===
    parser.add_argument('--loss_type', type=str, default='denoise_hybrid',
                        choices=['cosine', 'denoise_hybrid', 'hybrid', 'smooth_l1', 'mse'],
                        help='Denoising loss: denoise_hybrid (SmoothL1 + small cosine, default), '
                             'hybrid (legacy cosine+MSE), smooth_l1, cosine, mse')
    parser.add_argument('--cosine_weight', type=float, default=0.1,
                        help='Cosine weight used by denoise_hybrid (default: 0.1)')
    parser.add_argument('--mse_weight', type=float, default=0.1,
                        help='MSE weight in legacy hybrid loss (default: 0.1)')



    args = parser.parse_args()

    if args.parent_cap_lambda > 0.0 and args.parent_cap_target <= 0.0:
        parser.error('--parent_cap_target must be > 0 when --parent_cap_lambda is enabled')
    if args.ungated_symmetry_lambda > 0.0 and not args.directional_kappa_gate:
        parser.error('--ungated_symmetry_lambda requires --directional_kappa_gate')
    if args.fixed_support_mask_mode != 'none' and args.structure_parameterization != 'support_direction':
        parser.error('--fixed_support_mask_mode requires --structure_parameterization support_direction')
    if args.direction_lr_multiplier <= 0.0:
        parser.error('--direction_lr_multiplier must be > 0')
    if args.freeze_direction_after_epoch < -1:
        parser.error('--freeze_direction_after_epoch must be >= -1')
    if args.directional_loss_end_epoch < -1:
        parser.error('--directional_loss_end_epoch must be >= -1')
    if args.disable_directional_loss and args.directional_loss_end_epoch >= 0:
        parser.error('--directional_loss_end_epoch cannot be used with --disable_directional_loss')
    if args.gradient_alignment_probe_seed < 0:
        parser.error('--gradient_alignment_probe_seed must be >= 0')
    if (
        (abs(args.direction_lr_multiplier - 1.0) > 1e-12 or args.freeze_direction_after_epoch >= 0) and
        args.structure_parameterization != 'support_direction'
    ):
        parser.error(
            '--direction_lr_multiplier and --freeze_direction_after_epoch require '
            '--structure_parameterization support_direction'
        )
    if args.enable_gradient_alignment_probe and args.structure_parameterization != 'support_direction':
        parser.error(
            '--enable_gradient_alignment_probe requires --structure_parameterization support_direction'
        )

    

    # Set device

    if args.device == 'cuda' and not torch.cuda.is_available():

        print("CUDA not available, falling back to CPU")

        args.device = 'cpu'

    

    print("=" * 60)

    print("Brain Connectivity Learning with DDM")

    print("=" * 60)

    print(f"Device: {args.device}")

    print(f"Time points per subject (raw): {args.time_points}")
    print(f"Requested subject_limit/time_limit: {args.subject_limit}/{args.time_limit}")

    print(f"L1 regularization (lambda): {args.lambda_l1}")
    print(f"Optimizer step mode: {args.optimizer_step_mode} (batch_size={args.batch_size})")
    print(f"Structure parameterization: {args.structure_parameterization}")
    print(f"Adjacency activation: {args.adj_activation}")
    print(f"Kappa logit bias scale: {args.kappa_logit_bias_scale}")
    print(f"Main DDM loss weight: {args.main_loss_weight}")

    print("=" * 60)

    

    set_seed(args.seed)

    

    # Load and reshape fMRI data

    data_3d, data_2d, num_subjects, num_nodes = load_fmri_data(

        csv_path=args.csv_path,

        time_points_per_subject=args.time_points,
        subject_limit=args.subject_limit,
        time_limit=args.time_limit,

    )

    effective_time_points = int(data_3d.shape[-1])
    print(f"Effective dataset: subjects={num_subjects}, nodes={num_nodes}, time_points={effective_time_points}")

    

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

    structure_init_matrix = build_structure_init_matrix(
        mode=args.structure_init_mode,
        patel_score_matrix=patel_score_matrix,
        patel_kappa_matrix=patel_kappa_matrix,
        pearson_matrix=pearson_matrix.float(),
        seed=args.seed,
    )
    print(f"Structure init mode: {args.structure_init_mode} | "
          f"scale={args.structure_init_scale} | "
          f"range=[{structure_init_matrix.min():.4f}, {structure_init_matrix.max():.4f}]")
    if args.structure_parameterization == 'support_direction':
        print(f"Support/direction factorization: enabled | direction_init={args.direction_init_mode} | "
              f"fixed_support_mask={args.fixed_support_mask_mode}")
    if args.kappa_logit_bias_scale != 0.0:
        print(
            "Kappa logit bias: enabled | "
            f"scale={args.kappa_logit_bias_scale} | "
            f"sym_range=[{torch.clamp(patel_kappa_matrix, min=0.0).min():.4f}, "
            f"{torch.clamp(patel_kappa_matrix, min=0.0).max():.4f}]"
        )
    if args.enable_cross_prediction:
        print(f"Cross-prediction: enabled | target_ratio={args.cross_pred_target_ratio} | "
              f"schedule={args.cross_pred_schedule} | "
              f"aggregation={args.cross_pred_aggregation} | "
              f"softmax_temp={args.cross_pred_softmax_temp}")
    if args.parent_entropy_lambda > 0.0:
        print(f"Parent concentration: enabled | entropy_lambda={args.parent_entropy_lambda} | "
              f"warmup={args.parent_entropy_warmup_epochs} | "
              f"ramp={args.parent_entropy_ramp_epochs}")
    if args.parent_cap_lambda > 0.0:
        print(f"Parent cap: enabled | lambda={args.parent_cap_lambda} | "
              f"target={args.parent_cap_target} | "
              f"warmup={args.parent_cap_warmup_epochs} | "
              f"ramp={args.parent_cap_ramp_epochs}")
    if args.ungated_symmetry_lambda > 0.0:
        print(f"Ungated-pair symmetry: enabled | lambda={args.ungated_symmetry_lambda} | "
              f"warmup={args.ungated_symmetry_warmup_epochs} | "
              f"ramp={args.ungated_symmetry_ramp_epochs}")
    if not args.disable_directional_loss:
        print(f"Directional prior: enabled | mode={args.directional_prior_mode} | "
              f"schedule={args.directional_schedule} | "
              f"target_ratio={args.directional_target_ratio} | "
              f"end_epoch={args.directional_loss_end_epoch} | "
              f"lag_source={args.lag_direction_source}")
        if args.directional_kappa_gate:
            print(f"Directional kappa gate requested | quantile={args.directional_kappa_gate_quantile}")
    if args.structure_parameterization == 'support_direction':
        print(f"Direction retention: lr_multiplier={args.direction_lr_multiplier:g} | "
              f"freeze_after_epoch={args.freeze_direction_after_epoch}")
    if args.enable_gradient_alignment_probe:
        print(f"Gradient alignment probe: enabled | seed={args.gradient_alignment_probe_seed}")

    # Step 3: Build noise-guide skeleton
    if args.directed_noise:
        noise_guide_adj = build_directed_noise_guide_adjacency(
            patel_kappa=torch.clamp(patel_kappa_matrix, min=0.0),
            patel_tau=patel_tau_matrix,
            top_k_pairs=args.top_k_edges,
            direction_alpha=args.direction_alpha,
        )
        adj_binary = None
        asym_norm = torch.norm(noise_guide_adj - noise_guide_adj.t()).item()
        print(f"Directed noise guide adj: top {args.top_k_edges} pairs, "
              f"alpha={args.direction_alpha}, ||A-A^T||_F={asym_norm:.4f}")
        # For compatibility: estimate k_pairs from non-self-loop entries
        k_pairs = args.top_k_edges
    else:
        noise_selection_mode = 'maxgap' if args.fixed_support_mask_mode == 'maxgap_kappa' else 'topk'
        noise_guide_adj, adj_binary, k_pairs, threshold, selection_gap = build_noise_guide_adjacency(
            patel_strength_matrix=torch.clamp(patel_kappa_matrix, min=0.0),
            top_k_pairs=args.top_k_edges,
            selection_mode=noise_selection_mode,
        )
        if noise_selection_mode == 'maxgap':
            print(
                f"Keeping max-gap kappa skeleton: {k_pairs} undirected pairs "
                f"(threshold: {threshold:.4f}, gap: {selection_gap:.4f})"
            )
        else:
            print(f"Keeping top {k_pairs} undirected pairs (threshold: {threshold:.4f})")
        print(f"Noise guide adj: {adj_binary.sum().item() / 2:.0f} undirected pairs + {num_nodes} self-loops")

    fixed_support_mask = None
    if args.fixed_support_mask_mode in {'topk_kappa', 'maxgap_kappa'}:
        if adj_binary is None:
            raise ValueError(f'--fixed_support_mask_mode {args.fixed_support_mask_mode} requires the undirected kappa skeleton path')
        if args.fixed_support_mask_mode == 'topk_kappa':
            support_label = 'top-k kappa'
        else:
            support_label = 'max-gap kappa'
        fixed_support_mask = adj_binary.clone().float()
        print(f"Fixed support mask: {support_label} | undirected_pairs={fixed_support_mask.sum().item() / 2:.0f}")

    if args.direction_init_mode == 'patel_tau':
        direction_init_matrix = patel_tau_matrix.clone()
    elif args.direction_init_mode == 'zeros':
        direction_init_matrix = torch.zeros_like(patel_tau_matrix)
    else:
        random_direction = torch.randn_like(patel_tau_matrix)
        direction_init_matrix = random_direction - random_direction.t()
        direction_init_matrix.fill_diagonal_(0.0)

    

    # Create results folder with timestamp

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    result_dir = f'./results/run_{timestamp}'

    os.makedirs(result_dir, exist_ok=True)

    print(f"\nResults will be saved to: {result_dir}")

    

    # Step 6: Train model

    # - structure_init_matrix: chosen matrix for sender/receiver initialization
    # - patel_tau_matrix: weak directional prior only
    # - patel_kappa_matrix: skeleton prior for proxy scoring / noise guidance

    model, adj_matrix, loss_history, collapse_history, best_epoch = train_brain_connectivity(

        data_3d=data_3d,

        pearson_matrix=pearson_matrix,  # Pearson for reference/saving

        num_nodes=num_nodes,

        time_points=effective_time_points,

        noise_guide_adj=noise_guide_adj,  # For neighbor-based noise

        patel_matrix=structure_init_matrix,  # Chosen matrix for SVD init
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
        optimizer_step_mode=args.optimizer_step_mode,

        debug_checks=args.debug_checks,

        skip_pretrain=args.skip_pretrain,
        pretrain_checkpoint=args.pretrain_checkpoint,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        result_dir=result_dir,
        selection_top_k=args.selection_top_k,
        selection_start_epoch=args.selection_start_epoch,
        selection_min_skeleton_overlap=args.selection_min_skeleton_overlap,
        selection_min_skeleton_retention=args.selection_min_skeleton_retention,
        selection_min_density_factor=args.selection_min_density_factor,
        selection_max_density_ratio=args.selection_max_density_ratio,
        enable_directional_loss=not args.disable_directional_loss,
        directional_prior_mode=args.directional_prior_mode,
        directional_schedule=args.directional_schedule,
        lag_direction_source=args.lag_direction_source,
        directional_kappa_gate=args.directional_kappa_gate,
        directional_kappa_gate_quantile=args.directional_kappa_gate_quantile,
        directional_target_ratio=args.directional_target_ratio,
        main_loss_weight=args.main_loss_weight,
        directional_loss_end_epoch=args.directional_loss_end_epoch,
        direction_lr_multiplier=args.direction_lr_multiplier,
        freeze_direction_after_epoch=args.freeze_direction_after_epoch,
        enable_gradient_alignment_probe=args.enable_gradient_alignment_probe,
        gradient_alignment_probe_seed=args.gradient_alignment_probe_seed,
        enable_cross_prediction=args.enable_cross_prediction,
        cross_pred_target_ratio=args.cross_pred_target_ratio,
        cross_pred_schedule=args.cross_pred_schedule,
        cross_pred_aggregation=args.cross_pred_aggregation,
        cross_pred_softmax_temp=args.cross_pred_softmax_temp,
        parent_entropy_lambda=args.parent_entropy_lambda,
        parent_entropy_warmup_epochs=args.parent_entropy_warmup_epochs,
        parent_entropy_ramp_epochs=args.parent_entropy_ramp_epochs,
        parent_cap_lambda=args.parent_cap_lambda,
        parent_cap_target=args.parent_cap_target,
        parent_cap_warmup_epochs=args.parent_cap_warmup_epochs,
        parent_cap_ramp_epochs=args.parent_cap_ramp_epochs,
        ungated_symmetry_lambda=args.ungated_symmetry_lambda,
        ungated_symmetry_warmup_epochs=args.ungated_symmetry_warmup_epochs,
        ungated_symmetry_ramp_epochs=args.ungated_symmetry_ramp_epochs,
        selection_agreement_weight=args.selection_agreement_weight,
        ddm_kwargs={
            'use_temporal_encoder': not args.disable_temporal_encoder,
            'uniform_timestep': not args.per_node_timestep,
            'noise_norm_mode': args.noise_norm_mode,
            'noise_zero_mean': not args.noise_with_mean,
            'init_logit_scale': args.structure_init_scale,
            'emb_dim': None if args.emb_dim <= 0 else args.emb_dim,
            'structure_parameterization': args.structure_parameterization,
            'structure_message_graph_mode': args.structure_message_graph_mode,
            'adj_activation': args.adj_activation,
            'kappa_logit_bias_scale': args.kappa_logit_bias_scale,
            'direction_init_features': direction_init_matrix,
            'fixed_support_mask': fixed_support_mask,
            'loss_type': args.loss_type,
            'cosine_weight': args.cosine_weight,
            'mse_weight': args.mse_weight,
        },

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
    best_epoch_adj_causal = getattr(model, 'best_epoch_adj_matrix_causal', to_causal_matrix_np(adj_matrix))
    final_epoch_adj_causal = getattr(model, 'last_epoch_adj_matrix_causal', None)

    # Save adjacency matrix to results folder (both npy and csv)

    adj_save_path = os.path.join(result_dir, 'learned_adjacency.npy')

    np.save(adj_save_path, adj_matrix)

    

    # Save as CSV for easy viewing

    adj_csv_path = os.path.join(result_dir, 'learned_adjacency.csv')

    pd.DataFrame(adj_matrix).to_csv(adj_csv_path, index=False, header=False, float_format='%.8f')

    adj_causal_save_path = os.path.join(result_dir, 'learned_adjacency_causal.npy')
    adj_causal_csv_path = os.path.join(result_dir, 'learned_adjacency_causal.csv')
    np.save(adj_causal_save_path, best_epoch_adj_causal)
    pd.DataFrame(best_epoch_adj_causal).to_csv(
        adj_causal_csv_path, index=False, header=False, float_format='%.8f'
    )

    if final_epoch_adj is not None:
        final_adj_save_path = os.path.join(result_dir, 'final_epoch_adjacency.npy')
        final_adj_csv_path = os.path.join(result_dir, 'final_epoch_adjacency.csv')
        np.save(final_adj_save_path, final_epoch_adj)
        pd.DataFrame(final_epoch_adj).to_csv(
            final_adj_csv_path, index=False, header=False, float_format='%.8f'
        )
    if final_epoch_adj_causal is not None:
        final_adj_causal_save_path = os.path.join(result_dir, 'final_epoch_adjacency_causal.npy')
        final_adj_causal_csv_path = os.path.join(result_dir, 'final_epoch_adjacency_causal.csv')
        np.save(final_adj_causal_save_path, final_epoch_adj_causal)
        pd.DataFrame(final_epoch_adj_causal).to_csv(
            final_adj_causal_csv_path, index=False, header=False, float_format='%.8f'
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
    config['effective_time_points'] = int(data_3d.shape[-1])
    config['main_loss_weight'] = float(getattr(model, 'main_loss_weight', args.main_loss_weight))

    config['num_nodes'] = int(num_nodes)
    config['noise_guide_pairs'] = int(k_pairs)
    config['selection_top_k'] = int(args.selection_top_k) if args.selection_top_k is not None else int(k_pairs)
    config['exported_epoch'] = int(best_epoch)
    config['best_proxy_score'] = float(getattr(model, 'best_epoch_score', -1.0))
    config['best_epoch_selection_mode'] = str(getattr(model, 'best_epoch_selection_mode', 'unknown'))
    config['best_epoch_guarded'] = int(getattr(model, 'best_epoch_guarded', -1))
    config['best_epoch_guarded_score'] = float(getattr(model, 'best_epoch_guarded_score', -1.0))
    config['best_epoch_score_only'] = int(getattr(model, 'best_epoch_score_only', -1))
    config['best_epoch_score_only_score'] = float(getattr(model, 'best_epoch_score_only_score', -1.0))
    config['raw_adjacency_convention'] = RAW_ADJ_CONVENTION
    config['causal_adjacency_convention'] = CAUSAL_ADJ_CONVENTION
    config['learned_adjacency_file_semantics'] = 'raw_internal_convention'
    config['learned_adjacency_causal_file_semantics'] = 'causal_convention_for_eval'
    config['requested_emb_dim'] = int(args.emb_dim)
    config['effective_emb_dim'] = int(getattr(model, 'emb_dim', num_nodes))
    config['structure_message_graph_mode'] = str(getattr(model, 'structure_message_graph_mode', args.structure_message_graph_mode))
    config['adj_activation'] = str(getattr(model, 'adj_activation', args.adj_activation))
    config['directional_loss_deactivated_from_epoch'] = int(getattr(model, 'directional_loss_deactivated_from_epoch', -1))
    config['direction_branch_frozen_from_epoch'] = int(getattr(model, 'direction_branch_frozen_from_epoch', -1))
    final_adj_diag = getattr(model, 'last_epoch_adj_diagnostics', None)
    if final_adj_diag is not None:
        for key, value in final_adj_diag.items():
            config[f'final_{key}'] = float(value)

    np.save(os.path.join(result_dir, 'config.npy'), config, allow_pickle=True)

    

    # Save Pearson matrix for reference (both npy and csv)

    np.save(os.path.join(result_dir, 'pearson_matrix.npy'), pearson_matrix.numpy())

    pd.DataFrame(pearson_matrix.numpy()).to_csv(

        os.path.join(result_dir, 'pearson_matrix.csv'), index=False, header=False, float_format='%.4f'

    )

    np.save(os.path.join(result_dir, 'structure_init_matrix.npy'), structure_init_matrix.numpy())
    pd.DataFrame(structure_init_matrix.numpy()).to_csv(
        os.path.join(result_dir, 'structure_init_matrix.csv'),
        index=False,
        header=False,
        float_format='%.8f',
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
    print(f"  [Selection Mode: {getattr(model, 'best_epoch_selection_mode', 'unknown')}]")

    print(f"  - loss_curve.png          <- 查看此图判断收敛")

    print(f"  - learned_adjacency.csv   <- best-epoch 原始邻接矩阵（raw convention）")

    print(f"  - learned_adjacency_causal.csv <- best-epoch 因果方向邻接矩阵（causal convention）")

    print(f"  - final_epoch_adjacency.csv <- 最后一轮原始邻接矩阵（用于对照）")

    print(f"  - final_epoch_adjacency_causal.csv <- 最后一轮因果邻接矩阵（用于对照）")

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

    final_adj_diag = getattr(model, 'last_epoch_adj_diagnostics', None)
    if final_adj_diag is not None:
        print("Final adjacency diagnostics (causal, off-diagonal):")
        print(f"  - Offdiag Mean: {final_adj_diag['adj_offdiag_mean']:.4f}")
        print(f"  - Offdiag Std:  {final_adj_diag['adj_offdiag_std']:.4f}")
        print(f"  - Offdiag CV:   {final_adj_diag['adj_offdiag_cv']:.4f}")
        print(f"  - Offdiag Min:  {final_adj_diag['adj_offdiag_min']:.4f}")
        print(f"  - Offdiag Max:  {final_adj_diag['adj_offdiag_max']:.4f}")
        print(f"  - InDeg Mean:   {final_adj_diag['adj_in_degree_mean']:.4f}")
        print(f"  - InDeg Std:    {final_adj_diag['adj_in_degree_std']:.4f}")

    



if __name__ == '__main__':

    main()

