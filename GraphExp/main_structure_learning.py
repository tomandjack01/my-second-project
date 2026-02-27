#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Graph Structure Learning for fMRI Brain Connectivity



Learns a shared brain connectivity matrix from fMRI time-series data

using DDM (Directional Diffusion Models) with L1 sparsity regularization.

"""



import argparse

import os

import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F

from typing import Any, Dict, Optional

from datetime import datetime

from collections import defaultdict



from models import DDM

from utils.patel_util import compute_patel_matrix

from pretrain_temporal_encoder import pretrain_temporal_encoder

import matplotlib.pyplot as plt


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

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)





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





def train_brain_connectivity(

    data_3d: torch.Tensor,

    pearson_matrix: torch.Tensor,

    num_nodes: int,

    time_points: int,

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
    pretrain_split_ratio: float = 0.75,
    result_dir: Optional[str] = None,

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

        noise_guide_adj: Row-normalized adjacency matrix for neighbor-based noise

        ddm_kwargs: Optional extra keyword arguments forwarded to DDM

    

    Returns:

        model: Trained DDM model

        adj_matrix: Learned adjacency matrix [N, N]

    """

    num_subjects = data_3d.shape[0]

    data_3d = data_3d.to(device)

    ddm_kwargs = {} if ddm_kwargs is None else dict(ddm_kwargs)

    

    # Initialize DDM with Pearson matrix for structure learning

    # in_dim = TIME_POINTS (features per node)

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

        init_features=pearson_matrix,  # [N, N] for structure learning

        noise_guide_adj=noise_guide_adj,  # Row-normalized adj for neighbor-based noise

        **ddm_kwargs,

    )

    model = model.to(device)

    # ---- Encoder Pretraining / Loading ----
    if not skip_pretrain:
        if pretrain_checkpoint and os.path.exists(pretrain_checkpoint):
            print(f"\n[Pretrain] Loading encoder weights from: {pretrain_checkpoint}")
            state = torch.load(pretrain_checkpoint, map_location=device)
            model.temporal_encoder.load_state_dict(state)
        else:
            print(f"\n[Pretrain] Starting encoder pretraining for {pretrain_epochs} epochs...")
            pretrain_history = pretrain_temporal_encoder(
                encoder=model.temporal_encoder,
                data_3d=data_3d,
                device=device,
                num_epochs=pretrain_epochs,
                learning_rate=pretrain_lr,
                split_ratio=pretrain_split_ratio,
            )
            # Save pretrained weights
            if result_dir:
                save_path = os.path.join(result_dir, 'pretrained_encoder.pt')
                torch.save(model.temporal_encoder.state_dict(), save_path)
                print(f"[Pretrain] Saved encoder weights to: {save_path}")

        # Post-pretrain collapse diagnostics
        print("\n[Pretrain] Post-pretrain collapse diagnostics:")
        pt_metrics = diagnose_encoder_collapse(model, data_3d, device)
        print_collapse_diagnostics(pt_metrics, 0, 1)

    # ---- Freeze encoder ----
    if not skip_pretrain:
        print("\n[Freeze] Freezing temporal_encoder parameters")
        for param in model.temporal_encoder.parameters():
            param.requires_grad = False

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

    

    # Training loop

    for epoch in range(num_epochs):

        model.train()

        epoch_loss = 0.0

        epoch_sparsity = 0.0

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

                        x_encoded = model.temporal_encoder(x)

                        x_encoded = F.layer_norm(x_encoded, (x_encoded.shape[-1],))

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

                

                # L1 sparsity regularization on learned adjacency

                # We exclude self-loops (diagonal) from sparsity penalty to allow for auto-regression

                # NORMALIZED by the number of off-diagonal elements

                adj_sigmoid = torch.sigmoid(model.learned_adj)

                

                # Mask out diagonal (keep only off-diagonal for penalty)

                mask_off_diag = 1.0 - torch.eye(num_nodes, device=device)

                adj_off_diag = adj_sigmoid * mask_off_diag

                

                l1_norm = torch.norm(adj_off_diag, p=1)

                n_off_diag = num_nodes * num_nodes - num_nodes  # Total elements - Diagonal elements

                

                # Avoid division by zero (though N is typically > 1)

                if n_off_diag > 0:

                    sparsity_loss = lambda_l1 * (l1_norm / n_off_diag)

                else:

                    sparsity_loss = torch.tensor(0.0, device=device)

                

                # Total loss

                total_loss = loss + sparsity_loss

                

                # Backward pass

                total_loss.backward()

                if debug_checks and epoch == 0 and i == 0 and subj_idx == 0:

                    grad = model.temporal_encoder.input_proj.weight.grad

                    if grad is None:
                        print("[Debug] temporal_encoder grad is None")
                    else:
                        print(f"[Debug] temporal_encoder grad norm: {grad.norm().item():.6e}")

                optimizer.step()

                

                epoch_loss += loss.item()

                epoch_sparsity += sparsity_loss.item()

                num_batches += 1

        

        # Log progress

        if (epoch + 1) % log_interval == 0:

            avg_loss = epoch_loss / num_batches

            avg_sparsity = epoch_sparsity / num_batches

            

            with torch.no_grad():

                adj_sigmoid = torch.sigmoid(model.learned_adj)

                adj_mean = adj_sigmoid.mean().item()

                sparsity_ratio = (adj_sigmoid < 0.5).float().mean().item()

            

            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "

                  f"Diff Loss: {avg_loss:.4f} | "

                  f"Sparsity Loss: {avg_sparsity:.4f} | "

                  f"Adj Mean: {adj_mean:.3f} | "

                  f"Sparsity: {sparsity_ratio:.2%}")

            # --- Encoder collapse diagnostics ---
            collapse_metrics = diagnose_encoder_collapse(model, data_3d, device)
            print_collapse_diagnostics(collapse_metrics, epoch, num_epochs)
            collapse_history.append(collapse_metrics)



        # Record loss for every epoch

        loss_history.append(epoch_loss / num_batches)

    

    # Extract final adjacency matrix

    with torch.no_grad():

        adj_matrix = torch.sigmoid(model.learned_adj).cpu().numpy()

    

    return model, adj_matrix, loss_history, collapse_history





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

    parser.add_argument('--lambda_l1', type=float, default=0.1,

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

                        help='Number of top edges for Patel functional module detection')

    parser.add_argument('--debug_checks', action='store_true', default=False,

                        help='Run one-step debug checks (cos(x_t,x_encoded) and temporal encoder grad)')

    # Pretrain arguments
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                        help='Number of encoder pretrain epochs')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3,
                        help='Learning rate for encoder pretraining')
    parser.add_argument('--pretrain_split_ratio', type=float, default=0.75,
                        help='Fraction of time points as input (rest = forecast target)')
    parser.add_argument('--skip_pretrain', action='store_true', default=False,
                        help='Skip encoder pretraining entirely')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='Path to existing pretrained encoder weights to load')

    

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

    

    # Step 2: Compute Patel connectivity matrix for noise guidance

    print("\nComputing Patel connectivity matrix for neighbor-based noise...")

    patel_matrix = compute_patel_matrix(data_2d.numpy())  # [N, N]

    patel_matrix = torch.from_numpy(patel_matrix).float()

    print(f"Patel matrix range: [{patel_matrix.min():.4f}, {patel_matrix.max():.4f}]")

    

    # Step 3: Threshold to keep top edges (create sparse structure)

    total_edges = num_nodes * (num_nodes - 1)

    k_edges = min(args.top_k_edges, total_edges)

    

    mask_off_diag = 1.0 - torch.eye(num_nodes)

    patel_off_diag = patel_matrix * mask_off_diag

    flat_values = patel_off_diag.flatten()

    flat_values = flat_values[flat_values > 0]

    

    if len(flat_values) >= k_edges:

        threshold = torch.topk(flat_values, k_edges).values[-1]

    else:

        threshold = 0.0

    

    print(f"Keeping top {k_edges} edges (threshold: {threshold:.4f})")

    

    # Step 4: Binarize and add self-loops

    adj_binary = (patel_matrix >= threshold).float() * mask_off_diag

    adj_with_self = adj_binary + torch.eye(num_nodes)

    

    # Step 5: Row-normalize

    degree = adj_with_self.sum(dim=1, keepdim=True)

    noise_guide_adj = adj_with_self / (degree + 1e-9)

    

    print(f"Noise guide adj: {adj_binary.sum().item():.0f} edges + {num_nodes} self-loops")

    

    # Create results folder with timestamp

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    result_dir = f'./results/run_{timestamp}'

    os.makedirs(result_dir, exist_ok=True)

    print(f"\nResults will be saved to: {result_dir}")

    

    # Step 6: Train model

    # - pearson_matrix: Used as init_features (starting point for learned_adj)

    # - noise_guide_adj: Row-normalized adjacency for neighbor-based noise

    model, adj_matrix, loss_history, collapse_history = train_brain_connectivity(

        data_3d=data_3d,

        pearson_matrix=pearson_matrix,  # Pearson for initialization

        num_nodes=num_nodes,

        time_points=args.time_points,

        noise_guide_adj=noise_guide_adj,  # For neighbor-based noise

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
        pretrain_split_ratio=args.pretrain_split_ratio,
        result_dir=result_dir,

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
        log_epochs = [i * args.log_interval for i in range(len(collapse_history))]

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
        collapse_df.insert(0, 'epoch', [(i + 1) * args.log_interval for i in range(len(collapse_history))])
        collapse_csv_path = os.path.join(result_dir, 'collapse_diagnostics.csv')
        collapse_df.to_csv(collapse_csv_path, index=False, float_format='%.6f')
        print(f"Saved collapse metrics to: {collapse_csv_path}")

    

    # Save adjacency matrix to results folder (both npy and csv)

    adj_save_path = os.path.join(result_dir, 'learned_adjacency.npy')

    np.save(adj_save_path, adj_matrix)

    

    # Save as CSV for easy viewing

    adj_csv_path = os.path.join(result_dir, 'learned_adjacency.csv')

    pd.DataFrame(adj_matrix).to_csv(adj_csv_path, index=False, header=False, float_format='%.4f')

    

    # Save loss history (both npy and csv)

    np.save(os.path.join(result_dir, 'loss_history.npy'), np.array(loss_history))

    pd.DataFrame({'epoch': range(1, len(loss_history)+1), 'loss': loss_history}).to_csv(

        os.path.join(result_dir, 'loss_history.csv'), index=False

    )

    

    # Save config

    config = vars(args)

    config['num_neighbors_avg'] = float(adj_binary.sum() / num_nodes) if 'adj_binary' in dir() else 0

    config['num_subjects'] = int(data_3d.shape[0])

    config['num_nodes'] = int(num_nodes)

    np.save(os.path.join(result_dir, 'config.npy'), config, allow_pickle=True)

    

    # Save Pearson matrix for reference (both npy and csv)

    np.save(os.path.join(result_dir, 'pearson_matrix.npy'), pearson_matrix.numpy())

    pd.DataFrame(pearson_matrix.numpy()).to_csv(

        os.path.join(result_dir, 'pearson_matrix.csv'), index=False, header=False, float_format='%.4f'

    )

    

    print("=" * 60)

    print("Training Complete!")

    print("=" * 60)

    print(f"Results saved to: {result_dir}")

    print(f"  - loss_curve.png          <- 查看此图判断收敛")

    print(f"  - learned_adjacency.csv   <- 学习到的邻接矩阵")

    print(f"  - loss_history.csv")

    print(f"  - pearson_matrix.csv")

    print(f"  - config.npy")

    print(f"\nMatrix shape: {adj_matrix.shape}")

    print(f"Intensity stats:")

    print(f"  - Min:  {adj_matrix.min():.4f}")

    print(f"  - Max:  {adj_matrix.max():.4f}")

    print(f"  - Mean: {adj_matrix.mean():.4f}")

    print(f"  - Std:  {adj_matrix.std():.4f}")

    



if __name__ == '__main__':

    main()

