#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretrain NodeSpecificTemporalEncoder with three-objective loss:
  1. Reconstruction  — faithfulness to original time series
  2. Forecasting     — future-predictive information
  3. VICReg          — anti-collapse (variance + covariance)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from models.DDM import TemporalResidualBlock, NodeSpecificTemporalEncoder


# ============================================================================
# DECODER & PREDICTION HEADS
# ============================================================================

class TemporalDecoder(nn.Module):
    """
    Lightweight symmetric decoder: z [N, H] -> reconstructed [N, T].

    Intentionally shallow (2 blocks vs encoder's 4) so the bottleneck
    stays in z, preventing the decoder from being too powerful.
    """

    def __init__(self, hidden_channels: int, output_dim: int, time_points: int):
        super().__init__()
        self.time_points = time_points
        self.hidden_channels = hidden_channels

        # Project encoding back to temporal feature map
        self.expand = nn.Linear(output_dim, hidden_channels * time_points)

        # 2 residual blocks (lighter than encoder's 4)
        self.blocks = nn.ModuleList([
            TemporalResidualBlock(hidden_channels, kernel_size=3, dilation=1),
            TemporalResidualBlock(hidden_channels, kernel_size=3, dilation=2),
        ])

        # Final projection: hidden_channels -> 1 channel
        self.out_conv = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, z):
        """
        Args:
            z: [B*N, output_dim]
        Returns:
            recon: [B*N, T]
        """
        x = self.expand(z)  # [B*N, C*T]
        x = x.view(-1, self.hidden_channels, self.time_points)  # [B*N, C, T]

        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)  # [B*N, 1, T]
        return x.squeeze(1)   # [B*N, T]


class ForecastingHead(nn.Module):
    """3-layer MLP: z [N, H] -> predicted future [N, Q]."""

    def __init__(self, input_dim: int, forecast_len: int):
        super().__init__()
        mid = max(input_dim, forecast_len)
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.PReLU(),
            nn.Linear(mid, mid),
            nn.PReLU(),
            nn.Linear(mid, forecast_len),
        )

    def forward(self, z):
        return self.net(z)


# ============================================================================
# VICReg LOSS
# ============================================================================

def vicreg_loss(z: torch.Tensor):
    """
    Compute VICReg variance + covariance terms on encoding matrix z.

    Args:
        z: [M, H] where M = num samples, H = feature dim

    Returns:
        var_loss: scalar — penalizes dimensions with std < 1.0
        cov_loss: scalar — penalizes off-diagonal covariance
    """
    M, H = z.shape

    # --- Variance term ---
    std_per_dim = torch.sqrt(z.var(dim=0) + 1e-4)  # [H]
    var_loss = F.relu(1.0 - std_per_dim).mean()

    # --- Covariance term ---
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / (M - 1)  # [H, H]
    # Zero out diagonal
    off_diag_mask = 1.0 - torch.eye(H, device=z.device)
    cov_loss = (cov * off_diag_mask).pow(2).sum() / H

    return var_loss, cov_loss


# ============================================================================
# PRETRAIN MAIN FUNCTION
# ============================================================================

def pretrain_temporal_encoder(
    encoder: NodeSpecificTemporalEncoder,
    data_3d: torch.Tensor,
    device: str = 'cuda',
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    split_ratio: float = 0.75,
    log_interval: int = 5,
    # Loss weights
    w_recon: float = 1.0,
    w_forecast: float = 0.5,
    w_variance: float = 1.0,
    w_covariance: float = 0.04,
    forecast_warmup_epochs: int = 10,
) -> Dict[str, list]:
    """
    Pretrain the temporal encoder with reconstruction + forecasting + VICReg.

    Args:
        encoder: NodeSpecificTemporalEncoder instance (will be trained in-place)
        data_3d: [Num_Subjects, N, T] raw fMRI data
        device: torch device string
        num_epochs: number of pretrain epochs
        learning_rate: optimizer lr
        split_ratio: fraction of time points used as input (rest = forecast target)
        log_interval: epochs between log prints
        w_recon, w_forecast, w_variance, w_covariance: loss weights
        forecast_warmup_epochs: linear warmup epochs for forecast loss

    Returns:
        history: dict with per-epoch loss lists
    """
    encoder = encoder.to(device)
    data_3d = data_3d.to(device)

    num_subjects, N, T = data_3d.shape
    P = int(T * split_ratio)  # input length
    Q = T - P                 # forecast length

    print(f"[Pretrain] T={T}, P={P} (input), Q={Q} (forecast)")
    print(f"[Pretrain] Subjects={num_subjects}, Nodes={N}")
    print(f"[Pretrain] Weights: recon={w_recon}, forecast={w_forecast}, "
          f"var={w_variance}, cov={w_covariance}")

    # Build decoder and forecasting head
    decoder = TemporalDecoder(
        hidden_channels=encoder.hidden_channels,
        output_dim=encoder.output_dim,
        time_points=T,
    ).to(device)

    forecast_head = ForecastingHead(
        input_dim=encoder.output_dim,
        forecast_len=Q,
    ).to(device)

    # Optimizer over encoder + decoder + forecast_head
    all_params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(forecast_head.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=learning_rate)

    history = {
        'total': [], 'recon': [], 'forecast': [],
        'variance': [], 'covariance': [],
    }

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        forecast_head.train()

        epoch_losses = {k: 0.0 for k in history}
        perm = torch.randperm(num_subjects)

        # Forecast weight warmup: linear 0 → w_forecast over warmup epochs
        if epoch < forecast_warmup_epochs:
            cur_w_forecast = w_forecast * (epoch / forecast_warmup_epochs)
        else:
            cur_w_forecast = w_forecast

        for s_idx in perm:
            x = data_3d[s_idx]  # [N, T]

            # --- Loss 1: Reconstruction ---
            feature_map, z_full = encoder.encode_features(x)  # z_full: [N, H]
            # Flatten for decoder: feature_map is [N, C, T], z_full is [N, H]
            x_recon = decoder(z_full)  # [N, T]
            recon_loss = F.mse_loss(x_recon, x)

            # --- Loss 2: Forecasting ---
            x_partial = x[:, :P]  # [N, P]
            z_partial = encoder(x_partial)  # [N, H]  (uses forward, GAP adapts to shorter T)
            future_pred = forecast_head(z_partial)  # [N, Q]
            forecast_loss = F.mse_loss(future_pred, x[:, P:])

            # --- Loss 3: VICReg ---
            var_loss, cov_loss = vicreg_loss(z_full)

            # --- Total ---
            total = (
                w_recon * recon_loss
                + cur_w_forecast * forecast_loss
                + w_variance * var_loss
                + w_covariance * cov_loss
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_losses['total'] += total.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['forecast'] += forecast_loss.item()
            epoch_losses['variance'] += var_loss.item()
            epoch_losses['covariance'] += cov_loss.item()

        # Average over subjects
        for k in epoch_losses:
            epoch_losses[k] /= num_subjects
            history[k].append(epoch_losses[k])

        if (epoch + 1) % log_interval == 0:
            print(
                f"[Pretrain] Epoch [{epoch+1:3d}/{num_epochs}] | "
                f"Total: {epoch_losses['total']:.4f} | "
                f"Recon: {epoch_losses['recon']:.4f} | "
                f"Forecast: {epoch_losses['forecast']:.4f} | "
                f"Var: {epoch_losses['variance']:.4f} | "
                f"Cov: {epoch_losses['covariance']:.4f} | "
                f"w_fc={cur_w_forecast:.3f}"
            )

            # Quick collapse check
            with torch.no_grad():
                sample_z = encoder(data_3d[0])  # [N, H]
                sample_z = F.layer_norm(sample_z, (sample_z.shape[-1],))
                z_normed = F.normalize(sample_z, p=2, dim=-1)
                sim = (z_normed @ z_normed.T)
                mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
                mean_cos = sim[mask].mean().item()
                std_per_dim = sample_z.std(dim=0)
                dead_ratio = (std_per_dim < 1e-3).float().mean().item()
            print(
                f"  [Collapse] mean_cos={mean_cos:.4f} | "
                f"dead_dims={dead_ratio:.2%} | "
                f"feat_std={std_per_dim.mean().item():.4f}"
            )

    # Cleanup: we only keep the encoder, discard decoder & forecast_head
    del decoder, forecast_head, optimizer
    torch.cuda.empty_cache() if device != 'cpu' else None

    return history


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    from main_structure_learning import load_fmri_data, set_seed

    parser = argparse.ArgumentParser(description='Pretrain Temporal Encoder')
    parser.add_argument('--csv_path', type=str, default='../fMRI_dataset/sim4.csv')
    parser.add_argument('--time_points', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--split_ratio', type=float, default=0.75)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='./pretrained_encoder.pt')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    set_seed(args.seed)

    data_3d, _, _, num_nodes = load_fmri_data(args.csv_path, args.time_points)

    encoder = NodeSpecificTemporalEncoder(
        time_points=args.time_points,
        hidden_channels=32,
        output_dim=args.num_hidden,
        num_blocks=4,
        kernel_size=3,
    )

    history = pretrain_temporal_encoder(
        encoder=encoder,
        data_3d=data_3d,
        device=args.device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        split_ratio=args.split_ratio,
    )

    torch.save(encoder.state_dict(), args.save_path)
    print(f"\nSaved pretrained encoder to: {args.save_path}")
