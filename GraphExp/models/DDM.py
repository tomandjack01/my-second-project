#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File Name:     diffusion.py
# Author:        Yang Run
# Created Time:  2022-10-29  17:09
# Last Modified: <none>-<none>

import sys
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import math
import dgl
import dgl.function as fn
from utils.utils import make_edge_weights
from .mlp_gat import Denoising_Unet
import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# ============================================================================
# NODE-SPECIFIC CAUSAL DILATED TEMPORAL ENCODER
# ============================================================================

class CausalConv1d(nn.Module):
    """单层因果一维卷积算子"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]  # 严格截断未来信息
        return x


class NodeSpecificTemporalEncoder(nn.Module):
    """1-2-4 膨胀策略的时间因果编码器，带自回归预训练头"""
    def __init__(self, time_points=200, hidden_channels=32, output_dim=64, **kwargs):
        # 兼容旧代码的入参，但内部使用新逻辑
        super(NodeSpecificTemporalEncoder, self).__init__()
        self.time_points = time_points
        self.hidden_channels = hidden_channels
        self.output_dim = time_points  # 输出维度 = 时间点数，保持 [N, T] → [N, T]

        self.conv1 = CausalConv1d(1, hidden_channels, kernel_size=3, dilation=1)
        self.conv2 = CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=2)
        self.conv3 = CausalConv1d(hidden_channels, hidden_channels, kernel_size=3, dilation=4)

        # 将多通道特征压扁回单通道
        self.projector = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        # 预测头适配器：用于将特征还原到原始信号的尺度
        self.pred_head = nn.Linear(time_points, time_points)
        self.norm = nn.LayerNorm(time_points)

        # 兼容旧代码中对 input_proj 的引用（如 diagnose_encoder_collapse）
        self.input_proj = self.conv1.conv

    def forward(self, x, return_unnormalized=False):
        # x: [N, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [N, 1, T]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        unnormalized_x = self.projector(x).squeeze(1)  # [N, T]

        norm_x = self.norm(unnormalized_x)
        if return_unnormalized:
            return norm_x, unnormalized_x
        return norm_x

    def pretrain_forward(self, x):
        """自回归预训练：用 t 时刻预测 t+1 时刻，避开 LayerNorm 的尺度破坏"""
        _, unnormalized_x = self.forward(x, return_unnormalized=True)
        # 通过预测头映射尺度
        pred_x = self.pred_head(unnormalized_x)

        predictions = pred_x[:, :-1]  # 前 T-1 个时间点
        targets = x[:, 1:]            # 后 T-1 个时间点真实信号
        return F.mse_loss(predictions, targets)


class DDM(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            norm: Optional[str],
            alpha_l: float = 2,
            beta_schedule: str = 'linear',
            beta_1: float = 0.0001,
            beta_T: float = 0.02,
            T: int = 1000,
            init_features: Optional[torch.Tensor] = None,
            noise_guide_adj: Optional[torch.Tensor] = None,
            preserve_noise_sign: bool = False,
            # Temporal encoder parameters
            temporal_hidden_channels: int = 32,
            use_temporal_encoder: bool = True,
            **kwargs

         ):
        super(DDM, self).__init__()
        self.T = T
        beta = get_beta_schedule(beta_schedule, beta_1, beta_T, T)
        self.register_buffer(
                'betas', beta
                )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
                'sqrt_alphas_bar', torch.sqrt(alphas_bar)
                )
        self.register_buffer(
                'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
                )

        self.alpha_l = alpha_l
        self.in_dim = in_dim  # Store original time series length
        self.num_hidden = num_hidden
        self.use_temporal_encoder = use_temporal_encoder

        # Temporal Encoder: Causal dilated conv, output stays [N, in_dim]
        # Input: [N, in_dim] → Output: [N, in_dim] (same time dimension)
        if self.use_temporal_encoder:
            self.temporal_encoder = NodeSpecificTemporalEncoder(
                time_points=in_dim,
                hidden_channels=temporal_hidden_channels,
                output_dim=in_dim,  # 输出维度 = 时间点数，不再压缩
            )
            print(f"Temporal Encoder: {in_dim} time points → {in_dim} causal features")
        else:
            self.temporal_encoder = None
            print(f"Temporal Encoder: DISABLED - Using raw time series directly")

        # 扩散过程始终在原始时间维度空间中进行
        denoising_in_dim = in_dim

        assert num_hidden % nhead == 0
        # Denoising UNet works in either encoded feature space or raw time series space
        # Input/Output: denoising_in_dim (encoded features if encoder enabled, raw time points otherwise)
        self.net = Denoising_Unet(in_dim=denoising_in_dim,
                                  num_hidden=num_hidden,
                                  out_dim=denoising_in_dim,
                                  num_layers=num_layers,
                                  nhead=nhead,
                                  activation=activation,
                                  feat_drop=feat_drop,
                                  attn_drop=attn_drop,
                                  negative_slope=0.2,
                                  norm=norm)

        self.time_embedding = nn.Embedding(T, num_hidden)

        # Neighbor-Based Noise: Store row-normalized adjacency matrix for noise guidance
        # noise_guide_adj should be [N, N], row-normalized (each row sums to 1)
        if noise_guide_adj is not None:
            self.register_buffer('noise_guide_adj', noise_guide_adj)
            print(f"Using Neighbor-Based Noise with adj shape {noise_guide_adj.shape}")
        else:
            self.noise_guide_adj = None

        self.preserve_noise_sign = preserve_noise_sign

        # Graph Structure Learning: Initialize learnable adjacency matrix
        self.structure_learning_mode = init_features is not None
        if self.structure_learning_mode:
            N, D = init_features.shape
            # Compute Pearson correlation matrix
            feat_norm = (init_features - init_features.mean(dim=0)) / (init_features.std(dim=0) + 1e-8)
            pearson_corr = feat_norm @ feat_norm.T / D
            # Initialize learnable adjacency parameter
            self.learned_adj = nn.Parameter(pearson_corr.clone())
            # Create fully connected DGL graph (N x N edges)
            src = torch.arange(N).repeat_interleave(N)
            dst = torch.arange(N).repeat(N)
            self.register_buffer('full_g_src', src)
            self.register_buffer('full_g_dst', dst)
            self.num_nodes = N

    def _get_structure_graph(self, device):
        """Create fully connected graph and compute edge weights from learned_adj."""
        g = dgl.graph((self.full_g_src, self.full_g_dst), num_nodes=self.num_nodes)
        g = g.to(device)
        # Apply sigmoid to get edge weights in [0, 1]
        edge_weights = torch.sigmoid(self.learned_adj).flatten()
        return g, edge_weights

    def forward(self, g, x):
        """
        Forward pass with optional temporal encoding.

        Args:
            g: DGL graph (can be None if using structure learning mode)
            x: Raw time series input [N, T] or [B, N, T]
               where T = in_dim (time points, e.g., 200)

        Returns:
            loss: Diffusion loss
            loss_item: Dictionary with loss value
        """
        # Step 1: Optionally encode raw time series via causal conv
        if self.use_temporal_encoder:
            # [N, T] → [N, T] (causal features, same dimension)
            x_processed = self.temporal_encoder(x)
        else:
            # Use raw time series directly
            x_processed = x

        # Step 2: Apply layer normalization
        x_processed = F.layer_norm(x_processed, (x_processed.shape[-1], ))

        # Step 3: Sample random timestep for diffusion
        t = torch.randint(self.T, size=(x_processed.shape[0], ), device=x_processed.device)

        # Step 4: Get graph structure
        if self.structure_learning_mode:
            g, edge_weight = self._get_structure_graph(x_processed.device)
        else:
            edge_weight = None

        # Step 5: Diffusion forward process (add noise)
        x_t, time_embed, g = self.sample_q(t, x_processed, g)

        # Step 6: Denoise and compute loss
        loss = self.node_denoising(x_processed, x_t, time_embed, g, edge_weight=edge_weight)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def sample_q(self, t, x, g):
        """
        Diffusion forward process with Neighbor-Based Statistical Noise.
        
        For each node i, noise is drawn from N(mu_neighbors, sigma_neighbors)
        where neighbors are defined by self.noise_guide_adj.
        """
        # Determine input shape: [N, Feats] or [Batch, N, Feats]
        is_batched = x.dim() == 3
        
        if is_batched:
            B, N, D = x.shape
        else:
            N, D = x.shape
            # Add batch dimension for unified processing
            x = x.unsqueeze(0)  # [1, N, D]
            B = 1
        
        # Global statistics as fallback
        global_mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        global_std = x.std(dim=1, keepdim=True) + 1e-6  # [B, 1, D]
        
        # Generate base random noise
        eps = torch.randn_like(x)  # [B, N, D]
        
        if self.noise_guide_adj is not None:
            # Neighbor-Based Noise using matrix operations
            # noise_guide_adj: [N, N], x: [B, N, D]
            
            # Step 1: Compute neighbor mean
            # [N, N] @ [B, N, D] -> need to handle batch dimension
            # Reshape for batched matmul: adj @ x[b] for each batch
            adj = self.noise_guide_adj  # [N, N]
            
            # neighbor_mean[b, i, :] = sum_j(adj[i,j] * x[b, j, :])
            # Using einsum for clarity: 'ij, bjd -> bid'
            neighbor_mean = torch.einsum('ij,bjd->bid', adj, x)  # [B, N, D]
            
            # Step 2: Compute neighbor variance using E[X^2] - E[X]^2
            x_sq = x ** 2  # [B, N, D]
            neighbor_sq_mean = torch.einsum('ij,bjd->bid', adj, x_sq)  # [B, N, D]
            neighbor_var = neighbor_sq_mean - neighbor_mean ** 2
            
            # Clamp to avoid negative variance due to numerical issues
            neighbor_std = torch.sqrt(torch.clamp(neighbor_var, min=1e-6))  # [B, N, D]
            
            # Step 3: Generate noise with neighbor statistics
            noise = eps * neighbor_std + neighbor_mean  # [B, N, D]
        else:
            # Fallback to global statistics
            noise = eps * global_std + global_mean
        
        # Apply layer normalization to noise
        with torch.no_grad():
            noise = F.layer_norm(noise, (noise.shape[-1], ))
        
        # Optional: preserve sign alignment with the input
        if self.preserve_noise_sign:
            noise = torch.sign(x) * torch.abs(noise)
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            x = x.squeeze(0)
            noise = noise.squeeze(0)
        
        # Diffusion forward process
        x_t = (
            extract(self.sqrt_alphas_bar, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
        )
        time_embed = self.time_embedding(t)
        return x_t, time_embed, g

    def node_denoising(self, x, x_t, time_embed, g, edge_weight=None):
        out, _ = self.net(g, x_t=x_t, time_embed=time_embed, edge_weight=edge_weight)
        loss = loss_fn(out, x, self.alpha_l)

        return loss

    def embed(self, g, x, T):
        """
        Generate embeddings from raw time series.

        Args:
            g: DGL graph (can be None if using structure learning mode)
            x: Raw time series input [N, T] or [B, N, T]
            T: Diffusion timestep for embedding

        Returns:
            hidden: Encoded hidden representations
        """
        # Optionally encode raw time series first
        if self.use_temporal_encoder:
            x_processed = self.temporal_encoder(x)
        else:
            x_processed = x

        t = torch.full((1, ), T, device=x_processed.device)
        with torch.no_grad():
            x_processed = F.layer_norm(x_processed, (x_processed.shape[-1], ))

        # Use learned structure if in structure learning mode
        if self.structure_learning_mode:
            g, edge_weight = self._get_structure_graph(x_processed.device)
        else:
            edge_weight = None

        x_t, time_embed, g = self.sample_q(t, x_processed, g)
        _, hidden = self.net(g, x_t=x_t, time_embed=time_embed, edge_weight=edge_weight)
        return hidden


def loss_fn(x, y, alpha=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return torch.from_numpy(betas)
