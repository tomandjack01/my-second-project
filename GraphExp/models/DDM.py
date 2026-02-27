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
    """
    Causal 1D Convolution with left-side padding only.
    
    Ensures no information leakage from future time steps.
    Padding size = (kernel_size - 1) * dilation
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0,  # We apply causal padding manually
            dilation=dilation
        )
    
    def forward(self, x):
        # x: [B, C, T]
        # Causal padding: pad only on the left side
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalResidualBlock(nn.Module):
    """
    Residual block with two causal dilated convolutions.
    
    Structure:
        x → CausalConv → LayerNorm → Activation → CausalConv → LayerNorm → + → Activation
        └─────────────────────── Residual Connection ──────────────────────┘
    """
    def __init__(self, channels, kernel_size, dilation, activation='prelu'):
        super(TemporalResidualBlock, self).__init__()
        
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        
        # LayerNorm for temporal dimension (applied per-channel)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Activation
        if activation == 'prelu':
            self.act1 = nn.PReLU()
            self.act2 = nn.PReLU()
        elif activation == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        else:
            self.act1 = nn.GELU()
            self.act2 = nn.GELU()
    
    def forward(self, x):
        # x: [B, C, T]
        residual = x
        
        # First conv block
        out = self.conv1(x)  # [B, C, T]
        out = out.transpose(1, 2)  # [B, T, C] for LayerNorm
        out = self.norm1(out)
        out = out.transpose(1, 2)  # [B, C, T]
        out = self.act1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        
        # Residual connection
        out = out + residual
        out = self.act2(out)
        
        return out


class NodeSpecificTemporalEncoder(nn.Module):
    """
    Node-Specific Causal Dilated Temporal Encoder.
    
    Extracts deep temporal features from each brain region's time series
    using a stack of causal dilated residual blocks with exponentially
    increasing dilation rates.
    
    Each brain region is processed independently to learn region-specific
    temporal patterns.
    
    Architecture:
        Input: [B, N, T] or [N, T]
        → Reshape to [B*N, 1, T]
        → Input projection: 1 → hidden_channels
        → Stack of TemporalResidualBlocks (dilation: 1, 2, 4, 8, ...)
        → Global Average Pooling over time
        → Output projection: hidden_channels → output_dim
        → Reshape to [B, N, output_dim]
    
    Args:
        time_points: Number of input time points (e.g., 200)
        hidden_channels: Internal channel dimension for convolutions (e.g., 32)
        output_dim: Output feature dimension per node (e.g., 64)
        num_blocks: Number of residual blocks (default: 4)
        kernel_size: Convolution kernel size (default: 3)
        activation: Activation function name (default: 'prelu')
    """
    def __init__(
        self,
        time_points: int,
        hidden_channels: int = 32,
        output_dim: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 3,
        activation: str = 'prelu'
    ):
        super(NodeSpecificTemporalEncoder, self).__init__()
        
        self.time_points = time_points
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        
        # Input projection: 1 channel → hidden_channels
        self.input_proj = nn.Conv1d(1, hidden_channels, kernel_size=1)
        
        # Stack of residual blocks with exponential dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # 1, 2, 4, 8, ...
            self.blocks.append(
                TemporalResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation
                )
            )
        
        # Output projection: hidden_channels → output_dim
        self.output_proj = nn.Linear(hidden_channels, output_dim)
        
        # Calculate and store receptive field for reference
        self.receptive_field = self._calculate_receptive_field(num_blocks, kernel_size)
    
    def _calculate_receptive_field(self, num_blocks, kernel_size):
        """Calculate the total receptive field of the encoder."""
        rf = 1
        for i in range(num_blocks):
            dilation = 2 ** i
            # Each block has 2 conv layers
            rf += 2 * (kernel_size - 1) * dilation
        return rf
    
    def encode_features(self, x):
        """
        Encode and return both pre-GAP feature map and post-GAP encoding.

        Args:
            x: Input tensor of shape [N, T] or [B, N, T]

        Returns:
            feature_map: Pre-GAP features [B*N, hidden_channels, T]
            encoding: Post-projection encoding [N, output_dim] or [B, N, output_dim]
        """
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)

        B, N, T = x.shape
        x = x.view(B * N, 1, T)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)

        feature_map = x  # [B*N, hidden_channels, T]

        pooled = x.mean(dim=2)  # [B*N, hidden_channels]
        encoding = self.output_proj(pooled)  # [B*N, output_dim]
        encoding = encoding.view(B, N, self.output_dim)

        if not is_batched:
            encoding = encoding.squeeze(0)

        return feature_map, encoding

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [N, T] or [B, N, T]
               where N = num_nodes, T = time_points

        Returns:
            Encoded features of shape [N, output_dim] or [B, N, output_dim]
        """
        # Handle input dimensions
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)  # [1, N, T]

        B, N, T = x.shape

        # Reshape: [B, N, T] → [B*N, 1, T]
        # Each node becomes an independent sample with 1 channel
        x = x.view(B * N, 1, T)

        # Input projection: [B*N, 1, T] → [B*N, hidden_channels, T]
        x = self.input_proj(x)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Global Average Pooling over time: [B*N, hidden_channels, T] → [B*N, hidden_channels]
        x = x.mean(dim=2)

        # Output projection: [B*N, hidden_channels] → [B*N, output_dim]
        x = self.output_proj(x)

        # Reshape back: [B*N, output_dim] → [B, N, output_dim]
        x = x.view(B, N, self.output_dim)

        # Remove batch dimension if input was unbatched
        if not is_batched:
            x = x.squeeze(0)  # [N, output_dim]

        return x


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
            temporal_num_blocks: int = 4,
            temporal_kernel_size: int = 3,
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
        
        # Temporal Encoder: Convert raw time series to hidden features
        # Input: [N, in_dim] or [B, N, in_dim] → Output: [N, num_hidden] or [B, N, num_hidden]
        self.temporal_encoder = NodeSpecificTemporalEncoder(
            time_points=in_dim,
            hidden_channels=temporal_hidden_channels,
            output_dim=num_hidden,
            num_blocks=temporal_num_blocks,
            kernel_size=temporal_kernel_size,
            activation=activation
        )
        print(f"Temporal Encoder: {in_dim} time points → {num_hidden} features")
        print(f"  Receptive field: {self.temporal_encoder.receptive_field} time points")
        
        assert num_hidden % nhead == 0
        # Denoising UNet now works in encoded feature space
        # Input/Output: num_hidden (encoded features, not raw time points)
        self.net = Denoising_Unet(in_dim=num_hidden,
                                  num_hidden=num_hidden,
                                  out_dim=num_hidden,
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
        Forward pass with temporal encoding.
        
        Args:
            g: DGL graph (can be None if using structure learning mode)
            x: Raw time series input [N, T] or [B, N, T]
               where T = in_dim (time points, e.g., 200)
        
        Returns:
            loss: Diffusion loss
            loss_item: Dictionary with loss value
        """
        # Step 1: Encode raw time series to feature space
        # [N, T] → [N, H] or [B, N, T] → [B, N, H]
        x_encoded = self.temporal_encoder(x)
        
        # Step 2: Apply layer normalization to encoded features
        x_encoded = F.layer_norm(x_encoded, (x_encoded.shape[-1], ))

        # Step 3: Sample random timestep for diffusion
        t = torch.randint(self.T, size=(x_encoded.shape[0], ), device=x_encoded.device)
        
        # Step 4: Get graph structure
        if self.structure_learning_mode:
            g, edge_weight = self._get_structure_graph(x_encoded.device)
        else:
            edge_weight = None
            
        # Step 5: Diffusion forward process (add noise)
        x_t, time_embed, g = self.sample_q(t, x_encoded, g)

        # Step 6: Denoise and compute loss (reconstruct encoded features)
        loss = self.node_denoising(x_encoded, x_t, time_embed, g, edge_weight=edge_weight)
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
        # Encode raw time series first
        x_encoded = self.temporal_encoder(x)
        
        t = torch.full((1, ), T, device=x_encoded.device)
        with torch.no_grad():
            x_encoded = F.layer_norm(x_encoded, (x_encoded.shape[-1], ))
        
        # Use learned structure if in structure learning mode
        if self.structure_learning_mode:
            g, edge_weight = self._get_structure_graph(x_encoded.device)
        else:
            edge_weight = None
            
        x_t, time_embed, g = self.sample_q(t, x_encoded, g)
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
