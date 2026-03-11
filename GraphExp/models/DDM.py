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
            normalize_noise: bool = True,
            preserve_noise_sign: bool = False,
            emb_dim: Optional[int] = None,  # None = full rank (N)
            adj_bias_init: Optional[float] = None,  # Sparsity bias, e.g. logit(0.025) ≈ -3.66
            # Temporal encoder parameters
            temporal_hidden_channels: int = 32,
            use_temporal_encoder: bool = True,
            # Fix 4: Uniform timestep sampling
            uniform_timestep: bool = True,
            # Fix 2: Noise normalization mode ('global', 'layernorm', 'none')
            noise_norm_mode: Optional[str] = None,
            # Fix 3: Zero-mean noise (drop neighbor mean bias)
            noise_zero_mean: bool = True,
            # Fix 6: Loss function type
            loss_type: str = 'denoise_hybrid',
            cosine_weight: float = 0.1,
            mse_weight: float = 0.1,
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

        self.normalize_noise = normalize_noise
        self.preserve_noise_sign = preserve_noise_sign
        # Fix 4: Uniform timestep — all nodes share the same t per forward pass
        self.uniform_timestep = uniform_timestep
        # Fix 2: Noise normalization mode
        # Backward compat: if noise_norm_mode not explicitly set, derive from normalize_noise
        if noise_norm_mode is not None:
            self.noise_norm_mode = noise_norm_mode
        else:
            self.noise_norm_mode = 'layernorm' if normalize_noise else 'global'
        # Fix 3: Zero-mean noise
        self.noise_zero_mean = noise_zero_mean
        # Fix 6: Loss function selection
        self.loss_type = loss_type
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

        # Graph Structure Learning: Sender/Receiver embedding with SVD initialization
        self.structure_learning_mode = init_features is not None
        if self.structure_learning_mode:
            N = init_features.shape[0]
            # Full rank by default: emb_dim=N gives same expressiveness as [N,N] parameter
            self.emb_dim = min(emb_dim if emb_dim is not None else N, N)
            # SVD decomposition of init_features (Patel matrix) for asymmetric init
            U, S, V = torch.svd(init_features.float())
            U_trunc = U[:, :self.emb_dim]        # [N, emb_dim]
            S_trunc = S[:self.emb_dim]            # [emb_dim]
            V_trunc = V[:, :self.emb_dim]         # [N, emb_dim]
            sqrt_S = torch.sqrt(S_trunc).unsqueeze(0)  # [1, emb_dim]
            raw_sender = U_trunc * sqrt_S
            raw_receiver = V_trunc * sqrt_S
            # Rescale so logit std ≈ 1.0 (sigmoid linear region)
            with torch.no_grad():
                logits_est = raw_sender @ raw_receiver.T
                logit_std = logits_est.std()
                scale = (1.0 / (logit_std + 1e-8)).sqrt().item()
            self.node_emb_sender = nn.Parameter(raw_sender * scale)    # [N, emb_dim]
            self.node_emb_receiver = nn.Parameter(raw_receiver * scale)  # [N, emb_dim]
            # Learnable sparsity bias: shifts all logits to encourage sparse output
            if adj_bias_init is not None:
                self.adj_bias = nn.Parameter(torch.tensor(float(adj_bias_init)))
            else:
                self.adj_bias = nn.Parameter(torch.tensor(0.0))
            # Cached adj weights for external loss computation
            self.learned_adj_weights = None
            # Create fully connected DGL graph (N x N edges)
            src = torch.arange(N).repeat_interleave(N)
            dst = torch.arange(N).repeat(N)
            self.register_buffer('full_g_src', src)
            self.register_buffer('full_g_dst', dst)
            self.register_buffer('diag_mask', 1.0 - torch.eye(N))
            self.num_nodes = N

    def get_structure_logits(self):
        """Return the clamped structure logits used everywhere downstream."""
        if not self.structure_learning_mode:
            raise RuntimeError("Structure logits requested when structure learning mode is disabled.")
        adj_logits = self.node_emb_sender @ self.node_emb_receiver.T + self.adj_bias
        return torch.clamp(adj_logits, -6.0, 6.0)

    def get_structure_adj(self):
        """Return the masked adjacency matrix used for graph weights and export."""
        adj_logits = self.get_structure_logits()
        adj_weights = torch.sigmoid(adj_logits)
        adj_weights = adj_weights * self.diag_mask.to(adj_weights.device)
        self.learned_adj_weights = adj_weights
        return adj_weights

    def _get_structure_graph(self, device):
        """Create fully connected graph and compute edge weights from sender/receiver embeddings."""
        g = dgl.graph((self.full_g_src, self.full_g_dst), num_nodes=self.num_nodes)
        g = g.to(device)
        adj_weights = self.get_structure_adj()
        edge_weights = adj_weights.flatten()
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
        # Step 1: Build the clean target x0 for diffusion.
        # When the temporal encoder is enabled, use its unnormalized causal
        # representation instead of the doubly-normalized output.
        x_processed = self.prepare_clean_target(x)

        # Step 2: Sample random timestep for diffusion
        if self.uniform_timestep:
            t_val = torch.randint(self.T, size=(1,), device=x_processed.device)
            t = t_val.expand(x_processed.shape[0])
        else:
            t = torch.randint(self.T, size=(x_processed.shape[0], ), device=x_processed.device)

        # Step 3: Get graph structure
        if self.structure_learning_mode:
            g, edge_weight = self._get_structure_graph(x_processed.device)
        else:
            edge_weight = None

        # Step 4: Diffusion forward process (add noise)
        x_t, time_embed, g = self.sample_q(t, x_processed, g)

        # Step 5: Denoise and compute loss
        loss = self.node_denoising(x_processed, x_t, time_embed, g, edge_weight=edge_weight)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def prepare_clean_target(self, x):
        """
        Build the clean target x0 used by diffusion and denoising.

        With the temporal encoder enabled, we use the encoder's unnormalized
        causal representation so the denoiser learns to recover a meaningful
        temporal state instead of a doubly-normalized proxy.
        """
        if self.use_temporal_encoder:
            _, x_clean = self.temporal_encoder(x, return_unnormalized=True)
            return x_clean
        return x

    def sample_q(self, t, x, g):
        """
        Diffusion forward process with Neighbor-Based Statistical Noise.
        
        For each node i, noise is drawn from N(mu_neighbors, sigma_neighbors)
        where neighbors are defined by self.noise_guide_adj.
        """
        # Determine input shape: [N, Feats] or [Batch, N, Feats]
        is_batched = x.dim() == 3
        noise = self.build_noise(x)

        # Diffusion forward process
        x_t = (
            extract(self.sqrt_alphas_bar, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise
        )
        time_embed = self.time_embedding(t)
        return x_t, time_embed, g

    def build_noise(self, x, eps=None, normalize_noise: Optional[bool] = None,
                    noise_norm_mode: Optional[str] = None,
                    noise_zero_mean: Optional[bool] = None,
                    return_details: bool = False):
        """
        Build guided noise for x.

        Args:
            x: [N, D] or [B, N, D]
            eps: Optional pre-sampled Gaussian noise with the same shape as x
            normalize_noise: Legacy override (True→layernorm, False→skip)
            noise_norm_mode: Override noise normalization ('global', 'layernorm', 'none')
            noise_zero_mean: Override zero-mean noise flag
            return_details: Return intermediate mean/std tensors for diagnostics
        """
        is_batched = x.dim() == 3
        x_work = x if is_batched else x.unsqueeze(0)

        global_mean = x_work.mean(dim=1, keepdim=True)
        global_std = x_work.std(dim=1, keepdim=True) + 1e-6

        if eps is None:
            eps = torch.randn_like(x_work)
        elif eps.shape != x_work.shape:
            raise ValueError(f"eps shape {eps.shape} does not match x shape {x_work.shape}")

        if self.noise_guide_adj is not None:
            adj = self.noise_guide_adj
            base_mean = torch.einsum('ij,bjd->bid', adj, x_work)
            x_sq = x_work ** 2
            base_sq_mean = torch.einsum('ij,bjd->bid', adj, x_sq)
            base_var = base_sq_mean - base_mean ** 2
            base_std = torch.sqrt(torch.clamp(base_var, min=1e-6))
            noise_source = "neighbor"
        else:
            base_mean = global_mean
            base_std = global_std
            noise_source = "global"

        # Fix 3: Zero-mean noise — drop neighbor mean bias to reduce signal correlation
        use_zero_mean = noise_zero_mean if noise_zero_mean is not None else self.noise_zero_mean
        if use_zero_mean:
            raw_noise = eps * base_std
        else:
            raw_noise = eps * base_std + base_mean

        # Fix 2: Noise normalization mode
        # Resolve effective mode: explicit override > legacy override > instance default
        if noise_norm_mode is not None:
            eff_mode = noise_norm_mode
        elif normalize_noise is not None:
            eff_mode = 'layernorm' if normalize_noise else 'none'
        else:
            eff_mode = self.noise_norm_mode

        if eff_mode == 'layernorm':
            raw_noise = F.layer_norm(raw_noise, (raw_noise.shape[-1], ))
        elif eff_mode == 'global':
            noise_std = raw_noise.std()
            raw_noise = raw_noise / (noise_std + 1e-6)
        # 'none': no normalization

        if self.preserve_noise_sign:
            raw_noise = torch.sign(x_work) * torch.abs(raw_noise)

        noise = raw_noise if is_batched else raw_noise.squeeze(0)
        if not return_details:
            return noise

        details = {
            "base_mean": base_mean if is_batched else base_mean.squeeze(0),
            "base_std": base_std if is_batched else base_std.squeeze(0),
            "global_mean": global_mean if is_batched else global_mean.squeeze(0),
            "global_std": global_std if is_batched else global_std.squeeze(0),
            "eps": eps if is_batched else eps.squeeze(0),
            "noise_source": noise_source,
        }
        return noise, details

    def node_denoising(self, x, x_t, time_embed, g, edge_weight=None):
        out, _ = self.net(g, x_t=x_t, time_embed=time_embed, edge_weight=edge_weight)
        if self.loss_type == 'denoise_hybrid':
            loss = loss_fn_denoise_hybrid(out, x, self.alpha_l, self.cosine_weight)
        elif self.loss_type == 'hybrid':
            loss = loss_fn_hybrid(out, x, self.alpha_l, self.mse_weight)
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(out, x)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(out, x)
        else:  # 'cosine' or legacy default
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
        # Embed in the same clean-target space used during training.
        x_processed = self.prepare_clean_target(x)

        t = torch.full((1, ), T, device=x_processed.device)

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


def loss_fn_hybrid(x, y, alpha=2, mse_weight=0.1):
    """Legacy cosine similarity loss + weighted MSE loss."""
    cos_loss = loss_fn(x, y, alpha)
    mse_loss = F.mse_loss(x, y)
    return cos_loss + mse_weight * mse_loss


def loss_fn_denoise_hybrid(x, y, alpha=2, cosine_weight=0.1):
    """
    Default denoising loss: robust reconstruction in the clean-target space,
    with a light cosine term to keep directional similarity stable.
    """
    recon_loss = F.smooth_l1_loss(x, y)
    cos_loss = loss_fn(x, y, alpha)
    return recon_loss + cosine_weight * cos_loss


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
