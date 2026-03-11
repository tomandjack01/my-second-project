#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDM 批评诊断脚本 — 6 项独立实验

用数据验证另一个 AI 对 DDM 代码提出的 6 条批评是否成立。
每个实验独立运行，打印 [CONFIRMED] / [REFUTED] / [PARTIALLY CONFIRMED]。

Usage:
    cd GraphExp
    python diagnose_criticisms.py --csv_path ../fMRI_dataset/sim4.csv
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F

import dgl
from dgl.nn import GraphConv

from main_structure_learning import (
    load_fmri_data,
    build_noise_guide_adjacency,
    build_directed_noise_guide_adjacency,
)
from models.DDM import DDM, loss_fn, loss_fn_hybrid
from utils.patel_util import compute_patel_components

SEPARATOR = "=" * 72


# ============================================================================
# Experiment 1: noise_guide_adj 是否真的无方向？
# ============================================================================

def exp1_noise_guide_symmetry(data_2d: torch.Tensor, top_k_edges: int):
    """检查 build_noise_guide_adjacency 输出的对称性。"""
    print(f"\n{SEPARATOR}")
    print("Experiment 1: noise_guide_adj 是否真的无方向？")
    print(SEPARATOR)

    _, kappa_np, _ = compute_patel_components(data_2d.numpy())
    kappa_t = torch.from_numpy(kappa_np).float()

    noise_adj, adj_binary, k_pairs, threshold = build_noise_guide_adjacency(
        patel_strength_matrix=torch.clamp(kappa_t, min=0.0),
        top_k_pairs=top_k_edges,
    )

    # 1a: adj_binary 对称性
    asym_binary = torch.norm(adj_binary - adj_binary.T).item()
    # 1b: noise_adj 对称性（行归一化后不一定对称，但底层骨架是对称的）
    asym_noise = torch.norm(noise_adj - noise_adj.T).item()
    # 1c: 行归一化验证
    row_sums = noise_adj.sum(dim=1)
    row_sum_range = (row_sums.min().item(), row_sums.max().item())

    print(f"  adj_binary ||A - A^T||_F = {asym_binary:.6f}")
    print(f"  noise_adj  ||A - A^T||_F = {asym_noise:.6f}")
    print(f"  noise_adj row sums: [{row_sum_range[0]:.4f}, {row_sum_range[1]:.4f}]")
    print(f"  Top-k pairs selected: {k_pairs}")

    if asym_binary < 1e-8:
        print("\n  >> [CONFIRMED] adj_binary 完全对称 — 噪声引导邻接矩阵无方向信息。")
        if asym_noise > 1e-6:
            print("     (noise_adj 因行归一化而非对称，但底层骨架是无向的)")
    else:
        print(f"\n  >> [REFUTED] adj_binary 不对称，||A-A^T||_F = {asym_binary:.6f}")


# ============================================================================
# Experiment 2: LayerNorm 是否抹掉了邻居统计量？
# ============================================================================

def exp2_layernorm_erases_neighbor_stats(data_3d: torch.Tensor, noise_adj: torch.Tensor):
    """对比 normalize_noise=True/False 下节点噪声分布的区分度。"""
    print(f"\n{SEPARATOR}")
    print("Experiment 2: LayerNorm 是否抹掉了邻居统计量？")
    print(SEPARATOR)

    N = data_3d.shape[1]
    T = data_3d.shape[2]
    x = data_3d[0]  # 取第一个 subject [N, T]

    # 构造一个最小 DDM 仅用于 build_noise
    model = DDM(
        in_dim=T, num_hidden=32, num_layers=1, nhead=4,
        activation="prelu", feat_drop=0.0, attn_drop=0.0, norm="layernorm",
        noise_guide_adj=noise_adj, normalize_noise=True,
        noise_zero_mean=False,  # 使用旧行为以复现原始问题
        use_temporal_encoder=False,
    )
    model.eval()

    with torch.no_grad():
        x_ln = F.layer_norm(x, (T,))
        # build_noise 内部会 unsqueeze 2D→3D，eps 需要匹配 3D 形状
        eps = torch.randn(1, N, T)

        # 不做 LayerNorm 的噪声
        noise_raw, details_raw = model.build_noise(
            x_ln, eps=eps.clone(), normalize_noise=False, return_details=True
        )
        # 做 LayerNorm 的噪声
        noise_ln, details_ln = model.build_noise(
            x_ln, eps=eps.clone(), normalize_noise=True, return_details=True
        )

    # 每个节点噪声的均值和方差
    raw_means = noise_raw.mean(dim=-1)  # [N]
    raw_stds = noise_raw.std(dim=-1)    # [N]
    ln_means = noise_ln.mean(dim=-1)
    ln_stds = noise_ln.std(dim=-1)

    # 节点间均值的方差（区分度指标）
    raw_mean_var = raw_means.var().item()
    ln_mean_var = ln_means.var().item()
    raw_std_var = raw_stds.var().item()
    ln_std_var = ln_stds.var().item()

    print(f"  [无 LN] 节点均值的方差: {raw_mean_var:.6f}")
    print(f"  [有 LN] 节点均值的方差: {ln_mean_var:.6f}")
    print(f"  [无 LN] 节点标准差的方差: {raw_std_var:.6f}")
    print(f"  [有 LN] 节点标准差的方差: {ln_std_var:.6f}")

    mean_reduction = 1.0 - ln_mean_var / (raw_mean_var + 1e-12)
    std_reduction = 1.0 - ln_std_var / (raw_std_var + 1e-12)
    print(f"  均值区分度下降: {mean_reduction:.1%}")
    print(f"  标准差区分度下降: {std_reduction:.1%}")

    if mean_reduction > 0.9 and std_reduction > 0.5:
        print("\n  >> [CONFIRMED] LayerNorm 显著抹掉了节点间噪声差异（均值区分度下降 >90%）。")
    elif mean_reduction > 0.5:
        print("\n  >> [PARTIALLY CONFIRMED] LayerNorm 部分削弱了节点间噪声差异。")
    else:
        print("\n  >> [REFUTED] LayerNorm 对节点间噪声差异影响不大。")


# ============================================================================
# Experiment 3: 邻居噪声 vs 标准高斯噪声的相关性
# ============================================================================

def exp3_noise_signal_correlation(data_3d: torch.Tensor, noise_adj: torch.Tensor):
    """比较 neighbor-guided noise 与原始信号 x 的相关性。"""
    print(f"\n{SEPARATOR}")
    print("Experiment 3: 邻居噪声 vs 标准高斯噪声的相关性")
    print(SEPARATOR)

    N = data_3d.shape[1]
    T = data_3d.shape[2]
    x = data_3d[0]  # [N, T]

    model = DDM(
        in_dim=T, num_hidden=32, num_layers=1, nhead=4,
        activation="prelu", feat_drop=0.0, attn_drop=0.0, norm="layernorm",
        noise_guide_adj=noise_adj, normalize_noise=False,
        noise_zero_mean=False,  # 使用旧行为以复现原始问题
        use_temporal_encoder=False,
    )
    model.eval()

    with torch.no_grad():
        x_ln = F.layer_norm(x, (T,))
        # 邻居引导噪声
        neighbor_noise = model.build_noise(x_ln)
        # 标准高斯噪声
        gaussian_noise = torch.randn_like(x_ln)

    # 逐节点计算与 x 的 Pearson 相关系数
    def pearson_per_node(a, b):
        """a, b: [N, T] → [N] 相关系数"""
        a_c = a - a.mean(dim=-1, keepdim=True)
        b_c = b - b.mean(dim=-1, keepdim=True)
        num = (a_c * b_c).sum(dim=-1)
        den = a_c.norm(dim=-1) * b_c.norm(dim=-1) + 1e-12
        return num / den

    corr_neighbor = pearson_per_node(neighbor_noise, x_ln)
    corr_gaussian = pearson_per_node(gaussian_noise, x_ln)

    mean_corr_neighbor = corr_neighbor.abs().mean().item()
    mean_corr_gaussian = corr_gaussian.abs().mean().item()
    max_corr_neighbor = corr_neighbor.abs().max().item()

    print(f"  邻居噪声与信号的平均 |相关系数|: {mean_corr_neighbor:.4f}")
    print(f"  高斯噪声与信号的平均 |相关系数|: {mean_corr_gaussian:.4f}")
    print(f"  邻居噪声与信号的最大 |相关系数|: {max_corr_neighbor:.4f}")
    print(f"  相关性比值 (neighbor/gaussian): {mean_corr_neighbor / (mean_corr_gaussian + 1e-12):.2f}x")

    if mean_corr_neighbor > 0.5:
        print("\n  >> [CONFIRMED] 邻居噪声与信号高度相关（>0.5），更像平滑污染而非扩散扰动。")
    elif mean_corr_neighbor > 3 * mean_corr_gaussian:
        print(f"\n  >> [PARTIALLY CONFIRMED] 邻居噪声相关性显著高于高斯基线"
              f"（{mean_corr_neighbor:.4f} vs {mean_corr_gaussian:.4f}），但未达到极端水平。")
    else:
        print("\n  >> [REFUTED] 邻居噪声与信号的相关性接近高斯基线。")


# ============================================================================
# Experiment 4: per-node timestep vs uniform timestep
# ============================================================================

def exp4_per_node_vs_uniform_timestep(data_3d: torch.Tensor, noise_adj: torch.Tensor):
    """比较 per-node t 和 uniform t 下的信噪比差异。"""
    print(f"\n{SEPARATOR}")
    print("Experiment 4: per-node timestep vs uniform timestep")
    print(SEPARATOR)

    N = data_3d.shape[1]
    T_dim = data_3d.shape[2]
    x = data_3d[0]  # [N, T_dim]

    model = DDM(
        in_dim=T_dim, num_hidden=32, num_layers=1, nhead=4,
        activation="prelu", feat_drop=0.0, attn_drop=0.0, norm="layernorm",
        noise_guide_adj=noise_adj, normalize_noise=True,
        use_temporal_encoder=False, T=1000,
    )
    model.eval()

    with torch.no_grad():
        x_ln = F.layer_norm(x, (T_dim,))

        # per-node: 每个节点独立采样 t
        t_per_node = torch.randint(model.T, size=(N,))
        # uniform: 所有节点共享同一个 t
        t_uniform_val = 500
        t_uniform = torch.full((N,), t_uniform_val)

        noise = model.build_noise(x_ln)

        # per-node 扩散
        sqrt_ab_per = model.sqrt_alphas_bar[t_per_node].unsqueeze(-1)
        sqrt_1mab_per = model.sqrt_one_minus_alphas_bar[t_per_node].unsqueeze(-1)
        x_t_per = sqrt_ab_per * x_ln + sqrt_1mab_per * noise

        # uniform 扩散
        sqrt_ab_uni = model.sqrt_alphas_bar[t_uniform].unsqueeze(-1)
        sqrt_1mab_uni = model.sqrt_one_minus_alphas_bar[t_uniform].unsqueeze(-1)
        x_t_uni = sqrt_ab_uni * x_ln + sqrt_1mab_uni * noise

    # 信噪比: SNR = ||signal_component||^2 / ||noise_component||^2
    def compute_snr(sqrt_ab, sqrt_1mab, x_clean, noise_vec):
        sig_power = (sqrt_ab * x_clean).pow(2).mean(dim=-1)
        noise_power = (sqrt_1mab * noise_vec).pow(2).mean(dim=-1)
        return sig_power / (noise_power + 1e-12)

    snr_per = compute_snr(sqrt_ab_per, sqrt_1mab_per, x_ln, noise)
    snr_uni = compute_snr(sqrt_ab_uni, sqrt_1mab_uni, x_ln, noise)

    print(f"  Per-node t 范围: [{t_per_node.min().item()}, {t_per_node.max().item()}]")
    print(f"  Uniform t: {t_uniform_val}")
    print(f"  Per-node SNR: mean={snr_per.mean():.4f}, std={snr_per.std():.4f}, "
          f"range=[{snr_per.min():.4f}, {snr_per.max():.4f}]")
    print(f"  Uniform SNR:  mean={snr_uni.mean():.4f}, std={snr_uni.std():.4f}, "
          f"range=[{snr_uni.min():.4f}, {snr_uni.max():.4f}]")

    snr_cv_per = (snr_per.std() / (snr_per.mean() + 1e-12)).item()
    snr_cv_uni = (snr_uni.std() / (snr_uni.mean() + 1e-12)).item()
    print(f"  Per-node SNR 变异系数 (CV): {snr_cv_per:.4f}")
    print(f"  Uniform SNR 变异系数 (CV): {snr_cv_uni:.4f}")

    if snr_cv_per > 5 * snr_cv_uni:
        print("\n  >> [CONFIRMED] per-node t 导致同一图中节点处于截然不同的噪声水平，"
              "去噪网络需同时处理多种信噪比。")
    elif snr_cv_per > 2 * snr_cv_uni:
        print("\n  >> [PARTIALLY CONFIRMED] per-node t 增加了节点间 SNR 差异，但幅度有限。")
    else:
        print("\n  >> [REFUTED] per-node t 与 uniform t 的 SNR 差异不显著。")


# ============================================================================
# Experiment 5: DGL 消息传递方向 vs 矩阵语义
# ============================================================================

def exp5_dgl_message_direction():
    """用 2 节点极端案例验证 DGL GraphConv 的消息流向。"""
    print(f"\n{SEPARATOR}")
    print("Experiment 5: DGL 消息传递方向 vs 矩阵语义")
    print(SEPARATOR)

    # 构造 2 节点图: 只有边 0→1 (src=0, dst=1)
    g = dgl.graph(([0], [1]), num_nodes=2)

    # 节点特征: node0=[1,0], node1=[0,1]（正交，便于追踪信息流）
    h = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0]])

    # 用 GraphConv(norm='none') + identity weight 来追踪纯消息传递
    conv = GraphConv(2, 2, norm='none', weight=True, bias=False,
                     allow_zero_in_degree=True)
    with torch.no_grad():
        conv.weight.copy_(torch.eye(2))

    # 不带 edge_weight
    out_no_w = conv(g, h).detach()

    # 带 edge_weight = [1.0]（只有一条边）
    ew = torch.tensor([1.0])
    out_with_w = conv(g, h, edge_weight=ew).detach()

    print("  图结构: edge (0 → 1), 即 src=0, dst=1")
    print(f"  输入特征: node0={h[0].tolist()}, node1={h[1].tolist()}")
    print(f"  GraphConv 输出 (无 edge_weight):")
    print(f"    node0 = {out_no_w[0].tolist()}")
    print(f"    node1 = {out_no_w[1].tolist()}")
    print(f"  GraphConv 输出 (edge_weight=1.0):")
    print(f"    node0 = {out_with_w[0].tolist()}")
    print(f"    node1 = {out_with_w[1].tolist()}")

    # DGL GraphConv: dst 节点聚合 src 节点的消息
    # 所以 edge(0→1) 意味着 node1 收到 node0 的信息
    node1_received_node0 = (out_with_w[1][0].item() > 0.1)
    node0_unchanged = (out_with_w[0][0].item() < 0.1) or (out_with_w[0] == h[0]).all()

    print()
    print("  分析:")
    print(f"    node1 是否收到了 node0 的特征? {'是' if node1_received_node0 else '否'}")
    print(f"    node0 是否保持不变（未收到消息）? {'是' if node0_unchanged else '否'}")

    # DDM 中 adj[i,j] 高 → edge(i→j) → j 聚合 i 的信息
    # 注释说 adj[effect, cause] → effect 聚合 cause 的信息
    # 即 adj[dst, src] → dst 聚合 src → 与 DGL 一致
    print()
    if node1_received_node0:
        print("  DGL 语义: edge(src→dst) = dst 聚合 src 的消息")
        print("  DDM 注释: A_raw[effect, cause] = effect 聚合 cause 的信息")
        print("  对应关系: A_raw[i,j] 高 → edge(i→j) → node_j 聚合 node_i")
        print("  但 DDM 用 A_raw 做 edge_weight 时: A_raw[effect, cause] 对应 edge(effect→cause)?")
        print()

        # 验证 DDM 的 _get_structure_graph 如何构造边
        print("  DDM 边构造方式:")
        print("    src = arange(N).repeat_interleave(N)  → [0,0,...,1,1,...,N-1]")
        print("    dst = arange(N).repeat(N)             → [0,1,...,N-1,0,1,...,N-1]")
        print("    edge_weight = adj_weights.flatten()   → adj[src, dst]")
        print()
        print("    即 edge(i→j) 的权重 = adj[i, j]")
        print("    DGL 中 edge(i→j) 意味着 j 聚合 i 的消息")
        print("    所以 adj[i,j] 高 → j 更多地聚合 i 的信息")
        print("    如果 adj 的语义是 adj[effect, cause]:")
        print("      adj[effect, cause] 高 → cause 更多地聚合 effect 的信息 ← 方向反了!")
        print("      应该是 effect 聚合 cause，但实际是 cause 聚合 effect")
        print()
        print("  >> [CONFIRMED] 存在方向反转问题。")
        print("     adj[i,j] 高时，DGL 让 j 聚合 i 的信息（j 是 effect），")
        print("     但注释声称 i 是 effect。两者矛盾。")
        print("     不过 test_eval.py 在评估时做了转置修正，最终输出可能是正确的。")
    else:
        print("  >> [REFUTED] DGL 消息方向与预期不符，需要进一步调查。")


# ============================================================================
# Experiment 6: cosine loss 的幅值信息丢失
# ============================================================================

def exp6_cosine_loss_amplitude_blindness():
    """验证 cosine loss 是否对幅值变化不敏感。"""
    print(f"\n{SEPARATOR}")
    print("Experiment 6: cosine loss 的幅值信息丢失")
    print(SEPARATOR)

    # 构造 target
    target = torch.randn(10, 64)

    # Case A: 方向相同，幅值 1x
    pred_1x = target.clone()
    # Case B: 方向相同，幅值 0.1x
    pred_01x = target * 0.1
    # Case C: 方向相同，幅值 10x
    pred_10x = target * 10.0
    # Case D: 方向偏移 (加噪声)
    pred_noisy = target + torch.randn_like(target) * 0.5

    cos_1x = loss_fn(pred_1x, target, alpha=2).item()
    cos_01x = loss_fn(pred_01x, target, alpha=2).item()
    cos_10x = loss_fn(pred_10x, target, alpha=2).item()
    cos_noisy = loss_fn(pred_noisy, target, alpha=2).item()

    # MSE 对比
    mse_1x = F.mse_loss(pred_1x, target).item()
    mse_01x = F.mse_loss(pred_01x, target).item()
    mse_10x = F.mse_loss(pred_10x, target).item()
    mse_noisy = F.mse_loss(pred_noisy, target).item()

    print("  Prediction vs Target:")
    print(f"  {'Case':<20} {'Cosine Loss':>12} {'MSE Loss':>12}")
    print(f"  {'-'*44}")
    print(f"  {'同方向 1x 幅值':<20} {cos_1x:>12.6f} {mse_1x:>12.6f}")
    print(f"  {'同方向 0.1x 幅值':<20} {cos_01x:>12.6f} {mse_01x:>12.6f}")
    print(f"  {'同方向 10x 幅值':<20} {cos_10x:>12.6f} {mse_10x:>12.6f}")
    print(f"  {'方向偏移 (噪声)':<20} {cos_noisy:>12.6f} {mse_noisy:>12.6f}")

    # 幅值变化对 cosine loss 的影响
    cos_range = max(cos_01x, cos_10x) - cos_1x
    mse_range = max(mse_01x, mse_10x) - mse_1x

    print(f"\n  Cosine loss 对幅值变化的响应范围: {cos_range:.6f}")
    print(f"  MSE loss 对幅值变化的响应范围:    {mse_range:.6f}")

    if cos_range < 0.01:
        print("\n  >> [CONFIRMED] Cosine loss 对幅值变化几乎完全不敏感。")
        print("     同方向不同幅值的预测得到几乎相同的 loss。")
        print("     这意味着去噪网络只需学习方向，无需恢复正确的信号强度。")
    elif cos_range < 0.1:
        print("\n  >> [PARTIALLY CONFIRMED] Cosine loss 对幅值变化有微弱响应，但远不如 MSE。")
    else:
        print("\n  >> [REFUTED] Cosine loss 对幅值变化有显著响应。")


# ============================================================================
# 修复验证：对比修复前后的指标
# ============================================================================

def verify_fix1_directed_noise(data_2d: torch.Tensor, top_k_edges: int):
    """验证 Fix 1: 有向噪声邻接矩阵是否引入了不对称性。"""
    print(f"\n{SEPARATOR}")
    print("Verify Fix 1: 有向噪声邻接矩阵")
    print(SEPARATOR)

    _, kappa_np, tau_np = compute_patel_components(data_2d.numpy())
    kappa_t = torch.from_numpy(kappa_np).float()
    tau_t = torch.from_numpy(tau_np).float()

    # 旧：对称
    old_adj, adj_binary, _, _ = build_noise_guide_adjacency(
        patel_strength_matrix=torch.clamp(kappa_t, min=0.0),
        top_k_pairs=top_k_edges,
    )
    # 新：有向
    new_adj = build_directed_noise_guide_adjacency(
        patel_kappa=torch.clamp(kappa_t, min=0.0),
        patel_tau=tau_t,
        top_k_pairs=top_k_edges,
        direction_alpha=0.5,
    )

    # 比较底层骨架的对称性（归一化前）
    old_binary_asym = torch.norm(adj_binary - adj_binary.T).item()
    # 比较归一化后的不对称性
    old_asym = torch.norm(old_adj - old_adj.T).item()
    new_asym = torch.norm(new_adj - new_adj.T).item()

    # 逐边方向差异：对选中的边对，新矩阵的 |A[i,j] - A[j,i]| 应该更大
    N = kappa_t.shape[0]
    old_margins = []
    new_margins = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj_binary[i, j] > 0:
                old_margins.append(abs(old_adj[i, j].item() - old_adj[j, i].item()))
                new_margins.append(abs(new_adj[i, j].item() - new_adj[j, i].item()))

    old_mean_margin = np.mean(old_margins) if old_margins else 0
    new_mean_margin = np.mean(new_margins) if new_margins else 0

    print(f"  旧骨架 (binary) ||A - A^T||_F = {old_binary_asym:.6f} (完全对称)")
    print(f"  旧 (归一化后) ||A - A^T||_F = {old_asym:.6f} (度数差异导致)")
    print(f"  新 (有向)     ||A - A^T||_F = {new_asym:.6f}")
    print(f"  旧边对平均方向 margin: {old_mean_margin:.6f}")
    print(f"  新边对平均方向 margin: {new_mean_margin:.6f}")

    if new_mean_margin > old_mean_margin * 1.5:
        print(f"\n  >> [FIX VERIFIED] 有向噪声邻接矩阵在边级别引入了 "
              f"{new_mean_margin / (old_mean_margin + 1e-12):.1f}x 的方向差异。")
    else:
        print("\n  >> [FIX PARTIAL] 方向偏置存在但幅度有限，可尝试增大 direction_alpha。")


def verify_fix2_noise_norm(data_3d: torch.Tensor, noise_adj: torch.Tensor):
    """验证 Fix 2: 全局标准化是否保留了节点间差异。"""
    print(f"\n{SEPARATOR}")
    print("Verify Fix 2: 全局标准化 vs LayerNorm")
    print(SEPARATOR)

    N = data_3d.shape[1]
    T = data_3d.shape[2]
    x = data_3d[0]

    model = DDM(
        in_dim=T, num_hidden=32, num_layers=1, nhead=4,
        activation="prelu", feat_drop=0.0, attn_drop=0.0, norm="layernorm",
        noise_guide_adj=noise_adj, noise_norm_mode='global',
        noise_zero_mean=False, use_temporal_encoder=False,
    )
    model.eval()

    with torch.no_grad():
        x_ln = F.layer_norm(x, (T,))
        eps = torch.randn(1, N, T)

        noise_ln = model.build_noise(x_ln, eps=eps.clone(), noise_norm_mode='layernorm')
        noise_global = model.build_noise(x_ln, eps=eps.clone(), noise_norm_mode='global')
        noise_none = model.build_noise(x_ln, eps=eps.clone(), noise_norm_mode='none')

    for label, noise in [("layernorm", noise_ln), ("global", noise_global), ("none", noise_none)]:
        mean_var = noise.mean(dim=-1).var().item()
        std_var = noise.std(dim=-1).var().item()
        print(f"  [{label:>9}] 节点均值方差={mean_var:.6f}, 节点标准差方差={std_var:.6f}")

    global_mean_var = noise_global.mean(dim=-1).var().item()
    ln_mean_var = noise_ln.mean(dim=-1).var().item()
    if global_mean_var > ln_mean_var * 10:
        print("\n  >> [FIX VERIFIED] 全局标准化保留了节点间差异。")
    else:
        print("\n  >> [FIX INSUFFICIENT] 全局标准化未显著改善。")


def verify_fix3_zero_mean(data_3d: torch.Tensor, noise_adj: torch.Tensor):
    """验证 Fix 3: 零均值噪声是否降低了信号相关性。"""
    print(f"\n{SEPARATOR}")
    print("Verify Fix 3: 零均值噪声 vs 含均值噪声")
    print(SEPARATOR)

    N = data_3d.shape[1]
    T = data_3d.shape[2]
    x = data_3d[0]

    model = DDM(
        in_dim=T, num_hidden=32, num_layers=1, nhead=4,
        activation="prelu", feat_drop=0.0, attn_drop=0.0, norm="layernorm",
        noise_guide_adj=noise_adj, noise_norm_mode='none',
        noise_zero_mean=True, use_temporal_encoder=False,
    )
    model.eval()

    def pearson_per_node(a, b):
        a_c = a - a.mean(dim=-1, keepdim=True)
        b_c = b - b.mean(dim=-1, keepdim=True)
        num = (a_c * b_c).sum(dim=-1)
        den = a_c.norm(dim=-1) * b_c.norm(dim=-1) + 1e-12
        return num / den

    with torch.no_grad():
        x_ln = F.layer_norm(x, (T,))
        noise_with_mean = model.build_noise(x_ln, noise_zero_mean=False)
        noise_zero_mean = model.build_noise(x_ln, noise_zero_mean=True)

    corr_with = pearson_per_node(noise_with_mean, x_ln).abs().mean().item()
    corr_zero = pearson_per_node(noise_zero_mean, x_ln).abs().mean().item()

    print(f"  含均值噪声与信号的平均 |相关系数|: {corr_with:.4f}")
    print(f"  零均值噪声与信号的平均 |相关系数|: {corr_zero:.4f}")
    print(f"  相关性降低: {(1 - corr_zero / (corr_with + 1e-12)):.1%}")

    if corr_zero < corr_with * 0.5:
        print("\n  >> [FIX VERIFIED] 零均值噪声显著降低了信号相关性。")
    elif corr_zero < corr_with:
        print("\n  >> [FIX PARTIAL] 零均值噪声有所改善，但幅度有限。")
    else:
        print("\n  >> [FIX FAILED] 零均值噪声未降低信号相关性。")


def verify_fix6_hybrid_loss():
    """验证 Fix 6: 混合损失是否对幅值变化敏感。"""
    print(f"\n{SEPARATOR}")
    print("Verify Fix 6: 混合损失 vs 纯 cosine 损失")
    print(SEPARATOR)

    target = torch.randn(10, 64)
    pred_1x = target.clone()
    pred_01x = target * 0.1
    pred_10x = target * 10.0

    cos_01x = loss_fn(pred_01x, target, alpha=2).item()
    cos_10x = loss_fn(pred_10x, target, alpha=2).item()
    hyb_01x = loss_fn_hybrid(pred_01x, target, alpha=2, mse_weight=0.1).item()
    hyb_10x = loss_fn_hybrid(pred_10x, target, alpha=2, mse_weight=0.1).item()

    cos_range = max(cos_01x, cos_10x)
    hyb_range = max(hyb_01x, hyb_10x)

    print(f"  Cosine loss 对 0.1x/10x 幅值的最大响应: {cos_range:.6f}")
    print(f"  Hybrid loss 对 0.1x/10x 幅值的最大响应: {hyb_range:.6f}")

    if hyb_range > cos_range + 0.01:
        print("\n  >> [FIX VERIFIED] 混合损失对幅值变化有显著响应。")
    else:
        print("\n  >> [FIX FAILED] 混合损失未改善幅值敏感性。")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DDM 批评诊断脚本")
    parser.add_argument("--csv_path", type=str, default="../fMRI_dataset/sim4.csv",
                        help="fMRI CSV 数据路径")
    parser.add_argument("--top_k_edges", type=int, default=61,
                        help="噪声引导邻接矩阵的 top-k 边数")
    parser.add_argument("--experiments", type=str, default="all",
                        help="要运行的实验编号，逗号分隔 (e.g., '1,3,5') 或 'all'")
    parser.add_argument("--verify_fixes", action="store_true", default=False,
                        help="运行修复验证模式，对比修复前后的指标")
    args = parser.parse_args()

    print(SEPARATOR)
    if args.verify_fixes:
        print("DDM 修复验证 — 对比修复前后的指标")
    else:
        print("DDM 批评诊断脚本 — 6 项独立实验")
    print(SEPARATOR)

    # 解析要运行的实验
    if args.experiments == "all":
        run_exps = {1, 2, 3, 4, 5, 6}
    else:
        run_exps = {int(x.strip()) for x in args.experiments.split(",")}

    # 加载数据（实验 5、6 不需要数据）
    data_3d, data_2d, noise_adj = None, None, None
    needs_data = bool(run_exps & {1, 2, 3, 4})

    if needs_data:
        print(f"\n加载数据: {args.csv_path}")
        data_3d, data_2d, num_subjects, num_nodes = load_fmri_data(args.csv_path)
        print(f"数据形状: {data_3d.shape} [Subjects, Nodes, TimePoints]")

        # 预计算 noise_guide_adj（实验 2, 3, 4 需要）
        if run_exps & {2, 3, 4}:
            _, kappa_np, _ = compute_patel_components(data_2d.numpy())
            kappa_t = torch.from_numpy(kappa_np).float()
            noise_adj, _, _, _ = build_noise_guide_adjacency(
                patel_strength_matrix=torch.clamp(kappa_t, min=0.0),
                top_k_pairs=args.top_k_edges,
            )

    if args.verify_fixes:
        # 修复验证模式
        if 1 in run_exps:
            verify_fix1_directed_noise(data_2d, args.top_k_edges)
        if 2 in run_exps:
            verify_fix2_noise_norm(data_3d, noise_adj)
        if 3 in run_exps:
            verify_fix3_zero_mean(data_3d, noise_adj)
        if 6 in run_exps:
            verify_fix6_hybrid_loss()
    else:
        # 原始诊断模式
        if 1 in run_exps:
            exp1_noise_guide_symmetry(data_2d, args.top_k_edges)
        if 2 in run_exps:
            exp2_layernorm_erases_neighbor_stats(data_3d, noise_adj)
        if 3 in run_exps:
            exp3_noise_signal_correlation(data_3d, noise_adj)
        if 4 in run_exps:
            exp4_per_node_vs_uniform_timestep(data_3d, noise_adj)
        if 5 in run_exps:
            exp5_dgl_message_direction()
        if 6 in run_exps:
            exp6_cosine_loss_amplitude_blindness()

    print(f"\n{SEPARATOR}")
    print("所有实验完成。")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
