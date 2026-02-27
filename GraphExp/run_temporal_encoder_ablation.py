#!/usr/bin/env python3
# -*- coding: u     tf-8 -*-
"""
Minimal ablation for temporal encoder contribution in structure learning.

Compares three settings:
1) full_encoder: default causal dilated temporal encoder
2) no_temporal_stack: remove temporal residual stack (num_blocks=0)
3) reduced_receptive_field: shallower temporal stack (num_blocks=2)

Outputs:
- raw_runs.csv: per-seed metrics per setting
- summary.csv: mean/std metrics per setting
- meta.json: experiment configuration
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from main_structure_learning import (
    TIME_POINTS_PER_SUBJECT,
    compute_global_pearson,
    load_fmri_data,
    set_seed,
    train_brain_connectivity,
)
from utils.patel_util import compute_patel_matrix


def parse_seeds(seeds_text: str) -> List[int]:
    seeds = [int(item.strip()) for item in seeds_text.split(',') if item.strip()]
    if not seeds:
        raise ValueError('`--seeds` 至少需要一个整数种子，例如 "42" 或 "42,43,44"')
    return seeds


def build_noise_guide_adj(data_2d: torch.Tensor, num_nodes: int, top_k_edges: int) -> torch.Tensor:
    """Build row-normalized adjacency used for neighbor-based noise guidance."""
    patel_matrix = compute_patel_matrix(data_2d.numpy())
    patel_matrix = torch.from_numpy(patel_matrix).float()

    mask_off_diag = 1.0 - torch.eye(num_nodes)
    patel_off_diag = patel_matrix * mask_off_diag
    flat_values = patel_off_diag.flatten()
    flat_values = flat_values[flat_values > 0]

    total_edges = num_nodes * (num_nodes - 1)
    k_edges = min(top_k_edges, total_edges)

    if len(flat_values) >= k_edges and k_edges > 0:
        threshold = torch.topk(flat_values, k_edges).values[-1]
    else:
        threshold = 0.0

    adj_binary = (patel_matrix >= threshold).float() * mask_off_diag
    adj_with_self = adj_binary + torch.eye(num_nodes)
    degree = adj_with_self.sum(dim=1, keepdim=True)
    noise_guide_adj = adj_with_self / (degree + 1e-9)
    return noise_guide_adj


def compute_metrics(adj_matrix: np.ndarray, loss_history: List[float]) -> Dict[str, float]:
    """Compute minimal quantitative metrics for ablation comparison."""
    adj = np.asarray(adj_matrix)
    loss_arr = np.asarray(loss_history, dtype=np.float64)

    off_diag_mask = ~np.eye(adj.shape[0], dtype=bool)
    off_diag_values = adj[off_diag_mask]

    final_loss = float(loss_arr[-1])
    mean_last5_loss = float(loss_arr[-min(5, len(loss_arr)):].mean())
    normalized_auc = float(np.trapz(loss_arr) / max(len(loss_arr), 1))

    sparsity_ratio = float((off_diag_values < 0.5).mean())
    off_diag_mean = float(off_diag_values.mean())
    asymmetry = float(np.abs(adj - adj.T)[off_diag_mask].mean())

    return {
        'final_loss': final_loss,
        'mean_last5_loss': mean_last5_loss,
        'normalized_auc': normalized_auc,
        'offdiag_mean': off_diag_mean,
        'offdiag_sparsity_ratio': sparsity_ratio,
        'adj_asymmetry': asymmetry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Temporal encoder minimal ablation for DDM structure learning')
    parser.add_argument('--csv_path', type=str, default='../fMRI_dataset/fMRI.csv', help='Path to fMRI CSV')
    parser.add_argument('--time_points', type=int, default=TIME_POINTS_PER_SUBJECT, help='Time points per subject')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs per run')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=0.1, help='L1 coefficient for adjacency sparsity')
    parser.add_argument('--num_hidden', type=int, default=64, help='Hidden feature dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Denoising graph layers')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size in subjects')
    parser.add_argument('--top_k_edges', type=int, default=50, help='Top-k edges for noise guide adjacency')
    parser.add_argument('--seeds', type=str, default='42,43,44', help='Comma-separated random seeds')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for training')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, fallback to CPU')
        args.device = 'cpu'

    seeds = parse_seeds(args.seeds)

    print('=' * 70)
    print('Temporal Encoder Minimal Ablation')
    print('=' * 70)
    print(f'Device: {args.device}')
    print(f'Seeds: {seeds}')
    print(f'Epochs per run: {args.epochs}')

    data_3d, data_2d, num_subjects, num_nodes = load_fmri_data(
        csv_path=args.csv_path,
        time_points_per_subject=args.time_points,
    )
    pearson_matrix = compute_global_pearson(data_2d)
    noise_guide_adj = build_noise_guide_adj(data_2d, num_nodes=num_nodes, top_k_edges=args.top_k_edges)

    variants = [
        ('full_encoder', {}),
        ('no_temporal_stack', {'temporal_num_blocks': 0}),
        ('reduced_receptive_field', {'temporal_num_blocks': 2}),
    ]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(os.path.dirname(__file__), 'results', f'temporal_ablation_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)

    raw_records: List[Dict[str, float]] = []

    for variant_name, ddm_kwargs in variants:
        print('\n' + '-' * 70)
        print(f'Variant: {variant_name} | ddm_kwargs={ddm_kwargs}')
        print('-' * 70)

        for seed in seeds:
            set_seed(seed)
            run_start = time.time()

            _, adj_matrix, loss_history = train_brain_connectivity(
                data_3d=data_3d,
                pearson_matrix=pearson_matrix,
                num_nodes=num_nodes,
                time_points=args.time_points,
                noise_guide_adj=noise_guide_adj,
                num_epochs=args.epochs,
                learning_rate=args.lr,
                lambda_l1=args.lambda_l1,
                device=args.device,
                log_interval=args.log_interval,
                num_hidden=args.num_hidden,
                num_layers=args.num_layers,
                batch_size=args.batch_size,
                debug_checks=False,
                ddm_kwargs=ddm_kwargs,
            )

            metrics = compute_metrics(adj_matrix=adj_matrix, loss_history=loss_history)
            runtime_sec = float(time.time() - run_start)

            record = {
                'variant': variant_name,
                'seed': int(seed),
                'runtime_sec': runtime_sec,
                **metrics,
            }
            raw_records.append(record)

            print(
                f"seed={seed} | final_loss={record['final_loss']:.4f} "
                f"| mean_last5={record['mean_last5_loss']:.4f} "
                f"| auc={record['normalized_auc']:.4f} "
                f"| asym={record['adj_asymmetry']:.4f} "
                f"| runtime={runtime_sec:.1f}s"
            )

    raw_df = pd.DataFrame(raw_records)
    summary_df = raw_df.groupby('variant', as_index=False).agg(
        final_loss_mean=('final_loss', 'mean'),
        final_loss_std=('final_loss', 'std'),
        mean_last5_loss_mean=('mean_last5_loss', 'mean'),
        mean_last5_loss_std=('mean_last5_loss', 'std'),
        normalized_auc_mean=('normalized_auc', 'mean'),
        normalized_auc_std=('normalized_auc', 'std'),
        offdiag_mean_mean=('offdiag_mean', 'mean'),
        offdiag_sparsity_ratio_mean=('offdiag_sparsity_ratio', 'mean'),
        adj_asymmetry_mean=('adj_asymmetry', 'mean'),
        runtime_sec_mean=('runtime_sec', 'mean'),
    )

    raw_csv = os.path.join(result_dir, 'raw_runs.csv')
    summary_csv = os.path.join(result_dir, 'summary.csv')
    meta_json = os.path.join(result_dir, 'meta.json')

    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(meta_json, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'args': vars(args),
                'seeds': seeds,
                'variants': [{name: kwargs} for name, kwargs in variants],
                'num_subjects': int(num_subjects),
                'num_nodes': int(num_nodes),
                'result_dir': result_dir,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print('\n' + '=' * 70)
    print('Ablation finished')
    print(f'raw runs: {raw_csv}')
    print(f'summary : {summary_csv}')
    print(f'meta    : {meta_json}')
    print('=' * 70)


if __name__ == '__main__':
    main()
