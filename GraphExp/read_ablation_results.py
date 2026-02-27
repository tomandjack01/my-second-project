#!/usr/bin/env python3
"""Read and export ablation results to readable formats."""

import numpy as np
import pandas as pd
import json

# Load results
results = np.load('ablation_results.npy', allow_pickle=True).item()

print('=== ABLATION RESULTS ===')
print()

# History comparison
h_sa = results['history_structure_aware']
h_bl = results['history_baseline']

print('Final Metrics:')
sa_loss = h_sa['diff_loss'][-1]
bl_loss = h_bl['diff_loss'][-1]
sa_sparsity = h_sa['sparsity_ratio'][-1] * 100
bl_sparsity = h_bl['sparsity_ratio'][-1] * 100

print(f'  Structure-Aware Diff Loss: {sa_loss:.4f}')
print(f'  Baseline Diff Loss:        {bl_loss:.4f}')
print(f'  Structure-Aware Sparsity:  {sa_sparsity:.1f}%')
print(f'  Baseline Sparsity:         {bl_sparsity:.1f}%')
print()

# Adjacency stats
adj_sa = results['adj_structure_aware']
adj_bl = results['adj_baseline']

print('Adjacency Matrix Stats:')
print(f'  SA  - Mean: {adj_sa.mean():.4f}, Std: {adj_sa.std():.4f}')
print(f'  BL  - Mean: {adj_bl.mean():.4f}, Std: {adj_bl.std():.4f}')
print()

# Save as CSV for easy viewing
df = pd.DataFrame({
    'Epoch': h_sa['epoch'],
    'SA_DiffLoss': h_sa['diff_loss'],
    'BL_DiffLoss': h_bl['diff_loss'],
    'SA_TotalLoss': h_sa['total_loss'],
    'BL_TotalLoss': h_bl['total_loss'],
    'SA_Sparsity': h_sa['sparsity_ratio'],
    'BL_Sparsity': h_bl['sparsity_ratio'],
    'SA_AdjMean': h_sa['adj_mean'],
    'BL_AdjMean': h_bl['adj_mean'],
    'SA_AdjStd': h_sa['adj_std'],
    'BL_AdjStd': h_bl['adj_std'],
})
df.to_csv('ablation_results.csv', index=False)
print('Saved training history to: ablation_results.csv')

# Save summary as JSON
summary = {
    'final_metrics': {
        'structure_aware': {
            'diff_loss': float(sa_loss),
            'sparsity_percent': float(sa_sparsity),
            'adj_mean': float(adj_sa.mean()),
            'adj_std': float(adj_sa.std()),
        },
        'baseline': {
            'diff_loss': float(bl_loss),
            'sparsity_percent': float(bl_sparsity),
            'adj_mean': float(adj_bl.mean()),
            'adj_std': float(adj_bl.std()),
        },
    },
    'improvement': {
        'diff_loss_reduction_percent': float((bl_loss - sa_loss) / bl_loss * 100),
    },
    'adjacency_matrices': {
        'structure_aware': adj_sa.tolist(),
        'baseline': adj_bl.tolist(),
    }
}

with open('ablation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Saved summary to: ablation_summary.json')

print()
print('=== IMPROVEMENT ===')
improvement = (bl_loss - sa_loss) / bl_loss * 100
print(f'Structure-Aware reduces Diff Loss by {improvement:.1f}%')
