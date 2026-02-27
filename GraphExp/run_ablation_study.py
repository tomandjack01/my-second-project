#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study: Structure-Aware Noise vs Global Noise

Compares the training performance of DDM with:
1. Structure-Aware Noise (using Patel functional modules)
2. Global Noise (baseline, no module information)

Metrics compared:
- Training loss convergence
- Learned adjacency matrix properties
- Sparsity patterns
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DDM
from utils.patel_util import get_functional_modules

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
CONFIG = {
    'csv_path': '../fMRI_dataset/fMRI.csv',
    'time_points_per_subject': 200,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'lambda_l1': 0.01,
    'num_hidden': 64,
    'num_layers': 2,
    'batch_size': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'top_k_edges': 50,  # For Patel module detection
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fmri_data(config):
    """Load and preprocess fMRI data."""
    df = pd.read_csv(config['csv_path'], header=None)
    data = df.values.astype(np.float32)
    
    total_rows, num_nodes = data.shape
    time_points = config['time_points_per_subject']
    num_subjects = total_rows // time_points
    
    # 2D for Pearson computation
    data_2d = torch.from_numpy(data).float()
    
    # 3D for model input [Num_Subjects, N, TIME_POINTS]
    data_3d = data.reshape(num_subjects, time_points, num_nodes)
    data_3d = np.transpose(data_3d, (0, 2, 1))
    data_3d = torch.from_numpy(data_3d).float()
    
    return data_3d, data_2d, num_subjects, num_nodes


def compute_pearson_matrix(data_2d):
    """Compute Pearson correlation matrix."""
    data_t = data_2d.T
    data_norm = (data_t - data_t.mean(dim=1, keepdim=True)) / (data_t.std(dim=1, keepdim=True) + 1e-8)
    pearson_matrix = data_norm @ data_norm.T / data_norm.shape[1]
    return pearson_matrix


def train_model(data_3d, pearson_matrix, time_points, config, 
                use_structure_aware=True, connected_components=None):
    """
    Train DDM model and return training history.
    
    Args:
        use_structure_aware: If True, use structure-aware noise
        connected_components: Functional modules (only used if use_structure_aware=True)
    
    Returns:
        model: Trained model
        history: Dict with training metrics
    """
    device = config['device']
    num_subjects = data_3d.shape[0]
    data_3d = data_3d.to(device)
    
    # Initialize model
    model = DDM(
        in_dim=time_points,
        num_hidden=config['num_hidden'],
        num_layers=config['num_layers'],
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
        init_features=pearson_matrix,
        connected_components=connected_components if use_structure_aware else None,
    )
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'epoch': [],
        'diff_loss': [],
        'sparsity_loss': [],
        'total_loss': [],
        'adj_mean': [],
        'adj_std': [],
        'sparsity_ratio': [],
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_diff_loss = 0.0
        epoch_sparsity_loss = 0.0
        num_batches = 0
        
        perm = torch.randperm(num_subjects)
        
        for i in range(0, num_subjects, config['batch_size']):
            batch_idx = perm[i:i+config['batch_size']]
            batch_data = data_3d[batch_idx]
            
            for subj_idx in range(batch_data.shape[0]):
                optimizer.zero_grad()
                
                x = batch_data[subj_idx]
                loss, _ = model(g=None, x=x)
                
                adj_sigmoid = torch.sigmoid(model.learned_adj)
                sparsity_loss = config['lambda_l1'] * adj_sigmoid.mean()
                
                total_loss = loss + sparsity_loss
                total_loss.backward()
                optimizer.step()
                
                epoch_diff_loss += loss.item()
                epoch_sparsity_loss += sparsity_loss.item()
                num_batches += 1
        
        # Record metrics
        avg_diff_loss = epoch_diff_loss / num_batches
        avg_sparsity_loss = epoch_sparsity_loss / num_batches
        
        with torch.no_grad():
            adj_sigmoid = torch.sigmoid(model.learned_adj)
            adj_mean = adj_sigmoid.mean().item()
            adj_std = adj_sigmoid.std().item()
            sparsity_ratio = (adj_sigmoid < 0.5).float().mean().item()
        
        history['epoch'].append(epoch + 1)
        history['diff_loss'].append(avg_diff_loss)
        history['sparsity_loss'].append(avg_sparsity_loss)
        history['total_loss'].append(avg_diff_loss + avg_sparsity_loss)
        history['adj_mean'].append(adj_mean)
        history['adj_std'].append(adj_std)
        history['sparsity_ratio'].append(sparsity_ratio)
    
    # Extract final adjacency matrix
    with torch.no_grad():
        adj_matrix = torch.sigmoid(model.learned_adj).cpu().numpy()
    
    return model, history, adj_matrix


def plot_comparison(history_sa, history_baseline, save_path='ablation_comparison.png'):
    """Plot comparison of training metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Ablation Study: Structure-Aware Noise vs Global Noise', fontsize=14, fontweight='bold')
    
    epochs = history_sa['epoch']
    
    # Plot 1: Diffusion Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history_sa['diff_loss'], 'b-', label='Structure-Aware', linewidth=2)
    ax1.plot(epochs, history_baseline['diff_loss'], 'r--', label='Global (Baseline)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Diffusion Loss')
    ax1.set_title('Diffusion Loss Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, history_sa['total_loss'], 'b-', label='Structure-Aware', linewidth=2)
    ax2.plot(epochs, history_baseline['total_loss'], 'r--', label='Global (Baseline)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Total Loss (Diff + Sparsity)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sparsity Ratio
    ax3 = axes[0, 2]
    ax3.plot(epochs, [x * 100 for x in history_sa['sparsity_ratio']], 'b-', label='Structure-Aware', linewidth=2)
    ax3.plot(epochs, [x * 100 for x in history_baseline['sparsity_ratio']], 'r--', label='Global (Baseline)', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Sparsity Ratio (%)')
    ax3.set_title('Adjacency Matrix Sparsity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Adjacency Mean
    ax4 = axes[1, 0]
    ax4.plot(epochs, history_sa['adj_mean'], 'b-', label='Structure-Aware', linewidth=2)
    ax4.plot(epochs, history_baseline['adj_mean'], 'r--', label='Global (Baseline)', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Mean Value')
    ax4.set_title('Adjacency Matrix Mean')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Adjacency Std
    ax5 = axes[1, 1]
    ax5.plot(epochs, history_sa['adj_std'], 'b-', label='Structure-Aware', linewidth=2)
    ax5.plot(epochs, history_baseline['adj_std'], 'r--', label='Global (Baseline)', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Std Value')
    ax5.set_title('Adjacency Matrix Std (Edge Differentiation)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Final metrics comparison (bar chart)
    ax6 = axes[1, 2]
    metrics = ['Final Diff\nLoss', 'Final Adj\nMean', 'Final Adj\nStd', 'Final\nSparsity %']
    sa_values = [
        history_sa['diff_loss'][-1],
        history_sa['adj_mean'][-1],
        history_sa['adj_std'][-1],
        history_sa['sparsity_ratio'][-1] * 100
    ]
    baseline_values = [
        history_baseline['diff_loss'][-1],
        history_baseline['adj_mean'][-1],
        history_baseline['adj_std'][-1],
        history_baseline['sparsity_ratio'][-1] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax6.bar(x - width/2, sa_values, width, label='Structure-Aware', color='blue', alpha=0.7)
    bars2 = ax6.bar(x + width/2, baseline_values, width, label='Global (Baseline)', color='red', alpha=0.7)
    ax6.set_ylabel('Value')
    ax6.set_title('Final Metrics Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax6.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax6.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {save_path}")
    plt.close()


def plot_adjacency_comparison(adj_sa, adj_baseline, save_path='adjacency_comparison.png'):
    """Plot side-by-side adjacency matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Structure-Aware
    im1 = axes[0].imshow(adj_sa, cmap='RdBu_r', vmin=0, vmax=1)
    axes[0].set_title('Structure-Aware Noise', fontweight='bold')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Global (Baseline)
    im2 = axes[1].imshow(adj_baseline, cmap='RdBu_r', vmin=0, vmax=1)
    axes[1].set_title('Global Noise (Baseline)', fontweight='bold')
    axes[1].set_xlabel('Node')
    axes[1].set_ylabel('Node')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Difference
    diff = adj_sa - adj_baseline
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference (SA - Baseline)', fontweight='bold')
    axes[2].set_xlabel('Node')
    axes[2].set_ylabel('Node')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved adjacency comparison to: {save_path}")
    plt.close()


def print_summary(history_sa, history_baseline, adj_sa, adj_baseline):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Structure-Aware':>18} {'Global (Baseline)':>18}")
    print("-" * 70)
    
    # Final losses
    print(f"{'Final Diff Loss':<30} {history_sa['diff_loss'][-1]:>18.4f} {history_baseline['diff_loss'][-1]:>18.4f}")
    print(f"{'Final Total Loss':<30} {history_sa['total_loss'][-1]:>18.4f} {history_baseline['total_loss'][-1]:>18.4f}")
    print(f"{'Final Sparsity Ratio':<30} {history_sa['sparsity_ratio'][-1]*100:>17.2f}% {history_baseline['sparsity_ratio'][-1]*100:>17.2f}%")
    
    # Adjacency stats
    print(f"\n{'Adjacency Matrix Mean':<30} {adj_sa.mean():>18.4f} {adj_baseline.mean():>18.4f}")
    print(f"{'Adjacency Matrix Std':<30} {adj_sa.std():>18.4f} {adj_baseline.std():>18.4f}")
    print(f"{'Adjacency Matrix Min':<30} {adj_sa.min():>18.4f} {adj_baseline.min():>18.4f}")
    print(f"{'Adjacency Matrix Max':<30} {adj_sa.max():>18.4f} {adj_baseline.max():>18.4f}")
    
    # Convergence speed (epoch to reach 90% of final loss)
    final_sa = history_sa['diff_loss'][-1]
    final_baseline = history_baseline['diff_loss'][-1]
    
    thresh_sa = final_sa * 1.1  # Within 10% of final
    thresh_baseline = final_baseline * 1.1
    
    conv_epoch_sa = next((i+1 for i, l in enumerate(history_sa['diff_loss']) if l <= thresh_sa), len(history_sa['diff_loss']))
    conv_epoch_baseline = next((i+1 for i, l in enumerate(history_baseline['diff_loss']) if l <= thresh_baseline), len(history_baseline['diff_loss']))
    
    print(f"\n{'Convergence Epoch (90%)':<30} {conv_epoch_sa:>18} {conv_epoch_baseline:>18}")
    
    # Improvement percentage
    improvement = (history_baseline['diff_loss'][-1] - history_sa['diff_loss'][-1]) / history_baseline['diff_loss'][-1] * 100
    print(f"\n{'Structure-Aware Improvement':<30} {improvement:>17.2f}%")
    
    print("=" * 70)


def main():
    print("=" * 70)
    print("ABLATION STUDY: Structure-Aware Noise vs Global Noise")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(CONFIG['seed'])
    
    # Load data
    print("\n[1/5] Loading fMRI data...")
    data_3d, data_2d, num_subjects, num_nodes = load_fmri_data(CONFIG)
    print(f"      Data shape: {data_3d.shape} [Subjects, Nodes, Time]")
    
    # Compute Pearson matrix
    print("\n[2/5] Computing Pearson correlation matrix...")
    pearson_matrix = compute_pearson_matrix(data_2d)
    
    # Compute functional modules
    print("\n[3/5] Computing Patel functional modules...")
    functional_modules = get_functional_modules(data_2d.numpy(), top_k=CONFIG['top_k_edges'])
    print(f"      Found {len(functional_modules)} modules")
    if functional_modules:
        print(f"      Largest module: {len(functional_modules[0])} nodes")
    
    # Train with Structure-Aware Noise
    print("\n[4/5] Training with STRUCTURE-AWARE NOISE...")
    set_seed(CONFIG['seed'])  # Reset seed for fair comparison
    model_sa, history_sa, adj_sa = train_model(
        data_3d, pearson_matrix, CONFIG['time_points_per_subject'], CONFIG,
        use_structure_aware=True, connected_components=functional_modules
    )
    print(f"      Final Diff Loss: {history_sa['diff_loss'][-1]:.4f}")
    
    # Train with Global Noise (Baseline)
    print("\n[5/5] Training with GLOBAL NOISE (Baseline)...")
    set_seed(CONFIG['seed'])  # Reset seed for fair comparison
    model_baseline, history_baseline, adj_baseline = train_model(
        data_3d, pearson_matrix, CONFIG['time_points_per_subject'], CONFIG,
        use_structure_aware=False, connected_components=None
    )
    print(f"      Final Diff Loss: {history_baseline['diff_loss'][-1]:.4f}")
    
    # Print summary
    print_summary(history_sa, history_baseline, adj_sa, adj_baseline)
    
    # Create results folder with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f'./results/ablation_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    print(f"\nSaving results to: {result_dir}")
    
    # Generate plots
    print("Generating comparison plots...")
    plot_comparison(history_sa, history_baseline, os.path.join(result_dir, 'ablation_comparison.png'))
    plot_adjacency_comparison(adj_sa, adj_baseline, os.path.join(result_dir, 'adjacency_comparison.png'))
    
    results = {
        'history_structure_aware': history_sa,
        'history_baseline': history_baseline,
        'adj_structure_aware': adj_sa,
        'adj_baseline': adj_baseline,
        'config': CONFIG,
    }
    np.save(os.path.join(result_dir, 'ablation_results.npy'), results, allow_pickle=True)
    
    # Also save CSV for easy viewing
    import pandas as pd
    df = pd.DataFrame({
        'Epoch': history_sa['epoch'],
        'SA_DiffLoss': history_sa['diff_loss'],
        'BL_DiffLoss': history_baseline['diff_loss'],
        'SA_Sparsity': history_sa['sparsity_ratio'],
        'BL_Sparsity': history_baseline['sparsity_ratio'],
    })
    df.to_csv(os.path.join(result_dir, 'training_history.csv'), index=False)
    
    print(f"\nSaved files:")
    print(f"  - {result_dir}/ablation_comparison.png")
    print(f"  - {result_dir}/adjacency_comparison.png")
    print(f"  - {result_dir}/ablation_results.npy")
    print(f"  - {result_dir}/training_history.csv")
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
