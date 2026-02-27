#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patel's Kappa/Tau Connectivity Algorithm - Python Implementation

This module implements Patel's method for computing functional connectivity
from fMRI time-series data and identifying functional brain modules.

Reference: Patel, R. S., et al. (2006). "A Bayesian approach to determining 
connectivity of the human brain."

Author: Ported from MATLAB (Pate.m, patel2.m)
"""

import numpy as np
import networkx as nx
from typing import List, Optional, Tuple
import warnings


def _preprocess_timeseries(data: np.ndarray) -> np.ndarray:
    """
    Preprocess time-series data according to Patel's method.
    
    Steps:
        1. Compute 10th and 90th percentiles for each node
        2. Min-Max scale to [0, 1] using these percentiles
        3. Binarize: value = 1 if normalized > 0.75, else 0
    
    Args:
        data: Time-series data of shape [Time_Points, Num_Nodes]
    
    Returns:
        Binarized data of shape [Time_Points, Num_Nodes]
    """
    # Compute percentiles along time axis (axis=0)
    p10 = np.percentile(data, 10, axis=0, keepdims=True)  # [1, N]
    p90 = np.percentile(data, 90, axis=0, keepdims=True)  # [1, N]
    
    # Min-Max scale using percentiles, clamp to [0, 1]
    denom = p90 - p10
    denom[denom == 0] = 1e-8  # Avoid division by zero
    data_norm = (data - p10) / denom
    data_norm = np.clip(data_norm, 0, 1)
    
    # Binarize: 1 if > 0.75, else 0
    data_binary = (data_norm > 0.75).astype(np.float64)
    
    return data_binary


def _compute_joint_probabilities(data_binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute joint probability matrices (theta1-4) for all node pairs.
    
    For nodes i and j:
        theta1[i,j] = P(i=1, j=1) = d1' * d2 / T
        theta2[i,j] = P(i=1, j=0) = d1' * (1-d2) / T
        theta3[i,j] = P(i=0, j=1) = (1-d1)' * d2 / T
        theta4[i,j] = P(i=0, j=0) = (1-d1)' * (1-d2) / T
    
    Args:
        data_binary: Binarized data [T, N]
    
    Returns:
        Tuple of (theta1, theta2, theta3, theta4), each [N, N]
    """
    T, N = data_binary.shape
    
    # Transpose for matrix multiplication: [N, T]
    D = data_binary.T  # [N, T]
    D_not = 1 - D      # [N, T]
    
    # Compute joint probabilities via matrix multiplication
    # theta1[i,j] = sum_t(D[i,t] * D[j,t]) / T
    theta1 = (D @ D.T) / T           # P(i=1, j=1)
    theta2 = (D @ D_not.T) / T       # P(i=1, j=0)
    theta3 = (D_not @ D.T) / T       # P(i=0, j=1)
    theta4 = (D_not @ D_not.T) / T   # P(i=0, j=0)
    
    return theta1, theta2, theta3, theta4


def _compute_kappa(theta1: np.ndarray, theta2: np.ndarray, 
                   theta3: np.ndarray, theta4: np.ndarray) -> np.ndarray:
    """
    Compute Patel's Kappa coefficient matrix.
    
    Kappa measures the strength of association between two variables,
    normalized to account for chance agreement.
    
    Formula (from Pate.m):
        EEE = (theta1 + theta2) * (theta1 + theta3)  # Expected under independence
        max_theta1 = min(theta1+theta2, theta1+theta3)
        min_theta1 = max(0, 2*theta1 + theta2 + theta3 - 1)
        
        if theta1 > EEE:
            DDD = 0.5 + (theta1 - EEE) / (2 * (max_theta1 - EEE))
        else:
            DDD = 0.5 - (theta1 - EEE) / (2 * (EEE - min_theta1))
        
        Kappa = (theta1 - EEE) / (DDD * (max_theta1 - EEE) + (1-DDD) * (EEE - min_theta1))
    
    Args:
        theta1-4: Joint probability matrices [N, N]
    
    Returns:
        Kappa matrix [N, N]
    """
    # Expected joint probability under independence
    EEE = (theta1 + theta2) * (theta1 + theta3)
    
    # Bounds for theta1
    max_theta1 = np.minimum(theta1 + theta2, theta1 + theta3)
    min_theta1 = np.maximum(0, 2 * theta1 + theta2 + theta3 - 1)
    
    # Compute DDD based on whether theta1 > EEE
    DDD = np.zeros_like(theta1)
    
    # Case 1: theta1 > EEE
    mask_gt = theta1 > EEE
    denom_gt = 2 * (max_theta1 - EEE)
    denom_gt[denom_gt == 0] = 1e-8
    DDD[mask_gt] = 0.5 + (theta1[mask_gt] - EEE[mask_gt]) / denom_gt[mask_gt]
    
    # Case 2: theta1 <= EEE
    mask_le = ~mask_gt
    denom_le = 2 * (EEE - min_theta1)
    denom_le[denom_le == 0] = 1e-8
    DDD[mask_le] = 0.5 - (theta1[mask_le] - EEE[mask_le]) / denom_le[mask_le]
    
    # Compute Kappa
    numerator = theta1 - EEE
    denominator = DDD * (max_theta1 - EEE) + (1 - DDD) * (EEE - min_theta1)
    denominator[denominator == 0] = 1e-8
    
    kappa = numerator / denominator
    
    return kappa


def _compute_tau(theta1: np.ndarray, theta2: np.ndarray, 
                 theta3: np.ndarray) -> np.ndarray:
    """
    Compute Patel's Tau coefficient matrix.
    
    Tau measures the directionality of the relationship.
    
    Formula (from Pate.m):
        if theta2 > theta3:
            tau = 1 - (theta1 + theta3) / (theta1 + theta2)
        else:
            tau = (theta1 + theta2) / (theta1 + theta3) - 1
    
    Note: The output uses -tau to match Pate.m's out(2) = -tau_12
    
    Args:
        theta1, theta2, theta3: Joint probability matrices [N, N]
    
    Returns:
        Tau matrix [N, N] (negated as per Pate.m)
    """
    tau = np.zeros_like(theta1)
    
    # Case 1: theta2 > theta3
    mask_gt = theta2 > theta3
    denom1 = theta1 + theta2
    denom1[denom1 == 0] = 1e-8
    tau[mask_gt] = 1 - (theta1[mask_gt] + theta3[mask_gt]) / denom1[mask_gt]
    
    # Case 2: theta2 <= theta3
    mask_le = ~mask_gt
    denom2 = theta1 + theta3
    denom2[denom2 == 0] = 1e-8
    tau[mask_le] = (theta1[mask_le] + theta2[mask_le]) / denom2[mask_le] - 1
    
    # Return -tau to match Pate.m
    return -tau


def compute_patel_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute the Patel connectivity score matrix.
    
    This is the main function that computes the N x N connectivity matrix
    using Patel's Kappa and Tau metrics.
    
    Args:
        data: fMRI time-series of shape [Time_Points, Num_Nodes]
    
    Returns:
        score_matrix: Connectivity scores [N, N], where 
                      Score[i,j] = -Kappa[i,j] * Tau[i,j]
    
    Example:
        >>> data = np.random.randn(200, 50)  # 200 time points, 50 regions
        >>> score_matrix = compute_patel_matrix(data)
        >>> print(score_matrix.shape)
        (50, 50)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array [Time, Nodes], got shape {data.shape}")
    
    T, N = data.shape
    if T < N:
        warnings.warn(f"Data has more nodes ({N}) than time points ({T}). "
                      "Ensure data is [Time, Nodes] format.")
    
    # Step 1: Preprocess (percentile scaling + binarization)
    data_binary = _preprocess_timeseries(data)
    
    # Step 2: Compute joint probabilities
    theta1, theta2, theta3, theta4 = _compute_joint_probabilities(data_binary)
    
    # Step 3: Compute Kappa and Tau
    kappa = _compute_kappa(theta1, theta2, theta3, theta4)
    tau = _compute_tau(theta1, theta2, theta3)
    
    # Step 4: Compute final score = -Kappa * Tau (as per Pate.m out(3))
    score_matrix = -kappa * tau
    
    # Handle NaNs and set diagonal to 0
    score_matrix = np.nan_to_num(score_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(score_matrix, 0)
    
    return score_matrix


def get_functional_modules(
    data: np.ndarray,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None
) -> List[List[int]]:
    """
    Identify functional brain modules using Patel connectivity.
    
    This function computes connectivity, thresholds the graph, and returns
    connected components (functional modules).
    
    Args:
        data: fMRI time-series of shape [Time_Points, Num_Nodes]
        threshold: (Optional) Keep edges with Score > threshold
        top_k: (Optional) Keep the top K strongest edges. 
               If both threshold and top_k are set, top_k takes precedence.
    
    Returns:
        modules: List of lists, where each inner list contains node indices
                 belonging to the same functional module.
                 Sorted by module size (largest first).
    
    Example:
        >>> data = np.random.randn(200, 50)
        >>> modules = get_functional_modules(data, top_k=50)
        >>> print(f"Found {len(modules)} modules")
        >>> print(f"Largest module: {modules[0]}")
    """
    # Compute Patel score matrix
    score_matrix = compute_patel_matrix(data)
    N = score_matrix.shape[0]
    
    # Determine threshold
    if top_k is not None:
        # Get upper triangular values (avoid counting edges twice)
        upper_tri_indices = np.triu_indices(N, k=1)
        upper_tri_values = score_matrix[upper_tri_indices]
        
        # Sort descending and find k-th largest value as threshold
        if top_k >= len(upper_tri_values):
            effective_threshold = upper_tri_values.min() - 1e-8
        else:
            sorted_values = np.sort(upper_tri_values)[::-1]
            effective_threshold = sorted_values[top_k - 1]
    elif threshold is not None:
        effective_threshold = threshold
    else:
        # Default: use mean + std as threshold
        upper_tri_values = score_matrix[np.triu_indices(N, k=1)]
        effective_threshold = np.mean(upper_tri_values) + np.std(upper_tri_values)
        print(f"No threshold specified. Using mean+std: {effective_threshold:.4f}")
    
    # Build graph from edges above threshold
    G = nx.Graph()
    G.add_nodes_from(range(N))  # Add all nodes (even isolated ones)
    
    # Add edges where score > threshold
    for i in range(N):
        for j in range(i + 1, N):
            if score_matrix[i, j] > effective_threshold:
                G.add_edge(i, j, weight=score_matrix[i, j])
    
    # Extract connected components
    components = list(nx.connected_components(G))
    
    # Sort by size (largest first) and convert to sorted lists
    modules = [sorted(list(comp)) for comp in components]
    modules.sort(key=lambda x: len(x), reverse=True)
    
    return modules


def get_functional_modules_detailed(
    data: np.ndarray,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None
) -> Tuple[List[List[int]], np.ndarray, nx.Graph]:
    """
    Extended version that also returns the score matrix and graph.
    
    Args:
        data: fMRI time-series [Time, Nodes]
        threshold: Edge threshold
        top_k: Number of top edges to keep
    
    Returns:
        modules: List of functional modules
        score_matrix: The N x N Patel connectivity matrix
        graph: The thresholded NetworkX graph
    """
    score_matrix = compute_patel_matrix(data)
    N = score_matrix.shape[0]
    
    # Determine threshold
    if top_k is not None:
        upper_tri_values = score_matrix[np.triu_indices(N, k=1)]
        if top_k >= len(upper_tri_values):
            effective_threshold = upper_tri_values.min() - 1e-8
        else:
            sorted_values = np.sort(upper_tri_values)[::-1]
            effective_threshold = sorted_values[top_k - 1]
    elif threshold is not None:
        effective_threshold = threshold
    else:
        upper_tri_values = score_matrix[np.triu_indices(N, k=1)]
        effective_threshold = np.mean(upper_tri_values) + np.std(upper_tri_values)
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    
    for i in range(N):
        for j in range(i + 1, N):
            if score_matrix[i, j] > effective_threshold:
                G.add_edge(i, j, weight=score_matrix[i, j])
    
    # Extract components
    components = list(nx.connected_components(G))
    modules = [sorted(list(comp)) for comp in components]
    modules.sort(key=lambda x: len(x), reverse=True)
    
    return modules, score_matrix, G


# ============================================================================
# Main demonstration
# ============================================================================

if __name__ == "__main__":
    import pandas as pd
    import os
    
    print("=" * 60)
    print("Patel's Kappa/Tau Connectivity - Functional Module Detection")
    print("=" * 60)
    
    # Configuration
    TIME_POINTS_PER_SUBJECT = 200
    TOP_K_EDGES = 50
    
    # Try to find fMRI.csv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "..", "..", "fMRI_dataset", "fMRI.csv"),
        os.path.join(script_dir, "..", "datasets", "fMRI.csv"),
        "fMRI.csv",
    ]
    
    fmri_path = None
    for path in possible_paths:
        if os.path.exists(path):
            fmri_path = path
            break
    
    if fmri_path is None:
        print("\n[Demo Mode] fMRI.csv not found. Using synthetic data.\n")
        # Generate synthetic data: 200 time points, 20 brain regions
        np.random.seed(42)
        T, N = 200, 20
        
        # Create 3 clusters with correlated signals
        subject_data = np.random.randn(T, N)
        # Cluster 1: regions 0-6
        base_signal_1 = np.random.randn(T)
        for i in range(7):
            subject_data[:, i] += 2 * base_signal_1 + 0.3 * np.random.randn(T)
        # Cluster 2: regions 7-12
        base_signal_2 = np.random.randn(T)
        for i in range(7, 13):
            subject_data[:, i] += 2 * base_signal_2 + 0.3 * np.random.randn(T)
        # Cluster 3: regions 13-19
        base_signal_3 = np.random.randn(T)
        for i in range(13, 20):
            subject_data[:, i] += 2 * base_signal_3 + 0.3 * np.random.randn(T)
        
        print(f"Generated synthetic data: {subject_data.shape} [Time, Nodes]")
    else:
        print(f"\nLoading data from: {fmri_path}")
        df = pd.read_csv(fmri_path, header=None)
        data = df.values.astype(np.float32)
        
        print(f"Loaded data shape: {data.shape} [Total_Rows, Num_Nodes]")
        
        # Extract first subject
        subject_data = data[:TIME_POINTS_PER_SUBJECT, :]
        print(f"Using first subject: {subject_data.shape} [Time, Nodes]")
    
    # Compute functional modules
    print(f"\nComputing Patel connectivity with top_k={TOP_K_EDGES} edges...")
    modules = get_functional_modules(subject_data, top_k=TOP_K_EDGES)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Found {len(modules)} functional modules.\n")
    
    # Show top 5 modules
    for i, module in enumerate(modules[:5]):
        if len(module) > 1:
            print(f"Module {i+1}: {len(module)} nodes -> {module}")
        else:
            print(f"Module {i+1}: {len(module)} node (isolated) -> {module}")
    
    if len(modules) > 5:
        remaining = len(modules) - 5
        print(f"... and {remaining} smaller modules")
    
    # Show largest module details
    print(f"\n{'='*60}")
    print(f"Largest module contains {len(modules[0])} nodes: {modules[0]}")
    print("=" * 60)
