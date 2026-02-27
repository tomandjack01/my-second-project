import numpy as np
import pandas as pd

# ============================================================================
# Load Ground Truth
# ============================================================================
gt_edges = set()
with open('../fMRI_dataset/h3.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            # Convert from 1-indexed to 0-indexed
            src = int(parts[0]) - 1
            dst = int(parts[1]) - 1
            gt_edges.add((src, dst))

print(f"Ground Truth: {len(gt_edges)} directed edges")

# ============================================================================
# Load Learned Adjacency
# ============================================================================
adj = pd.read_csv('results/run_20260110_191837/learned_adjacency.csv', header=None).values
N = adj.shape[0]

print(f"Adjacency matrix shape: {adj.shape}")
print(f"Adjacency stats: min={adj.min():.4f}, max={adj.max():.4f}, mean={adj.mean():.4f}")

# ============================================================================
# Metric 1: Connection Sensitivity (c-sensitivity)
# Based on Smith et al. 2011 "Network modelling methods for FMRI"
# ============================================================================
print("\n" + "=" * 80)
print("Metric 1: Connection Sensitivity (c-sensitivity)")
print("=" * 80)

# Step A & B: Collect weights for non-edges (False Positive distribution)
non_edge_weights = []
for i in range(N):
    for j in range(N):
        if i != j and (i, j) not in gt_edges:
            non_edge_weights.append(adj[i, j])

non_edge_weights = np.array(non_edge_weights)

# Print FP distribution statistics
print(f"\nFalse Positive Distribution (non-edge weights):")
print(f"  Count: {len(non_edge_weights)}")
print(f"  Mean:  {non_edge_weights.mean():.4f}")
print(f"  Std:   {non_edge_weights.std():.4f}")
print(f"  Max:   {non_edge_weights.max():.4f}")
print(f"  Min:   {non_edge_weights.min():.4f}")

# Step C: Calculate 95th percentile threshold
threshold_95 = np.percentile(non_edge_weights, 95)
print(f"\n95th Percentile Threshold: {threshold_95:.4f}")

# Step D: Count true edges above threshold
true_edge_weights = []
true_edges_above_threshold = 0
for src, dst in gt_edges:
    weight = adj[src, dst]
    true_edge_weights.append(weight)
    if weight > threshold_95:
        true_edges_above_threshold += 1

# Calculate c-sensitivity
c_sensitivity = true_edges_above_threshold / len(gt_edges) if len(gt_edges) > 0 else 0

print(f"\nTrue edges above threshold: {true_edges_above_threshold} / {len(gt_edges)}")
print(f">>> c-sensitivity = {c_sensitivity:.4f} ({c_sensitivity*100:.1f}%)")

# ============================================================================
# Metric 2: Direction Accuracy (d-accuracy)
# ============================================================================
print("\n" + "=" * 80)
print("Metric 2: Direction Accuracy (d-accuracy)")
print("=" * 80)

correct_direction = 0
wrong_direction = 0
tie_count = 0

direction_details = []

for src, dst in gt_edges:
    w_forward = adj[src, dst]   # W_{i -> j}
    w_backward = adj[dst, src]  # W_{j -> i}
    diff = w_forward - w_backward
    
    if diff > 0:
        correct_direction += 1
        direction_details.append((src, dst, w_forward, w_backward, diff, "Correct"))
    elif diff < 0:
        wrong_direction += 1
        direction_details.append((src, dst, w_forward, w_backward, diff, "Wrong"))
    else:
        tie_count += 1
        direction_details.append((src, dst, w_forward, w_backward, diff, "Tie"))

# Calculate d-accuracy
total_edges = len(gt_edges)
d_accuracy = correct_direction / total_edges if total_edges > 0 else 0

print(f"\nDirection Analysis Results:")
print(f"  Correct direction (W_ij > W_ji): {correct_direction}")
print(f"  Wrong direction (W_ij < W_ji):   {wrong_direction}")
print(f"  Tie (W_ij = W_ji):               {tie_count}")
print(f"\n>>> d-accuracy = {d_accuracy:.4f} ({d_accuracy*100:.1f}%)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary (Smith et al. 2011 Metrics)")
print("=" * 80)
print(f"  c-sensitivity: {c_sensitivity:.4f} ({c_sensitivity*100:.1f}%)")
print(f"  d-accuracy:    {d_accuracy:.4f} ({d_accuracy*100:.1f}%)")
print("=" * 80)

# ============================================================================
# Detailed Direction Analysis (Optional)
# ============================================================================
print("\n" + "=" * 80)
print("Detailed Direction Analysis")
print("=" * 80)
print(f"{'Edge':<12} {'W_ij':<10} {'W_ji':<10} {'Diff':<10} {'Result'}")
print("-" * 52)

# Sort by absolute diff to show most confident predictions first
direction_details.sort(key=lambda x: abs(x[4]), reverse=True)
for src, dst, w_fwd, w_bwd, diff, result in direction_details[:15]:
    print(f"({src:2d},{dst:2d})    {w_fwd:8.4f}   {w_bwd:8.4f}   {diff:+8.4f}   {result}")

if len(direction_details) > 15:
    print(f"... and {len(direction_details) - 15} more edges")
