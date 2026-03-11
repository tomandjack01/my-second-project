#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDM Pipeline Diagnostic Script — Read-only root cause analysis.

Tests 5 hypotheses for why F1 is low on sim4 (50 nodes, 61 GT edges):
  H1: Patel cross-subject concatenation dilutes directional prior
  H2: Cosine-annealed auxiliary lambdas vanish in late training
  H3: L1 regularization over-sparsifies (signal below threshold)
  H4: Transpose convention mismatch
  H5: Per-subject gradient updates vs mini-batch accumulation

Usage:
    cd GraphExp
    python diagnose_pipeline.py \
        --csv_path ../fMRI_dataset/sim4.csv \
        --gt_path ../fMRI_dataset/h4.txt \
        --pred_path results/run_XXXXXXXX_XXXXXX/learned_adjacency.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reuse project utilities (read-only imports)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.patel_util import compute_patel_components
from test_eval import load_ground_truth, evaluate_directional

TIME_POINTS_PER_SUBJECT = 200


def _banner(hypothesis_num: int, title: str):
    print(f"\n{'=' * 72}")
    print(f"  HYPOTHESIS {hypothesis_num}: {title}")
    print(f"{'=' * 72}")


def _verdict(confirmed: bool, severity: str, reason: str):
    tag = "CONFIRMED" if confirmed else ("REJECTED" if confirmed is False else "INCONCLUSIVE")
    print(f"\nVERDICT:  {tag}")
    print(f"SEVERITY: {severity}")
    print(f"REASON:   {reason}")


# ============================================================================
# H1: Patel cross-subject concatenation vs per-subject averaging
# ============================================================================

def diagnose_patel_quality(data_2d_np: np.ndarray, gt_edges: set, num_nodes: int):
    """Compare concatenated vs per-subject-averaged Patel tau quality."""
    _banner(1, "Patel cross-subject concatenation dilutes directional prior")

    total_rows = data_2d_np.shape[0]
    num_subjects = total_rows // TIME_POINTS_PER_SUBJECT

    # --- Method A: Full concatenation (current pipeline) ---
    _, kappa_concat, tau_concat = compute_patel_components(data_2d_np)

    # --- Method B: Per-subject, then average ---
    tau_per_subj = []
    kappa_per_subj = []
    for s in range(num_subjects):
        start = s * TIME_POINTS_PER_SUBJECT
        end = start + TIME_POINTS_PER_SUBJECT
        subj_data = data_2d_np[start:end]
        _, k_s, t_s = compute_patel_components(subj_data)
        tau_per_subj.append(t_s)
        kappa_per_subj.append(k_s)
    tau_avg = np.mean(tau_per_subj, axis=0)
    kappa_avg = np.mean(kappa_per_subj, axis=0)

    # --- Metrics ---
    def _tau_metrics(tau, label, gt_edges_set):
        delta = tau - tau.T
        abs_delta = np.abs(delta)
        off_diag = abs_delta[~np.eye(tau.shape[0], dtype=bool)]

        # Direction agreement on GT edges
        agree = 0
        total_gt = 0
        for src, dst in gt_edges_set:
            total_gt += 1
            # tau[src,dst] > tau[dst,src] means src→dst direction
            if tau[src, dst] > tau[dst, src]:
                agree += 1
        agreement_rate = agree / total_gt if total_gt > 0 else 0.0

        print(f"\n  [{label}]")
        print(f"    Tau value range:       [{tau.min():.4f}, {tau.max():.4f}]")
        print(f"    |tau[i,j]-tau[j,i]| mean:   {off_diag.mean():.4f}")
        print(f"    |tau[i,j]-tau[j,i]| median: {np.median(off_diag):.4f}")
        print(f"    GT direction agreement:     {agree}/{total_gt} = {agreement_rate:.2%}")
        return agreement_rate

    rate_concat = _tau_metrics(tau_concat, "Concatenated (current)", gt_edges)
    rate_avg = _tau_metrics(tau_avg, "Per-subject averaged", gt_edges)

    # Also check per-subject variance of direction agreement
    per_subj_rates = []
    for t_s in tau_per_subj:
        agree_s = sum(1 for src, dst in gt_edges if t_s[src, dst] > t_s[dst, src])
        per_subj_rates.append(agree_s / len(gt_edges))
    print(f"\n  Per-subject agreement rates: "
          f"mean={np.mean(per_subj_rates):.2%}, "
          f"std={np.std(per_subj_rates):.2%}, "
          f"min={np.min(per_subj_rates):.2%}, "
          f"max={np.max(per_subj_rates):.2%}")

    diff = rate_avg - rate_concat
    if diff > 0.05:
        _verdict(True, "HIGH",
                 f"Per-subject avg agreement {rate_avg:.2%} > concat {rate_concat:.2%} "
                 f"(+{diff:.2%}). Concatenation dilutes direction signal.")
    elif diff > 0.02:
        _verdict(None, "MEDIUM",
                 f"Small improvement with per-subject avg (+{diff:.2%}). "
                 f"Marginal dilution effect.")
    else:
        _verdict(False, "LOW",
                 f"Concat ({rate_concat:.2%}) ≈ per-subject avg ({rate_avg:.2%}). "
                 f"No significant dilution.")

# ============================================================================
# H2: Auxiliary lambda cosine anneal — does direction constraint vanish?
# ============================================================================

def diagnose_lambda_schedule(num_epochs: int = 300):
    """Simulate the lambda schedule over training to check effective range."""
    _banner(2, "Cosine anneal causes auxiliary lambdas to vanish in late training")

    # Simulate with representative loss values
    # (ratio-adaptive means absolute lambda depends on loss magnitudes,
    #  but the *shape* of the schedule is deterministic)
    warmup_epochs = 5
    prev_dir, prev_ortho = 0.0, 0.0

    # Use typical loss magnitudes from training logs
    mock_main_loss = 0.5
    mock_dir_loss = 0.3
    mock_ortho_loss = 0.1

    import torch
    loss_main_t = torch.tensor(mock_main_loss)
    loss_dir_t = torch.tensor(mock_dir_loss)
    loss_ortho_t = torch.tensor(mock_ortho_loss)

    from main_structure_learning import compute_auxiliary_lambdas

    epochs = []
    lambda_dirs = []
    lambda_orthos = []
    epoch_factors = []

    for epoch in range(num_epochs):
        ld, lo = compute_auxiliary_lambdas(
            epoch=epoch,
            num_epochs=num_epochs,
            loss_ddm_main=loss_main_t,
            raw_loss_dir=loss_dir_t,
            raw_loss_ortho=loss_ortho_t,
            prev_lambda_dir=prev_dir,
            prev_lambda_ortho=prev_ortho,
            warmup_epochs=warmup_epochs,
        )
        prev_dir, prev_ortho = ld, lo
        epochs.append(epoch + 1)
        lambda_dirs.append(ld)
        lambda_orthos.append(lo)

        # Compute raw epoch_factor for reference
        if epoch < warmup_epochs:
            ef = 0.0
        else:
            post_warmup = max(num_epochs - warmup_epochs, 1)
            ramp_epochs = max(1, min(10, post_warmup))
            ramp = min(1.0, float(epoch - warmup_epochs + 1) / float(ramp_epochs))
            progress = float(epoch - warmup_epochs) / float(max(post_warmup - 1, 1))
            anneal = 0.5 * (1.0 + math.cos(math.pi * progress))
            ef = ramp * anneal
        epoch_factors.append(ef)

    lambda_dirs = np.array(lambda_dirs)
    lambda_orthos = np.array(lambda_orthos)
    epoch_factors = np.array(epoch_factors)

    # Find effective range (lambda_dir > 0.001)
    active_dir = np.where(lambda_dirs > 0.001)[0]
    active_ortho = np.where(lambda_orthos > 0.001)[0]

    print(f"\n  Schedule parameters: warmup={warmup_epochs}, total={num_epochs}")
    print(f"  Mock losses: main={mock_main_loss}, dir={mock_dir_loss}, ortho={mock_ortho_loss}")

    print(f"\n  Lambda_dir:")
    print(f"    Peak value:     {lambda_dirs.max():.6f} at epoch {lambda_dirs.argmax() + 1}")
    print(f"    Active range:   epoch {active_dir[0]+1 if len(active_dir) else 'N/A'} "
          f"to {active_dir[-1]+1 if len(active_dir) else 'N/A'} "
          f"({len(active_dir)} epochs)")
    print(f"    Value at 50%:   {lambda_dirs[num_epochs//2]:.6f}")
    print(f"    Value at 75%:   {lambda_dirs[int(num_epochs*0.75)]:.6f}")
    print(f"    Value at 90%:   {lambda_dirs[int(num_epochs*0.9)]:.6f}")
    print(f"    Final value:    {lambda_dirs[-1]:.6f}")

    print(f"\n  Lambda_ortho:")
    print(f"    Peak value:     {lambda_orthos.max():.6f} at epoch {lambda_orthos.argmax() + 1}")
    print(f"    Active range:   epoch {active_ortho[0]+1 if len(active_ortho) else 'N/A'} "
          f"to {active_ortho[-1]+1 if len(active_ortho) else 'N/A'} "
          f"({len(active_ortho)} epochs)")

    print(f"\n  Raw epoch_factor (ramp * cosine):")
    print(f"    Reaches 0 at:   ~epoch {np.where(epoch_factors[warmup_epochs:] < 0.01)[0][0] + warmup_epochs + 1 if np.any(epoch_factors[warmup_epochs:] < 0.01) else 'never'}")

    # Verdict
    half_point = num_epochs // 2
    if len(active_dir) > 0 and active_dir[-1] < half_point:
        _verdict(True, "CRITICAL",
                 f"Direction lambda drops below 0.001 by epoch {active_dir[-1]+1}, "
                 f"only {len(active_dir)}/{num_epochs} epochs have active constraint.")
    elif len(active_dir) > 0 and active_dir[-1] < int(num_epochs * 0.75):
        _verdict(True, "HIGH",
                 f"Direction lambda active for {len(active_dir)}/{num_epochs} epochs. "
                 f"Last 25% of training has no directional guidance.")
    else:
        _verdict(False, "LOW",
                 f"Direction lambda remains active for {len(active_dir)}/{num_epochs} epochs.")

# ============================================================================
# H3: L1 over-sparsification — signal exists but below threshold
# ============================================================================

def diagnose_adjacency_distribution(adj: np.ndarray, gt_edges: set, num_nodes: int):
    """Analyze weight distribution and multi-threshold precision/recall."""
    _banner(3, "L1 regularization over-sparsifies (signal below threshold)")

    n = adj.shape[0]
    off_diag = adj[~np.eye(n, dtype=bool)]

    print(f"\n  Adjacency shape: {adj.shape}")
    print(f"  Overall statistics:")
    print(f"    Mean:   {off_diag.mean():.4f}")
    print(f"    Median: {np.median(off_diag):.4f}")
    print(f"    Std:    {off_diag.std():.4f}")
    print(f"    Min:    {off_diag.min():.4f}")
    print(f"    Max:    {off_diag.max():.4f}")

    # Weight distribution buckets
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    print(f"\n  Weight distribution (directed edges, excl. diagonal):")
    for t in thresholds:
        count = int((off_diag > t).sum())
        print(f"    adj > {t:.2f}: {count:5d} edges ({count / off_diag.size:.2%})")

    # GT edge weight analysis
    gt_weights = []
    gt_margins = []
    for src, dst in gt_edges:
        w_fwd = float(adj[src, dst])
        w_rev = float(adj[dst, src])
        gt_weights.append(max(w_fwd, w_rev))
        gt_margins.append(abs(w_fwd - w_rev))

    gt_weights = np.array(gt_weights)
    gt_margins = np.array(gt_margins)

    print(f"\n  GT edge weights (max of both directions):")
    print(f"    Mean:   {gt_weights.mean():.4f}")
    print(f"    Median: {np.median(gt_weights):.4f}")
    print(f"    >0.5:   {(gt_weights > 0.5).sum()}/{len(gt_weights)}")
    print(f"    >0.1:   {(gt_weights > 0.1).sum()}/{len(gt_weights)}")
    print(f"    >0.05:  {(gt_weights > 0.05).sum()}/{len(gt_weights)}")

    print(f"\n  GT edge direction margins |adj[s,d]-adj[d,s]|:")
    print(f"    Mean:   {gt_margins.mean():.4f}")
    print(f"    Median: {np.median(gt_margins):.4f}")

    # Multi-threshold evaluation using test_eval logic
    print(f"\n  Multi-threshold evaluation (top-k = {len(gt_edges)}):")
    print(f"  {'Method':<20s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'TP':>4s}")
    print(f"  {'-'*42}")

    for top_k in [len(gt_edges), len(gt_edges) * 2, len(gt_edges) * 3]:
        result = evaluate_directional(adj, gt_edges, top_k=top_k)
        label = f"top_k={top_k}"
        print(f"  {label:<20s} {result['precision']:6.3f} {result['recall']:6.3f} "
              f"{result['f1']:6.3f} {result['tp']:4d}")

    # Verdict
    recall_at_3x = evaluate_directional(adj, gt_edges, top_k=len(gt_edges) * 3)['recall']
    recall_at_1x = evaluate_directional(adj, gt_edges, top_k=len(gt_edges))['recall']
    recall_gain = recall_at_3x - recall_at_1x

    if gt_weights.mean() < 0.1:
        _verdict(True, "CRITICAL",
                 f"Mean GT edge weight = {gt_weights.mean():.4f} (very low). "
                 f"L1 has suppressed nearly all signal.")
    elif recall_gain > 0.15:
        _verdict(True, "HIGH",
                 f"Recall jumps from {recall_at_1x:.2%} to {recall_at_3x:.2%} "
                 f"at 3x top-k. Signal exists but is below selection threshold.")
    else:
        _verdict(False, "LOW",
                 f"Weight distribution looks reasonable. "
                 f"Recall gain at 3x is only +{recall_gain:.2%}.")

# ============================================================================
# H4: Transpose convention — does flipping improve or hurt?
# ============================================================================

def diagnose_transpose_direction(adj_raw: np.ndarray, gt_edges: set, num_nodes: int):
    """Compare evaluation with and without transpose."""
    _banner(4, "Transpose convention mismatch")

    adj_transposed = adj_raw.T  # raw → causal convention

    top_k = len(gt_edges)

    print(f"\n  Evaluating with top_k = {top_k} (= |GT edges|)")

    for label, adj in [("Transposed (raw→causal)", adj_transposed),
                       ("No transpose (raw as-is)", adj_raw)]:
        result = evaluate_directional(adj, gt_edges, top_k=top_k)
        print(f"\n  [{label}]")
        print(f"    TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
        print(f"    Precision={result['precision']:.4f}, "
              f"Recall={result['recall']:.4f}, F1={result['f1']:.4f}")

    # Per-edge analysis on GT: which direction does the model prefer?
    tp_trans = evaluate_directional(adj_transposed, gt_edges, top_k=top_k)
    tp_raw = evaluate_directional(adj_raw, gt_edges, top_k=top_k)

    print(f"\n  Per-GT-edge direction analysis (on transposed adj):")
    correct_dir = 0
    wrong_dir = 0
    weak_signal = 0
    for src, dst in sorted(gt_edges):
        w_fwd = float(adj_transposed[src, dst])
        w_rev = float(adj_transposed[dst, src])
        margin = abs(w_fwd - w_rev)
        max_w = max(w_fwd, w_rev)

        if max_w < 0.05:
            weak_signal += 1
            tag = "WEAK"
        elif w_fwd > w_rev:
            correct_dir += 1
            tag = "CORRECT"
        else:
            wrong_dir += 1
            tag = "WRONG"

        if len(gt_edges) <= 80:  # Only print for small edge sets
            print(f"    GT({src+1:2d}→{dst+1:2d}): "
                  f"adj[s,d]={w_fwd:.4f} adj[d,s]={w_rev:.4f} "
                  f"margin={margin:.4f} [{tag}]")

    print(f"\n  Direction summary on GT edges:")
    print(f"    Correct direction: {correct_dir}/{len(gt_edges)} ({correct_dir/len(gt_edges):.2%})")
    print(f"    Wrong direction:   {wrong_dir}/{len(gt_edges)} ({wrong_dir/len(gt_edges):.2%})")
    print(f"    Weak signal:       {weak_signal}/{len(gt_edges)} ({weak_signal/len(gt_edges):.2%})")

    tp_diff = tp_trans['tp'] - tp_raw['tp']
    if tp_diff > 3:
        _verdict(False, "LOW",
                 f"Transpose improves TP by {tp_diff}. Convention is correct.")
    elif tp_diff < -3:
        _verdict(True, "HIGH",
                 f"Transpose HURTS TP by {abs(tp_diff)}. Convention may be inverted.")
    else:
        _verdict(None, "MEDIUM",
                 f"TP difference is small ({tp_diff}). "
                 f"Transpose convention is not the main issue.")


# ============================================================================
# H5: Per-subject updates vs batch accumulation
# ============================================================================

def diagnose_gradient_updates(num_subjects: int, batch_size: int = 4):
    """Analyze gradient update frequency."""
    _banner(5, "Per-subject gradient updates vs batch accumulation")

    updates_per_epoch = num_subjects  # current: optimizer.step() per subject
    batches_per_epoch = math.ceil(num_subjects / batch_size)

    print(f"\n  Current behavior (from code):")
    print(f"    Subjects: {num_subjects}")
    print(f"    Batch size: {batch_size}")
    print(f"    optimizer.zero_grad() + backward() + step(): per SUBJECT")
    print(f"    → {updates_per_epoch} parameter updates per epoch")

    print(f"\n  Expected mini-batch behavior:")
    print(f"    optimizer.zero_grad() per batch, backward() per subject, step() per batch")
    print(f"    → {batches_per_epoch} parameter updates per epoch")

    print(f"\n  Impact:")
    ratio = updates_per_epoch / batches_per_epoch
    print(f"    Current updates are {ratio:.1f}x more frequent than mini-batch")
    print(f"    Each update uses gradient from 1 subject (high variance)")
    print(f"    Mini-batch would average {batch_size} subjects (lower variance)")

    # This is a code-level observation, always confirmed
    _verdict(True, "MEDIUM",
             f"Per-subject SGD ({updates_per_epoch} updates/epoch) instead of "
             f"mini-batch ({batches_per_epoch} updates/epoch). "
             f"High gradient variance may destabilize structure learning.")

# ============================================================================
# MAIN
# ============================================================================

def find_latest_pred(results_dir: Path) -> Path:
    """Find the most recent learned_adjacency file."""
    run_dirs = sorted(
        [p for p in results_dir.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        for fname in ("learned_adjacency.csv", "learned_adjacency.npy"):
            pred = run_dir / fname
            if pred.exists():
                return pred
    raise FileNotFoundError(f"No learned_adjacency found under {results_dir}")


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="DDM Pipeline Diagnostic — root cause analysis for low F1"
    )
    parser.add_argument("--csv_path", type=str,
                        default=str(repo_root / "fMRI_dataset" / "sim4.csv"))
    parser.add_argument("--gt_path", type=str,
                        default=str(repo_root / "fMRI_dataset" / "h4.txt"))
    parser.add_argument("--pred_path", type=str, default=None,
                        help="Path to learned_adjacency.csv (default: latest run)")
    parser.add_argument("--results_dir", type=str,
                        default=str(script_dir / "results"))
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="Number of epochs to simulate for lambda schedule")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # --- Load data ---
    print("=" * 72)
    print("  DDM PIPELINE DIAGNOSTIC")
    print("=" * 72)

    csv_path = Path(args.csv_path)
    gt_path = Path(args.gt_path)

    df = pd.read_csv(csv_path, header=None)
    data_2d_np = df.values.astype(np.float64)
    total_rows, num_nodes = data_2d_np.shape
    num_subjects = total_rows // TIME_POINTS_PER_SUBJECT

    print(f"Data:     {csv_path.name} ({total_rows} rows, {num_nodes} nodes, "
          f"{num_subjects} subjects)")

    gt_edges = load_ground_truth(gt_path)
    print(f"GT:       {gt_path.name} ({len(gt_edges)} directed edges)")

    # Load prediction (raw convention — NOT transposed)
    if args.pred_path:
        pred_path = Path(args.pred_path)
    else:
        pred_path = find_latest_pred(Path(args.results_dir))
    print(f"Pred:     {pred_path}")

    if pred_path.suffix == ".npy":
        adj_raw = np.load(pred_path)
    else:
        adj_raw = np.loadtxt(pred_path, delimiter=",")
    print(f"Adj shape: {adj_raw.shape}")

    # --- Run all diagnostics ---
    diagnose_patel_quality(data_2d_np, gt_edges, num_nodes)
    diagnose_lambda_schedule(num_epochs=args.num_epochs)
    diagnose_adjacency_distribution(adj_raw.T, gt_edges, num_nodes)  # transpose to causal
    diagnose_transpose_direction(adj_raw, gt_edges, num_nodes)
    diagnose_gradient_updates(num_subjects, batch_size=args.batch_size)

    # --- Summary ---
    print(f"\n{'=' * 72}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'=' * 72}")
    print("Review VERDICT for each hypothesis above to prioritize fixes.")


if __name__ == "__main__":
    main()
