#!/usr/bin/env python3
"""
Final-only multi-seed validation with margin diagnostics.

Purpose:
1. Remove early best-epoch selection bias by evaluating only final epoch outputs.
2. Probe whether non-Patel initialization can sustain directional asymmetry.
3. Distinguish two failure modes:
   - symmetric collapse: margins go to ~0, any "correct" directions are noise/ties
   - wrong-direction asymmetry: margins stay nonzero but directions are inaccurate
4. Compare baseline/treatment conditions across cross-prediction and
   directional-prior variants.
"""

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


FAILURE_MODES = (
    "symmetric_collapse",
    "wrong_direction_asymmetry",
    "weak_asymmetry",
    "mixed_or_partial",
)

BASE_AGGREGATE_METRICS = (
    "final_precision",
    "final_recall",
    "final_f1",
    "best_precision",
    "best_recall",
    "best_f1",
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "strict_pred_count",
    "best_strict_precision",
    "best_strict_recall",
    "best_strict_f1",
    "best_strict_pred_count",
    "final_diff_loss",
    "margin_mean",
    "margin_median",
    "margin_std",
    "margin_p90",
    "margin_max",
    "margin_lt_1e3_frac",
    "margin_lt_1e2_frac",
    "final_same_dir_vs_tau",
    "adj_offdiag_mean",
    "adj_offdiag_cv",
    "adj_in_degree_mean",
    "adj_eff_parents_mean",
    "adj_eff_parents_p90",
    "adj_top1_share_mean",
    "adj_top2_share_mean",
    "gt_forward_weight_mean",
    "gt_reverse_weight_mean",
    "gt_signed_margin_mean",
    "gt_signed_margin_median",
    "gt_signed_margin_p10",
    "gt_signed_margin_p90",
    "gt_signed_margin_frac_pos",
    "non_gt_weight_mean",
    "best_margin_median",
    "best_margin_p90",
    "best_margin_lt_1e2_frac",
    "best_same_dir_vs_tau",
    "best_gt_signed_margin_median",
    "best_gt_signed_margin_frac_pos",
    "best_adj_eff_parents_mean",
    "best_final_gap_gt_signed_margin_median",
)

BASE_PAIRED_DELTA_METRICS = (
    "final_f1",
    "best_f1",
    "strict_f1",
    "best_strict_f1",
    "strict_precision",
    "strict_recall",
    "strict_pred_count",
    "best_strict_precision",
    "best_strict_recall",
    "best_strict_pred_count",
    "margin_median",
    "margin_p90",
    "margin_lt_1e2_frac",
    "margin_lt_1e3_frac",
    "final_same_dir_vs_tau",
    "gt_signed_margin_mean",
    "gt_signed_margin_median",
    "gt_signed_margin_frac_pos",
    "gt_forward_weight_mean",
    "gt_reverse_weight_mean",
    "non_gt_weight_mean",
    "best_margin_median",
    "best_margin_p90",
    "best_margin_lt_1e2_frac",
    "best_gt_signed_margin_median",
    "best_gt_signed_margin_frac_pos",
    "best_adj_eff_parents_mean",
    "best_final_gap_gt_signed_margin_median",
)

def cross_pred_schedule_name(enable_cross_prediction: bool, schedule: str) -> str:
    return schedule if enable_cross_prediction else "disabled"


def cross_pred_aggregation_name(enable_cross_prediction: bool, aggregation: str) -> str:
    return aggregation if enable_cross_prediction else "disabled"


def resolve_causal_lag_main_weight(
    enable_cross_prediction: bool,
    cross_pred_fixed_weight: float,
    cross_pred_target_ratio: float,
) -> float:
    """Backward-compatible mapping from legacy cross-pred knobs to causal_lag_main."""
    if not enable_cross_prediction:
        return 0.0
    if cross_pred_fixed_weight > 0.0:
        return float(cross_pred_fixed_weight)
    return max(0.0, float(cross_pred_target_ratio))


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_text_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def normalize_margin_eps(value: float) -> float:
    return 0.0 if abs(value) <= 1e-12 else float(value)


def margin_eps_label(value: float) -> str:
    normalized = normalize_margin_eps(value)
    if normalized == 0.0:
        return "0"
    return np.format_float_positional(normalized, trim="-").replace(".", "p")


def strict_metric_field(metric: str, margin_eps: float) -> str:
    return f"{metric}_eps_{margin_eps_label(margin_eps)}"


def build_extra_strict_metric_names(margin_eps_values: Sequence[float]) -> List[str]:
    metrics: List[str] = []
    for margin_eps in margin_eps_values:
        for metric in ("strict_precision", "strict_recall", "strict_f1", "strict_pred_count"):
            metrics.append(strict_metric_field(metric, margin_eps))
    return metrics


def best_strict_metric_field(metric: str, margin_eps: float) -> str:
    return f"best_{strict_metric_field(metric, margin_eps)}"


def build_best_extra_strict_metric_names(margin_eps_values: Sequence[float]) -> List[str]:
    metrics: List[str] = []
    for margin_eps in margin_eps_values:
        for metric in ("strict_precision", "strict_recall", "strict_f1", "strict_pred_count"):
            metrics.append(best_strict_metric_field(metric, margin_eps))
    return metrics


def best_final_gap_metric_field(metric: str, margin_eps: float) -> str:
    return f"best_final_gap_{metric}_eps_{margin_eps_label(margin_eps)}"


def build_best_final_gap_metric_names(margin_eps_values: Sequence[float]) -> List[str]:
    return [
        best_final_gap_metric_field("strict_f1", margin_eps)
        for margin_eps in margin_eps_values
    ]


def format_margin_eps_summary(row: Dict[str, object], margin_eps_values: Sequence[float]) -> str:
    parts = []
    for margin_eps in margin_eps_values:
        f1_key = strict_metric_field("strict_f1", margin_eps)
        pred_key = strict_metric_field("strict_pred_count", margin_eps)
        parts.append(
            f"eps={normalize_margin_eps(margin_eps):g}:F1={float(row[f1_key]):.4f}/pred={int(row[pred_key])}"
        )
    return " ".join(parts)


def load_final_diff_loss(result_dir: Path) -> float:
    loss_history_path = result_dir / "loss_history.csv"
    with loss_history_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No loss rows found in {loss_history_path}")
    return float(rows[-1]["loss"])


def adjacency_density_stats(adj: np.ndarray) -> Dict[str, float]:
    offdiag_mask = ~np.eye(adj.shape[0], dtype=bool)
    offdiag = adj[offdiag_mask]
    if offdiag.size == 0:
        return {
            "adj_offdiag_mean": 0.0,
            "adj_offdiag_cv": 0.0,
            "adj_in_degree_mean": 0.0,
        }
    offdiag_mean = float(offdiag.mean())
    offdiag_std = float(offdiag.std())
    return {
        "adj_offdiag_mean": offdiag_mean,
        "adj_offdiag_cv": offdiag_std / (offdiag_mean + 1e-8),
        "adj_in_degree_mean": float(adj.sum(axis=0).mean()),
    }


def adjacency_parent_concentration_stats(adj: np.ndarray) -> Dict[str, float]:
    """Summarize incoming-edge concentration for each target node on a causal adjacency."""
    effective_parents = []
    top1_shares = []
    top2_shares = []
    for target_idx in range(adj.shape[1]):
        incoming = np.array(adj[:, target_idx], dtype=float)
        total = float(incoming.sum())
        if total <= 1e-12:
            effective_parents.append(0.0)
            top1_shares.append(0.0)
            top2_shares.append(0.0)
            continue
        probs = incoming / total
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
        effective_parents.append(float(np.exp(entropy)))
        sorted_probs = np.sort(probs)[::-1]
        top1_shares.append(float(sorted_probs[0]))
        top2_shares.append(float(sorted_probs[:2].sum()))
    return {
        "adj_eff_parents_mean": float(np.mean(effective_parents)),
        "adj_eff_parents_p90": float(np.quantile(np.array(effective_parents), 0.90)),
        "adj_top1_share_mean": float(np.mean(top1_shares)),
        "adj_top2_share_mean": float(np.mean(top2_shares)),
    }


def gt_edge_margin_stats(adj: np.ndarray, gt_edges: set) -> Dict[str, float]:
    gt_forward_weights = []
    gt_reverse_weights = []
    gt_signed_margins = []
    non_gt_weights = []

    for src in range(adj.shape[0]):
        for dst in range(adj.shape[1]):
            if src == dst:
                continue
            weight = float(adj[src, dst])
            if (src, dst) in gt_edges:
                reverse_weight = float(adj[dst, src])
                gt_forward_weights.append(weight)
                gt_reverse_weights.append(reverse_weight)
                gt_signed_margins.append(weight - reverse_weight)
            else:
                non_gt_weights.append(weight)

    if not gt_forward_weights:
        return {
            "gt_forward_weight_mean": 0.0,
            "gt_reverse_weight_mean": 0.0,
            "gt_signed_margin_mean": 0.0,
            "gt_signed_margin_median": 0.0,
            "gt_signed_margin_p10": 0.0,
            "gt_signed_margin_p90": 0.0,
            "gt_signed_margin_frac_pos": 0.0,
            "non_gt_weight_mean": 0.0,
        }

    gt_signed_margins_np = np.array(gt_signed_margins, dtype=float)
    non_gt_weight_mean = float(np.mean(non_gt_weights)) if non_gt_weights else 0.0
    return {
        "gt_forward_weight_mean": float(np.mean(gt_forward_weights)),
        "gt_reverse_weight_mean": float(np.mean(gt_reverse_weights)),
        "gt_signed_margin_mean": float(np.mean(gt_signed_margins_np)),
        "gt_signed_margin_median": float(np.median(gt_signed_margins_np)),
        "gt_signed_margin_p10": float(np.quantile(gt_signed_margins_np, 0.10)),
        "gt_signed_margin_p90": float(np.quantile(gt_signed_margins_np, 0.90)),
        "gt_signed_margin_frac_pos": float(np.mean(gt_signed_margins_np > 0.0)),
        "non_gt_weight_mean": non_gt_weight_mean,
    }


def parse_cross_pred_conditions(text: str) -> List[bool]:
    mapping = {
        "off": False,
        "baseline": False,
        "no": False,
        "0": False,
        "on": True,
        "treatment": True,
        "yes": True,
        "1": True,
    }
    seen = set()
    conditions: List[bool] = []
    for token in text.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(
                f"Unsupported cross-pred condition '{token}'. "
                "Use comma-separated values from: off,on,baseline,treatment."
            )
        value = mapping[key]
        if value in seen:
            continue
        conditions.append(value)
        seen.add(value)
    if not conditions:
        raise ValueError("At least one cross-pred condition must be provided.")
    return conditions


def cross_pred_condition_name(enable_cross_prediction: bool) -> str:
    return "treatment_cross_on" if enable_cross_prediction else "baseline_cross_off"


def parse_directional_conditions(text: str) -> List[Tuple[bool, str, str]]:
    mapping = {
        "off": (False, "disabled", "disabled"),
        "baseline": (False, "disabled", "disabled"),
        "none": (False, "disabled", "disabled"),
        "0": (False, "disabled", "disabled"),
        "patel": (True, "patel", "disabled"),
        "patel_tau": (True, "patel", "disabled"),
        "lag_corr_raw": (True, "lag_corr", "raw"),
        "lag_raw": (True, "lag_corr", "raw"),
        "lag_corr_encoder": (True, "lag_corr", "encoder"),
        "lag_encoder": (True, "lag_corr", "encoder"),
    }
    seen = set()
    conditions: List[Tuple[bool, str, str]] = []
    for token in text.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(
                f"Unsupported directional condition '{token}'. Use comma-separated values "
                "from: off, patel, lag_corr_raw, lag_corr_encoder."
            )
        value = mapping[key]
        if value in seen:
            continue
        conditions.append(value)
        seen.add(value)
    if not conditions:
        raise ValueError("At least one directional condition must be provided.")
    return conditions


def directional_prior_mode_name(enable_directional_loss: bool, prior_mode: str) -> str:
    return prior_mode if enable_directional_loss else "disabled"


def directional_schedule_name(enable_directional_loss: bool, schedule: str) -> str:
    return schedule if enable_directional_loss else "disabled"


def lag_direction_source_name(
    enable_directional_loss: bool,
    prior_mode: str,
    lag_direction_source: str,
) -> str:
    if not enable_directional_loss or prior_mode != "lag_corr":
        return "disabled"
    return lag_direction_source


def directional_condition_name(
    enable_directional_loss: bool,
    prior_mode: str,
    lag_direction_source: str,
) -> str:
    if not enable_directional_loss:
        return "directional_off"
    if prior_mode == "patel":
        return "directional_patel"
    if prior_mode == "lag_corr":
        return f"directional_lag_corr_{lag_direction_source}"
    return f"directional_{prior_mode}"


def as_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean-like value: {value!r}")


def format_failure_counts(counter: Counter, total: int) -> str:
    return ";".join(
        f"{mode}:{counter.get(mode, 0)}/{total}" for mode in FAILURE_MODES if counter.get(mode, 0)
    ) or "none"


def sanitize_tag(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip())
    return clean.strip("-_.")


def load_gt(path: Path) -> set:
    gt = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.replace(",", " ").split()
            if len(parts) < 2:
                continue
            src = int(parts[0]) - 1
            dst = int(parts[1]) - 1
            if src != dst:
                gt.add((src, dst))
    return gt


def directional_predictions(adj: np.ndarray) -> List[Tuple[int, int, float]]:
    preds = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            w_ij = float(adj[i, j])
            w_ji = float(adj[j, i])
            if w_ij >= w_ji:
                src, dst = i, j
            else:
                src, dst = j, i
            preds.append((src, dst, abs(w_ij - w_ji)))
    preds.sort(key=lambda x: x[2], reverse=True)
    return preds


def evaluate_directional(adj: np.ndarray, gt_edges: set) -> Dict[str, object]:
    preds = directional_predictions(adj)
    pred_edges = {(src, dst) for src, dst, _ in preds}
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tie_count = sum(1 for _, _, margin in preds if margin == 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tie_count": tie_count,
        "predictions": preds,
    }


def evaluate_directional_strict(
    adj: np.ndarray,
    gt_edges: set,
    margin_eps: float = 1e-12,
) -> Dict[str, float]:
    pred_edges = set()
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            delta = float(adj[i, j] - adj[j, i])
            if delta > margin_eps:
                pred_edges.add((i, j))
            elif delta < -margin_eps:
                pred_edges.add((j, i))

    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "strict_precision": precision,
        "strict_recall": recall,
        "strict_f1": f1,
        "strict_pred_count": float(len(pred_edges)),
    }


def margin_stats(adj: np.ndarray) -> Dict[str, float]:
    margins = np.abs(adj - adj.T)
    mask = ~np.eye(adj.shape[0], dtype=bool)
    vals = margins[mask]
    if vals.size == 0:
        return {
            "margin_mean": 0.0,
            "margin_median": 0.0,
            "margin_std": 0.0,
            "margin_p90": 0.0,
            "margin_max": 0.0,
            "margin_lt_1e3_frac": 1.0,
            "margin_lt_1e2_frac": 1.0,
        }
    return {
        "margin_mean": float(vals.mean()),
        "margin_median": float(np.median(vals)),
        "margin_std": float(vals.std()),
        "margin_p90": float(np.quantile(vals, 0.90)),
        "margin_max": float(vals.max()),
        "margin_lt_1e3_frac": float(np.mean(vals < 1e-3)),
        "margin_lt_1e2_frac": float(np.mean(vals < 1e-2)),
    }


def failure_mode(stats: Dict[str, float], direction_eval: Dict[str, object]) -> str:
    if stats["margin_p90"] < 1e-3 or stats["margin_lt_1e2_frac"] > 0.95:
        return "symmetric_collapse"
    if direction_eval["f1"] <= 0.2 and stats["margin_median"] > 1e-2:
        return "wrong_direction_asymmetry"
    if direction_eval["f1"] <= 0.4 and stats["margin_p90"] < 5e-2:
        return "weak_asymmetry"
    return "mixed_or_partial"


def tau_alignment(adj: np.ndarray, tau: np.ndarray) -> Tuple[int, int]:
    same = 0
    total = 0
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            pred = (i, j) if adj[i, j] >= adj[j, i] else (j, i)
            ref = (i, j) if tau[i, j] >= tau[j, i] else (j, i)
            same += int(pred == ref)
            total += 1
    return same, total


def evaluate_best_adjacency_metrics(
    adj: np.ndarray,
    gt_edges: set,
    tau: np.ndarray,
    strict_margin_eps_values: Sequence[float],
) -> Dict[str, float]:
    direction_eval = evaluate_directional(adj, gt_edges)
    primary_strict_eval = evaluate_directional_strict(
        adj,
        gt_edges,
        margin_eps=strict_margin_eps_values[0],
    )
    stats = margin_stats(adj)
    adj_concentration = adjacency_parent_concentration_stats(adj)
    gt_margin = gt_edge_margin_stats(adj, gt_edges)
    same_tau, _ = tau_alignment(adj, tau)

    row: Dict[str, float] = {
        "best_precision": float(direction_eval["precision"]),
        "best_recall": float(direction_eval["recall"]),
        "best_f1": float(direction_eval["f1"]),
        "best_strict_precision": float(primary_strict_eval["strict_precision"]),
        "best_strict_recall": float(primary_strict_eval["strict_recall"]),
        "best_strict_f1": float(primary_strict_eval["strict_f1"]),
        "best_strict_pred_count": float(primary_strict_eval["strict_pred_count"]),
        "best_same_dir_vs_tau": float(same_tau),
        "best_tie_count": float(direction_eval["tie_count"]),
        "best_margin_median": float(stats["margin_median"]),
        "best_margin_p90": float(stats["margin_p90"]),
        "best_margin_lt_1e2_frac": float(stats["margin_lt_1e2_frac"]),
        "best_gt_signed_margin_median": float(gt_margin["gt_signed_margin_median"]),
        "best_gt_signed_margin_frac_pos": float(gt_margin["gt_signed_margin_frac_pos"]),
        "best_adj_eff_parents_mean": float(adj_concentration["adj_eff_parents_mean"]),
    }
    for margin_eps in strict_margin_eps_values:
        strict_eval = evaluate_directional_strict(
            adj,
            gt_edges,
            margin_eps=margin_eps,
        )
        for metric_name, metric_value in strict_eval.items():
            row[best_strict_metric_field(metric_name, margin_eps)] = float(metric_value)
    return row


def build_command(
    args: argparse.Namespace,
    seed: int,
    scale: float,
    emb_dim: int,
    structure_parameterization: str,
    fixed_support_mask_mode: str,
    direction_init_mode: str,
    optimizer_step_mode: str,
    adj_activation: str,
    kappa_logit_bias_scale: float,
    direction_logit_bias_scale: float,
    main_loss_weight: float,
    selection_agreement_weight: float,
    direction_lr_multiplier: float,
    freeze_direction_after_epoch: int,
    directional_target_ratio: float,
    directional_loss_end_epoch: int,
    lambda_l1: float,
    parent_entropy_lambda: float,
    parent_cap_lambda: float,
    parent_cap_target: float,
    ungated_symmetry_lambda: float,
    cross_pred_fixed_weight: float,
    enable_cross_prediction: bool,
    enable_directional_loss: bool,
    directional_prior_mode: str,
    lag_direction_source: str,
    directional_prior_scope: str,
) -> List[str]:
    causal_lag_main_weight = resolve_causal_lag_main_weight(
        enable_cross_prediction=enable_cross_prediction,
        cross_pred_fixed_weight=cross_pred_fixed_weight,
        cross_pred_target_ratio=args.cross_pred_target_ratio,
    )
    cmd = [
        sys.executable,
        "main_structure_learning.py",
        "--csv_path",
        args.csv_path,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--pretrain_epochs",
        str(args.pretrain_epochs),
        "--subject_limit",
        str(args.subject_limit),
        "--time_limit",
        str(args.time_limit),
        "--lambda_l1",
        str(lambda_l1),
        "--main_loss_weight",
        str(main_loss_weight),
        "--optimizer_step_mode",
        optimizer_step_mode,
        "--pretrain_checkpoint",
        args.pretrain_checkpoint,
        "--seed",
        str(seed),
        "--log_interval",
        str(args.log_interval),
        "--top_k_edges",
        str(args.top_k_edges),
        "--selection_agreement_weight",
        str(selection_agreement_weight),
        "--structure_init_mode",
        args.structure_init_mode,
        "--structure_init_scale",
        str(scale),
        "--emb_dim",
        str(emb_dim),
        "--structure_parameterization",
        structure_parameterization,
        "--fixed_support_mask_mode",
        fixed_support_mask_mode,
        "--direction_init_mode",
        direction_init_mode,
        "--structure_message_graph_mode",
        args.structure_message_graph_mode,
        "--adj_activation",
        adj_activation,
        "--kappa_logit_bias_scale",
        str(kappa_logit_bias_scale),
        "--direction_logit_bias_scale",
        str(direction_logit_bias_scale),
        "--direction_lr_multiplier",
        str(direction_lr_multiplier),
        "--freeze_direction_after_epoch",
        str(freeze_direction_after_epoch),
        "--parent_entropy_lambda",
        str(parent_entropy_lambda),
        "--parent_entropy_warmup_epochs",
        str(args.parent_entropy_warmup_epochs),
        "--parent_entropy_ramp_epochs",
        str(args.parent_entropy_ramp_epochs),
        "--parent_cap_lambda",
        str(parent_cap_lambda),
        "--parent_cap_target",
        str(parent_cap_target),
        "--parent_cap_warmup_epochs",
        str(args.parent_cap_warmup_epochs),
        "--parent_cap_ramp_epochs",
        str(args.parent_cap_ramp_epochs),
        "--directional_prior_lags",
        args.directional_prior_lags,
        "--causal_lag_main_lags",
        args.cross_pred_lags,
        "--causal_lag_main_weight",
        str(causal_lag_main_weight),
        "--causal_lag_main_aggregation",
        args.cross_pred_aggregation,
        "--causal_lag_main_softmax_temp",
        str(args.cross_pred_softmax_temp),
    ]
    if args.directional_prior_lag_weights.strip():
        cmd.extend(
            [
                "--directional_prior_lag_weights",
                args.directional_prior_lag_weights,
            ]
        )
    if args.cross_pred_lag_weights.strip():
        cmd.extend(
            [
                "--causal_lag_main_lag_weights",
                args.cross_pred_lag_weights,
            ]
        )
    if not enable_directional_loss:
        cmd.append("--disable_directional_loss")
    else:
        cmd.extend(
            [
                "--directional_prior_mode",
                directional_prior_mode,
                "--directional_prior_scope",
                directional_prior_scope,
                "--directional_schedule",
                args.directional_schedule,
                "--directional_target_ratio",
                str(directional_target_ratio),
                "--directional_loss_end_epoch",
                str(directional_loss_end_epoch),
            ]
        )
        if args.directional_kappa_gate:
            cmd.extend(
                [
                    "--directional_kappa_gate",
                    "--directional_kappa_gate_quantile",
                    str(args.directional_kappa_gate_quantile),
                ]
            )
        if directional_prior_mode == "lag_corr":
            cmd.extend(
                [
                    "--lag_direction_source",
                    lag_direction_source,
                ]
            )
        if ungated_symmetry_lambda > 0.0:
            cmd.extend(
                [
                    "--ungated_symmetry_lambda",
                    str(ungated_symmetry_lambda),
                    "--ungated_symmetry_warmup_epochs",
                    str(args.ungated_symmetry_warmup_epochs),
                    "--ungated_symmetry_ramp_epochs",
                    str(args.ungated_symmetry_ramp_epochs),
                ]
            )
    if args.disable_temporal_encoder:
        cmd.append("--disable_temporal_encoder")
    return cmd


def run_single_experiment(
    *,
    args: argparse.Namespace,
    script_dir: Path,
    gt_edges: set,
    strict_margin_eps_values: Sequence[float],
    condition: str,
    enable_cross_prediction: bool,
    enable_directional_loss: bool,
    directional_prior_mode: str,
    lag_direction_source: str,
    directional_prior_scope: str,
    parent_entropy_lambda: float,
    parent_cap_lambda: float,
    parent_cap_target: float,
    ungated_symmetry_lambda: float,
    cross_pred_fixed_weight: float,
    directional_target_ratio: float,
    directional_loss_end_epoch: int,
    lambda_l1: float,
    structure_parameterization: str,
    fixed_support_mask_mode: str,
    direction_init_mode: str,
    emb_dim: int,
    optimizer_step_mode: str,
    adj_activation: str,
    kappa_logit_bias_scale: float,
    direction_logit_bias_scale: float,
    main_loss_weight: float,
    selection_agreement_weight: float,
    direction_lr_multiplier: float,
    freeze_direction_after_epoch: int,
    scale: float,
    seed: int,
) -> Dict[str, object]:
    cmd = build_command(
        args,
        seed=seed,
        scale=scale,
        emb_dim=emb_dim,
        structure_parameterization=structure_parameterization,
        fixed_support_mask_mode=fixed_support_mask_mode,
        direction_init_mode=direction_init_mode,
        optimizer_step_mode=optimizer_step_mode,
        adj_activation=adj_activation,
        kappa_logit_bias_scale=kappa_logit_bias_scale,
        direction_logit_bias_scale=direction_logit_bias_scale,
        main_loss_weight=main_loss_weight,
        selection_agreement_weight=selection_agreement_weight,
        direction_lr_multiplier=direction_lr_multiplier,
        freeze_direction_after_epoch=freeze_direction_after_epoch,
        directional_target_ratio=directional_target_ratio,
        directional_loss_end_epoch=directional_loss_end_epoch,
        lambda_l1=lambda_l1,
        parent_entropy_lambda=parent_entropy_lambda,
        parent_cap_lambda=parent_cap_lambda,
        parent_cap_target=parent_cap_target,
        ungated_symmetry_lambda=ungated_symmetry_lambda,
        cross_pred_fixed_weight=cross_pred_fixed_weight,
        enable_cross_prediction=enable_cross_prediction,
        enable_directional_loss=enable_directional_loss,
        directional_prior_mode=directional_prior_mode,
        lag_direction_source=lag_direction_source,
        directional_prior_scope=directional_prior_scope,
    )
    proc = subprocess.run(
        cmd,
        cwd=str(script_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Training subprocess failed for "
            f"condition={condition}, lambda_l1={lambda_l1}, "
            f"parent_entropy={parent_entropy_lambda}, "
            f"parent_cap={parent_cap_lambda}@{parent_cap_target}, "
            f"dir_ratio={directional_target_ratio}, "
            f"struct={structure_parameterization}, "
            f"support_mask={fixed_support_mask_mode}, "
            f"dir_init={direction_init_mode}, "
            f"emb_dim={emb_dim}, msg_mode={args.structure_message_graph_mode}, "
            f"opt_step={optimizer_step_mode}, "
            f"adj_act={adj_activation}, "
            f"kappa_bias={kappa_logit_bias_scale}, "
            f"tau_bias={direction_logit_bias_scale}, "
            f"main_w={main_loss_weight}, "
            f"sel_agree={selection_agreement_weight}, "
            f"ungated_sym={ungated_symmetry_lambda}, "
            f"seed={seed}, scale={scale}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    stdout = proc.stdout
    match = re.search(r"Results will be saved to: (.+)", stdout)
    if not match:
        raise RuntimeError(
            "Could not parse result dir for "
            f"condition={condition}, lambda_l1={lambda_l1}, "
            f"parent_entropy={parent_entropy_lambda}, "
            f"parent_cap={parent_cap_lambda}@{parent_cap_target}, "
            f"dir_ratio={directional_target_ratio}, "
            f"struct={structure_parameterization}, "
            f"support_mask={fixed_support_mask_mode}, "
            f"dir_init={direction_init_mode}, "
            f"emb_dim={emb_dim}, msg_mode={args.structure_message_graph_mode}, "
            f"opt_step={optimizer_step_mode}, "
            f"adj_act={adj_activation}, "
            f"kappa_bias={kappa_logit_bias_scale}, "
            f"tau_bias={direction_logit_bias_scale}, "
            f"main_w={main_loss_weight}, "
            f"sel_agree={selection_agreement_weight}, "
            f"ungated_sym={ungated_symmetry_lambda}, "
            f"seed={seed}, scale={scale}"
        )
    result_dir = (script_dir / match.group(1).strip()).resolve()
    (result_dir / "cross_pred_v1_final_only_compare_stdout.log").write_text(
        stdout,
        encoding="utf-8",
    )
    if proc.stderr:
        (result_dir / "cross_pred_v1_final_only_compare_stderr.log").write_text(
            proc.stderr,
            encoding="utf-8",
        )

    best_adj = np.loadtxt(result_dir / "learned_adjacency_causal.csv", delimiter=",")
    final_adj = np.loadtxt(result_dir / "final_epoch_adjacency_causal.csv", delimiter=",")
    tau = np.loadtxt(result_dir / "patel_tau.csv", delimiter=",")
    final_diff_loss = load_final_diff_loss(result_dir)
    best_metrics = evaluate_best_adjacency_metrics(
        best_adj,
        gt_edges,
        tau,
        strict_margin_eps_values,
    )
    direction_eval = evaluate_directional(final_adj, gt_edges)
    primary_strict_eval = evaluate_directional_strict(
        final_adj,
        gt_edges,
        margin_eps=strict_margin_eps_values[0],
    )
    extra_strict_eval: Dict[str, float] = {}
    for margin_eps in strict_margin_eps_values:
        strict_eval = evaluate_directional_strict(
            final_adj,
            gt_edges,
            margin_eps=margin_eps,
        )
        for metric_name, metric_value in strict_eval.items():
            extra_strict_eval[strict_metric_field(metric_name, margin_eps)] = metric_value
    stats = margin_stats(final_adj)
    adj_density = adjacency_density_stats(final_adj)
    adj_concentration = adjacency_parent_concentration_stats(final_adj)
    gt_margin = gt_edge_margin_stats(final_adj, gt_edges)
    same_tau, total_pairs = tau_alignment(final_adj, tau)
    best_final_gap_metrics = {
        "best_final_gap_gt_signed_margin_median": (
            float(best_metrics["best_gt_signed_margin_median"]) -
            float(gt_margin["gt_signed_margin_median"])
        ),
    }
    for margin_eps in strict_margin_eps_values:
        best_final_gap_metrics[best_final_gap_metric_field("strict_f1", margin_eps)] = (
            float(best_metrics[best_strict_metric_field("strict_f1", margin_eps)]) -
            float(extra_strict_eval[strict_metric_field("strict_f1", margin_eps)])
        )

    return {
        "condition": condition,
        "enable_cross_prediction": int(enable_cross_prediction),
        "cross_pred_target_ratio": args.cross_pred_target_ratio if enable_cross_prediction else 0.0,
        "cross_pred_schedule": cross_pred_schedule_name(enable_cross_prediction, args.cross_pred_schedule),
        "cross_pred_aggregation": cross_pred_aggregation_name(enable_cross_prediction, args.cross_pred_aggregation),
        "cross_pred_softmax_temp": args.cross_pred_softmax_temp if enable_cross_prediction else 0.0,
        "cross_pred_lags": args.cross_pred_lags if enable_cross_prediction else "disabled",
        "cross_pred_lag_weights": args.cross_pred_lag_weights if enable_cross_prediction else "disabled",
        "cross_pred_fixed_weight": cross_pred_fixed_weight if enable_cross_prediction else 0.0,
        "cross_pred_fixed_warmup_epochs": args.cross_pred_fixed_warmup_epochs if enable_cross_prediction else 0,
        "cross_pred_fixed_ramp_epochs": args.cross_pred_fixed_ramp_epochs if enable_cross_prediction else 1,
        "enable_directional_loss": int(enable_directional_loss),
        "directional_prior_mode": directional_prior_mode_name(
            enable_directional_loss,
            directional_prior_mode,
        ),
        "directional_prior_scope": (
            directional_prior_scope
            if enable_directional_loss and directional_prior_mode == "lag_corr"
            else "disabled"
        ),
        "directional_schedule": directional_schedule_name(
            enable_directional_loss,
            args.directional_schedule,
        ),
        "lag_direction_source": lag_direction_source_name(
            enable_directional_loss,
            directional_prior_mode,
            lag_direction_source,
        ),
        "directional_prior_lags": args.directional_prior_lags if enable_directional_loss else "disabled",
        "directional_prior_lag_weights": args.directional_prior_lag_weights if enable_directional_loss else "disabled",
        "directional_kappa_gate": int(enable_directional_loss and args.directional_kappa_gate),
        "directional_kappa_gate_quantile": (
            args.directional_kappa_gate_quantile
            if enable_directional_loss and args.directional_kappa_gate
            else 0.0
        ),
        "directional_target_ratio": directional_target_ratio if enable_directional_loss else 0.0,
        "directional_loss_end_epoch": directional_loss_end_epoch if enable_directional_loss else -1,
        "parent_entropy_lambda": parent_entropy_lambda,
        "parent_entropy_warmup_epochs": args.parent_entropy_warmup_epochs,
        "parent_entropy_ramp_epochs": args.parent_entropy_ramp_epochs,
        "parent_cap_lambda": parent_cap_lambda,
        "parent_cap_target": parent_cap_target,
        "parent_cap_warmup_epochs": args.parent_cap_warmup_epochs,
        "parent_cap_ramp_epochs": args.parent_cap_ramp_epochs,
        "ungated_symmetry_lambda": ungated_symmetry_lambda,
        "ungated_symmetry_warmup_epochs": args.ungated_symmetry_warmup_epochs,
        "ungated_symmetry_ramp_epochs": args.ungated_symmetry_ramp_epochs,
        "strict_margin_eps_values": ",".join(
            f"{normalize_margin_eps(v):g}" for v in strict_margin_eps_values
        ),
        "strict_primary_margin_eps": strict_margin_eps_values[0],
        "structure_parameterization": structure_parameterization,
        "fixed_support_mask_mode": fixed_support_mask_mode,
        "direction_init_mode": direction_init_mode,
        "emb_dim": emb_dim,
        "structure_message_graph_mode": args.structure_message_graph_mode,
        "optimizer_step_mode": optimizer_step_mode,
        "adj_activation": adj_activation,
        "kappa_logit_bias_scale": kappa_logit_bias_scale,
        "direction_logit_bias_scale": direction_logit_bias_scale,
        "main_loss_weight": main_loss_weight,
        "selection_agreement_weight": selection_agreement_weight,
        "direction_lr_multiplier": direction_lr_multiplier,
        "freeze_direction_after_epoch": freeze_direction_after_epoch,
        "subject_limit": args.subject_limit,
        "time_limit": args.time_limit,
        "lambda_l1": lambda_l1,
        "structure_init_mode": args.structure_init_mode,
        "structure_init_scale": scale,
        "seed": seed,
        "result_dir": str(result_dir),
        "final_precision": float(direction_eval["precision"]),
        "final_recall": float(direction_eval["recall"]),
        "final_f1": float(direction_eval["f1"]),
        **primary_strict_eval,
        **extra_strict_eval,
        "final_diff_loss": final_diff_loss,
        "final_tie_count": int(direction_eval["tie_count"]),
        "final_same_dir_vs_tau": int(same_tau),
        "total_pairs": int(total_pairs),
        **best_metrics,
        **stats,
        **adj_density,
        **adj_concentration,
        **gt_margin,
        **best_final_gap_metrics,
        "failure_mode": failure_mode(stats, direction_eval),
        "final_top5": str(
            [
                (src + 1, dst + 1, round(margin, 4))
                for src, dst, margin in direction_eval["predictions"][:5]
            ]
        ),
    }


def aggregate_rows(
    rows: Sequence[Dict[str, object]],
    aggregate_metric_names: Sequence[str],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            row["condition"],
            row["enable_cross_prediction"],
            row["cross_pred_target_ratio"],
            row["cross_pred_schedule"],
            row["cross_pred_aggregation"],
            row["cross_pred_softmax_temp"],
            row.get("cross_pred_lags", "disabled"),
            row.get("cross_pred_lag_weights", "disabled"),
            row.get("cross_pred_fixed_weight", 0.0),
            row.get("cross_pred_fixed_warmup_epochs", 0),
            row.get("cross_pred_fixed_ramp_epochs", 1),
            row["enable_directional_loss"],
            row["directional_prior_mode"],
            row.get("directional_prior_scope", "disabled"),
            row["directional_schedule"],
            row["lag_direction_source"],
            row.get("directional_prior_lags", "disabled"),
            row.get("directional_prior_lag_weights", "disabled"),
            row.get("structure_message_graph_mode", "raw"),
            row["directional_kappa_gate"],
            row["directional_kappa_gate_quantile"],
            row.get("directional_target_ratio", 0.0),
            row.get("directional_loss_end_epoch", -1),
            row["parent_entropy_lambda"],
            row["parent_entropy_warmup_epochs"],
            row["parent_entropy_ramp_epochs"],
            row["parent_cap_lambda"],
            row["parent_cap_target"],
            row["parent_cap_warmup_epochs"],
            row["parent_cap_ramp_epochs"],
            row.get("ungated_symmetry_lambda", 0.0),
            row.get("ungated_symmetry_warmup_epochs", 0),
            row.get("ungated_symmetry_ramp_epochs", 1),
            row.get("structure_parameterization", "coupled"),
            row.get("fixed_support_mask_mode", "none"),
            row.get("direction_init_mode", "patel_tau"),
            row.get("emb_dim", 0),
            row.get("optimizer_step_mode", "subject"),
            row.get("adj_activation", "sigmoid"),
            row["lambda_l1"],
            row["structure_init_mode"],
            row["structure_init_scale"],
            row.get("kappa_logit_bias_scale", 0.0),
            row.get("direction_logit_bias_scale", 0.0),
            row.get("main_loss_weight", 1.0),
            row.get("selection_agreement_weight", 0.0),
            row.get("direction_lr_multiplier", 1.0),
            row.get("freeze_direction_after_epoch", -1),
            row.get("subject_limit", -1),
            row.get("time_limit", -1),
        )
        grouped.setdefault(key, []).append(row)

    aggregates: List[Dict[str, object]] = []
    for key in sorted(grouped.keys()):
        group_rows = grouped[key]
        sample = group_rows[0]
        count = len(group_rows)
        failure_counter = Counter(str(r["failure_mode"]) for r in group_rows)
        aggregate_row: Dict[str, object] = {
            "condition": sample["condition"],
            "enable_cross_prediction": sample["enable_cross_prediction"],
            "cross_pred_target_ratio": sample["cross_pred_target_ratio"],
            "cross_pred_schedule": sample["cross_pred_schedule"],
            "cross_pred_aggregation": sample["cross_pred_aggregation"],
            "cross_pred_softmax_temp": sample["cross_pred_softmax_temp"],
            "cross_pred_lags": sample.get("cross_pred_lags", "disabled"),
            "cross_pred_lag_weights": sample.get("cross_pred_lag_weights", "disabled"),
            "cross_pred_fixed_weight": sample.get("cross_pred_fixed_weight", 0.0),
            "cross_pred_fixed_warmup_epochs": sample.get("cross_pred_fixed_warmup_epochs", 0),
            "cross_pred_fixed_ramp_epochs": sample.get("cross_pred_fixed_ramp_epochs", 1),
            "enable_directional_loss": sample["enable_directional_loss"],
            "directional_prior_mode": sample["directional_prior_mode"],
            "directional_prior_scope": sample.get("directional_prior_scope", "disabled"),
            "directional_schedule": sample["directional_schedule"],
            "lag_direction_source": sample["lag_direction_source"],
            "directional_prior_lags": sample.get("directional_prior_lags", "disabled"),
            "directional_prior_lag_weights": sample.get("directional_prior_lag_weights", "disabled"),
            "structure_message_graph_mode": sample.get("structure_message_graph_mode", "raw"),
            "directional_kappa_gate": sample["directional_kappa_gate"],
            "directional_kappa_gate_quantile": sample["directional_kappa_gate_quantile"],
            "directional_target_ratio": sample.get("directional_target_ratio", 0.0),
            "directional_loss_end_epoch": sample.get("directional_loss_end_epoch", -1),
            "parent_entropy_lambda": sample["parent_entropy_lambda"],
            "parent_entropy_warmup_epochs": sample["parent_entropy_warmup_epochs"],
            "parent_entropy_ramp_epochs": sample["parent_entropy_ramp_epochs"],
            "parent_cap_lambda": sample["parent_cap_lambda"],
            "parent_cap_target": sample["parent_cap_target"],
            "parent_cap_warmup_epochs": sample["parent_cap_warmup_epochs"],
            "parent_cap_ramp_epochs": sample["parent_cap_ramp_epochs"],
            "ungated_symmetry_lambda": sample.get("ungated_symmetry_lambda", 0.0),
            "ungated_symmetry_warmup_epochs": sample.get("ungated_symmetry_warmup_epochs", 0),
            "ungated_symmetry_ramp_epochs": sample.get("ungated_symmetry_ramp_epochs", 1),
            "strict_margin_eps_values": sample.get("strict_margin_eps_values", ""),
            "strict_primary_margin_eps": sample.get("strict_primary_margin_eps", 0.0),
            "structure_parameterization": sample.get("structure_parameterization", "coupled"),
            "fixed_support_mask_mode": sample.get("fixed_support_mask_mode", "none"),
            "direction_init_mode": sample.get("direction_init_mode", "patel_tau"),
            "emb_dim": sample.get("emb_dim", 0),
            "optimizer_step_mode": sample.get("optimizer_step_mode", "subject"),
            "adj_activation": sample.get("adj_activation", "sigmoid"),
            "lambda_l1": sample["lambda_l1"],
            "structure_init_mode": sample["structure_init_mode"],
            "structure_init_scale": sample["structure_init_scale"],
            "kappa_logit_bias_scale": sample.get("kappa_logit_bias_scale", 0.0),
            "direction_logit_bias_scale": sample.get("direction_logit_bias_scale", 0.0),
            "main_loss_weight": sample.get("main_loss_weight", 1.0),
            "selection_agreement_weight": sample.get("selection_agreement_weight", 0.0),
            "direction_lr_multiplier": sample.get("direction_lr_multiplier", 1.0),
            "freeze_direction_after_epoch": sample.get("freeze_direction_after_epoch", -1),
            "subject_limit": sample.get("subject_limit", -1),
            "time_limit": sample.get("time_limit", -1),
            "run_count": count,
            "seed_list": ",".join(str(r["seed"]) for r in group_rows),
            "failure_mode_counts": format_failure_counts(failure_counter, count),
        }
        for metric in aggregate_metric_names:
            values = np.array([float(r[metric]) for r in group_rows], dtype=float)
            aggregate_row[f"{metric}_mean"] = float(values.mean())
            aggregate_row[f"{metric}_std"] = float(values.std())
        for mode in FAILURE_MODES:
            mode_count = failure_counter.get(mode, 0)
            aggregate_row[f"{mode}_count"] = int(mode_count)
            aggregate_row[f"{mode}_frac"] = float(mode_count / count)
        aggregates.append(aggregate_row)
    return aggregates


def baseline_match_key(row: Dict[str, object], *, include_seed: bool = False) -> Tuple[object, ...]:
    key: List[object] = [
        row["structure_init_mode"],
        row["structure_init_scale"],
    ]
    if include_seed:
        key.append(row["seed"])
    key.extend(
        [
            row["lambda_l1"],
            row.get("structure_parameterization", "coupled"),
            row.get("fixed_support_mask_mode", "none"),
            row.get("direction_init_mode", "patel_tau"),
            row.get("emb_dim", 0),
            row.get("structure_message_graph_mode", "raw"),
            row.get("optimizer_step_mode", "subject"),
            row.get("adj_activation", "sigmoid"),
            row.get("kappa_logit_bias_scale", 0.0),
            row.get("direction_logit_bias_scale", 0.0),
            row.get("main_loss_weight", 1.0),
            row.get("selection_agreement_weight", 0.0),
            row.get("direction_lr_multiplier", 1.0),
            row.get("freeze_direction_after_epoch", -1),
            row.get("subject_limit", -1),
            row.get("time_limit", -1),
        ]
    )
    return tuple(key)


def build_condition_deltas(
    aggregate_rows_data: Sequence[Dict[str, object]],
    delta_metric_names: Sequence[str],
) -> List[Dict[str, object]]:
    baseline_by_key: Dict[Tuple[object, ...], Dict[str, object]] = {}
    treatments_by_key: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in aggregate_rows_data:
        base_key = baseline_match_key(row)
        if (
            not as_bool_flag(row["enable_cross_prediction"]) and
            not as_bool_flag(row["enable_directional_loss"]) and
            float(row.get("parent_entropy_lambda", 0.0)) == 0.0 and
            float(row.get("parent_cap_lambda", 0.0)) == 0.0 and
            float(row.get("ungated_symmetry_lambda", 0.0)) == 0.0
        ):
            baseline_by_key[base_key] = row
            continue
        treatment_key = (
            row["structure_init_mode"],
            row["structure_init_scale"],
            row["lambda_l1"],
            row.get("enable_cross_prediction", 0),
            row.get("cross_pred_target_ratio", 0.0),
            row.get("cross_pred_schedule", "disabled"),
            row.get("cross_pred_aggregation", "disabled"),
            row.get("cross_pred_softmax_temp", 0.0),
            row.get("cross_pred_lags", "disabled"),
            row.get("cross_pred_lag_weights", "disabled"),
            row.get("cross_pred_fixed_weight", 0.0),
            row.get("cross_pred_fixed_warmup_epochs", 0),
            row.get("cross_pred_fixed_ramp_epochs", 1),
            row.get("enable_directional_loss", 0),
            row.get("directional_prior_mode", "disabled"),
            row.get("directional_prior_scope", "disabled"),
            row.get("directional_schedule", "disabled"),
            row.get("lag_direction_source", "disabled"),
            row.get("directional_prior_lags", "disabled"),
            row.get("directional_prior_lag_weights", "disabled"),
            row.get("structure_message_graph_mode", "raw"),
            row.get("directional_kappa_gate", 0),
            row.get("directional_kappa_gate_quantile", 0.0),
            row.get("directional_target_ratio", 0.0),
            row.get("directional_loss_end_epoch", -1),
            row.get("parent_entropy_lambda", 0.0),
            row.get("parent_entropy_warmup_epochs", 0),
            row.get("parent_entropy_ramp_epochs", 1),
            row.get("parent_cap_lambda", 0.0),
            row.get("parent_cap_target", 0.0),
            row.get("parent_cap_warmup_epochs", 0),
            row.get("parent_cap_ramp_epochs", 1),
            row.get("ungated_symmetry_lambda", 0.0),
            row.get("ungated_symmetry_warmup_epochs", 0),
            row.get("ungated_symmetry_ramp_epochs", 1),
            row.get("emb_dim", 0),
            row.get("optimizer_step_mode", "subject"),
            row.get("adj_activation", "sigmoid"),
            row.get("kappa_logit_bias_scale", 0.0),
            row.get("direction_logit_bias_scale", 0.0),
            row.get("main_loss_weight", 1.0),
            row.get("selection_agreement_weight", 0.0),
            row.get("direction_lr_multiplier", 1.0),
            row.get("freeze_direction_after_epoch", -1),
            row.get("subject_limit", -1),
            row.get("time_limit", -1),
        )
        treatments_by_key.setdefault(treatment_key, []).append(row)

    delta_rows: List[Dict[str, object]] = []
    for key in sorted(treatments_by_key.keys()):
        treatment = treatments_by_key[key][0]
        base_key = baseline_match_key(treatment)
        baseline = baseline_by_key.get(base_key)
        if baseline is None:
            continue
        delta_row: Dict[str, object] = {
            "structure_init_mode": baseline["structure_init_mode"],
            "structure_init_scale": baseline["structure_init_scale"],
            "lambda_l1": baseline["lambda_l1"],
            "structure_parameterization": baseline.get("structure_parameterization", "coupled"),
            "fixed_support_mask_mode": baseline.get("fixed_support_mask_mode", "none"),
            "direction_init_mode": baseline.get("direction_init_mode", "patel_tau"),
            "emb_dim": baseline.get("emb_dim", 0),
            "structure_message_graph_mode": baseline.get("structure_message_graph_mode", "raw"),
            "optimizer_step_mode": baseline.get("optimizer_step_mode", "subject"),
            "adj_activation": baseline.get("adj_activation", "sigmoid"),
            "kappa_logit_bias_scale": baseline.get("kappa_logit_bias_scale", 0.0),
            "direction_logit_bias_scale": baseline.get("direction_logit_bias_scale", 0.0),
            "main_loss_weight": baseline.get("main_loss_weight", 1.0),
            "selection_agreement_weight": baseline.get("selection_agreement_weight", 0.0),
            "direction_lr_multiplier": baseline.get("direction_lr_multiplier", 1.0),
            "freeze_direction_after_epoch": baseline.get("freeze_direction_after_epoch", -1),
            "directional_prior_lags": baseline.get("directional_prior_lags", "disabled"),
            "directional_prior_lag_weights": baseline.get("directional_prior_lag_weights", "disabled"),
            "cross_pred_lags": baseline.get("cross_pred_lags", "disabled"),
            "cross_pred_lag_weights": baseline.get("cross_pred_lag_weights", "disabled"),
            "cross_pred_fixed_weight": baseline.get("cross_pred_fixed_weight", 0.0),
            "subject_limit": baseline.get("subject_limit", -1),
            "time_limit": baseline.get("time_limit", -1),
            "baseline_runs": baseline["run_count"],
            "treatment_runs": treatment["run_count"],
            "treatment_enable_cross_prediction": treatment["enable_cross_prediction"],
            "treatment_cross_pred_target_ratio": treatment["cross_pred_target_ratio"],
            "treatment_cross_pred_schedule": treatment["cross_pred_schedule"],
            "treatment_cross_pred_aggregation": treatment.get("cross_pred_aggregation", "disabled"),
            "treatment_cross_pred_softmax_temp": treatment.get("cross_pred_softmax_temp", 0.0),
            "treatment_cross_pred_lags": treatment.get("cross_pred_lags", "disabled"),
            "treatment_cross_pred_lag_weights": treatment.get("cross_pred_lag_weights", "disabled"),
            "treatment_cross_pred_fixed_weight": treatment.get("cross_pred_fixed_weight", 0.0),
            "treatment_enable_directional_loss": treatment["enable_directional_loss"],
            "treatment_directional_prior_mode": treatment["directional_prior_mode"],
            "treatment_directional_prior_scope": treatment.get("directional_prior_scope", "disabled"),
            "treatment_directional_schedule": treatment.get("directional_schedule", "disabled"),
            "treatment_lag_direction_source": treatment["lag_direction_source"],
            "treatment_directional_prior_lags": treatment.get("directional_prior_lags", "disabled"),
            "treatment_directional_prior_lag_weights": treatment.get("directional_prior_lag_weights", "disabled"),
            "treatment_structure_message_graph_mode": treatment.get("structure_message_graph_mode", "raw"),
            "treatment_directional_kappa_gate": treatment.get("directional_kappa_gate", 0),
            "treatment_directional_kappa_gate_quantile": treatment.get("directional_kappa_gate_quantile", 0.0),
            "treatment_directional_target_ratio": treatment.get("directional_target_ratio", 0.0),
            "treatment_directional_loss_end_epoch": treatment.get("directional_loss_end_epoch", -1),
            "treatment_parent_entropy_lambda": treatment.get("parent_entropy_lambda", 0.0),
            "treatment_parent_entropy_warmup_epochs": treatment.get("parent_entropy_warmup_epochs", 0),
            "treatment_parent_entropy_ramp_epochs": treatment.get("parent_entropy_ramp_epochs", 1),
            "treatment_parent_cap_lambda": treatment.get("parent_cap_lambda", 0.0),
            "treatment_parent_cap_target": treatment.get("parent_cap_target", 0.0),
            "treatment_parent_cap_warmup_epochs": treatment.get("parent_cap_warmup_epochs", 0),
            "treatment_parent_cap_ramp_epochs": treatment.get("parent_cap_ramp_epochs", 1),
            "treatment_ungated_symmetry_lambda": treatment.get("ungated_symmetry_lambda", 0.0),
            "treatment_ungated_symmetry_warmup_epochs": treatment.get("ungated_symmetry_warmup_epochs", 0),
            "treatment_ungated_symmetry_ramp_epochs": treatment.get("ungated_symmetry_ramp_epochs", 1),
            "treatment_structure_parameterization": treatment.get("structure_parameterization", "coupled"),
            "treatment_fixed_support_mask_mode": treatment.get("fixed_support_mask_mode", "none"),
            "treatment_direction_init_mode": treatment.get("direction_init_mode", "patel_tau"),
            "treatment_emb_dim": treatment.get("emb_dim", 0),
            "treatment_optimizer_step_mode": treatment.get("optimizer_step_mode", "subject"),
            "treatment_adj_activation": treatment.get("adj_activation", "sigmoid"),
            "treatment_kappa_logit_bias_scale": treatment.get("kappa_logit_bias_scale", 0.0),
            "treatment_direction_logit_bias_scale": treatment.get("direction_logit_bias_scale", 0.0),
            "treatment_main_loss_weight": treatment.get("main_loss_weight", 1.0),
            "treatment_selection_agreement_weight": treatment.get("selection_agreement_weight", 0.0),
            "treatment_direction_lr_multiplier": treatment.get("direction_lr_multiplier", 1.0),
            "treatment_freeze_direction_after_epoch": treatment.get("freeze_direction_after_epoch", -1),
            "treatment_subject_limit": treatment.get("subject_limit", -1),
            "treatment_time_limit": treatment.get("time_limit", -1),
            "baseline_failure_mode_counts": baseline["failure_mode_counts"],
            "treatment_failure_mode_counts": treatment["failure_mode_counts"],
        }
        for metric in delta_metric_names:
            baseline_mean = float(baseline[f"{metric}_mean"])
            treatment_mean = float(treatment[f"{metric}_mean"])
            delta_row[f"{metric}_baseline_mean"] = baseline_mean
            delta_row[f"{metric}_treatment_mean"] = treatment_mean
            delta_row[f"{metric}_delta_treat_minus_base"] = treatment_mean - baseline_mean
        for mode in FAILURE_MODES:
            baseline_frac = float(baseline[f"{mode}_frac"])
            treatment_frac = float(treatment[f"{mode}_frac"])
            delta_row[f"{mode}_baseline_frac"] = baseline_frac
            delta_row[f"{mode}_treatment_frac"] = treatment_frac
            delta_row[f"{mode}_frac_delta_treat_minus_base"] = treatment_frac - baseline_frac
        delta_rows.append(delta_row)
    return delta_rows


def build_paired_seed_deltas(
    rows: Sequence[Dict[str, object]],
    delta_metric_names: Sequence[str],
) -> List[Dict[str, object]]:
    baseline_by_key: Dict[Tuple[object, ...], Dict[str, object]] = {}
    treatments_by_key: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        base_key = baseline_match_key(row, include_seed=True)
        if (
            not as_bool_flag(row["enable_cross_prediction"]) and
            not as_bool_flag(row["enable_directional_loss"]) and
            float(row.get("parent_entropy_lambda", 0.0)) == 0.0 and
            float(row.get("parent_cap_lambda", 0.0)) == 0.0 and
            float(row.get("ungated_symmetry_lambda", 0.0)) == 0.0
        ):
            baseline_by_key[base_key] = row
            continue
        treatment_key = (
            row["structure_init_mode"],
            row["structure_init_scale"],
            row["seed"],
            row["lambda_l1"],
            row.get("enable_cross_prediction", 0),
            row.get("cross_pred_target_ratio", 0.0),
            row.get("cross_pred_schedule", "disabled"),
            row.get("cross_pred_aggregation", "disabled"),
            row.get("cross_pred_softmax_temp", 0.0),
            row.get("cross_pred_lags", "disabled"),
            row.get("cross_pred_lag_weights", "disabled"),
            row.get("cross_pred_fixed_weight", 0.0),
            row.get("cross_pred_fixed_warmup_epochs", 0),
            row.get("cross_pred_fixed_ramp_epochs", 1),
            row.get("enable_directional_loss", 0),
            row.get("directional_prior_mode", "disabled"),
            row.get("directional_prior_scope", "disabled"),
            row.get("directional_schedule", "disabled"),
            row.get("lag_direction_source", "disabled"),
            row.get("directional_prior_lags", "disabled"),
            row.get("directional_prior_lag_weights", "disabled"),
            row.get("structure_message_graph_mode", "raw"),
            row.get("directional_kappa_gate", 0),
            row.get("directional_kappa_gate_quantile", 0.0),
            row.get("directional_target_ratio", 0.0),
            row.get("directional_loss_end_epoch", -1),
            row.get("parent_entropy_lambda", 0.0),
            row.get("parent_entropy_warmup_epochs", 0),
            row.get("parent_entropy_ramp_epochs", 1),
            row.get("parent_cap_lambda", 0.0),
            row.get("parent_cap_target", 0.0),
            row.get("parent_cap_warmup_epochs", 0),
            row.get("parent_cap_ramp_epochs", 1),
            row.get("ungated_symmetry_lambda", 0.0),
            row.get("ungated_symmetry_warmup_epochs", 0),
            row.get("ungated_symmetry_ramp_epochs", 1),
            row.get("emb_dim", 0),
            row.get("optimizer_step_mode", "subject"),
            row.get("adj_activation", "sigmoid"),
            row.get("kappa_logit_bias_scale", 0.0),
            row.get("direction_logit_bias_scale", 0.0),
            row.get("main_loss_weight", 1.0),
            row.get("selection_agreement_weight", 0.0),
            row.get("direction_lr_multiplier", 1.0),
            row.get("freeze_direction_after_epoch", -1),
            row.get("subject_limit", -1),
            row.get("time_limit", -1),
        )
        treatments_by_key.setdefault(treatment_key, []).append(row)

    paired_rows: List[Dict[str, object]] = []
    for key in sorted(treatments_by_key.keys()):
        treatment = treatments_by_key[key][0]
        base_key = baseline_match_key(treatment, include_seed=True)
        baseline = baseline_by_key.get(base_key)
        if baseline is None:
            continue
        paired_row: Dict[str, object] = {
            "structure_init_mode": baseline["structure_init_mode"],
            "structure_init_scale": baseline["structure_init_scale"],
            "seed": baseline["seed"],
            "lambda_l1": baseline["lambda_l1"],
            "structure_parameterization": baseline.get("structure_parameterization", "coupled"),
            "fixed_support_mask_mode": baseline.get("fixed_support_mask_mode", "none"),
            "direction_init_mode": baseline.get("direction_init_mode", "patel_tau"),
            "emb_dim": baseline.get("emb_dim", 0),
            "structure_message_graph_mode": baseline.get("structure_message_graph_mode", "raw"),
            "optimizer_step_mode": baseline.get("optimizer_step_mode", "subject"),
            "adj_activation": baseline.get("adj_activation", "sigmoid"),
            "kappa_logit_bias_scale": baseline.get("kappa_logit_bias_scale", 0.0),
            "direction_logit_bias_scale": baseline.get("direction_logit_bias_scale", 0.0),
            "main_loss_weight": baseline.get("main_loss_weight", 1.0),
            "selection_agreement_weight": baseline.get("selection_agreement_weight", 0.0),
            "direction_lr_multiplier": baseline.get("direction_lr_multiplier", 1.0),
            "freeze_direction_after_epoch": baseline.get("freeze_direction_after_epoch", -1),
            "directional_prior_lags": baseline.get("directional_prior_lags", "disabled"),
            "directional_prior_lag_weights": baseline.get("directional_prior_lag_weights", "disabled"),
            "cross_pred_lags": baseline.get("cross_pred_lags", "disabled"),
            "cross_pred_lag_weights": baseline.get("cross_pred_lag_weights", "disabled"),
            "cross_pred_fixed_weight": baseline.get("cross_pred_fixed_weight", 0.0),
            "subject_limit": baseline.get("subject_limit", -1),
            "time_limit": baseline.get("time_limit", -1),
            "treatment_enable_cross_prediction": treatment["enable_cross_prediction"],
            "treatment_cross_pred_target_ratio": treatment["cross_pred_target_ratio"],
            "treatment_cross_pred_schedule": treatment["cross_pred_schedule"],
            "treatment_cross_pred_aggregation": treatment.get("cross_pred_aggregation", "disabled"),
            "treatment_cross_pred_softmax_temp": treatment.get("cross_pred_softmax_temp", 0.0),
            "treatment_cross_pred_lags": treatment.get("cross_pred_lags", "disabled"),
            "treatment_cross_pred_lag_weights": treatment.get("cross_pred_lag_weights", "disabled"),
            "treatment_cross_pred_fixed_weight": treatment.get("cross_pred_fixed_weight", 0.0),
            "treatment_enable_directional_loss": treatment["enable_directional_loss"],
            "treatment_directional_prior_mode": treatment["directional_prior_mode"],
            "treatment_directional_prior_scope": treatment.get("directional_prior_scope", "disabled"),
            "treatment_directional_schedule": treatment.get("directional_schedule", "disabled"),
            "treatment_lag_direction_source": treatment["lag_direction_source"],
            "treatment_directional_prior_lags": treatment.get("directional_prior_lags", "disabled"),
            "treatment_directional_prior_lag_weights": treatment.get("directional_prior_lag_weights", "disabled"),
            "treatment_structure_message_graph_mode": treatment.get("structure_message_graph_mode", "raw"),
            "treatment_directional_kappa_gate": treatment.get("directional_kappa_gate", 0),
            "treatment_directional_kappa_gate_quantile": treatment.get("directional_kappa_gate_quantile", 0.0),
            "treatment_directional_target_ratio": treatment.get("directional_target_ratio", 0.0),
            "treatment_directional_loss_end_epoch": treatment.get("directional_loss_end_epoch", -1),
            "treatment_parent_entropy_lambda": treatment.get("parent_entropy_lambda", 0.0),
            "treatment_parent_entropy_warmup_epochs": treatment.get("parent_entropy_warmup_epochs", 0),
            "treatment_parent_entropy_ramp_epochs": treatment.get("parent_entropy_ramp_epochs", 1),
            "treatment_parent_cap_lambda": treatment.get("parent_cap_lambda", 0.0),
            "treatment_parent_cap_target": treatment.get("parent_cap_target", 0.0),
            "treatment_parent_cap_warmup_epochs": treatment.get("parent_cap_warmup_epochs", 0),
            "treatment_parent_cap_ramp_epochs": treatment.get("parent_cap_ramp_epochs", 1),
            "treatment_ungated_symmetry_lambda": treatment.get("ungated_symmetry_lambda", 0.0),
            "treatment_ungated_symmetry_warmup_epochs": treatment.get("ungated_symmetry_warmup_epochs", 0),
            "treatment_ungated_symmetry_ramp_epochs": treatment.get("ungated_symmetry_ramp_epochs", 1),
            "treatment_structure_parameterization": treatment.get("structure_parameterization", "coupled"),
            "treatment_fixed_support_mask_mode": treatment.get("fixed_support_mask_mode", "none"),
            "treatment_direction_init_mode": treatment.get("direction_init_mode", "patel_tau"),
            "treatment_emb_dim": treatment.get("emb_dim", 0),
            "treatment_optimizer_step_mode": treatment.get("optimizer_step_mode", "subject"),
            "treatment_adj_activation": treatment.get("adj_activation", "sigmoid"),
            "treatment_kappa_logit_bias_scale": treatment.get("kappa_logit_bias_scale", 0.0),
            "treatment_direction_logit_bias_scale": treatment.get("direction_logit_bias_scale", 0.0),
            "treatment_main_loss_weight": treatment.get("main_loss_weight", 1.0),
            "treatment_selection_agreement_weight": treatment.get("selection_agreement_weight", 0.0),
            "treatment_direction_lr_multiplier": treatment.get("direction_lr_multiplier", 1.0),
            "treatment_freeze_direction_after_epoch": treatment.get("freeze_direction_after_epoch", -1),
            "treatment_subject_limit": treatment.get("subject_limit", -1),
            "treatment_time_limit": treatment.get("time_limit", -1),
            "baseline_result_dir": baseline["result_dir"],
            "treatment_result_dir": treatment["result_dir"],
            "baseline_failure_mode": baseline["failure_mode"],
            "treatment_failure_mode": treatment["failure_mode"],
            "failure_mode_shift": (
                f"{baseline['failure_mode']}->{treatment['failure_mode']}"
            ),
        }
        for metric in delta_metric_names:
            baseline_value = float(baseline[metric])
            treatment_value = float(treatment[metric])
            paired_row[f"{metric}_baseline"] = baseline_value
            paired_row[f"{metric}_treatment"] = treatment_value
            paired_row[f"{metric}_delta_treat_minus_base"] = treatment_value - baseline_value
        paired_rows.append(paired_row)
    return paired_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_output_stem(
    args: argparse.Namespace,
    seeds: Sequence[int],
    cross_pred_conditions: Sequence[bool],
    directional_conditions: Sequence[Tuple[bool, str, str]],
    timestamp: str,
) -> str:
    if len(cross_pred_conditions) > 1 and len(directional_conditions) > 1:
        condition_tag = "cross_direction_compare"
    elif len(cross_pred_conditions) > 1:
        condition_tag = "cross_compare"
    elif len(directional_conditions) > 1:
        condition_tag = "direction_compare"
    elif cross_pred_conditions[0]:
        condition_tag = "cross_on"
    elif directional_conditions[0][0]:
        condition_tag = directional_condition_name(*directional_conditions[0]).replace("directional_", "dir_")
    else:
        condition_tag = "cross_off"
    stem = (
        f"cross_pred_v1_final_only_compare_{args.structure_init_mode}_"
        f"{condition_tag}_{len(seeds)}seeds_{timestamp}"
    )
    if args.experiment_tag:
        stem = f"{stem}_{sanitize_tag(args.experiment_tag)}"
    return stem


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Final-only multi-seed experiment with margin diagnostics."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(repo_root / "fMRI_dataset" / "fMRI.csv"),
        help="Path to fMRI CSV file.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=str(repo_root / "fMRI_dataset" / "h1.txt"),
        help="Path to directed GT edge list.",
    )
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=str(script_dir / "results" / "run_20260310_185625" / "pretrained_encoder.pt"),
        help="Path to pretrained encoder checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--top_k_edges", type=int, default=50)
    parser.add_argument(
        "--subject_limit",
        type=int,
        default=-1,
        help="Optional subject cap forwarded to main_structure_learning.py; -1 keeps all.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=-1,
        help="Optional per-subject time cap forwarded to main_structure_learning.py; -1 keeps all.",
    )
    parser.add_argument(
        "--structure_init_mode",
        type=str,
        default="random",
        choices=["patel_score", "patel_kappa", "pearson", "random"],
        help="Non-Patel directional init to probe.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0.01,0.05,0.1",
        help="Comma-separated init_logit_scale values.",
    )
    parser.add_argument(
        "--emb_dims",
        type=str,
        default="0",
        help="Comma-separated structure embedding dimensions; 0 means full rank.",
    )
    parser.add_argument(
        "--structure_parameterizations",
        type=str,
        default="coupled",
        help="Comma-separated structure parameterizations: coupled,support_direction",
    )
    parser.add_argument(
        "--fixed_support_mask_modes",
        type=str,
        default="none",
        help="Comma-separated fixed support-mask modes: none,topk_kappa,maxgap_kappa",
    )
    parser.add_argument(
        "--direction_init_modes",
        type=str,
        default="patel_tau",
        help="Comma-separated direction init modes: patel_tau,zeros,random",
    )
    parser.add_argument(
        "--optimizer_step_modes",
        type=str,
        default="subject",
        help="Comma-separated optimizer step modes: subject,batch_mean",
    )
    parser.add_argument(
        "--adj_activations",
        type=str,
        default="sigmoid",
        help="Comma-separated adjacency activations: sigmoid,sparsemax,entmax15",
    )
    parser.add_argument(
        "--kappa_logit_bias_scales",
        type=str,
        default="0.0",
        help="Comma-separated persistent Patel-kappa logit-bias scales.",
    )
    parser.add_argument(
        "--direction_logit_bias_scales",
        type=str,
        default="0.0",
        help="Comma-separated persistent Patel-tau direction-logit bias scales.",
    )
    parser.add_argument(
        "--main_loss_weights",
        type=str,
        default="1.0",
        help="Comma-separated weights applied to the main DDM loss.",
    )
    parser.add_argument(
        "--selection_agreement_weights",
        type=str,
        default="0.0",
        help="Comma-separated Patel-agreement weights used by guarded best-epoch selection.",
    )
    parser.add_argument(
        "--direction_lr_multipliers",
        type=str,
        default="1.0",
        help="Comma-separated LR multipliers for the separate direction branch in support_direction mode.",
    )
    parser.add_argument(
        "--freeze_direction_after_epochs",
        type=str,
        default="-1",
        help="Comma-separated epoch counts after which the separate direction branch is frozen; -1 disables.",
    )
    parser.add_argument(
        "--lambda_l1_values",
        type=str,
        default="0.02",
        help="Comma-separated L1 values passed to main_structure_learning.py.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="11,22,33,44,55",
        help="Comma-separated random seeds.",
    )
    parser.add_argument(
        "--cross_pred_conditions",
        type=str,
        default="off",
        help="Comma-separated sweep over cross-pred states: off,on.",
    )
    parser.add_argument(
        "--directional_conditions",
        type=str,
        default="off",
        help="Comma-separated sweep over directional-loss states: off,patel,lag_corr_raw,lag_corr_encoder.",
    )
    parser.add_argument(
        "--directional_schedule",
        type=str,
        default="cosine_anneal",
        choices=["cosine_anneal", "plateau"],
        help="Directional auxiliary schedule when enabled.",
    )
    parser.add_argument(
        "--structure_message_graph_mode",
        type=str,
        default="raw",
        choices=["raw", "causal"],
        help="Adjacency convention used for GraphConv message passing inside the model.",
    )
    parser.add_argument(
        "--directional_kappa_gate",
        action="store_true",
        default=False,
        help="Gate directional margin loss to high-kappa pairs only.",
    )
    parser.add_argument(
        "--directional_kappa_gate_quantile",
        type=float,
        default=0.5,
        help="Quantile over positive Patel kappa used to define the directional gate.",
    )
    parser.add_argument(
        "--directional_target_ratios",
        type=str,
        default="0.01",
        help="Comma-separated target ratios for adaptive directional-margin weighting.",
    )
    parser.add_argument(
        "--directional_loss_end_epochs",
        type=str,
        default="-1",
        help="Comma-separated last epochs that keep directional supervision active; -1 keeps it on for the full run.",
    )
    parser.add_argument(
        "--directional_prior_lags",
        type=str,
        default="1",
        help="Comma-separated lag steps forwarded to lag-corr directional supervision.",
    )
    parser.add_argument(
        "--directional_prior_lag_weights",
        type=str,
        default="",
        help="Optional comma-separated lag weights for lag-corr directional supervision.",
    )
    parser.add_argument(
        "--directional_prior_scope",
        type=str,
        default="online_subject",
        choices=["online_subject", "global_dataset"],
        help="Whether lag-corr directional prior is recomputed per subject or fixed once from the full dataset.",
    )
    parser.add_argument(
        "--cross_pred_target_ratio",
        type=float,
        default=0.02,
        help="Target ratio for adaptive cross-pred weighting when enabled.",
    )
    parser.add_argument(
        "--cross_pred_schedule",
        type=str,
        default="cosine_anneal",
        choices=["cosine_anneal", "plateau"],
        help="Cross-pred auxiliary schedule when enabled.",
    )
    parser.add_argument(
        "--cross_pred_aggregation",
        type=str,
        default="mean",
        choices=["mean", "softmax"],
        help="Cross-pred aggregation when enabled.",
    )
    parser.add_argument(
        "--cross_pred_softmax_temp",
        type=float,
        default=1.0,
        help="Temperature for softmax cross-pred aggregation.",
    )
    parser.add_argument(
        "--cross_pred_lags",
        type=str,
        default="1",
        help="Comma-separated lag steps forwarded to cross-prediction supervision.",
    )
    parser.add_argument(
        "--cross_pred_lag_weights",
        type=str,
        default="",
        help="Optional comma-separated lag weights for cross-prediction supervision.",
    )
    parser.add_argument(
        "--cross_pred_fixed_weights",
        type=str,
        default="0.0",
        help="Comma-separated fixed cross-prediction weights; 0 keeps adaptive weighting only.",
    )
    parser.add_argument(
        "--cross_pred_fixed_warmup_epochs",
        type=int,
        default=0,
        help="Warmup epochs before fixed cross-prediction weight activates.",
    )
    parser.add_argument(
        "--cross_pred_fixed_ramp_epochs",
        type=int,
        default=1,
        help="Linear ramp epochs for fixed cross-prediction weight.",
    )
    parser.add_argument(
        "--parent_entropy_values",
        type=str,
        default="0.0",
        help="Comma-separated parent-entropy lambda values passed to main_structure_learning.py.",
    )
    parser.add_argument(
        "--parent_entropy_warmup_epochs",
        type=int,
        default=0,
        help="Warmup epochs before parent-entropy regularization activates.",
    )
    parser.add_argument(
        "--parent_entropy_ramp_epochs",
        type=int,
        default=1,
        help="Linear ramp epochs for parent-entropy regularization.",
    )
    parser.add_argument(
        "--parent_cap_values",
        type=str,
        default="0.0",
        help="Comma-separated parent-cap lambda values passed to main_structure_learning.py.",
    )
    parser.add_argument(
        "--parent_cap_targets",
        type=str,
        default="0.0",
        help="Comma-separated effective-parent targets used when parent-cap lambda is positive.",
    )
    parser.add_argument(
        "--parent_cap_warmup_epochs",
        type=int,
        default=0,
        help="Warmup epochs before parent-cap regularization activates.",
    )
    parser.add_argument(
        "--parent_cap_ramp_epochs",
        type=int,
        default=1,
        help="Linear ramp epochs for parent-cap regularization.",
    )
    parser.add_argument(
        "--ungated_symmetry_values",
        type=str,
        default="0.0",
        help="Comma-separated ungated-pair symmetry lambda values passed to main_structure_learning.py.",
    )
    parser.add_argument(
        "--ungated_symmetry_warmup_epochs",
        type=int,
        default=0,
        help="Warmup epochs before ungated-pair symmetry regularization activates.",
    )
    parser.add_argument(
        "--ungated_symmetry_ramp_epochs",
        type=int,
        default=1,
        help="Linear ramp epochs for ungated-pair symmetry regularization.",
    )
    parser.add_argument(
        "--strict_margin_eps_values",
        type=str,
        default="0",
        help="Comma-separated strict directional margin deadzones to report, e.g. 0,3e-4,0.1",
    )
    parser.add_argument(
        "--disable_temporal_encoder",
        action="store_true",
        default=False,
        help="Disable temporal encoder.",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default="",
        help="Optional tag appended to output CSV names.",
    )
    args = parser.parse_args()
    args.csv_path = str(Path(args.csv_path).resolve())
    args.gt_path = str(Path(args.gt_path).resolve())
    args.pretrain_checkpoint = str(Path(args.pretrain_checkpoint).resolve())

    gt_edges = load_gt(Path(args.gt_path))
    scales = parse_float_list(args.scales)
    emb_dims = parse_int_list(args.emb_dims)
    structure_parameterizations = parse_text_list(args.structure_parameterizations)
    fixed_support_mask_modes = parse_text_list(args.fixed_support_mask_modes)
    direction_init_modes = parse_text_list(args.direction_init_modes)
    optimizer_step_modes = parse_text_list(args.optimizer_step_modes)
    adj_activations = parse_text_list(args.adj_activations)
    kappa_logit_bias_scales = parse_float_list(args.kappa_logit_bias_scales)
    direction_logit_bias_scales = parse_float_list(args.direction_logit_bias_scales)
    main_loss_weights = parse_float_list(args.main_loss_weights)
    selection_agreement_weights = parse_float_list(args.selection_agreement_weights)
    direction_lr_multipliers = parse_float_list(args.direction_lr_multipliers)
    freeze_direction_after_epochs = parse_int_list(args.freeze_direction_after_epochs)
    directional_target_ratios = parse_float_list(args.directional_target_ratios)
    directional_loss_end_epochs = parse_int_list(args.directional_loss_end_epochs)
    directional_prior_lags = parse_int_list(args.directional_prior_lags)
    directional_prior_lag_weights = (
        parse_float_list(args.directional_prior_lag_weights)
        if args.directional_prior_lag_weights.strip()
        else []
    )
    cross_pred_lags = parse_int_list(args.cross_pred_lags)
    cross_pred_lag_weights = (
        parse_float_list(args.cross_pred_lag_weights)
        if args.cross_pred_lag_weights.strip()
        else []
    )
    cross_pred_fixed_weights = parse_float_list(args.cross_pred_fixed_weights)
    lambda_l1_values = parse_float_list(args.lambda_l1_values)
    parent_entropy_values = parse_float_list(args.parent_entropy_values)
    parent_cap_values = parse_float_list(args.parent_cap_values)
    parent_cap_targets = parse_float_list(args.parent_cap_targets)
    ungated_symmetry_values = parse_float_list(args.ungated_symmetry_values)
    strict_margin_eps_values = [normalize_margin_eps(v) for v in parse_float_list(args.strict_margin_eps_values)]
    seeds = parse_int_list(args.seeds)
    cross_pred_conditions = parse_cross_pred_conditions(args.cross_pred_conditions)
    directional_conditions = parse_directional_conditions(args.directional_conditions)
    if not emb_dims:
        parser.error("--emb_dims must include at least one integer value")
    if not structure_parameterizations:
        parser.error("--structure_parameterizations must include at least one value")
    if not fixed_support_mask_modes:
        parser.error("--fixed_support_mask_modes must include at least one value")
    if not direction_init_modes:
        parser.error("--direction_init_modes must include at least one value")
    if not optimizer_step_modes:
        parser.error("--optimizer_step_modes must include at least one value")
    if not adj_activations:
        parser.error("--adj_activations must include at least one value")
    if not kappa_logit_bias_scales:
        parser.error("--kappa_logit_bias_scales must include at least one value")
    if not direction_logit_bias_scales:
        parser.error("--direction_logit_bias_scales must include at least one value")
    if not main_loss_weights:
        parser.error("--main_loss_weights must include at least one value")
    if not selection_agreement_weights:
        parser.error("--selection_agreement_weights must include at least one value")
    if not direction_lr_multipliers:
        parser.error("--direction_lr_multipliers must include at least one value")
    if not freeze_direction_after_epochs:
        parser.error("--freeze_direction_after_epochs must include at least one value")
    if not directional_target_ratios:
        parser.error("--directional_target_ratios must include at least one value")
    if not directional_loss_end_epochs:
        parser.error("--directional_loss_end_epochs must include at least one value")
    if not directional_prior_lags:
        parser.error("--directional_prior_lags must include at least one value")
    if any(value <= 0 for value in directional_prior_lags):
        parser.error("--directional_prior_lags must contain positive integers")
    if directional_prior_lag_weights and len(directional_prior_lag_weights) != len(directional_prior_lags):
        parser.error("--directional_prior_lag_weights must match --directional_prior_lags length")
    if any(value < 0.0 for value in directional_prior_lag_weights):
        parser.error("--directional_prior_lag_weights must be non-negative")
    if directional_prior_lag_weights and sum(directional_prior_lag_weights) <= 0.0:
        parser.error("--directional_prior_lag_weights must sum to a positive value")
    if not cross_pred_lags:
        parser.error("--cross_pred_lags must include at least one value")
    if any(value <= 0 for value in cross_pred_lags):
        parser.error("--cross_pred_lags must contain positive integers")
    if cross_pred_lag_weights and len(cross_pred_lag_weights) != len(cross_pred_lags):
        parser.error("--cross_pred_lag_weights must match --cross_pred_lags length")
    if any(value < 0.0 for value in cross_pred_lag_weights):
        parser.error("--cross_pred_lag_weights must be non-negative")
    if cross_pred_lag_weights and sum(cross_pred_lag_weights) <= 0.0:
        parser.error("--cross_pred_lag_weights must sum to a positive value")
    if not cross_pred_fixed_weights:
        parser.error("--cross_pred_fixed_weights must include at least one value")
    if any(value < 0.0 for value in cross_pred_fixed_weights):
        parser.error("--cross_pred_fixed_weights must be >= 0")
    valid_optimizer_step_modes = {"subject", "batch_mean"}
    invalid_optimizer_step_modes = [
        value for value in optimizer_step_modes if value not in valid_optimizer_step_modes
    ]
    if invalid_optimizer_step_modes:
        parser.error(
            "--optimizer_step_modes contains invalid values: "
            + ",".join(invalid_optimizer_step_modes)
        )
    valid_adj_activations = {"sigmoid", "sparsemax", "entmax15"}
    invalid_adj_activations = [value for value in adj_activations if value not in valid_adj_activations]
    if invalid_adj_activations:
        parser.error(
            "--adj_activations contains invalid values: "
            + ",".join(invalid_adj_activations)
        )
    valid_structure_parameterizations = {"coupled", "support_direction"}
    invalid_structure_parameterizations = [
        value for value in structure_parameterizations if value not in valid_structure_parameterizations
    ]
    if invalid_structure_parameterizations:
        parser.error(
            "--structure_parameterizations contains invalid values: "
            + ",".join(invalid_structure_parameterizations)
        )
    valid_fixed_support_mask_modes = {"none", "topk_kappa", "maxgap_kappa"}
    invalid_fixed_support_mask_modes = [
        value for value in fixed_support_mask_modes if value not in valid_fixed_support_mask_modes
    ]
    if invalid_fixed_support_mask_modes:
        parser.error(
            "--fixed_support_mask_modes contains invalid values: "
            + ",".join(invalid_fixed_support_mask_modes)
        )
    valid_direction_init_modes = {"patel_tau", "zeros", "random"}
    invalid_direction_init_modes = [
        value for value in direction_init_modes if value not in valid_direction_init_modes
    ]
    if invalid_direction_init_modes:
        parser.error(
            "--direction_init_modes contains invalid values: "
            + ",".join(invalid_direction_init_modes)
        )
    if any(value < 0.0 for value in main_loss_weights):
        parser.error("--main_loss_weights must be >= 0")
    if any(value < 0.0 for value in selection_agreement_weights):
        parser.error("--selection_agreement_weights must be >= 0")
    if any(value <= 0.0 for value in direction_lr_multipliers):
        parser.error("--direction_lr_multipliers must be > 0")
    if any(value < -1 for value in freeze_direction_after_epochs):
        parser.error("--freeze_direction_after_epochs must be >= -1")
    if any(value < -1 for value in directional_loss_end_epochs):
        parser.error("--directional_loss_end_epochs must be >= -1")
    for structure_parameterization in structure_parameterizations:
        if structure_parameterization != "support_direction":
            incompatible_masks = [
                value for value in fixed_support_mask_modes if value != "none"
            ]
            if incompatible_masks:
                parser.error(
                    "--fixed_support_mask_modes with non-none values require "
                    "--structure_parameterizations support_direction"
                )
            incompatible_direction_inits = [
                value for value in direction_init_modes if value != "patel_tau"
            ]
            if incompatible_direction_inits:
                parser.error(
                    "--direction_init_modes other than patel_tau require "
                    "--structure_parameterizations support_direction"
                )
            if any(abs(value - 1.0) > 1e-12 for value in direction_lr_multipliers):
                parser.error(
                    "--direction_lr_multipliers other than 1.0 require "
                    "--structure_parameterizations support_direction"
                )
            if any(value >= 0 for value in freeze_direction_after_epochs):
                parser.error(
                    "--freeze_direction_after_epochs >= 0 require "
                    "--structure_parameterizations support_direction"
                )
    if any(value < 0.0 for value in directional_target_ratios):
        parser.error("--directional_target_ratios must be non-negative")
    if args.cross_pred_fixed_warmup_epochs < 0:
        parser.error("--cross_pred_fixed_warmup_epochs must be >= 0")
    if args.cross_pred_fixed_ramp_epochs < 1:
        parser.error("--cross_pred_fixed_ramp_epochs must be >= 1")
    if args.subject_limit == 0 or args.subject_limit < -1:
        parser.error("--subject_limit must be -1 or a positive integer")
    if args.time_limit == 0 or args.time_limit < -1:
        parser.error("--time_limit must be -1 or a positive integer")
    if any(value > 0.0 for value in parent_cap_values):
        positive_targets = [target for target in parent_cap_targets if target > 0.0]
        if not positive_targets:
            parser.error("--parent_cap_targets must include at least one value > 0 when parent cap is enabled")
    else:
        positive_targets = []
    if any(value > 0.0 for value in ungated_symmetry_values) and not args.directional_kappa_gate:
        parser.error("--ungated_symmetry_values with a positive value requires --directional_kappa_gate")
    args.directional_prior_lags = ",".join(str(value) for value in directional_prior_lags)
    args.directional_prior_lag_weights = ",".join(f"{value:g}" for value in directional_prior_lag_weights)
    args.cross_pred_lags = ",".join(str(value) for value in cross_pred_lags)
    args.cross_pred_lag_weights = ",".join(f"{value:g}" for value in cross_pred_lag_weights)
    print(
        "[Compat] Legacy cross-pred sweep is mapped to causal_lag_main_* in main_structure_learning.py; "
        "cross_pred_schedule and cross_pred_fixed warmup/ramp are recorded in CSV but not forwarded as trainer flags."
    )
    aggregate_metric_names = tuple(BASE_AGGREGATE_METRICS) + tuple(
        build_extra_strict_metric_names(strict_margin_eps_values)
    ) + tuple(
        build_best_extra_strict_metric_names(strict_margin_eps_values)
    ) + tuple(
        build_best_final_gap_metric_names(strict_margin_eps_values)
    )
    delta_metric_names = tuple(BASE_PAIRED_DELTA_METRICS) + tuple(
        build_extra_strict_metric_names(strict_margin_eps_values)
    ) + tuple(
        build_best_extra_strict_metric_names(strict_margin_eps_values)
    ) + tuple(
        build_best_final_gap_metric_names(strict_margin_eps_values)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = build_output_stem(
        args,
        seeds,
        cross_pred_conditions,
        directional_conditions,
        timestamp,
    )
    results_dir = script_dir / "results"
    run_summary_path = results_dir / f"{output_stem}.csv"
    aggregate_path = results_dir / f"{output_stem}_aggregate.csv"
    comparison_path = results_dir / f"{output_stem}_comparison.csv"
    paired_path = results_dir / f"{output_stem}_paired.csv"

    rows: List[Dict[str, object]] = []

    for enable_cross_prediction in cross_pred_conditions:
        cross_condition = cross_pred_condition_name(enable_cross_prediction)
        for enable_directional_loss, directional_prior_mode, lag_direction_source in directional_conditions:
            directional_condition = directional_condition_name(
                enable_directional_loss,
                directional_prior_mode,
                lag_direction_source,
            )
            condition = f"{cross_condition}__{directional_condition}"
            for parent_entropy_lambda in parent_entropy_values:
                for parent_cap_lambda in parent_cap_values:
                    cap_targets = [0.0] if parent_cap_lambda <= 0.0 else positive_targets
                    for parent_cap_target in cap_targets:
                        symmetry_values = [0.0]
                        if enable_directional_loss and args.directional_kappa_gate:
                            symmetry_values = ungated_symmetry_values
                        for ungated_symmetry_lambda in symmetry_values:
                            directional_ratio_values = [0.0] if not enable_directional_loss else directional_target_ratios
                            directional_end_values = [-1] if not enable_directional_loss else directional_loss_end_epochs
                            for directional_target_ratio in directional_ratio_values:
                                for directional_loss_end_epoch in directional_end_values:
                                    fixed_cross_weight_values = [0.0] if not enable_cross_prediction else cross_pred_fixed_weights
                                    for cross_pred_fixed_weight in fixed_cross_weight_values:
                                        for lambda_l1 in lambda_l1_values:
                                            for structure_parameterization in structure_parameterizations:
                                                active_fixed_support_mask_modes = (
                                                    fixed_support_mask_modes
                                                    if structure_parameterization == "support_direction"
                                                    else ["none"]
                                                )
                                                active_direction_init_modes = (
                                                    direction_init_modes
                                                    if structure_parameterization == "support_direction"
                                                    else ["patel_tau"]
                                                )
                                                for (
                                                    fixed_support_mask_mode,
                                                    direction_init_mode,
                                                    emb_dim,
                                                    optimizer_step_mode,
                                                    adj_activation,
                                                    kappa_logit_bias_scale,
                                                    direction_logit_bias_scale,
                                                    main_loss_weight,
                                                    selection_agreement_weight,
                                                ) in product(
                                                    active_fixed_support_mask_modes,
                                                    active_direction_init_modes,
                                                    emb_dims,
                                                    optimizer_step_modes,
                                                    adj_activations,
                                                    kappa_logit_bias_scales,
                                                    direction_logit_bias_scales,
                                                    main_loss_weights,
                                                    selection_agreement_weights,
                                                ):
                                                    active_direction_lr_multipliers = (
                                                        direction_lr_multipliers
                                                        if structure_parameterization == "support_direction"
                                                        else [1.0]
                                                    )
                                                    active_freeze_direction_after_epochs = (
                                                        freeze_direction_after_epochs
                                                        if structure_parameterization == "support_direction"
                                                        else [-1]
                                                    )
                                                    for (
                                                        direction_lr_multiplier,
                                                        freeze_direction_after_epoch,
                                                        scale,
                                                        seed,
                                                    ) in product(
                                                        active_direction_lr_multipliers,
                                                        active_freeze_direction_after_epochs,
                                                        scales,
                                                        seeds,
                                                    ):
                                                                    print(
                                                                        f"=== RUN condition={condition} init={args.structure_init_mode} "
                                                                        f"lambda_l1={lambda_l1} parent_entropy={parent_entropy_lambda} "
                                                                        f"parent_cap={parent_cap_lambda}@{parent_cap_target} "
                                                                        f"dir_ratio={directional_target_ratio} "
                                                                        f"struct={structure_parameterization} "
                                                                        f"support_mask={fixed_support_mask_mode} "
                                                                        f"dir_init={direction_init_mode} "
                                                                        f"emb_dim={emb_dim} msg_mode={args.structure_message_graph_mode} "
                                                                        f"opt_step={optimizer_step_mode} "
                                                                        f"adj_act={adj_activation} "
                                                                        f"kappa_bias={kappa_logit_bias_scale} "
                                                                        f"tau_bias={direction_logit_bias_scale} "
                                                                        f"main_w={main_loss_weight} "
                                                                        f"sel_agree={selection_agreement_weight} "
                                                                        f"dir_end={directional_loss_end_epoch} "
                                                                        f"cross_fixed={cross_pred_fixed_weight} "
                                                                        f"dir_lr_mult={direction_lr_multiplier} "
                                                                        f"dir_freeze={freeze_direction_after_epoch} "
                                                                        f"ungated_sym={ungated_symmetry_lambda} "
                                                                        f"subj_limit={args.subject_limit} time_limit={args.time_limit} "
                                                                        f"scale={scale} seed={seed} ==="
                                                                    )
                                                                    row = run_single_experiment(
                                                                        args=args,
                                                                        script_dir=script_dir,
                                                                        gt_edges=gt_edges,
                                                                        strict_margin_eps_values=strict_margin_eps_values,
                                                                        condition=condition,
                                                                        enable_cross_prediction=enable_cross_prediction,
                                                                        enable_directional_loss=enable_directional_loss,
                                                                        directional_prior_mode=directional_prior_mode,
                                                                        lag_direction_source=lag_direction_source,
                                                                        directional_prior_scope=args.directional_prior_scope,
                                                                        parent_entropy_lambda=parent_entropy_lambda,
                                                                        parent_cap_lambda=parent_cap_lambda,
                                                                        parent_cap_target=parent_cap_target,
                                                                        ungated_symmetry_lambda=ungated_symmetry_lambda,
                                                                        cross_pred_fixed_weight=cross_pred_fixed_weight,
                                                                        directional_target_ratio=directional_target_ratio,
                                                                        directional_loss_end_epoch=directional_loss_end_epoch,
                                                                        lambda_l1=lambda_l1,
                                                                        structure_parameterization=structure_parameterization,
                                                                        fixed_support_mask_mode=fixed_support_mask_mode,
                                                                        direction_init_mode=direction_init_mode,
                                                                        emb_dim=emb_dim,
                                                                        optimizer_step_mode=optimizer_step_mode,
                                                                        adj_activation=adj_activation,
                                                                        kappa_logit_bias_scale=kappa_logit_bias_scale,
                                                                        direction_logit_bias_scale=direction_logit_bias_scale,
                                                                        main_loss_weight=main_loss_weight,
                                                                        selection_agreement_weight=selection_agreement_weight,
                                                                        direction_lr_multiplier=direction_lr_multiplier,
                                                                        freeze_direction_after_epoch=freeze_direction_after_epoch,
                                                                        scale=scale,
                                                                        seed=seed,
                                                                    )
                                                                    rows.append(row)
                                                                    print(
                                                                        f"{condition} l1={lambda_l1} parent_ent={parent_entropy_lambda} "
                                                                        f"parent_cap={parent_cap_lambda}@{parent_cap_target} "
                                                                        f"dir_ratio={directional_target_ratio} "
                                                                        f"struct={structure_parameterization} "
                                                                        f"support_mask={fixed_support_mask_mode} "
                                                                        f"dir_init={direction_init_mode} "
                                                                        f"emb_dim={emb_dim} msg_mode={args.structure_message_graph_mode} "
                                                                        f"opt_step={optimizer_step_mode} "
                                                                        f"adj_act={adj_activation} "
                                                                        f"kappa_bias={kappa_logit_bias_scale} "
                                                                        f"tau_bias={direction_logit_bias_scale} "
                                                                        f"main_w={main_loss_weight} "
                                                                        f"sel_agree={selection_agreement_weight} "
                                                                        f"dir_end={directional_loss_end_epoch} "
                                                                        f"cross_fixed={cross_pred_fixed_weight} "
                                                                        f"dir_lr_mult={direction_lr_multiplier} "
                                                                        f"dir_freeze={freeze_direction_after_epoch} "
                                                                        f"ungated_sym={ungated_symmetry_lambda} "
                                                                        f"subj_limit={args.subject_limit} time_limit={args.time_limit} "
                                                                        f"scale={scale} seed={seed}: "
                                                                        f"diff={row['final_diff_loss']:.4f}, "
                                                                        f"best_strict={row['best_strict_f1']:.4f}, "
                                                                        f"final_F1={row['final_f1']:.4f}, "
                                                                        f"strict_primary_F1={row['strict_f1']:.4f}, "
                                                                        f"gap_primary={row[best_final_gap_metric_field('strict_f1', strict_margin_eps_values[0])]:.4f}, "
                                                                        f"{format_margin_eps_summary(row, strict_margin_eps_values)}, "
                                                                        f"margin_med={row['margin_median']:.4e}, "
                                                                        f"gt_margin_med={row['gt_signed_margin_median']:.4e}, "
                                                                        f"gt_margin_gap={row['best_final_gap_gt_signed_margin_median']:.4e}, "
                                                                        f"p90={row['margin_p90']:.4e}, "
                                                                        f"eff_par={row['adj_eff_parents_mean']:.2f}, "
                                                                        f"gt_pos={row['gt_signed_margin_frac_pos']:.2%}, "
                                                                        f"near0(<1e-2)={row['margin_lt_1e2_frac']:.2%}, "
                                                                        f"mode={row['failure_mode']}"
                                                                    )

    write_csv(run_summary_path, rows)
    aggregate_rows_data = aggregate_rows(rows, aggregate_metric_names)
    write_csv(aggregate_path, aggregate_rows_data)

    comparison_rows = build_condition_deltas(aggregate_rows_data, delta_metric_names)
    if comparison_rows:
        write_csv(comparison_path, comparison_rows)
    paired_rows = build_paired_seed_deltas(rows, delta_metric_names)
    if paired_rows:
        write_csv(paired_path, paired_rows)

    print(f"RUN_SUMMARY_CSV {run_summary_path}")
    print(f"AGGREGATE_CSV {aggregate_path}")
    for row in aggregate_rows_data:
        print(
            f"AGG {row['condition']} l1={row['lambda_l1']} "
            f"struct={row.get('structure_parameterization', 'coupled')} "
            f"support_mask={row.get('fixed_support_mask_mode', 'none')} "
            f"dir_init={row.get('direction_init_mode', 'patel_tau')} "
            f"emb_dim={row.get('emb_dim', 0)} "
            f"msg_mode={row.get('structure_message_graph_mode', 'raw')} "
            f"opt_step={row.get('optimizer_step_mode', 'subject')} "
            f"adj_act={row.get('adj_activation', 'sigmoid')} "
            f"kappa_bias={row.get('kappa_logit_bias_scale', 0.0)} "
            f"tau_bias={row.get('direction_logit_bias_scale', 0.0)} "
            f"main_w={row.get('main_loss_weight', 1.0)} "
            f"sel_agree={row.get('selection_agreement_weight', 0.0)} "
            f"dir_end={row.get('directional_loss_end_epoch', -1)} "
            f"dir_lr_mult={row.get('direction_lr_multiplier', 1.0)} "
            f"dir_freeze={row.get('freeze_direction_after_epoch', -1)} "
            f"subj_limit={row.get('subject_limit', -1)} "
            f"time_limit={row.get('time_limit', -1)} "
            f"dir_ratio={row.get('directional_target_ratio', 0.0)} "
            f"parent_ent={row.get('parent_entropy_lambda', 0.0)} "
            f"parent_cap={row.get('parent_cap_lambda', 0.0)}@{row.get('parent_cap_target', 0.0)} "
            f"ungated_sym={row.get('ungated_symmetry_lambda', 0.0)} "
            f"scale={row['structure_init_scale']}: "
            f"diff={row['final_diff_loss_mean']:.4f}+/-{row['final_diff_loss_std']:.4f}, "
            f"best_strict={row['best_strict_f1_mean']:.4f}+/-{row['best_strict_f1_std']:.4f}, "
            f"f1={row['final_f1_mean']:.4f}+/-{row['final_f1_std']:.4f}, "
            f"strict@{normalize_margin_eps(strict_margin_eps_values[0]):g}="
            f"{row['strict_f1_mean']:.4f}+/-{row['strict_f1_std']:.4f}, "
            f"gap@{normalize_margin_eps(strict_margin_eps_values[0]):g}="
            f"{row[best_final_gap_metric_field('strict_f1', strict_margin_eps_values[0]) + '_mean']:.4f}, "
            f"margin_med={row['margin_median_mean']:.4e}+/-{row['margin_median_std']:.4e}, "
            f"gt_margin_gap={row['best_final_gap_gt_signed_margin_median_mean']:.4e}, "
            f"p90={row['margin_p90_mean']:.4e}+/-{row['margin_p90_std']:.4e}, "
            f"eff_par={row['adj_eff_parents_mean_mean']:.2f}, "
            f"near0(<1e-2)={row['margin_lt_1e2_frac_mean']:.2%}, "
            f"failure={row['failure_mode_counts']}"
        )
    if comparison_rows:
        print(f"COMPARISON_CSV {comparison_path}")
        for row in comparison_rows:
            print(
                f"DELTA scale={row['structure_init_scale']} "
                f"dir={row['treatment_directional_prior_mode']} "
                f"dir_sched={row.get('treatment_directional_schedule', 'disabled')} "
                f"emb_dim={row.get('treatment_emb_dim', row.get('emb_dim', 0))} "
                f"msg_mode={row.get('treatment_structure_message_graph_mode', row.get('structure_message_graph_mode', 'raw'))} "
                f"opt_step={row.get('treatment_optimizer_step_mode', row.get('optimizer_step_mode', 'subject'))} "
                f"adj_act={row.get('treatment_adj_activation', row.get('adj_activation', 'sigmoid'))} "
                f"kappa_bias={row.get('treatment_kappa_logit_bias_scale', row.get('kappa_logit_bias_scale', 0.0))} "
                f"tau_bias={row.get('treatment_direction_logit_bias_scale', row.get('direction_logit_bias_scale', 0.0))} "
                f"main_w={row.get('treatment_main_loss_weight', row.get('main_loss_weight', 1.0))} "
                f"sel_agree={row.get('treatment_selection_agreement_weight', row.get('selection_agreement_weight', 0.0))} "
                f"dir_end={row.get('treatment_directional_loss_end_epoch', row.get('directional_loss_end_epoch', -1))} "
                f"dir_lr_mult={row.get('treatment_direction_lr_multiplier', row.get('direction_lr_multiplier', 1.0))} "
                f"dir_freeze={row.get('treatment_freeze_direction_after_epoch', row.get('freeze_direction_after_epoch', -1))} "
                f"subj_limit={row.get('treatment_subject_limit', row.get('subject_limit', -1))} "
                f"time_limit={row.get('treatment_time_limit', row.get('time_limit', -1))} "
                f"dir_gate={row.get('treatment_directional_kappa_gate', 0)}@"
                f"{row.get('treatment_directional_kappa_gate_quantile', 0.0)} "
                f"dir_ratio={row.get('treatment_directional_target_ratio', row.get('directional_target_ratio', 0.0))} "
                f"parent_ent={row.get('treatment_parent_entropy_lambda', 0.0)}: "
                f"parent_cap={row.get('treatment_parent_cap_lambda', 0.0)}@"
                f"{row.get('treatment_parent_cap_target', 0.0)} "
                f"ungated_sym={row.get('treatment_ungated_symmetry_lambda', 0.0)}: "
                f"d_margin_med={row['margin_median_delta_treat_minus_base']:.4e}, "
                f"d_p90={row['margin_p90_delta_treat_minus_base']:.4e}, "
                f"d_near0={row['margin_lt_1e2_frac_delta_treat_minus_base']:.2%}, "
                f"d_f1={row['final_f1_delta_treat_minus_base']:.4f}, "
                f"d_sym_collapse={row['symmetric_collapse_frac_delta_treat_minus_base']:.2%}"
            )
    if paired_rows:
        print(f"PAIRED_COMPARISON_CSV {paired_path}")


if __name__ == "__main__":
    main()
