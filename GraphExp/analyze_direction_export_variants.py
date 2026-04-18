from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from main_structure_learning import (
    load_gt_edges,
    selector_audit_evaluate_directional_strict,
    selector_audit_failure_mode,
    selector_audit_gt_edge_margin_stats,
    selector_audit_margin_stats,
    to_causal_matrix_np,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate offline export variants on saved support/direction snapshots "
            "without retraining."
        )
    )
    parser.add_argument(
        "--run_dir",
        action="append",
        dest="run_dirs",
        type=Path,
        required=True,
        help="Run directory containing support_direction_snapshots.npz and config.npy. May be passed multiple times.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Optional GT edge path override. Defaults to selector_audit_gt_path from config.npy.",
    )
    parser.add_argument(
        "--margin_eps",
        type=float,
        default=0.0,
        help="Strict directional margin eps used for F1 evaluation.",
    )
    parser.add_argument(
        "--keep_fracs",
        type=str,
        default="0.75,0.5,0.25",
        help="Comma-separated confident-pair keep fractions used by mask_hard_topfrac variants.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="export_variants",
        help="Short suffix used in the output filenames.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict[str, Any]:
    cfg = np.load(run_dir / "config.npy", allow_pickle=True).item()
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict-like config in {run_dir / 'config.npy'}")
    return dict(cfg)


def resolve_repo_relative_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def load_selector_summary(run_dir: Path) -> Dict[str, str]:
    path = run_dir / "selector_audit_summary.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        raise RuntimeError(f"No rows found in {path}")
    return row


def parse_keep_fracs(text: str) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"keep_frac must be in (0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one keep_frac must be provided")
    deduped = sorted(set(values), reverse=True)
    return deduped


def metric_from_selector_row(row: Dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in {"", None} else float("nan")


def build_epoch_index(epochs: np.ndarray) -> Dict[int, int]:
    return {int(epoch): idx for idx, epoch in enumerate(epochs.tolist())}


def strict_metrics_with_failure_mode(
    adj_causal: np.ndarray,
    gt_edges: Set[Tuple[int, int]],
    margin_eps: float,
) -> Dict[str, Any]:
    strict_eval = selector_audit_evaluate_directional_strict(
        adj_causal,
        gt_edges,
        margin_eps=margin_eps,
    )
    margin_stats = selector_audit_margin_stats(adj_causal)
    gt_margin_stats = selector_audit_gt_edge_margin_stats(adj_causal, gt_edges)
    failure_mode = selector_audit_failure_mode(
        {
            "selector_audit_margin_p90": margin_stats["selector_audit_margin_p90"],
            "selector_audit_margin_lt_1e2_frac": margin_stats["selector_audit_margin_lt_1e2_frac"],
            "selector_audit_margin_median": margin_stats["selector_audit_margin_median"],
        },
        {
            "selector_audit_f1": strict_eval["strict_f1"],
        },
    )
    return {
        **strict_eval,
        **margin_stats,
        **gt_margin_stats,
        "failure_mode": failure_mode,
    }


def unordered_pair_indices(mask: np.ndarray) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for i in range(mask.shape[0]):
        for j in range(i + 1, mask.shape[1]):
            if bool(mask[i, j] or mask[j, i]):
                pairs.append((i, j))
    return pairs


def build_confident_pair_mask(
    delta_raw: np.ndarray,
    support_mask: np.ndarray,
    keep_frac: float,
) -> np.ndarray:
    confident_mask = np.zeros_like(support_mask, dtype=bool)
    pairs = unordered_pair_indices(support_mask)
    if not pairs:
        return confident_mask
    num_keep = max(1, int(np.ceil(len(pairs) * float(keep_frac))))
    ranked_pairs = sorted(
        pairs,
        key=lambda ij: (abs(float(delta_raw[ij[0], ij[1]])), -ij[0], -ij[1]),
        reverse=True,
    )
    for i, j in ranked_pairs[:num_keep]:
        confident_mask[i, j] = True
        confident_mask[j, i] = True
    return confident_mask


def build_raw_export_variant(
    *,
    variant: str,
    support_weights: np.ndarray,
    direction_logits: np.ndarray,
    support_mask: np.ndarray,
    keep_frac: Optional[float],
) -> np.ndarray:
    num_nodes = support_weights.shape[0]
    diag_mask = ~np.eye(num_nodes, dtype=bool)
    delta_raw = direction_logits - direction_logits.T
    direction_gate = 1.0 / (1.0 + np.exp(-delta_raw))

    if variant == "current_soft":
        raw = support_weights * direction_gate
    elif variant == "mask_soft":
        raw = support_mask.astype(np.float32) * direction_gate
    elif variant in {"mask_hard", "mask_hard_topfrac"}:
        active_mask = support_mask.astype(bool)
        if variant == "mask_hard_topfrac":
            if keep_frac is None:
                raise ValueError("keep_frac is required for mask_hard_topfrac")
            active_mask = build_confident_pair_mask(delta_raw, support_mask.astype(bool), keep_frac)
        raw = np.zeros_like(support_weights, dtype=np.float32)
        pairs = unordered_pair_indices(active_mask)
        for i, j in pairs:
            if delta_raw[i, j] > 0.0:
                raw[i, j] = 1.0
                raw[j, i] = 0.0
            elif delta_raw[i, j] < 0.0:
                raw[i, j] = 0.0
                raw[j, i] = 1.0
            else:
                raw[i, j] = 0.5
                raw[j, i] = 0.5
    else:
        raise ValueError(f"Unsupported export variant: {variant}")

    raw = raw * diag_mask.astype(np.float32)
    return raw.astype(np.float32, copy=False)


def evaluate_anchor_variants(
    *,
    run_dir: Path,
    dataset: str,
    seed: int,
    gt_edges: Set[Tuple[int, int]],
    margin_eps: float,
    anchor_name: str,
    anchor_epoch: int,
    support_weights: np.ndarray,
    direction_logits: np.ndarray,
    support_mask: np.ndarray,
    keep_fracs: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    variant_specs: List[Tuple[str, Optional[float], str]] = [
        ("current_soft", None, "current_soft"),
        ("mask_soft", None, "mask_soft"),
        ("mask_hard", None, "mask_hard"),
    ]
    for keep_frac in keep_fracs:
        label = f"mask_hard_top{int(round(keep_frac * 100.0)):02d}"
        variant_specs.append(("mask_hard_topfrac", float(keep_frac), label))

    baseline_strict_f1: Optional[float] = None
    for variant, keep_frac, label in variant_specs:
        raw_adj = build_raw_export_variant(
            variant=variant,
            support_weights=support_weights,
            direction_logits=direction_logits,
            support_mask=support_mask,
            keep_frac=keep_frac,
        )
        adj_causal = to_causal_matrix_np(raw_adj)
        metrics = strict_metrics_with_failure_mode(adj_causal, gt_edges, margin_eps)
        row = {
            "run_dir": str(run_dir),
            "dataset": dataset,
            "seed": int(seed),
            "anchor_name": anchor_name,
            "anchor_epoch": int(anchor_epoch),
            "variant": label,
            "keep_frac": (float(keep_frac) if keep_frac is not None else 1.0),
            "strict_f1": float(metrics["strict_f1"]),
            "strict_precision": float(metrics["strict_precision"]),
            "strict_recall": float(metrics["strict_recall"]),
            "strict_pred_count": float(metrics["strict_pred_count"]),
            "margin_median": float(metrics["selector_audit_margin_median"]),
            "margin_p90": float(metrics["selector_audit_margin_p90"]),
            "gt_signed_margin_median": float(metrics["selector_audit_gt_signed_margin_median"]),
            "failure_mode": str(metrics["failure_mode"]),
        }
        if label == "current_soft":
            baseline_strict_f1 = float(row["strict_f1"])
        row["strict_f1_gain_vs_current"] = (
            float(row["strict_f1"]) - float(baseline_strict_f1)
            if baseline_strict_f1 is not None else 0.0
        )
        rows.append(row)
    return rows


def aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = ["dataset", "anchor_name", "variant", "keep_frac"]
    summary_rows: List[Dict[str, Any]] = []
    for group_key, group_df in df.groupby(group_cols, dropna=False):
        dataset, anchor_name, variant, keep_frac = group_key
        strict_f1_vals = group_df["strict_f1"].astype(float).tolist()
        pred_count_vals = group_df["strict_pred_count"].astype(float).tolist()
        margin_vals = group_df["margin_median"].astype(float).tolist()
        gain_vals = group_df["strict_f1_gain_vs_current"].astype(float).tolist()
        gt_margin_vals = group_df["gt_signed_margin_median"].astype(float).tolist()
        summary_rows.append({
            "dataset": dataset,
            "anchor_name": anchor_name,
            "variant": variant,
            "keep_frac": float(keep_frac),
            "run_count": int(len(group_df)),
            "seed_list": ",".join(str(int(v)) for v in group_df["seed"].tolist()),
            "strict_f1_mean": mean(strict_f1_vals),
            "strict_f1_std": pstdev(strict_f1_vals) if len(strict_f1_vals) > 1 else 0.0,
            "strict_pred_count_mean": mean(pred_count_vals),
            "margin_median_mean": mean(margin_vals),
            "gt_signed_margin_median_mean": mean(gt_margin_vals),
            "strict_f1_gain_vs_current_mean": mean(gain_vals),
            "failure_mode_counts": dict(group_df["failure_mode"].value_counts()),
        })
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    keep_fracs = parse_keep_fracs(args.keep_fracs)

    all_rows: List[Dict[str, Any]] = []
    for run_dir_arg in args.run_dirs:
        run_dir = run_dir_arg.resolve()
        snapshot_path = run_dir / "support_direction_snapshots.npz"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Missing snapshot file: {snapshot_path}")

        cfg = load_config(run_dir)
        dataset = Path(str(cfg.get("csv_path", ""))).stem
        seed = int(cfg.get("seed", -1))
        gt_path = args.gt_path or cfg.get("selector_audit_gt_path")
        if not gt_path:
            raise ValueError(
                "GT path is required; pass --gt_path or ensure selector_audit_gt_path "
                f"exists in config.npy for {run_dir}."
            )
        gt_edges = load_gt_edges(str(resolve_repo_relative_path(str(gt_path))))
        selector_summary = load_selector_summary(run_dir)
        anchor_epochs = {
            "best": int(metric_from_selector_row(selector_summary, "selector_audit_best_gt_epoch")),
            "exported": int(metric_from_selector_row(selector_summary, "selector_audit_exported_epoch")),
            "final": int(metric_from_selector_row(selector_summary, "selector_audit_final_epoch")),
        }

        with np.load(snapshot_path) as data:
            epochs = data["epochs"].astype(np.int32, copy=False)
            support_weights_all = data["support_weights"].astype(np.float32, copy=False)
            direction_logits_all = data["direction_logits"].astype(np.float32, copy=False)
            if "fixed_support_mask" in data:
                support_mask = data["fixed_support_mask"].astype(np.float32, copy=False)
            else:
                support_mask = (support_weights_all[0] > 0.0).astype(np.float32, copy=False)

        epoch_to_idx = build_epoch_index(epochs)
        for anchor_name, anchor_epoch in anchor_epochs.items():
            if anchor_epoch not in epoch_to_idx:
                raise RuntimeError(
                    f"Anchor epoch {anchor_epoch} ({anchor_name}) not found in snapshots for {run_dir}"
                )
            anchor_idx = epoch_to_idx[anchor_epoch]
            all_rows.extend(
                evaluate_anchor_variants(
                    run_dir=run_dir,
                    dataset=dataset,
                    seed=seed,
                    gt_edges=gt_edges,
                    margin_eps=args.margin_eps,
                    anchor_name=anchor_name,
                    anchor_epoch=anchor_epoch,
                    support_weights=support_weights_all[anchor_idx],
                    direction_logits=direction_logits_all[anchor_idx],
                    support_mask=support_mask,
                    keep_fracs=keep_fracs,
                )
            )

    full_df = pd.DataFrame(all_rows)
    if full_df.empty:
        raise RuntimeError("No rows generated.")
    aggregate_df = aggregate_rows(all_rows)

    dataset_label = "-".join(sorted(set(full_df["dataset"].astype(str).tolist())))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = RESULTS_DIR / f"unify_export_variants_{dataset_label}_{timestamp}_{args.tag}.csv"
    aggregate_path = RESULTS_DIR / f"unify_export_variants_{dataset_label}_{timestamp}_{args.tag}_aggregate.csv"
    full_df.to_csv(full_path, index=False, float_format="%.6f")
    aggregate_df.to_csv(aggregate_path, index=False, float_format="%.6f")
    print(f"Full rows written to: {full_path}")
    print(f"Aggregate written to: {aggregate_path}")


if __name__ == "__main__":
    main()
