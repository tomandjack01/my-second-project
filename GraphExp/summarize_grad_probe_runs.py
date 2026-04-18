from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize gradient-alignment probe diagnostics from replayed run dirs."
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Run directories containing config.npy, quality_history.csv, and selector_audit_summary.csv.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="grad_probe",
        help="Suffix used in output filenames.",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help="Optional explicit output stem without extension.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.npy"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.npy in {run_dir}")
    cfg = np.load(config_path, allow_pickle=True).item()
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict config in {config_path}")
    return dict(cfg)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_single_csv_row(path: Path) -> Dict[str, str]:
    rows = load_csv_rows(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one data row in {path}, found {len(rows)}")
    return rows[0]


def to_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or value == "":
        return default
    return int(float(value))


def dataset_name_from_config(cfg: Dict[str, Any]) -> str:
    csv_path = str(cfg.get("csv_path", ""))
    return Path(csv_path).stem if csv_path else "unknown"


def finite_values(rows: Sequence[Dict[str, str]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = to_float(row, key, float("nan"))
        if math.isfinite(value):
            values.append(value)
    return values


def stat_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else float("nan")


def stat_median(values: Sequence[float]) -> float:
    return float(median(values)) if values else float("nan")


def stat_std(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0 if values else float("nan")


def build_run_row(run_dir: Path) -> Dict[str, Any]:
    cfg = load_config(run_dir)
    quality_history_path = run_dir / "quality_history.csv"
    selector_summary_path = run_dir / "selector_audit_summary.csv"
    if not quality_history_path.exists():
        raise FileNotFoundError(f"Missing quality_history.csv in {run_dir}")
    if not selector_summary_path.exists():
        raise FileNotFoundError(f"Missing selector_audit_summary.csv in {run_dir}")

    quality_rows = load_csv_rows(quality_history_path)
    selector_summary = load_single_csv_row(selector_summary_path)
    if not quality_rows:
        raise RuntimeError(f"No quality rows found in {quality_history_path}")

    dataset = dataset_name_from_config(cfg)
    probe_rows = [row for row in quality_rows if to_float(row, "grad_probe_available", 0.0) > 0.5]
    joint_rows = [row for row in probe_rows if to_int(row, "detach_direction_from_main_active", 0) <= 0]
    isolated_rows = [row for row in probe_rows if to_int(row, "detach_direction_from_main_active", 0) > 0]
    last5_rows = probe_rows[-5:] if probe_rows else []
    final_row = quality_rows[-1]

    row: Dict[str, Any] = {
        "dataset": dataset,
        "seed": int(cfg.get("seed", -1)),
        "run_dir": str(run_dir),
        "gradient_routing_mode": str(cfg.get("gradient_routing_mode", "")),
        "detach_direction_from_main_after_epoch": int(cfg.get("detach_direction_from_main_after_epoch", -1)),
        "freeze_direction_after_epoch": int(cfg.get("freeze_direction_after_epoch", -1)),
        "directional_loss_end_epoch": int(cfg.get("directional_loss_end_epoch", -1)),
        "causal_lag_main_weight": float(cfg.get("causal_lag_main_weight", 0.0)),
        "best_primary_strict_f1": to_float(selector_summary, "selector_audit_best_gt_primary_strict_f1"),
        "exported_primary_strict_f1": to_float(selector_summary, "selector_audit_exported_primary_strict_f1"),
        "final_primary_strict_f1": to_float(selector_summary, "selector_audit_final_primary_strict_f1"),
        "best_failure_mode": str(selector_summary.get("selector_audit_best_gt_failure_mode", "")),
        "exported_failure_mode": str(selector_summary.get("selector_audit_exported_failure_mode", "")),
        "final_failure_mode": str(selector_summary.get("selector_audit_final_failure_mode", "")),
        "final_gt_signed_margin_median": to_float(final_row, "selector_audit_gt_signed_margin_median"),
        "probe_epoch_count": len(probe_rows),
        "joint_epoch_count": len(joint_rows),
        "isolated_epoch_count": len(isolated_rows),
        "probe_ratio_mean": stat_mean(finite_values(probe_rows, "grad_probe_dir_to_diff_norm_ratio")),
        "probe_ratio_median": stat_median(finite_values(probe_rows, "grad_probe_dir_to_diff_norm_ratio")),
        "probe_ratio_last5_mean": stat_mean(finite_values(last5_rows, "grad_probe_dir_to_diff_norm_ratio")),
        "probe_diff_norm_mean": stat_mean(finite_values(probe_rows, "grad_probe_diff_norm")),
        "probe_dir_norm_weighted_mean": stat_mean(finite_values(probe_rows, "grad_probe_dir_norm_weighted")),
        "probe_cosine_mean": stat_mean(finite_values(probe_rows, "grad_probe_cosine")),
        "probe_cosine_min": min(finite_values(probe_rows, "grad_probe_cosine"), default=float("nan")),
        "probe_cosine_last5_mean": stat_mean(finite_values(last5_rows, "grad_probe_cosine")),
        "probe_negative_frac": stat_mean(finite_values(probe_rows, "grad_probe_cosine_negative")),
        "joint_ratio_mean": stat_mean(finite_values(joint_rows, "grad_probe_dir_to_diff_norm_ratio")),
        "joint_cosine_mean": stat_mean(finite_values(joint_rows, "grad_probe_cosine")),
        "joint_negative_frac": stat_mean(finite_values(joint_rows, "grad_probe_cosine_negative")),
        "isolated_ratio_mean": stat_mean(finite_values(isolated_rows, "grad_probe_dir_to_diff_norm_ratio")),
        "isolated_cosine_mean": stat_mean(finite_values(isolated_rows, "grad_probe_cosine")),
        "isolated_negative_frac": stat_mean(finite_values(isolated_rows, "grad_probe_cosine_negative")),
        "final_routing_label": str(final_row.get("gradient_routing_label", "")),
        "final_detach_direction_from_main_active": to_int(final_row, "detach_direction_from_main_active", 0),
        "final_dir_lambda_current": to_float(final_row, "dir_lambda_current"),
    }
    return row


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["dataset"])].append(row)

    numeric_keys = [
        "best_primary_strict_f1",
        "exported_primary_strict_f1",
        "final_primary_strict_f1",
        "final_gt_signed_margin_median",
        "probe_epoch_count",
        "joint_epoch_count",
        "isolated_epoch_count",
        "probe_ratio_mean",
        "probe_ratio_median",
        "probe_ratio_last5_mean",
        "probe_diff_norm_mean",
        "probe_dir_norm_weighted_mean",
        "probe_cosine_mean",
        "probe_cosine_min",
        "probe_cosine_last5_mean",
        "probe_negative_frac",
        "joint_ratio_mean",
        "joint_cosine_mean",
        "joint_negative_frac",
        "isolated_ratio_mean",
        "isolated_cosine_mean",
        "isolated_negative_frac",
        "final_dir_lambda_current",
    ]

    aggregate_out: List[Dict[str, Any]] = []
    for dataset, group_rows in sorted(groups.items()):
        agg: Dict[str, Any] = {
            "dataset": dataset,
            "run_count": len(group_rows),
            "seed_list": ",".join(str(row["seed"]) for row in group_rows),
            "run_dirs": ";".join(str(row["run_dir"]) for row in group_rows),
            "best_failure_mode_counts": dict(Counter(str(row["best_failure_mode"]) for row in group_rows)),
            "exported_failure_mode_counts": dict(Counter(str(row["exported_failure_mode"]) for row in group_rows)),
            "final_failure_mode_counts": dict(Counter(str(row["final_failure_mode"]) for row in group_rows)),
            "gradient_routing_mode_counts": dict(Counter(str(row["gradient_routing_mode"]) for row in group_rows)),
            "final_routing_label_counts": dict(Counter(str(row["final_routing_label"]) for row in group_rows)),
        }
        for key in numeric_keys:
            vals = [float(row[key]) for row in group_rows if math.isfinite(float(row[key]))]
            agg[f"{key}_mean"] = stat_mean(vals)
            agg[f"{key}_std"] = stat_std(vals)
        aggregate_out.append(agg)
    return aggregate_out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(text).resolve() for text in args.run_dirs]
    rows = [build_run_row(run_dir) for run_dir in run_dirs]
    aggregate = aggregate_rows(rows)

    datasets = sorted({str(row["dataset"]) for row in rows})
    dataset_label = datasets[0] if len(datasets) == 1 else "mixed"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = (
        args.output_stem
        if args.output_stem
        else f"unify_grad_probe_{dataset_label}_{timestamp}_{args.tag}"
    )
    summary_path = RESULTS_DIR / f"{output_stem}.csv"
    aggregate_path = RESULTS_DIR / f"{output_stem}_aggregate.csv"
    write_csv(summary_path, rows)
    write_csv(aggregate_path, aggregate)
    print(f"Summary written to: {summary_path}")
    print(f"Aggregate written to: {aggregate_path}")


if __name__ == "__main__":
    main()
