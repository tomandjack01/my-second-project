from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

DEFAULT_SELECTOR_MODES = [
    "legacy",
    "causal_lag_composite",
    "causal_lag_entropy_composite",
]
DEFAULT_AGREEMENT_WEIGHTS = [0.0, 0.25]

SCORE_FIELD_BY_MODE = {
    "legacy": "score_legacy_total",
    "causal_lag_composite": "score_composite_total",
    "causal_lag_entropy_composite": "score_entropy_composite_total",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline replay of selector score modes on saved quality_history.csv files."
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Run directories that contain quality_history.csv, selector_audit_summary.csv, and config.npy.",
    )
    parser.add_argument(
        "--selector_modes",
        type=str,
        default=",".join(DEFAULT_SELECTOR_MODES),
        help="Comma-separated selector modes to replay.",
    )
    parser.add_argument(
        "--agreement_weights",
        type=str,
        default=",".join(str(v) for v in DEFAULT_AGREEMENT_WEIGHTS),
        help="Comma-separated selection_agreement_weight values to test.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="selector_replay",
        help="Suffix used in output filenames.",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help="Optional explicit output stem without extension.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


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


def dataset_name_from_config(cfg: Dict[str, Any]) -> str:
    csv_path = str(cfg.get("csv_path", ""))
    return Path(csv_path).stem if csv_path else "unknown"


def normalize_margin_eps_value(value: float) -> float:
    return 0.0 if abs(float(value)) <= 1e-12 else float(value)


def margin_eps_label(value: float) -> str:
    normalized = normalize_margin_eps_value(value)
    if normalized == 0.0:
        return "0"
    return np.format_float_positional(normalized, trim="-").replace(".", "p")


def selector_audit_strict_metric_field(metric: str, margin_eps: float) -> str:
    return f"selector_audit_{metric}_eps_{margin_eps_label(margin_eps)}"


def format_float_key(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def compute_legacy_score(row: Dict[str, str], agreement_weight: float) -> float:
    return (
        to_float(row, "score_term_legacy_skeleton", 0.0)
        + agreement_weight * to_float(row, "effective_agreement_score", 0.5)
        + to_float(row, "score_term_legacy_density", 0.0)
        + to_float(row, "score_term_legacy_margin", 0.0)
        + to_float(row, "score_term_legacy_asymmetry", 0.0)
    )


def compute_replay_score(
    row: Dict[str, str],
    *,
    selector_mode: str,
    agreement_weight: float,
) -> Tuple[float, str, int]:
    if selector_mode == "legacy":
        return compute_legacy_score(row, agreement_weight), "score_legacy_total", 1
    if selector_mode not in SCORE_FIELD_BY_MODE:
        raise ValueError(f"Unsupported selector mode: {selector_mode}")
    score_field = SCORE_FIELD_BY_MODE[selector_mode]
    return to_float(row, score_field, float("-inf")), score_field, 0


def choose_epoch(
    quality_rows: Sequence[Dict[str, str]],
    *,
    selector_mode: str,
    agreement_weight: float,
) -> Dict[str, Any]:
    fallback_best_row: Dict[str, str] | None = None
    fallback_best_score = float("-inf")
    best_guarded_row: Dict[str, str] | None = None
    best_guarded_score = float("-inf")
    score_field = SCORE_FIELD_BY_MODE[selector_mode]
    agreement_weight_applies = int(selector_mode == "legacy")

    for row in quality_rows:
        selection_eligible = to_int(row, "selection_eligible", 0)
        if selection_eligible <= 0:
            continue
        row_score, score_field, agreement_weight_applies = compute_replay_score(
            row,
            selector_mode=selector_mode,
            agreement_weight=agreement_weight,
        )
        if row_score > fallback_best_score:
            fallback_best_score = row_score
            fallback_best_row = row
        if to_int(row, "guardrail_pass", 0) > 0 and row_score > best_guarded_score:
            best_guarded_score = row_score
            best_guarded_row = row

    if best_guarded_row is not None:
        chosen_row = best_guarded_row
        chosen_score = best_guarded_score
        selection_mode = "guarded"
    elif fallback_best_row is not None:
        chosen_row = fallback_best_row
        chosen_score = fallback_best_score
        selection_mode = "score_only_fallback"
    else:
        chosen_row = quality_rows[-1]
        chosen_score, score_field, agreement_weight_applies = compute_replay_score(
            chosen_row,
            selector_mode=selector_mode,
            agreement_weight=agreement_weight,
        )
        selection_mode = "final_epoch_fallback"

    return {
        "chosen_row": chosen_row,
        "chosen_score": chosen_score,
        "score_field": score_field,
        "agreement_weight_applies": agreement_weight_applies,
        "chosen_selection_mode": selection_mode,
    }


def build_replay_rows(
    run_dir: Path,
    *,
    selector_modes: Sequence[str],
    agreement_weights: Sequence[float],
) -> List[Dict[str, Any]]:
    quality_history_path = run_dir / "quality_history.csv"
    selector_summary_path = run_dir / "selector_audit_summary.csv"
    if not quality_history_path.exists():
        raise FileNotFoundError(f"Missing quality_history.csv in {run_dir}")
    if not selector_summary_path.exists():
        raise FileNotFoundError(f"Missing selector_audit_summary.csv in {run_dir}")

    cfg = load_config(run_dir)
    dataset = dataset_name_from_config(cfg)
    seed = int(cfg.get("seed", -1))
    quality_rows = load_csv_rows(quality_history_path)
    selector_summary = load_single_csv_row(selector_summary_path)
    primary_eps = normalize_margin_eps_value(
        to_float(selector_summary, "selector_audit_primary_margin_eps", 0.0)
    )
    primary_metric_field = selector_audit_strict_metric_field("strict_f1", primary_eps)

    replay_rows: List[Dict[str, Any]] = []
    for selector_mode in selector_modes:
        for agreement_weight in agreement_weights:
            chosen = choose_epoch(
                quality_rows,
                selector_mode=selector_mode,
                agreement_weight=agreement_weight,
            )
            chosen_row = chosen["chosen_row"]
            chosen_primary = to_float(chosen_row, primary_metric_field)
            trained_exported_primary = to_float(
                selector_summary,
                "selector_audit_exported_primary_strict_f1",
            )
            trained_final_primary = to_float(
                selector_summary,
                "selector_audit_final_primary_strict_f1",
            )
            gt_best_primary = to_float(
                selector_summary,
                "selector_audit_best_gt_primary_strict_f1",
            )
            replay_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "original_selection_score_mode": str(cfg.get("selection_score_mode", "")),
                    "original_selection_agreement_weight": float(
                        cfg.get("selection_agreement_weight", 0.25)
                    ),
                    "selector_mode": selector_mode,
                    "agreement_weight_requested": agreement_weight,
                    "agreement_weight_applies": chosen["agreement_weight_applies"],
                    "metric_field": chosen["score_field"],
                    "primary_metric_field": primary_metric_field,
                    "chosen_epoch": to_int(chosen_row, "epoch"),
                    "chosen_score": float(chosen["chosen_score"]),
                    "chosen_selection_mode": str(chosen["chosen_selection_mode"]),
                    "chosen_guardrail_pass": to_int(chosen_row, "guardrail_pass"),
                    "chosen_guardrail_reason": str(chosen_row.get("guardrail_reason", "")),
                    "chosen_primary_strict_f1": chosen_primary,
                    "chosen_strict_f1_eps_0": to_float(
                        chosen_row,
                        "selector_audit_strict_f1_eps_0",
                    ),
                    "chosen_strict_f1_eps_0p1": to_float(
                        chosen_row,
                        "selector_audit_strict_f1_eps_0p1",
                    ),
                    "chosen_failure_mode": str(
                        chosen_row.get("selector_audit_failure_mode", "")
                    ),
                    "chosen_gt_signed_margin_median": to_float(
                        chosen_row,
                        "selector_audit_gt_signed_margin_median",
                    ),
                    "chosen_strict_precision_eps_0": to_float(
                        chosen_row,
                        "selector_audit_strict_precision_eps_0",
                    ),
                    "chosen_strict_recall_eps_0": to_float(
                        chosen_row,
                        "selector_audit_strict_recall_eps_0",
                    ),
                    "chosen_strict_pred_count_eps_0": to_float(
                        chosen_row,
                        "selector_audit_strict_pred_count_eps_0",
                    ),
                    "trained_exported_epoch": to_int(
                        selector_summary,
                        "selector_audit_exported_epoch",
                    ),
                    "trained_exported_primary_strict_f1": trained_exported_primary,
                    "trained_final_epoch": to_int(
                        selector_summary,
                        "selector_audit_final_epoch",
                    ),
                    "trained_final_primary_strict_f1": trained_final_primary,
                    "gt_best_epoch": to_int(selector_summary, "selector_audit_best_gt_epoch"),
                    "gt_best_primary_strict_f1": gt_best_primary,
                    "gt_best_failure_mode": str(
                        selector_summary.get("selector_audit_best_gt_failure_mode", "")
                    ),
                    "gt_best_signed_margin_median": to_float(
                        selector_summary,
                        "selector_audit_best_gt_signed_margin_median",
                    ),
                    "chosen_vs_trained_exported_delta_primary_strict_f1": (
                        chosen_primary - trained_exported_primary
                    ),
                    "chosen_vs_trained_final_delta_primary_strict_f1": (
                        chosen_primary - trained_final_primary
                    ),
                    "chosen_vs_gt_best_delta_primary_strict_f1": (
                        chosen_primary - gt_best_primary
                    ),
                    "matches_trained_exported_epoch": int(
                        to_int(chosen_row, "epoch")
                        == to_int(selector_summary, "selector_audit_exported_epoch")
                    ),
                    "matches_trained_exported_primary_strict_f1": int(
                        abs(chosen_primary - trained_exported_primary) <= 1e-12
                    ),
                }
            )

    return replay_rows


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["selector_mode"]),
            format_float_key(float(row["agreement_weight_requested"])),
        )
        groups[key].append(dict(row))

    aggregate_rows_out: List[Dict[str, Any]] = []
    numeric_metrics = [
        "chosen_epoch",
        "chosen_score",
        "chosen_primary_strict_f1",
        "chosen_strict_f1_eps_0",
        "chosen_strict_f1_eps_0p1",
        "chosen_gt_signed_margin_median",
        "chosen_vs_trained_exported_delta_primary_strict_f1",
        "chosen_vs_trained_final_delta_primary_strict_f1",
        "chosen_vs_gt_best_delta_primary_strict_f1",
    ]

    for (dataset, selector_mode, agreement_weight_key), group_rows in sorted(groups.items()):
        agg: Dict[str, Any] = {
            "dataset": dataset,
            "selector_mode": selector_mode,
            "agreement_weight_requested": agreement_weight_key,
            "agreement_weight_applies": int(group_rows[0]["agreement_weight_applies"]),
            "run_count": len(group_rows),
            "seed_list": ",".join(str(row["seed"]) for row in group_rows),
            "run_dirs": ";".join(str(row["run_dir"]) for row in group_rows),
            "selection_mode_counts": dict(
                Counter(str(row["chosen_selection_mode"]) for row in group_rows)
            ),
            "failure_mode_counts": dict(
                Counter(str(row["chosen_failure_mode"]) for row in group_rows)
            ),
            "matches_trained_exported_epoch_count": sum(
                int(row["matches_trained_exported_epoch"]) for row in group_rows
            ),
            "matches_trained_exported_primary_strict_f1_count": sum(
                int(row["matches_trained_exported_primary_strict_f1"]) for row in group_rows
            ),
        }
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in group_rows]
            agg[f"{metric}_mean"] = mean(values)
            agg[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregate_rows_out.append(agg)
    return aggregate_rows_out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_dataset_label(rows: Sequence[Dict[str, Any]]) -> str:
    datasets = sorted({str(row["dataset"]) for row in rows})
    if not datasets:
        return "unknown"
    if len(datasets) == 1:
        return datasets[0]
    return "mixed"


def main() -> None:
    args = parse_args()
    selector_modes = parse_csv_list(args.selector_modes)
    agreement_weights = parse_float_list(args.agreement_weights)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows: List[Dict[str, Any]] = []
    for run_dir_text in args.run_dirs:
        run_dir = Path(run_dir_text).resolve()
        all_rows.extend(
            build_replay_rows(
                run_dir,
                selector_modes=selector_modes,
                agreement_weights=agreement_weights,
            )
        )

    dataset_label = infer_dataset_label(all_rows)
    output_stem = (
        args.output_stem
        if args.output_stem
        else f"unify_phase0_selector_{dataset_label}_{timestamp}_{args.tag}"
    )
    summary_path = RESULTS_DIR / f"{output_stem}.csv"
    aggregate_path = RESULTS_DIR / f"{output_stem}_aggregate.csv"
    aggregate = aggregate_rows(all_rows)

    write_csv(summary_path, all_rows)
    write_csv(aggregate_path, aggregate)

    print(f"Summary written to: {summary_path}")
    print(f"Aggregate written to: {aggregate_path}")


if __name__ == "__main__":
    main()
