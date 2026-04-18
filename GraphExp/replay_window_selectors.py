from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Sequence

from replay_selector_modes import (
    DEFAULT_SELECTOR_MODES,
    choose_epoch,
    dataset_name_from_config,
    load_config,
    load_csv_rows,
    load_single_csv_row,
    normalize_margin_eps_value,
    parse_csv_list,
    selector_audit_strict_metric_field,
    to_float,
    to_int,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline replay for simple late-window checkpoint selection policies "
            "on saved quality_history.csv trajectories."
        )
    )
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Run directories that contain quality_history.csv and selector_audit_summary.csv.",
    )
    parser.add_argument(
        "--selector_modes",
        type=str,
        default=",".join(DEFAULT_SELECTOR_MODES),
        help="Comma-separated selector modes to replay inside each window.",
    )
    parser.add_argument(
        "--window_sizes",
        type=str,
        default="3,5,8,10",
        help="Comma-separated late-window sizes counted from the final epoch.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="window_selector_replay",
        help="Suffix used in output filenames.",
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default=None,
        help="Optional explicit output stem without extension.",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def safe_pstdev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def choose_oracle_epoch(
    quality_rows: Sequence[Dict[str, str]],
    *,
    primary_metric_field: str,
) -> Dict[str, Any]:
    eligible_rows = [row for row in quality_rows if to_int(row, "selection_eligible", 0) > 0]
    pool = eligible_rows if eligible_rows else list(quality_rows)
    chosen_row = max(pool, key=lambda row: to_float(row, primary_metric_field, float("-inf")))
    return {
        "chosen_row": chosen_row,
        "chosen_selection_mode": "oracle_gt",
    }


def filter_last_k_rows(
    quality_rows: Sequence[Dict[str, str]],
    *,
    window_size: int,
) -> List[Dict[str, str]]:
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    max_epoch = max(to_int(row, "epoch", 0) for row in quality_rows)
    min_epoch = max_epoch - window_size + 1
    return [row for row in quality_rows if to_int(row, "epoch", 0) >= min_epoch]


def build_policy_rows(
    run_dir: Path,
    *,
    selector_modes: Sequence[str],
    window_sizes: Sequence[int],
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

    global_oracle = choose_oracle_epoch(
        quality_rows,
        primary_metric_field=primary_metric_field,
    )
    global_oracle_row = global_oracle["chosen_row"]
    final_row = quality_rows[-1]

    policy_rows: List[Dict[str, Any]] = []

    def append_row(
        *,
        policy_family: str,
        policy_name: str,
        selector_mode: str,
        window_size: int,
        chosen: Dict[str, Any],
    ) -> None:
        chosen_row = chosen["chosen_row"]
        chosen_primary = to_float(chosen_row, primary_metric_field)
        global_primary = to_float(global_oracle_row, primary_metric_field)
        final_primary = to_float(final_row, primary_metric_field)
        policy_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "run_dir": str(run_dir),
                "policy_family": policy_family,
                "policy_name": policy_name,
                "selector_mode": selector_mode,
                "window_size": window_size,
                "chosen_epoch": to_int(chosen_row, "epoch", 0),
                "chosen_primary_strict_f1": chosen_primary,
                "chosen_strict_f1_eps_0p1": to_float(
                    chosen_row,
                    selector_audit_strict_metric_field("strict_f1", 0.1),
                ),
                "chosen_failure_mode": chosen_row.get("selector_audit_failure_mode", ""),
                "chosen_selection_mode": chosen.get("chosen_selection_mode", ""),
                "final_epoch": to_int(final_row, "epoch", 0),
                "final_primary_strict_f1": final_primary,
                "global_best_epoch": to_int(global_oracle_row, "epoch", 0),
                "global_best_primary_strict_f1": global_primary,
                "chosen_vs_global_best_delta_primary_strict_f1": chosen_primary - global_primary,
                "final_vs_global_best_delta_primary_strict_f1": final_primary - global_primary,
            }
        )

    append_row(
        policy_family="global_oracle",
        policy_name="global_oracle_gt",
        selector_mode="oracle_gt",
        window_size=-1,
        chosen=global_oracle,
    )
    append_row(
        policy_family="final",
        policy_name="final_epoch",
        selector_mode="final",
        window_size=1,
        chosen={"chosen_row": final_row, "chosen_selection_mode": "final_epoch"},
    )

    for window_size in window_sizes:
        window_rows = filter_last_k_rows(quality_rows, window_size=window_size)
        window_oracle = choose_oracle_epoch(
            window_rows,
            primary_metric_field=primary_metric_field,
        )
        append_row(
            policy_family="window_oracle",
            policy_name=f"last{window_size}_oracle_gt",
            selector_mode="oracle_gt",
            window_size=window_size,
            chosen=window_oracle,
        )
        for selector_mode in selector_modes:
            chosen = choose_epoch(
                window_rows,
                selector_mode=selector_mode,
                agreement_weight=0.0,
            )
            append_row(
                policy_family="window_selector",
                policy_name=f"{selector_mode}_last{window_size}",
                selector_mode=selector_mode,
                window_size=window_size,
                chosen=chosen,
            )

    return policy_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["policy_family"],
            row["policy_name"],
            row["selector_mode"],
            row["window_size"],
        )
        groups[key].append(row)

    aggregated: List[Dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        dataset, policy_family, policy_name, selector_mode, window_size = key
        chosen_epochs = [int(row["chosen_epoch"]) for row in group_rows]
        chosen_primary = [float(row["chosen_primary_strict_f1"]) for row in group_rows]
        chosen_eps01 = [float(row["chosen_strict_f1_eps_0p1"]) for row in group_rows]
        chosen_delta = [
            float(row["chosen_vs_global_best_delta_primary_strict_f1"]) for row in group_rows
        ]
        final_delta = [
            float(row["final_vs_global_best_delta_primary_strict_f1"]) for row in group_rows
        ]
        exact_hits = sum(abs(delta) <= 1e-9 for delta in chosen_delta)
        noninferior_hits = sum(delta >= -0.03 for delta in chosen_delta)
        aggregated.append(
            {
                "dataset": dataset,
                "policy_family": policy_family,
                "policy_name": policy_name,
                "selector_mode": selector_mode,
                "window_size": window_size,
                "run_count": len(group_rows),
                "seed_list": ",".join(str(row["seed"]) for row in group_rows),
                "selection_mode_counts": dict(
                    Counter(row["chosen_selection_mode"] for row in group_rows)
                ),
                "failure_mode_counts": dict(
                    Counter(row["chosen_failure_mode"] for row in group_rows)
                ),
                "chosen_epoch_mean": mean(chosen_epochs),
                "chosen_epoch_std": safe_pstdev(chosen_epochs),
                "chosen_primary_strict_f1_mean": mean(chosen_primary),
                "chosen_primary_strict_f1_std": safe_pstdev(chosen_primary),
                "chosen_strict_f1_eps_0p1_mean": mean(chosen_eps01),
                "chosen_strict_f1_eps_0p1_std": safe_pstdev(chosen_eps01),
                "chosen_vs_global_best_delta_primary_strict_f1_mean": mean(chosen_delta),
                "chosen_vs_global_best_delta_primary_strict_f1_std": safe_pstdev(chosen_delta),
                "final_vs_global_best_delta_primary_strict_f1_mean": mean(final_delta),
                "final_vs_global_best_delta_primary_strict_f1_std": safe_pstdev(final_delta),
                "exact_global_best_match_count": exact_hits,
                "noninferior_match_count_delta_ge_neg0p03": noninferior_hits,
            }
        )
    return aggregated


def main() -> None:
    args = parse_args()
    selector_modes = parse_csv_list(args.selector_modes)
    window_sizes = parse_int_list(args.window_sizes)

    all_rows: List[Dict[str, Any]] = []
    for run_dir_text in args.run_dirs:
        all_rows.extend(
            build_policy_rows(
                Path(run_dir_text),
                selector_modes=selector_modes,
                window_sizes=window_sizes,
            )
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets = sorted({row["dataset"] for row in all_rows})
    dataset_label = datasets[0] if len(datasets) == 1 else "multi"
    output_stem = args.output_stem or f"window_replay_{dataset_label}_{timestamp}_{args.tag}"
    summary_path = RESULTS_DIR / f"{output_stem}.csv"
    aggregate_path = RESULTS_DIR / f"{output_stem}_aggregate.csv"

    fieldnames = list(all_rows[0].keys())
    write_csv(summary_path, all_rows, fieldnames)
    aggregate_rows_data = aggregate_rows(all_rows)
    write_csv(aggregate_path, aggregate_rows_data, list(aggregate_rows_data[0].keys()))

    print(f"Summary written to: {summary_path}")
    print(f"Aggregate written to: {aggregate_path}")


if __name__ == "__main__":
    main()
