from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from run_replay_saved_config import (
    RESULTS_DIR,
    aggregate_rows,
    apply_overrides,
    dataset_name_from_cfg,
    load_config,
    read_epoch_csv_row,
    read_single_row_csv,
    to_float,
    write_aggregate_csv,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build replay summary/aggregate CSVs from existing completed run dirs."
    )
    parser.add_argument("--base_run_dir", type=Path, required=True)
    parser.add_argument("--run_dirs", type=str, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def parse_csv_ints(text: str) -> List[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def parse_run_dirs(text: str) -> List[Path]:
    return [Path(token.strip()).resolve() for token in text.split(";") if token.strip()]


def row_from_run_dir(
    cfg: Dict[str, Any],
    *,
    seed: int,
    base_run_dir: Path,
    result_dir: Path,
    override_spec: str,
) -> Dict[str, Any]:
    selector = read_single_row_csv(result_dir / "selector_audit_summary.csv")
    best_epoch = int(to_float(selector, "selector_audit_best_gt_epoch"))
    exported_epoch = int(to_float(selector, "selector_audit_exported_epoch"))
    final_epoch = int(to_float(selector, "selector_audit_final_epoch"))
    selector_epoch = (
        int(to_float(selector, "selector_audit_selector_epoch"))
        if selector.get("selector_audit_selector_epoch")
        else exported_epoch
    )
    quality_history_path = result_dir / "quality_history.csv"
    best_quality = read_epoch_csv_row(quality_history_path, best_epoch)
    exported_quality = read_epoch_csv_row(quality_history_path, exported_epoch)
    final_quality = read_epoch_csv_row(quality_history_path, final_epoch)

    return {
        "dataset": dataset_name_from_cfg(cfg),
        "seed": seed,
        "base_run_dir": str(base_run_dir),
        "override_spec": override_spec,
        "result_dir": str(result_dir),
        "best_primary_strict_f1": to_float(selector, "selector_audit_best_gt_primary_strict_f1"),
        "exported_primary_strict_f1": to_float(selector, "selector_audit_exported_primary_strict_f1"),
        "final_primary_strict_f1": to_float(selector, "selector_audit_final_primary_strict_f1"),
        "best_epoch": best_epoch,
        "exported_epoch": exported_epoch,
        "selector_epoch": selector_epoch,
        "export_epoch_policy": selector.get(
            "selector_audit_export_epoch_policy",
            str(cfg.get("export_epoch_policy", "")),
        ),
        "diffusion_noise_mode": str(cfg.get("diffusion_noise_mode", "guided")),
        "final_epoch": final_epoch,
        "best_failure_mode": selector.get("selector_audit_best_gt_failure_mode", ""),
        "exported_failure_mode": selector.get("selector_audit_exported_failure_mode", ""),
        "final_failure_mode": selector.get("selector_audit_final_failure_mode", ""),
        "best_signed_margin_median": to_float(selector, "selector_audit_best_gt_signed_margin_median"),
        "exported_signed_margin_median": to_float(selector, "selector_audit_exported_signed_margin_median"),
        "final_signed_margin_median": to_float(selector, "selector_audit_final_signed_margin_median"),
        "best_exported_adj_margin_median": to_float(selector, "selector_audit_best_gt_exported_signed_margin_median"),
        "exported_exported_adj_margin_median": to_float(selector, "selector_audit_exported_gt_exported_signed_margin_median"),
        "final_exported_adj_margin_median": to_float(selector, "selector_audit_final_gt_exported_signed_margin_median"),
        "best_gate_margin_median": to_float(selector, "selector_audit_best_gt_gate_signed_margin_median"),
        "exported_gate_margin_median": to_float(selector, "selector_audit_exported_gt_gate_signed_margin_median"),
        "final_gate_margin_median": to_float(selector, "selector_audit_final_gt_gate_signed_margin_median"),
        "best_support_median": to_float(selector, "selector_audit_best_gt_support_median"),
        "exported_support_median": to_float(selector, "selector_audit_exported_gt_support_median"),
        "final_support_median": to_float(selector, "selector_audit_final_gt_support_median"),
        "best_support_p10": to_float(selector, "selector_audit_best_gt_support_p10"),
        "exported_support_p10": to_float(selector, "selector_audit_exported_gt_support_p10"),
        "final_support_p10": to_float(selector, "selector_audit_final_gt_support_p10"),
        "best_strict_f1_eps_0p1": to_float(best_quality, "selector_audit_strict_f1_eps_0p1"),
        "exported_strict_f1_eps_0p1": to_float(exported_quality, "selector_audit_strict_f1_eps_0p1"),
        "final_strict_f1_eps_0p1": to_float(final_quality, "selector_audit_strict_f1_eps_0p1"),
        "exported_vs_best_gap": to_float(selector, "selector_audit_exported_vs_best_gt_gap_primary_strict_f1"),
        "final_vs_best_gap": to_float(selector, "selector_audit_final_vs_best_gt_gap_primary_strict_f1"),
    }


def main() -> None:
    args = parse_args()
    base_run_dir = args.base_run_dir.resolve()
    cfg = apply_overrides(load_config(base_run_dir), args.overrides)
    seeds = parse_csv_ints(args.seeds)
    run_dirs = parse_run_dirs(args.run_dirs)
    if len(seeds) != len(run_dirs):
        raise ValueError(f"Expected equal seeds/run_dirs counts, got {len(seeds)} and {len(run_dirs)}")

    override_spec = ";".join(args.overrides)
    rows = [
        row_from_run_dir(
            cfg,
            seed=seed,
            base_run_dir=base_run_dir,
            result_dir=run_dir,
            override_spec=override_spec,
        )
        for seed, run_dir in zip(seeds, run_dirs)
    ]

    dataset = dataset_name_from_cfg(cfg)
    output_stem = f"unify_replay_{dataset}_{args.tag}"
    summary_path = RESULTS_DIR / f"{output_stem}.csv"
    aggregate_path = RESULTS_DIR / f"{output_stem}_aggregate.csv"
    write_summary_csv(summary_path, rows)
    write_aggregate_csv(aggregate_path, aggregate_rows(rows))
    print(f"Summary written to: {summary_path}")
    print(f"Aggregate written to: {aggregate_path}")


if __name__ == "__main__":
    main()
