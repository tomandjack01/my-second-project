from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULT_DIR_PATTERN = re.compile(r"Results will be saved to:\s*(.+)")


# Conservative defaults for older saved configs that predate some fields.
DEFAULTS: Dict[str, Any] = {
    "device": "cuda",
    "epochs": 100,
    "pretrain_epochs": 50,
    "pretrain_lr": 1e-3,
    "subject_limit": -1,
    "time_limit": -1,
    "lambda_l1": 0.02,
    "main_loss_weight": 1.0,
    "optimizer_step_mode": "subject",
    "log_interval": 10,
    "top_k_edges": 50,
    "selection_agreement_weight": 0.25,
    "selection_agreement_mode": "hard_coverage",
    "selection_score_mode": "legacy",
    "selection_soft_agreement_weight": 0.20,
    "selection_causal_lag_weight": 1.0,
    "selection_margin_penalty_weight": 0.05,
    "selection_causal_lag_subject_limit": -1,
    "selection_causal_lag_std_penalty_weight": 0.0,
    "selection_parent_entropy_penalty_weight": 0.0,
    "selection_primary_causal_lag_weight": 1.0,
    "selection_primary_soft_tiebreak_weight": 0.05,
    "selection_primary_skeleton_tiebreak_weight": 0.05,
    "selection_primary_density_tiebreak_weight": 0.0,
    "selection_start_epoch": 6,
    "selection_top_k": None,
    "selector_audit_strict_margin_eps_values": "0,3e-4,0.1",
    "structure_init_mode": "patel_score",
    "support_prior_mode": "patel_kappa",
    "support_prior_algorithm": "patel",
    "direction_prior_algorithm": "patel",
    "structure_init_scale": 1.0,
    "emb_dim": 0,
    "structure_parameterization": "coupled",
    "direction_parameterization": "factorized",
    "fixed_support_mask_mode": "none",
    "direction_init_mode": "patel_tau",
    "structure_message_graph_mode": "raw",
    "structure_message_edge_mode": "full",
    "adj_activation": "sigmoid",
    "kappa_logit_bias_scale": 0.0,
    "direction_logit_bias_scale": 0.0,
    "direction_lr_multiplier": 1.0,
    "freeze_direction_after_epoch": -1,
    "detach_direction_from_main_after_epoch": -1,
    "gradient_routing_mode": "legacy",
    "parent_entropy_lambda": 0.0,
    "parent_entropy_warmup_epochs": 0,
    "parent_entropy_ramp_epochs": 1,
    "parent_cap_lambda": 0.0,
    "parent_cap_target": 0.0,
    "parent_cap_warmup_epochs": 0,
    "parent_cap_ramp_epochs": 1,
    "ungated_symmetry_lambda": 0.0,
    "ungated_symmetry_warmup_epochs": 0,
    "ungated_symmetry_ramp_epochs": 1,
    "self_distill_direction_retention_lambda": 0.0,
    "self_distill_direction_retention_start_epoch": -1,
    "self_distill_direction_retention_ema": 0.9,
    "self_distill_direction_retention_active_quantile": 0.5,
    "self_distill_direction_retention_margin_scale": 0.5,
    "self_distill_direction_retention_margin_floor": 0.0,
    "directional_schedule": "cosine_anneal",
    "directional_target_ratio": 0.01,
    "directional_loss_end_epoch": -1,
    "directional_kappa_gate_quantile": 0.5,
    "causal_lag_main_weight": 0.0,
    "causal_lag_main_aggregation": "mean",
    "causal_lag_main_softmax_temp": 1.0,
    "causal_lag_main_lags": "1",
    "causal_lag_main_lag_weights": "",
    "batch_size": 4,
    "lr": 1e-3,
    "num_hidden": 64,
    "num_layers": 2,
    "loss_type": "denoise_hybrid",
    "cosine_weight": 0.1,
    "mse_weight": 0.1,
    "noise_norm_mode": "global",
    "training_noise_guide_mode": "fixed_patel",
    "training_noise_guide_blend_target": 0.5,
    "training_noise_guide_warmup_epochs": 5,
    "training_noise_guide_ramp_epochs": 5,
    "skip_pretrain": False,
    "disable_temporal_encoder": False,
    "debug_checks": False,
    "directional_kappa_gate": False,
    "enable_gradient_alignment_probe": False,
    "save_support_direction_snapshots": False,
    "disable_directional_loss": False,
    "uniform_timestep": True,
    "noise_zero_mean": True,
}


VALUE_KEYS: List[str] = [
    "csv_path",
    "device",
    "epochs",
    "pretrain_epochs",
    "pretrain_lr",
    "subject_limit",
    "time_limit",
    "lambda_l1",
    "main_loss_weight",
    "optimizer_step_mode",
    "log_interval",
    "top_k_edges",
    "selection_agreement_weight",
    "selection_agreement_mode",
    "selection_score_mode",
    "selection_soft_agreement_weight",
    "selection_causal_lag_weight",
    "selection_margin_penalty_weight",
    "selection_causal_lag_subject_limit",
    "selection_causal_lag_std_penalty_weight",
    "selection_parent_entropy_penalty_weight",
    "selection_primary_causal_lag_weight",
    "selection_primary_soft_tiebreak_weight",
    "selection_primary_skeleton_tiebreak_weight",
    "selection_primary_density_tiebreak_weight",
    "selection_start_epoch",
    "selection_top_k",
    "selector_audit_gt_path",
    "selector_audit_strict_margin_eps_values",
    "structure_init_mode",
    "support_prior_mode",
    "support_prior_algorithm",
    "direction_prior_algorithm",
    "structure_init_scale",
    "emb_dim",
    "structure_parameterization",
    "direction_parameterization",
    "fixed_support_mask_mode",
    "direction_init_mode",
    "structure_message_graph_mode",
    "structure_message_edge_mode",
    "adj_activation",
    "kappa_logit_bias_scale",
    "direction_logit_bias_scale",
    "direction_lr_multiplier",
    "freeze_direction_after_epoch",
    "detach_direction_from_main_after_epoch",
    "gradient_routing_mode",
    "parent_entropy_lambda",
    "parent_entropy_warmup_epochs",
    "parent_entropy_ramp_epochs",
    "parent_cap_lambda",
    "parent_cap_target",
    "parent_cap_warmup_epochs",
    "parent_cap_ramp_epochs",
    "ungated_symmetry_lambda",
    "ungated_symmetry_warmup_epochs",
    "ungated_symmetry_ramp_epochs",
    "self_distill_direction_retention_lambda",
    "self_distill_direction_retention_start_epoch",
    "self_distill_direction_retention_ema",
    "self_distill_direction_retention_active_quantile",
    "self_distill_direction_retention_margin_scale",
    "self_distill_direction_retention_margin_floor",
    "directional_schedule",
    "directional_target_ratio",
    "directional_loss_end_epoch",
    "directional_kappa_gate_quantile",
    "causal_lag_main_weight",
    "causal_lag_main_aggregation",
    "causal_lag_main_softmax_temp",
    "causal_lag_main_lags",
    "causal_lag_main_lag_weights",
    "batch_size",
    "lr",
    "num_hidden",
    "num_layers",
    "loss_type",
    "cosine_weight",
    "mse_weight",
    "noise_norm_mode",
    "training_noise_guide_mode",
    "training_noise_guide_blend_target",
    "training_noise_guide_warmup_epochs",
    "training_noise_guide_ramp_epochs",
]


NUMERIC_METRICS: List[str] = [
    "selector_audit_best_gt_primary_strict_f1",
    "selector_audit_exported_primary_strict_f1",
    "selector_audit_final_primary_strict_f1",
    "selector_audit_best_gt_epoch",
    "selector_audit_exported_epoch",
    "selector_audit_final_epoch",
    "selector_audit_exported_vs_best_gt_gap_primary_strict_f1",
    "selector_audit_final_vs_best_gt_gap_primary_strict_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a saved GraphExp run config across multiple seeds."
    )
    parser.add_argument(
        "--base_run_dir",
        type=Path,
        required=True,
        help="Existing run directory that contains config.npy.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="11,22,33,44,55",
        help="Comma-separated replay seeds.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override. If omitted, uses the saved config/device default.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="replay",
        help="Short suffix used in the summary/aggregate filenames.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a saved config entry with KEY=VALUE. May be passed multiple times.",
    )
    return parser.parse_args()


def parse_seed_list(text: str) -> List[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def load_config(base_run_dir: Path) -> Dict[str, Any]:
    config_path = base_run_dir / "config.npy"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.npy under {base_run_dir}")
    cfg = np.load(config_path, allow_pickle=True).item()
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict-like config in {config_path}")
    return dict(cfg)


def parse_bool_text(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean override value: {text!r}")


def coerce_override_value(raw_value: str, reference_value: Any) -> Any:
    normalized = raw_value.strip()
    lowered = normalized.lower()
    if lowered in {"none", "null"}:
        return None
    if isinstance(reference_value, bool):
        return parse_bool_text(normalized)
    if isinstance(reference_value, int) and not isinstance(reference_value, bool):
        return int(normalized)
    if isinstance(reference_value, float):
        return float(normalized)
    if reference_value is None:
        if lowered in {"true", "false", "yes", "no", "on", "off", "1", "0"}:
            return parse_bool_text(normalized)
        try:
            return int(normalized)
        except ValueError:
            pass
        try:
            return float(normalized)
        except ValueError:
            pass
        return normalized
    return normalized


def apply_overrides(cfg: Dict[str, Any], override_specs: Iterable[str]) -> Dict[str, Any]:
    updated = dict(cfg)
    for spec in override_specs:
        if "=" not in spec:
            raise ValueError(f"Override must be KEY=VALUE, got {spec!r}")
        key, raw_value = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {spec!r}")
        reference_value = cfg[key] if key in cfg else DEFAULTS.get(key)
        updated[key] = coerce_override_value(raw_value, reference_value)
    return updated


def get_config_value(cfg: Dict[str, Any], key: str) -> Any:
    if key in cfg:
        return cfg[key]
    return DEFAULTS.get(key)


def build_command(cfg: Dict[str, Any], seed: int, device_override: Optional[str]) -> List[str]:
    cmd = [sys.executable, "main_structure_learning.py"]

    for key in VALUE_KEYS:
        value = get_config_value(cfg, key)
        if value is None:
            continue
        if key == "device" and device_override is not None:
            value = device_override
        if key == "selection_top_k" and value is None:
            continue
        if key == "pretrain_checkpoint" and not value:
            continue
        if key == "selector_audit_gt_path" and not value:
            continue
        cli_key = f"--{key}"
        cmd.extend([cli_key, str(value)])

    pretrain_checkpoint = cfg.get("pretrain_checkpoint")
    if pretrain_checkpoint:
        cmd.extend(["--pretrain_checkpoint", str(pretrain_checkpoint)])

    cmd.extend(["--seed", str(seed)])

    if bool(get_config_value(cfg, "skip_pretrain")):
        cmd.append("--skip_pretrain")
    if bool(get_config_value(cfg, "disable_temporal_encoder")):
        cmd.append("--disable_temporal_encoder")
    if bool(get_config_value(cfg, "debug_checks")):
        cmd.append("--debug_checks")
    if bool(get_config_value(cfg, "directional_kappa_gate")):
        cmd.append("--directional_kappa_gate")
    if bool(get_config_value(cfg, "enable_gradient_alignment_probe")):
        cmd.append("--enable_gradient_alignment_probe")
    if bool(get_config_value(cfg, "save_support_direction_snapshots")):
        cmd.append("--save_support_direction_snapshots")
    if bool(get_config_value(cfg, "disable_directional_loss")):
        cmd.append("--disable_directional_loss")
    if not bool(get_config_value(cfg, "uniform_timestep")):
        cmd.append("--per_node_timestep")
    if not bool(get_config_value(cfg, "noise_zero_mean")):
        cmd.append("--noise_with_mean")

    return cmd


def parse_result_dir(stdout_text: str) -> Path:
    match = RESULT_DIR_PATTERN.search(stdout_text)
    if not match:
        raise RuntimeError("Could not parse result directory from training stdout.")
    result_dir = Path(match.group(1).strip())
    if not result_dir.is_absolute():
        result_dir = (SCRIPT_DIR / result_dir).resolve()
    return result_dir


def read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        raise RuntimeError(f"Expected one-row CSV at {path}, found no data.")
    return row


def read_epoch_csv_row(path: Path, epoch: int) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(float(row["epoch"])) == epoch:
                return row
    raise RuntimeError(f"Could not find epoch={epoch} in {path}")


def to_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return float("nan")
    return float(value)


def dataset_name_from_cfg(cfg: Dict[str, Any]) -> str:
    csv_path = str(get_config_value(cfg, "csv_path"))
    return Path(csv_path).stem


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "run_count": len(rows),
        "seed_list": ",".join(str(row["seed"]) for row in rows),
        "result_dirs": ";".join(str(row["result_dir"]) for row in rows),
        "override_spec": rows[0].get("override_spec", ""),
        "failure_mode_counts": dict(Counter(str(row["best_failure_mode"]) for row in rows)),
        "final_failure_mode_counts": dict(Counter(str(row["final_failure_mode"]) for row in rows)),
    }
    for metric in (
        "best_primary_strict_f1",
        "exported_primary_strict_f1",
        "final_primary_strict_f1",
        "best_signed_margin_median",
        "exported_signed_margin_median",
        "final_signed_margin_median",
        "best_exported_adj_margin_median",
        "exported_exported_adj_margin_median",
        "final_exported_adj_margin_median",
        "best_gate_margin_median",
        "exported_gate_margin_median",
        "final_gate_margin_median",
        "best_support_median",
        "exported_support_median",
        "final_support_median",
        "best_support_p10",
        "exported_support_p10",
        "final_support_p10",
        "best_strict_f1_eps_0p1",
        "exported_strict_f1_eps_0p1",
        "final_strict_f1_eps_0p1",
        "exported_vs_best_gap",
        "final_vs_best_gap",
    ):
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = mean(values)
        result[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return result


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_csv(path: Path, row: Dict[str, Any]) -> None:
    fieldnames = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def run_single_seed(
    cfg: Dict[str, Any],
    *,
    seed: int,
    base_run_dir: Path,
    device_override: Optional[str],
    tag: str,
    dry_run: bool,
    override_spec: str,
) -> Optional[Dict[str, Any]]:
    cmd = build_command(cfg, seed=seed, device_override=device_override)
    if dry_run:
        print("DRY RUN:", subprocess.list2cmdline(cmd))
        return None

    proc = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Replay run failed for seed={seed}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    result_dir = parse_result_dir(proc.stdout)
    stdout_path = result_dir / f"{tag}_stdout.log"
    stderr_path = result_dir / f"{tag}_stderr.log"
    save_text(stdout_path, proc.stdout)
    save_text(stderr_path, proc.stderr)

    selector_path = result_dir / "selector_audit_summary.csv"
    quality_history_path = result_dir / "quality_history.csv"
    if not selector_path.exists():
        raise FileNotFoundError(f"Missing selector_audit_summary.csv in {result_dir}")
    if not quality_history_path.exists():
        raise FileNotFoundError(f"Missing quality_history.csv in {result_dir}")
    selector = read_single_row_csv(selector_path)
    best_epoch = int(to_float(selector, "selector_audit_best_gt_epoch"))
    exported_epoch = int(to_float(selector, "selector_audit_exported_epoch"))
    final_epoch = int(to_float(selector, "selector_audit_final_epoch"))
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
        "final_epoch": final_epoch,
        "best_failure_mode": selector.get("selector_audit_best_gt_failure_mode", ""),
        "exported_failure_mode": selector.get("selector_audit_exported_failure_mode", ""),
        "final_failure_mode": selector.get("selector_audit_final_failure_mode", ""),
        "best_signed_margin_median": to_float(selector, "selector_audit_best_gt_signed_margin_median"),
        "exported_signed_margin_median": to_float(selector, "selector_audit_exported_signed_margin_median"),
        "final_signed_margin_median": to_float(selector, "selector_audit_final_signed_margin_median"),
        "best_exported_adj_margin_median": to_float(
            selector,
            "selector_audit_best_gt_exported_signed_margin_median",
        ),
        "exported_exported_adj_margin_median": to_float(
            selector,
            "selector_audit_exported_gt_exported_signed_margin_median",
        ),
        "final_exported_adj_margin_median": to_float(
            selector,
            "selector_audit_final_gt_exported_signed_margin_median",
        ),
        "best_gate_margin_median": to_float(
            selector,
            "selector_audit_best_gt_gate_signed_margin_median",
        ),
        "exported_gate_margin_median": to_float(
            selector,
            "selector_audit_exported_gt_gate_signed_margin_median",
        ),
        "final_gate_margin_median": to_float(
            selector,
            "selector_audit_final_gt_gate_signed_margin_median",
        ),
        "best_support_median": to_float(
            selector,
            "selector_audit_best_gt_support_median",
        ),
        "exported_support_median": to_float(
            selector,
            "selector_audit_exported_gt_support_median",
        ),
        "final_support_median": to_float(
            selector,
            "selector_audit_final_gt_support_median",
        ),
        "best_support_p10": to_float(
            selector,
            "selector_audit_best_gt_support_p10",
        ),
        "exported_support_p10": to_float(
            selector,
            "selector_audit_exported_gt_support_p10",
        ),
        "final_support_p10": to_float(
            selector,
            "selector_audit_final_gt_support_p10",
        ),
        "best_strict_f1_eps_0p1": to_float(best_quality, "selector_audit_strict_f1_eps_0p1"),
        "exported_strict_f1_eps_0p1": to_float(exported_quality, "selector_audit_strict_f1_eps_0p1"),
        "final_strict_f1_eps_0p1": to_float(final_quality, "selector_audit_strict_f1_eps_0p1"),
        "exported_vs_best_gap": to_float(
            selector,
            "selector_audit_exported_vs_best_gt_gap_primary_strict_f1",
        ),
        "final_vs_best_gap": to_float(
            selector,
            "selector_audit_final_vs_best_gt_gap_primary_strict_f1",
        ),
    }


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.base_run_dir.resolve())
    cfg = apply_overrides(cfg, args.overrides)
    override_spec = ";".join(args.overrides)
    seeds = parse_seed_list(args.seeds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = dataset_name_from_cfg(cfg)
    output_stem = f"unify_replay_{dataset}_{timestamp}_{args.tag}"

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"=== Replay {dataset} seed={seed} ===")
        row = run_single_seed(
            cfg,
            seed=seed,
            base_run_dir=args.base_run_dir.resolve(),
            device_override=args.device,
            tag=args.tag,
            dry_run=args.dry_run,
            override_spec=override_spec,
        )
        if row is not None:
            rows.append(row)
            print(
                f"seed={seed} "
                f"best={row['best_primary_strict_f1']:.6f} "
                f"exported={row['exported_primary_strict_f1']:.6f} "
                f"final={row['final_primary_strict_f1']:.6f}"
            )

    if args.dry_run:
        sys.exit(0)

    if not rows:
        raise RuntimeError("No replay rows were collected.")

    summary_path = RESULTS_DIR / f"{output_stem}.csv"
    aggregate_path = RESULTS_DIR / f"{output_stem}_aggregate.csv"
    aggregate_row = aggregate_rows(rows)

    write_summary_csv(summary_path, rows)
    write_aggregate_csv(aggregate_path, aggregate_row)

    print(f"Summary written to: {summary_path}")
    print(f"Aggregate written to: {aggregate_path}")
