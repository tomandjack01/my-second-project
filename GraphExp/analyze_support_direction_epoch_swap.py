import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from main_structure_learning import (
    load_gt_edges,
    selector_audit_evaluate_directional_strict,
    selector_audit_margin_stats,
    to_causal_matrix_np,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze support/direction epoch-swap combinations from "
            "support_direction_snapshots.npz."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing support_direction_snapshots.npz and config.npy.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Optional GT path override. Defaults to selector_audit_gt_path in config.npy.",
    )
    parser.add_argument(
        "--margin_eps",
        type=float,
        default=0.0,
        help="Strict directional margin eps used for F1 evaluation.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="epoch_swap",
        help="Short suffix used in the output filenames.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top epoch-swap rows written to the companion top-N CSV.",
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


def metric_from_selector_row(row: Dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in {"", None} else float("nan")


def build_epoch_index(epochs: np.ndarray) -> Dict[int, int]:
    return {int(epoch): idx for idx, epoch in enumerate(epochs.tolist())}


def eval_combo(
    *,
    support_epoch: int,
    direction_epoch: int,
    support_weights: np.ndarray,
    direction_gate: np.ndarray,
    gt_edges,
    margin_eps: float,
) -> Dict[str, float]:
    adj_raw = support_weights * direction_gate
    adj_causal = to_causal_matrix_np(adj_raw)
    strict_metrics = selector_audit_evaluate_directional_strict(
        adj_causal,
        gt_edges,
        margin_eps=margin_eps,
    )
    margin_stats = selector_audit_margin_stats(adj_causal)
    return {
        "support_epoch": float(support_epoch),
        "direction_epoch": float(direction_epoch),
        "strict_f1": float(strict_metrics["strict_f1"]),
        "strict_precision": float(strict_metrics["strict_precision"]),
        "strict_recall": float(strict_metrics["strict_recall"]),
        "strict_pred_count": float(strict_metrics["strict_pred_count"]),
        "signed_margin_median": float(margin_stats["selector_audit_margin_median"]),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    snapshot_path = run_dir / "support_direction_snapshots.npz"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing snapshot file: {snapshot_path}")

    cfg = load_config(run_dir)
    gt_path = args.gt_path or cfg.get("selector_audit_gt_path")
    if not gt_path:
        raise ValueError(
            "GT path is required; pass --gt_path or ensure selector_audit_gt_path "
            "exists in config.npy."
        )
    gt_edges = load_gt_edges(str(resolve_repo_relative_path(gt_path)))

    selector_summary = load_selector_summary(run_dir)
    best_epoch = int(metric_from_selector_row(selector_summary, "selector_audit_best_gt_epoch"))
    exported_epoch = int(metric_from_selector_row(selector_summary, "selector_audit_exported_epoch"))
    final_epoch = int(metric_from_selector_row(selector_summary, "selector_audit_final_epoch"))

    with np.load(snapshot_path) as data:
        epochs = data["epochs"].astype(np.int32, copy=False)
        support_weights = data["support_weights"].astype(np.float32, copy=False)
        direction_gate = data["direction_gate"].astype(np.float32, copy=False)

    epoch_to_idx = build_epoch_index(epochs)
    if best_epoch not in epoch_to_idx or final_epoch not in epoch_to_idx or exported_epoch not in epoch_to_idx:
        raise RuntimeError(
            "Snapshot epochs do not cover the selector summary anchor epochs. "
            f"anchors=best:{best_epoch}, exported:{exported_epoch}, final:{final_epoch}"
        )

    rows: List[Dict[str, float]] = []
    epoch_list = [int(epoch) for epoch in epochs.tolist()]
    for support_idx, support_epoch in enumerate(epoch_list):
        for direction_idx, direction_epoch in enumerate(epoch_list):
            rows.append(
                eval_combo(
                    support_epoch=support_epoch,
                    direction_epoch=direction_epoch,
                    support_weights=support_weights[support_idx],
                    direction_gate=direction_gate[direction_idx],
                    gt_edges=gt_edges,
                    margin_eps=args.margin_eps,
                )
            )

    combo_df = pd.DataFrame(rows)
    combo_df["support_epoch"] = combo_df["support_epoch"].astype(int)
    combo_df["direction_epoch"] = combo_df["direction_epoch"].astype(int)
    combo_df["is_diagonal"] = (
        combo_df["support_epoch"] == combo_df["direction_epoch"]
    ).astype(int)

    def get_combo_row(support_epoch_value: int, direction_epoch_value: int) -> pd.Series:
        row = combo_df[
            (combo_df["support_epoch"] == int(support_epoch_value)) &
            (combo_df["direction_epoch"] == int(direction_epoch_value))
        ]
        if row.empty:
            raise RuntimeError(
                "Missing epoch-swap row for "
                f"support={support_epoch_value}, direction={direction_epoch_value}"
            )
        return row.iloc[0]

    sort_cols = ["strict_f1", "signed_margin_median", "support_epoch", "direction_epoch"]
    sort_asc = [False, False, True, True]
    best_any = combo_df.sort_values(sort_cols, ascending=sort_asc).iloc[0]
    best_diagonal = combo_df[combo_df["is_diagonal"] == 1].sort_values(
        ["strict_f1", "signed_margin_median", "support_epoch"],
        ascending=[False, False, True],
    ).iloc[0]
    best_support_with_final_direction = combo_df[
        combo_df["direction_epoch"] == final_epoch
    ].sort_values(
        ["strict_f1", "signed_margin_median", "support_epoch"],
        ascending=[False, False, True],
    ).iloc[0]
    best_direction_with_final_support = combo_df[
        combo_df["support_epoch"] == final_epoch
    ].sort_values(
        ["strict_f1", "signed_margin_median", "direction_epoch"],
        ascending=[False, False, True],
    ).iloc[0]

    anchor_rows = {
        "best_best": get_combo_row(best_epoch, best_epoch),
        "best_final": get_combo_row(best_epoch, final_epoch),
        "final_best": get_combo_row(final_epoch, best_epoch),
        "final_final": get_combo_row(final_epoch, final_epoch),
        "exported_exported": get_combo_row(exported_epoch, exported_epoch),
    }

    summary_row = {
        "run_dir": str(run_dir),
        "dataset": Path(str(cfg.get("csv_path", ""))).stem,
        "seed": int(cfg.get("seed", -1)),
        "margin_eps": float(args.margin_eps),
        "best_epoch": best_epoch,
        "exported_epoch": exported_epoch,
        "final_epoch": final_epoch,
        "baseline_best_gt_strict_f1": metric_from_selector_row(
            selector_summary,
            "selector_audit_best_gt_primary_strict_f1",
        ),
        "baseline_exported_strict_f1": metric_from_selector_row(
            selector_summary,
            "selector_audit_exported_primary_strict_f1",
        ),
        "baseline_final_strict_f1": metric_from_selector_row(
            selector_summary,
            "selector_audit_final_primary_strict_f1",
        ),
        "diag_best_support_epoch": int(best_diagonal["support_epoch"]),
        "diag_best_direction_epoch": int(best_diagonal["direction_epoch"]),
        "diag_best_strict_f1": float(best_diagonal["strict_f1"]),
        "any_best_support_epoch": int(best_any["support_epoch"]),
        "any_best_direction_epoch": int(best_any["direction_epoch"]),
        "any_best_strict_f1": float(best_any["strict_f1"]),
        "best_support_with_final_direction_epoch": int(best_support_with_final_direction["support_epoch"]),
        "best_support_with_final_direction_strict_f1": float(
            best_support_with_final_direction["strict_f1"]
        ),
        "best_direction_with_final_support_epoch": int(best_direction_with_final_support["direction_epoch"]),
        "best_direction_with_final_support_strict_f1": float(
            best_direction_with_final_support["strict_f1"]
        ),
        "anchor_best_best_strict_f1": float(anchor_rows["best_best"]["strict_f1"]),
        "anchor_best_final_strict_f1": float(anchor_rows["best_final"]["strict_f1"]),
        "anchor_final_best_strict_f1": float(anchor_rows["final_best"]["strict_f1"]),
        "anchor_final_final_strict_f1": float(anchor_rows["final_final"]["strict_f1"]),
        "anchor_exported_exported_strict_f1": float(anchor_rows["exported_exported"]["strict_f1"]),
        "support_swap_gain_over_final": float(
            best_support_with_final_direction["strict_f1"] - anchor_rows["final_final"]["strict_f1"]
        ),
        "direction_swap_gain_over_final": float(
            best_direction_with_final_support["strict_f1"] - anchor_rows["final_final"]["strict_f1"]
        ),
        "any_swap_gain_over_final": float(best_any["strict_f1"] - anchor_rows["final_final"]["strict_f1"]),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = summary_row["dataset"]
    output_stem = f"unify_epoch_swap_{dataset}_{timestamp}_{args.tag}"
    combo_path = RESULTS_DIR / f"{output_stem}.csv"
    top_path = RESULTS_DIR / f"{output_stem}_top.csv"
    summary_path = RESULTS_DIR / f"{output_stem}_summary.csv"

    combo_df.to_csv(combo_path, index=False, float_format="%.6f")
    combo_df.sort_values(sort_cols, ascending=sort_asc).head(max(args.top_n, 1)).to_csv(
        top_path,
        index=False,
        float_format="%.6f",
    )
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False, float_format="%.6f")

    print(f"Epoch-swap grid written to: {combo_path}")
    print(f"Epoch-swap top rows written to: {top_path}")
    print(f"Epoch-swap summary written to: {summary_path}")
    print(
        "Key anchors | "
        f"best_best={summary_row['anchor_best_best_strict_f1']:.4f} | "
        f"best_final={summary_row['anchor_best_final_strict_f1']:.4f} | "
        f"final_best={summary_row['anchor_final_best_strict_f1']:.4f} | "
        f"final_final={summary_row['anchor_final_final_strict_f1']:.4f}"
    )
    print(
        "Best recoveries | "
        f"support@*+dir@final={summary_row['best_support_with_final_direction_strict_f1']:.4f} "
        f"(epoch {summary_row['best_support_with_final_direction_epoch']}) | "
        f"support@final+dir@*={summary_row['best_direction_with_final_support_strict_f1']:.4f} "
        f"(epoch {summary_row['best_direction_with_final_support_epoch']}) | "
        f"any_swap={summary_row['any_best_strict_f1']:.4f} "
        f"(support {summary_row['any_best_support_epoch']}, direction {summary_row['any_best_direction_epoch']})"
    )


if __name__ == "__main__":
    main()
