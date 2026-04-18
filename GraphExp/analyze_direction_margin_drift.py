import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from main_structure_learning import load_gt_edges


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pair-level direction margin drift from "
            "support_direction_snapshots.npz across one or more runs."
        )
    )
    parser.add_argument(
        "--run_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run directories containing support_direction_snapshots.npz.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Optional GT path override. Defaults to selector_audit_gt_path in config.npy.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="direction_margin_drift",
        help="Short suffix used in output filenames.",
    )
    parser.add_argument(
        "--anchor_epoch",
        type=int,
        default=23,
        help="Routing-switch epoch used for the fixed anchor summary row.",
    )
    parser.add_argument(
        "--extra_epochs",
        type=str,
        default="30",
        help="Comma-separated extra fixed epochs to summarize if available.",
    )
    parser.add_argument(
        "--low_margin_thresholds",
        type=str,
        default="0.002,0.005,0.01",
        help="Comma-separated absolute-margin thresholds used for collapse diagnostics.",
    )
    return parser.parse_args()


def parse_float_csv(text: str) -> List[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def parse_int_csv(text: str) -> List[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def resolve_repo_relative_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def load_config(run_dir: Path) -> Dict[str, Any]:
    cfg = np.load(run_dir / "config.npy", allow_pickle=True).item()
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict-like config in {run_dir / 'config.npy'}")
    return dict(cfg)


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


def format_float_token(value: float) -> str:
    token = np.format_float_positional(float(value), trim="-")
    return token.replace(".", "p").replace("-", "m")


def epoch_index_map(epochs: np.ndarray) -> Dict[int, int]:
    return {int(epoch): idx for idx, epoch in enumerate(epochs.tolist())}


def sorted_unique_epochs(values: Iterable[int]) -> List[int]:
    return sorted({int(v) for v in values if int(v) > 0})


def build_gt_pair_labels(gt_edges: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    pair_to_direction: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for src, dst in sorted(gt_edges):
        a, b = sorted((src, dst))
        if (a, b) in pair_to_direction:
            raise ValueError(
                "GT contains both directions for the same unordered pair, "
                f"which this analysis does not support: {(a + 1, b + 1)}"
            )
        pair_to_direction[(a, b)] = (src, dst)
    return pair_to_direction


def compute_gt_signed_margins(
    adj_causal: np.ndarray,
    gt_pair_labels: Dict[Tuple[int, int], Tuple[int, int]],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    margins: List[float] = []
    pair_order: List[Tuple[int, int]] = []
    for unordered_pair, directed_edge in sorted(gt_pair_labels.items()):
        src, dst = directed_edge
        margins.append(float(adj_causal[src, dst] - adj_causal[dst, src]))
        pair_order.append(unordered_pair)
    return np.asarray(margins, dtype=np.float64), pair_order


def compute_all_pair_margins(adj_causal: np.ndarray) -> np.ndarray:
    n = adj_causal.shape[0]
    margins: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            margins.append(float(adj_causal[i, j] - adj_causal[j, i]))
    return np.asarray(margins, dtype=np.float64)


def summarize_epoch(
    *,
    margins_gt: np.ndarray,
    margins_all_pairs: np.ndarray,
    thresholds: Sequence[float],
) -> Dict[str, float]:
    if margins_gt.size == 0:
        raise ValueError("Expected at least one GT pair margin.")

    summary: Dict[str, float] = {
        "gt_margin_mean": float(margins_gt.mean()),
        "gt_margin_median": float(np.median(margins_gt)),
        "gt_margin_p10": float(np.quantile(margins_gt, 0.10)),
        "gt_margin_min": float(margins_gt.min()),
        "gt_positive_frac": float(np.mean(margins_gt > 0.0)),
        "gt_nonpositive_frac": float(np.mean(margins_gt <= 0.0)),
        "pair_abs_margin_mean": float(np.mean(np.abs(margins_all_pairs))),
        "pair_abs_margin_median": float(np.median(np.abs(margins_all_pairs))),
    }
    for threshold in thresholds:
        label = format_float_token(threshold)
        summary[f"gt_abs_lt_{label}_frac"] = float(np.mean(np.abs(margins_gt) < threshold))
        summary[f"pair_abs_lt_{label}_frac"] = float(
            np.mean(np.abs(margins_all_pairs) < threshold)
        )
    return summary


def categorize_best_to_final(
    *,
    best_margins_gt: np.ndarray,
    final_margins_gt: np.ndarray,
    low_margin_threshold: float,
) -> Dict[str, float]:
    best_correct = best_margins_gt > 0.0
    final_correct = final_margins_gt > 0.0
    final_low = np.abs(final_margins_gt) < low_margin_threshold
    same_sign = np.sign(best_margins_gt) == np.sign(final_margins_gt)

    categories = {
        "best_correct_final_correct": float(np.mean(best_correct & final_correct)),
        "best_correct_final_wrong_or_zero": float(np.mean(best_correct & ~final_correct)),
        "best_correct_final_low_margin": float(
            np.mean(best_correct & final_correct & final_low)
        ),
        "best_correct_final_strong_margin": float(
            np.mean(best_correct & final_correct & ~final_low)
        ),
        "best_wrong_final_correct": float(np.mean(~best_correct & final_correct)),
        "best_to_final_same_sign": float(np.mean(same_sign)),
        "best_to_final_sign_flip": float(np.mean(~same_sign)),
        "gt_margin_delta_mean": float(np.mean(final_margins_gt - best_margins_gt)),
        "gt_margin_delta_median": float(np.median(final_margins_gt - best_margins_gt)),
    }
    return categories


def build_pair_detail_rows(
    *,
    run_dir: Path,
    dataset: str,
    seed: int,
    gt_pair_labels: Dict[Tuple[int, int], Tuple[int, int]],
    epoch_to_adj: Dict[int, np.ndarray],
    selected_epochs: Sequence[int],
    best_epoch: int,
    final_epoch: int,
) -> List[Dict[str, Any]]:
    low_margin_threshold = 0.005
    rows: List[Dict[str, Any]] = []
    for unordered_pair, directed_edge in sorted(gt_pair_labels.items()):
        src, dst = directed_edge
        row: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "dataset": dataset,
            "seed": int(seed),
            "src_1based": int(src + 1),
            "dst_1based": int(dst + 1),
            "pair_i_1based": int(unordered_pair[0] + 1),
            "pair_j_1based": int(unordered_pair[1] + 1),
        }
        margin_by_epoch: Dict[int, float] = {}
        for epoch in selected_epochs:
            adj = epoch_to_adj[epoch]
            margin = float(adj[src, dst] - adj[dst, src])
            margin_by_epoch[epoch] = margin
            row[f"margin_epoch_{epoch}"] = margin
        best_margin = margin_by_epoch[best_epoch]
        final_margin = margin_by_epoch[final_epoch]
        row["best_margin"] = best_margin
        row["final_margin"] = final_margin
        row["margin_delta_final_minus_best"] = final_margin - best_margin
        row["best_correct"] = int(best_margin > 0.0)
        row["final_correct"] = int(final_margin > 0.0)
        row["final_low_margin"] = int(abs(final_margin) < low_margin_threshold)
        if best_margin > 0.0 and final_margin > 0.0:
            if abs(final_margin) < low_margin_threshold:
                transition = "correct_to_near_zero"
            else:
                transition = "correct_to_correct"
        elif best_margin > 0.0 and final_margin <= 0.0:
            transition = "correct_to_wrong"
        elif best_margin <= 0.0 and final_margin > 0.0:
            transition = "wrong_to_correct"
        else:
            transition = "wrong_to_wrong"
        row["best_final_transition"] = transition
        rows.append(row)
    return rows


def analyze_run(
    *,
    run_dir: Path,
    gt_path_override: Optional[str],
    anchor_epoch: int,
    extra_epochs: Sequence[int],
    low_margin_thresholds: Sequence[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    run_dir = run_dir.resolve()
    snapshot_path = run_dir / "support_direction_snapshots.npz"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing snapshot file: {snapshot_path}")

    cfg = load_config(run_dir)
    selector = load_selector_summary(run_dir)
    dataset = Path(str(cfg.get("csv_path", ""))).stem
    seed = int(cfg.get("seed", -1))
    gt_path = gt_path_override or cfg.get("selector_audit_gt_path")
    if not gt_path:
        raise ValueError(
            f"GT path missing for {run_dir}; pass --gt_path or ensure selector_audit_gt_path exists."
        )
    gt_edges = load_gt_edges(str(resolve_repo_relative_path(gt_path)))
    gt_pair_labels = build_gt_pair_labels(gt_edges)

    with np.load(snapshot_path) as data:
        epochs = data["epochs"].astype(np.int32, copy=False)
        adj_causal = data["adj_causal"].astype(np.float32, copy=False)
        direction_gate = data["direction_gate"].astype(np.float32, copy=False)

    best_epoch = int(metric_from_selector_row(selector, "selector_audit_best_gt_epoch"))
    exported_epoch = int(metric_from_selector_row(selector, "selector_audit_exported_epoch"))
    final_epoch = int(metric_from_selector_row(selector, "selector_audit_final_epoch"))
    epoch_map = epoch_index_map(epochs)
    requested_epochs = sorted_unique_epochs(
        [anchor_epoch, best_epoch, exported_epoch, final_epoch, *extra_epochs]
    )
    selected_epochs = [epoch for epoch in requested_epochs if epoch in epoch_map]
    epoch_to_adj = {epoch: adj_causal[epoch_map[epoch]] for epoch in selected_epochs}
    epoch_to_gate = {epoch: direction_gate[epoch_map[epoch]] for epoch in selected_epochs}

    epoch_rows: List[Dict[str, Any]] = []
    for epoch in selected_epochs:
        gt_margins, _ = compute_gt_signed_margins(epoch_to_adj[epoch], gt_pair_labels)
        all_pair_margins = compute_all_pair_margins(epoch_to_adj[epoch])
        row: Dict[str, Any] = {
            "run_dir": str(run_dir),
            "dataset": dataset,
            "seed": int(seed),
            "epoch": int(epoch),
            "is_best_epoch": int(epoch == best_epoch),
            "is_exported_epoch": int(epoch == exported_epoch),
            "is_final_epoch": int(epoch == final_epoch),
            "is_anchor_epoch": int(epoch == anchor_epoch),
            "direction_gate_mean": float(epoch_to_gate[epoch].mean()),
            "direction_gate_std": float(epoch_to_gate[epoch].std()),
        }
        row.update(
            summarize_epoch(
                margins_gt=gt_margins,
                margins_all_pairs=all_pair_margins,
                thresholds=low_margin_thresholds,
            )
        )
        epoch_rows.append(row)

    best_gt_margins, _ = compute_gt_signed_margins(epoch_to_adj[best_epoch], gt_pair_labels)
    final_gt_margins, pair_order = compute_gt_signed_margins(epoch_to_adj[final_epoch], gt_pair_labels)
    pair_rows = build_pair_detail_rows(
        run_dir=run_dir,
        dataset=dataset,
        seed=seed,
        gt_pair_labels=gt_pair_labels,
        epoch_to_adj=epoch_to_adj,
        selected_epochs=selected_epochs,
        best_epoch=best_epoch,
        final_epoch=final_epoch,
    )
    transition_summary = categorize_best_to_final(
        best_margins_gt=best_gt_margins,
        final_margins_gt=final_gt_margins,
        low_margin_threshold=0.005,
    )
    summary_row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "exported_epoch": int(exported_epoch),
        "final_epoch": int(final_epoch),
        "baseline_best_gt_strict_f1": float(
            metric_from_selector_row(selector, "selector_audit_best_gt_primary_strict_f1")
        ),
        "baseline_exported_strict_f1": float(
            metric_from_selector_row(selector, "selector_audit_exported_primary_strict_f1")
        ),
        "baseline_final_strict_f1": float(
            metric_from_selector_row(selector, "selector_audit_final_primary_strict_f1")
        ),
        "baseline_best_signed_margin_median": float(
            metric_from_selector_row(selector, "selector_audit_best_gt_signed_margin_median")
        ),
        "baseline_final_signed_margin_median": float(
            metric_from_selector_row(selector, "selector_audit_final_signed_margin_median")
        ),
        "gt_pair_count": int(len(gt_pair_labels)),
    }
    summary_row.update(transition_summary)
    summary_row["best_to_final_mean_abs_margin_drop"] = float(
        np.mean(np.abs(best_gt_margins) - np.abs(final_gt_margins))
    )
    summary_row["best_to_final_pair_abs_margin_lt_0p005_increase"] = float(
        np.mean(np.abs(final_gt_margins) < 0.005) - np.mean(np.abs(best_gt_margins) < 0.005)
    )
    summary_row["best_to_final_pair_abs_margin_lt_0p01_increase"] = float(
        np.mean(np.abs(final_gt_margins) < 0.01) - np.mean(np.abs(best_gt_margins) < 0.01)
    )
    return epoch_rows, pair_rows, summary_row


def aggregate_summary_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "run_count": len(rows),
        "seed_list": ",".join(str(int(row["seed"])) for row in rows),
        "result_dirs": ";".join(str(row["run_dir"]) for row in rows),
    }
    for key in (
        "baseline_best_gt_strict_f1",
        "baseline_exported_strict_f1",
        "baseline_final_strict_f1",
        "baseline_best_signed_margin_median",
        "baseline_final_signed_margin_median",
        "best_correct_final_correct",
        "best_correct_final_wrong_or_zero",
        "best_correct_final_low_margin",
        "best_correct_final_strong_margin",
        "best_wrong_final_correct",
        "best_to_final_same_sign",
        "best_to_final_sign_flip",
        "gt_margin_delta_mean",
        "gt_margin_delta_median",
        "best_to_final_mean_abs_margin_drop",
        "best_to_final_pair_abs_margin_lt_0p005_increase",
        "best_to_final_pair_abs_margin_lt_0p01_increase",
    ):
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = mean(values)
        result[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return result


def aggregate_epoch_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"run_dir", "dataset", "seed"}
    ]
    grouped_rows: List[Dict[str, Any]] = []
    for epoch, group in df.groupby("epoch", sort=True):
        agg_row: Dict[str, Any] = {
            "epoch": int(epoch),
            "run_count": int(group.shape[0]),
            "seed_list": ",".join(str(int(v)) for v in group["seed"].tolist()),
        }
        for col in numeric_cols:
            if col == "epoch":
                continue
            values = [float(v) for v in group[col].tolist()]
            agg_row[f"{col}_mean"] = mean(values)
            agg_row[f"{col}_std"] = pstdev(values) if len(values) > 1 else 0.0
        grouped_rows.append(agg_row)
    return pd.DataFrame(grouped_rows)


def aggregate_pair_transitions(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped_rows: List[Dict[str, Any]] = []
    for transition, group in df.groupby("best_final_transition", sort=True):
        grouped_rows.append(
            {
                "best_final_transition": transition,
                "pair_count": int(group.shape[0]),
                "seed_count": int(group["seed"].nunique()),
                "margin_delta_final_minus_best_mean": float(
                    group["margin_delta_final_minus_best"].mean()
                ),
                "margin_delta_final_minus_best_median": float(
                    group["margin_delta_final_minus_best"].median()
                ),
            }
        )
    return pd.DataFrame(grouped_rows)


def main() -> None:
    args = parse_args()
    extra_epochs = parse_int_csv(args.extra_epochs)
    low_margin_thresholds = parse_float_csv(args.low_margin_thresholds)

    all_epoch_rows: List[Dict[str, Any]] = []
    all_pair_rows: List[Dict[str, Any]] = []
    all_summary_rows: List[Dict[str, Any]] = []
    for run_dir in args.run_dirs:
        epoch_rows, pair_rows, summary_row = analyze_run(
            run_dir=run_dir,
            gt_path_override=args.gt_path,
            anchor_epoch=args.anchor_epoch,
            extra_epochs=extra_epochs,
            low_margin_thresholds=low_margin_thresholds,
        )
        all_epoch_rows.extend(epoch_rows)
        all_pair_rows.extend(pair_rows)
        all_summary_rows.append(summary_row)

    if not all_summary_rows:
        raise RuntimeError("No runs were analyzed.")

    dataset = str(all_summary_rows[0]["dataset"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = f"unify_margin_drift_{dataset}_{timestamp}_{args.tag}"
    summary_path = RESULTS_DIR / f"{output_stem}.csv"
    aggregate_path = RESULTS_DIR / f"{output_stem}_aggregate.csv"
    epoch_path = RESULTS_DIR / f"{output_stem}_epochs.csv"
    epoch_aggregate_path = RESULTS_DIR / f"{output_stem}_epochs_aggregate.csv"
    pair_path = RESULTS_DIR / f"{output_stem}_pairs.csv"
    transition_path = RESULTS_DIR / f"{output_stem}_pair_transitions.csv"

    pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False, float_format="%.6f")
    pd.DataFrame([aggregate_summary_rows(all_summary_rows)]).to_csv(
        aggregate_path,
        index=False,
        float_format="%.6f",
    )
    pd.DataFrame(all_epoch_rows).to_csv(epoch_path, index=False, float_format="%.6f")
    aggregate_epoch_rows(all_epoch_rows).to_csv(
        epoch_aggregate_path,
        index=False,
        float_format="%.6f",
    )
    pd.DataFrame(all_pair_rows).to_csv(pair_path, index=False, float_format="%.6f")
    aggregate_pair_transitions(all_pair_rows).to_csv(
        transition_path,
        index=False,
        float_format="%.6f",
    )

    print(f"Run summary written to: {summary_path}")
    print(f"Aggregate written to: {aggregate_path}")
    print(f"Epoch rows written to: {epoch_path}")
    print(f"Epoch aggregate written to: {epoch_aggregate_path}")
    print(f"Pair rows written to: {pair_path}")
    print(f"Pair transition aggregate written to: {transition_path}")


if __name__ == "__main__":
    main()
