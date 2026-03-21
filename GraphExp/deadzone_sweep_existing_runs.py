from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


def load_gt(path: Path) -> Set[Tuple[int, int]]:
    gt: Set[Tuple[int, int]] = set()
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


def load_adj(run_dir: Path) -> np.ndarray:
    npy_path = run_dir / "final_epoch_adjacency_causal.npy"
    if npy_path.exists():
        return np.load(npy_path)
    csv_path = run_dir / "final_epoch_adjacency_causal.csv"
    if csv_path.exists():
        return np.loadtxt(csv_path, delimiter=",")
    raise FileNotFoundError(f"Could not find final causal adjacency under {run_dir}")


def load_config(run_dir: Path) -> Dict[str, object]:
    cfg_path = run_dir / "config.npy"
    if not cfg_path.exists():
        return {}
    cfg = np.load(cfg_path, allow_pickle=True).item()
    return cfg if isinstance(cfg, dict) else {}


def pred_edges_by_eps(adj: np.ndarray, margin_eps: float) -> Set[Tuple[int, int]]:
    pred_edges: Set[Tuple[int, int]] = set()
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            delta = float(adj[i, j] - adj[j, i])
            if delta > margin_eps:
                pred_edges.add((i, j))
            elif delta < -margin_eps:
                pred_edges.add((j, i))
    return pred_edges


def gt_signed_margins(adj: np.ndarray, gt_edges: Iterable[Tuple[int, int]]) -> np.ndarray:
    values = [float(adj[src, dst] - adj[dst, src]) for src, dst in gt_edges]
    if not values:
        return np.zeros(0, dtype=float)
    return np.asarray(values, dtype=float)


def evaluate(pred_edges: Set[Tuple[int, int]], gt_edges: Set[Tuple[int, int]]) -> Dict[str, object]:
    tp_edges = pred_edges & gt_edges
    fp_edges = pred_edges - gt_edges
    fn_edges = gt_edges - pred_edges
    tp = len(tp_edges)
    fp = len(fp_edges)
    fn = len(fn_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "pred_edges": pred_edges,
        "tp_edges": tp_edges,
        "fp_edges": fp_edges,
        "fn_edges": fn_edges,
        "strict_pred_count": len(pred_edges),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "strict_precision": precision,
        "strict_recall": recall,
        "strict_f1": f1,
    }


def summarize_run(
    run_dir: Path,
    gt_edges: Set[Tuple[int, int]],
    eps_values: Sequence[float],
    baseline_eps: float,
) -> List[Dict[str, object]]:
    adj = load_adj(run_dir)
    cfg = load_config(run_dir)
    gt_margins = gt_signed_margins(adj, gt_edges)
    baseline = evaluate(pred_edges_by_eps(adj, baseline_eps), gt_edges)
    baseline_tp_edges = set(baseline["tp_edges"])
    baseline_fp_edges = set(baseline["fp_edges"])
    baseline_tp = max(int(baseline["tp"]), 1)
    baseline_fp = max(int(baseline["fp"]), 1)

    rows: List[Dict[str, object]] = []
    for eps in eps_values:
        result = evaluate(pred_edges_by_eps(adj, eps), gt_edges)
        tp_edges = set(result["tp_edges"])
        fp_edges = set(result["fp_edges"])
        tp_lost = len(baseline_tp_edges - tp_edges)
        fp_removed = len(baseline_fp_edges - fp_edges)
        gt_fragile_count = int(np.sum(gt_margins < eps))
        row: Dict[str, object] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            "eps": eps,
            "strict_precision": float(result["strict_precision"]),
            "strict_recall": float(result["strict_recall"]),
            "strict_f1": float(result["strict_f1"]),
            "strict_pred_count": int(result["strict_pred_count"]),
            "tp": int(result["tp"]),
            "fp": int(result["fp"]),
            "fn": int(result["fn"]),
            "baseline_eps": baseline_eps,
            "baseline_tp": int(baseline["tp"]),
            "baseline_fp": int(baseline["fp"]),
            "tp_lost_by_deadzone": tp_lost,
            "tp_lost_by_deadzone_frac": tp_lost / baseline_tp,
            "fp_removed_by_deadzone": fp_removed,
            "fp_removed_by_deadzone_frac": fp_removed / baseline_fp,
            "all_gt_fragile_count": gt_fragile_count,
            "all_gt_fragile_frac": gt_fragile_count / len(gt_edges) if gt_edges else 0.0,
            "gt_margin_min": float(gt_margins.min()) if gt_margins.size else 0.0,
            "gt_margin_p10": float(np.quantile(gt_margins, 0.10)) if gt_margins.size else 0.0,
            "gt_margin_p25": float(np.quantile(gt_margins, 0.25)) if gt_margins.size else 0.0,
            "gt_margin_median": float(np.median(gt_margins)) if gt_margins.size else 0.0,
            "gt_edge_count": len(gt_edges),
            "directional_kappa_gate": int(bool(cfg.get("directional_kappa_gate", False))),
            "directional_kappa_gate_quantile": float(cfg.get("directional_kappa_gate_quantile", 0.0) or 0.0),
            "parent_cap_lambda": float(cfg.get("parent_cap_lambda", 0.0) or 0.0),
            "parent_cap_target": float(cfg.get("parent_cap_target", 0.0) or 0.0),
            "ungated_symmetry_lambda": float(cfg.get("ungated_symmetry_lambda", 0.0) or 0.0),
            "seed": int(cfg.get("seed", -1) or -1),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep strict directional deadzone over existing result dirs.")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Result directories containing final_epoch_adjacency_causal.npy/csv",
    )
    parser.add_argument("--gt_path", required=True, help="Ground-truth edge list path")
    parser.add_argument(
        "--eps_values",
        nargs="+",
        type=float,
        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1],
        help="Deadzone eps values to sweep",
    )
    parser.add_argument(
        "--baseline_eps",
        type=float,
        default=1e-12,
        help="Reference epsilon used to define tp_lost/fp_removed",
    )
    parser.add_argument("--output_csv", required=True, help="Where to write the sweep CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_edges = load_gt(Path(args.gt_path))
    all_rows: List[Dict[str, object]] = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        rows = summarize_run(
            run_dir=run_dir,
            gt_edges=gt_edges,
            eps_values=args.eps_values,
            baseline_eps=args.baseline_eps,
        )
        all_rows.extend(rows)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, all_rows)

    for row in all_rows:
        print(
            f"{row['run_name']} eps={row['eps']:.4g}: "
            f"strict_f1={row['strict_f1']:.4f}, "
            f"precision={row['strict_precision']:.4f}, "
            f"recall={row['strict_recall']:.4f}, "
            f"pred={row['strict_pred_count']}, "
            f"tp_lost={row['tp_lost_by_deadzone']}, "
            f"fp_removed={row['fp_removed_by_deadzone']}, "
            f"gt_fragile={row['all_gt_fragile_count']}"
        )
    print(f"DEADZONE_SWEEP_CSV {output_csv.resolve()}")


if __name__ == "__main__":
    main()
