from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
REPO_ROOT = SCRIPT_DIR.parent
DATASET_ORDER = ("fMRI", "sim2", "sim3", "sim4", "sim8", "sim10", "sim11", "sim12")
GT_PATHS = {
    "fMRI": REPO_ROOT / "fMRI_dataset" / "h1.txt",
    "sim2": REPO_ROOT / "fMRI_dataset" / "h2.txt",
    "sim3": REPO_ROOT / "fMRI_dataset" / "h3.txt",
    "sim4": REPO_ROOT / "fMRI_dataset" / "h4.txt",
    "sim8": REPO_ROOT / "fMRI_dataset" / "h8.txt",
    "sim10": REPO_ROOT / "fMRI_dataset" / "h10.txt",
    "sim11": REPO_ROOT / "fMRI_dataset" / "h11.txt",
    "sim12": REPO_ROOT / "fMRI_dataset" / "h12.txt",
}

Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize formal best-base topk/lag tuning manifests."
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=RESULTS_DIR / "param_tuning_bestbase_20260516",
        help="Output prefix. Writes *_summary.csv, *_topk.csv, *_lag.csv, *_best.csv and .md.",
    )
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="Skip rows with missing aggregate/summary/adjacency instead of failing.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_one_csv_row(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")
    return rows[0]


def load_gt(path: Path) -> Set[Edge]:
    edges: Set[Edge] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parts = text.replace(",", " ").split()
            if len(parts) < 2:
                continue
            src = int(parts[0]) - 1
            dst = int(parts[1]) - 1
            if src != dst:
                edges.add((src, dst))
    return edges


def load_final_adj(run_dir: Path) -> np.ndarray:
    for filename in ("final_epoch_adjacency_causal.npy", "final_epoch_adjacency_causal.csv"):
        path = run_dir / filename
        if path.exists():
            return np.load(path) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
    for filename in ("learned_adjacency_causal.npy", "learned_adjacency_causal.csv"):
        path = run_dir / filename
        if path.exists():
            return np.load(path) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
    raise FileNotFoundError(f"Missing final/exported causal adjacency under {run_dir}")


def topk_directed_edges(adj: np.ndarray, k: int) -> Set[Edge]:
    scored: List[Tuple[float, int, int, int, int]] = []
    n = int(adj.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            delta = float(adj[i, j] - adj[j, i])
            if delta >= 0.0:
                src, dst = i, j
            else:
                src, dst = j, i
            scored.append((abs(delta), i, j, src, dst))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return {(src, dst) for _, _, _, src, dst in scored[:k]}


def directed_shd(pred_edges: Set[Edge], gt_edges: Set[Edge]) -> int:
    pred_remaining = set(pred_edges)
    gt_remaining = set(gt_edges)
    reversals = 0
    for edge in list(pred_remaining):
        rev = (edge[1], edge[0])
        if rev in gt_remaining:
            reversals += 1
            pred_remaining.remove(edge)
            gt_remaining.remove(rev)
    return reversals + len(pred_remaining - gt_remaining) + len(gt_remaining - pred_remaining)


def edge_accuracy(pred_edges: Set[Edge], gt_edges: Set[Edge], n: int) -> float:
    directed_edges = {(i, j) for i in range(n) for j in range(n) if i != j}
    if not directed_edges:
        return 0.0
    correct = sum(((edge in pred_edges) == (edge in gt_edges)) for edge in directed_edges)
    return correct / len(directed_edges)


def evaluate_final_adj(adj: np.ndarray, gt_edges: Set[Edge]) -> Dict[str, float]:
    pred_edges = topk_directed_edges(adj, len(gt_edges))
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "final_directional_precision": precision,
        "final_directional_recall": recall,
        "final_directional_f1": f1,
        "final_shd": float(directed_shd(pred_edges, gt_edges)),
        "final_edge_accuracy": edge_accuracy(pred_edges, gt_edges, int(adj.shape[0])),
    }


def aggregate_directional(summary_path: Path, dataset: str) -> Dict[str, object]:
    gt_edges = load_gt(GT_PATHS[dataset])
    per_seed = []
    for row in read_csv_rows(summary_path):
        run_dir = Path(row["result_dir"])
        metrics = evaluate_final_adj(load_final_adj(run_dir), gt_edges)
        metrics["seed"] = int(row["seed"])
        per_seed.append(metrics)

    out: Dict[str, object] = {
        "directional_run_count": len(per_seed),
        "directional_seed_list": ",".join(str(row["seed"]) for row in per_seed),
    }
    for metric in (
        "final_directional_precision",
        "final_directional_recall",
        "final_directional_f1",
        "final_shd",
        "final_edge_accuracy",
    ):
        values = [float(row[metric]) for row in per_seed]
        out[f"{metric}_mean"] = mean(values)
        out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return out


def row_value(row: Dict[str, str], key: str, default: str = "") -> str:
    value = row.get(key)
    return default if value is None else value


def collect_rows(manifests: Sequence[Path], *, allow_incomplete: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for manifest in manifests:
        for manifest_row in read_csv_rows(manifest):
            if manifest_row.get("status") != "ok":
                continue
            aggregate_path = Path(manifest_row["aggregate_path"])
            summary_path = Path(manifest_row["summary_path"])
            try:
                aggregate = read_one_csv_row(aggregate_path)
                directional = aggregate_directional(summary_path, manifest_row["dataset"])
            except Exception:
                if allow_incomplete:
                    continue
                raise

            rows.append(
                {
                    "experiment": manifest_row["experiment"],
                    "dataset": manifest_row["dataset"],
                    "candidate_key": manifest_row["candidate_key"],
                    "base_run_dir": manifest_row["base_run_dir"],
                    "seeds": manifest_row["seeds"],
                    "top_k_edges": manifest_row["top_k_edges"],
                    "selection_top_k": manifest_row["selection_top_k"],
                    "lag_weight": manifest_row["lag_weight"],
                    "override_spec": manifest_row["override_spec"],
                    "run_count": aggregate["run_count"],
                    "best_primary_strict_f1_mean": aggregate["best_primary_strict_f1_mean"],
                    "best_primary_strict_f1_std": aggregate["best_primary_strict_f1_std"],
                    "exported_primary_strict_f1_mean": aggregate["exported_primary_strict_f1_mean"],
                    "exported_primary_strict_f1_std": aggregate["exported_primary_strict_f1_std"],
                    "final_primary_strict_f1_mean": aggregate["final_primary_strict_f1_mean"],
                    "final_primary_strict_f1_std": aggregate["final_primary_strict_f1_std"],
                    "final_strict_f1_eps_0p1_mean": aggregate["final_strict_f1_eps_0p1_mean"],
                    "final_strict_f1_eps_0p1_std": aggregate["final_strict_f1_eps_0p1_std"],
                    "final_signed_margin_median_mean": aggregate["final_signed_margin_median_mean"],
                    "final_signed_margin_median_std": aggregate["final_signed_margin_median_std"],
                    "final_vs_best_gap_mean": aggregate["final_vs_best_gap_mean"],
                    "final_vs_best_gap_std": aggregate["final_vs_best_gap_std"],
                    "final_failure_mode_counts": aggregate["final_failure_mode_counts"],
                    **directional,
                    "summary_path": str(summary_path),
                    "aggregate_path": str(aggregate_path),
                    "manifest_path": str(manifest),
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: Dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def score_key(row: Dict[str, object]) -> tuple[float, float, float, float, float]:
    final_f1 = as_float(row, "final_primary_strict_f1_mean", -math.inf)
    eps = as_float(row, "final_strict_f1_eps_0p1_mean", -math.inf)
    shd = as_float(row, "final_shd_mean", math.inf)
    acc = as_float(row, "final_edge_accuracy_mean", -math.inf)
    gap = as_float(row, "final_vs_best_gap_mean", math.inf)
    return final_f1, eps, -shd, acc, -abs(gap)


def dataset_rank(dataset: str) -> int:
    try:
        return DATASET_ORDER.index(dataset)
    except ValueError:
        return len(DATASET_ORDER)


def sort_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["experiment"]),
            dataset_rank(str(row["dataset"])),
            as_float(row, "lag_weight", 0.0),
            as_float(row, "top_k_edges", 0.0),
        ),
    )


def best_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment"]), str(row["dataset"]))].append(row)

    out: List[Dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: (item[0][0], dataset_rank(item[0][1]))):
        best = max(group, key=score_key)
        best_row = dict(best)
        best_row["best_selection_scope"] = f"{key[0]}:{key[1]}"
        out.append(best_row)
    return out


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_mean_std(row: Dict[str, object], metric: str) -> str:
    return f"{fmt_float(row.get(metric + '_mean', ''))} +/- {fmt_float(row.get(metric + '_std', ''))}"


def markdown_table(rows: Sequence[Dict[str, object]], title: str) -> List[str]:
    lines = [
        f"## {title}",
        "",
        "| dataset | candidate | top_k | sel_k | lag | runs | best | exported | final | final eps=0.1 | margin | gap | dir F1 | SHD | edge acc | failure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {candidate} | {topk} | {selk} | {lag} | {runs} | {best} | {exported} | {final} | {eps} | {margin} | {gap} | {dirf1} | {shd} | {acc} | `{failure}` |".format(
                dataset=row["dataset"],
                candidate=row["candidate_key"],
                topk=row.get("top_k_edges", ""),
                selk=row.get("selection_top_k", ""),
                lag=row.get("lag_weight", ""),
                runs=row.get("run_count", ""),
                best=fmt_mean_std(row, "best_primary_strict_f1"),
                exported=fmt_mean_std(row, "exported_primary_strict_f1"),
                final=fmt_mean_std(row, "final_primary_strict_f1"),
                eps=fmt_mean_std(row, "final_strict_f1_eps_0p1"),
                margin=fmt_mean_std(row, "final_signed_margin_median"),
                gap=fmt_mean_std(row, "final_vs_best_gap"),
                dirf1=fmt_mean_std(row, "final_directional_f1"),
                shd=fmt_mean_std(row, "final_shd"),
                acc=fmt_mean_std(row, "final_edge_accuracy"),
                failure=row.get("final_failure_mode_counts", ""),
            )
        )
    lines.append("")
    return lines


def write_markdown(path: Path, all_rows: Sequence[Dict[str, object]], best: Sequence[Dict[str, object]]) -> None:
    topk = [row for row in sort_rows(all_rows) if row["experiment"] == "topk"]
    lag = [row for row in sort_rows(all_rows) if row["experiment"] == "lag"]
    lines = [
        "# Best-Base Parameter Tuning Summary",
        "",
        "Ranking rule: final primary strict F1, final strict F1 @ eps=0.1, lower final SHD, higher edge accuracy, then smaller absolute final-best gap.",
        "",
        "Support scan is topk-only: conclusions can only compare `topk_kappa` candidates inside each fixed-base dataset.",
        "",
    ]
    lines.extend(markdown_table(topk, "Experiment A: topk_kappa / top_k_edges"))
    lines.extend(markdown_table(lag, "Experiment B: causal_lag_main_weight"))
    lines.extend(markdown_table(sort_rows(best), "Best Under Fixed Base"))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.manifest, allow_incomplete=args.allow_incomplete)
    if not rows:
        raise RuntimeError("No completed manifest rows found.")

    rows = sort_rows(rows)
    topk_rows = [row for row in rows if row["experiment"] == "topk"]
    lag_rows = [row for row in rows if row["experiment"] == "lag"]
    best = best_rows(rows)

    prefix = args.output_prefix
    write_csv(Path(str(prefix) + "_summary.csv"), rows)
    if topk_rows:
        write_csv(Path(str(prefix) + "_topk.csv"), topk_rows)
    if lag_rows:
        write_csv(Path(str(prefix) + "_lag.csv"), lag_rows)
    write_csv(Path(str(prefix) + "_best.csv"), best)
    write_markdown(Path(str(prefix) + ".md"), rows, best)

    print(f"Rows: {len(rows)}")
    print(f"Summary CSV: {Path(str(prefix) + '_summary.csv')}")
    print(f"Best CSV: {Path(str(prefix) + '_best.csv')}")
    print(f"Markdown: {Path(str(prefix) + '.md')}")


if __name__ == "__main__":
    main()
