from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


Edge = Tuple[int, int]


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


def load_adj(run_dir: Path, split: str) -> np.ndarray:
    filenames = {
        "exported": ("learned_adjacency_causal.npy", "learned_adjacency_causal.csv"),
        "final": ("final_epoch_adjacency_causal.npy", "final_epoch_adjacency_causal.csv"),
    }[split]
    for filename in filenames:
        path = run_dir / filename
        if path.exists():
            return np.load(path) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
    raise FileNotFoundError(f"Missing {split} causal adjacency under {run_dir}")


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
    fp = len(pred_remaining - gt_remaining)
    fn = len(gt_remaining - pred_remaining)
    return reversals + fp + fn


def edge_accuracy(pred_edges: Set[Edge], gt_edges: Set[Edge], n: int) -> float:
    directed_edges = {
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j
    }
    correct = sum(((edge in pred_edges) == (edge in gt_edges)) for edge in directed_edges)
    return correct / len(directed_edges) if directed_edges else 0.0


def evaluate_adj(adj: np.ndarray, gt_edges: Set[Edge]) -> Dict[str, float]:
    pred_edges = topk_directed_edges(adj, len(gt_edges))
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "num_nodes": float(adj.shape[0]),
        "gt_edges": float(len(gt_edges)),
        "pred_edges": float(len(pred_edges)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": float(directed_shd(pred_edges, gt_edges)),
        "edge_accuracy": edge_accuracy(pred_edges, gt_edges, int(adj.shape[0])),
    }


def read_summary_runs(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add_rows(
    rows: List[Dict[str, object]],
    *,
    dataset: str,
    variant: str,
    summary_path: Path,
    gt_path: Path,
) -> None:
    gt_edges = load_gt(gt_path)
    for run in read_summary_runs(summary_path):
        run_dir = Path(run["result_dir"])
        seed = int(run["seed"])
        for split in ("exported", "final"):
            metrics = evaluate_adj(load_adj(run_dir, split), gt_edges)
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "split": split,
                "seed": seed,
                "run_dir": str(run_dir),
                "summary_file": str(summary_path),
                "gt_file": str(gt_path),
                **metrics,
            })


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["variant"]), str(row["split"]))].append(row)

    metric_names = ("precision", "recall", "f1", "shd", "edge_accuracy")
    out: List[Dict[str, object]] = []
    for (dataset, variant, split), group in sorted(grouped.items()):
        agg: Dict[str, object] = {
            "dataset": dataset,
            "variant": variant,
            "split": split,
            "run_count": len(group),
            "seed_list": ",".join(str(row["seed"]) for row in group),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            agg[f"{metric}_mean"] = mean(values)
            agg[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(agg)
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_mean_std(row: Dict[str, object], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_std']):.4f}"


def write_markdown(path: Path, aggregate_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Ablation Precision/Recall/F1/SHD",
        "",
        "Evaluation convention:",
        "",
        "- Use saved causal adjacency snapshots.",
        "- `exported` uses `learned_adjacency_causal.*`; `final` uses `final_epoch_adjacency_causal.*`.",
        "- Predicted directed edges are the top `|GT|` unordered pairs ranked by `abs(A[i,j] - A[j,i])`.",
        "- Direction is chosen by the sign of `A[i,j] - A[j,i]`.",
        "- SHD is directed SHD: add/delete/reverse each cost `1`.",
        "",
        "| dataset | variant | split | precision | recall | F1 | SHD | edge accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| {dataset} | {variant} | {split} | {precision} | {recall} | {f1} | {shd} | {acc} |".format(
                dataset=row["dataset"],
                variant=row["variant"],
                split=row["split"],
                precision=fmt_mean_std(row, "precision"),
                recall=fmt_mean_std(row, "recall"),
                f1=fmt_mean_std(row, "f1"),
                shd=fmt_mean_std(row, "shd"),
                acc=fmt_mean_std(row, "edge_accuracy"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_prefix", type=str, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    results = root / "results"
    repo = root.parent

    specs = [
        ("fMRI", "disable_encoder", results / "unify_replay_fMRI_20260513_173101_fmri_ablation_disable_encoder_5seed_v2.csv", repo / "fMRI_dataset" / "h1.txt"),
        ("fMRI", "gaussian_iid", results / "unify_replay_fMRI_20260513_165348_fmri_ablation_gaussian_iid_5seed.csv", repo / "fMRI_dataset" / "h1.txt"),
        ("sim2", "disable_encoder", results / "unify_replay_sim2_20260513_185047_sim2_ablation_disable_encoder_5seed.csv", repo / "fMRI_dataset" / "h2.txt"),
        ("sim2", "gaussian_iid", results / "unify_replay_sim2_20260513_185725_sim2_ablation_gaussian_iid_5seed.csv", repo / "fMRI_dataset" / "h2.txt"),
        ("sim3", "disable_encoder", results / "unify_replay_sim3_20260513_192354_sim3_ablation_disable_encoder_5seed.csv", repo / "fMRI_dataset" / "h3.txt"),
        ("sim3", "gaussian_iid", results / "unify_replay_sim3_20260513_sim3_ablation_gaussian_iid_5seed_combined.csv", repo / "fMRI_dataset" / "h3.txt"),
        ("sim4", "disable_encoder", results / "unify_replay_sim4_20260513_211344_sim4_ablation_disable_encoder_5seed.csv", repo / "fMRI_dataset" / "h4.txt"),
        ("sim4", "gaussian_iid", results / "unify_replay_sim4_20260514_sim4_ablation_gaussian_iid_5seed_combined.csv", repo / "fMRI_dataset" / "h4.txt"),
        ("sim8", "disable_encoder", results / "unify_replay_sim8_20260514_022611_sim8_ablation_disable_encoder_5seed.csv", repo / "fMRI_dataset" / "h8.txt"),
        ("sim8", "gaussian_iid", results / "unify_replay_sim8_20260514_023604_sim8_ablation_gaussian_iid_5seed.csv", repo / "fMRI_dataset" / "h8.txt"),
        ("sim10", "disable_encoder", results / "unify_replay_sim10_20260514_032112_sim10_ablation_disable_encoder_5seed.csv", repo / "fMRI_dataset" / "h10.txt"),
        ("sim10", "gaussian_iid", results / "unify_replay_sim10_20260514_033111_sim10_ablation_gaussian_iid_5seed.csv", repo / "fMRI_dataset" / "h10.txt"),
        ("sim11", "disable_encoder", results / "unify_replay_sim11_20260514_050123_sim11_ablation_disable_encoder_5seed.csv", repo / "fMRI_dataset" / "h11.txt"),
        ("sim11", "gaussian_iid", results / "unify_replay_sim11_20260514_050543_sim11_ablation_gaussian_iid_5seed.csv", repo / "fMRI_dataset" / "h11.txt"),
        ("sim12", "disable_encoder", results / "unify_replay_sim12_20260514_041620_sim12_ablation_disable_encoder_5seed.csv", repo / "fMRI_dataset" / "h12.txt"),
        ("sim12", "gaussian_iid", results / "unify_replay_sim12_20260514_042044_sim12_ablation_gaussian_iid_5seed.csv", repo / "fMRI_dataset" / "h12.txt"),
    ]

    per_run_rows: List[Dict[str, object]] = []
    for dataset, variant, summary_path, gt_path in specs:
        add_rows(
            per_run_rows,
            dataset=dataset,
            variant=variant,
            summary_path=summary_path,
            gt_path=gt_path,
        )

    aggregate = summarize(per_run_rows)
    prefix = results / args.output_prefix
    write_csv(prefix.with_suffix(".csv"), per_run_rows)
    write_csv(Path(str(prefix) + "_aggregate.csv"), aggregate)
    write_markdown(Path(str(prefix) + ".md"), aggregate)
    print(f"Per-run CSV: {prefix.with_suffix('.csv')}")
    print(f"Aggregate CSV: {Path(str(prefix) + '_aggregate.csv')}")
    print(f"Markdown: {Path(str(prefix) + '.md')}")


if __name__ == "__main__":
    main()
