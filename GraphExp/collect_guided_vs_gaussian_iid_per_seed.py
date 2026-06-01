from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


Edge = Tuple[int, int]


DATASET_ORDER = ("fMRI", "sim2", "sim3", "sim4", "sim8", "sim10", "sim11", "sim12")
MODE_ORDER = ("guided", "gaussian_iid")


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
            if path.suffix == ".npy":
                return np.load(path)
            return np.loadtxt(path, delimiter=",")
    raise FileNotFoundError(f"Missing final causal adjacency under {run_dir}")


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
    total = n * (n - 1)
    if total <= 0:
        return 0.0
    correct = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge = (i, j)
            if (edge in pred_edges) == (edge in gt_edges):
                correct += 1
    return correct / total


def evaluate(adj: np.ndarray, gt_edges: Set[Edge]) -> Dict[str, float]:
    pred_edges = topk_directed_edges(adj, len(gt_edges))
    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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


def read_replay(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def add_rows(
    rows: List[Dict[str, object]],
    *,
    dataset: str,
    noise_mode: str,
    summary_file: Path,
    gt_file: Path,
) -> None:
    gt_edges = load_gt(gt_file)
    for run in read_replay(summary_file):
        run_dir = Path(run["result_dir"])
        metrics = evaluate(load_final_adj(run_dir), gt_edges)
        rows.append(
            {
                "dataset": dataset,
                "seed": int(run["seed"]),
                "noise_mode": noise_mode,
                "split": "final",
                "run_dir": str(run_dir),
                "summary_file": str(summary_file),
                "gt_file": str(gt_file),
                **metrics,
            }
        )


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["noise_mode"]))].append(row)

    metric_names = ("precision", "recall", "f1", "shd", "edge_accuracy")
    summary: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for mode in MODE_ORDER:
            group = grouped.get((dataset, mode), [])
            if not group:
                continue
            out: Dict[str, object] = {
                "dataset": dataset,
                "noise_mode": mode,
                "split": "final",
                "run_count": len(group),
                "seed_list": ",".join(str(row["seed"]) for row in sorted(group, key=lambda item: int(item["seed"]))),
            }
            for metric in metric_names:
                values = [float(row[metric]) for row in group]
                out[f"{metric}_mean"] = mean(values)
                out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
            summary.append(out)
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    return f"{float(value):.4f}"


def write_markdown(path: Path, rows: Sequence[Dict[str, object]], summary_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Guided vs Gaussian IID Per-Seed Final Metrics",
        "",
        "Evaluation convention:",
        "",
        "- Split: `final` only; adjacency source is `final_epoch_adjacency_causal.*`.",
        "- Predicted directed edges are the top `|GT|` unordered pairs ranked by `abs(A[i,j] - A[j,i])`.",
        "- Direction is chosen by the sign of `A[i,j] - A[j,i]`.",
        "- SHD is directed SHD: add/delete/reverse each cost `1`.",
        "",
    ]
    for dataset in DATASET_ORDER:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| seed | noise | TP | FP | FN | Precision | Recall | F1 | SHD | Edge Acc |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for seed in sorted({int(row["seed"]) for row in dataset_rows}):
            for mode in MODE_ORDER:
                matches = [row for row in dataset_rows if int(row["seed"]) == seed and row["noise_mode"] == mode]
                for row in matches:
                    lines.append(
                        "| {seed} | {mode} | {tp:.0f} | {fp:.0f} | {fn:.0f} | {precision} | {recall} | {f1} | {shd:.0f} | {acc} |".format(
                            seed=seed,
                            mode=mode,
                            tp=float(row["tp"]),
                            fp=float(row["fp"]),
                            fn=float(row["fn"]),
                            precision=fmt(row["precision"]),
                            recall=fmt(row["recall"]),
                            f1=fmt(row["f1"]),
                            shd=float(row["shd"]),
                            acc=fmt(row["edge_accuracy"]),
                        )
                    )
        lines.append("")

    lines.extend(
        [
            "## Aggregate",
            "",
            "| dataset | noise | Precision | Recall | F1 | SHD | Edge Acc |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {dataset} | {mode} | {precision:.4f} +/- {precision_std:.4f} | {recall:.4f} +/- {recall_std:.4f} | {f1:.4f} +/- {f1_std:.4f} | {shd:.4f} +/- {shd_std:.4f} | {acc:.4f} +/- {acc_std:.4f} |".format(
                dataset=row["dataset"],
                mode=row["noise_mode"],
                precision=float(row["precision_mean"]),
                precision_std=float(row["precision_std"]),
                recall=float(row["recall_mean"]),
                recall_std=float(row["recall_std"]),
                f1=float(row["f1_mean"]),
                f1_std=float(row["f1_std"]),
                shd=float(row["shd_mean"]),
                shd_std=float(row["shd_std"]),
                acc=float(row["edge_accuracy_mean"]),
                acc_std=float(row["edge_accuracy_std"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    repo = root.parent
    results = root / "results"
    gt_root = repo / "fMRI_dataset"

    specs = [
        ("fMRI", "guided", "unify_replay_fMRI_20260511_124620_fmri_F3_lag035_final_export_5seed.csv", "h1.txt"),
        ("fMRI", "gaussian_iid", "unify_replay_fMRI_20260513_165348_fmri_ablation_gaussian_iid_5seed.csv", "h1.txt"),
        ("sim2", "guided", "unify_replay_sim2_20260420_090228_sim2_2x2_incumbent_selfloop_alpha001.csv", "h2.txt"),
        ("sim2", "gaussian_iid", "unify_replay_sim2_20260513_185725_sim2_ablation_gaussian_iid_5seed.csv", "h2.txt"),
        ("sim3", "guided", "unify_replay_sim3_20260420_152303_sim3_dirend10_epochs100_seed11_probe.csv", "h3.txt"),
        ("sim3", "guided", "unify_replay_sim3_20260420_164742_sim3_dirend10_epochs100_part234.csv", "h3.txt"),
        ("sim3", "guided", "unify_replay_sim3_20260420_173624_sim3_dirend10_epochs100_seed55.csv", "h3.txt"),
        ("sim3", "gaussian_iid", "unify_replay_sim3_20260513_sim3_ablation_gaussian_iid_5seed_combined.csv", "h3.txt"),
        ("sim4", "guided", "unify_replay_sim4_20260420_175553_sim4_l15a_low_epochs100_seed11_probe.csv", "h4.txt"),
        ("sim4", "guided", "unify_replay_sim4_20260420_214502_sim4_l15a_low_epochs100_part2455.csv", "h4.txt"),
        ("sim4", "gaussian_iid", "unify_replay_sim4_20260514_sim4_ablation_gaussian_iid_5seed_combined.csv", "h4.txt"),
        ("sim8", "guided", "unify_replay_sim8_20260512_091903_sim8_gt_5seed_repretrain.csv", "h8.txt"),
        ("sim8", "gaussian_iid", "unify_replay_sim8_20260514_023604_sim8_ablation_gaussian_iid_5seed.csv", "h8.txt"),
        ("sim10", "guided", "unify_replay_sim10_20260512_095726_sim10_gt_5seed_repretrain.csv", "h10.txt"),
        ("sim10", "gaussian_iid", "unify_replay_sim10_20260514_033111_sim10_ablation_gaussian_iid_5seed.csv", "h10.txt"),
        ("sim11", "guided", "unify_replay_sim11_20260512_184811_sim11_D3_lag035_5seed.csv", "h11.txt"),
        ("sim11", "gaussian_iid", "unify_replay_sim11_20260514_050543_sim11_ablation_gaussian_iid_5seed.csv", "h11.txt"),
        ("sim12", "guided", "unify_replay_sim12_20260512_111301_sim12_gt_5seed_repretrain.csv", "h12.txt"),
        ("sim12", "gaussian_iid", "unify_replay_sim12_20260514_042044_sim12_ablation_gaussian_iid_5seed.csv", "h12.txt"),
    ]

    rows: List[Dict[str, object]] = []
    for dataset, mode, summary_name, gt_name in specs:
        add_rows(
            rows,
            dataset=dataset,
            noise_mode=mode,
            summary_file=results / summary_name,
            gt_file=gt_root / gt_name,
        )

    dataset_rank = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    mode_rank = {mode: index for index, mode in enumerate(MODE_ORDER)}
    rows.sort(key=lambda row: (dataset_rank[str(row["dataset"])], int(row["seed"]), mode_rank[str(row["noise_mode"])]))
    summary_rows = summarize(rows)

    out_prefix = results / "guided_vs_gaussian_iid_per_seed_metrics_20260601"
    write_csv(out_prefix.with_suffix(".csv"), rows)
    write_csv(Path(str(out_prefix) + "_aggregate.csv"), summary_rows)
    write_markdown(out_prefix.with_suffix(".md"), rows, summary_rows)
    print(out_prefix.with_suffix(".csv"))
    print(Path(str(out_prefix) + "_aggregate.csv"))
    print(out_prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
