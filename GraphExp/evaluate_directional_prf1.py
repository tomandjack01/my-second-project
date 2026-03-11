# Usage:
# 1) Run with latest result under GraphExp/results/run_*/learned_adjacency.csv
#    python .\GraphExp\evaluate_directional_prf1.py --gt .\fMRI_dataset\h1.txt
#
# 2) Run with a specific prediction file
#    python .\GraphExp\evaluate_directional_prf1.py --pred .\GraphExp\results\run_20260302_214639\learned_adjacency.csv --gt .\fMRI_dataset\h1.txt
#
# 3) Optional: limit number of printed predicted edges
#    python .\GraphExp\evaluate_directional_prf1.py --gt .\fMRI_dataset\h1.txt --list_limit 50
#
# Direction rule used in evaluation:
# For each node pair (i, j), if A[i, j] > A[j, i], predict i->j; otherwise predict j->i.

import argparse
from pathlib import Path

import numpy as np


def load_ground_truth(gt_path: Path):
    gt_edges = set()
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.replace(",", " ").split()
            if len(parts) < 2:
                continue
            src = int(parts[0]) - 1
            dst = int(parts[1]) - 1
            if src == dst:
                continue
            gt_edges.add((src, dst))
    return gt_edges


def find_latest_pred(results_dir: Path) -> Path:
    run_dirs = sorted(
        [p for p in results_dir.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        for filename in (
            "learned_adjacency_causal.npy",
            "learned_adjacency_causal.csv",
            "learned_adjacency.npy",
            "learned_adjacency.csv",
        ):
            pred_path = run_dir / filename
            if pred_path.exists():
                return pred_path
    raise FileNotFoundError(
        f"No learned adjacency file found under: {results_dir}"
    )


def infer_adjacency_convention(pred_path: Path) -> str:
    return "causal" if "causal" in pred_path.stem.lower() else "raw"


def load_adjacency(pred_path: Path, adj_convention: str = "auto"):
    if pred_path.suffix.lower() == ".npy":
        adj = np.load(pred_path)
    else:
        adj = np.loadtxt(pred_path, delimiter=",")
    if adj_convention == "auto":
        adj_convention = infer_adjacency_convention(pred_path)
    if adj_convention == "raw":
        adj = adj.T
    elif adj_convention != "causal":
        raise ValueError(f"Unsupported --adj_convention={adj_convention!r}")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape={adj.shape}")
    return adj, adj_convention


def evaluate_directional(adj, gt_edges):
    n = adj.shape[0]
    predictions = []
    tie_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            w_ij = float(adj[i, j])
            w_ji = float(adj[j, i])
            if w_ij > w_ji:
                src, dst = i, j
            elif w_ij < w_ji:
                src, dst = j, i
            else:
                tie_count += 1
                src, dst = i, j
            predictions.append((src, dst, abs(w_ij - w_ji)))

    pred_edges = {(src, dst) for src, dst, _ in predictions}
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
        "predictions": predictions,
        "tp_edges": tp_edges,
        "fn_edges": fn_edges,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tie_count": tie_count,
    }


def format_edge(edge):
    return f"({edge[0] + 1},{edge[1] + 1})"


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Evaluate directed edges with rule: A[i,j] > A[j,i] => i->j"
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Path to learned adjacency csv/npy (default: latest run in GraphExp/results)",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=str(repo_root / "fMRI_dataset" / "h1.txt"),
        help="Path to ground-truth edge list txt",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(script_dir / "results"),
        help="Directory containing run_* folders (used when --pred is not set)",
    )
    parser.add_argument(
        "--list_limit",
        type=int,
        default=100,
        help="Max number of predicted edges to print by confidence",
    )
    parser.add_argument(
        "--adj_convention",
        type=str,
        choices=["auto", "raw", "causal"],
        default="auto",
        help="Interpretation of the prediction file. raw means transpose before evaluation.",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred) if args.pred else find_latest_pred(Path(args.results_dir))
    gt_path = Path(args.gt)

    adj, used_convention = load_adjacency(pred_path, adj_convention=args.adj_convention)
    gt_edges = load_ground_truth(gt_path)
    result = evaluate_directional(adj, gt_edges)

    print("=" * 72)
    print(f"Pred file: {pred_path}")
    print(f"Adj convention used: {used_convention} -> evaluated as causal")
    print(f"GT file:   {gt_path}")
    print(f"Matrix shape: {adj.shape}")
    print("-" * 72)
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")
    print(f"TP={result['tp']}, FP={result['fp']}, FN={result['fn']}, Ties={result['tie_count']}")
    print("=" * 72)

    matched = sorted(result["tp_edges"])
    missed = sorted(result["fn_edges"])
    print("Matched edges (TP):", [format_edge(e) for e in matched])
    print("Missed GT edges (FN):", [format_edge(e) for e in missed])
    print("-" * 72)

    predictions = sorted(result["predictions"], key=lambda x: x[2], reverse=True)
    limit = max(0, args.list_limit)
    if limit:
        print(f"Top {min(limit, len(predictions))} predicted edges by |A[i,j]-A[j,i]|:")
        for src, dst, margin in predictions[:limit]:
            tag = "TP" if (src, dst) in result["tp_edges"] else "FP"
            print(f"{format_edge((src, dst))} margin={margin:.4f} [{tag}]")


if __name__ == "__main__":
    main()
