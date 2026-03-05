# Usage:
# 1) 按稀疏度评估 (默认保留前 5% 的边，最推荐处理未知数据):
#    python .\GraphExp\evaluate_directional_prf1.py --gt .\fMRI_dataset\h1.txt --sparsity 0.05
#
# 2) 按固定边数评估 (例如知道 GT 只有大约 60 条边):
#    python .\GraphExp\evaluate_directional_prf1.py --gt .\fMRI_dataset\h1.txt --top_k 60
#
# 3) Run with a specific prediction file
#    python .\GraphExp\evaluate_directional_prf1.py --pred .\GraphExp\results\run_20260302_214639\learned_adjacency.csv --gt .\fMRI_dataset\h1.txt

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
        pred_path = run_dir / "learned_adjacency.csv"
        if pred_path.exists():
            return pred_path
    raise FileNotFoundError(
        f"No learned_adjacency.csv found under: {results_dir}"
    )


def load_adjacency(pred_path: Path):
    adj = np.loadtxt(pred_path, delimiter=",")
    # ========== 加入这极其关键的一行 ==========
    adj = adj.T  # 将 GNN 视角的矩阵转置为 标准因果图视角
    # ==========================================
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape={adj.shape}")
    return adj


def evaluate_directional(adj, gt_edges, top_k=None, sparsity=0.05):
    n = adj.shape[0]
    candidate_edges = []
    tie_count = 0

    # Step 1: 提取所有可能的方向边及其绝对权重
    for i in range(n):
        for j in range(i + 1, n):
            w_ij = float(adj[i, j])
            w_ji = float(adj[j, i])
            
            # 确定方向，并将该方向的绝对权重作为置信度
            if w_ij > w_ji:
                src, dst = i, j
                weight = w_ij
            elif w_ij < w_ji:
                src, dst = j, i
                weight = w_ji
            else:
                tie_count += 1
                src, dst = i, j
                weight = w_ij
                
            candidate_edges.append((src, dst, weight, abs(w_ij - w_ji)))

    # Step 2: 按绝对权重从大到小排序 (核心修改点)
    candidate_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Step 3: 计算需要保留的边数 K
    total_possible_pairs = n * (n - 1) // 2
    if top_k is not None:
        k = top_k
    else:
        k = int(total_possible_pairs * sparsity)
        
    k = max(1, min(k, total_possible_pairs)) # 保证 K 的合法范围

    # Step 4: 截断并生成最终预测
    predictions = candidate_edges[:k]
    pred_edges = {(src, dst) for src, dst, _, _ in predictions}
    
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
        "k_used": k,
        "total_pairs": total_possible_pairs
    }


def format_edge(edge):
    return f"({edge[0] + 1},{edge[1] + 1})"


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Evaluate directed edges using Top-K or Sparsity truncation."
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Path to learned_adjacency.csv (default: latest run in GraphExp/results)",
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
        "--top_k",
        type=int,
        default=None,
        help="Number of edges to keep (e.g., 60). If set, overrides --sparsity.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.05,
        help="Fraction of possible edges to keep (default: 0.05).",
    )
    parser.add_argument(
        "--list_limit",
        type=int,
        default=100,
        help="Max number of predicted edges to print by confidence",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred) if args.pred else find_latest_pred(Path(args.results_dir))
    gt_path = Path(args.gt)

    adj = load_adjacency(pred_path)
    gt_edges = load_ground_truth(gt_path)
    
    # 传入新的参数
    result = evaluate_directional(adj, gt_edges, top_k=args.top_k, sparsity=args.sparsity)

    print("=" * 72)
    print(f"Pred file: {pred_path}")
    print(f"GT file:   {gt_path}")
    print(f"Matrix shape: {adj.shape} (Total possible undirected edges: {result['total_pairs']})")
    print(f"Truncation: Kept Top {result['k_used']} edges")
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

    predictions = result["predictions"]
    limit = max(0, args.list_limit)
    if limit:
        print(f"Top {min(limit, len(predictions))} predicted edges by absolute weight:")
        for src, dst, weight, margin in predictions[:limit]:
            tag = "TP" if (src, dst) in result["tp_edges"] else "FP"
            print(f"{format_edge((src, dst))} weight={weight:.4f} (margin={margin:.4f}) [{tag}]")

if __name__ == "__main__":
    main()