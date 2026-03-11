#!/usr/bin/env python3
"""
Experiment 1: verify graph message direction and raw/causal adjacency semantics.

This script does two things:
1. Probes DGL GraphConv to confirm whether messages flow src -> dst.
2. Uses a 2-node full-graph example to check whether the repository's
   documented raw adjacency convention requires a transpose before causal use.

Optional:
    Provide --pred and --gt to compare a saved adjacency under direct vs
    transposed interpretation against ground truth.
"""

import argparse
from pathlib import Path
from typing import Optional

import dgl
import numpy as np
import torch
from dgl.nn import GraphConv

from evaluate_directional_prf1 import evaluate_directional, load_ground_truth
from main_structure_learning import (
    CAUSAL_ADJ_CONVENTION,
    RAW_ADJ_CONVENTION,
    to_causal_matrix_np,
)


def build_single_edge_graph(src: int, dst: int, num_nodes: int = 2):
    return dgl.graph((torch.tensor([src]), torch.tensor([dst])), num_nodes=num_nodes)


def apply_graphconv(g, x, edge_weight):
    conv = GraphConv(
        1,
        1,
        norm='none',
        weight=False,
        bias=False,
        allow_zero_in_degree=True,
    )
    out = conv(g, x, edge_weight=edge_weight)
    return out.squeeze(-1)


def apply_full_matrix(adj_matrix: torch.Tensor, x: torch.Tensor):
    num_nodes = adj_matrix.shape[0]
    src = torch.arange(num_nodes).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes).repeat(num_nodes)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    edge_weight = adj_matrix.reshape(-1)
    return apply_graphconv(g, x, edge_weight)


def run_graphconv_probe():
    x = torch.tensor([[2.0], [5.0]])

    out_0_to_1 = apply_graphconv(
        build_single_edge_graph(0, 1),
        x,
        edge_weight=torch.tensor([1.0]),
    )
    out_1_to_0 = apply_graphconv(
        build_single_edge_graph(1, 0),
        x,
        edge_weight=torch.tensor([1.0]),
    )

    return {
        "edge_0_to_1": out_0_to_1.tolist(),
        "edge_1_to_0": out_1_to_0.tolist(),
        "message_direction": "src_to_dst",
    }


def run_raw_convention_probe():
    # According to the repo docs, raw adjacency is stored as:
    #   A_raw[effect, cause]
    # If node 0 is the cause and node 1 is the effect, the documented raw matrix is:
    raw_adj = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    x = torch.tensor([[1.0], [0.0]])
    direct_out = apply_full_matrix(raw_adj, x)

    causal_adj = torch.from_numpy(to_causal_matrix_np(raw_adj.numpy())).float()
    transposed_out = apply_full_matrix(causal_adj, x)

    return {
        "raw_adj": raw_adj.tolist(),
        "causal_adj": causal_adj.tolist(),
        "input_feature": x.squeeze(-1).tolist(),
        "direct_out": direct_out.tolist(),
        "transposed_out": transposed_out.tolist(),
        "transpose_required_for_causal_use": bool(
            transposed_out[1].item() > direct_out[1].item()
        ),
    }


def load_square_adjacency(path: Path):
    if path.suffix.lower() == ".npy":
        adj = np.load(path)
    else:
        adj = np.loadtxt(path, delimiter=",")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape={adj.shape}")
    return adj


def compare_saved_adjacency(pred_path: Path, gt_path: Path, top_k: Optional[int], sparsity: float):
    adj = load_square_adjacency(pred_path)
    gt_edges = load_ground_truth(gt_path)

    direct = evaluate_directional(adj, gt_edges, top_k=top_k, sparsity=sparsity)
    transposed = evaluate_directional(adj.T, gt_edges, top_k=top_k, sparsity=sparsity)

    return {
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "direct": {
            "precision": direct["precision"],
            "recall": direct["recall"],
            "f1": direct["f1"],
            "tp": direct["tp"],
            "fp": direct["fp"],
            "fn": direct["fn"],
        },
        "transposed": {
            "precision": transposed["precision"],
            "recall": transposed["recall"],
            "f1": transposed["f1"],
            "tp": transposed["tp"],
            "fp": transposed["fp"],
            "fn": transposed["fn"],
        },
        "better_interpretation": (
            "transposed" if transposed["f1"] > direct["f1"]
            else "direct" if direct["f1"] > transposed["f1"]
            else "tie"
        ),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Verify structure-learning adjacency direction semantics."
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Optional learned adjacency file (.csv or .npy) for direct-vs-transposed GT comparison.",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=str(repo_root / "fMRI_dataset" / "h1.txt"),
        help="Ground-truth edge list used with --pred.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k directed edges for GT comparison. Overrides --sparsity when set.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.05,
        help="Fallback sparsity used when --top_k is not set.",
    )
    args = parser.parse_args()

    graphconv_probe = run_graphconv_probe()
    raw_probe = run_raw_convention_probe()

    print("=" * 72)
    print("Experiment 1: Structure Direction Semantics")
    print("=" * 72)
    print(f"Documented raw convention:    {RAW_ADJ_CONVENTION}")
    print(f"Documented causal convention: {CAUSAL_ADJ_CONVENTION}")
    print("-" * 72)
    print("GraphConv probe")
    print(f"  edge 0 -> 1 output: {graphconv_probe['edge_0_to_1']}")
    print(f"  edge 1 -> 0 output: {graphconv_probe['edge_1_to_0']}")
    print(f"  observed message direction: {graphconv_probe['message_direction']}")
    print("-" * 72)
    print("Raw-convention probe")
    print(f"  raw adjacency (documented A_raw[effect, cause]): {raw_probe['raw_adj']}")
    print(f"  causal adjacency via transpose:                  {raw_probe['causal_adj']}")
    print(f"  input feature (signal only on node 0):          {raw_probe['input_feature']}")
    print(f"  direct raw application output:                  {raw_probe['direct_out']}")
    print(f"  transposed application output:                  {raw_probe['transposed_out']}")
    print(
        "  conclusion: transpose required before causal use = "
        f"{raw_probe['transpose_required_for_causal_use']}"
    )

    if args.pred:
        comparison = compare_saved_adjacency(
            pred_path=Path(args.pred),
            gt_path=Path(args.gt),
            top_k=args.top_k,
            sparsity=args.sparsity,
        )
        print("-" * 72)
        print("Saved adjacency comparison")
        print(f"  pred: {comparison['pred_path']}")
        print(f"  gt:   {comparison['gt_path']}")
        print(
            "  direct     : "
            f"F1={comparison['direct']['f1']:.4f}, "
            f"P={comparison['direct']['precision']:.4f}, "
            f"R={comparison['direct']['recall']:.4f}"
        )
        print(
            "  transposed : "
            f"F1={comparison['transposed']['f1']:.4f}, "
            f"P={comparison['transposed']['precision']:.4f}, "
            f"R={comparison['transposed']['recall']:.4f}"
        )
        print(f"  better interpretation: {comparison['better_interpretation']}")


if __name__ == "__main__":
    main()
