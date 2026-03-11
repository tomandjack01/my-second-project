#!/usr/bin/env python3
"""
Experiment 2: single-factor ablation for LayerNorm applied to guided noise.

This script compares:
1. Noise statistics with `normalize_noise=True` vs `False` using the same input and eps.
2. Training outcomes under the two settings with all other knobs fixed.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from main_structure_learning import (
    build_noise_guide_adjacency,
    compute_global_pearson,
    load_fmri_data,
    set_seed,
    train_brain_connectivity,
)
from models import DDM
from utils.patel_util import compute_patel_components


def summarize_noise_tensor(noise: torch.Tensor, base_mean: torch.Tensor, base_std: torch.Tensor):
    node_mean = noise.mean(dim=-1)
    node_std = noise.std(dim=-1)
    base_node_mean = base_mean.mean(dim=-1)
    base_node_std = base_std.mean(dim=-1)

    return {
        "node_mean_avg": float(node_mean.mean().item()),
        "node_mean_std": float(node_mean.std().item()),
        "node_std_avg": float(node_std.mean().item()),
        "node_std_std": float(node_std.std().item()),
        "mean_abs_shift_vs_base_mean": float((node_mean - base_node_mean).abs().mean().item()),
        "std_abs_shift_vs_base_std": float((node_std - base_node_std).abs().mean().item()),
        "mean_cosine_to_base_mean": float(
            F.cosine_similarity(noise, base_mean, dim=-1).mean().item()
        ),
    }


def build_probe_model(
    num_nodes: int,
    time_points: int,
    init_features: torch.Tensor,
    noise_guide_adj: Optional[torch.Tensor],
    device: str,
    num_hidden: int,
    num_layers: int,
    use_temporal_encoder: bool,
    normalize_noise: bool,
):
    model = DDM(
        in_dim=time_points,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=4,
        activation='prelu',
        feat_drop=0.1,
        attn_drop=0.1,
        norm='layernorm',
        alpha_l=2,
        init_features=init_features,
        noise_guide_adj=noise_guide_adj,
        use_temporal_encoder=use_temporal_encoder,
        normalize_noise=normalize_noise,
    )
    return model.to(device)


def collect_noise_probe_stats(
    x: torch.Tensor,
    init_features: torch.Tensor,
    noise_guide_adj: Optional[torch.Tensor],
    device: str,
    num_hidden: int,
    num_layers: int,
    use_temporal_encoder: bool,
    seed: int,
):
    set_seed(seed)
    probe_model = build_probe_model(
        num_nodes=x.shape[0],
        time_points=x.shape[1],
        init_features=init_features,
        noise_guide_adj=noise_guide_adj,
        device=device,
        num_hidden=num_hidden,
        num_layers=num_layers,
        use_temporal_encoder=use_temporal_encoder,
        normalize_noise=True,
    )

    x = x.to(device)
    if use_temporal_encoder:
        x = probe_model.temporal_encoder(x)
    x = F.layer_norm(x, (x.shape[-1],))

    set_seed(seed + 1)
    eps = torch.randn(1, *x.shape, device=x.device)

    noise_on, details = probe_model.build_noise(
        x,
        eps=eps,
        normalize_noise=True,
        return_details=True,
    )
    noise_off = probe_model.build_noise(
        x,
        eps=eps,
        normalize_noise=False,
        return_details=False,
    )

    stats_on = summarize_noise_tensor(noise_on, details["base_mean"], details["base_std"])
    stats_off = summarize_noise_tensor(noise_off, details["base_mean"], details["base_std"])

    return {
        "noise_source": details["noise_source"],
        "with_layernorm": stats_on,
        "without_layernorm": stats_off,
    }


def run_training_variant(
    variant_name: str,
    normalize_noise: bool,
    data_3d: torch.Tensor,
    pearson_matrix: torch.Tensor,
    patel_score_matrix: torch.Tensor,
    patel_tau_matrix: torch.Tensor,
    patel_kappa_matrix: torch.Tensor,
    noise_guide_adj: Optional[torch.Tensor],
    k_pairs: int,
    args,
    result_dir: Optional[str],
):
    set_seed(args.seed)
    model, adj_matrix, loss_history, collapse_history, best_epoch = train_brain_connectivity(
        data_3d=data_3d,
        pearson_matrix=pearson_matrix,
        num_nodes=data_3d.shape[1],
        time_points=args.time_points,
        patel_matrix=patel_score_matrix,
        patel_direction_matrix=patel_tau_matrix,
        patel_strength_matrix=torch.clamp(patel_kappa_matrix, min=0.0),
        noise_guide_adj=noise_guide_adj,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lambda_l1=args.lambda_l1,
        device=args.device,
        log_interval=args.log_interval,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        ddm_kwargs={
            "use_temporal_encoder": not args.disable_temporal_encoder,
            "normalize_noise": normalize_noise,
        },
        skip_pretrain=args.skip_pretrain or args.pretrain_epochs <= 0,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_lr=args.pretrain_lr,
        result_dir=result_dir,
        target_edge_count=k_pairs,
        selection_top_k=k_pairs,
        selection_start_epoch=args.selection_start_epoch,
        selection_min_skeleton_overlap=args.selection_min_skeleton_overlap,
        selection_min_skeleton_retention=args.selection_min_skeleton_retention,
        selection_min_density_factor=args.selection_min_density_factor,
        selection_max_density_ratio=args.selection_max_density_ratio,
    )

    best_quality = model.best_epoch_quality or {}
    final_quality = model.quality_history[-1] if getattr(model, "quality_history", None) else {}
    raw_best_score = float(getattr(model, "best_epoch_score", -1.0))
    raw_best_epoch = int(best_epoch)

    if raw_best_score >= 0.0:
        reported_score = raw_best_score
        reported_quality = best_quality
        score_source = "best_epoch"
    else:
        reported_score = float(final_quality.get("score", -1.0))
        reported_quality = final_quality
        score_source = "final_epoch_fallback"

    return {
        "variant": variant_name,
        "normalize_noise": normalize_noise,
        "best_epoch": raw_best_epoch,
        "best_score": reported_score,
        "score_source": score_source,
        "selection_mode": str(getattr(model, "best_epoch_selection_mode", "unknown")),
        "best_quality": {
            "agreement": float(reported_quality.get("agreement", 0.0)),
            "agreement_score": float(reported_quality.get("agreement_score", 0.0)),
            "dir_margin": float(reported_quality.get("dir_margin", 0.0)),
            "density_factor": float(reported_quality.get("density_factor", 0.0)),
            "skeleton_overlap": float(reported_quality.get("skeleton_overlap", 0.0)),
            "actual_pair_density": float(reported_quality.get("actual_pair_density", 0.0)),
            "target_pair_density": float(reported_quality.get("target_pair_density", 0.0)),
        },
        "final_quality": {
            "score": float(final_quality.get("score", 0.0)),
            "agreement": float(final_quality.get("agreement", 0.0)),
            "dir_margin": float(final_quality.get("dir_margin", 0.0)),
            "density_factor": float(final_quality.get("density_factor", 0.0)),
            "skeleton_overlap": float(final_quality.get("skeleton_overlap", 0.0)),
        },
        "final_loss": float(loss_history[-1]) if loss_history else None,
        "num_logged_collapse_epochs": int(len(collapse_history)),
        "adjacency_mean": float(np.mean(adj_matrix)),
        "adjacency_l1_mean": float(np.mean(np.abs(adj_matrix))),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Single-factor ablation for LayerNorm applied to guided noise."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=str(repo_root / "fMRI_dataset" / "fMRI.csv"),
        help="Path to fMRI CSV file.",
    )
    parser.add_argument("--time_points", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_l1", type=float, default=0.02)
    parser.add_argument("--num_hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_k_edges", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--pretrain_epochs", type=int, default=0)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--skip_pretrain", action="store_true", default=False)
    parser.add_argument(
        "--disable_temporal_encoder",
        action="store_true",
        default=False,
        help="Disable the temporal encoder and operate directly on raw time series.",
    )
    parser.add_argument("--selection_start_epoch", type=int, default=6)
    parser.add_argument("--selection_min_skeleton_overlap", type=float, default=0.50)
    parser.add_argument("--selection_min_skeleton_retention", type=float, default=0.85)
    parser.add_argument("--selection_min_density_factor", type=float, default=0.65)
    parser.add_argument("--selection_max_density_ratio", type=float, default=2.50)
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Optional directory for per-variant artifacts and summary json.",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Optional path to save the final summary json.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to cpu")
        args.device = "cpu"

    set_seed(args.seed)

    data_3d, data_2d, _, num_nodes = load_fmri_data(args.csv_path, args.time_points)
    pearson_matrix = compute_global_pearson(data_2d)

    patel_score_np, patel_kappa_np, patel_tau_np = compute_patel_components(data_2d.numpy())
    patel_score_matrix = torch.from_numpy(patel_score_np).float()
    patel_kappa_matrix = torch.from_numpy(patel_kappa_np).float()
    patel_tau_matrix = torch.from_numpy(patel_tau_np).float()

    noise_guide_adj, _, k_pairs, threshold = build_noise_guide_adjacency(
        patel_strength_matrix=torch.clamp(patel_kappa_matrix, min=0.0),
        top_k_pairs=args.top_k_edges,
    )

    noise_probe = collect_noise_probe_stats(
        x=data_3d[0],
        init_features=patel_score_matrix,
        noise_guide_adj=noise_guide_adj,
        device=args.device,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        use_temporal_encoder=not args.disable_temporal_encoder,
        seed=args.seed,
    )

    results_root = Path(args.results_dir) if args.results_dir else None
    if results_root is not None:
        results_root.mkdir(parents=True, exist_ok=True)

    with_ln_result = run_training_variant(
        variant_name="with_noise_layernorm",
        normalize_noise=True,
        data_3d=data_3d,
        pearson_matrix=pearson_matrix,
        patel_score_matrix=patel_score_matrix,
        patel_tau_matrix=patel_tau_matrix,
        patel_kappa_matrix=patel_kappa_matrix,
        noise_guide_adj=noise_guide_adj,
        k_pairs=k_pairs,
        args=args,
        result_dir=str(results_root / "with_noise_layernorm") if results_root else None,
    )

    without_ln_result = run_training_variant(
        variant_name="without_noise_layernorm",
        normalize_noise=False,
        data_3d=data_3d,
        pearson_matrix=pearson_matrix,
        patel_score_matrix=patel_score_matrix,
        patel_tau_matrix=patel_tau_matrix,
        patel_kappa_matrix=patel_kappa_matrix,
        noise_guide_adj=noise_guide_adj,
        k_pairs=k_pairs,
        args=args,
        result_dir=str(results_root / "without_noise_layernorm") if results_root else None,
    )

    adjacency_l1_gap = abs(with_ln_result["adjacency_l1_mean"] - without_ln_result["adjacency_l1_mean"])
    best_score_gap = with_ln_result["best_score"] - without_ln_result["best_score"]

    summary = {
        "config": {
            "csv_path": args.csv_path,
            "time_points": args.time_points,
            "epochs": args.epochs,
            "lr": args.lr,
            "lambda_l1": args.lambda_l1,
            "num_hidden": args.num_hidden,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "top_k_edges": args.top_k_edges,
            "noise_guide_threshold": float(threshold),
            "noise_guide_pairs": int(k_pairs),
            "device": args.device,
            "seed": args.seed,
            "use_temporal_encoder": not args.disable_temporal_encoder,
            "pretrain_epochs": args.pretrain_epochs,
        },
        "noise_probe": noise_probe,
        "training": {
            "with_layernorm": with_ln_result,
            "without_layernorm": without_ln_result,
            "best_score_gap_with_minus_without": best_score_gap,
            "adjacency_l1_mean_gap": adjacency_l1_gap,
        },
    }

    print("=" * 72)
    print("Experiment 2: Noise LayerNorm Ablation")
    print("=" * 72)
    print(f"Dataset: {args.csv_path}")
    print(f"Device: {args.device}")
    print(f"Temporal encoder enabled: {not args.disable_temporal_encoder}")
    print(f"Noise-guide pairs: {k_pairs} (threshold={threshold:.4f})")
    print("-" * 72)
    print("Noise probe")
    print(
        "  with LayerNorm   : "
        f"node_mean_avg={noise_probe['with_layernorm']['node_mean_avg']:.4f}, "
        f"node_std_avg={noise_probe['with_layernorm']['node_std_avg']:.4f}, "
        f"mean_shift={noise_probe['with_layernorm']['mean_abs_shift_vs_base_mean']:.4f}, "
        f"std_shift={noise_probe['with_layernorm']['std_abs_shift_vs_base_std']:.4f}"
    )
    print(
        "  without LayerNorm: "
        f"node_mean_avg={noise_probe['without_layernorm']['node_mean_avg']:.4f}, "
        f"node_std_avg={noise_probe['without_layernorm']['node_std_avg']:.4f}, "
        f"mean_shift={noise_probe['without_layernorm']['mean_abs_shift_vs_base_mean']:.4f}, "
        f"std_shift={noise_probe['without_layernorm']['std_abs_shift_vs_base_std']:.4f}"
    )
    print("-" * 72)
    print("Training summary")
    print(
        "  with LayerNorm   : "
        f"best_score={with_ln_result['best_score']:.4f}, "
        f"best_epoch={with_ln_result['best_epoch']}, "
        f"source={with_ln_result['score_source']}, "
        f"skeleton={with_ln_result['best_quality']['skeleton_overlap']:.4f}, "
        f"density={with_ln_result['best_quality']['density_factor']:.4f}"
    )
    print(
        "  without LayerNorm: "
        f"best_score={without_ln_result['best_score']:.4f}, "
        f"best_epoch={without_ln_result['best_epoch']}, "
        f"source={without_ln_result['score_source']}, "
        f"skeleton={without_ln_result['best_quality']['skeleton_overlap']:.4f}, "
        f"density={without_ln_result['best_quality']['density_factor']:.4f}"
    )
    print(
        "  gaps             : "
        f"best_score(with-minus-without)={best_score_gap:.4f}, "
        f"adj_l1_gap={adjacency_l1_gap:.4f}"
    )

    summary_path = Path(args.summary_path) if args.summary_path else None
    if summary_path is None and results_root is not None:
        summary_path = results_root / "summary.json"
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
