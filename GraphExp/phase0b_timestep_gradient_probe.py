import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import dgl

from models.DDM import DDM
from utils.patel_util import compute_patel_components
from main_structure_learning import (
    build_directed_noise_guide_adjacency,
    build_noise_guide_adjacency,
    build_structure_init_matrix,
    compute_target_density,
    get_current_directional_logits,
    load_fmri_data,
)


def parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_gt_edges(gt_path: Path) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            edges.append((int(parts[0]) - 1, int(parts[1]) - 1))
    if not edges:
        raise ValueError(f"No GT edges found in {gt_path}")
    return edges


def build_model_from_run_dir(run_dir: Path, device: str) -> Tuple[DDM, Dict[str, object], torch.Tensor]:
    config = np.load(run_dir / "config.npy", allow_pickle=True).item()
    checkpoint = torch.load(run_dir / "model_final.pt", map_location=device)

    csv_path = Path(config["csv_path"])
    if not csv_path.is_absolute():
        csv_path = (run_dir.parent.parent / csv_path).resolve()

    data_3d, data_2d, _, num_nodes = load_fmri_data(
        str(csv_path),
        time_points_per_subject=int(config.get("time_points", 200)),
        subject_limit=int(config.get("subject_limit", -1)),
        time_limit=int(config.get("time_limit", -1)),
    )
    time_points = int(data_3d.shape[-1])

    patel_score_np, patel_kappa_np, patel_tau_np = compute_patel_components(data_2d.numpy())
    patel_score_matrix = torch.from_numpy(patel_score_np).float().to(device)
    patel_kappa_matrix = torch.from_numpy(patel_kappa_np).float().to(device)
    patel_tau_matrix = torch.from_numpy(patel_tau_np).float().to(device)

    data_norm = (data_2d - data_2d.mean(dim=0, keepdim=True)) / (data_2d.std(dim=0, keepdim=True) + 1e-8)
    pearson_matrix = (data_norm.T @ data_norm / data_norm.shape[0]).float().to(device)

    structure_init_matrix = build_structure_init_matrix(
        mode=str(config.get("structure_init_mode", "patel_kappa")),
        patel_score_matrix=patel_score_matrix,
        patel_kappa_matrix=patel_kappa_matrix,
        pearson_matrix=pearson_matrix,
        seed=int(config.get("seed", 0)),
    )

    direction_init_mode = str(config.get("direction_init_mode", "random"))
    if direction_init_mode == "patel_tau":
        direction_init_matrix = patel_tau_matrix.clone()
    elif direction_init_mode == "zeros":
        direction_init_matrix = torch.zeros_like(patel_tau_matrix)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.get("seed", 0)))
        direction_init_matrix = torch.randn(
            patel_tau_matrix.shape,
            generator=generator,
            dtype=patel_tau_matrix.dtype,
        ).to(device)
    direction_init_matrix.fill_diagonal_(0.0)

    fixed_support_mask = None
    noise_guide_adj = None
    k_pairs = int(config.get("top_k_edges", config.get("noise_guide_pairs", 50)))
    fixed_support_mask_mode = str(config.get("fixed_support_mask_mode", "none"))
    directed_noise = bool(config.get("directed_noise", False))
    if directed_noise:
        noise_guide_adj = build_directed_noise_guide_adjacency(
            patel_kappa=torch.clamp(patel_kappa_matrix, min=0.0),
            patel_tau=patel_tau_matrix,
            top_k_pairs=k_pairs,
            direction_alpha=float(config.get("direction_alpha", 0.5)),
        )
    else:
        selection_mode = "maxgap" if fixed_support_mask_mode == "maxgap_kappa" else "topk"
        noise_guide_adj, adj_binary, _, _, _ = build_noise_guide_adjacency(
            patel_strength_matrix=torch.clamp(patel_kappa_matrix, min=0.0),
            top_k_pairs=k_pairs,
            selection_mode=selection_mode,
        )
        if fixed_support_mask_mode in {"topk_kappa", "maxgap_kappa"}:
            fixed_support_mask = adj_binary.clone().float()

    target_edge_count = int(config.get("selection_top_k", config.get("noise_guide_pairs", k_pairs)))
    target_density = compute_target_density(num_nodes, target_edge_count)
    adj_bias_init = float(np.log(target_density / (1.0 - target_density)))
    kappa_logit_bias_prior = torch.maximum(patel_kappa_matrix, patel_kappa_matrix.t()).clone()
    kappa_logit_bias_prior.fill_diagonal_(0.0)

    requested_emb_dim = int(config.get("requested_emb_dim", config.get("emb_dim", 0)))
    emb_dim = None if requested_emb_dim <= 0 else requested_emb_dim

    model = DDM(
        in_dim=time_points,
        num_hidden=int(config.get("num_hidden", 64)),
        num_layers=int(config.get("num_layers", 2)),
        nhead=4,
        activation="prelu",
        feat_drop=0.1,
        attn_drop=0.1,
        norm="layernorm",
        alpha_l=2,
        beta_schedule="linear",
        beta_1=0.0001,
        beta_T=0.02,
        T=1000,
        init_features=structure_init_matrix,
        noise_guide_adj=noise_guide_adj,
        kappa_logit_bias_prior=kappa_logit_bias_prior,
        direction_init_features=direction_init_matrix,
        fixed_support_mask=fixed_support_mask,
        adj_bias_init=adj_bias_init,
        init_logit_scale=float(config.get("structure_init_scale", 1.0)),
        emb_dim=emb_dim,
        structure_parameterization=str(config.get("structure_parameterization", "support_direction")),
        structure_message_graph_mode=str(config.get("structure_message_graph_mode", "raw")),
        adj_activation=str(config.get("adj_activation", "sigmoid")),
        kappa_logit_bias_scale=float(config.get("kappa_logit_bias_scale", 0.0)),
        use_temporal_encoder=not bool(config.get("disable_temporal_encoder", False)),
        uniform_timestep=not bool(config.get("per_node_timestep", False)),
        noise_norm_mode=str(config.get("noise_norm_mode", "global")),
        noise_zero_mean=not bool(config.get("noise_with_mean", False)),
        loss_type=str(config.get("loss_type", "denoise_hybrid")),
        cosine_weight=float(config.get("cosine_weight", 0.1)),
        mse_weight=float(config.get("mse_weight", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, data_3d.to(device)


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_probe_loss(
    model: DDM,
    x: torch.Tensor,
    t_value: int,
    noise_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_clean = model.prepare_clean_target(x)
    if model.structure_learning_mode:
        support_logits = model.get_structure_logits()
        support_weights = torch.sigmoid(support_logits)
        diag_mask = model.diag_mask.to(support_weights.device)
        support_weights = support_weights * diag_mask
        if getattr(model, "fixed_support_mask", None) is not None:
            support_weights = support_weights * model.fixed_support_mask.to(support_weights.device)

        direction_logits_raw = model.get_direction_logits()
        direction_gate = torch.sigmoid(
            direction_logits_raw - direction_logits_raw.transpose(0, 1)
        )
        adj_weights = support_weights * direction_gate
        adj_weights = adj_weights * diag_mask
        message_adj = (
            adj_weights.transpose(0, 1)
            if getattr(model, "structure_message_graph_mode", "raw") == "causal"
            else adj_weights
        )
        g = dgl.graph((model.full_g_src, model.full_g_dst), num_nodes=model.num_nodes).to(x_clean.device)
        edge_weight = message_adj.flatten()
    else:
        g, edge_weight = None, None
        direction_logits_raw = get_current_directional_logits(model, causal=False, detach=False)

    t = torch.full((x_clean.shape[0],), t_value, device=x_clean.device, dtype=torch.long)
    original_noise_guide = getattr(model, "noise_guide_adj", None)
    if noise_mode == "isotropic":
        model.noise_guide_adj = None
    try:
        x_t, time_embed, g = model.sample_q(t, x_clean, g)
        loss = model.node_denoising(x_clean, x_t, time_embed, g, edge_weight=edge_weight)
        return loss, direction_logits_raw
    finally:
        model.noise_guide_adj = original_noise_guide


def collect_probe_stats(
    model: DDM,
    probe_batch: torch.Tensor,
    gt_edges: Sequence[Tuple[int, int]],
    t_value: int,
    noise_mode: str,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    gt_edge_set = set(gt_edges)
    num_nodes = int(probe_batch.shape[1])
    signed_push_values: List[float] = []
    abs_margin_gt: List[float] = []
    abs_margin_non_gt: List[float] = []
    detail_rows: List[Dict[str, str]] = []

    for subj_idx in range(probe_batch.shape[0]):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed + 997 * subj_idx + 13 * t_value)
        loss, direction_logits_raw = compute_probe_loss(
            model=model,
            x=probe_batch[subj_idx],
            t_value=t_value,
            noise_mode=noise_mode,
        )
        grad_logits_raw = torch.autograd.grad(loss, direction_logits_raw, retain_graph=False)[0]
        grad_logits = grad_logits_raw.transpose(0, 1)

        for u, v in gt_edges:
            signed_push = float(grad_logits[v, u].item() - grad_logits[u, v].item())
            signed_push_values.append(signed_push)
            abs_margin_gt.append(float(abs(grad_logits[u, v].item() - grad_logits[v, u].item())))
            detail_rows.append(
                {
                    "subject_index": str(subj_idx),
                    "t_value": str(t_value),
                    "noise_mode": noise_mode,
                    "edge": f"{u + 1}->{v + 1}",
                    "signed_push": f"{signed_push:.6f}",
                    "pushes_correct_direction": str(int(signed_push > 0.0)),
                }
            )

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (i, j) in gt_edge_set or (j, i) in gt_edge_set:
                    continue
                abs_margin_non_gt.append(float(abs(grad_logits[i, j].item() - grad_logits[j, i].item())))

    signed_push_arr = np.asarray(signed_push_values, dtype=np.float64)
    abs_margin_gt_arr = np.asarray(abs_margin_gt, dtype=np.float64)
    abs_margin_non_gt_arr = np.asarray(abs_margin_non_gt, dtype=np.float64)
    ratio = (
        float(abs_margin_gt_arr.mean() / (abs_margin_non_gt_arr.mean() + 1e-8))
        if abs_margin_non_gt_arr.size > 0 else 0.0
    )
    summary = {
        "gt_signed_push_mean": float(signed_push_arr.mean()),
        "gt_signed_push_median": float(np.median(signed_push_arr)),
        "gt_signed_push_p10": float(np.percentile(signed_push_arr, 10)),
        "gt_signed_push_p90": float(np.percentile(signed_push_arr, 90)),
        "gt_push_correct_frac": float((signed_push_arr > 0.0).mean()),
        "gt_abs_grad_margin_mean": float(abs_margin_gt_arr.mean()),
        "non_gt_abs_grad_margin_mean": float(abs_margin_non_gt_arr.mean()) if abs_margin_non_gt_arr.size > 0 else 0.0,
        "gt_to_non_gt_abs_grad_ratio": ratio,
    }
    return summary, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0B timestep-stratified direction-gradient probe.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--timesteps", type=str, default="50,300,800")
    parser.add_argument("--noise_modes", type=str, default="isotropic,patel")
    parser.add_argument("--num_probe_subjects", type=int, default=8)
    parser.add_argument("--subject_offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_tag", type=str, default="model")
    parser.add_argument("--tag", type=str, default="phase0b")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    gt_path = Path(args.gt_path).resolve()
    timesteps = parse_int_list(args.timesteps)
    noise_modes = [part.strip() for part in args.noise_modes.split(",") if part.strip()]
    if not timesteps:
        raise ValueError("--timesteps must include at least one timestep")
    if not noise_modes:
        raise ValueError("--noise_modes must include at least one mode")

    model, config, data_3d = build_model_from_run_dir(run_dir, device=args.device)
    gt_edges = load_gt_edges(gt_path)

    start = int(args.subject_offset)
    stop = min(data_3d.shape[0], start + int(args.num_probe_subjects))
    if start >= stop:
        raise ValueError("No probe subjects selected")
    probe_batch = data_3d[start:stop]

    print(f"Loaded model from {run_dir}")
    print(f"Model tag: {args.model_tag}")
    print(f"Probe subjects: {start}:{stop} (count={probe_batch.shape[0]})")
    print(f"Timesteps: {timesteps}")
    print(f"Noise modes: {noise_modes}")

    summary_rows: List[Dict[str, str]] = []
    detail_rows: List[Dict[str, str]] = []

    for noise_mode in noise_modes:
        for t_value in timesteps:
            stats, details = collect_probe_stats(
                model=model,
                probe_batch=probe_batch,
                gt_edges=gt_edges,
                t_value=t_value,
                noise_mode=noise_mode,
                seed=int(config.get("seed", 0)),
            )
            row = {
                "run_dir": str(run_dir),
                "model_tag": args.model_tag,
                "noise_mode": noise_mode,
                "t_value": str(t_value),
                "num_probe_subjects": str(probe_batch.shape[0]),
                "num_gt_edges": str(len(gt_edges)),
            }
            row.update({k: f"{v:.6f}" for k, v in stats.items()})
            summary_rows.append(row)
            print(
                f"[{args.model_tag}][{noise_mode}][t={t_value}] "
                f"push_frac={row['gt_push_correct_frac']} "
                f"mean_push={row['gt_signed_push_mean']} "
                f"gt/non_gt={row['gt_to_non_gt_abs_grad_ratio']}"
            )
            for detail in details:
                detail_rows.append({
                    "run_dir": str(run_dir),
                    "model_tag": args.model_tag,
                    **detail,
                })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir.parent
    stem = f"phase0b_gradient_probe_{args.model_tag}_{timestamp}_{args.tag}"
    summary_path = output_dir / f"{stem}_summary.csv"
    details_path = output_dir / f"{stem}_details.csv"
    write_csv(summary_path, summary_rows)
    write_csv(details_path, detail_rows)
    print(f"SUMMARY_CSV {summary_path}")
    print(f"DETAILS_CSV {details_path}")


if __name__ == "__main__":
    main()
