import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from phase0b_timestep_gradient_probe import (
    build_model_from_run_dir,
    load_gt_edges,
    parse_int_list,
    write_csv,
)


def build_probe_graph(model, device: torch.device) -> dgl.DGLGraph:
    if not getattr(model, "structure_learning_mode", False):
        raise ValueError("Phase 0C requires structure learning mode to be enabled.")
    return dgl.graph((model.full_g_src, model.full_g_dst), num_nodes=model.num_nodes).to(device)


def raw_to_message_adj(model, raw_adj: torch.Tensor) -> torch.Tensor:
    if getattr(model, "structure_message_graph_mode", "raw") == "causal":
        return raw_adj.transpose(0, 1)
    return raw_adj


def sample_noisy_input(
    model,
    x_clean: torch.Tensor,
    g: dgl.DGLGraph,
    t_value: int,
    noise_mode: str,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.full((x_clean.shape[0],), t_value, device=x_clean.device, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    eps = torch.randn((1, *x_clean.shape), generator=generator, dtype=x_clean.dtype).to(x_clean.device)

    original_noise_guide = getattr(model, "noise_guide_adj", None)
    if noise_mode == "isotropic":
        model.noise_guide_adj = None
    try:
        x_t, time_embed, _ = model.sample_q(t, x_clean, g, eps_override=eps)
    finally:
        model.noise_guide_adj = original_noise_guide
    return x_t, time_embed


def compute_node_losses(
    model,
    g: dgl.DGLGraph,
    x_clean: torch.Tensor,
    x_t: torch.Tensor,
    time_embed: torch.Tensor,
    message_adj: torch.Tensor,
) -> torch.Tensor:
    out, _ = model.net(g, x_t=x_t, time_embed=time_embed, edge_weight=message_adj.flatten())
    return F.smooth_l1_loss(out, x_clean, reduction="none").mean(dim=-1)


def collect_ablation_stats(
    model,
    probe_batch: torch.Tensor,
    gt_edges: Sequence[Tuple[int, int]],
    t_value: int,
    noise_mode: str,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    raw_adj_base = model.get_structure_adj().detach()
    probe_graph = build_probe_graph(model, probe_batch.device)
    full_message_adj = raw_to_message_adj(model, raw_adj_base)
    directed_pairs = sorted(set(gt_edges) | {(dst, src) for src, dst in gt_edges})

    margins: List[float] = []
    correct_importances: List[float] = []
    reverse_importances: List[float] = []
    detail_rows: List[Dict[str, str]] = []

    with torch.no_grad():
        for subj_idx in range(probe_batch.shape[0]):
            x = probe_batch[subj_idx]
            x_clean = model.prepare_clean_target(x)
            x_t, time_embed = sample_noisy_input(
                model=model,
                x_clean=x_clean,
                g=probe_graph,
                t_value=t_value,
                noise_mode=noise_mode,
                seed=seed + 997 * subj_idx + 13 * t_value,
            )
            full_node_losses = compute_node_losses(
                model=model,
                g=probe_graph,
                x_clean=x_clean,
                x_t=x_t,
                time_embed=time_embed,
                message_adj=full_message_adj,
            )

            importance_map: Dict[Tuple[int, int], float] = {}
            for src, dst in directed_pairs:
                masked_raw_adj = raw_adj_base.clone()
                masked_raw_adj[dst, src] = 0.0
                masked_message_adj = raw_to_message_adj(model, masked_raw_adj)
                masked_node_losses = compute_node_losses(
                    model=model,
                    g=probe_graph,
                    x_clean=x_clean,
                    x_t=x_t,
                    time_embed=time_embed,
                    message_adj=masked_message_adj,
                )
                importance_map[(src, dst)] = float(
                    masked_node_losses[dst].item() - full_node_losses[dst].item()
                )

            for src, dst in gt_edges:
                importance_correct = importance_map[(src, dst)]
                importance_reverse = importance_map[(dst, src)]
                margin = importance_correct - importance_reverse
                margins.append(margin)
                correct_importances.append(importance_correct)
                reverse_importances.append(importance_reverse)
                detail_rows.append(
                    {
                        "subject_index": str(subj_idx),
                        "t_value": str(t_value),
                        "noise_mode": noise_mode,
                        "edge": f"{src + 1}->{dst + 1}",
                        "importance_correct": f"{importance_correct:.6f}",
                        "importance_reverse": f"{importance_reverse:.6f}",
                        "margin": f"{margin:.6f}",
                        "prefers_correct_direction": str(int(margin > 0.0)),
                        "raw_weight_correct": f"{float(raw_adj_base[dst, src].item()):.6f}",
                        "raw_weight_reverse": f"{float(raw_adj_base[src, dst].item()):.6f}",
                    }
                )

    margin_arr = np.asarray(margins, dtype=np.float64)
    correct_arr = np.asarray(correct_importances, dtype=np.float64)
    reverse_arr = np.asarray(reverse_importances, dtype=np.float64)
    summary = {
        "direction_accuracy": float((margin_arr > 0.0).mean()),
        "margin_mean": float(margin_arr.mean()),
        "margin_median": float(np.median(margin_arr)),
        "margin_p10": float(np.percentile(margin_arr, 10)),
        "margin_p90": float(np.percentile(margin_arr, 90)),
        "importance_correct_mean": float(correct_arr.mean()),
        "importance_reverse_mean": float(reverse_arr.mean()),
        "importance_correct_median": float(np.median(correct_arr)),
        "importance_reverse_median": float(np.median(reverse_arr)),
    }
    return summary, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0C edge-ablation denoise direction probe.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--timesteps", type=str, default="50,800")
    parser.add_argument("--noise_modes", type=str, default="isotropic")
    parser.add_argument("--num_probe_subjects", type=int, default=8)
    parser.add_argument("--subject_offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_tag", type=str, default="model")
    parser.add_argument("--tag", type=str, default="phase0c")
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
            stats, details = collect_ablation_stats(
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
            row.update({key: f"{value:.6f}" for key, value in stats.items()})
            summary_rows.append(row)
            print(
                f"[{args.model_tag}][{noise_mode}][t={t_value}] "
                f"acc={row['direction_accuracy']} "
                f"margin_mean={row['margin_mean']} "
                f"margin_median={row['margin_median']}"
            )
            for detail in details:
                detail_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "model_tag": args.model_tag,
                        **detail,
                    }
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir.parent
    stem = f"phase0c_edge_ablation_probe_{args.model_tag}_{timestamp}_{args.tag}"
    summary_path = output_dir / f"{stem}_summary.csv"
    details_path = output_dir / f"{stem}_details.csv"
    write_csv(summary_path, summary_rows)
    write_csv(details_path, detail_rows)
    print(f"SUMMARY_CSV {summary_path}")
    print(f"DETAILS_CSV {details_path}")


if __name__ == "__main__":
    main()
