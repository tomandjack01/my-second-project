import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from models.DDM import NodeSpecificTemporalEncoder, get_beta_schedule


TIME_POINTS_PER_SUBJECT = 200


@dataclass
class PairwiseResult:
    edge_u: int
    edge_v: int
    correct_mse: float
    reverse_mse: float
    gap: float
    correct: int


def parse_int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_time_series(csv_path: Path, time_points: int) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    data = df.values.astype(np.float32)
    total_rows, num_nodes = data.shape
    if total_rows % time_points != 0:
        raise ValueError(
            f"Total rows {total_rows} not divisible by time_points {time_points}"
        )
    num_subjects = total_rows // time_points
    data_3d = data.reshape(num_subjects, time_points, num_nodes)
    return np.transpose(data_3d, (0, 2, 1))


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


def fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge_lambda: float,
) -> np.ndarray:
    feat_mean = x_train.mean(axis=0, keepdims=True)
    feat_std = x_train.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    x_train_z = (x_train - feat_mean) / feat_std
    x_test_z = (x_test - feat_mean) / feat_std

    y_mean = y_train.mean()
    y_centered = y_train - y_mean

    xtx = x_train_z.T @ x_train_z
    reg = ridge_lambda * np.eye(xtx.shape[0], dtype=np.float64)
    beta = np.linalg.solve(xtx + reg, x_train_z.T @ y_centered)
    return x_test_z @ beta + y_mean


def build_single_source_lag_dataset(
    source_series: np.ndarray,
    target_series: np.ndarray,
    lags: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if source_series.shape != target_series.shape:
        raise ValueError("source and target must share shape [subjects, time]")
    max_lag = max(lags)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for subj in range(source_series.shape[0]):
        src = source_series[subj]
        tgt = target_series[subj]
        feat_cols = [src[max_lag - lag : src.shape[0] - lag] for lag in lags]
        x_sub = np.stack(feat_cols, axis=1)
        y_sub = tgt[max_lag:]
        xs.append(x_sub)
        ys.append(y_sub)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def compute_pair_direction_result(
    data: np.ndarray,
    edge: Tuple[int, int],
    lags: Sequence[int],
    train_subjects: int,
    ridge_lambda: float,
) -> PairwiseResult:
    u, v = edge
    src_correct_train = data[:train_subjects, u, :]
    tgt_correct_train = data[:train_subjects, v, :]
    src_correct_test = data[train_subjects:, u, :]
    tgt_correct_test = data[train_subjects:, v, :]

    x_train, y_train = build_single_source_lag_dataset(
        src_correct_train, tgt_correct_train, lags,
    )
    x_test, y_test = build_single_source_lag_dataset(
        src_correct_test, tgt_correct_test, lags,
    )
    pred_correct = fit_ridge_predict(x_train, y_train, x_test, ridge_lambda)
    correct_mse = float(np.mean((pred_correct - y_test) ** 2))

    src_reverse_train = data[:train_subjects, v, :]
    tgt_reverse_train = data[:train_subjects, u, :]
    src_reverse_test = data[train_subjects:, v, :]
    tgt_reverse_test = data[train_subjects:, u, :]

    x_train_rev, y_train_rev = build_single_source_lag_dataset(
        src_reverse_train, tgt_reverse_train, lags,
    )
    x_test_rev, y_test_rev = build_single_source_lag_dataset(
        src_reverse_test, tgt_reverse_test, lags,
    )
    pred_reverse = fit_ridge_predict(x_train_rev, y_train_rev, x_test_rev, ridge_lambda)
    reverse_mse = float(np.mean((pred_reverse - y_test_rev) ** 2))

    gap = reverse_mse - correct_mse
    return PairwiseResult(
        edge_u=u + 1,
        edge_v=v + 1,
        correct_mse=correct_mse,
        reverse_mse=reverse_mse,
        gap=gap,
        correct=int(gap > 0.0),
    )


def summarise_pairwise_results(
    condition: str,
    rows: Sequence[PairwiseResult],
    extra: Dict[str, str],
) -> Dict[str, str]:
    gaps = np.array([row.gap for row in rows], dtype=np.float64)
    acc = np.array([row.correct for row in rows], dtype=np.float64)
    correct_mse = np.array([row.correct_mse for row in rows], dtype=np.float64)
    reverse_mse = np.array([row.reverse_mse for row in rows], dtype=np.float64)
    summary = {
        "condition": condition,
        "num_edges": str(len(rows)),
        "direction_accuracy": f"{acc.mean():.6f}",
        "mean_gap_reverse_minus_correct": f"{gaps.mean():.6f}",
        "median_gap_reverse_minus_correct": f"{np.median(gaps):.6f}",
        "p10_gap_reverse_minus_correct": f"{np.percentile(gaps, 10):.6f}",
        "p90_gap_reverse_minus_correct": f"{np.percentile(gaps, 90):.6f}",
        "gap_positive_frac": f"{(gaps > 0.0).mean():.6f}",
        "mean_correct_mse": f"{correct_mse.mean():.6f}",
        "mean_reverse_mse": f"{reverse_mse.mean():.6f}",
    }
    summary.update(extra)
    return summary


def build_encoder_features(
    data: np.ndarray,
    checkpoint_path: Path,
    device: str,
) -> np.ndarray:
    num_subjects, _, time_points = data.shape
    encoder = NodeSpecificTemporalEncoder(
        time_points=time_points,
        hidden_channels=32,
        output_dim=time_points,
    )
    state = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state)
    encoder.to(device)
    encoder.eval()

    outputs = []
    with torch.no_grad():
        for subj in range(num_subjects):
            x = torch.from_numpy(data[subj]).to(device)
            _, unnormalized = encoder(x, return_unnormalized=True)
            outputs.append(unnormalized.detach().cpu().numpy().astype(np.float32))
    return np.stack(outputs, axis=0)


def build_noisy_variant(
    clean: np.ndarray,
    t: int,
    noise_repeat: int,
    alpha_bars: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha_bar_t = float(alpha_bars[t])
    eps = rng.standard_normal(size=clean.shape).astype(np.float32)
    return (
        np.sqrt(alpha_bar_t, dtype=np.float32) * clean +
        np.sqrt(1.0 - alpha_bar_t, dtype=np.float32) * eps
    ).astype(np.float32)


def evaluate_condition(
    data: np.ndarray,
    gt_edges: Sequence[Tuple[int, int]],
    lags: Sequence[int],
    train_subjects: int,
    ridge_lambda: float,
    noise_repeats: int,
    condition: str,
    alpha_bars: np.ndarray,
    t_value: int,
) -> Tuple[List[PairwiseResult], Dict[str, str]]:
    if t_value < 0:
        rows = [
            compute_pair_direction_result(
                data=data,
                edge=edge,
                lags=lags,
                train_subjects=train_subjects,
                ridge_lambda=ridge_lambda,
            )
            for edge in gt_edges
        ]
        extra = {
            "lags": ",".join(str(v) for v in lags),
            "train_subjects": str(train_subjects),
            "test_subjects": str(data.shape[0] - train_subjects),
            "t_value": "",
            "noise_repeats": "1",
        }
        return rows, summarise_pairwise_results(condition, rows, extra)

    aggregated: Dict[Tuple[int, int], List[PairwiseResult]] = {}
    for rep in range(noise_repeats):
        noisy = build_noisy_variant(
            clean=data,
            t=t_value,
            noise_repeat=rep,
            alpha_bars=alpha_bars,
            seed=1000 + 97 * rep + t_value,
        )
        rep_rows = [
            compute_pair_direction_result(
                data=noisy,
                edge=edge,
                lags=lags,
                train_subjects=train_subjects,
                ridge_lambda=ridge_lambda,
            )
            for edge in gt_edges
        ]
        for row in rep_rows:
            aggregated.setdefault((row.edge_u, row.edge_v), []).append(row)

    rows = []
    for key in sorted(aggregated.keys()):
        rep_rows = aggregated[key]
        mean_correct_mse = float(np.mean([r.correct_mse for r in rep_rows]))
        mean_reverse_mse = float(np.mean([r.reverse_mse for r in rep_rows]))
        gap = mean_reverse_mse - mean_correct_mse
        rows.append(
            PairwiseResult(
                edge_u=key[0],
                edge_v=key[1],
                correct_mse=mean_correct_mse,
                reverse_mse=mean_reverse_mse,
                gap=gap,
                correct=int(gap > 0.0),
            )
        )
    extra = {
        "lags": ",".join(str(v) for v in lags),
        "train_subjects": str(train_subjects),
        "test_subjects": str(data.shape[0] - train_subjects),
        "t_value": str(t_value),
        "noise_repeats": str(noise_repeats),
    }
    return rows, summarise_pairwise_results(condition, rows, extra)


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline multi-lag direction diagnostic for the Option B go/no-go test.",
    )
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--pretrain_checkpoint", type=str, required=True)
    parser.add_argument("--lags", type=str, default="1,2,3")
    parser.add_argument("--time_points", type=int, default=TIME_POINTS_PER_SUBJECT)
    parser.add_argument("--train_subject_ratio", type=float, default=0.7)
    parser.add_argument("--ridge_lambda", type=float, default=1.0)
    parser.add_argument("--low_t", type=int, default=50)
    parser.add_argument("--mid_t", type=int, default=300)
    parser.add_argument("--noise_repeats", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tag", type=str, default="phase0_multilag")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    gt_path = Path(args.gt_path).resolve()
    pretrain_checkpoint = Path(args.pretrain_checkpoint).resolve()
    lags = parse_int_list(args.lags)
    if not lags:
        raise ValueError("--lags must include at least one lag")

    data = load_time_series(csv_path, time_points=args.time_points)
    gt_edges = load_gt_edges(gt_path)
    num_subjects = data.shape[0]
    train_subjects = max(1, min(num_subjects - 1, int(round(num_subjects * args.train_subject_ratio))))

    print(f"Loaded {csv_path.name}: subjects={num_subjects}, nodes={data.shape[1]}, time={data.shape[2]}")
    print(f"GT edges: {len(gt_edges)}")
    print(f"Lags: {lags}")
    print(f"Train/Test subjects: {train_subjects}/{num_subjects - train_subjects}")

    encoded = build_encoder_features(
        data=data,
        checkpoint_path=pretrain_checkpoint,
        device=args.device,
    )
    print(f"Encoded representation built from {pretrain_checkpoint.name}")

    betas = get_beta_schedule("linear", 0.0001, 0.02, 1000).cpu().numpy()
    alpha_bars = np.cumprod(1.0 - betas, axis=0)

    summaries: List[Dict[str, str]] = []
    detail_rows: List[Dict[str, str]] = []

    conditions = [
        ("raw_clean", data, -1),
        ("encoder_clean", encoded, -1),
        (f"raw_noisy_t{args.low_t}", data, args.low_t),
        (f"raw_noisy_t{args.mid_t}", data, args.mid_t),
        (f"encoder_noisy_t{args.low_t}", encoded, args.low_t),
        (f"encoder_noisy_t{args.mid_t}", encoded, args.mid_t),
    ]

    for condition_name, source_data, t_value in conditions:
        rows, summary = evaluate_condition(
            data=source_data,
            gt_edges=gt_edges,
            lags=lags,
            train_subjects=train_subjects,
            ridge_lambda=args.ridge_lambda,
            noise_repeats=args.noise_repeats,
            condition=condition_name,
            alpha_bars=alpha_bars,
            t_value=t_value,
        )
        summaries.append(summary)
        print(
            f"[{condition_name}] acc={summary['direction_accuracy']} "
            f"mean_gap={summary['mean_gap_reverse_minus_correct']} "
            f"median_gap={summary['median_gap_reverse_minus_correct']}"
        )
        for row in rows:
            detail_rows.append({
                "condition": condition_name,
                "edge": f"{row.edge_u}->{row.edge_v}",
                "correct_mse": f"{row.correct_mse:.6f}",
                "reverse_mse": f"{row.reverse_mse:.6f}",
                "gap_reverse_minus_correct": f"{row.gap:.6f}",
                "correct": str(row.correct),
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_stem = (
        f"multilag_direction_diagnostic_{csv_path.stem}_{timestamp}_{args.tag}"
    )
    output_dir = csv_path.parent.parent / "GraphExp" / "results"
    summary_path = output_dir / f"{output_stem}_summary.csv"
    details_path = output_dir / f"{output_stem}_details.csv"
    write_csv(summary_path, summaries)
    write_csv(details_path, detail_rows)
    print(f"SUMMARY_CSV {summary_path}")
    print(f"DETAILS_CSV {details_path}")


if __name__ == "__main__":
    main()
