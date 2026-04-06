#!/usr/bin/env python3
"""Offline audit for support-selection rule sensitivity across support priors."""

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from main_structure_learning import (  # noqa: E402
    TIME_POINTS_PER_SUBJECT,
    build_support_prior_matrix,
    build_undirected_kappa_skeleton,
    compute_global_pearson,
    load_fmri_data,
)
from utils.patel_util import compute_patel_components  # noqa: E402


DEFAULT_DATASET_SPECS: Sequence[Tuple[str, Path, int]] = (
    ("fMRI", REPO_ROOT / "fMRI_dataset" / "fMRI.csv", 5),
    ("sim2", REPO_ROOT / "fMRI_dataset" / "sim2.csv", 11),
    ("sim3", REPO_ROOT / "fMRI_dataset" / "sim3.csv", 18),
    ("sim4", REPO_ROOT / "fMRI_dataset" / "sim4.csv", 61),
)

TOPK_FRACTIONS: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50)
QUANTILES: Sequence[float] = (0.50, 0.70, 0.80, 0.90, 0.95)


@dataclass
class RuleSelectionResult:
    rule_name: str
    rule_family: str
    rule_param: str
    adj_binary: torch.Tensor
    selected_pairs: int
    threshold: float
    selection_gap: float


def dataset_specs_from_args(args: argparse.Namespace) -> List[Tuple[str, Path, int]]:
    if not args.csv_paths:
        return list(DEFAULT_DATASET_SPECS)

    csv_paths = [Path(path).resolve() for path in args.csv_paths]
    if args.top_k_pairs:
        if len(args.top_k_pairs) != len(csv_paths):
            raise ValueError("--top_k_pairs must match --csv_paths length")
        top_k_pairs = list(args.top_k_pairs)
    else:
        default_map = {name.lower(): top_k for name, _, top_k in DEFAULT_DATASET_SPECS}
        top_k_pairs = []
        for path in csv_paths:
            stem = path.stem.lower()
            if stem not in default_map:
                raise ValueError(
                    f"No default top-k pair count for dataset '{path.stem}'. "
                    "Pass --top_k_pairs explicitly."
                )
            top_k_pairs.append(default_map[stem])

    specs: List[Tuple[str, Path, int]] = []
    for path, top_k in zip(csv_paths, top_k_pairs):
        specs.append((path.stem, path, int(top_k)))
    return specs


def pair_strength_and_indices(
    prior_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = prior_matrix.shape[0]
    pair_strength = torch.maximum(prior_matrix, prior_matrix.t()).clone()
    pair_strength = torch.clamp(pair_strength, min=0.0)
    pair_strength.fill_diagonal_(0.0)
    triu_i, triu_j = torch.triu_indices(num_nodes, num_nodes, offset=1)
    flat_strength = pair_strength[triu_i, triu_j]
    return pair_strength, triu_i, triu_j, flat_strength


def flat_pair_order(prior_matrix: torch.Tensor) -> np.ndarray:
    _, _, _, flat_strength = pair_strength_and_indices(prior_matrix)
    flat_np = flat_strength.cpu().numpy()
    pair_index = np.arange(flat_np.shape[0], dtype=np.int64)
    order = np.lexsort((pair_index, -flat_np))
    return order.astype(np.int64)


def inversion_count(values: Sequence[int]) -> int:
    values_list = list(values)

    def sort_count(seq: List[int]) -> Tuple[List[int], int]:
        if len(seq) <= 1:
            return seq, 0
        mid = len(seq) // 2
        left, left_inv = sort_count(seq[:mid])
        right, right_inv = sort_count(seq[mid:])
        merged: List[int] = []
        inv = left_inv + right_inv
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv += len(left) - i
                j += 1
        if i < len(left):
            merged.extend(left[i:])
        if j < len(right):
            merged.extend(right[j:])
        return merged, inv

    _, inv = sort_count(values_list)
    return int(inv)


def kendall_tau_tiebroken(order_a: np.ndarray, order_b: np.ndarray) -> float:
    num_items = int(order_a.shape[0])
    if num_items < 2:
        return 1.0

    ranks_b = np.empty(num_items, dtype=np.int64)
    ranks_b[order_b] = np.arange(num_items, dtype=np.int64)
    perm = ranks_b[order_a]
    discordant = inversion_count(perm.tolist())
    total_pairs = num_items * (num_items - 1) // 2
    return 1.0 - (2.0 * float(discordant) / float(total_pairs))


def build_pair_set(adj_binary: torch.Tensor) -> set:
    triu_i, triu_j = torch.triu_indices(adj_binary.shape[0], adj_binary.shape[1], offset=1)
    selected = torch.nonzero(adj_binary[triu_i, triu_j] > 0, as_tuple=False).flatten()
    return {
        (int(triu_i[idx].item()), int(triu_j[idx].item()))
        for idx in selected
    }


def select_by_quantile(prior_matrix: torch.Tensor, quantile: float) -> RuleSelectionResult:
    pair_strength, triu_i, triu_j, flat_strength = pair_strength_and_indices(prior_matrix)
    positive_idx = torch.nonzero(flat_strength > 0, as_tuple=False).flatten()
    adj_binary = torch.zeros_like(pair_strength)
    if positive_idx.numel() == 0:
        return RuleSelectionResult(
            rule_name=f"quantile_{quantile:.2f}",
            rule_family="quantile",
            rule_param=f"{quantile:.2f}",
            adj_binary=adj_binary,
            selected_pairs=0,
            threshold=0.0,
            selection_gap=0.0,
        )

    positive_strength = flat_strength[positive_idx]
    threshold = float(torch.quantile(positive_strength, float(quantile)).item())
    selected_idx = positive_idx[positive_strength >= threshold]
    if selected_idx.numel() > 0:
        src = triu_i[selected_idx]
        dst = triu_j[selected_idx]
        adj_binary[src, dst] = 1.0
        adj_binary[dst, src] = 1.0

    return RuleSelectionResult(
        rule_name=f"quantile_{quantile:.2f}",
        rule_family="quantile",
        rule_param=f"{quantile:.2f}",
        adj_binary=adj_binary,
        selected_pairs=int(selected_idx.numel()),
        threshold=threshold,
        selection_gap=0.0,
    )


def select_by_topk(prior_matrix: torch.Tensor, top_k_pairs: int, label: str) -> RuleSelectionResult:
    adj_binary, selected_pairs, threshold, selection_gap = build_undirected_kappa_skeleton(
        prior_matrix,
        top_k_pairs=top_k_pairs,
        selection_mode="topk",
    )
    return RuleSelectionResult(
        rule_name=label,
        rule_family="topk",
        rule_param=str(int(top_k_pairs)),
        adj_binary=adj_binary,
        selected_pairs=int(selected_pairs),
        threshold=float(threshold),
        selection_gap=float(selection_gap),
    )


def select_by_maxgap(prior_matrix: torch.Tensor) -> RuleSelectionResult:
    adj_binary, selected_pairs, threshold, selection_gap = build_undirected_kappa_skeleton(
        prior_matrix,
        top_k_pairs=0,
        selection_mode="maxgap",
    )
    return RuleSelectionResult(
        rule_name="maxgap",
        rule_family="maxgap",
        rule_param="auto",
        adj_binary=adj_binary,
        selected_pairs=int(selected_pairs),
        threshold=float(threshold),
        selection_gap=float(selection_gap),
    )


def build_rule_grid(total_pairs: int, reference_top_k: int) -> List[Tuple[str, str, float]]:
    rules: List[Tuple[str, str, float]] = [("maxgap", "maxgap", 0.0), ("topk_current", "topk", float(reference_top_k))]

    seen_topk = {int(reference_top_k)}
    for frac in TOPK_FRACTIONS:
        k_pairs = max(1, min(total_pairs, int(round(total_pairs * frac))))
        if k_pairs in seen_topk:
            continue
        seen_topk.add(k_pairs)
        rules.append((f"topk_frac_{frac:.2f}", "topk", float(k_pairs)))

    for quantile in QUANTILES:
        rules.append((f"quantile_{quantile:.2f}", "quantile", float(quantile)))
    return rules


def run_rule(prior_matrix: torch.Tensor, rule_family: str, rule_name: str, rule_value: float) -> RuleSelectionResult:
    if rule_family == "maxgap":
        return select_by_maxgap(prior_matrix)
    if rule_family == "topk":
        return select_by_topk(prior_matrix, top_k_pairs=int(rule_value), label=rule_name)
    if rule_family == "quantile":
        return select_by_quantile(prior_matrix, quantile=float(rule_value))
    raise ValueError(f"Unsupported rule family: {rule_family}")


def first_mismatch_rank(order_a: np.ndarray, order_b: np.ndarray) -> int:
    mismatch = np.nonzero(order_a != order_b)[0]
    if mismatch.size == 0:
        return 0
    return int(mismatch[0] + 1)


def scan_dataset(
    dataset_name: str,
    csv_path: Path,
    reference_top_k: int,
    subject_limit: int,
    time_limit: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"CSV path: {csv_path}")
    print(f"Reference top-k pairs: {reference_top_k}")

    _, data_2d, effective_subjects, num_nodes = load_fmri_data(
        csv_path=str(csv_path),
        time_points_per_subject=TIME_POINTS_PER_SUBJECT,
        subject_limit=subject_limit,
        time_limit=time_limit,
    )
    pearson_matrix = compute_global_pearson(data_2d)
    _, patel_kappa_np, _ = compute_patel_components(data_2d.numpy())
    patel_kappa_matrix = torch.from_numpy(patel_kappa_np).float()

    patel_prior = build_support_prior_matrix(
        mode="patel_kappa",
        patel_kappa_matrix=patel_kappa_matrix,
        pearson_matrix=pearson_matrix,
    )
    pearson_prior = build_support_prior_matrix(
        mode="pearson_abs",
        patel_kappa_matrix=patel_kappa_matrix,
        pearson_matrix=pearson_matrix,
    )

    order_patel = flat_pair_order(patel_prior)
    order_pearson = flat_pair_order(pearson_prior)
    total_pairs = int(order_patel.shape[0])
    mismatch_rank = first_mismatch_rank(order_patel, order_pearson)
    tau = kendall_tau_tiebroken(order_patel, order_pearson)

    dataset_summary = {
        "dataset": dataset_name,
        "csv_path": str(csv_path),
        "num_nodes": int(num_nodes),
        "num_subjects": int(effective_subjects),
        "total_pairs": total_pairs,
        "reference_top_k": int(reference_top_k),
        "same_sorted_pair_order": int(np.array_equal(order_patel, order_pearson)),
        "first_order_mismatch_rank": int(mismatch_rank),
        "kendall_tau_tiebroken": float(tau),
    }

    rule_rows: List[Dict[str, float]] = []
    for rule_name, rule_family, rule_value in build_rule_grid(total_pairs, reference_top_k):
        patel_result = run_rule(patel_prior, rule_family=rule_family, rule_name=rule_name, rule_value=rule_value)
        pearson_result = run_rule(pearson_prior, rule_family=rule_family, rule_name=rule_name, rule_value=rule_value)

        patel_set = build_pair_set(patel_result.adj_binary)
        pearson_set = build_pair_set(pearson_result.adj_binary)
        overlap = len(patel_set & pearson_set)
        union = len(patel_set | pearson_set)
        diff = len(patel_set ^ pearson_set)
        jaccard = 1.0 if union == 0 else float(overlap) / float(union)

        row = {
            "dataset": dataset_name,
            "rule_name": rule_name,
            "rule_family": rule_family,
            "rule_param": patel_result.rule_param,
            "reference_top_k": int(reference_top_k),
            "num_nodes": int(num_nodes),
            "total_pairs": total_pairs,
            "patel_pairs": int(patel_result.selected_pairs),
            "pearson_pairs": int(pearson_result.selected_pairs),
            "same_skeleton": int(patel_set == pearson_set),
            "jaccard": float(jaccard),
            "overlap_pair_count": int(overlap),
            "union_pair_count": int(union),
            "different_pair_count": int(diff),
            "patel_only_pair_count": int(len(patel_set - pearson_set)),
            "pearson_only_pair_count": int(len(pearson_set - patel_set)),
            "patel_threshold": float(patel_result.threshold),
            "pearson_threshold": float(pearson_result.threshold),
            "patel_gap": float(patel_result.selection_gap),
            "pearson_gap": float(pearson_result.selection_gap),
            "same_sorted_pair_order": int(dataset_summary["same_sorted_pair_order"]),
            "first_order_mismatch_rank": int(dataset_summary["first_order_mismatch_rank"]),
            "kendall_tau_tiebroken": float(dataset_summary["kendall_tau_tiebroken"]),
        }
        rule_rows.append(row)

    return dataset_summary, rule_rows


def write_csv(path: Path, rows: Iterable[Dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(dataset_summaries: Sequence[Dict[str, float]], rule_rows: Sequence[Dict[str, float]]) -> None:
    print("\n=== Rank Diagnostics ===")
    for summary in dataset_summaries:
        dataset = summary["dataset"]
        same_order = bool(summary["same_sorted_pair_order"])
        mismatch_rank = int(summary["first_order_mismatch_rank"])
        tau = float(summary["kendall_tau_tiebroken"])
        if same_order:
            print(f"- {dataset}: identical pair ordering across Patel and Pearson (tau={tau:.4f})")
        else:
            print(
                f"- {dataset}: ordering diverges at rank {mismatch_rank} "
                f"(tau={tau:.4f})"
            )

    print("\n=== Rule Trigger Summary ===")
    for dataset in [summary["dataset"] for summary in dataset_summaries]:
        dataset_rows = [row for row in rule_rows if row["dataset"] == dataset]
        differing = [row for row in dataset_rows if int(row["same_skeleton"]) == 0]
        if not differing:
            print(f"- {dataset}: no scanned rule produced different Patel/Pearson skeletons")
            continue
        first_diff = differing[0]
        print(
            f"- {dataset}: first differing rule = {first_diff['rule_name']} "
            f"(Jaccard={float(first_diff['jaccard']):.4f}, "
            f"Patel pairs={int(first_diff['patel_pairs'])}, "
            f"Pearson pairs={int(first_diff['pearson_pairs'])})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan support selection rules across Patel and Pearson priors.")
    parser.add_argument(
        "--csv_paths",
        nargs="*",
        default=None,
        help="Optional CSV paths to scan. Defaults to fMRI/sim2/sim3/sim4.",
    )
    parser.add_argument(
        "--top_k_pairs",
        nargs="*",
        type=int,
        default=None,
        help="Optional top-k pair counts aligned with --csv_paths.",
    )
    parser.add_argument(
        "--subject_limit",
        type=int,
        default=-1,
        help="Optional subject limit forwarded to load_fmri_data.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=-1,
        help="Optional time limit forwarded to load_fmri_data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "GraphExp" / "results"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="support_selection_rule_scan",
        help="Filename prefix for generated CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_specs = dataset_specs_from_args(args)

    dataset_summaries: List[Dict[str, float]] = []
    all_rule_rows: List[Dict[str, float]] = []
    for dataset_name, csv_path, top_k_pairs in dataset_specs:
        summary, rule_rows = scan_dataset(
            dataset_name=dataset_name,
            csv_path=csv_path,
            reference_top_k=top_k_pairs,
            subject_limit=args.subject_limit,
            time_limit=args.time_limit,
        )
        dataset_summaries.append(summary)
        all_rule_rows.extend(rule_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    detail_path = output_dir / f"{args.tag}_{timestamp}.csv"
    summary_path = output_dir / f"{args.tag}_rank_{timestamp}.csv"

    write_csv(detail_path, all_rule_rows)
    write_csv(summary_path, dataset_summaries)

    print_summary(dataset_summaries, all_rule_rows)
    print(f"\nSaved detailed rule scan to: {detail_path}")
    print(f"Saved rank diagnostics to: {summary_path}")


if __name__ == "__main__":
    main()
