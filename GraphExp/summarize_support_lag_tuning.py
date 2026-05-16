from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize support/lag tuning manifests into compact CSV and Markdown tables."
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_aggregate(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in aggregate CSV: {path}")
    return rows[0]


def collect_rows(manifests: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for manifest in manifests:
        for manifest_row in read_csv_rows(manifest):
            if manifest_row.get("status") != "ok":
                continue
            aggregate_path = Path(manifest_row["aggregate_path"])
            if not aggregate_path.exists():
                continue
            agg = read_aggregate(aggregate_path)
            rows.append(
                {
                    "phase": manifest_row["phase"],
                    "dataset": manifest_row["dataset"],
                    "mask_mode": manifest_row["mask_mode"],
                    "top_k_edges": manifest_row["top_k_edges"],
                    "selection_top_k": manifest_row["selection_top_k"],
                    "lag_weight": manifest_row["lag_weight"],
                    "seeds": manifest_row["seeds"],
                    "run_count": agg["run_count"],
                    "best_primary_strict_f1_mean": agg["best_primary_strict_f1_mean"],
                    "exported_primary_strict_f1_mean": agg["exported_primary_strict_f1_mean"],
                    "final_primary_strict_f1_mean": agg["final_primary_strict_f1_mean"],
                    "final_strict_f1_eps_0p1_mean": agg["final_strict_f1_eps_0p1_mean"],
                    "final_signed_margin_median_mean": agg["final_signed_margin_median_mean"],
                    "final_vs_best_gap_mean": agg["final_vs_best_gap_mean"],
                    "final_failure_mode_counts": agg["final_failure_mode_counts"],
                    "aggregate_path": str(aggregate_path),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(text: str, digits: int = 4) -> str:
    try:
        return f"{float(text):.{digits}f}"
    except ValueError:
        return text


def write_markdown(path: Path, rows: List[Dict[str, str]]) -> None:
    lines = [
        "# Support Mask / Causal-Lag Tuning Summary",
        "",
        "| phase | dataset | mask | top_k | sel_k | lag | runs | best | exported | final | final eps=0.1 | final margin | final-best gap | failure |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        rows,
        key=lambda r: (
            r["phase"],
            r["dataset"],
            r["mask_mode"],
            int(r["top_k_edges"]),
            float(r["lag_weight"]),
        ),
    ):
        lines.append(
            "| {phase} | {dataset} | {mask} | {topk} | {selk} | {lag} | {runs} | {best} | {exported} | {final} | {eps} | {margin} | {gap} | `{failure}` |".format(
                phase=row["phase"],
                dataset=row["dataset"],
                mask=row["mask_mode"],
                topk=row["top_k_edges"],
                selk=row["selection_top_k"],
                lag=row["lag_weight"],
                runs=row["run_count"],
                best=fmt_float(row["best_primary_strict_f1_mean"]),
                exported=fmt_float(row["exported_primary_strict_f1_mean"]),
                final=fmt_float(row["final_primary_strict_f1_mean"]),
                eps=fmt_float(row["final_strict_f1_eps_0p1_mean"]),
                margin=fmt_float(row["final_signed_margin_median_mean"]),
                gap=fmt_float(row["final_vs_best_gap_mean"]),
                failure=row["final_failure_mode_counts"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_best_by_dataset(rows: List[Dict[str, str]]) -> None:
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["dataset"])].append(row)
    for key, group in sorted(grouped.items()):
        best = max(
            group,
            key=lambda r: (
                float(r["final_primary_strict_f1_mean"]),
                float(r["final_strict_f1_eps_0p1_mean"]),
                -abs(float(r["final_vs_best_gap_mean"])),
                -int(r["top_k_edges"]),
            ),
        )
        print(
            f"{key[0]} {key[1]}: "
            f"mask={best['mask_mode']} top_k={best['top_k_edges']} "
            f"sel_k={best['selection_top_k']} lag={best['lag_weight']} "
            f"final={float(best['final_primary_strict_f1_mean']):.4f} "
            f"eps0.1={float(best['final_strict_f1_eps_0p1_mean']):.4f} "
            f"gap={float(best['final_vs_best_gap_mean']):.4f}"
        )


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.manifest)
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    print_best_by_dataset(rows)


if __name__ == "__main__":
    main()
