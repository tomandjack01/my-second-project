from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SUMMARY_RE = re.compile(r"Summary written to:\s*(.+)")
AGGREGATE_RE = re.compile(r"Aggregate written to:\s*(.+)")


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    base_run_dir: str
    current_mask: str
    current_top_k_edges: int
    selection_top_k: int
    current_lag_weight: float
    phase1_topk_values: tuple[int, ...]
    lag_values: tuple[float, ...]


@dataclass(frozen=True)
class RunSpec:
    phase: str
    dataset: str
    mask_mode: str
    top_k_edges: int
    selection_top_k: int
    lag_weight: float
    seeds: str
    tag: str
    base_run_dir: str


DATASETS: Dict[str, DatasetSpec] = {
    "fMRI": DatasetSpec(
        "fMRI",
        "results/run_20260511_124622",
        "maxgap_kappa",
        5,
        5,
        0.35,
        (5, 7, 9),
        (0.25, 0.30, 0.35, 0.40, 0.50),
    ),
    "sim2": DatasetSpec(
        "sim2",
        "results/run_20260420_090231",
        "maxgap_kappa",
        11,
        11,
        0.25,
        (11, 13, 16),
        (0.15, 0.25, 0.30, 0.35, 0.40),
    ),
    "sim3": DatasetSpec(
        "sim3",
        "results/run_20260420_152306",
        "maxgap_kappa",
        18,
        18,
        0.25,
        (18, 21, 24),
        (0.15, 0.25, 0.30, 0.35, 0.40),
    ),
    "sim4": DatasetSpec(
        "sim4",
        "results/run_20260420_175556",
        "maxgap_kappa",
        61,
        61,
        0.25,
        (61, 70, 80),
        (0.15, 0.25, 0.30, 0.35, 0.40),
    ),
    "sim8": DatasetSpec(
        "sim8",
        "results/run_20260512_091906",
        "maxgap_kappa",
        5,
        5,
        0.35,
        (5, 7, 9),
        (0.25, 0.30, 0.35, 0.40, 0.50),
    ),
    "sim10": DatasetSpec(
        "sim10",
        "results/run_20260512_095729",
        "maxgap_kappa",
        5,
        5,
        0.35,
        (5, 7, 9),
        (0.25, 0.30, 0.35, 0.40, 0.50),
    ),
    "sim11": DatasetSpec(
        "sim11",
        "results/run_20260512_184814",
        "topk_kappa",
        16,
        11,
        0.35,
        (14, 16, 18, 20),
        (0.25, 0.30, 0.35, 0.40, 0.50),
    ),
    "sim12": DatasetSpec(
        "sim12",
        "results/run_20260512_111304",
        "maxgap_kappa",
        11,
        11,
        0.25,
        (11, 13, 16),
        (0.15, 0.25, 0.30, 0.35, 0.40),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the support-mask and causal-lag tuning plan."
    )
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2"],
        required=True,
        help="phase1 scans support masks; phase2 scans causal_lag_main_weight.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names to include.",
    )
    parser.add_argument("--seeds", default="11,22")
    parser.add_argument(
        "--run_id",
        default="param_tuning_20260514",
        help="Stable ID used in tags and manifest filenames.",
    )
    parser.add_argument(
        "--phase1_manifest",
        type=Path,
        default=None,
        help="Manifest produced by phase1; required for phase2 unless using defaults.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument(
        "--include_existing_mask_baseline",
        action="store_true",
        default=True,
        help="Include each dataset's current mask as the phase1 baseline.",
    )
    parser.add_argument(
        "--no_existing_mask_baseline",
        dest="include_existing_mask_baseline",
        action="store_false",
    )
    return parser.parse_args()


def parse_dataset_names(text: str) -> List[str]:
    names = [token.strip() for token in text.split(",") if token.strip()]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return names


def lag_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def phase1_specs(datasets: Iterable[DatasetSpec], run_id: str, seeds: str, include_baseline: bool) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for ds in datasets:
        if include_baseline:
            specs.append(
                RunSpec(
                    phase="phase1",
                    dataset=ds.dataset,
                    mask_mode=ds.current_mask,
                    top_k_edges=ds.current_top_k_edges,
                    selection_top_k=ds.selection_top_k,
                    lag_weight=ds.current_lag_weight,
                    seeds=seeds,
                    tag=(
                        f"{run_id}_phase1_{ds.dataset}_"
                        f"{ds.current_mask}_k{ds.current_top_k_edges}_lag{lag_tag(ds.current_lag_weight)}"
                    ),
                    base_run_dir=ds.base_run_dir,
                )
            )
        for top_k in ds.phase1_topk_values:
            specs.append(
                RunSpec(
                    phase="phase1",
                    dataset=ds.dataset,
                    mask_mode="topk_kappa",
                    top_k_edges=top_k,
                    selection_top_k=ds.selection_top_k,
                    lag_weight=ds.current_lag_weight,
                    seeds=seeds,
                    tag=f"{run_id}_phase1_{ds.dataset}_topk_kappa_k{top_k}_lag{lag_tag(ds.current_lag_weight)}",
                    base_run_dir=ds.base_run_dir,
                )
            )
    return dedupe_specs(specs)


def dedupe_specs(specs: Iterable[RunSpec]) -> List[RunSpec]:
    seen = set()
    out: List[RunSpec] = []
    for spec in specs:
        key = (
            spec.phase,
            spec.dataset,
            spec.mask_mode,
            spec.top_k_edges,
            spec.selection_top_k,
            spec.lag_weight,
            spec.seeds,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def score_aggregate(path: Path) -> tuple[float, float, float]:
    row = read_csv_rows(path)[0]
    final_f1 = float(row["final_primary_strict_f1_mean"])
    eps_f1 = float(row.get("final_strict_f1_eps_0p1_mean", "nan"))
    gap = float(row.get("final_vs_best_gap_mean", "nan"))
    return final_f1, eps_f1, gap


def select_phase1_support(manifest_path: Path, datasets: Iterable[DatasetSpec]) -> Dict[str, Dict[str, str]]:
    rows = [
        row
        for row in read_csv_rows(manifest_path)
        if row.get("phase") == "phase1" and row.get("status") == "ok"
    ]
    selected: Dict[str, Dict[str, str]] = {}
    for ds in datasets:
        candidates = [row for row in rows if row["dataset"] == ds.dataset]
        if not candidates:
            raise ValueError(f"No phase1 candidates found for {ds.dataset} in {manifest_path}")

        def sort_key(row: Dict[str, str]) -> tuple[float, float, float, int]:
            final_f1, eps_f1, gap = score_aggregate(Path(row["aggregate_path"]))
            # Higher final/eps is better; gap closer to zero is better; smaller k is preferred.
            return final_f1, eps_f1, -abs(gap), -int(row["top_k_edges"])

        selected[ds.dataset] = max(candidates, key=sort_key)
    return selected


def phase2_specs(
    datasets: Iterable[DatasetSpec],
    *,
    run_id: str,
    seeds: str,
    phase1_manifest: Optional[Path],
) -> List[RunSpec]:
    selected = select_phase1_support(phase1_manifest, datasets) if phase1_manifest else {}
    specs: List[RunSpec] = []
    for ds in datasets:
        chosen = selected.get(ds.dataset)
        mask_mode = chosen["mask_mode"] if chosen else ds.current_mask
        top_k_edges = int(chosen["top_k_edges"]) if chosen else ds.current_top_k_edges
        for lag in ds.lag_values:
            specs.append(
                RunSpec(
                    phase="phase2",
                    dataset=ds.dataset,
                    mask_mode=mask_mode,
                    top_k_edges=top_k_edges,
                    selection_top_k=ds.selection_top_k,
                    lag_weight=lag,
                    seeds=seeds,
                    tag=(
                        f"{run_id}_phase2_{ds.dataset}_{mask_mode}_"
                        f"k{top_k_edges}_lag{lag_tag(lag)}"
                    ),
                    base_run_dir=ds.base_run_dir,
                )
            )
    return dedupe_specs(specs)


def manifest_path(run_id: str, phase: str) -> Path:
    return RESULTS_DIR / f"{run_id}_{phase}_manifest.csv"


def command_for_spec(spec: RunSpec) -> List[str]:
    return [
        sys.executable,
        "run_replay_saved_config.py",
        "--base_run_dir",
        str(SCRIPT_DIR / spec.base_run_dir),
        "--seeds",
        spec.seeds,
        "--tag",
        spec.tag,
        "--set",
        "export_epoch_policy=final",
        "--set",
        f"fixed_support_mask_mode={spec.mask_mode}",
        "--set",
        f"top_k_edges={spec.top_k_edges}",
        "--set",
        f"selection_top_k={spec.selection_top_k}",
        "--set",
        f"causal_lag_main_weight={spec.lag_weight}",
    ]


def parse_output_path(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    path = Path(match.group(1).strip())
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return str(path)


def append_manifest(path: Path, row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "timestamp",
        "status",
        "phase",
        "dataset",
        "mask_mode",
        "top_k_edges",
        "selection_top_k",
        "lag_weight",
        "seeds",
        "tag",
        "base_run_dir",
        "summary_path",
        "aggregate_path",
        "driver_stdout_path",
        "driver_stderr_path",
        "error",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def already_done(path: Path, spec: RunSpec) -> bool:
    if not path.exists():
        return False
    for row in read_csv_rows(path):
        if (
            row.get("status") == "ok"
            and row.get("phase") == spec.phase
            and row.get("dataset") == spec.dataset
            and row.get("mask_mode") == spec.mask_mode
            and int(row.get("top_k_edges", "-1")) == spec.top_k_edges
            and int(row.get("selection_top_k", "-1")) == spec.selection_top_k
            and abs(float(row.get("lag_weight", "nan")) - spec.lag_weight) < 1e-9
            and row.get("seeds") == spec.seeds
            and Path(row.get("aggregate_path", "")).exists()
        ):
            return True
    return False


def run_spec(spec: RunSpec, *, manifest: Path, dry_run: bool) -> None:
    if already_done(manifest, spec):
        print(f"SKIP existing {spec.tag}", flush=True)
        return

    cmd = command_for_spec(spec)
    print(f"RUN {spec.tag}", flush=True)
    print(subprocess.list2cmdline(cmd), flush=True)
    if dry_run:
        return

    proc = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    stdout_path = RESULTS_DIR / f"{spec.tag}_driver_stdout.log"
    stderr_path = RESULTS_DIR / f"{spec.tag}_driver_stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    status = "ok" if proc.returncode == 0 else "failed"
    error = "" if proc.returncode == 0 else f"returncode={proc.returncode}"
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "phase": spec.phase,
        "dataset": spec.dataset,
        "mask_mode": spec.mask_mode,
        "top_k_edges": str(spec.top_k_edges),
        "selection_top_k": str(spec.selection_top_k),
        "lag_weight": f"{spec.lag_weight:g}",
        "seeds": spec.seeds,
        "tag": spec.tag,
        "base_run_dir": str((SCRIPT_DIR / spec.base_run_dir).resolve()),
        "summary_path": parse_output_path(SUMMARY_RE, proc.stdout),
        "aggregate_path": parse_output_path(AGGREGATE_RE, proc.stdout),
        "driver_stdout_path": str(stdout_path),
        "driver_stderr_path": str(stderr_path),
        "error": error,
    }
    append_manifest(manifest, row)
    if proc.returncode != 0:
        raise RuntimeError(f"{spec.tag} failed; see {stdout_path} and {stderr_path}")


def main() -> None:
    args = parse_args()
    dataset_specs = [DATASETS[name] for name in parse_dataset_names(args.datasets)]
    if args.phase == "phase1":
        specs = phase1_specs(
            dataset_specs,
            run_id=args.run_id,
            seeds=args.seeds,
            include_baseline=args.include_existing_mask_baseline,
        )
    else:
        specs = phase2_specs(
            dataset_specs,
            run_id=args.run_id,
            seeds=args.seeds,
            phase1_manifest=args.phase1_manifest,
        )

    out_manifest = manifest_path(args.run_id, args.phase)
    print(f"Manifest: {out_manifest}", flush=True)
    print(f"Specs: {len(specs)}", flush=True)
    for spec in specs:
        try:
            run_spec(spec, manifest=out_manifest, dry_run=args.dry_run)
        except Exception as exc:
            print(f"ERROR {spec.tag}: {exc}", flush=True)
            if not args.keep_going:
                raise


if __name__ == "__main__":
    main()
