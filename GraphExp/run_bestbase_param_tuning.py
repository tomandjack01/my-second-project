from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from run_replay_saved_config import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SUMMARY_RE = re.compile(r"Summary written to:\s*(.+)")
AGGREGATE_RE = re.compile(r"Aggregate written to:\s*(.+)")
DEFAULT_SEEDS = "11,22,33,44,55"
DEFAULT_RUN_ID = "param_tuning_bestbase_20260516"


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    base_run_dir: str
    selection_top_k: int
    base_lag_weight: float
    topk_values: tuple[int, ...]
    lag_values: tuple[float, ...]
    new_dataset_repretrain: bool = False


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    dataset: str
    base_run_dir: str
    seeds: str
    tag: str
    top_k_edges: Optional[int]
    selection_top_k: Optional[int]
    lag_weight: float
    overrides: tuple[str, ...]

    @property
    def candidate_key(self) -> str:
        if self.experiment == "topk":
            return f"topk_kappa_k{self.top_k_edges}_lag{format_tag_float(self.lag_weight)}"
        return f"base_support_lag{format_tag_float(self.lag_weight)}"


DATASETS: Dict[str, DatasetSpec] = {
    "fMRI": DatasetSpec(
        dataset="fMRI",
        base_run_dir="results/run_20260511_124622",
        selection_top_k=5,
        base_lag_weight=0.35,
        topk_values=(5, 7, 9),
        lag_values=(0.25, 0.30, 0.35, 0.40, 0.50),
    ),
    "sim2": DatasetSpec(
        dataset="sim2",
        base_run_dir="results/run_20260420_090231",
        selection_top_k=11,
        base_lag_weight=0.25,
        topk_values=(11, 13, 16),
        lag_values=(0.15, 0.25, 0.30, 0.35, 0.40),
    ),
    "sim3": DatasetSpec(
        dataset="sim3",
        base_run_dir="results/run_20260420_152306",
        selection_top_k=18,
        base_lag_weight=0.25,
        topk_values=(18, 21, 24),
        lag_values=(0.15, 0.25, 0.30, 0.35, 0.40),
    ),
    "sim4": DatasetSpec(
        dataset="sim4",
        base_run_dir="results/run_20260420_175556",
        selection_top_k=61,
        base_lag_weight=0.25,
        topk_values=(61, 70, 80),
        lag_values=(0.15, 0.25, 0.30, 0.35, 0.40),
    ),
    "sim8": DatasetSpec(
        dataset="sim8",
        base_run_dir="results/run_20260512_091906",
        selection_top_k=5,
        base_lag_weight=0.35,
        topk_values=(5, 7, 9),
        lag_values=(0.25, 0.30, 0.35, 0.40, 0.50),
        new_dataset_repretrain=True,
    ),
    "sim10": DatasetSpec(
        dataset="sim10",
        base_run_dir="results/run_20260512_095729",
        selection_top_k=5,
        base_lag_weight=0.35,
        topk_values=(5, 7, 9),
        lag_values=(0.25, 0.30, 0.35, 0.40, 0.50),
        new_dataset_repretrain=True,
    ),
    "sim11": DatasetSpec(
        dataset="sim11",
        base_run_dir="results/run_20260512_184814",
        selection_top_k=11,
        base_lag_weight=0.35,
        topk_values=(14, 16, 18, 20),
        lag_values=(0.25, 0.30, 0.35, 0.40, 0.50),
        new_dataset_repretrain=True,
    ),
    "sim12": DatasetSpec(
        dataset="sim12",
        base_run_dir="results/run_20260512_111304",
        selection_top_k=11,
        base_lag_weight=0.25,
        topk_values=(11, 13, 16),
        lag_values=(0.15, 0.25, 0.30, 0.35, 0.40),
        new_dataset_repretrain=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run formal best-known-base parameter tuning for topk_kappa/top_k_edges "
            "and causal_lag_main_weight."
        )
    )
    parser.add_argument(
        "--phase",
        choices=["topk", "lag", "all"],
        default="all",
        help="Experiment A topk scan, Experiment B lag scan, or both.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--run_id", default=DEFAULT_RUN_ID)
    parser.add_argument("--device", default=None, help="Optional device override forwarded to replay.")
    parser.add_argument(
        "--python_executable",
        default=sys.executable,
        help="Python executable used to invoke run_replay_saved_config.py.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Write a dry-run manifest and print commands.")
    parser.add_argument("--keep_going", action="store_true", help="Continue after a failed replay spec.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path. Defaults to results/<run_id>_<phase>_manifest.csv.",
    )
    parser.add_argument(
        "--max_specs",
        type=int,
        default=0,
        help="Optional cap on specs after filtering; useful for smoke checks.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help=(
            "Optional candidate filter. For topk use DATASET:K, for lag use DATASET:LAG. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--no_validate_base",
        action="store_true",
        help="Skip validation that base configs are non-ablation guided-noise configs.",
    )
    return parser.parse_args()


def parse_dataset_names(text: str) -> List[str]:
    names = [token.strip() for token in text.split(",") if token.strip()]
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return names


def parse_candidate_filters(filters: Sequence[str]) -> Dict[str, set[str]]:
    parsed: Dict[str, set[str]] = {}
    for item in filters:
        if ":" not in item:
            raise ValueError(f"Candidate filter must be DATASET:VALUE, got {item!r}")
        dataset, value = item.split(":", 1)
        dataset = dataset.strip()
        value = value.strip()
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset in candidate filter: {dataset}")
        parsed.setdefault(dataset, set()).add(value)
    return parsed


def format_tag_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def format_manifest_float(value: float) -> str:
    return f"{value:g}"


def base_path(spec: DatasetSpec) -> Path:
    return (SCRIPT_DIR / spec.base_run_dir).resolve()


def new_dataset_overrides(spec: DatasetSpec) -> List[str]:
    if not spec.new_dataset_repretrain:
        return []
    return ["pretrain_checkpoint=", "pretrain_epochs=50"]


def topk_specs(specs: Iterable[DatasetSpec], *, run_id: str, seeds: str) -> List[RunSpec]:
    out: List[RunSpec] = []
    for spec in specs:
        for top_k in spec.topk_values:
            overrides = [
                "export_epoch_policy=final",
                "fixed_support_mask_mode=topk_kappa",
                f"top_k_edges={top_k}",
                f"selection_top_k={spec.selection_top_k}",
                f"causal_lag_main_weight={format_manifest_float(spec.base_lag_weight)}",
                *new_dataset_overrides(spec),
            ]
            out.append(
                RunSpec(
                    experiment="topk",
                    dataset=spec.dataset,
                    base_run_dir=spec.base_run_dir,
                    seeds=seeds,
                    tag=(
                        f"{run_id}_topk_{spec.dataset}_"
                        f"k{top_k}_lag{format_tag_float(spec.base_lag_weight)}"
                    ),
                    top_k_edges=top_k,
                    selection_top_k=spec.selection_top_k,
                    lag_weight=spec.base_lag_weight,
                    overrides=tuple(overrides),
                )
            )
    return out


def lag_specs(specs: Iterable[DatasetSpec], *, run_id: str, seeds: str) -> List[RunSpec]:
    out: List[RunSpec] = []
    for spec in specs:
        for lag in spec.lag_values:
            overrides = [
                "export_epoch_policy=final",
                f"causal_lag_main_weight={format_manifest_float(lag)}",
                *new_dataset_overrides(spec),
            ]
            out.append(
                RunSpec(
                    experiment="lag",
                    dataset=spec.dataset,
                    base_run_dir=spec.base_run_dir,
                    seeds=seeds,
                    tag=f"{run_id}_lag_{spec.dataset}_lag{format_tag_float(lag)}",
                    top_k_edges=None,
                    selection_top_k=None,
                    lag_weight=lag,
                    overrides=tuple(overrides),
                )
            )
    return out


def build_specs(phase: str, datasets: List[DatasetSpec], *, run_id: str, seeds: str) -> List[RunSpec]:
    specs: List[RunSpec] = []
    if phase in {"topk", "all"}:
        specs.extend(topk_specs(datasets, run_id=run_id, seeds=seeds))
    if phase in {"lag", "all"}:
        specs.extend(lag_specs(datasets, run_id=run_id, seeds=seeds))
    return specs


def filter_specs(specs: List[RunSpec], filters: Dict[str, set[str]]) -> List[RunSpec]:
    if not filters:
        return specs
    out: List[RunSpec] = []
    for spec in specs:
        values = filters.get(spec.dataset)
        if not values:
            continue
        if spec.experiment == "topk" and str(spec.top_k_edges) in values:
            out.append(spec)
        elif spec.experiment == "lag" and (
            format_manifest_float(spec.lag_weight) in values
            or f"{spec.lag_weight:.2f}" in values
        ):
            out.append(spec)
    return out


def validate_base_configs(dataset_specs: Iterable[DatasetSpec]) -> None:
    for spec in dataset_specs:
        cfg = load_config(base_path(spec))
        if bool(cfg.get("disable_temporal_encoder", False)):
            raise ValueError(f"{spec.dataset} base config has disable_temporal_encoder=True")
        if str(cfg.get("diffusion_noise_mode", "guided")) == "gaussian_iid":
            raise ValueError(f"{spec.dataset} base config uses gaussian_iid diffusion noise")


def command_for_spec(spec: RunSpec, device: Optional[str], python_executable: str) -> List[str]:
    cmd = [
        python_executable,
        "run_replay_saved_config.py",
        "--base_run_dir",
        str((SCRIPT_DIR / spec.base_run_dir).resolve()),
        "--seeds",
        spec.seeds,
        "--tag",
        spec.tag,
    ]
    if device:
        cmd.extend(["--device", device])
    for override in spec.overrides:
        cmd.extend(["--set", override])
    return cmd


def manifest_path(run_id: str, phase: str, dry_run: bool) -> Path:
    suffix = "_dry_run" if dry_run else ""
    return RESULTS_DIR / f"{run_id}_{phase}{suffix}_manifest.csv"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_one_csv_row(path: Path) -> Dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")
    return rows[0]


def seed_count(seeds: str) -> int:
    return len([token for token in seeds.split(",") if token.strip()])


def aggregate_complete(path_text: str, expected_count: int) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if not path.exists():
        return False
    try:
        row = read_one_csv_row(path)
    except Exception:
        return False
    try:
        return int(float(row.get("run_count", "0"))) == expected_count
    except ValueError:
        return False


def already_done(path: Path, spec: RunSpec) -> bool:
    if not path.exists():
        return False
    expected_count = seed_count(spec.seeds)
    for row in read_csv_rows(path):
        if (
            row.get("status") == "ok"
            and row.get("experiment") == spec.experiment
            and row.get("dataset") == spec.dataset
            and row.get("candidate_key") == spec.candidate_key
            and row.get("seeds") == spec.seeds
            and row.get("override_spec") == ";".join(spec.overrides)
            and aggregate_complete(row.get("aggregate_path", ""), expected_count)
        ):
            return True
    return False


def parse_output_path(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    path = Path(match.group(1).strip())
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return str(path)


def manifest_fieldnames() -> List[str]:
    return [
        "timestamp",
        "status",
        "experiment",
        "dataset",
        "candidate_key",
        "base_run_dir",
        "seeds",
        "top_k_edges",
        "selection_top_k",
        "lag_weight",
        "override_spec",
        "tag",
        "summary_path",
        "aggregate_path",
        "driver_stdout_path",
        "driver_stderr_path",
        "command",
        "error",
    ]


def append_manifest(path: Path, row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fieldnames())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def base_manifest_row(spec: RunSpec, command: List[str]) -> Dict[str, str]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "",
        "experiment": spec.experiment,
        "dataset": spec.dataset,
        "candidate_key": spec.candidate_key,
        "base_run_dir": str((SCRIPT_DIR / spec.base_run_dir).resolve()),
        "seeds": spec.seeds,
        "top_k_edges": "" if spec.top_k_edges is None else str(spec.top_k_edges),
        "selection_top_k": "" if spec.selection_top_k is None else str(spec.selection_top_k),
        "lag_weight": format_manifest_float(spec.lag_weight),
        "override_spec": ";".join(spec.overrides),
        "tag": spec.tag,
        "summary_path": "",
        "aggregate_path": "",
        "driver_stdout_path": "",
        "driver_stderr_path": "",
        "command": subprocess.list2cmdline(command),
        "error": "",
    }


def run_spec(
    spec: RunSpec,
    *,
    manifest: Path,
    dry_run: bool,
    device: Optional[str],
    python_executable: str,
) -> None:
    cmd = command_for_spec(spec, device, python_executable)
    print(f"{'DRY' if dry_run else 'RUN'} {spec.experiment} {spec.dataset} {spec.candidate_key}", flush=True)
    print(subprocess.list2cmdline(cmd), flush=True)

    if dry_run:
        row = base_manifest_row(spec, cmd)
        row["status"] = "dry_run"
        append_manifest(manifest, row)
        return

    if already_done(manifest, spec):
        print(f"SKIP existing {spec.tag}", flush=True)
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

    row = base_manifest_row(spec, cmd)
    row["status"] = "ok" if proc.returncode == 0 else "failed"
    row["summary_path"] = parse_output_path(SUMMARY_RE, proc.stdout)
    row["aggregate_path"] = parse_output_path(AGGREGATE_RE, proc.stdout)
    row["driver_stdout_path"] = str(stdout_path)
    row["driver_stderr_path"] = str(stderr_path)
    row["error"] = "" if proc.returncode == 0 else f"returncode={proc.returncode}"
    append_manifest(manifest, row)

    if proc.returncode != 0:
        raise RuntimeError(f"{spec.tag} failed; see {stdout_path} and {stderr_path}")


def main() -> None:
    args = parse_args()
    dataset_specs = [DATASETS[name] for name in parse_dataset_names(args.datasets)]
    if not args.no_validate_base:
        validate_base_configs(dataset_specs)

    specs = build_specs(args.phase, dataset_specs, run_id=args.run_id, seeds=args.seeds)
    specs = filter_specs(specs, parse_candidate_filters(args.candidate))
    if args.max_specs > 0:
        specs = specs[: args.max_specs]

    out_manifest = args.manifest or manifest_path(args.run_id, args.phase, args.dry_run)
    print(f"Manifest: {out_manifest}", flush=True)
    print(f"Specs: {len(specs)}", flush=True)

    for spec in specs:
        try:
            run_spec(
                spec,
                manifest=out_manifest,
                dry_run=args.dry_run,
                device=args.device,
                python_executable=args.python_executable,
            )
        except Exception as exc:
            print(f"ERROR {spec.tag}: {exc}", flush=True)
            if not args.keep_going:
                raise


if __name__ == "__main__":
    main()
