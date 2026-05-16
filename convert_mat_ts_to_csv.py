import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def convert_file(mat_path: Path, output_dir: Path, overwrite: bool = False) -> Path:
    data = loadmat(mat_path)
    if "ts" not in data:
        raise KeyError(f"{mat_path} does not contain variable 'ts'")

    ts = np.asarray(data["ts"])
    if ts.ndim != 2:
        raise ValueError(f"{mat_path} variable 'ts' must be 2-D, got shape {ts.shape}")

    output_path = output_dir / f"{mat_path.stem}.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; use --overwrite to replace it")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, ts, delimiter=",", fmt="%.18g")
    return output_path


def convert_directory(input_dir: Path, output_dir: Path, overwrite: bool = False) -> int:
    mat_files = sorted(input_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {input_dir}")

    for mat_path in mat_files:
        output_path = convert_file(mat_path, output_dir, overwrite=overwrite)
        print(f"{mat_path.name} -> {output_path}")

    return len(mat_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the MATLAB variable 'ts' from each .mat file to headerless CSV."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("mat_dataset"),
        help="Directory containing .mat files. Default: mat_dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("mat_dataset_csv"),
        help="Directory for generated .csv files. Default: mat_dataset_csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_directory(args.input_dir, args.output_dir, overwrite=args.overwrite)
    print(f"Converted {count} file(s).")


if __name__ == "__main__":
    main()
