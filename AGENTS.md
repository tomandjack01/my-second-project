# Repository Guidelines

## Project Structure & Module Organization

- `GraphExp/`: graph-classification experiments and utilities (also contains `main_structure_learning.py` for fMRI connectivity learning).
- `NodeExp/`: node-classification experiments.
- `nni_search/`: NNI-based hyperparameter search entrypoints/config.
- `json_config/`, `*/yamls/`: experiment configuration files.
- `fMRI_dataset/`: local dataset assets (keep large/private data out of PRs).
- `GraphExp/results/`: generated outputs (e.g., `run_YYYYmmdd_HHMMSS/`); do not commit.

## Build, Test, and Development Commands

Environment (CUDA/DGL/PyTorch may need separate install per your system):
```sh
conda create -n ddm python=3.8
conda activate ddm
pip install -r requirements.txt
```

Run experiments:
```sh
cd GraphExp
python main_graph.py --yaml_dir ./yamls/MUTAG.yaml

cd ../NodeExp
python main_node.py --yaml_dir ./yamls/photo.yaml
```

Structure learning (fMRI):
```sh
python .\GraphExp\main_structure_learning.py --csv_path ..\fMRI_dataset\fMRI.csv --device cpu
```

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8–style naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Prefer small, targeted diffs; avoid sweeping reformatting unless required.
- Type hints are welcome for new/changed public functions.
- Static checks: keep `pyrightconfig.json` passing when applicable.

## Testing Guidelines

- No unified test runner is enforced in this repo snapshot. Use targeted script checks where available, e.g.:
  - `python .\GraphExp\test_temporal_encoder.py`
- When adding tests, prefer lightweight `pytest`-style functions only if the project already uses it; otherwise keep runnable scripts near the feature.

## Commit & Pull Request Guidelines

- Git history may be unavailable in some workspace snapshots; use clear, imperative commit messages (e.g., “Fix temporal encoder gradients”).
- PRs should include: purpose, how to run/verify (commands + config paths), and any expected output artifacts (do not include large `results/` files).
- Link related issues/notes and mention hardware assumptions (CPU/CUDA) when relevant.

## Agent-Specific Notes

- Avoid committing generated artifacts under `GraphExp/results/` and caches like `__pycache__/`.
- Keep dataset paths configurable via CLI flags/YAML; don’t hardcode machine-specific absolute paths.
