# Repository Guidelines

## Project Structure & Module Organization
Core pipeline lives in `src/`, with domain folders for data loaders (`src/data`), geometric losses & models (`src/models`), training loops (`src/training`), and evaluation/visualization helpers (`src/evaluation`). Entry points such as `main.py`, `train_complete_simple.py`, and `train_symmetry_simple.py` orchestrate phase workflows. Configuration defaults sit in `configs/config.yaml` (baseline) and `configs/config_geometric.yaml`; keep experiment-specific overrides in `configs/` and persist generated runs under `logs/` and `checkpoints/`. Raw assets reside in `data/` (images, annotations) and derivative outputs in `visualization_results/`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: set up PyTorch ROCm stack and tooling.
- `python main.py train1` / `python main.py train_geometric_complete`: run the phase-1 warmup or full geometric curriculum; add `--config configs/config_geometric.yaml` for overrides.
- `python main.py evaluate` or `python evaluate_complete.py`: report metrics using the latest checkpoint.
- `python main.py visualize_test_complete_loss`: export annotated predictions to `visualization_results/`.
- `python test_dataset.py` / `python test_gpu.py`: sanity-check dataloaders and GPU availability before long runs.

## Coding Style & Naming Conventions
Write Python 3.12 with 4-space indentation, trailing commas for multiline literals, and snake_case for variables, functions, and module names. Follow the existing bilingual docstring style (English headings, Spanish narrative) and keep public APIs typed (see `src/training/utils.py`). Place reusable logic under `src/`; CLI scripts should stay in the repository root and call into package functions. YAML keys use lower_snake_case.

## Testing Guidelines
Prefer lightweight smoke tests that hit dataloaders, model forward passes, and evaluation scripts. Name ad-hoc scripts `test_<scope>.py` and keep them executable. After any training change, run `python test_dataset.py` plus an evaluation command (`python main.py evaluate`) to confirm metrics and check for regressions. Attach visual outputs when altering loss functions so reviewers can compare heatmaps.

## Commit & Pull Request Guidelines
Commits follow Conventional Commits (`feat:`, `docs:`, `fix:`) per `git log`. Group related file changes together and include metric deltas or dataset notes in the body. Pull requests should: describe the phase affected, list configs or checkpoints touched, link tracking issues, and include evaluation results (pixel error, clinical tier) plus optional visualizations from `visualization_results/`.

## Experiment Tracking
Store new checkpoints under `checkpoints/<phase>_<date>` and mirror run metadata in `logs/`. For TensorBoard, run `tensorboard --logdir logs` and add the URL screenshot to PRs when sharing learning curves.
