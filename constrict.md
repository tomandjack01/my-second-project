# Mainline Constriction Log

Last updated: 2026-03-29

## Why Constrict Now

- Experiment conclusions are converging faster than the code path.
- The shared training loop in `GraphExp/main_structure_learning.py` now mixes:
  - mechanism changes
  - retention fixes
  - selection fixes
  - multiple auxiliary losses
- That makes attribution weak and turns each new tweak into a small local patch on
  top of a crowded objective stack.
- Current evidence supports a narrower diagnosis:
  - time supervision is not ruled out as a signal source
  - but adding more time-supervised auxiliary losses into the current shared
    loop is near exhaustion
- The recent soft-consistency weighting follow-up reinforces this:
  - same teacher
  - cleaner weighting
  - no improvement over the current best time-supervised point on `sim3`

## Locked Judgment

- Freeze the current shared main training branch as the reference branch.
- Do not add new auxiliary losses to the main shared loop for now.
- Treat "time supervision as auxiliary loss inside the existing loop" as the
  implementation path that is currently near its limit.
- Split further work into two independent lines:
  - mechanism line
  - selection line

## Frozen Mainline

- Keep the current branch for:
  - baselines
  - confirmation runs
  - regression checks
- Do not expand the parallel mechanism stack in
  `GraphExp/main_structure_learning.py` unless one of the constrained branches
  below shows a decisive result.

## Line 1: Mechanism

### Goal

Test whether the remaining directional bottleneck is mainly in exported space,
not in raw directional-logit space.

### Working Model

- `support_direction` decomposes the graph into:
  - symmetric support
  - directional gate
- Exported adjacency is:
  - `support_weights * direction_gate`
- `direction_gate = sigmoid(D - D^T)`
- Therefore the next mechanism work should target exported quantities directly,
  instead of first shaping raw `D - D^T` and hoping the effect survives export.

### Rules

- Prefer constraints on:
  - exported gate
  - exported directional margin
  - support-preserved directional pairs
- Do not start by adding another raw-logit-only auxiliary loss.
- Keep each mechanism experiment minimal and isolated.
- Compare against the frozen mainline, not against a moving target.

### First Candidate Experiments

1. Exported-gate floor on active directional pairs.
2. Exported-margin hinge on active directional pairs.
3. Support-preservation constraint on direction-supervised pairs.

## Line 2: Selection

### Goal

Separate checkpoint-selection failures from training-mechanism failures.

### Working Model

- `fMRI.csv` currently looks more like a selection failure than a training
  failure.
- The branch can reach a useful final result there, but the current best-epoch
  proxy can export a much worse checkpoint.
- Therefore this line should modify selection logic, not training losses.

### Rules

- Do not change training objectives on this line.
- Focus on:
  - `compute_epoch_quality(...)`
  - guardrails
  - dataset-dependent export behavior
- Use `fMRI.csv` + `h1.txt` as the primary debugging target.
- Only after the selector is trustworthy should this line be promoted to the
  larger synthetic sets.

### First Candidate Experiments

1. Audit the current proxy terms against actual best/final behavior on
   `fMRI.csv`.
2. Test selector variants that reduce early wrong exports without touching
   training.
3. Separate "best by proxy" from "final epoch" reporting in every selector run.

## Execution Policy

- No new broad sweeps until a branch has a clearly stated mechanism claim.
- No mixed branch runs that modify both mechanism and selector at once.
- Each run must state:
  - branch name
  - dataset
  - exact claim being tested
  - stop condition

## Experiment Log

### 2026-03-29 - Mainline constriction adopted

- Status:
  - active
- Decision:
  - freeze the current main shared training branch
  - stop adding new auxiliary losses to that loop
  - split work into a mechanism line and a selection line
- Immediate next priority:
  - selection-line audit on `fMRI.csv`
  - minimal exported-space mechanism experiment

### 2026-03-29 - Selection-line GT audit plumbing smoke

- Branch line:
  - selection
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Exact config delta:
  - added optional selector-only GT audit logging to
    `GraphExp/main_structure_learning.py`
  - new CLI:
    - `--selector_audit_gt_path`
    - `--selector_audit_strict_margin_eps_values`
  - audit writes per-epoch GT metrics into `quality_history.csv`
  - audit writes a run-level `selector_audit_summary.csv`
  - audit does **not** affect:
    - training loss
    - checkpoint selection
    - exported adjacency choice
- Smoke command:
  - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\fMRI.csv --device cpu --epochs 1 --pretrain_epochs 0 --skip_pretrain --subject_limit 2 --time_limit 20 --top_k_edges 5 --log_interval 1 --selector_audit_gt_path ..\fMRI_dataset\h1.txt`
- Result:
  - pass
  - run dir:
    - `GraphExp/results/run_20260329_165415`
  - confirmed:
    - training still runs end-to-end
    - `quality_history.csv` is produced
    - `selector_audit_summary.csv` is produced
    - terminal summary now reports:
      - best GT epoch
      - exported epoch
      - final epoch
      - primary strict-F1 gaps
- Interpretation:
  - selection-line audit is now instrumented enough for a real
    `fMRI.csv` selector diagnosis run
  - this is observability only, not a selector fix
- Keep / drop:
  - keep

### 2026-03-29 - Selection-line audit on current `fMRI.csv` recommended branch (`seed=11`)

- Branch line:
  - selection
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Exact config delta:
  - keep the current recommended training branch unchanged
  - enable selector-only GT audit
  - config:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `direction_init_mode = random`
    - `structure_init_mode = patel_kappa`
    - `structure_init_scale = 0.5`
    - `adj_activation = sigmoid`
    - `directional_prior_mode = patel`
    - `directional_schedule = plateau`
    - `directional_kappa_gate = True`
    - `directional_kappa_gate_quantile = 0.50`
    - `directional_target_ratio = 0.01`
    - `lambda_l1 = 0.02`
    - `optimizer_step_mode = subject`
    - `main_loss_weight = 1.0`
    - `selection_agreement_weight = 0.0`
    - `direction_lr_multiplier = 1.0`
    - `freeze_direction_after_epoch = 30`
    - `seed = 11`
- Command:
  - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\fMRI.csv --selector_audit_gt_path ..\fMRI_dataset\h1.txt --device cuda --epochs 100 --pretrain_epochs 50 --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --top_k_edges 5 --log_interval 10 --structure_parameterization support_direction --fixed_support_mask_mode maxgap_kappa --direction_init_mode random --structure_init_mode patel_kappa --structure_init_scale 0.5 --adj_activation sigmoid --directional_prior_mode patel --directional_schedule plateau --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratio 0.01 --lambda_l1 0.02 --optimizer_step_mode subject --main_loss_weight 1.0 --selection_agreement_weight 0.0 --direction_lr_multiplier 1.0 --freeze_direction_after_epoch 30 --seed 11`
- Result:
  - run dir:
    - `GraphExp/results/run_20260329_170115`
  - selector audit summary:
    - best GT epoch = `18`
    - best GT strict-F1@`eps=0` = `1.0000`
    - exported epoch = `6`
    - exported strict-F1@`eps=0` = `0.2000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.8000`
  - primary gaps:
    - exported vs best-GT = `-0.8000`
    - final vs best-GT = `-0.2000`
- Key observation:
  - every eligible epoch failed the same guardrail:
    - `low_density_factor|density_ratio_out_of_range`
  - under the current selector definition, `actual_pair_density` stayed `0.0`
    through the audited epochs on this run
  - therefore guarded selection never activates here
  - the selector falls back to score-only and chooses an early high-margin epoch
    that is directionally much worse than later epochs
- Interpretation:
  - this strongly supports the earlier diagnosis that `fMRI.csv` is currently a
    selector problem, not primarily a training problem
  - the present proxy is not just "slightly noisy" on this branch
  - it is structurally mismatched to this regime because the density term and
    density guardrail collapse to a degenerate state
- Keep / drop:
  - keep the audit result
  - drop any interpretation that this run's poor export is mainly a training-loss
    issue

### 2026-03-29 - Selection-line audit with `selection_agreement_weight = 0.25` (`seed=11`)

- Branch line:
  - selection
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Exact config delta:
  - same audited branch as above
  - only change:
    - `selection_agreement_weight: 0.0 -> 0.25`
- Result:
  - run dir:
    - `GraphExp/results/run_20260329_171027`
  - selector audit summary:
    - best GT epoch = `18`
    - exported epoch = `12`
    - final epoch = `100`
    - exported strict-F1@`eps=0` = `0.8000`
    - final strict-F1@`eps=0` = `0.8000`
  - compared with the audited `selection_agreement_weight = 0.0` run:
    - exported strict-F1@`eps=0`: `0.2000 -> 0.8000`
    - exported epoch: `6 -> 12`
- Interpretation:
  - on this branch, reintroducing Patel-agreement into selector scoring is a
    large first-order improvement
  - this does **not** fully solve the selector:
    - best GT epoch still remains `18`
    - exported epoch still lands earlier at `12`
  - but it removes the worst early wrong-direction export behavior
- Keep / drop:
  - keep as the current best selector setting for this `fMRI.csv` branch

### 2026-03-29 - Fixed-support selector density neutralization (`seed=11` verification)

- Branch line:
  - selection
- Code change:
  - in `GraphExp/main_structure_learning.py`, the proxy now neutralizes
    `density_factor` and density guardrails when `fixed_support_mask` is active
  - rationale:
    - when support pairs are fixed externally, hard `> 0.5` pair counting is no
      longer a meaningful selection diagnostic
- Verification run:
  - same config as the previous `selection_agreement_weight = 0.25` audit
  - run dir:
    - `GraphExp/results/run_20260329_171638`
- Result:
  - selector mode:
    - `score_only_fallback -> guarded`
  - exported epoch remained:
    - `12`
  - selector audit summary remained:
    - exported strict-F1@`eps=0` = `0.8000`
    - best GT epoch = `18`
    - final strict-F1@`eps=0` = `0.8000`
- Interpretation:
  - this code change does not by itself improve the chosen epoch beyond the
    `selection_agreement_weight = 0.25` configuration
  - but it fixes a structural selector bug:
    - guarded selection no longer collapses into permanent density failure under
      fixed-support regimes
  - therefore the selector is now better calibrated for this branch, even
    though it is not yet fully optimal
- Keep / drop:
  - keep

### 2026-03-29 - Soft-weighted Patel agreement selector follow-up (`seed=11`)

- Branch line:
  - selection
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Exact config delta:
  - same fixed-support audited branch
  - keep:
    - `selection_agreement_weight = 0.25`
  - change:
    - `selection_agreement_mode = soft_weighted`
- Result:
  - run dir:
    - `GraphExp/results/run_20260329_182755`
  - selector audit summary:
    - best GT epoch = `18`
    - exported epoch = `12`
    - final epoch = `100`
    - exported strict-F1@`eps=0` = `0.8000`
    - final strict-F1@`eps=0` = `0.8000`
  - detailed readout:
    - epoch `12`:
      - `agreement_soft_score = 1.0000`
      - `selector_audit_strict_f1@eps=0 = 0.8000`
    - epoch `18`:
      - `agreement_soft_score = 0.8989`
      - `selector_audit_strict_f1@eps=0 = 1.0000`
- Interpretation:
  - this is a strong negative result for "make the selector more Patel-like"
  - the remaining `epoch 12 -> epoch 18` gap is **not** fixed by moving from
    hard Patel agreement to soft Patel-weighted agreement
  - on this run, Patel-weighted agreement actually prefers the worse exported
    epoch (`12`) over the GT-best epoch (`18`)
  - therefore the current residual selector gap on `fMRI.csv` is unlikely to be
    solved by stronger Patel-based agreement terms
- Keep / drop:
  - keep the result
  - drop Patel-agreement strengthening as the next selector lever for this
    branch

## Logging Rule

All future experiments launched under this constriction phase should be appended
to this file with:

- date
- branch line (`mechanism` or `selection`)
- dataset
- exact config delta
- result
- interpretation
- keep / drop decision
