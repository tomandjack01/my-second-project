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

## Architecture Judgment

### 2026-03-29 - Selector mismatch is downstream of objective mismatch

- Status:
  - active
- Scope:
  - frozen mainline interpretation
  - no new experiment launched in this entry
- Decision:
  - treat the current selector difficulty as a downstream symptom, not the
    primary root cause
  - do not keep expanding proxy terms in `compute_epoch_quality(...)` as the
    main long-term fix
  - move the next mechanism work toward an objective-aligned denoising path
- Why:
  - current training optimizes denoising reconstruction in the clean-target
    space, while exported checkpoint selection still depends on a separate
    Patel-based proxy
  - therefore training, selection, and final GT evaluation are not the same
    objective
  - this mismatch forces the selector to behave like a correction layer for the
    training loop
- Current reading:
  - `sim3` / `sim4` mainly expose retention drift:
    - a good directional checkpoint appears and later degrades
  - `fMRI.csv` mainly exposes selector mismatch:
    - the branch can reach a useful later graph, but the proxy can still export
      an earlier worse epoch
  - these are different failure modes, but both become easier only if the main
    denoising objective becomes more direction-sensitive
- Mechanism implication:
  - the intended end state is:
    - when denoising is best under the training objective, the exported causal
      graph should also tend to be best
  - to approach that, the main denoising path must make wrong direction
    materially worse, instead of relying on Patel direction terms as auxiliary
    losses beside the shared loop
  - the most plausible route remains a constrained causal-lag denoising path:
    - use lagged candidate-parent information inside the main denoiser
    - let exported adjacency participate directly in that path
    - keep Patel as prior / candidate constraint / reliability signal, not as
      the sole downstream selector teacher
- Selector implication:
  - selector logic will still exist, but it should shrink back to ordinary
    checkpoint selection on an aligned validation objective
  - it should no longer carry the primary burden of correcting an objective
    mismatch after training
- Keep / drop:
  - keep the selector audit tooling
  - keep the fixed-support selector bug fix
  - drop "stronger Patel-like proxy terms" as the main next strategy
  - promote objective alignment to the next mechanism-line design target

### 2026-03-29 - Minimal causal-lag main-branch plumbing + smoke

- Branch line:
  - mechanism
- Dataset:
  - `sim2.csv`
- Exact config delta:
  - added an opt-in causal-lag reconstruction term inside the main denoising
    branch of `GraphExp/main_structure_learning.py`
  - new CLI:
    - `--causal_lag_main_weight`
    - `--causal_lag_main_aggregation`
    - `--causal_lag_main_softmax_temp`
    - `--causal_lag_main_lags`
    - `--causal_lag_main_lag_weights`
  - implementation details:
    - reuse exported causal adjacency for lagged candidate-parent aggregation
    - reconstruct each node's future from lagged source-node signals using the
      current learned graph
    - inject this term into the main branch rather than the selector proxy
    - add per-epoch forward-vs-reverse causal-lag diagnostics to
      `quality_history.csv`
    - add config export fields for the causal-lag main settings
  - ancillary compatibility fix:
    - `compute_noise_guide_probe_diagnostics(...)` now unwraps tuple returns
      defensively so the existing probe does not block mechanism-line runs
- Smoke command:
  - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\sim2.csv --device cpu --epochs 1 --pretrain_epochs 0 --skip_pretrain --subject_limit 2 --time_limit 20 --top_k_edges 5 --selection_start_epoch 1 --log_interval 1 --structure_parameterization support_direction --fixed_support_mask_mode maxgap_kappa --direction_init_mode random --structure_init_mode patel_kappa --structure_init_scale 0.5 --adj_activation sigmoid --directional_prior_mode patel --directional_schedule plateau --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratio 0.01 --lambda_l1 0.02 --optimizer_step_mode subject --main_loss_weight 1.0 --selection_agreement_weight 0.0 --direction_lr_multiplier 1.0 --freeze_direction_after_epoch 30 --causal_lag_main_weight 0.25 --causal_lag_main_lags 1,2 --causal_lag_main_aggregation mean`
- Result:
  - pass
  - run dir:
    - `GraphExp/results/run_20260329_220016`
  - confirmed:
    - training runs end-to-end with the new causal-lag main branch enabled
    - `quality_history.csv` records:
      - `causal_lag_main_raw`
      - `causal_lag_main_weight`
      - `causal_lag_main_weighted`
      - forward / reverse causal-lag diagnostics
    - `config.npy` records the causal-lag main settings
  - smoke readout:
    - `CausalLagMain(raw/w) = 0.4086 / 0.1135`
    - `clean_fwd = 0.4086`
    - `clean_rev = 0.4086`
    - `delta(rev-fwd) = +0.0000`
- Interpretation:
  - this run validates plumbing only
  - the causal-lag main path is now available as an isolated mechanism-line
    experiment without changing frozen-mainline defaults
  - however, the `1`-epoch smoke run is not evidence that the objective is
    already direction-sensitive
  - in this tiny smoke:
    - the learned graph remained near-zero / near-degenerate
    - therefore forward-vs-reverse causal-lag losses were indistinguishable
  - the next meaningful test is no longer "does the code run"
  - it is:
    - whether a nontrivial training run can make
      `reverse_loss - forward_loss > 0`
      while also improving GT direction metrics
- Keep / drop:
  - keep the implementation
  - keep forward-vs-reverse diagnostics as the primary readout for this branch
  - do not over-interpret the smoke run as mechanism evidence yet

### 2026-03-30 - Short paired mechanism pilot on `sim4` (`seed=11`, `20` epochs)

- Branch line:
  - mechanism
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - compare the frozen reference branch against the minimal causal-lag
    main-branch variant under the same short-run setup
  - primary readout:
    - whether forward-vs-reverse causal-lag reconstruction starts preferring the
      learned forward direction
  - secondary readout:
    - whether GT direction metrics move in the same direction
- Shared setup:
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
  - `epochs = 20`
  - `seed = 11`
  - `selector_audit_gt_path = ..\fMRI_dataset\h4.txt`
  - `pretrain_checkpoint = .\results\run_20260310_185625\pretrained_encoder.pt`
- Baseline run:
  - run dir:
    - `GraphExp/results/run_20260330_091421`
  - config delta:
    - `causal_lag_main_weight = 0.0`
  - selector audit summary:
    - best GT epoch = `16`
    - best GT strict-F1@`eps=0` = `0.836066`
    - exported epoch = `9`
    - exported strict-F1@`eps=0` = `0.786885`
    - final epoch = `20`
    - final strict-F1@`eps=0` = `0.754098`
  - key checkpoints:
    - epoch `9`:
      - strict-F1@`eps=0` = `0.786885`
      - GT margin median = `+0.044854`
    - epoch `16`:
      - strict-F1@`eps=0` = `0.836066`
      - GT margin median = `+0.029528`
- Causal-lag variant:
  - run dir:
    - `GraphExp/results/run_20260330_091808`
  - config delta:
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - `causal_lag_main_aggregation = mean`
  - selector audit summary:
    - best GT epoch = `10`
    - best GT strict-F1@`eps=0` = `0.836066`
    - exported epoch = `9`
    - exported strict-F1@`eps=0` = `0.836066`
    - final epoch = `20`
    - final strict-F1@`eps=0` = `0.770492`
  - causal-lag diagnostics:
    - delta definition:
      - `reverse_loss - forward_loss`
    - across `20` epochs:
      - mean delta = `+0.002928`
      - min / max delta = `-0.003344 / +0.006144`
      - positive epochs = `15 / 20`
    - high-strict epochs:
      - epoch `9`:
        - strict-F1@`eps=0` = `0.836066`
        - GT margin median = `+0.046238`
        - delta = `+0.006144`
      - epoch `10`:
        - strict-F1@`eps=0` = `0.836066`
        - GT margin median = `+0.047956`
        - delta = `+0.005618`
      - epoch `16`:
        - strict-F1@`eps=0` = `0.836066`
        - GT margin median = `+0.032572`
        - delta = `+0.005651`
- Interpretation:
  - this is the first positive mechanism-line signal worth keeping
  - compared with the same short baseline run:
    - the causal-lag branch reached the same best strict-F1 earlier
    - the exported epoch improved from:
      - `0.786885 -> 0.836066`
    - final strict-F1 improved modestly:
      - `0.754098 -> 0.770492`
  - more importantly, the causal-lag diagnostic no longer stayed neutral
    - forward reconstruction was usually better than reverse reconstruction
    - the strongest positive deltas appeared on the strongest GT-strict epochs
  - however, the result is still only a pilot-level positive
    - the effect size is small
    - failure mode remains `symmetric_collapse`
    - the branch has not yet produced a decisive final-epoch mechanism change
  - current best reading:
    - adding lagged candidate-parent reconstruction into the main branch may be
      starting to make the objective weakly direction-sensitive
    - but the present `mean` aggregation and short training horizon are not yet
      strong enough to claim a robust mechanism repair
- Keep / drop:
  - keep the causal-lag main branch
  - keep the forward-vs-reverse diagnostic as the mechanism-line gate
  - keep this result as a weak positive
  - next follow-up should test stronger edge-specific contrast inside the same
    branch before adding any new loss family
    - most likely:
      - `causal_lag_main_aggregation = softmax`
      - or a longer same-seed run to see whether the positive delta survives
        beyond the short pilot

### 2026-03-30 - `softmax` causal-lag follow-up on `sim4` (`seed=11`, `20` epochs)

- Branch line:
  - mechanism
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - test whether stronger parent competition inside the causal-lag main branch
    improves the weak positive signal seen under `aggregation = mean`
- Important implementation correction before interpreting this follow-up:
  - the first `softmax` attempt revealed a structural bug in the aggregation
    helper under `support_direction`
  - the old `softmax` path was reading `get_structure_logits()`
  - under `support_direction`, that tensor is symmetric support, so reversing the
    causal direction could not change the softmax scores materially
  - consequence:
    - the initial run
      - `GraphExp/results/run_20260330_092726`
    - showed degenerate `reverse - forward = 0` diagnostics
    - that run is invalid as mechanism evidence and should not be used for
      interpretation
  - code fix:
    - the `softmax` aggregation path now uses the exported causal adjacency with
      support masking, so reversed direction really changes the aggregation
      weights
- Valid rerun:
  - run dir:
    - `GraphExp/results/run_20260330_093647`
  - config delta versus the earlier `mean` pilot:
    - `causal_lag_main_aggregation: mean -> softmax`
    - `causal_lag_main_softmax_temp = 1.0`
    - keep:
      - `causal_lag_main_weight = 0.25`
      - `causal_lag_main_lags = 1,2`
- Result:
  - selector audit summary:
    - best GT epoch = `8`
    - best GT strict-F1@`eps=0` = `0.819672`
    - exported epoch = `9`
    - exported strict-F1@`eps=0` = `0.786885`
    - final epoch = `20`
    - final strict-F1@`eps=0` = `0.721311`
  - causal-lag diagnostics:
    - mean delta:
      - `reverse_loss - forward_loss = +0.00004955`
    - min / max delta:
      - `-0.000357 / +0.000466`
    - positive epochs:
      - `14 / 20`
  - comparison against the valid `mean` pilot
    - `GraphExp/results/run_20260330_091808`
    - `mean` branch:
      - best GT strict-F1@`eps=0` = `0.836066`
      - exported strict-F1@`eps=0` = `0.836066`
      - final strict-F1@`eps=0` = `0.770492`
      - delta mean = `+0.00292840`
    - `softmax` branch:
      - best GT strict-F1@`eps=0` = `0.819672`
      - exported strict-F1@`eps=0` = `0.786885`
      - final strict-F1@`eps=0` = `0.721311`
      - delta mean = `+0.00004955`
- Interpretation:
  - after fixing the aggregation semantics, `softmax` no longer degenerates, but
    it does **not** improve this branch
  - compared with the `mean` causal-lag run, `softmax` is worse on all of the
    primary short-pilot readouts:
    - lower best GT strict-F1
    - worse exported checkpoint
    - worse final strict-F1
    - much weaker forward-vs-reverse preference
  - this means the next lever is **not** simply "make parent competition harder"
    using the current `softmax temp=1.0` path
  - current best reading:
    - the weak positive mechanism signal is real only for the present `mean`
      branch
    - `softmax` contrast in this form erodes that signal rather than
      strengthening it
- Keep / drop:
  - keep the `softmax` aggregation bug fix
  - drop `causal_lag_main_aggregation = softmax` at `temp=1.0` as the next
    mechanism-line default
  - keep `aggregation = mean` as the current best causal-lag main setting
  - next follow-up should return to the `mean` branch and test either:
    - a longer same-seed run
    - or a weight sweep on `causal_lag_main_weight`

### 2026-03-30 - Longer same-seed `mean` mechanism tracking on `sim4` (`seed=11`, `40` epochs)

- Branch line:
  - mechanism
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - test whether the weak positive `mean`-branch signal survives past the
    `freeze_direction_after_epoch = 30` point
  - use a matched `40`-epoch baseline as control
- Matched baseline:
  - run dir:
    - `GraphExp/results/run_20260330_095819`
  - config delta:
    - `causal_lag_main_weight = 0.0`
  - selector audit summary:
    - best GT epoch = `13`
    - best GT strict-F1@`eps=0` = `0.819672`
    - exported epoch = `9`
    - exported strict-F1@`eps=0` = `0.803279`
    - final epoch = `40`
    - final strict-F1@`eps=0` = `0.770492`
- Mean causal-lag run:
  - run dir:
    - `GraphExp/results/run_20260330_100529`
  - config delta:
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - `causal_lag_main_aggregation = mean`
  - selector audit summary:
    - best GT epoch = `26`
    - best GT strict-F1@`eps=0` = `0.868852`
    - exported epoch = `9`
    - exported strict-F1@`eps=0` = `0.803279`
    - final epoch = `40`
    - final strict-F1@`eps=0` = `0.803279`
  - causal-lag diagnostics:
    - all epochs:
      - mean delta:
        - `reverse_loss - forward_loss = +0.003964`
      - min / max:
        - `-0.003344 / +0.006184`
      - positive epochs:
        - `35 / 40`
    - pre-freeze (`epoch <= 30`):
      - mean delta = `+0.003633`
    - post-freeze (`epoch > 30`):
      - mean delta = `+0.004958`
      - positive epochs = `10 / 10`
    - selected checkpoints:
      - epoch `9`:
        - strict-F1@`eps=0` = `0.803279`
        - GT margin median = `+0.047875`
        - delta = `+0.006184`
      - epoch `26`:
        - strict-F1@`eps=0` = `0.868852`
        - GT margin median = `+0.024849`
        - delta = `+0.004835`
      - epoch `40`:
        - strict-F1@`eps=0` = `0.803279`
        - GT margin median = `+0.016208`
        - delta = `+0.004886`
        - direction branch frozen = `1`
- Interpretation:
  - this is the strongest positive mechanism-line evidence so far
  - compared with the matched `40`-epoch baseline:
    - best GT strict-F1 improved:
      - `0.819672 -> 0.868852`
    - final strict-F1 improved:
      - `0.770492 -> 0.803279`
    - the forward-vs-reverse diagnostic stayed positive through the whole late
      phase and remained positive after the direction branch froze
  - this means the `mean` causal-lag branch is no longer just producing a short
    transient
    - it is sustaining a weak-but-stable forward preference in the main
      objective even after epoch `30`
  - however, this run also sharpens a remaining split diagnosis:
    - mechanism quality improved
    - selector alignment did not
  - the best GT epoch moved to `26`, but export still stayed at `9`
    - exported strict-F1 remained `0.803279`
    - best GT strict-F1 reached `0.868852`
  - so the current reading is:
    - objective alignment is improving
    - but the checkpoint proxy is still not tracking the GT-best late epoch on
      this branch
- Keep / drop:
  - keep `causal_lag_main_aggregation = mean` as the active mechanism-line
    setting
  - keep longer-run tracking across the freeze point
  - keep the judgment that mechanism and selector are still partially decoupled
  - next follow-up should stay on the same `mean` branch and test:
    - a small weight sweep on `causal_lag_main_weight`
    - or a selector-only audit on this improved branch specifically

### 2026-03-30 - Selector-only audit on the improved `mean` branch (`run_20260330_100529`)

- Branch line:
  - selection
- Scope:
  - offline analysis only
  - no new training run launched in this entry
- Dataset / source run:
  - `sim4.csv`
  - `h4.txt`
  - source run:
    - `GraphExp/results/run_20260330_100529`
- Objective:
  - explain why the selector still exports epoch `9` even though the mechanism
    branch reaches a better GT epoch later (`26`)
- Key epoch comparison:
  - exported epoch `9`:
    - proxy score = `0.616264`
    - strict-F1@`eps=0` = `0.803279`
    - GT margin median = `+0.047875`
    - `dir_margin = 0.114691`
    - `agreement_score = 0.745902`
    - `causal_lag_diag_reverse_minus_forward = +0.006184`
  - GT-best epoch `26`:
    - proxy score = `0.573915`
    - strict-F1@`eps=0` = `0.868852`
    - GT margin median = `+0.024849`
    - `dir_margin = 0.039093`
    - `agreement_score = 0.745902`
    - `causal_lag_diag_reverse_minus_forward = +0.004835`
- Rank / trend readout:
  - GT-best epoch `26` is only proxy-score rank `20`
  - GT-best epoch `28` is proxy-score rank `22`
  - exported epoch `9` is proxy-score rank `1`
  - within eligible epochs (`epoch >= 6`):
    - correlation(proxy score, GT strict-F1@`eps=0`) = `-0.088446`
    - correlation(dir_margin, GT strict-F1@`eps=0`) = `-0.096661`
    - correlation(causal-lag delta, GT strict-F1@`eps=0`) = `+0.600838`
  - within the stronger subset (`strict-F1@eps=0 >= 0.75`):
    - correlation(proxy score, GT strict-F1@`eps=0`) = `-0.402417`
    - correlation(dir_margin, GT strict-F1@`eps=0`) = `-0.404259`
    - correlation(causal-lag delta, GT strict-F1@`eps=0`) = `+0.260444`
- Interpretation:
  - on this improved branch, the selector is now missing the late better epoch
    for a concrete reason:
    - `skeleton_overlap` and `density_factor` are effectively constant
    - `agreement_score` is also flat across the main competitive region
    - so the proxy is mainly ranking epochs by early `dir_margin` /
      `global_asymmetry`
  - that was acceptable when higher margin roughly tracked quality
  - but after the mechanism improvement, the best GT epochs are later lower-
    margin but cleaner-direction epochs
  - therefore the current selector is systematically over-rewarding "early high
    margin" and under-rewarding "later better direction"
  - importantly, the new causal-lag mechanism diagnostic is more aligned with GT
    than the current proxy score on this branch
- Keep / drop:
  - keep the diagnosis that selector mismatch remains after the mechanism gain
  - drop any interpretation that the remaining export miss is mainly a training
    failure on this branch
  - promote selector work that uses the new objective-aligned diagnostics rather
    than further strengthening the old margin-heavy proxy

### 2026-03-30 - Minimal causal-lag-aware selector on the improved `mean` branch

- Branch line:
  - selection
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - implement the smallest selector-only replacement that stops rewarding
    positive `dir_margin` monotonically
  - test whether the improved mechanism branch can be exported correctly without
    changing training
- Exact config delta:
  - code change in `GraphExp/main_structure_learning.py`
    - add opt-in selector score mode:
      - `legacy`
      - `causal_lag_composite`
    - add new CLI:
      - `--selection_score_mode`
      - `--selection_soft_agreement_weight`
      - `--selection_causal_lag_weight`
      - `--selection_margin_penalty_weight`
    - composite definition:
      - `score = w_soft * agreement_soft_score + w_lag * causal_lag_diag_reverse_minus_forward - w_margin * dir_margin`
    - keep all guardrails unchanged:
      - skeleton overlap retention
      - density guardrail neutralization under fixed support
  - tested selector setting:
    - `selection_score_mode = causal_lag_composite`
    - `selection_soft_agreement_weight = 0.20`
    - `selection_causal_lag_weight = 1.0`
    - `selection_margin_penalty_weight = 0.05`
  - matched run command keeps the full training setup of
    `GraphExp/results/run_20260330_100529`
    unchanged apart from the selector arguments above
- Offline replay on the existing improved branch:
  - source run:
    - `GraphExp/results/run_20260330_100529`
  - legacy export:
    - epoch `9`
    - strict-F1@`eps=0` = `0.803279`
  - best GT epoch:
    - epoch `26`
    - strict-F1@`eps=0` = `0.868852`
  - replaying the composite score on the saved `quality_history.csv` changes the
    ranking to:
    - rank `1`: epoch `26`, strict-F1@`eps=0` = `0.868852`, composite score = `0.185882`
    - rank `2`: epoch `28`, strict-F1@`eps=0` = `0.868852`, composite score = `0.185846`
    - rank `3`: epoch `9`, strict-F1@`eps=0` = `0.803279`, composite score = `0.185702`
  - correlation over eligible guarded epochs:
    - legacy proxy vs GT strict-F1@`eps=0` = `-0.088446`
    - composite proxy vs GT strict-F1@`eps=0` = `+0.709049`
  - this confirms the selector miss on `run_20260330_100529` can be repaired by
    re-ranking the already trained checkpoints; it does **not** require another
    training loss
- Matched live rerun:
  - run dir:
    - `GraphExp/results/run_20260330_105521`
  - result:
    - best GT epoch = `23`
    - best GT strict-F1@`eps=0` = `0.868852`
    - exported epoch = `26`
    - exported strict-F1@`eps=0` = `0.852459`
    - final epoch = `40`
    - final strict-F1@`eps=0` = `0.786885`
    - exported vs best GT gap:
      - `-0.016393`
  - same rerun, if ranked by the logged legacy proxy instead:
    - legacy-best epoch = `9`
    - legacy-best strict-F1@`eps=0` = `0.803279`
  - correlation over eligible guarded epochs in this rerun:
    - legacy proxy vs GT strict-F1@`eps=0` = `+0.077985`
    - composite proxy vs GT strict-F1@`eps=0` = `+0.753953`
  - ranking summary in this rerun:
    - composite top `3` epochs:
      - `26`, `28`, `9`
    - legacy top `3` epochs:
      - `9`, `8`, `10`
- Interpretation:
  - this is the first selector-line result strong enough to keep
  - the important point is not just that the exported epoch moved
  - it is that the move happened for the **right reason**:
    - the selector stopped treating larger early `dir_margin` as automatic
      evidence of a better checkpoint
    - it started preferring later epochs with better objective-aligned causal-lag
      behavior and lower overconfident margin
  - the offline replay is the key isolation test:
    - on the exact saved checkpoints from `run_20260330_100529`, the new score
      already picks the GT-best late epoch
    - therefore this is genuinely a selector repair, not a hidden mechanism
      change
  - the live rerun still shows small late-epoch competition between `23`, `26`,
    and `28`
    - so the selector is not globally solved yet
    - but the old systematic "export epoch `9` because margin is high" failure
      is largely removed on this branch
- Keep / drop:
  - keep `selection_score_mode = causal_lag_composite` as the active next
    selector-line baseline for the improved `mean` mechanism branch
  - keep `legacy` as the frozen comparison mode
  - drop any next step that strengthens positive margin reward inside the
    selector
  - next selector follow-up should be:
    - either a small weight sweep around `0.20 / 1.0 / 0.05`
    - or a transfer test on `fMRI.csv` / `h1.txt` without changing training

### 2026-03-30 - `fMRI.csv` transfer pair with mechanism enabled (`seed=11`)

- Branch line:
  - selection
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Objective:
  - test whether the new selector result from `sim4` transfers to the real
    `fMRI.csv` branch
  - hold training fixed and compare:
    - legacy selector
    - causal-lag-composite selector
  - but unlike the older `fMRI.csv` audits, keep the new mechanism line enabled:
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - `causal_lag_main_aggregation = mean`
- Shared training setup:
  - start from the previously audited `fMRI.csv` branch:
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
    - `selection_agreement_weight = 0.25`
    - `selection_agreement_mode = hard_coverage`
    - `direction_lr_multiplier = 1.0`
    - `freeze_direction_after_epoch = 30`
    - `epochs = 100`
    - `seed = 11`
- Legacy selector run:
  - run dir:
    - `GraphExp/results/run_20260330_141426`
  - extra selector config:
    - `selection_score_mode = legacy`
  - selector audit summary:
    - best GT epoch = `12`
    - best GT strict-F1@`eps=0` = `1.0000`
    - exported epoch = `12`
    - exported strict-F1@`eps=0` = `1.0000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.8000`
  - compared with the older fixed-support audited branch without causal-lag
    mechanism:
    - `GraphExp/results/run_20260329_171638`
    - old exported vs best-GT gap = `-0.2000`
    - new exported vs best-GT gap = `0.0000`
- Composite selector run:
  - run dir:
    - `GraphExp/results/run_20260330_142007`
  - extra selector config:
    - `selection_score_mode = causal_lag_composite`
    - `selection_soft_agreement_weight = 0.20`
    - `selection_causal_lag_weight = 1.0`
    - `selection_margin_penalty_weight = 0.05`
  - selector audit summary:
    - best GT epoch = `12`
    - best GT strict-F1@`eps=0` = `1.0000`
    - exported epoch = `13`
    - exported strict-F1@`eps=0` = `1.0000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.8000`
  - practical gap:
    - exported vs best-GT = `0.0000`
- Score readout:
  - legacy run:
    - correlation(proxy score, GT strict-F1@`eps=0`) over eligible guarded
      epochs = `+0.818449`
    - top score epoch = `12`, strict-F1@`eps=0` = `1.0000`
  - composite run:
    - correlation(proxy score, GT strict-F1@`eps=0`) over eligible guarded
      epochs = `+0.904242`
    - top score epoch = `13`, strict-F1@`eps=0` = `1.0000`
  - note:
    - epochs `12`, `13`, `14`, `16`, `18` all lie on the same best strict-F1
      plateau under this mechanism-enabled branch
- Interpretation:
  - this transfer result does **not** show a practical selector win for the new
    composite score on `fMRI.csv`
  - instead, it shows something more important:
    - once the causal-lag main mechanism is enabled, the old severe `fMRI.csv`
      selector mismatch largely disappears
    - even the legacy selector with `selection_agreement_weight = 0.25` now
      lands directly on a best-GT epoch
  - this materially strengthens the earlier architecture judgment:
    - a substantial part of the old `fMRI.csv` selector problem was downstream
      of objective mismatch, not purely a bad checkpoint proxy
  - the composite selector is still acceptable here:
    - it stays on the same best strict-F1 plateau
    - it slightly improves score-vs-GT correlation
  - but there is no practical export gain on this branch because the mechanism
    change already repaired the damaging early-wrong-export behavior
  - the remaining problem on `fMRI.csv` is now mainly **retention / late drift**:
    - final strict-F1 remains `0.8000`
    - best strict-F1 reaches `1.0000`
- Keep / drop:
  - keep the conclusion that objective alignment can remove selector failures
    downstream
  - keep `causal_lag_main(mean)` as the more important change on the
    `fMRI.csv` branch
  - keep `causal_lag_composite` as a valid selector option, but do not claim it
    adds practical value on `fMRI.csv` yet
  - drop `fMRI.csv` as the immediate next selector battleground under the
    mechanism-enabled branch
  - next follow-up should move back to mechanism / retention:
    - small sweep on `causal_lag_main_weight`
    - or a late-phase stabilization experiment
    - but not another selector-only tweak first

### 2026-03-30 - `fMRI.csv` mechanism sweep on `causal_lag_main_weight` (`seed=11`)

- Branch line:
  - mechanism
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Objective:
  - test whether the mechanism-enabled `fMRI.csv` branch can improve retention
    by moving `causal_lag_main_weight` away from `0.25`
  - keep selector fixed to the practical baseline that already works here:
    - `selection_score_mode = legacy`
    - `selection_agreement_weight = 0.25`
- Shared setup:
  - same mechanism-enabled `fMRI.csv` branch as above
  - keep:
    - `causal_lag_main_lags = 1,2`
    - `causal_lag_main_aggregation = mean`
- Reference center point:
  - existing run:
    - `GraphExp/results/run_20260330_141426`
  - weight:
    - `causal_lag_main_weight = 0.25`
  - result:
    - best GT epoch = `12`
    - best GT strict-F1@`eps=0` = `1.0000`
    - exported epoch = `12`
    - exported strict-F1@`eps=0` = `1.0000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.8000`
    - exported vs best-GT gap = `0.0000`
    - final vs best-GT gap = `-0.2000`
- Lower weight:
  - run dir:
    - `GraphExp/results/run_20260330_143839`
  - config delta:
    - `causal_lag_main_weight = 0.15`
  - result:
    - best GT epoch = `14`
    - best GT strict-F1@`eps=0` = `1.0000`
    - exported epoch = `13`
    - exported strict-F1@`eps=0` = `0.6000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.8000`
    - exported vs best-GT gap = `-0.4000`
    - final vs best-GT gap = `-0.2000`
  - causal-lag diagnostic:
    - mean `reverse - forward` = `-0.001241`
    - post-30 mean = `-0.000815`
    - positive epochs = `10 / 100`
- Higher weight:
  - run dir:
    - `GraphExp/results/run_20260330_144408`
  - config delta:
    - `causal_lag_main_weight = 0.35`
  - result:
    - best GT epoch = `14`
    - best GT strict-F1@`eps=0` = `1.0000`
    - exported epoch = `13`
    - exported strict-F1@`eps=0` = `0.8000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.8000`
    - exported vs best-GT gap = `-0.2000`
    - final vs best-GT gap = `-0.2000`
  - causal-lag diagnostic:
    - mean `reverse - forward` = `-0.001067`
    - post-30 mean = `-0.000194`
    - positive epochs = `1 / 100`
- Comparison:
  - all three weights still reach a best GT epoch with strict-F1@`eps=0 = 1.0000`
  - none of the three weights improve final strict-F1 beyond `0.8000`
  - export behavior is strongly weight-sensitive:
    - `0.15`:
      - under-shoots badly
      - exported checkpoint drops to `0.6000`
    - `0.25`:
      - current sweet spot
      - exported checkpoint lands exactly on the best-GT epoch
    - `0.35`:
      - loses the exact best export again
      - exported checkpoint falls back to `0.8000`
  - causal-lag diagnostics do **not** improve monotonically with larger weight
    on `fMRI.csv`
    - `0.25` is the least negative overall and the only one with mildly positive
      post-30 mean
    - `0.35` is the most consistently negative
- Interpretation:
  - on this branch, `causal_lag_main_weight = 0.25` is a real local optimum
    among the tested values
  - lower weight is too weak to stabilize export
  - higher weight does not improve retention and appears to over-push the branch
    away from the narrow best-GT plateau
  - the current unresolved problem is now cleaner:
    - not "find a better selector"
    - not "turn the same weight knob harder"
    - but "why does the branch pass through a perfect epoch and still drift back
      to `0.8000` by the end"
- Keep / drop:
  - keep `causal_lag_main_weight = 0.25` as the active `fMRI.csv` mechanism
    setting
  - drop further immediate scalar sweeps on this weight as the main next move
  - next follow-up should target late-phase stabilization directly, e.g.:
    - retention after epoch `12-20`
    - post-freeze drift control
    - or a schedule / early-stop strategy tied to the improved objective, not a
      larger fixed weight

### 2026-03-30 - `fMRI.csv` late-phase stabilization isolation (`seed=11`)

- Branch line:
  - mechanism
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Objective:
  - isolate the source of the remaining late-phase drift after the
    mechanism-enabled branch already reaches a perfect epoch
  - test two hypotheses separately:
    - H1: the branch simply trains too long
    - H2: the direction branch keeps updating past the best phase and destroys
      the good solution
- Reference branch:
  - `GraphExp/results/run_20260330_141426`
  - config:
    - `causal_lag_main_weight = 0.25`
    - `freeze_direction_after_epoch = 30`
    - `epochs = 100`
  - reference outcome:
    - best GT epoch = `12`
    - exported epoch = `12`
    - final epoch = `100`
    - best / exported strict-F1@`eps=0` = `1.0000`
    - final strict-F1@`eps=0` = `0.8000`
- Experiment A: short-horizon control
  - run dir:
    - `GraphExp/results/run_20260330_155626`
  - config delta:
    - `epochs: 100 -> 20`
    - keep `freeze_direction_after_epoch = 30` unchanged
  - result:
    - best GT epoch = `12`
    - exported epoch = `12`
    - final epoch = `20`
    - best / exported strict-F1@`eps=0` = `1.0000`
    - final strict-F1@`eps=0` = `0.8000`
  - interpretation:
    - the branch has **already** left the perfect plateau by epoch `20`
    - therefore the current drift is not a purely "very late training" effect
      that only appears near epoch `100`
- Experiment B: earlier direction-branch freeze
  - run dir:
    - `GraphExp/results/run_20260330_155759`
  - config delta:
    - `freeze_direction_after_epoch: 30 -> 15`
    - keep `epochs = 100`
  - result:
    - best GT epoch = `12`
    - exported epoch = `12`
    - final epoch = `100`
    - best / exported / final strict-F1@`eps=0` = `1.0000`
    - final vs best-GT gap = `0.0000`
  - epoch trace:
    - epoch `16`:
      - direction branch frozen = `1`
      - strict-F1@`eps=0` still = `1.0000`
    - epoch `20`:
      - strict-F1@`eps=0` still = `1.0000`
    - epoch `30`, `40`, `60`, `80`, `100`:
      - strict-F1@`eps=0` all remain `1.0000`
- Comparison:
  - reference (`freeze=30`, `epochs=100`):
    - epoch `20` already dropped to `0.8000`
    - final stays `0.8000`
  - short-horizon (`epochs=20`):
    - final is also `0.8000`
  - early-freeze (`freeze=15`, `epochs=100`):
    - final remains `1.0000`
- Interpretation:
  - this is a strong localization result
  - the remaining `fMRI.csv` retention failure is **not** mainly "we trained too
    long overall"
  - it is primarily driven by continued direction-branch updates after the
    branch has already reached the good phase around epochs `12-15`
  - the decisive evidence is:
    - stopping total training at `20` does not rescue final quality
    - freezing the direction branch at `15` does rescue final quality all the
      way to epoch `100`
  - therefore the next stabilization step should be designed around
    **direction-branch retention**, not around more selector changes or larger
    causal-lag weights
- Keep / drop:
  - keep `freeze_direction_after_epoch = 15` as the current best stabilization
    baseline on `fMRI.csv`
  - drop the hypothesis that the current `fMRI.csv` degradation is mainly just
    "too many total epochs"
  - promote next follow-up toward direction-retention scheduling, e.g.:
    - verify transfer of early freeze on `sim4`
    - test a small window around `freeze = 12 / 15 / 18`
    - optionally pair early freeze with an earlier directional-loss end epoch if
      we want to separate "branch parameter updates" from "supervision window"

### 2026-03-30 - Separate late Patel supervision from late direction-branch updates (`fMRI.csv`, `seed=11`)

- Branch line:
  - mechanism
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Objective:
  - after the earlier isolation run showed that `freeze_direction_after_epoch = 15`
    stabilizes final performance, test whether the real problem is:
    - late Patel supervision itself
    - or late updates to the direction branch even after Patel supervision stops
- Shared setup:
  - keep the current best mechanism branch:
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - `causal_lag_main_aggregation = mean`
  - keep selector fixed:
    - `selection_score_mode = legacy`
    - `selection_agreement_weight = 0.25`
- Reference points:
  - ongoing-direction reference:
    - `GraphExp/results/run_20260330_141426`
    - `directional_loss_end_epoch = -1`
    - `freeze_direction_after_epoch = 30`
    - final strict-F1@`eps=0` = `0.8000`
  - early-freeze stabilization:
    - `GraphExp/results/run_20260330_155759`
    - `directional_loss_end_epoch = -1`
    - `freeze_direction_after_epoch = 15`
    - final strict-F1@`eps=0` = `1.0000`
- Experiment A: stop Patel supervision early, keep late freeze
  - run dir:
    - `GraphExp/results/run_20260330_175302`
  - config delta:
    - `directional_loss_end_epoch = 15`
    - keep `freeze_direction_after_epoch = 30`
  - result:
    - best GT epoch = `12`
    - exported epoch = `12`
    - best / exported strict-F1@`eps=0` = `1.0000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.8000`
  - interpretation:
    - merely turning off Patel supervision after epoch `15` does **not** prevent
      the later degradation
- Experiment B: stop Patel supervision early, never freeze the direction branch
  - run dir:
    - `GraphExp/results/run_20260330_180028`
  - config delta:
    - `directional_loss_end_epoch = 15`
    - `freeze_direction_after_epoch = -1`
  - result:
    - best GT epoch = `12`
    - exported epoch = `12`
    - best / exported strict-F1@`eps=0` = `1.0000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.2000`
    - failure mode = `weak_asymmetry`
  - trajectory note:
    - epoch `20` still remains at strict-F1@`eps=0 = 1.0000`
    - by epoch `80` strict-F1@`eps=0` collapses to `0.0000`
    - by epoch `100` it only recovers to `0.2000`
- Comparison:
  - `directional_loss_end_epoch = 15` + `freeze = 30`
    - final remains `0.8000`
  - `directional_loss_end_epoch = 15` + `freeze = -1`
    - final collapses to `0.2000`
  - `directional_loss_end_epoch = -1` + `freeze = 15`
    - final remains `1.0000`
  - important local trace:
    - both early-stop-supervision runs still sit at `1.0000` through epoch `20`
    - both have already fallen away from the perfect branch by epoch `30` if the
      direction branch is not frozen early
- Interpretation:
  - this is the strongest current localization result
  - the harmful late updates are **not** caused only by ongoing Patel margin
    supervision
  - even after Patel supervision is switched off at epoch `15`, the direction
    branch continues to move under the rest of the objective and this movement is
    enough to destroy the best graph
  - early freezing works because it blocks those later parameter updates
  - therefore the structural issue is sharper than "Patel supervision lasts too
    long":
    - the main denoising / causal-lag objective is still sending late gradients
      through the direction branch in a way that is misaligned with GT
  - this matches the broader judgment that the branch is passing through a good
    region and then being pulled away by a multi-objective training path whose
    later optimum is not the GT optimum
- Keep / drop:
  - keep `freeze_direction_after_epoch = 15` as the best known stabilization
    behavior
  - drop "just shorten directional supervision" as a sufficient fix
  - promote the next follow-up toward a more principled retention mechanism:
    - late-stage direction-branch LR -> 0 schedule
    - or code-level detachment of direction-branch updates from the main /
      causal-lag path after an early epoch
    - rather than more selector work or more scalar weight sweeps

### 2026-03-30 - Detach late main/causal-lag gradients from the direction gate (`fMRI.csv`, `seed=11`)

- Branch line:
  - mechanism
- Dataset:
  - `fMRI.csv`
  - `h1.txt`
- Objective:
  - test the more flexible alternative to hard early freeze:
    - keep the direction branch trainable
    - keep Patel supervision on
    - but after an early epoch, stop sending late main denoising /
      causal-lag gradients through the direction gate
  - because the codebase has evolved since the older `freeze=15` run, first make
    a **matched current-code baseline** before interpreting the detach result
- Code change:
  - added `--detach_direction_from_main_after_epoch`
  - implementation detail:
    - in `support_direction`, the adjacency used by the main denoising path and
      `causal_lag_main` can now use `direction_gate.detach()` after a chosen
      epoch
    - export / selector / audit still read the normal exported adjacency
- Shared setup:
  - `causal_lag_main_weight = 0.25`
  - `causal_lag_main_lags = 1,2`
  - `causal_lag_main_aggregation = mean`
  - `directional_loss_end_epoch = -1`
  - `selection_score_mode = legacy`
  - `selection_agreement_weight = 0.25`
  - no branch freeze in either matched run:
    - `freeze_direction_after_epoch = -1`
- Experiment A: matched baseline, no freeze, no detach
  - run dir:
    - `GraphExp/results/run_20260330_190104`
  - config delta:
    - `freeze_direction_after_epoch = -1`
    - `detach_direction_from_main_after_epoch = -1`
  - result:
    - best GT epoch = `13`
    - exported epoch = `13`
    - best / exported strict-F1@`eps=0` = `0.8000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.6000`
    - final vs best-GT gap = `-0.2000`
  - trajectory note:
    - epoch `13`: strict-F1@`eps=0 = 0.8000`
    - epoch `20`: drops to `0.6000`
    - epoch `30-80`: partially recovers to `0.8000`
    - epoch `90-100`: drops again to `0.6000`
    - `causal_lag_diag_reverse_minus_forward` drifts from `-0.0076` at epoch
      `13` to `+0.0009` at epoch `100`
- Experiment B: detach late main path, still no freeze
  - run dir:
    - `GraphExp/results/run_20260330_185256`
  - config delta:
    - `freeze_direction_after_epoch = -1`
    - `detach_direction_from_main_after_epoch = 15`
  - result:
    - best GT epoch = `13`
    - exported epoch = `13`
    - best / exported strict-F1@`eps=0` = `0.8000`
    - final epoch = `100`
    - final strict-F1@`eps=0` = `0.8000`
    - final vs best-GT gap = `0.0000`
  - trajectory note:
    - epoch `16`: `detach_direction_from_main_active = 1`
    - epoch `16-100`: `direction_branch_frozen = 0` throughout
    - epoch `20`: strict-F1@`eps=0` stays at `0.8000`
    - epoch `90-100`: strict-F1@`eps=0` still stays at `0.8000`
    - `causal_lag_diag_reverse_minus_forward` stays near `-0.0058` through
      epoch `100`, instead of drifting back to `0`
- Comparison:
  - up to the best epoch (`13`), the matched baseline and detach run are
    identical:
    - same best epoch
    - same best strict-F1@`eps=0 = 0.8000`
  - the difference appears only after detach activates:
    - epoch `20`: baseline `0.6000`, detach `0.8000`
    - epoch `90`: baseline `0.6000`, detach `0.8000`
    - epoch `100`: baseline `0.6000`, detach `0.8000`
  - because the detach run never freezes the direction branch, this is not
    "freeze in disguise"
- Interpretation:
  - this is the clearest current evidence that the harmful late updates are
    specifically coming from the **main / causal-lag gradient path through the
    direction gate**
  - once that path is cut after the good phase, the best-final gap disappears
    even though:
    - the direction branch remains trainable
    - Patel supervision remains enabled
  - therefore hard freeze was a workaround, but not the only workable one:
    - a more selective gradient-path intervention can stabilize retention
      without globally freezing the branch
  - at the same time, this experiment does **not** raise the best-GT ceiling in
    the current code state:
    - best remains `0.8000`
    - so detach is a **retention fix**, not yet a stronger early-learning fix
- Keep / drop:
  - keep `detach_direction_from_main_after_epoch = 15` as the current more
    extensible stabilization baseline than hard freeze
  - drop the claim that "only freezing the whole direction branch can stop the
    drift"
  - next follow-up should move to raising the best-GT ceiling under this
    detached schedule, not back to selector tweaking or more scalar sweeps

### 2026-03-30 - `sim4` transfer of late main-path detach under the current mechanism+selector protocol (`seed=11`, `40` epochs)

- Branch line:
  - mechanism
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - test whether the `fMRI.csv` retention fix
    - `detach_direction_from_main_after_epoch`
    transfers to `sim4`
  - and explicitly inspect the full epoch-level strict-F1 trajectory, because
    `sim4` had already shown later GT-best epochs in earlier runs
- Shared setup:
  - use the current `sim4` mechanism+selector stack:
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - `causal_lag_main_aggregation = mean`
    - `selection_score_mode = causal_lag_composite`
    - `selection_soft_agreement_weight = 0.20`
    - `selection_causal_lag_weight = 1.0`
    - `selection_margin_penalty_weight = 0.05`
  - keep:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `direction_init_mode = random`
    - `structure_init_mode = patel_kappa`
    - `structure_init_scale = 0.5`
    - `directional_prior_mode = patel`
    - `directional_schedule = plateau`
    - `directional_kappa_gate = True`
    - `directional_kappa_gate_quantile = 0.50`
    - `directional_target_ratio = 0.01`
    - `directional_loss_end_epoch = -1`
    - `direction_lr_multiplier = 1.0`
    - `selection_agreement_weight = 0.0`
    - `epochs = 40`
    - `pretrain_checkpoint = .\results\run_20260310_185625\pretrained_encoder.pt`
    - `selector_audit_gt_path = ..\fMRI_dataset\h4.txt`
  - all new runs below use:
    - `freeze_direction_after_epoch = -1`
    - to isolate detach without global branch freezing
- Reference point from the existing frozen branch:
  - run dir:
    - `GraphExp/results/run_20260330_105521`
  - config:
    - `freeze_direction_after_epoch = 30`
    - `detach_direction_from_main_after_epoch = -1`
  - summary:
    - best GT epoch = `23`
    - best GT strict-F1@`eps=0` = `0.868852`
    - exported epoch = `26`
    - exported strict-F1@`eps=0` = `0.852459`
    - final strict-F1@`eps=0` = `0.786885`
- Experiment A: matched no-freeze baseline
  - run dir:
    - `GraphExp/results/run_20260330_192940`
  - config delta:
    - `freeze_direction_after_epoch = -1`
    - `detach_direction_from_main_after_epoch = -1`
  - result:
    - best GT epoch = `25`
    - best GT strict-F1@`eps=0` = `0.885246`
    - exported epoch = `10`
    - exported strict-F1@`eps=0` = `0.819672`
    - final epoch = `40`
    - final strict-F1@`eps=0` = `0.786885`
    - final vs best-GT gap = `-0.098361`
  - strict-F1 trajectory highlights:
    - epoch `10`: `0.819672`
    - epoch `15`: `0.836066`
    - epoch `20`: `0.786885`
    - epoch `25`: `0.885246` (GT-best)
    - epoch `30`: `0.786885`
    - epoch `40`: `0.786885`
  - reading:
    - `sim4` late improvement is real
    - the GT-best region sits around epochs `23-26`, not around epoch `10-15`
- Experiment B: detach after epoch `15`
  - run dir:
    - `GraphExp/results/run_20260330_194935`
  - config delta:
    - `detach_direction_from_main_after_epoch = 15`
  - result:
    - best GT epoch = `10`
    - best GT strict-F1@`eps=0` = `0.819672`
    - exported epoch = `9`
    - exported strict-F1@`eps=0` = `0.803279`
    - final epoch = `40`
    - final strict-F1@`eps=0` = `0.819672`
    - final vs best-GT gap = `0.000000`
  - strict-F1 trajectory highlights:
    - epoch `16`: `detach_direction_from_main_active = 1`
    - epoch `20`: `0.819672`
    - epoch `25`: `0.819672`
    - epoch `30`: `0.819672`
    - epoch `40`: `0.819672`
  - comparison to the matched baseline:
    - retention is stabilized
    - but the late improvement region around epoch `25` disappears entirely
    - best GT drops:
      - `0.885246 -> 0.819672`
  - reading:
    - on `sim4`, `detach_after_15` is too early
    - it prevents the harmful late drift, but it also cuts off useful late
      learning
- Experiment C: detach after epoch `25`
  - run dir:
    - `GraphExp/results/run_20260330_200644`
  - config delta:
    - `detach_direction_from_main_after_epoch = 25`
  - result:
    - best GT epoch = `25`
    - best GT strict-F1@`eps=0` = `0.868852`
    - exported epoch = `10`
    - exported strict-F1@`eps=0` = `0.836066`
    - final epoch = `40`
    - final strict-F1@`eps=0` = `0.852459`
    - final vs best-GT gap = `-0.016393`
  - strict-F1 trajectory highlights:
    - epoch `26`: `detach_direction_from_main_active = 1`
    - epoch `25`: `0.868852` (GT-best)
    - epoch `30`: `0.852459`
    - epoch `35`: `0.852459`
    - epoch `40`: `0.852459`
  - comparison to the matched baseline:
    - keeps almost all of the late best-epoch gain:
      - `0.885246 -> 0.868852`
    - while greatly reducing late degradation:
      - final `0.786885 -> 0.852459`
    - exported checkpoint also improves over the no-freeze baseline:
      - `0.819672 -> 0.836066`
- Comparison:
  - no-freeze baseline:
    - highest ceiling (`0.885246`)
    - worst late drift (`final-best = -0.098361`)
  - detach `15`:
    - fully removes drift
    - but cuts off the late-learning window and lowers the ceiling too much
  - detach `25`:
    - preserves the late-learning window
    - nearly removes the late drift
    - gives the best overall train/final tradeoff among the new no-freeze runs
  - frozen reference (`freeze=30`):
    - exported checkpoint still best (`0.852459`)
    - but final retention is weaker than `detach_after_25`
      - `0.786885` vs `0.852459`
- Interpretation:
  - this transfer result is qualitatively different from `fMRI.csv`
  - on `sim4`, the useful learning phase lasts much longer
    - the GT-best zone is late (`~23-26`)
  - therefore the detach mechanism **does** transfer, but not with the same
    timing:
    - `15` is too early
    - `25` is much better matched to the task
  - the main structural conclusion still holds:
    - once the late-learning phase has passed, cutting the main / causal-lag
      gradient path into the direction gate improves retention
  - but `sim4` also shows that detach timing must be dataset-dependent
  - selector mismatch is not fully solved here:
    - even with `detach_after_25`, export still stays at epoch `10` while the
      GT-best epoch is `25`
    - so this entry is mainly a training-retention result, not a selector fix
- Keep / drop:
  - keep `detach_direction_from_main_after_epoch` as a transferable mechanism
    for retention control
  - drop `detach_after_15` as a universal default across datasets
  - keep `detach_after_25` as the current best `sim4` no-freeze stabilization
    point
  - next follow-up should be one of:
    - a small detach-timing window on `sim4` around `22 / 25 / 28`
    - or pairing `detach_after_25` with selector work so export can follow the
      late GT-best region

### 2026-03-30 - Recheck the Patel-only ceiling against the new `sim4` detach result

- Branch line:
  - mechanism
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - quantify how much the current DDM branch is actually contributing beyond the
    Patel prior itself
  - because the next step should differ depending on whether current gains are:
    - mostly "protect Patel better"
    - or real improvement beyond the prior ceiling
- Method:
  - no training
  - recompute Patel on `sim4.csv`
  - build the simple heuristic:
    - `top-k kappa + tau_sign`
    - `k = |GT edges| = 61`
  - evaluate with the same strict directional top-`k` metric
- Result:
  - recomputed `sim4 top-k kappa + tau_sign strict-F1 = 0.770492`
  - this matches the older tracker entry (`~0.7705`)
- Comparison to the new `sim4` runs:
  - matched no-freeze baseline:
    - best = `0.885246`
    - final = `0.786885`
  - `detach_after_25`:
    - best = `0.868852`
    - exported = `0.836066`
    - final = `0.852459`
- Interpretation:
  - Claude's broader framing is directionally right:
    - the Patel ceiling question is important
  - but on `sim4`, this diagnostic is no longer hypothetical
  - the current branch is **already** above the Patel-only baseline by a clear
    margin:
    - best gain over Patel-only:
      - `0.885246 - 0.770492 = +0.114754`
    - `detach_after_25` final gain over Patel-only:
      - `0.852459 - 0.770492 = +0.081967`
    - exported gain over Patel-only:
      - `0.836066 - 0.770492 = +0.065574`
  - so the right reading is:
    - current work is not merely preserving a Patel-level solution
    - DDM + causal-lag + learned optimization is adding a real margin beyond
      the raw `kappa/tau` heuristic
  - that makes selector / retention work still meaningful:
    - they are not just polishing a fixed Patel answer
    - they are helping preserve a solution that is already stronger than Patel
- Keep / drop:
  - keep the Patel-only baseline as a required comparison point when judging new
    `sim4` branches
  - drop the idea that current `sim4` gains are only re-packaged Patel
  - next follow-up should optimize preservation/export of the already-above-
    Patel learned solution, not revert to a Patel-only framing

### 2026-03-30 - Selector-only follow-up on `sim4 detach_after_25` (`seed=11`, `40` epochs)

- Branch line:
  - selection
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - test the smallest selector-only modification on top of the best current
    no-freeze stabilization run:
    - `detach_direction_from_main_after_epoch = 25`
  - specifically:
    - reduce the soft-agreement dominance in the current composite selector
    - see whether export can move from the early epoch `10` toward the later
      higher-quality plateau
- Starting point:
  - source run:
    - `GraphExp/results/run_20260330_200644`
  - current selector setting:
    - `selection_score_mode = causal_lag_composite`
    - `selection_soft_agreement_weight = 0.20`
    - `selection_causal_lag_weight = 1.0`
    - `selection_margin_penalty_weight = 0.05`
  - source result:
    - best GT epoch = `25`
    - best GT strict-F1@`eps=0` = `0.868852`
    - exported epoch = `10`
    - exported strict-F1@`eps=0` = `0.836066`
    - final strict-F1@`eps=0` = `0.852459`
- Offline replay before launching the rerun:
  - replaying the saved `quality_history.csv` from
    `GraphExp/results/run_20260330_200644`
    suggested that lowering soft-agreement weight strongly could push selection
    to the later `0.852459` plateau
  - simple weight-grid readout:
    - current weights (`0.20 / 1.0 / 0.05`) pick epoch `10`
    - several lower-soft settings pick epoch `35`
    - but even offline, the current composite family still did **not** recover
      the GT-best epoch `25`
- Matched live rerun:
  - run dir:
    - `GraphExp/results/run_20260330_205921`
  - exact config delta:
    - `selection_soft_agreement_weight: 0.20 -> 0.05`
    - keep:
      - `selection_causal_lag_weight = 1.0`
      - `selection_margin_penalty_weight = 0.05`
      - `detach_direction_from_main_after_epoch = 25`
      - all training-side settings unchanged
- Result:
  - best GT epoch = `25`
  - best GT strict-F1@`eps=0` = `0.868852`
  - exported epoch = `20`
  - exported strict-F1@`eps=0` = `0.786885`
  - final epoch = `40`
  - final strict-F1@`eps=0` = `0.836066`
  - exported vs best-GT gap = `-0.081967`
  - final vs best-GT gap = `-0.032787`
- Additional offline sweep on the new rerun's own trajectory:
  - running the same composite-family weight sweep over
    `GraphExp/results/run_20260330_205921/quality_history.csv`
    showed:
    - the best current-family selector choices only reach the later
      `0.836066` plateau (e.g. epoch `35`)
    - none of the swept weights recover the GT-best epoch `25`
      with strict-F1 `0.868852`
- Interpretation:
  - this is a negative but useful selector result
  - merely reducing `soft_agreement_weight` inside the **current composite
    family** is not a reliable fix
  - two separate issues are now visible:
    - selector-only reruns are not perfectly reproducible end-to-end, even when
      the training objective is unchanged
    - more importantly, on this trajectory, the current weighted-sum composite
      family itself cannot identify the GT-best epoch
  - so the remaining export gap on `sim4 detach_after_25` is no longer best
    framed as:
    - "find a slightly better scalar weight"
  - it is better framed as:
    - "the selector family lacks the right discriminative signal for the late
      best region"
- Keep / drop:
  - keep `detach_after_25` as the best current no-freeze stabilization result
  - drop further scalar sweeps within the current
    `soft_agreement + causal_lag_delta - margin_penalty` selector family as the
    main next move
  - next follow-up should require a new selector signal or selector structure,
    not more weight tuning on the existing one

### 2026-03-30 - Selector-line plumbing for cross-subject causal-lag + parent-entropy signals

- Branch line:
  - selection
- Dataset:
  - code-only plumbing
- Objective:
  - add a new selector family that can use signals missing from the old
    `soft_agreement + causal_lag_delta - margin_penalty` family
  - specifically:
    - cross-subject causal-lag mean/std
    - current adjacency parent-entropy
- Exact code delta:
  - `GraphExp/main_structure_learning.py`
  - added:
    - `compute_dataset_causal_lag_selector_diagnostics(...)`
    - `selection_score_mode = causal_lag_entropy_composite`
    - CLI:
      - `--selection_causal_lag_subject_limit`
      - `--selection_causal_lag_std_penalty_weight`
      - `--selection_parent_entropy_penalty_weight`
  - training loop now logs into `quality_history.csv`:
    - `selection_causal_lag_forward_mean/std`
    - `selection_causal_lag_reverse_mean/std`
    - `selection_causal_lag_delta_mean/std`
    - `selection_causal_lag_prefers_forward_frac`
    - existing `adj_parent_entropy_mean` is now also consumed by the selector
  - the new score is:
    - `soft_agreement + subject_delta_mean - subject_delta_std - parent_entropy - dir_margin`
  - `python -m py_compile GraphExp/main_structure_learning.py`
    - pass
- Result:
  - new selector family is fully wired end-to-end
  - no training objective was changed
  - this is selector instrumentation / export logic only
- Interpretation:
  - the selector line now has access to graph-explanation signals that are
    structurally closer to "does this graph explain the data consistently
    across subjects?" than the older scalar family
- Keep / drop:
  - keep the new selector family as the next selection-line baseline
  - drop any further selector experiments that do not record the new
    cross-subject diagnostics

### 2026-03-30 - First live `causal_lag_entropy_composite` test on `sim4 detach_after_25`

- Branch line:
  - selection
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - test the new selector family with a minimal first-pass weighting that
    leans on:
    - cross-subject causal-lag delta
    - parent-entropy penalty
  - while avoiding another large weight sweep
- Matched run:
  - source stabilization baseline:
    - `GraphExp/results/run_20260330_200644`
  - live run:
    - `GraphExp/results/run_20260330_221359`
  - exact config delta relative to the source baseline:
    - `selection_score_mode: causal_lag_composite -> causal_lag_entropy_composite`
    - `selection_soft_agreement_weight: 0.20 -> 0.00`
    - `selection_causal_lag_weight: 1.0 -> 1.0`
    - `selection_margin_penalty_weight: 0.05 -> 0.00`
    - `selection_causal_lag_std_penalty_weight = 0.0`
    - `selection_parent_entropy_penalty_weight = 0.05`
    - keep training-side settings unchanged
- Result:
  - best GT epoch = `25`
  - best GT strict-F1@`eps=0` = `0.885246`
  - exported epoch = `15`
  - exported strict-F1@`eps=0` = `0.819672`
  - final epoch = `40`
  - final strict-F1@`eps=0` = `0.852459`
  - exported vs best-GT gap = `-0.065574`
- Interpretation:
  - this first online weighting is a **negative** selector result
  - but the negative is informative:
    - the new score over-rewarded the early high-`delta_mean` region
    - and did not penalize that region enough for having higher parent entropy
  - this is not evidence that the new signal family is useless
  - it is evidence that the first weighting is biased toward early directional
    intensity
- Keep / drop:
  - keep the new family
  - drop this specific weight setting as the online default

### 2026-03-30 - Cross-trajectory offline audit of the new selector family

- Branch line:
  - selection
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - determine whether the new family truly lacks GT-best information, or
    whether the first live failure was just a bad weighting
- Data used:
  - `GraphExp/results/run_20260330_221359/quality_history.csv`
  - `GraphExp/results/run_20260330_224518/quality_history.csv`
  - note:
    - `run_20260330_224518` was a follow-up live validation using a
      single-run-derived candidate weighting
      (`soft=0.05, lag=0.5, parent_entropy=0.06, margin=0`)
    - it still failed online:
      - best GT = `0.868852` at epoch `25`
      - exported = `0.819672` at epoch `10`
      - final = `0.836066`
- Key offline findings:
  - on `run_20260330_221359`, correlations with strict-F1 were:
    - `selection_causal_lag_delta_mean`: `+0.819690`
    - `adj_parent_entropy_mean`: `-0.929640`
    - `agreement_soft_score`: `+0.954810`
  - on `run_20260330_224518`, correlations with strict-F1 were:
    - `selection_causal_lag_delta_mean`: `+0.842536`
    - `adj_parent_entropy_mean`: `-0.914049`
    - `agreement_soft_score`: `+0.957237`
  - sweeping the new-family score offline over both trajectories showed:
    - many weight settings can pick the GT-best epoch on each run
    - there are `64` shared weight combinations in the tested grid that select
      epoch `25` on **both** runs
- Interpretation:
  - this is the critical selector result of the round
  - unlike the old selector family, the new family is **not** failing because
    it lacks information
  - it does contain GT-best-relevant signal
  - the harder problem is now:
    - weighting robustness across slightly different training trajectories
  - this also exposed a separate issue:
    - repeated CUDA runs with the same nominal config are not perfectly
      reproducible
    - so selector quality should be judged by `exported vs per-run best` first,
      not only by absolute strict-F1 across reruns
- Keep / drop:
  - keep the new signal family as a real improvement over the old selector
    family
  - drop the claim that the selector still "has no GT-best information"
  - keep future selector analysis focused on:
    - robust weight regions
    - or a less weight-sensitive selector structure

### 2026-03-30 - Robust-weight end-to-end validation of the new selector family

- Branch line:
  - selection
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - test one online weighting chosen from the intersection region that selected
    the GT-best epoch on both audited trajectories
- Matched live run:
  - run dir:
    - `GraphExp/results/run_20260330_232227`
  - exact config:
    - `selection_score_mode = causal_lag_entropy_composite`
    - `selection_soft_agreement_weight = 0.10`
    - `selection_causal_lag_weight = 0.50`
    - `selection_margin_penalty_weight = 0.05`
    - `selection_causal_lag_std_penalty_weight = 0.0`
    - `selection_parent_entropy_penalty_weight = 0.05`
    - `detach_direction_from_main_after_epoch = 25`
    - all training-side settings unchanged
- Result:
  - best GT epoch = `23`
  - best GT strict-F1@`eps=0` = `0.852459`
  - exported epoch = `39`
  - exported strict-F1@`eps=0` = `0.852459`
  - final epoch = `40`
  - final strict-F1@`eps=0` = `0.852459`
  - exported vs best-GT gap = `0.000000`
  - final vs best-GT gap = `0.000000`
- Interpretation:
  - this run does **not** prove the selector has learned a perfect universal
    epoch index
  - what it does show is more important:
    - under a shifted but still plausible training trajectory, the new family
      can close the selector gap completely
    - and it can do so without changing training losses
  - on this trajectory the GT-best region became a late plateau
    (`23` through `39/40` in strict-F1 terms), and the selector successfully
    stayed inside that plateau
  - combined with the two-run offline audit, this is enough to upgrade the new
    family from:
    - "interesting diagnostic"
    - to "viable selection-line direction"
- Keep / drop:
  - keep `causal_lag_entropy_composite` as the new selector-line baseline for
    `sim4`
  - keep evaluating selector behavior by:
    - per-run GT-best gap
    - strict-F1 trajectory shape
    - cross-run robustness
  - drop the old conclusion that selection on `sim4 detach_after_25` is still
    fundamentally blocked by missing signal
  - next follow-up should be:
    - test this selector family on another synthetic set before touching the
      main training loop again

### 2026-03-31 - First `causal_lag_entropy_composite` migration read on `sim3` (`seed=11`, `30` epochs)

- Branch line:
  - selection
- Dataset:
  - `sim3.csv`
  - `h3.txt`
- Objective:
  - test whether the new selector family that became viable on `sim4` transfers
    to another synthetic dataset without changing the training loop
  - and compare selector families on the **same** audited training trajectory
    to avoid CUDA rerun noise dominating the read
- Shared training setup:
  - keep the historical `sim3` baseline training branch unchanged:
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
    - `epochs = 30`
    - `selection_agreement_weight = 0.0`
    - `freeze_direction_after_epoch = -1`
    - `detach_direction_from_main_after_epoch = -1`
  - selector-only diagnostics:
    - `selector_audit_gt_path = ..\fMRI_dataset\h3.txt`
    - `selection_top_k = 18`
    - `causal_lag_main_weight = 0.0`
    - `causal_lag_main_lags = 1,2`
    - this keeps training unchanged while still letting the new selector family
      log cross-subject lag diagnostics
- Live run:
  - run dir:
    - `GraphExp/results/run_20260331_081441`
  - transferred selector weighting from the current `sim4`-viable setting:
    - `selection_score_mode = causal_lag_entropy_composite`
    - `selection_soft_agreement_weight = 0.10`
    - `selection_causal_lag_weight = 0.50`
    - `selection_margin_penalty_weight = 0.05`
    - `selection_causal_lag_std_penalty_weight = 0.0`
    - `selection_parent_entropy_penalty_weight = 0.05`
- Live result:
  - best GT epoch = `11`
  - best GT strict-F1@`eps=0` = `0.888889`
  - exported epoch = `30`
  - exported strict-F1@`eps=0` = `0.833333`
  - final epoch = `30`
  - final strict-F1@`eps=0` = `0.833333`
  - exported vs best-GT gap = `-0.055556`
  - strict-F1 best plateau:
    - epochs `8,9,10,11` all reach `0.888889`
- Same-trajectory offline replay:
  - on this exact `quality_history.csv`:
    - `legacy` default picks epoch `11`
      - strict-F1@`eps=0` = `0.888889`
    - old `causal_lag_composite` default (`soft=0.2, lag=1.0, margin=0.05`)
      picks epoch `30`
      - strict-F1@`eps=0` = `0.833333`
    - transferred `sim4` entropy weighting also picks epoch `30`
      - strict-F1@`eps=0` = `0.833333`
  - signal correlations on this run:
    - `agreement_soft_score` vs strict-F1:
      - `+0.796562`
    - `selection_causal_lag_delta_mean` vs strict-F1:
      - `+0.379731`
    - `adj_parent_entropy_mean` vs strict-F1:
      - `-0.407147`
  - best offline entropy-family sweep in the tested grid:
    - epoch `9`
    - strict-F1@`eps=0` = `0.888889`
    - weighting:
      - `soft = 0.02`
      - `lag = 0.25`
      - `lag_std_penalty = 1.0`
      - `parent_entropy_penalty = 0.0`
      - `margin_penalty = 0.0`
- Interpretation:
  - the new family **does** transfer to `sim3`, but not with the same subterm
    emphasis that worked on `sim4`
  - the important contrast is:
    - `sim4` benefited from rewarding high `delta_mean` plus lower entropy
    - `sim3` is different:
      - the GT-best region is an earlier plateau
      - later epochs keep increasing soft agreement / lag mean while GT quality
        stays flat or degrades
  - so on `sim3`, the transferred `sim4` weighting over-ranks the late plateau
  - this is not a "missing signal" failure
  - it is a dataset-shape mismatch:
    - the useful selector dimension on `sim3` is much closer to
      **cross-subject lag stability**
      than to `parent_entropy`
- Keep / drop:
  - keep the new selector family as a valid direction on `sim3`
  - keep same-trajectory offline replay as the primary comparison method when
    selector families are being evaluated
  - drop direct reuse of the `sim4` weighting on `sim3`
  - keep `legacy` as the current strong baseline on `sim3`

### 2026-03-31 - Minimal `- selection_causal_lag_delta_std` selector confirmation on `sim3`

- Branch line:
  - selection
- Dataset:
  - `sim3.csv`
  - `h3.txt`
- Objective:
  - test the simplest online weighting suggested by the same-trajectory offline
    audit:
    - let the selector prefer epochs with lower cross-subject lag-direction
      variance
  - avoid mixing back in `sim4`-specific terms that looked misleading on
    `sim3`
- Matched live run:
  - run dir:
    - `GraphExp/results/run_20260331_082314`
  - exact selector config:
    - `selection_score_mode = causal_lag_entropy_composite`
    - `selection_soft_agreement_weight = 0.0`
    - `selection_causal_lag_weight = 0.0`
    - `selection_margin_penalty_weight = 0.0`
    - `selection_causal_lag_std_penalty_weight = 1.0`
    - `selection_parent_entropy_penalty_weight = 0.0`
  - all training-side settings unchanged from the prior `sim3` run
- Result:
  - best GT epoch = `11`
  - best GT strict-F1@`eps=0` = `0.888889`
  - exported epoch = `8`
  - exported strict-F1@`eps=0` = `0.888889`
  - final epoch = `30`
  - final strict-F1@`eps=0` = `0.833333`
  - exported vs best-GT gap = `0.000000`
  - final vs best-GT gap = `-0.055556`
- Interpretation:
  - this is the decisive `sim3` selector result
  - the new family can match the GT-best plateau on `sim3`
    **without changing training**
  - but the reason is different from `sim4`
  - on `sim3`, the most useful new term is not:
    - high lag mean
    - low parent entropy
  - it is:
    - low cross-subject lag-direction variance
  - in other words:
    - `sim4` rewarded late consistent sharpening
    - `sim3` rewarded finding the early stable window before the later
      confidence inflation pulls selection toward worse checkpoints
  - this also sharpens the overall selector-line judgment:
    - the new family is not converging toward one universal scalar recipe
    - it is converging toward a broader **signal basis**
      where different datasets can rely on different subterms
- Keep / drop:
  - keep `causal_lag_entropy_composite` as a viable selector family on `sim3`
  - keep `selection_causal_lag_delta_std` as the key new `sim3` signal
  - keep `legacy` as a strong control baseline, since it already works well on
    this dataset
  - drop the assumption that `parent_entropy` should always be part of the
    active weighting
  - next follow-up should be:
    - one small cross-dataset audit comparing which subterms are actually useful
      on `sim3` vs `sim4`
    - before promoting any single default weighting across datasets

### 2026-03-31 - Design note: do not fold selector surrogate into the shared total loss

- Type:
  - design record
- Status:
  - not implemented
- Core judgment:
  - it is **not** enough to say "selector signals are useful, so add them as one
    more auxiliary loss inside the current shared training loop"
  - under the current code structure, that would most likely repeat the same
    failure pattern already seen with other direction-side auxiliaries:
    - early help
    - late drift
    - then a new need for freeze / detach to protect the branch
- Why this matters:
  - the current constriction result is not merely:
    - "we need a better direction loss"
  - it is:
    - "late main / causal-lag gradients through the direction gate are
      structurally misaligned with GT"
  - therefore adding another direction-sensitive term into the same total-loss
    pool does not address the place where the mismatch happens
- Current code-state reading:
  - training total loss already mixes many terms in
    `GraphExp/main_structure_learning.py`
  - selector scoring remains outside training and runs under `@torch.no_grad()`
  - only some selector-adjacent ideas are present in loss form:
    - `parent_entropy`
    - forward `causal_lag_main`
  - but the selector signals that turned out to matter in recent audits:
    - `selection_causal_lag_delta_mean`
    - `selection_causal_lag_delta_std`
    are still selection-only diagnostics, not training objectives
- More precise design target:
  - keep the existing two-stage interpretation:
    1. before detach:
       - allow joint optimization to move the branch into a good direction region
    2. after detach:
       - stop letting the main denoising / causal-lag path update the direction
         branch
       - also stop relying on Patel-only late supervision as the sole direction
         teacher
  - in that post-detach stage, introduce a **direction-only** training signal
    instead of another global auxiliary
- Proposed post-detach supervision shape:
  - per-subject direction contrast:
    - `delta_s = L_reverse(s) - L_forward(s)`
  - post-detach direction loss:
    - reward larger cross-subject mean contrast
    - penalize cross-subject variance of that contrast
    - optionally retain a small parent-entropy term
  - practical form:
    - prefer `variance` over `std` for stability
    - avoid ratio forms such as `mean / std` as the first implementation
- Important architectural constraint:
  - this post-detach loss should update the **direction branch only**
  - it should not be allowed to re-shape the support branch while claiming to be
    a direction-isolated repair
  - in practice this likely means:
    - support weights detached
    - direction gate trainable
- Important optimizer constraint:
  - cross-subject `mean/var` statistics are not naturally compatible with the
    current pure `optimizer_step_mode = subject` formulation
  - a real post-detach direction-only objective will likely need:
    - batch-level subject statistics
    - or a small post-detach update path that accumulates several subjects
      before stepping
- Updated recommendation:
  - do **not** make the next move:
    - "add selector surrogate to `total_loss`"
  - make the next move:
    - "design a minimal post-detach direction-only objective that replaces
      Patel-only late supervision"
- Keep / drop:
  - keep selector-derived signals as a source of supervision ideas
  - keep `detach` as the structural isolation mechanism
  - drop the idea that one more shared auxiliary loss is the right next
    abstraction
  - next follow-up should be a minimal code design for:
    - post-detach direction-only contrast loss
    - direction-only parameter routing
    - batch-level cross-subject variance computation

### 2026-03-31 - First live post-detach direction-only mechanism test on `sim3` under `batch_mean`

- Branch line:
  - mechanism
- Dataset:
  - `sim3.csv`
  - `h3.txt`
- Objective:
  - implement the minimal post-detach direction-only objective proposed above
  - test it in the cleanest matched setting first:
    - separate direction branch
    - `optimizer_step_mode = batch_mean`
    - Patel late supervision turned off at the same epoch as detach
  - isolate three questions:
    1. does `batch_mean` itself shift the `sim3` best-epoch window
    2. if we turn Patel late supervision off, does the direction branch need a
       replacement objective
    3. if yes, does the proposed `- mean(delta) + var(delta)` objective help
- Implementation sanity check:
  - tiny subset smoke:
    - `GraphExp/results/run_20260331_183958`
  - result:
    - code path ran successfully
    - new `quality_history.csv` fields were written:
      - `post_detach_direction_active`
      - `post_detach_direction_delta_mean`
      - `post_detach_direction_delta_var`
      - `post_detach_direction_weighted`
- Matched run family:
  - `GraphExp/results/run_20260331_185949`
    - `batch_mean` no-detach baseline
    - `detach = -1`
    - `directional_loss_end_epoch = -1`
    - `post_detach_direction_contrast_weight = 0`
    - `post_detach_direction_variance_weight = 0`
  - `GraphExp/results/run_20260331_184225`
    - early detach control
    - `detach = 10`
    - `directional_loss_end_epoch = 10`
    - no post-detach objective
  - `GraphExp/results/run_20260331_185053`
    - early detach + post-detach objective
    - `detach = 10`
    - `directional_loss_end_epoch = 10`
    - `post_detach_direction_contrast_weight = 1.0`
    - `post_detach_direction_variance_weight = 10.0`
  - `GraphExp/results/run_20260331_191723`
    - corrected-timing detach control
    - `detach = 15`
    - `directional_loss_end_epoch = 15`
    - no post-detach objective
  - `GraphExp/results/run_20260331_190749`
    - corrected-timing detach + post-detach objective
    - `detach = 15`
    - `directional_loss_end_epoch = 15`
    - `post_detach_direction_contrast_weight = 1.0`
    - `post_detach_direction_variance_weight = 10.0`
- Result:
  - core comparison:

```text
run                            detach  dir_end  post(c,v)   best GT        exported       final

run_20260331_185949            -1      -1       0,0         0.9444 @ 15    0.8889 @ 21   0.7778 @ 30
run_20260331_184225            10      10       0,0         0.7778 @ 3     0.7222 @ 30   0.7222 @ 30
run_20260331_185053            10      10       1,10        0.8889 @ 24    0.7222 @ 11   0.7778 @ 30
run_20260331_191723            15      15       0,0         0.9444 @ 15    0.9444 @ 30   0.9444 @ 30
run_20260331_190749            15      15       1,10        0.9444 @ 18    0.9444 @ 15   0.8333 @ 30
```

  - corrected-timing detail:
    - `detach = 15` no-post:
      - best / exported / final all landed on the same strict-F1 plateau:
        `0.9444`
      - final no longer drifted away from the GT-best region
    - `detach = 15` + post objective:
      - exported still hit the best plateau
      - but final fell back to `0.8333`
  - post-detach trajectory contrast on the corrected-timing pair:
    - no-post:
      - from epoch `15` to `30`, selector lag stats changed only gently:
        - `delta_mean`: `0.0010  0.0025`
        - `delta_std`: `0.0055  0.0045`
      - strict-F1 stayed flat at `0.9444`
    - with post objective:
      - after detach, the new objective kept pushing the branch:
        - `post_detach_direction_delta_mean`: `0.0010  0.0114`
        - `selection_causal_lag_delta_std`: `0.0055  0.0103`
      - strict-F1 dropped:
        - `0.9444  0.8333`
- Interpretation:
  - the first important finding is about **timing**, not about the new loss:
    - under `batch_mean`, `sim3`'s GT-best window moved to around epoch `15`
    - so the original `detach = 10` trials were cutting the branch too early
  - once detach timing is aligned with the actual best window, `sim3` does **not**
    need a new post-detach direction objective:
    - `detach = 15` plus Patel late-off already solves the retention problem in
      this setting
  - the proposed post-detach objective is **not** a safe default, even though it
    is now isolated to the direction branch:
    - it continued to increase reverse-vs-forward contrast after detach
    - but that extra sharpening did not preserve GT quality
    - on `sim3`, it actually pushed the branch off the best plateau
  - this is the most important mechanism-line update from the run family:
    - even after gradient isolation, `delta_mean / delta_var` is still a
      **direction-sharpening surrogate**
    - it is not automatically a GT-preserving target
    - in other words:
      - "selector-useful signal" still does **not** imply
      - "good training target"
  - the corrected-timing no-post control is especially informative:
    - it shows that the branch can already hold the correct solution on `sim3`
      if we simply stop updating it at the right time
    - so on this dataset, the clean answer is still constriction by timing, not
      extra late-phase optimization
- Keep / drop:
  - keep:
    - the code support for post-detach direction-only losses
    - `batch_mean` as the required mode for any future batch-level direction-only
      experiment
    - `detach = 15` + `directional_loss_end_epoch = 15` as the current best
      `sim3` mechanism result under `batch_mean`
  - drop:
    - the idea that the new post-detach objective should become the default
      replacement for Patel late supervision
    - the earlier `detach = 10` read as a fair verdict on the new objective
  - next follow-up should be:
    - move back to `sim4`, where the no-freeze retention problem was harder
    - test whether the post-detach objective helps there **only after** first
      aligning detach timing with the actual `batch_mean` best window
    - keep `sim3` as the negative-control reminder that isolated late direction
      optimization can still overshoot

### 2026-03-31 - `sim4` migration of the `batch_mean` post-detach mechanism line

- Branch line:
  - mechanism
- Dataset:
  - `sim4.csv`
  - `h4.txt`
- Objective:
  - continue the post-detach mechanism line on the harder dataset
  - answer three concrete questions in `batch_mean` mode:
    1. where is the real `sim4` GT-best window under `batch_mean`
    2. if detach timing is aligned to that window, does timing alone solve the
       final drift
    3. after timing is aligned, does the new post-detach direction-only
       objective add anything
- Matched run family:
  - `GraphExp/results/run_20260331_194238`
    - `batch_mean` no-detach baseline
    - training matched to the current `sim4` selector family:
      - `selection_score_mode = causal_lag_entropy_composite`
      - `selection_soft_agreement_weight = 0.1`
      - `selection_causal_lag_weight = 0.5`
      - `selection_margin_penalty_weight = 0.05`
      - `selection_parent_entropy_penalty_weight = 0.05`
    - `detach = -1`
    - `directional_loss_end_epoch = -1`
    - no post-detach objective
  - `GraphExp/results/run_20260331_201124`
    - aligned-timing detach control
    - `detach = 27`
    - `directional_loss_end_epoch = 27`
    - no post-detach objective
  - `GraphExp/results/run_20260331_211150`
    - aligned-timing detach + post objective
    - `detach = 27`
    - `directional_loss_end_epoch = 27`
    - `post_detach_direction_contrast_weight = 1.0`
    - `post_detach_direction_variance_weight = 10.0`
    - `post_detach_direction_parent_entropy_weight = 0.0`
    - operational note:
      - an earlier attempt timed out after only finishing pretraining
      - the completed run reused the saved encoder checkpoint so the actual
        training comparison stayed matched
- Result:
  - core comparison:

```text
run                               detach  dir_end  post(c,v)   best GT        exported       final

run_20260331_194238               -1      -1       0,0         0.8033 @ 27    0.7705 @ 22   0.7541 @ 40
run_20260331_201124               27      27       0,0         0.8033 @ 27    0.7705 @ 22   0.8033 @ 40
run_20260331_211150               27      27       1,10        0.8033 @ 28    0.7705 @ 22   0.7869 @ 40
```

  - `batch_mean` baseline read:
    - the GT-best window moved to around epoch `27`
    - exported still came from epoch `22`
    - final dropped by `-0.0492` vs best GT
  - aligned timing without the new loss:
    - final drift disappeared
    - from epoch `27` onward, strict-F1 stayed on the best plateau:
      - epoch `27`: `0.8033`
      - epoch `30`: `0.8033`
      - epoch `40`: `0.8033`
    - selector still exported epoch `22`, so the residual problem was selection,
      not retention
  - aligned timing with the new post-detach objective:
    - the new objective strongly increased late lag-direction separation:
      - `post_detach_direction_delta_mean`:
        - epoch `28`: `0.0024`
        - epoch `30`: `0.0079`
        - epoch `40`: `0.0147`
      - `selection_causal_lag_delta_mean`:
        - epoch `28`: `0.0037`
        - epoch `30`: `0.0090`
        - epoch `40`: `0.0148`
    - but GT quality did not improve:
      - final only reached `0.7869`
      - this is better than the no-detach final `0.7541`
      - but worse than the aligned no-post final `0.8033`
    - selector still exported epoch `22`
- Interpretation:
  - the first decisive `sim4 batch_mean` finding matches the updated `sim3`
    mechanism reading:
    - **timing alignment is the primary fix**
  - once detach timing is aligned to the actual GT-best window, the final drift
    problem is solved without needing any new late objective:
    - on `sim4`, `detach = 27` + Patel late-off was enough to hold the best
      plateau through epoch `40`
  - the post-detach direction-only objective again failed to become a clean
    default:
    - it increased the exact surrogate it was designed to increase
    - but that increase did not translate into better GT strict-F1
    - instead it partially reintroduced late degradation
  - this makes the cross-dataset pattern much sharper:
    - on `sim3`, post-detach sharpening overshot badly
    - on `sim4`, it overshot more mildly, but it still underperformed the
      aligned no-post control
  - the remaining unsolved issue on `sim4 batch_mean` is now clearly **selector**
    rather than **retention**:
    - the training path can hold the good late solution
    - but the current selector still prefers epoch `22` while the best plateau
      sits at `27 40`
- Keep / drop:
  - keep:
    - `detach = 27` + `directional_loss_end_epoch = 27` as the current best
      `sim4 batch_mean` stabilization result
    - the conclusion that timing-aligned detach is the cleanest retention fix on
      both `sim3` and `sim4`
    - the post-detach code path as an experimental tool
  - drop:
    - the idea that post-detach `delta_mean / variance` optimization should be
      the default late-phase replacement objective
    - the assumption that improving late lag-direction contrast automatically
      improves GT direction quality
  - next follow-up should be:
    - return to the selector line on top of the stabilized `sim4 batch_mean`
      branch
    - specifically audit why the current selector keeps preferring epoch `22`
      when epochs `27 40` sit on the GT-best plateau
    - if needed, evaluate whether the best selector signal on this stabilized
      branch is still the old composite, or whether the plateau shape now
      changes which term is trustworthy

### 2026-04-02 - `warmup_then_orthogonal` timing window on `sim4` (`detach = 22/23/24/25`)

- Branch line:
  - mechanism
- Dataset:
  - `sim4`
  - GT: `h4`
- Exact config delta:
  - fixed base:
    - `GraphExp/main_structure_learning.py`
    - `--gradient_routing_mode warmup_then_orthogonal`
    - `--structure_parameterization support_direction`
    - `--fixed_support_mask_mode maxgap_kappa`
    - `--direction_init_mode random`
    - `--structure_init_mode patel_kappa`
    - `--structure_init_scale 0.5`
    - `--adj_activation sigmoid`
    - `--directional_prior_mode patel`
    - `--directional_schedule plateau`
    - `--directional_kappa_gate --directional_kappa_gate_quantile 0.5`
    - `--directional_target_ratio 0.01`
    - `--lambda_l1 0.02`
    - `--optimizer_step_mode subject`
    - `--main_loss_weight 1.0`
    - `--selection_agreement_weight 0.0`
    - `--direction_lr_multiplier 1.0`
    - `--freeze_direction_after_epoch -1`
    - `--causal_lag_main_weight 0.25`
    - `--causal_lag_main_lags 1,2`
    - `--causal_lag_main_aggregation mean`
    - `--seed 11`
    - pretrained encoder:
      - `GraphExp/results/run_20260310_185625/pretrained_encoder.pt`
  - only varying flag:
    - `--detach_direction_from_main_after_epoch`
      - `22` -> `GraphExp/results/run_20260402_162717`
      - `23` -> `GraphExp/results/run_20260402_164439`
      - `24` -> `GraphExp/results/run_20260402_172237`
      - `25` -> `GraphExp/results/run_20260402_153935`
- Result:
  - `detach = 22`
    - best GT:
      - epoch `16`
      - strict-F1 `0.8361`
    - exported:
      - epoch `9`
      - strict-F1 `0.8033`
    - final:
      - epoch `40`
      - strict-F1 `0.8361`
    - final-best gap:
      - `0.0000`
  - `detach = 23`
    - best GT:
      - epoch `25`
      - strict-F1 `0.8689`
    - exported:
      - epoch `9`
      - strict-F1 `0.8033`
    - final:
      - epoch `40`
      - strict-F1 `0.8689`
    - final-best gap:
      - `0.0000`
  - `detach = 24`
    - best GT:
      - epoch `30`
      - strict-F1 `0.8852`
    - exported:
      - epoch `9`
      - strict-F1 `0.8033`
    - final:
      - epoch `40`
      - strict-F1 `0.8689`
    - final-best gap:
      - `-0.0164`
  - `detach = 25`
    - best GT:
      - epoch `23`
      - strict-F1 `0.8689`
    - exported:
      - epoch `9`
      - strict-F1 `0.8033`
    - final:
      - epoch `40`
      - strict-F1 `0.8525`
    - final-best gap:
      - `-0.0164`
- Interpretation:
  - the new routing split is doing the intended mechanism job:
    - after the switch, main denoising stays on `support`
    - `causal_lag_main` is left to move `direction`
    - this clearly changes late-phase retention behavior
  - but the switch timing is still decisive:
    - `detach = 22` is too early
    - it removes late drift completely, but it also lowers the reachable GT
      ceiling
    - `detach = 23` is the cleanest stabilization point in this window
    - it preserves the `0.8689` GT-best plateau all the way to epoch `40`
    - `detach = 24` gives the highest observed GT-best (`0.8852`), but it no
      longer preserves the peak to the end
    - `detach = 25` is later than needed and reintroduces the same `-0.0164`
      late drop seen before
  - selector/export is still unchanged across the entire window:
    - every run exported epoch `9` at `0.8033`
    - so the routing fix is improving retention / late mechanism behavior, not
      selector alignment
- Keep / drop:
  - keep:
    - `warmup_then_orthogonal` as the clearer mechanism line than the old
      implicit detach logic
    - `detach = 23` as the current best default when the goal is to preserve the
      best late plateau through the final epoch
    - `detach = 24` as the current higher-ceiling but less stable branch worth
      further study if the goal shifts from retention to peak GT
  - drop:
    - `detach = 22` as a default
    - it is stable, but it cuts too early and suppresses the best score
    - the assumption that fixing late drift will automatically fix best/export
      separation

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

## Code Records

### 2026-04-02 - Explicit gradient-routing refactor for support vs direction

- Branch line:
  - mechanism
- Dataset:
  - none
  - code refactor only
- Exact config delta:
  - `GraphExp/main_structure_learning.py`
    - added explicit routing structs:
      - `AdjacencyGradRouting`
      - `EpochGradientRouting`
    - added `build_epoch_gradient_routing(...)` so each epoch now spells out:
      - whether the main denoising path updates support
      - whether the main denoising path updates direction
      - whether adjacency regularizers update support or direction
      - whether `causal_lag_main` updates support or direction
    - new CLI:
      - `--gradient_routing_mode {legacy,orthogonal,warmup_then_orthogonal}`
    - `quality_history.csv` now records:
      - `gradient_routing_mode`
      - `gradient_routing_label`
      - per-loss support/direction update flags
    - `config.npy` now records:
      - `gradient_routing_mode`
      - `gradient_routing_last_label`
  - `GraphExp/models/DDM.py`
    - `DDM.forward(...)` now accepts:
      - `detach_support_from_main`
    - this makes the main-path routing explicit at the graph-construction call
      site instead of being implied by caller-side assumptions
- Routing semantics:
  - `legacy`
    - preserves the old behavior
    - before `--detach_direction_from_main_after_epoch`:
      - main / structure regularizers / `causal_lag_main` all update both
        support and direction
    - after that epoch:
      - those same paths stop updating direction
  - `orthogonal`
    - from epoch `1`:
      - main denoising path updates support only
      - adjacency regularizers update support only
      - `causal_lag_main` updates direction only
  - `warmup_then_orthogonal`
    - keep the old joint behavior through
      `--detach_direction_from_main_after_epoch`
    - then switch to the orthogonal split above
- Result:
  - no training run launched yet under this record
  - static verification:
    - `python -m py_compile GraphExp/main_structure_learning.py GraphExp/models/DDM.py`
      passed
- Interpretation:
  - this refactor does **not** claim a new mechanism win by itself
  - its purpose is to make loss-to-branch responsibility explicit before any new
    retention experiment is launched
  - the old code path remains available as the default, so historical runs stay
    interpretable
- Rollback note:
  - to recover the pre-refactor training behavior, use:
    - `--gradient_routing_mode legacy`
  - if needed, existing detach timing can still be reproduced with:
    - `--detach_direction_from_main_after_epoch ...`
- Keep / drop:
  - keep:
    - explicit routing scaffolding
    - backward-compatible `legacy` default
    - per-epoch routing diagnostics for later mechanism analysis
  - drop:
    - the previous implicit assumption that readers can infer branch updates from
      scattered `detach` booleans inside the loop

### 2026-04-02 - Warmup-then-orthogonal detach timing window on `sim4`

- Branch line:
  - mechanism
- Dataset:
  - `sim4`
  - GT audit: `h4`
  - `seed = 11`
- Exact config delta:
  - base line:
    - `gradient_routing_mode = warmup_then_orthogonal`
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `direction_init_mode = random`
    - `structure_init_mode = patel_kappa`
    - `structure_init_scale = 0.5`
    - `adj_activation = sigmoid`
    - `directional_prior_mode = patel`
    - `directional_schedule = plateau`
    - `directional_kappa_gate = on`
    - `directional_kappa_gate_quantile = 0.5`
    - `directional_target_ratio = 0.01`
    - `lambda_l1 = 0.02`
    - `optimizer_step_mode = subject`
    - `main_loss_weight = 1.0`
    - `selection_agreement_weight = 0.0`
    - `direction_lr_multiplier = 1.0`
    - `freeze_direction_after_epoch = -1`
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - `causal_lag_main_aggregation = mean`
    - `epochs = 40`
    - pretrained encoder:
      - `GraphExp/results/run_20260310_185625/pretrained_encoder.pt`
  - only sweep:
    - `detach_direction_from_main_after_epoch in {22, 23, 24, 25}`
- Result:
  - `detach = 22`
    - run: `run_20260402_162717`
    - best GT:
      - epoch `16`
      - strict-F1 `0.836066`
    - exported:
      - epoch `9`
      - strict-F1 `0.803279`
    - final:
      - epoch `40`
      - strict-F1 `0.836066`
    - final-best gap:
      - `0.000000`
  - `detach = 23`
    - run: `run_20260402_164439`
    - best GT:
      - epoch `25`
      - strict-F1 `0.868852`
    - exported:
      - epoch `9`
      - strict-F1 `0.803279`
    - final:
      - epoch `40`
      - strict-F1 `0.868852`
    - final-best gap:
      - `0.000000`
  - `detach = 24`
    - run: `run_20260402_172237`
    - best GT:
      - epoch `30`
      - strict-F1 `0.885246`
    - exported:
      - epoch `9`
      - strict-F1 `0.803279`
    - final:
      - epoch `40`
      - strict-F1 `0.868852`
    - final-best gap:
      - `-0.016393`
  - `detach = 25`
    - run: `run_20260402_153935`
    - best GT:
      - epoch `23`
      - strict-F1 `0.868852`
    - exported:
      - epoch `9`
      - strict-F1 `0.803279`
    - final:
      - epoch `40`
      - strict-F1 `0.852459`
    - final-best gap:
      - `-0.016393`
- Interpretation:
  - the gradient-routing split is materially affecting late retention, but the
    switch timing still controls the tradeoff between ceiling and stability
  - `detach = 22` is too early:
    - it preserves final perfectly
    - but it lowers the ceiling before the model reaches the stronger late
      plateau
  - `detach = 23` is the cleanest current retention point:
    - it keeps the stronger `0.868852` plateau all the way to epoch `40`
    - this is the first point in the window that preserves the higher late
      solution without loss
  - `detach = 24` pushes the GT best score higher to `0.885246`
    - so orthogonal routing is not only a stabilization trick
    - but final still slips back by one notch, so the selector/export mismatch
      remains unresolved
  - `detach = 25` is slightly too late for retention:
    - it keeps more late joint training
    - but that reintroduces part of the final drift
- Keep / drop:
  - keep:
    - `warmup_then_orthogonal` as the current mechanism line
    - `detach = 23` as the best default if the immediate goal is to preserve
      the late GT plateau to final
    - `detach = 24` as the best ceiling point to reuse for selector work,
      because it reaches the strongest GT epoch even though it does not fully
      retain it
  - drop:
    - earlier-than-`23` switching as a default retention policy on `sim4`
    - the assumption that one fixed detach timing automatically solves both
      retention and selector/export at once

### 2026-04-02 - Mechanism-line reset after the routing experiments

- Branch line:
  - mechanism
- Dataset:
  - conclusion record only
  - based on the recent `sim4` routing window and the broader Patel-assisted
    experiment line
- Core reflection:
  - the routing refactor produced a real mechanism signal:
    - writing `support` and `direction` update ownership explicitly does change
      late retention in a coherent way
  - but the broader experiment line still carries too much
    "know-the-answer-then-fit-the-solution" pressure:
    - Patel still defines too much of the target shape
    - detach timing was getting close to a seed/window fitting exercise
    - GT audit remained cleanly separated from training, but the selector and
      several priors still shape the model toward Patel-like answers
- Current mechanism conclusions worth keeping:
  - keep:
    - `support / direction` factorization
    - explicit gradient routing
    - the late-phase rule:
      - main denoising path updates `support`
      - `causal_lag_main` updates `direction`
  - current stable anchor:
    - `gradient_routing_mode = warmup_then_orthogonal`
    - `detach_direction_from_main_after_epoch = 23`
    - use this when the objective is retention stability
  - current high-ceiling anchor:
    - same routing with `detach = 24`
    - use this only as a comparison point when checking whether a cleaner
      direction objective can still reach the stronger late peak
- Largest Patel residues still present in the current line:
  - `direction` target:
    - `directional_prior_mode = patel`
    - Patel tau still supervises what direction should be learned
  - `direction` supervision scope:
    - `directional_kappa_gate`
    - Patel kappa still decides which pairs are worth directional supervision
  - `support` space:
    - `fixed_support_mask_mode = maxgap_kappa`
    - Patel kappa still constrains which undirected pairs are even learnable
  - selector:
    - best-epoch export still uses Patel direction/agreement and Patel skeleton
      overlap terms
- Mainline reset decision:
  - stop treating detach-window micro-search as the primary path
  - next experiments should answer a cleaner question:
    - can the model learn direction from causal-lag evidence with explicit
      routing, without Patel acting as the direction teacher?
  - GT should remain audit-only
  - selector should stop being Patel-shaped before any claim about "the model
    chooses the right checkpoint" is made
- Keep / drop:
  - keep:
    - the routing refactor as a valid mechanism improvement
    - the `detach = 23` and `detach = 24` runs as rollback-friendly anchors
      for later comparison
  - drop:
    - using a narrow detach sweep on a small number of seeds as the main source
      of architectural confidence
    - making claims of self-consistent direction learning while Patel still
      serves as the explicit direction target

### 2026-04-02 - Phase A check: remove Patel as the direction teacher on the `detach = 23` anchor

- Branch line:
  - mechanism
- Dataset:
  - `sim4`
  - GT audit: `h4`
  - `seed = 11`
- Fixed anchor:
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `structure_init_mode = patel_kappa`
  - `direction_init_mode = random`
  - `causal_lag_main_weight = 0.25`
  - `causal_lag_main_lags = 1,2`
- Compared runs:
  - Patel direction teacher:
    - run: `run_20260402_164439`
    - config:
      - `directional_prior_mode = patel`
      - `directional_kappa_gate = on`
    - result:
      - best GT:
        - epoch `25`
        - strict-F1 `0.868852`
      - exported:
        - epoch `9`
        - strict-F1 `0.803279`
      - final:
        - epoch `40`
        - strict-F1 `0.868852`
  - `lag_corr` direction teacher:
    - run: `run_20260402_184959`
    - config:
      - `directional_prior_mode = lag_corr`
      - `lag_direction_source = raw`
      - `directional_prior_lags = 1,2`
      - `directional_kappa_gate = on`
    - result:
      - best GT:
        - epoch `26`
        - strict-F1 `0.655738`
      - exported:
        - epoch `40`
        - strict-F1 `0.655738`
      - final:
        - epoch `40`
        - strict-F1 `0.655738`
  - no direction teacher:
    - run: `run_20260402_190757`
    - config:
      - `disable_directional_loss = on`
      - only `causal_lag_main` remains to push direction
    - result:
      - best GT:
        - epoch `39`
        - strict-F1 `0.786885`
      - exported:
        - epoch `40`
        - strict-F1 `0.786885`
      - final:
        - epoch `40`
        - strict-F1 `0.786885`
- Interpretation:
  - Patel-free direction learning is **not** yet solved by the current Phase A
    substitute
  - the current `lag_corr` margin teacher is not a drop-in replacement for
    Patel tau on this backbone:
    - GT quality drops sharply from `0.8689` to `0.6557`
    - this is too large to treat as mere selector noise
  - removing the explicit direction teacher entirely is materially better than
    using the current `lag_corr` teacher:
    - `0.7869` vs `0.6557`
    - therefore the present `lag_corr` supervisory formulation is likely
      misaligned or over-constraining, rather than simply "we need any teacher"
  - but causal-lag alone still does not match the Patel-teacher anchor:
    - `0.7869` remains well below `0.8689`
    - so the existing Patel teacher is still carrying real performance in the
      current Patel-constrained support setting
  - an important side observation:
    - both Patel-free variants exported the final epoch rather than epoch `9`
    - so the old best/export split is not an immutable selector bug
    - it is entangled with the Patel-assisted training trajectory itself
- Keep / drop:
  - keep:
    - the `detach = 23` Patel-teacher run as the strongest current stable anchor
    - the no-direction-teacher run as the cleaner first-principles reference
      point
  - drop:
    - the assumption that switching from `patel` to the current `lag_corr`
      formulation is already a valid mainline replacement
    - moving immediately to selector redesign on top of the weak `lag_corr`
      branch

### 2026-04-02 - Phase A diagnostics on the weak `lag_corr` teacher

- Branch line:
  - mechanism
- Dataset:
  - `sim4`
  - GT audit: `h4`
  - `seed = 11`
- Fixed anchor:
  - same as the previous Phase A record
  - base weak branch:
    - `run_20260402_184959`
    - `directional_prior_mode = lag_corr`
    - `directional_prior_scope = online_subject`
    - `directional_target_ratio = 0.01`
    - `directional_kappa_gate = on`
- Goal:
  - determine whether the weak `lag_corr` result is mainly caused by:
    - teacher instability
    - teacher over-weighting
    - Patel-defined supervision scope mismatch
- Single-variable diagnostics:
  - `global_dataset` scope:
    - run: `run_20260402_193341`
    - delta:
      - `directional_prior_scope = global_dataset`
    - result:
      - best GT:
        - epoch `38`
        - strict-F1 `0.655738`
      - exported:
        - epoch `9`
        - strict-F1 `0.590164`
      - final:
        - epoch `40`
        - strict-F1 `0.655738`
    - read:
      - virtually no recovery in GT ceiling or final quality
      - so online-subject prior noise is not the dominant failure mode
  - lower teacher strength:
    - run: `run_20260402_195042`
    - delta:
      - `directional_target_ratio = 0.003`
    - result:
      - best GT:
        - epoch `30`
        - strict-F1 `0.688525`
      - exported:
        - epoch `40`
        - strict-F1 `0.655738`
      - final:
        - epoch `40`
        - strict-F1 `0.655738`
    - read:
      - best improves modestly relative to the weak base
      - final does not recover
      - therefore excessive weight is a contributing factor, but not the main
        one
  - remove Patel kappa gate:
    - run: `run_20260402_200757`
    - delta:
      - `directional_kappa_gate = off`
    - result:
      - best GT:
        - epoch `33`
        - strict-F1 `0.754098`
      - exported:
        - epoch `40`
        - strict-F1 `0.688525`
      - final:
        - epoch `40`
        - strict-F1 `0.688525`
    - read:
      - this is the largest improvement among the three single-variable checks
      - so the current strongest conflict is the combination:
        - `lag_corr` direction teacher
        - plus a Patel-kappa-defined supervision gate
- Interpretation:
  - the present failure of `lag_corr` is **not** primarily explained by
    subject-level teacher noise
  - teacher strength matters somewhat, but only secondarily
  - the main incompatibility appears to be:
    - using a lag-based direction teacher
    - while still letting Patel kappa choose which pairs are supervised
  - but even the best `lag_corr` diagnostic point (`0.7541`) remains below the
    no-direction-teacher reference (`0.7869`)
  - therefore the current `lag_corr` formulation is still not trustworthy as a
    mainline teacher even after removing the most suspicious Patel gate
- Keep / drop:
  - keep:
    - the conclusion that `directional_kappa_gate` should not be treated as
      compatible-by-default once the teacher changes from Patel to lag-based
    - `run_20260402_200757` as the current best `lag_corr` diagnostic point
  - drop:
    - `global_dataset` as the immediate rescue path for this branch
    - the assumption that reducing weight alone will make the current `lag_corr`
      teacher competitive

### 2026-04-02 - Final rescue check for `lag_corr`: no gate + lower weight

- Branch line:
  - mechanism
- Dataset:
  - `sim4`
  - GT audit: `h4`
  - `seed = 11`
- Purpose:
  - test the best remaining rescue hypothesis for `lag_corr` before removing it
    from the mainline path
- Compared runs:
  - no direction teacher reference:
    - run: `run_20260402_190757`
    - config:
      - `disable_directional_loss = on`
    - result:
      - best GT:
        - epoch `39`
        - strict-F1 `0.786885`
      - exported:
        - epoch `40`
        - strict-F1 `0.786885`
      - final:
        - epoch `40`
        - strict-F1 `0.786885`
  - best previous `lag_corr` single-variable point:
    - run: `run_20260402_200757`
    - config:
      - `directional_kappa_gate = off`
      - `directional_target_ratio = 0.01`
    - result:
      - best GT:
        - epoch `33`
        - strict-F1 `0.754098`
      - exported:
        - epoch `40`
        - strict-F1 `0.688525`
      - final:
        - epoch `40`
        - strict-F1 `0.688525`
  - final rescue combo:
    - run: `run_20260402_204728`
    - config:
      - `directional_kappa_gate = off`
      - `directional_target_ratio = 0.003`
      - `directional_prior_mode = lag_corr`
    - result:
      - best GT:
        - epoch `40`
        - strict-F1 `0.786885`
      - exported:
        - epoch `40`
        - strict-F1 `0.786885`
      - final:
        - epoch `40`
        - strict-F1 `0.786885`
- Interpretation:
  - this final combo rescues `lag_corr` from being actively harmful
  - but it still does **not** outperform the simpler no-direction-teacher
    reference:
    - best/export/final all tie at `0.786885`
  - therefore the current `lag_corr` branch offers no demonstrated value add on
    top of `causal_lag_main` alone
  - the practical read is:
    - with the Patel gate removed and the teacher weight reduced, the extra
      `lag_corr` loss becomes close to neutral
    - but "neutral" is not enough to justify keeping it in the mainline
      experimental budget
- Mainline decision:
  - `lag_corr` should be downgraded out of the mainline path
  - keep it only as an experimental branch for future redesign, not as a
    default or recommended direction-teacher setting
- Keep / drop:
  - keep:
    - `run_20260402_204728` as the final rollback/reference point showing the
      strongest currently-known safe `lag_corr` configuration
    - `disable_directional_loss` as the cleaner Patel-free reference branch
  - drop:
    - any claim that the current `lag_corr` implementation improves over the
      no-direction-teacher baseline
    - allocating further mainline tuning budget to the present `lag_corr`
      formulation

### 2026-04-03 - Architecture conclusion: denoising currently does not justify directed learning

- Branch line:
  - mechanism
- Dataset:
  - conclusion record only
  - based on the recent `sim4` routing and Patel-free direction experiments
- Core conclusion:
  - the routing refactor successfully stabilized training behavior
  - but it did **not** resolve the deeper architectural inconsistency:
    - the project is framed as a diffusion / denoising model
    - yet the best current training behavior requires stopping denoising-path
      gradients from updating `direction`
- What is now empirically clear:
  - under `warmup_then_orthogonal`, the useful late-phase split is:
    - denoising / structure regularizers update `support`
    - `causal_lag_main` updates `direction`
  - this improves retention
  - but it also means:
    - the denoising objective is not presently a trustworthy direction-learning
      objective
    - if left unconstrained, late denoising gradients tend to wash directional
      structure away rather than refine it
- Why this matters conceptually:
  - if direction has to be protected from the main denoising path, then the
    current system is not a self-consistent "directed diffusion model"
  - the more accurate description is:
    - diffusion backbone learns / stabilizes the support skeleton
    - an extra time-lag objective is asked to carry directional learning
  - this is an acceptable engineering decomposition
  - but it should not be described as if the denoising loss itself were learning
    direction in a principled way
- Theoretical boundary:
  - `causal_lag_main` currently acts as a lagged predictive regularizer
  - it is **not** yet a standard SEM objective
  - it is **not** yet a standard Granger-causality objective
  - so the current model should not claim first-principles causal grounding for
    its directional component
- Mainline interpretation to keep:
  - keep:
    - the empirical claim:
      - late denoising gradients are harmful to direction
      - explicit routing improves stability
    - the practical Patel-free reference:
      - `disable_directional_loss`
      - with direction carried by `causal_lag_main` in the late phase
  - do not claim:
    - that the present denoising loss naturally learns directed edges
    - that the current directional objective already has a clean causal-theory
      justification
- Next-step boundary:
  - future design work should start from one of two explicit positions:
    - position A:
      - accept the current split honestly
      - diffusion learns support, another objective learns direction
    - position B:
      - redesign the main objective so direction is learned inside a principled
        temporal / causal formulation rather than protected from denoising

### 2026-04-03 - Terminology boundary: `lag_corr` is not `causal_lag_main`

- Branch line:
  - mechanism
- Purpose:
  - prevent later design discussions from conflating two different time-based
    components
- Shared source:
  - both components use lagged temporal information
  - both may use the same lag list such as `1,2`
- But their roles are different:
  - `lag_corr`
    - role:
      - teacher / prior
    - mechanism:
      - first compute an ordered-pair lag score matrix from the time series
      - then use that matrix as a directional supervision target for the current
        `direction` logits
    - practical meaning:
      - "I first infer which direction looks plausible from lagged data, then I
        ask the model to align with that direction."
  - `causal_lag_main`
    - role:
      - task / reconstruction objective
    - mechanism:
      - do **not** first create a direction label
      - instead use the current learned graph to aggregate candidate-parent past
        signals and predict each node's future
      - then backpropagate the prediction error through the graph
    - practical meaning:
      - "I do not tell the model the direction first; I ask whether the current
        direction can actually predict the future better."
- Short mnemonic:
  - `lag_corr`:
    - teacher
  - `causal_lag_main`:
    - exam
- Consequence for current experiments:
  - removing `lag_corr` does **not** remove all directional learning
  - as long as `causal_lag_main` is active and routed to `direction`, the model
    still has a time-based directional training signal
- Current Patel-free reference interpretation:
  - in the preferred Patel-free branch:
    - `disable_directional_loss = True`
    - `lag_corr` / Patel direction teachers are absent
    - late-phase `direction` learning is carried by `causal_lag_main`
  - therefore:
    - that branch is not "direction-free"
    - it is specifically "teacher-free but still task-driven"

### 2026-04-03 - Code cleanup boundary: remove `post_detach_direction_*`

- Scope:
  - file:
    - `GraphExp/main_structure_learning.py`
- What was removed:
  - the standalone `post_detach_direction_*` branch
  - its training-loop accumulation / backward path
  - its epoch logging / quality-history fields
  - its CLI arguments and config-save entries
- Why this cleanup is low risk:
  - this branch was an isolated historical experiment
  - it is outside the current mainline interpretation of
    `support <- diffusion` and `direction <- routed temporal objective`
  - removing it does not change the active `causal_lag_main` / routing design
- Rollback boundary:
  - if a future comparison really needs this branch again, restore it as a
    dedicated experimental patch rather than keeping dead flags in the mainline
  - after removal, `python -m py_compile GraphExp/main_structure_learning.py GraphExp/models/DDM.py`
    passes

### 2026-04-03 - Code cleanup boundary: remove `directed_noise`

- Scope:
  - file:
    - `GraphExp/main_structure_learning.py`
- What was removed:
  - the Patel-tau-biased noise-guide constructor
  - its CLI flags:
    - `--directed_noise`
    - `--direction_alpha`
  - the branch that built an asymmetric noise guide instead of the symmetric
    kappa skeleton
- Why this cleanup is acceptable:
  - the current mainline interpretation already treats Patel direction as a
    teacher / prior issue, not as something that should quietly leak into the
    denoising noise path
  - `directed_noise` was an isolated historical branch and was outside the
    current support/direction routing story
  - after removal, the noise guide path is simpler and always uses the same
    symmetric kappa skeleton semantics
- Rollback boundary:
  - if a future experiment explicitly wants Patel-directed noise again, restore
    it as a separate experiment patch rather than a dormant mainline option
  - after removal, `python -m py_compile GraphExp/main_structure_learning.py GraphExp/models/DDM.py`
    passes

### 2026-04-03 - Code cleanup boundary: remove `lag_corr` from the main program

- Scope:
  - file:
    - `GraphExp/main_structure_learning.py`
- What was removed:
  - the `lag_corr` directional-teacher mode from the main CLI / training path
  - its auxiliary lag-prior constructors and dataset-cache branch
  - its mode/scope/source/lag config plumbing
- What remains intentionally:
  - `causal_lag_main`
    - this is still the active task-level temporal objective
    - it was **not** removed by this cleanup
  - directional supervision now has only two main-program states:
    - on:
      - Patel tau margin teacher
    - off:
      - `disable_directional_loss`
- Why this cleanup is acceptable:
  - the recent experiment line already downgraded the current `lag_corr`
    formulation to experimental-only
  - keeping it inside the main program meant continuing to maintain a branch
    that the current conclusions explicitly say should not be mainline
  - this cleanup removes teacher-branch complexity without touching the current
    `causal_lag_main`-based direction mechanism
- Rollback boundary:
  - if a future redesign wants a new lag-based teacher again, it should return
    as a fresh experiment patch with a cleaner formulation, not as revival of
    the old `lag_corr` wiring
  - after removal, `python -m py_compile GraphExp/main_structure_learning.py GraphExp/models/DDM.py`
    passes

### 2026-04-03 - Four-dataset replay of the two retained models

- Branch line:
  - mechanism
- Purpose:
  - replay the two currently retained model definitions on all four available
    datasets after the main-program cleanup
- Compared models:
  - `patel_assisted`
    - `support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `structure_init_mode = patel_kappa`
    - `direction_init_mode = random`
    - `gradient_routing_mode = warmup_then_orthogonal`
    - `detach_direction_from_main_after_epoch = 23`
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - Patel tau directional teacher on
    - Patel kappa gate on
  - `patel_free`
    - same backbone / routing
    - `disable_directional_loss = on`
    - only `causal_lag_main` remains as the active directional training signal
- Shared run settings:
  - `device = cuda`
  - `epochs = 40`
  - `pretrain_checkpoint = .\results\run_20260310_185625\pretrained_encoder.pt`
  - `optimizer_step_mode = subject`
  - `lambda_l1 = 0.02`
  - `main_loss_weight = 1.0`
  - `selection_agreement_weight = 0.0`
  - `direction_lr_multiplier = 1.0`
  - `freeze_direction_after_epoch = -1`
- Dataset-specific `top_k_edges`:
  - `fMRI / h1`: `5`
  - `sim2 / h2`: `11`
  - `sim3 / h3`: `18`
  - `sim4 / h4`: `61`
- Result summary:
  - `fMRI`
    - `patel_assisted`
      - run: `run_20260403_190615`
      - best/export/final:
        - `1.000000 / 0.400000 / 1.000000`
    - `patel_free`
      - run: `run_20260403_190853`
      - best/export/final:
        - `0.400000 / 0.400000 / 0.400000`
  - `sim2`
    - `patel_assisted`
      - run: `run_20260403_191122`
      - best/export/final:
        - `0.818182 / 0.727273 / 0.818182`
    - `patel_free`
      - run: `run_20260403_191558`
      - best/export/final:
        - `0.727273 / 0.727273 / 0.545455`
  - `sim3`
    - `patel_assisted`
      - run: `run_20260403_192045`
      - best/export/final:
        - `0.944444 / 0.888889 / 0.888889`
    - `patel_free`
      - run: `run_20260403_192727`
      - best/export/final:
        - `0.833333 / 0.777778 / 0.722222`
  - `sim4`
    - `patel_assisted`
      - run: `run_20260403_193344`
      - best/export/final:
        - `0.868852 / 0.803279 / 0.868852`
    - `patel_free`
      - run: `run_20260403_195049`
      - best/export/final:
        - `0.770492 / 0.770492 / 0.770492`
- Direct comparison:
  - `patel_assisted` wins on all four datasets in GT best score
  - `patel_free` does **not** currently beat it on any dataset
  - `patel_free` is only clearly better in one narrower sense:
    - on `sim4`, best/export/final collapse to the same point
    - so it is cleaner and more stable there
  - but that stability does not transfer uniformly:
    - on `sim2` and `sim3`, `patel_free` still degrades by the final epoch
- Current read:
  - no four-dataset evidence supports replacing the current best-performing
    Patel-assisted branch with the Patel-free branch yet
  - the Patel-free branch remains valuable as:
    - the cleaner mechanism reference
    - the better testbed for direction-learning redesign
  - the Patel-assisted branch remains the empirical score leader, but still
    carries export/best mismatch on several datasets, especially:
    - `fMRI`
    - `sim4`
- Artifact:
  - summary csv:
    - `GraphExp/results/two_model_four_dataset_summary_20260403.csv`

### 2026-04-03 - Two single-variable Patel-reduction controls on the `sim4` Patel-assisted anchor

- Branch line:
  - mechanism
- Purpose:
  - test the two lowest-ambiguity Patel-reduction moves one at a time on the
    current `sim4` Patel-assisted winner
- Fixed baseline:
  - run:
    - `run_20260403_193344`
  - dataset / GT:
    - `sim4`
    - `h4`
  - fixed backbone:
    - `support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `structure_init_mode = patel_kappa`
    - `direction_init_mode = random`
    - `gradient_routing_mode = warmup_then_orthogonal`
    - `detach_direction_from_main_after_epoch = 23`
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - Patel tau direction teacher on
    - Patel kappa gate on
  - baseline result:
    - best/export/final:
      - `0.868852 / 0.803279 / 0.868852`
- Control 1:
  - delta:
    - turn `directional_kappa_gate` off
  - run:
    - `run_20260403_220311`
  - result:
    - best/export/final:
      - `0.885246 / 0.754098 / 0.819672`
  - read:
    - removing the gate raises the GT ceiling
    - but export gets worse and final also drops below the baseline final
    - so the gate is not helping ceiling, but it was still acting as a
      stabilizer on this branch
- Control 2:
  - delta:
    - keep the Patel direction teacher and kappa gate
    - change `structure_init_mode: patel_kappa -> random`
  - run:
    - `run_20260403_222030`
  - result:
    - best/export/final:
      - `0.885246 / 0.770492 / 0.852459`
  - read:
    - random support init also raises the GT ceiling
    - export is still worse than the baseline export
    - final remains close to the baseline final and clearly better than the
      no-kappa-gate final
    - among the two one-step Patel reductions, this is the cleaner candidate
      for a next follow-up
- Joint interpretation:
  - both single-variable Patel reductions improved the GT best score from
    `0.868852` to `0.885246`
  - neither one solved the selector/export mismatch
  - `directional_kappa_gate` and `fixed_support_mask_mode = maxgap_kappa` are
    not identical:
    - turning off the gate changes direction-supervision scope without removing
      the support mask
    - the result shows that this still matters materially
  - if a next step must choose only one Patel reduction to keep pushing, the
    current better candidate is:
    - keep `fixed_support_mask_mode = maxgap_kappa`
    - keep the Patel direction teacher for now
    - test the `structure_init_mode = random` line before removing the kappa
      gate entirely
- Artifact:
  - summary csv:
    - `GraphExp/results/sim4_two_step_single_variable_controls_20260403.csv`

### 2026-04-04 - Combined follow-up: `structure_init_mode = random` plus `directional_kappa_gate = off` on `sim4`

- Branch line:
  - mechanism
- Purpose:
  - test whether the two previously promising Patel-reduction moves become
    stronger when combined, or whether they interfere
- Fixed reference points:
  - baseline:
    - `run_20260403_193344`
    - best/export/final:
      - `0.868852 / 0.803279 / 0.868852`
  - single-variable controls:
    - no kappa gate:
      - `run_20260403_220311`
      - `0.885246 / 0.754098 / 0.819672`
    - random structure init:
      - `run_20260403_222030`
      - `0.885246 / 0.770492 / 0.852459`
- Combined treatment:
  - delta from the baseline:
    - `structure_init_mode: patel_kappa -> random`
    - `directional_kappa_gate: on -> off`
  - run:
    - `run_20260404_095648`
  - result:
    - best/export/final:
      - `0.836066 / 0.770492 / 0.803279`
- Read:
  - the two Patel-reduction moves do **not** combine constructively on the
    current branch
  - compared with either single-variable control:
    - GT best drops back down
    - final also drops
  - compared with the original baseline:
    - export is still worse
    - final is also worse
- Current decision:
  - do **not** treat the combined `random_init + no_kappa_gate` setting as the
    next mainline direction
  - among the currently tested Patel-reduction moves on this `sim4` anchor, the
    better next candidate remains:
    - `structure_init_mode = random`
    - while keeping `directional_kappa_gate` on
- Artifact:
  - summary csv:
    - `GraphExp/results/sim4_random_init_no_kappa_gate_20260404.csv`

### 2026-04-04 - Selector replay on the `random structure init + kappa gate on` trajectory

- Branch line:
  - selection
- Purpose:
  - check whether the current weakness of the `random structure init + kappa
    gate on` branch is still mainly selector-side by replaying multiple selector
    scores on the **same** audited training trajectory
- Fixed trajectory:
  - run:
    - `run_20260403_222030`
  - training result:
    - best/export/final:
      - `0.885246 / 0.770492 / 0.852459`
- Offline selector replay on the same `quality_history.csv`:
  - `legacy`
    - chosen epoch:
      - `8`
    - strict-F1@`eps=0`:
      - `0.770492`
  - `causal_lag_composite`
    - chosen epoch:
      - `13`
    - strict-F1@`eps=0`:
      - `0.836066`
  - `causal_lag_entropy_composite`
    - chosen epoch:
      - `13`
    - strict-F1@`eps=0`:
      - `0.836066`
  - `causal_lag_primary`
    - chosen epoch:
      - `40`
    - strict-F1@`eps=0`:
      - `0.852459`
  - GT-best reference on the same trajectory:
    - chosen epoch:
      - `33`
    - strict-F1@`eps=0`:
      - `0.885246`
- Read:
  - this branch is now clearly more selector-limited than training-limited
  - the current `legacy` selector leaves a large amount of already-trained GT
    quality on the table
  - even a pure offline replay, without changing one training gradient, can move
    exported quality from:
    - `0.770492`
    to:
    - `0.836066` with `causal_lag_composite`
    - `0.852459` with `causal_lag_primary`
- Current decision:
  - before doing further training-side Patel reductions on this branch, the more
    valuable next test is a selector-only formalization
  - the first selector candidate to prioritize on this branch is:
    - `causal_lag_primary`
- Artifact:
  - replay csv:
    - `GraphExp/results/run_20260403_222030_selector_mode_replay_20260404.csv`

### 2026-04-04 - Finer selector-only controls on the same `random structure init + kappa gate on` trajectory

- Branch line:
  - selection
- Purpose:
  - refine the selector diagnosis on `run_20260403_222030` without changing any
    training gradient
  - check whether the remaining export gap is mainly caused by
    `soft_agreement` being too strong relative to `causal_lag` and
    `dir_margin`
- Fixed trajectory:
  - run:
    - `run_20260403_222030`
  - training result:
    - best/export/final:
      - `0.885246 / 0.770492 / 0.852459`
- Offline control 1: pseudo-`primary + margin penalty` sweep
  - read:
    - if we keep the current `causal_lag_primary` shape but add a
      `dir_margin` penalty offline, the selector can be pulled back from epoch
      `40` to epoch `34`
    - the recovery only appears once the soft tiebreak is reduced
  - representative points:
    - current-style primary:
      - soft=`0.05`, margin=`0.00`
      - chosen epoch:
        - `40`
      - strict-F1@`eps=0`:
        - `0.852459`
    - reduced-soft primary + margin:
      - soft=`0.02`, margin=`0.10`
      - chosen epoch:
        - `34`
      - strict-F1@`eps=0`:
        - `0.885246`
  - interpretation:
    - the current primary default is not mainly failing because causal-lag is
      too weak
    - it is failing because the selector keeps rewarding late soft-agreement
      growth while not penalizing the simultaneous increase in `dir_margin`
- Offline control 2: existing `causal_lag_composite` parameter sweep
  - reason:
    - unlike the primary family, this can be tested immediately with existing
      CLI flags because `causal_lag_composite` already exposes both
      `selection_soft_agreement_weight` and
      `selection_margin_penalty_weight`
  - representative points:
    - current composite default:
      - soft=`0.20`
      - margin=`0.05`
      - chosen epoch:
        - `13`
      - strict-F1@`eps=0`:
        - `0.836066`
    - improved composite window:
      - soft=`0.02` to `0.03`
      - margin=`0.10` to `0.15`
      - chosen epoch:
        - `34`
      - strict-F1@`eps=0`:
        - `0.885246`
- Read:
  - on this branch, the current export gap is now more specifically:
    - not "causal-lag selector is useless"
    - but "soft agreement is overweighted relative to margin control"
  - the good news is that this does **not** require a training-side redesign to
    test next
  - there is already an existing no-code path in the current program:
    - `selection_score_mode = causal_lag_composite`
    - lower `selection_soft_agreement_weight`
    - raise `selection_margin_penalty_weight`
- Current decision:
  - for the next formal rerun on this branch, prioritize an existing-CLI
    selector-only confirmation before inventing a new selector family
  - preferred first candidate:
    - `selection_score_mode = causal_lag_composite`
    - `selection_soft_agreement_weight = 0.03`
    - `selection_margin_penalty_weight = 0.10`
    - `selection_causal_lag_weight = 1.0`
  - treat this as a selector validation run, not yet as a mainline hyperparameter
    lock-in
- Artifact:
  - sweep csv:
    - `GraphExp/results/run_20260403_222030_selector_primary_margin_sweep_20260404.csv`
    - `GraphExp/results/run_20260403_222030_selector_composite_sweep_20260404.csv`

### 2026-04-04 - Real selector-only rerun for the same branch

- Branch line:
  - selection
- Purpose:
  - validate whether the offline-selected composite selector point really holds
    on a fresh full rerun with training backbone fixed
- Fixed training/backbone target:
  - same branch family as `run_20260403_222030`
  - selector-only candidate from the previous offline sweep:
    - `selection_score_mode = causal_lag_composite`
    - `selection_soft_agreement_weight = 0.03`
    - `selection_margin_penalty_weight = 0.10`
    - `selection_causal_lag_weight = 1.0`
- Real rerun:
  - run:
    - `run_20260404_111017`
  - result:
    - best/export/final:
      - `0.885246 / 0.852459 / 0.868852`
    - exported epoch:
      - `24`
    - GT-best epoch:
      - `30`
- Read:
  - this is still materially better than the old `legacy`-style export level on
    this branch family
  - but it does **not** robustly lock onto the GT-best epoch
  - therefore the earlier offline hit at `epoch 34 / 0.885246` should be read
    as:
    - evidence that selector-only improvements are possible
    - not evidence that `soft=0.03, margin=0.10` is already a stable final
      selector setting
- Follow-up offline replay on the new rerun trajectory:
  - current rerun trajectory still shows selector sensitivity
  - on `run_20260404_111017`, the better composite window shifts further toward:
    - lower soft-agreement weight:
      - `0.00` to `0.01`
    - higher margin penalty:
      - `0.12` to `0.15`
  - representative point:
    - soft=`0.01`, margin=`0.15`
    - chosen epoch:
      - `27`
    - strict-F1@`eps=0`:
      - `0.885246`
- Current decision:
  - keep the training-side conclusion unchanged:
    - this branch is still selector-limited
  - tighten the selector-side conclusion:
    - the current `causal_lag_composite` family is promising
    - but the exact weight point is still sensitive across reruns
  - if doing the next selector-only control, bias further toward:
    - weaker `soft_agreement`
    - stronger `dir_margin` penalty
  - do **not** yet freeze `0.03 / 0.10` as a new default
- Artifact:
  - rerun:
    - `GraphExp/results/run_20260404_111017`
  - replay csv:
    - `GraphExp/results/run_20260404_111017_selector_composite_sweep_20260404.csv`

### 2026-04-05 - Direction-side Patel removal on the current `sim4` random-support backbone (`3` seeds)

- Branch line:
  - direction
- Purpose:
  - test which remaining Patel-linked direction components still carry real
    signal after the support side has already been reduced
  - specifically separate:
    - Patel tau directional teacher
    - Patel kappa gate
- Important implementation boundary:
  - in the current code, `directional_kappa_gate` only gates the Patel
    directional margin-loss path
  - so a literal `2x2` is not meaningful today:
    - once `--disable_directional_loss` is used, the practical branch becomes
      `teacher off`, and "gate on" has no active supervision path left to gate
  - therefore the correct current control family is a **3-way** comparison:
    - `teacher on + gate on`
    - `teacher on + gate off`
    - `teacher off`
- Fixed backbone:
  - dataset / GT:
    - `sim4`
    - `h4`
  - fixed settings:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `support_prior_mode = patel_kappa`
    - `structure_init_mode = random`
    - `direction_init_mode = random`
    - `gradient_routing_mode = warmup_then_orthogonal`
    - `detach_direction_from_main_after_epoch = 23`
    - `causal_lag_main_weight = 0.25`
    - `causal_lag_main_lags = 1,2`
    - `causal_lag_main_aggregation = mean`
    - `selection_score_mode = legacy`
    - `epochs = 40`
    - `directional_schedule = plateau`
  - seeds:
    - `11,22,33`
- Reference line: `teacher on + gate on`
  - reused runs:
    - `run_20260405_091450`
    - `run_20260405_093012`
    - `run_20260405_094634`
  - aggregate:
    - best/export/final:
      - `0.857924 +/- 0.020446`
      - `0.819672 +/- 0.026770`
      - `0.808743 +/- 0.040893`
    - final GT signed-margin median:
      - `0.007301 +/- 0.000252`
- Control line: `teacher on + gate off`
  - runs:
    - `run_20260404_095648`
    - `run_20260405_143032`
    - `run_20260405_144840`
  - aggregate:
    - best/export/final:
      - `0.857923 +/- 0.015455`
      - `0.765027 +/- 0.007728`
      - `0.825137 +/- 0.030911`
    - final GT signed-margin median:
      - `0.009570 +/- 0.002637`
- Control line: `teacher off`
  - runs:
    - `run_20260405_150713`
    - `run_20260405_152548`
    - `run_20260405_154511`
  - aggregate:
    - best/export/final:
      - `0.715847 +/- 0.038640`
      - `0.704918 +/- 0.046368`
      - `0.704918 +/- 0.046368`
    - final GT signed-margin median:
      - `0.001895 +/- 0.000081`
- Seed-level read:
  - `teacher off` is worse on all `3/3` seeds than either teacher-on branch in
    final strict-F1
  - the direction-margin median also collapses consistently when the teacher is
    removed:
    - `~0.0073 to 0.0096` with teacher on
    - `~0.0019` with teacher off
  - between the two teacher-on branches:
    - GT-best ceiling is effectively unchanged
    - `gate off` worsens export selection
    - but `gate off` slightly improves final retention on this backbone
- Read:
  - the current direction-side Patel remnants are **not** equally important
  - Patel tau teacher is still providing real directional signal on this branch
  - removing the teacher drops both:
    - strict-F1
    - GT signed-margin separation
  - Patel kappa gate is different:
    - it is **not** the main source of direction signal
    - removing it does not collapse the branch
    - empirically it behaves more like a stability / trajectory-shaping control
      than the core teacher
  - the practical direction-side priority is now clearer:
    - the real open problem is replacing the Patel tau teacher
    - not treating the kappa gate as the main bottleneck
- Current decision:
  - keep the support-side conclusion unchanged:
    - support-side Patel dependence is no longer the main question
  - update the direction-side conclusion:
    - `teacher off` is now the cleaner Patel-free reference branch
    - but it is materially weaker than the teacher-on branch on the current
      `sim4` backbone
  - for future de-Patel work, prioritize:
    - replacing the tau teacher signal
    - then optionally re-checking whether a gate is still useful under the new
      teacher
- Artifacts:
  - summary:
    - `GraphExp/results/direction_patelside_randombackbone_sim4_3way_20260405_summary.csv`
  - aggregate:
    - `GraphExp/results/direction_patelside_randombackbone_sim4_3way_20260405_aggregate.csv`

### 2026-04-05 - What the support branch still learns under `fixed_support_mask_mode = maxgap_kappa`

- Trigger:
  - once `fixed_support_mask_mode = maxgap_kappa` is enabled, it is easy to
    overstate the role of the diffusion/support branch as if it were still
    performing full support discovery
- Code-path audit:
  - the hard undirected skeleton is built **before** training from the chosen
    support prior:
    - `main_structure_learning.py`
    - `build_noise_guide_adjacency(...)`
    - then copied into `fixed_support_mask`
  - the model then applies that mask directly to learned support weights:
    - `support_logits -> sigmoid(support_weights)`
    - then:
      - `support_weights = support_weights * fixed_support_mask`
  - final causal adjacency in factorized mode is:
    - `adj_weights = support_weights * direction_gate`
- Exact implication:
  - with `fixed_support_mask_mode = maxgap_kappa`, the model does **not**
    continue to learn "which undirected pairs are allowed to exist" globally
  - that combinatorial support set has already been fixed by the prior-derived
    skeleton
- What is already fixed:
  - which undirected pairs are permitted to carry nonzero weight at all
  - which pairs are excluded forever:
    - any pair outside the mask is multiplied to `0`
    - no training loss can resurrect it later
  - the neighbor-based diffusion noise-guide skeleton is also built from the
    same pretraining support set
- What is still learned by the support branch:
  - the **continuous weight** of each allowed support pair inside the mask
  - whether an allowed pair is kept relatively strong or shrunk toward near-zero
  - the effective support reweighting that controls graph message passing during
    denoising
- Parameters still carrying that support-side learning:
  - `node_emb_sender`
  - `node_emb_receiver`
  - shared scalar `adj_bias`
  - these produce symmetric `support_logits`
  - in `support_direction` mode the separate direction parameters are **not**
    part of this support branch
- Losses that still act on support under the current routed branch:
  - main denoising loss:
    - uses the graph built from masked support weights, so it still updates the
      support branch
  - `lambda_l1` sparsity:
    - still shrinks allowed edges, but only inside the fixed mask
  - hub regularization
  - structure orthogonality regularizer
  - after `warmup_then_orthogonal` switches on:
    - these paths are support-only
    - `causal_lag_main` is routed to direction-only
- Correct wording boundary:
  - it is no longer accurate to say:
    - "the current diffusion branch learns the full support structure"
  - the accurate statement is:
    - "the current diffusion branch learns **support reweighting inside a
      prior-fixed hard skeleton**"
- Architecture consequence:
  - under the current mainline, support has already been decomposed into:
    - stage A:
      - prior chooses the admissible undirected skeleton
    - stage B:
      - diffusion/support parameters learn continuous edge strengths within that
        skeleton
  - therefore the model is much closer to:
    - `prior-defined support carrier + learned reweighting`
- than to:
  - unconstrained structure discovery

### 2026-04-05 - Conservative `scheduled_blend` noise-guide smoke

- Goal:
  - turn the previously diagnostic-only learned-noise-guide idea into a real
    training branch with minimal code risk
  - keep the exported/base Patel noise guide unchanged
  - only allow a detached training-time override:
    - `active_noise_guide = (1 - w) * patel + w * learned_detached`
- Implementation boundary:
  - `DDM.forward(...)` now accepts `noise_guide_adj_override`
  - added helper:
    - `build_training_noise_guide_override(...)`
  - the helper currently supports:
    - `training_noise_guide_mode = fixed_patel`
    - `training_noise_guide_mode = scheduled_blend`
  - the main training loop now passes the same override into:
    - the primary diffusion forward
    - the `causal_lag_main` sampling path
  - `quality_history.csv` now records:
    - `training_noise_guide_mode`
    - `training_noise_guide_active`
    - `training_noise_guide_blend_weight`
    - `training_noise_guide_guide_l1_mean`
  - `config.npy` now records the scheduled-blend settings for rollback/replay
- Smoke run:
  - run dir:
    - `GraphExp/results/run_20260405_202707`
  - branch:
    - `sim4`
    - `seed = 11`
    - `support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `structure_init_mode = random`
    - `direction_init_mode = random`
    - `directional_kappa_gate = on`
    - `gradient_routing_mode = warmup_then_orthogonal`
    - `detach_direction_from_main_after_epoch = 23`
    - `causal_lag_main_weight = 0.25`
  - scheduled-blend setting:
    - `training_noise_guide_mode = scheduled_blend`
    - `training_noise_guide_blend_target = 0.5`
    - `training_noise_guide_warmup_epochs = 5`
    - `training_noise_guide_ramp_epochs = 5`
  - smoke length:
    - `10` epochs
- What the smoke verified:
  - code path compiles and trains end-to-end
  - the schedule activates exactly as intended:
    - epochs `1-5`:
      - `active = 0`
      - `blend_weight = 0.0`
    - epochs `6-10`:
      - `blend_weight = 0.1 -> 0.5`
  - the learned-vs-Patel guide difference stayed modest during this smoke:
    - `training_noise_guide_guide_l1_mean ≈ 0.0262`
- Selector-audit outcome for this smoke:
  - best GT epoch:
    - epoch `7`
    - strict `0.8525`
  - exported epoch:
    - epoch `8`
    - strict `0.8361`
  - final epoch:
    - epoch `10`
    - strict `0.8197`
- Correct interpretation:
  - this run proves the scheduled-blend branch is now a **real training
    mechanism**, not just a probe
  - it does **not** yet prove that scheduled blend improves the model
  - because this was only a short plumbing smoke, not a paired control against
    `fixed_patel`
- Next clean comparison:
  - run a paired `fixed_patel` vs `scheduled_blend` control on the same branch
  - preferably reuse the saved pretrain checkpoint from:
    - `GraphExp/results/run_20260405_202707/pretrained_encoder.pt`
  - so the comparison isolates the training noise-guide change instead of
    paying another full encoder-pretrain cost

### 2026-04-05 - Paired control: `fixed_patel` vs `scheduled_blend`

- Shared backbone:
  - dataset:
    - `sim4`
  - seed:
    - `11`
  - `support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `structure_init_mode = random`
  - `direction_init_mode = random`
  - `directional_kappa_gate = on`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - reused encoder checkpoint:
    - `GraphExp/results/run_20260405_202707/pretrained_encoder.pt`
- Runs:
  - `fixed_patel`
    - `GraphExp/results/run_20260405_205715`
  - `scheduled_blend`
    - `GraphExp/results/run_20260405_211525`
  - compact summary:
    - `GraphExp/results/noise_guide_paired_control_sim4_seed11_20260405_summary.csv`
- Selector-audit comparison:
  - `fixed_patel`
    - best GT:
      - epoch `19`
      - strict `0.8197`
    - exported/final:
      - epoch `40`
      - strict `0.7377`
      - gap vs best GT:
        - `-0.0820`
  - `scheduled_blend`
    - best GT:
      - epoch `15`
      - strict `0.8197`
    - exported/final:
      - epoch `40`
      - strict `0.7377`
      - gap vs best GT:
        - `-0.0820`
- Structural-shape comparison at final epoch:
  - `scheduled_blend` clearly softened the collapse geometry relative to
    `fixed_patel`
  - final off-diagonal max:
    - `fixed_patel`
      - `0.6317`
    - `scheduled_blend`
      - `0.1946`
  - final off-diagonal std:
    - `fixed_patel`
      - `0.0184`
    - `scheduled_blend`
      - `0.0099`
  - final parent-entropy mean:
    - `fixed_patel`
      - `0.1084`
    - `scheduled_blend`
      - `0.1231`
- Interpretation:
  - `scheduled_blend` did **not** improve:
    - best GT strict F1
    - exported strict F1
    - final strict F1
    - best/export gap
  - what it **did** improve is the *shape* of the late adjacency:
    - less concentrated
    - fewer extreme spikes
    - slightly higher parent entropy
  - so the current evidence is:
    - scheduled blend is a stabilizing/regularizing change to support-side
      geometry
    - but not yet a direction-quality gain on this branch
- Practical conclusion:
  - keep `scheduled_blend` as an available experimental knob
  - do not promote it into the mainline on the basis of performance yet
  - if revisited later, treat it as:
    - a collapse-shape control
    - not a demonstrated accuracy improvement
