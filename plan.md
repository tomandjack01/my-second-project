# Causal-Lag Diffusion Plan

Last updated: 2026-03-26

## Goal

Test whether direction-sensitive denoising can be made **endogenous** in the
current diffusion framework by constraining denoising information flow to
lagged candidate parents, instead of relying on Patel supervision as the sole
direction source.

This plan is intentionally staged:

1. First verify that multi-lag predictive asymmetry is actually present.
2. Only if the signal is strong enough, implement the minimal new auxiliary
   loss.
3. Run decisive ablations that can clearly answer whether Option B works.

## Current Starting Point

- Current Option A conclusion is stable:
  - diffusion-only does **not** learn direction in the present framework
  - Patel + support/direction factorization is the actual direction source
  - diffusion can help some best checkpoints, but does not autonomously recover
    direction
- The proposed Option B idea is:
  - keep the standard forward diffusion process
  - do **not** redesign the forward kernel first
  - instead add a new denoising-side auxiliary loss where each node is
    reconstructed only from lagged candidate-parent information

## Decision Rule

- We will **not** implement the full causal-lag denoiser immediately.
- We will first run a cheap offline feasibility diagnostic.
- Only if that diagnostic shows a clear multi-lag directional signal will we
  proceed to code changes.

## Phase 0: Offline Go / No-Go Diagnostic

### Objective

Measure whether a node's future is predictably better explained by its true
lagged causes than by the reversed direction, under simple predictors.

### Primary Dataset Order

- First:
  - `sim4.csv`
- Then:
  - `sim3.csv`

Reason:

- `sim4` has more GT edges and gives a more stable signal test
- `sim3` is still useful as a smaller secondary check

### Signal Sources To Test

For each dataset, run the same diagnostic on:

1. Clean raw series:
   - `x`
2. Clean encoded representation:
   - `h = model.prepare_clean_target(x)` or an equivalent frozen encoder output
3. Noisy observations at low diffusion noise:
   - `x_t`, low-noise timesteps
4. Noisy observations at medium diffusion noise:
   - `x_t`, medium-noise timesteps

### Predictor Families

Start simple. Do **not** use a large MLP first.

1. Single-edge linear predictor
   - compare correct direction vs reversed direction per GT edge
2. Multi-parent linear predictor
   - predict each node from all of its GT parents jointly
3. Optional small shared nonlinear predictor
   - only if the linear test is borderline but promising

### Lag Grid

- First pass:
  - `lags = {1, 2, 3}`
- If needed:
  - also test `lags = {1, 2, 3, 4, 5}`

### Metrics

At minimum record:

- edge-level direction accuracy
- mean prediction error gap:
  - reversed minus correct
- GT-edge margin summary:
  - median / p10 / p90
- fraction of GT edges with positive correct-vs-reverse gap

### Go / No-Go Threshold

Proceed to implementation only if **both** of the following are true on
`sim4`:

1. Clean-source multi-lag direction accuracy is clearly above weak-chance:
   - target: `>= 0.70`
2. At least one noisy-input setting still preserves meaningful asymmetry:
   - preferred target: `>= 0.65`
   - or, equivalently, a clearly positive median error gap

Interpretation:

- If clean and noisy both fail:
  - stop Option B; signal is too weak
- If clean passes but noisy fails:
  - only consider a low-noise-gated auxiliary loss
- If noisy also passes:
  - proceed with the full minimal implementation

## Phase 1: Minimal Implementation

This phase starts **only if Phase 0 passes**.

### Design Principle

- Keep standard forward diffusion unchanged
- Add a new auxiliary denoising loss
- Do not replace the main denoising UNet in the first version
- Do not modify checkpoint selection logic in the first version

### New Module

Add a small denoising-side module that predicts a node from:

- its own noisy current signal
- lagged aggregated candidate-parent signals
- diffusion timestep embedding

### First-Version Loss

Use reconstruction-to-clean-target, not epsilon-prediction, to stay aligned
with the current `DDM` training objective.

For node `i`:

- build lagged parent context from candidate parents only
- predict `x_clean[i, max_lag:]`
- optimize `smooth_l1`

### Candidate Parent Aggregation

Use:

- `adj_causal = to_causal_matrix_torch(model.get_structure_adj())`

For each lag `tau`:

- aggregate only through `adj_causal[cause, effect]`
- no reverse-direction mixing

### New Flags

Planned CLI additions:

- `--enable_causal_lag_denoise`
- `--causal_lag_values`
- `--causal_lag_target_ratio`
- `--causal_lag_schedule`
- `--causal_lag_head_hidden`
- `--causal_lag_use_self_input`
- `--causal_lag_use_noisy_input`
- `--causal_lag_timestep_mode`
  - e.g. `all`, `low_only`

### Diagnostics To Log

- `causal_lag_loss_raw`
- `causal_lag_loss_weighted`
- `causal_lag_lambda_current`
- `causal_lag_parent_count_mean`
- `causal_lag_active_lag_count`
- optional low-cost error-gap probe on GT edges during synthetic runs

## Phase 2: Decisive Experiments

This phase starts **only if Phase 1 is implemented**.

### Primary Decisive Pilot

Dataset:

- `sim4`

Core condition:

- `disable_directional_loss = True`
- `main_loss_weight = 1.0`
- `enable_causal_lag_denoise = True`

Question:

- can diffusion-side learning alone now break `symmetric_collapse`?

### Secondary Validation

Dataset:

- `sim3`

Question:

- does the same effect transfer to the smaller benchmark?

### Comparison Branches

At minimum compare:

1. diffusion-only baseline
   - no Patel directional supervision
   - no causal-lag loss
2. causal-lag only
   - no Patel directional supervision
   - causal-lag loss on
3. Patel-only reference
4. Patel + causal-lag combined

### Success Criteria

The new direction-sensitive denoising path is only considered successful if it
achieves most of the following:

- diffusion-only no longer ends in `symmetric_collapse`
- GT signed margin median rises substantially above the old diffusion-only
  baseline
- gradient conflict between denoising and direction learning becomes neutral or
  positive
- correct-vs-reverse direction preference appears without Patel supervision

## Phase 3: `fMRI.csv` Follow-Up

Only run this if synthetic decisive pilots are positive.

Dataset:

- `fMRI.csv` with `h1.txt`

Goal:

- test whether the same mechanism also helps on the small synthetic
  `fMRI.csv` benchmark paired with `h1.txt`
- explicitly separate:
  - true mechanism gain
  - selector artifacts

## Files Expected To Change In Phase 1

If implementation proceeds, likely files are:

- `GraphExp/models/DDM.py`
- `GraphExp/main_structure_learning.py`
- `GraphExp/run_cross_pred_v1_final_only_compare.py`

Phase 0 reproducibility script:

- `GraphExp/phase0_multilag_direction_diagnostic.py`

## Phase 0B: Final Low-Cost Gradient Diagnostic

This phase is a **last cheap follow-up** after the Phase 0 multi-lag predictor
test failed.

It is intended to answer a narrower question:

- even if offline multi-lag prediction is weak, does the denoising loss itself
  show a stronger directional preference at high noise levels?

This is the final diagnostic worth running before closing the DiffuGC-style
"signal amplification" idea in the current framework.

### Why Phase 0B Exists

Phase 0 showed:

- multi-lag prediction asymmetry is too weak to justify direct implementation
  of a causal-lag denoiser

However, there is still one remaining possibility:

- the denoising objective may exhibit a timestep-dependent directional bias
  that is **not** captured by the offline predictor test

### Important Constraints

This diagnostic must obey all of the following:

1. Do **not** rely on reverse-chain sampling
   - use the model's single-step denoiser output only
2. Do **not** recompute Patel tau from `x_hat`
   - current clean target lives in encoder space
   - Patel statistics on that representation are not semantically reliable
3. Do **not** allow Patel noise-guide leakage
   - diagnostics must explicitly remove Patel-guided forward noise
4. Focus on **direction parameters**, not just support edges
   - the bottleneck is directional identifiability, not support existence

### Core Question

When we remove Patel-guided noise during the probe and bucket samples by noise
level, does the denoising loss produce a stronger **correct-direction**
gradient on the direction branch at high `t` than at low `t`?

### Probe Target

For each GT pair `(c, e)`, define a directional contrast quantity on the
support/direction branch:

- `delta_ce = direction_logit[c,e] - direction_logit[e,c]`

The probe should measure the denoising-loss gradient with respect to
`delta_ce`, not just raw edge-weight gradients.

Reason:

- the main unresolved problem is direction
- support-edge gradient magnitude can be misleading under fixed masks

### Model Sources To Compare

Use two model families if feasible:

1. Patel-trained reference model
   - current strong branch
2. No-Patel-direction model
   - same support branch if possible
   - but no Patel directional supervision

Reason:

- a Patel-trained model may carry directional bias from training history
- without the second reference, high-`t` directional preference is difficult to
  attribute to diffusion itself

### Noise Conditions

At minimum compare:

1. Probe with isotropic/global noise only
   - `noise_guide_adj = None`
2. Probe with current Patel-guided noise
   - current training-style noise guide

Interpretation:

- if only Patel-guided noise shows a high-`t` preference:
  - it is likely just Patel self-reinforcement
- if isotropic noise also shows a stronger high-`t` directional bias:
  - there is evidence of genuine denoising-side amplification

### Timestep Buckets

Use at least three buckets:

- low:
  - `t = 50`
- mid:
  - `t = 300`
- high:
  - `t = 800`

### Minimal Procedure

For each model, each timestep bucket, and each noise condition:

1. take clean input `x`
2. build `x_clean = model.prepare_clean_target(x)`
3. sample `x_t` using the chosen noise mode
4. run the denoiser once to get the single-step reconstruction output
5. compute the denoising loss against `x_clean`
6. backprop only this probe loss
7. read gradients on the directional branch
8. aggregate gradients into GT-pair directional contrasts

### Metrics

Primary metrics:

- GT-pair directional-gradient sign accuracy
  - fraction of GT pairs where the probe gradient pushes the correct direction
- mean directional-gradient margin on GT pairs
- median directional-gradient margin on GT pairs
- high-`t` vs low-`t` improvement in the above metrics

Secondary metrics:

- GT vs reversed-pair absolute gradient ratio
- GT vs non-GT gradient ratio within the active support set

### Pass / Fail Criterion

Treat Phase 0B as positive only if **both** are true:

1. Under isotropic/global probe noise, high-`t` directional-gradient preference
   is clearly stronger than low-`t`
2. That improvement is not explainable solely by Patel-trained initialization
   or Patel-guided noise

Practical stop rule:

- if the isotropic-noise probe does not show a meaningful high-`t` directional
  advantage, close the DiffuGC-style amplification branch

### If Phase 0B Fails

- Stop Option B work in the current framework
- Do not implement:
  - timestep-dependent Patel recomputation
  - timestep-dependent edge weighting
  - causal-lag denoiser integration

### If Phase 0B Passes

- Re-open a narrow Option B branch
- First implementation target should be:
  - a low-noise / high-noise gated auxiliary probe
  - not a full architectural rewrite

## Documentation Rule

Everything for this Option B branch should be recorded in **this file**:

- planned diagnostics
- execution commands
- produced artifact paths
- result summaries
- interpretations
- stop / continue decisions

## Execution Log

### 2026-03-26

- Plan created.
- Next action:
  - run Phase 0 offline multi-lag feasibility diagnostic on `sim4` first.

### 2026-03-26 Phase 0 Execution

- Script added:
  - `GraphExp/phase0_multilag_direction_diagnostic.py`
- Commands:
  - `python GraphExp\phase0_multilag_direction_diagnostic.py --csv_path fMRI_dataset\sim4.csv --gt_path fMRI_dataset\h4.txt --pretrain_checkpoint GraphExp\results\run_20260310_185625\pretrained_encoder.pt --lags 1,2,3 --device cpu --tag sim4_go_nogo`
  - `python GraphExp\phase0_multilag_direction_diagnostic.py --csv_path fMRI_dataset\sim3.csv --gt_path fMRI_dataset\h3.txt --pretrain_checkpoint GraphExp\results\run_20260310_185625\pretrained_encoder.pt --lags 1,2,3 --device cpu --tag sim3_confirm`
- Shell logs:
  - `GraphExp/results/phase0_multilag_sim4_shell.log`
  - `GraphExp/results/phase0_multilag_sim3_shell.log`
- Summary artifacts:
  - `GraphExp/results/multilag_direction_diagnostic_sim4_20260326_190042_sim4_go_nogo_summary.csv`
  - `GraphExp/results/multilag_direction_diagnostic_sim4_20260326_190042_sim4_go_nogo_details.csv`
  - `GraphExp/results/multilag_direction_diagnostic_sim3_20260326_190125_sim3_confirm_summary.csv`
  - `GraphExp/results/multilag_direction_diagnostic_sim3_20260326_190125_sim3_confirm_details.csv`

### 2026-03-26 Phase 0 Result Summary

- Setup:
  - predictor:
    - single-edge linear ridge predictor
  - lags:
    - `{1,2,3}`
  - split:
    - `35` train subjects / `15` test subjects
  - signal sources:
    - clean raw
    - clean encoder
    - noisy raw at `t=50`
    - noisy raw at `t=300`
    - noisy encoder at `t=50`
    - noisy encoder at `t=300`

- `sim4`:
  - best condition:
    - `encoder_clean` or `encoder_noisy_t50`
    - direction accuracy `= 0.5246`
  - raw conditions:
    - `0.4754 - 0.4918`
  - median gap:
    - mostly near zero
    - best median gap only `0.0694`
  - interpretation:
    - far below the Phase 0 go threshold (`>= 0.70`)
    - noisy input does not preserve a strong directional asymmetry

- `sim3`:
  - all conditions:
    - direction accuracy `= 0.4444`
  - mean and median gaps:
    - generally negative
  - interpretation:
    - multi-lag linear prediction is worse than chance-level direction
      discrimination on this dataset

### 2026-03-26 Phase 0 Decision

- Decision:
  - `NO-GO` for the full Phase 1 causal-lag denoiser implementation in the
    current form
- Reason:
  - the prerequisite signal test failed on the primary benchmark `sim4`
  - and the secondary check on `sim3` was even weaker
- What this means:
  - the current proposal does **not** have enough evidence that lagged
    parent-only denoising will create a useful endogenous direction gradient
  - implementing the full auxiliary module now would likely be high-effort and
    low-information
- Recommended next steps instead of Phase 1:
  - either stop Option B here
  - or, if revisiting, only do a stronger offline diagnostic first:
    - multi-parent predictor
    - optional small nonlinear head
    - low-noise-only regime
  - do **not** proceed directly to integration into `DDM.py`

### 2026-03-26 Plan Refinement After Follow-Up Discussion

- The earlier "noise-stratified Patel recomputation" idea is now explicitly
  downgraded:
  - current `x_hat` lives in the model's clean-target / encoder space
  - recomputing Patel tau there is not semantically reliable
- If this branch is revisited, the next diagnostic should be Phase 0B instead:
  - timestep-stratified **direction-parameter gradient** probing
  - with explicit isotropic-noise control
  - and, if possible, Patel-trained vs no-Patel-direction model comparison
- Current status:
  - Phase 0 completed
  - Phase 1 remains blocked
  - Phase 0B is the last low-cost follow-up worth considering before closing
    the branch

### 2026-03-26 Phase 0B Execution

- Code support added for this phase:
  - `GraphExp/main_structure_learning.py`
    - now saves `model_final.pt` at the end of each structure-learning run
  - `GraphExp/phase0b_timestep_gradient_probe.py`
    - added as the Phase 0B diagnostic script
    - final probe version uses explicitly constructed `direction_logits` inside
      the probe graph so gradients are taken on the actual directional contrast
      used by the denoising path
    - output path corrected to write directly into `GraphExp/results`
- Fresh checkpoint commands:
  - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\sim4.csv --device cpu --epochs 100 --pretrain_epochs 50 --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --structure_parameterization support_direction --fixed_support_mask_mode maxgap_kappa --direction_init_mode random --structure_init_mode patel_kappa --structure_init_scale 0.5 --adj_activation sigmoid --directional_prior_mode patel --directional_schedule plateau --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratio 0.01 --lambda_l1 0.02 --optimizer_step_mode subject --main_loss_weight 1.0 --selection_agreement_weight 0.0 --direction_lr_multiplier 1.0 --freeze_direction_after_epoch 30 --seed 11`
  - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\sim4.csv --device cpu --epochs 100 --pretrain_epochs 50 --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --structure_parameterization support_direction --fixed_support_mask_mode maxgap_kappa --direction_init_mode random --structure_init_mode patel_kappa --structure_init_scale 0.5 --adj_activation sigmoid --lambda_l1 0.02 --optimizer_step_mode subject --main_loss_weight 1.0 --selection_agreement_weight 0.0 --disable_directional_loss --seed 11`
- Produced checkpoint runs:
  - Patel-trained reference:
    - `GraphExp/results/run_20260326_200714`
  - no-Patel-direction control:
    - `GraphExp/results/run_20260326_201004`
- Probe commands:
  - `python .\phase0b_timestep_gradient_probe.py --run_dir .\results\run_20260326_200714 --gt_path ..\fMRI_dataset\h4.txt --timesteps 50,300,800 --noise_modes isotropic,patel --num_probe_subjects 8 --device cpu --model_tag patel_ref --tag sim4_phase0b`
  - `python .\phase0b_timestep_gradient_probe.py --run_dir .\results\run_20260326_201004 --gt_path ..\fMRI_dataset\h4.txt --timesteps 50,300,800 --noise_modes isotropic,patel --num_probe_subjects 8 --device cpu --model_tag diff_only --tag sim4_phase0b`
- Probe artifacts:
  - Patel reference:
    - `GraphExp/results/phase0b_gradient_probe_patel_ref_20260326_201607_sim4_phase0b_summary.csv`
    - `GraphExp/results/phase0b_gradient_probe_patel_ref_20260326_201607_sim4_phase0b_details.csv`
  - diffusion-only control:
    - `GraphExp/results/phase0b_gradient_probe_diff_only_20260326_201623_sim4_phase0b_summary.csv`
    - `GraphExp/results/phase0b_gradient_probe_diff_only_20260326_201623_sim4_phase0b_details.csv`

### 2026-03-26 Phase 0B Result Summary

- Setup:
  - dataset:
    - `sim4.csv`
  - GT:
    - `h4.txt`
  - models:
    - Patel-trained reference
    - no-Patel-direction fixed-support control
  - probe subset:
    - first `8` subjects
  - timesteps:
    - `50`, `300`, `800`
  - noise modes:
    - `isotropic`
    - `patel`
- Primary metric summary:
  - Patel-trained reference, isotropic noise:
    - `gt_push_correct_frac = 0.3893 -> 0.3852 -> 0.4816`
    - `gt_signed_push_mean = -7.7e-05 -> -1.07e-04 -> 4.0e-06`
  - Patel-trained reference, Patel-guided noise:
    - `gt_push_correct_frac = 0.3893 -> 0.3689 -> 0.4898`
    - `gt_signed_push_mean = -7.7e-05 -> -1.24e-04 -> 1.0e-06`
  - diffusion-only control, isotropic noise:
    - `gt_push_correct_frac = 0.4611 -> 0.4508 -> 0.5451`
    - `gt_signed_push_mean = -2.0e-05 -> -5.6e-05 -> 9.0e-06`
  - diffusion-only control, Patel-guided noise:
    - `gt_push_correct_frac = 0.4467 -> 0.4426 -> 0.5184`
    - `gt_signed_push_mean = -2.1e-05 -> -5.5e-05 -> 1.2e-05`
- Magnitude summary:
  - all GT signed-push means stay in the `1e-05 - 1e-04` range
  - GT signed-push medians remain essentially `0`
  - GT absolute gradient margins are also tiny:
    - roughly `1.1e-04 - 3.0e-04`
- Important caveat:
  - `gt_to_non_gt_abs_grad_ratio` is not informative in this setup
  - with `fixed_support_mask_mode=maxgap_kappa`, non-support directional logits
    are mostly disconnected from the probe loss, so non-GT gradients collapse to
    `~0` by construction

### 2026-03-26 Phase 0B Interpretation

- There is a mild high-`t` increase in sign accuracy:
  - roughly `+0.07` to `+0.10` from `t=50` to `t=800`
- However, that shift is not accompanied by a meaningful directional-gradient
  magnitude increase:
  - mean signed push stays near `0`
  - median signed push stays at `0`
  - absolute margins remain extremely small
- Patel-guided probe noise and isotropic probe noise behave very similarly:
  - this argues against a strong Patel-noise self-reinforcement story
  - but it also means the high-`t` effect is weak regardless of probe noise mode
- The most optimistic cell is the diffusion-only control under isotropic noise:
  - `gt_push_correct_frac = 0.5451` at `t=800`
  - but this is only a weak majority and is not matched by a meaningful margin
- The Patel-trained reference does **not** show a robust correct-direction
  majority even at high noise:
  - `0.4816` under isotropic probe noise
  - `0.4898` under Patel-guided probe noise
- Therefore:
  - the probe does **not** reveal a decisive high-`t` endogenous direction
    signal
  - at most, it suggests a very weak timestep effect that is too small to
    justify new Option B implementation work

### 2026-03-26 Phase 0B Decision

- Decision:
  - `FAIL`
- Pass/fail reading:
  - the practical stop rule is triggered
  - isotropic-noise probing does **not** show a meaningful high-`t`
    directional advantage
- Updated Option B status:
  - close the DiffuGC-style amplification branch in the current framework
  - do **not** implement:
    - timestep-dependent Patel recomputation
    - timestep-dependent edge weighting
    - causal-lag denoiser integration
- Updated overall conclusion:
  - current evidence still supports Option A
  - diffusion in the present framework does not provide a strong endogenous
    causal-direction learning signal

### 2026-03-27 Phase 0C Plan

- Objective:
  - test whether the already-trained denoiser encodes usable direction
    information under a stricter edge-ablation diagnostic
  - avoid changing training or forward diffusion
- Diagnostic definition:
  - for each directed pair `src -> dst`, define:
    - `importance(src -> dst) = loss_dst(mask src->dst) - loss_dst(full)`
  - for each GT edge `u -> v`, compare:
    - `importance(u -> v)` vs `importance(v -> u)`
  - success means:
    - direction accuracy clearly above weak chance
    - and positive mean / median margin
- Scope for the first smoke:
  - dataset:
    - `sim4.csv`
  - GT:
    - `h4.txt`
  - models:
    - Patel-trained reference
    - no-Patel-direction control
  - probe noise:
    - `isotropic` only
  - timesteps:
    - `50`, `800`
  - subjects:
    - first `8`

### 2026-03-27 Phase 0C Execution

- Script added:
  - `GraphExp/phase0c_edge_ablation_direction_probe.py`
- Commands:
  - `python .\phase0c_edge_ablation_direction_probe.py --run_dir .\results\run_20260326_200714 --gt_path ..\fMRI_dataset\h4.txt --timesteps 50,800 --noise_modes isotropic --num_probe_subjects 8 --device cpu --model_tag patel_ref --tag sim4_phase0c_smoke`
  - `python .\phase0c_edge_ablation_direction_probe.py --run_dir .\results\run_20260326_201004 --gt_path ..\fMRI_dataset\h4.txt --timesteps 50,800 --noise_modes isotropic --num_probe_subjects 8 --device cpu --model_tag diff_only --tag sim4_phase0c_smoke`
- Artifacts:
  - Patel reference:
    - `GraphExp/results/phase0c_edge_ablation_probe_patel_ref_20260327_091756_sim4_phase0c_smoke_summary.csv`
    - `GraphExp/results/phase0c_edge_ablation_probe_patel_ref_20260327_091756_sim4_phase0c_smoke_details.csv`
  - diffusion-only control:
    - `GraphExp/results/phase0c_edge_ablation_probe_diff_only_20260327_091824_sim4_phase0c_smoke_summary.csv`
    - `GraphExp/results/phase0c_edge_ablation_probe_diff_only_20260327_091824_sim4_phase0c_smoke_details.csv`

### 2026-03-27 Phase 0C Result Summary

- Metric:
  - target-node `smooth_l1` reconstruction loss under single-edge ablation
  - report:
    - `direction_accuracy = frac(importance_correct > importance_reverse)`
    - `margin = importance_correct - importance_reverse`
- Patel-trained reference:
  - `t=50`
    - `direction_accuracy = 0.3873`
    - `margin_mean = -0.004251`
    - `margin_median = 0.000000`
  - `t=800`
    - `direction_accuracy = 0.3299`
    - `margin_mean = -0.000933`
    - `margin_median = 0.000000`
- diffusion-only control:
  - `t=50`
    - `direction_accuracy = 0.5533`
    - `margin_mean = 0.001721`
    - `margin_median = 0.000027`
  - `t=800`
    - `direction_accuracy = 0.4980`
    - `margin_mean = -0.001934`
    - `margin_median = -0.000001`

### 2026-03-27 Phase 0C Interpretation

- The denoiser does **not** reveal strong hidden directional information under
  edge ablation.
- Patel-trained reference:
  - the probe strongly prefers the wrong direction on average
  - and high noise makes this worse, not better
- diffusion-only control:
  - there is only a weak low-noise effect:
    - `0.5533`
  - but it is below the practical usefulness threshold
  - and it disappears at high noise:
    - `0.4980`
- This directly contradicts the hoped-for "high-noise denoiser exposes
  direction better" story.
- It also weakens the case for adding a more complex forward-noise propagation
  mechanism:
  - if the current denoiser cannot already expose direction under this stricter
    local-ablation probe, there is no evidence that a larger diffusion rewrite
    will unlock a strong signal

### 2026-03-27 Phase 0C Decision

- Decision:
  - `FAIL`
- Meaning:
  - the agreed Phase 0C smoke does not justify further Option B implementation
  - especially not a causal-noise-propagation training change
- Updated branch status:
  - Option B is now negative under:
    - offline lag prediction
    - timestep-stratified direction-gradient probing
    - denoiser edge-ablation direction probing
- Practical conclusion:
  - stop further diffusion-side direction-mechanism exploration in the current
    codebase unless a substantially different external signal source is
    identified first

## 2026-03-27 Residual Patel Fusion Branch

### Motivation

- After Phase 0 / 0B / 0C, the current evidence is still:
  - diffusion does **not** provide a strong endogenous direction signal
  - but diffusion is also not useless
  - under Option A, it behaves more like a structured carrier / residual
    optimizer around a strong prior
- That suggests a narrower integration idea:
  - do **not** ask diffusion to discover direction from scratch
  - instead make Patel a persistent parameter-space baseline
  - let the learnable graph branch model only the correction around that
    baseline

### Intended Effect

- Reframe the graph branch as:
  - `support_logits = prior_from_kappa + delta_support`
  - `direction_logits = prior_from_tau + delta_direction`
- The hoped-for effect is not "diffusion replaces Patel".
- The hoped-for effect is:
  - Patel provides the default support/direction template
  - denoising gradients only need to learn residual deviations that improve
    reconstruction / selection quality
- Working expectation:
  - support-side residual learning is the more plausible gain path
  - direction-side residual learning may still be weak, so the first round
    should be treated as a smoke comparison, not a decisive validation

### Code Changes Added For This Branch

- `GraphExp/models/DDM.py`
  - added `direction_logit_bias_prior`
  - added `direction_logit_bias_scale`
  - `get_direction_logits()` now supports a persistent Patel-tau bias term
  - the stored direction prior is skew-symmetrized so the downstream
    `direction_logits - direction_logits.T` contrast receives the intended
    Patel-tau directional signal
- `GraphExp/main_structure_learning.py`
  - added CLI flag:
    - `--direction_logit_bias_scale`
  - training now passes `patel_direction_matrix` into `DDM` as
    `direction_logit_bias_prior` under the model's raw/internal convention
  - first smoke exposed a raw/causal mismatch in that handoff
  - call-site was then corrected so the persistent direction prior is
    transposed before entering the raw direction branch
  - run/config logging now records both:
    - `kappa_logit_bias_scale`
    - `direction_logit_bias_scale`

### First-Round Comparison Design

- Objective:
  - check whether a minimal persistent-prior version of the current
    `support_direction + maxgap_kappa` branch is at least competitive with the
    current non-persistent baseline on a cheap synthetic smoke
- Dataset:
  - `sim3.csv`
- GT:
  - `h3.txt`
- Runtime target:
  - single-seed smoke on GPU
  - keep the experiment small enough to run immediately and inspect manually

### Baseline

- Keep the current strong non-persistent branch:
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `structure_init_mode = patel_kappa`
  - `direction_init_mode = random`
  - `directional_prior_mode = patel`
  - `directional_kappa_gate = True`
  - `directional_schedule = plateau`
  - `main_loss_weight = 1.0`
  - `kappa_logit_bias_scale = 0.0`
  - `direction_logit_bias_scale = 0.0`

### Treatment

- Enable the new residual-style persistent priors:
  - same branch as baseline, except
  - `direction_init_mode = zeros`
  - `kappa_logit_bias_scale > 0`
  - `direction_logit_bias_scale > 0`
- Reasoning:
  - `direction_init_mode=zeros` avoids double-counting a strong Patel direction
    both as initialization and as persistent bias
  - persistent `kappa/tau` priors make the trainable branch behave more like a
    residual corrector around Patel rather than a free graph learner

### Planned Readout

- For each run record:
  - training command
  - result directory
  - shell log path
  - top-`k` directional evaluation on `learned_adjacency_causal.csv`
  - quick interpretation
- First-round decision rule:
  - if the treatment is clearly worse even on this cheap smoke, keep the code
    path but do not prioritize a larger sweep
  - if the treatment is competitive or slightly better, extend to a multi-seed
    synthetic follow-up before touching the `fMRI.csv` branch

### 2026-03-27 First-Round Execution

- Actual setup:
  - dataset:
    - `sim3.csv`
  - GT:
    - `h3.txt`
  - device:
    - `cuda`
  - seed:
    - `11`
  - epochs:
    - `30`
  - evaluator:
    - `GraphExp/test_eval.py --top_k 18`

- Baseline command:
  - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\sim3.csv --device cuda --epochs 30 --pretrain_epochs 50 --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --structure_parameterization support_direction --fixed_support_mask_mode maxgap_kappa --direction_init_mode random --structure_init_mode patel_kappa --structure_init_scale 0.5 --adj_activation sigmoid --directional_prior_mode patel --directional_schedule plateau --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratio 0.01 --lambda_l1 0.02 --optimizer_step_mode subject --main_loss_weight 1.0 --selection_agreement_weight 0.0 --seed 11 --log_interval 10`
- Baseline artifacts:
  - run dir:
    - `GraphExp/results/run_20260327_160233`
  - shell log:
    - `GraphExp/results/residual_patel_baseline_sim3_seed11_smoke_shell.log`

- Initial treatment command:
  - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\sim3.csv --device cuda --epochs 30 --pretrain_epochs 50 --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --structure_parameterization support_direction --fixed_support_mask_mode maxgap_kappa --direction_init_mode zeros --structure_init_mode patel_kappa --structure_init_scale 0.5 --adj_activation sigmoid --directional_prior_mode patel --directional_schedule plateau --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratio 0.01 --lambda_l1 0.02 --optimizer_step_mode subject --main_loss_weight 1.0 --selection_agreement_weight 0.0 --kappa_logit_bias_scale 0.3 --direction_logit_bias_scale 0.3 --seed 11 --log_interval 10`
- Initial treatment artifacts:
  - run dir:
    - `GraphExp/results/run_20260327_160527`
  - shell log:
    - `GraphExp/results/residual_patel_treatment_sim3_seed11_smoke_shell.log`

### 2026-03-27 First-Round Hotfix

- Observation from the initial treatment smoke:
  - training-side quality agreement collapsed to the wrong direction despite
    using the same Patel tau supervision as baseline
  - the failure was too severe to read as a normal hyperparameter issue
- Root-cause reading:
  - the new persistent `direction_logit_bias_prior` was fed into the DDM
    direction branch in causal convention
  - but the branch itself operates in the model's raw/internal convention
- Hotfix:
  - transpose the persistent tau prior before passing it into `DDM`
- Corrected treatment rerun:
  - same command / same hyperparameters as above after the hotfix
- Corrected treatment artifacts:
  - run dir:
    - `GraphExp/results/run_20260327_160927`
  - shell log:
    - `GraphExp/results/residual_patel_treatment_sim3_seed11_smoke_fixed_shell.log`

### 2026-03-27 First-Round Result Summary

- Baseline (`run_20260327_160233`)
  - exported epoch:
    - `11`
  - training proxy:
    - `best score_only quality = 0.4460`
  - evaluator on exported best:
    - `precision = 0.8889`
    - `recall = 0.8889`
    - `F1 = 0.8889`
    - `TP/FP/FN = 16/2/2`
  - evaluator on final epoch:
    - `precision = 0.8333`
    - `recall = 0.8333`
    - `F1 = 0.8333`
    - `TP/FP/FN = 15/3/3`

- Initial treatment before the hotfix (`run_20260327_160527`)
  - exported epoch:
    - `7`
  - training proxy:
    - `best score_only quality = 0.3537`
  - evaluator on exported best:
    - `precision = 0.2222`
    - `recall = 0.2222`
    - `F1 = 0.2222`
    - `TP/FP/FN = 4/14/14`
  - reading:
    - this run is not comparable to baseline as a mechanism result
    - it mainly served to reveal the raw/causal convention bug in the new
      persistent direction-prior path

- Corrected treatment after the hotfix (`run_20260327_160927`)
  - exported epoch:
    - `7`
  - training proxy:
    - `best score_only quality = 0.3537`
  - evaluator on exported best:
    - `precision = 0.7778`
    - `recall = 0.7778`
    - `F1 = 0.7778`
    - `TP/FP/FN = 14/4/4`
  - evaluator on final epoch:
    - `precision = 0.7778`
    - `recall = 0.7778`
    - `F1 = 0.7778`
    - `TP/FP/FN = 14/4/4`

### 2026-03-27 First-Round Interpretation

- The hotfix was necessary:
  - without it, the new treatment path mostly learned the wrong direction
  - so the first treatment run should be read as an implementation audit, not
    as a scientific negative result
- After the convention fix, the residual branch becomes reasonable but still
  underperforms the baseline on this smoke:
  - exported `F1: 0.8889 -> 0.7778`
  - quality proxy: `0.4460 -> 0.3537`
- The main treatment failure mode after the hotfix is not global sign reversal.
- Instead it is **low asymmetry / low margin retention**:
  - many treatment top-18 predictions have extremely small margins
  - example best-run margins are mostly around `1e-4 - 1e-2`
  - by contrast, the baseline exported run still contains several large,
    cleanly separated directional margins
- Training logs support the same reading:
  - corrected treatment keeps a large raw directional loss
    - `raw_dir_loss ≈ 0.928`
  - yet exported adjacency asymmetry remains tiny
    - `dir_margin ≈ 0.005`
- Working interpretation:
  - the current residual treatment is probably **over-anchored to a weak Patel
    tau magnitude scale**
  - with `direction_init_mode=zeros` and moderate persistent bias
    (`direction_logit_bias_scale=0.3`), the branch stays too close to a
    low-margin regime instead of amplifying useful direction contrast

### 2026-03-27 First-Round Decision

- Decision:
  - `mixed / not yet competitive`
- Practical conclusion:
  - keep the residual-Patel code path
  - do **not** promote this exact setting as a new baseline
  - if this branch is continued, the next follow-up should target one of:
    - larger `direction_logit_bias_scale`
    - nonzero random direction residual init on top of the persistent prior
    - support-only persistent prior first (`kappa` on, `tau` off) to isolate
      whether the underperformance is specifically in the direction branch

### 2026-03-27 Follow-Up: Support-Only Persistent Prior (`kappa` on, `tau` off)

- Objective:
  - isolate whether the previous underperformance came mainly from the new
    persistent direction prior rather than from the support-side persistent
    kappa prior
- Setup:
  - reuse the same `sim3` / `seed=11` / `30`-epoch smoke
  - keep the baseline direction branch unchanged
  - only turn on:
    - `kappa_logit_bias_scale = 0.3`
  - keep:
    - `direction_logit_bias_scale = 0.0`
    - `direction_init_mode = random`

- Command:
  - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\sim3.csv --device cuda --epochs 30 --pretrain_epochs 50 --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --structure_parameterization support_direction --fixed_support_mask_mode maxgap_kappa --direction_init_mode random --structure_init_mode patel_kappa --structure_init_scale 0.5 --adj_activation sigmoid --directional_prior_mode patel --directional_schedule plateau --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratio 0.01 --lambda_l1 0.02 --optimizer_step_mode subject --main_loss_weight 1.0 --selection_agreement_weight 0.0 --kappa_logit_bias_scale 0.3 --direction_logit_bias_scale 0.0 --seed 11 --log_interval 10`

- Artifacts:
  - run dir:
    - `GraphExp/results/run_20260327_162130`
  - shell log:
    - `GraphExp/results/residual_patel_kappa_only_sim3_seed11_smoke_shell.log`

- Result summary:
  - exported epoch:
    - `11`
  - training proxy:
    - `best score_only quality = 0.4480`
  - evaluator on exported best:
    - `precision = 0.8889`
    - `recall = 0.8889`
    - `F1 = 0.8889`
    - `TP/FP/FN = 16/2/2`
  - evaluator on final epoch:
    - `precision = 0.8333`
    - `recall = 0.8333`
    - `F1 = 0.8333`
    - `TP/FP/FN = 15/3/3`

- Comparison against the baseline smoke (`run_20260327_160233`):
  - exported best:
    - identical directional evaluation
      - `F1 = 0.8889`
      - `TP/FP/FN = 16/2/2`
  - final epoch:
    - identical directional evaluation
      - `F1 = 0.8333`
      - `TP/FP/FN = 15/3/3`
  - quality proxy:
    - slightly higher than baseline
      - `0.4460 -> 0.4480`

- Interpretation:
  - this follow-up weakens the hypothesis that "persistent Patel priors are
    broadly harmful" in this branch
  - at least on this smoke:
    - support-side persistent kappa prior is essentially neutral-to-slightly
      helpful
    - the earlier degradation appears much more specific to the persistent tau
      direction path
  - practical reading:
    - the residual-Patel idea should now be decomposed into two subclaims:
      - `kappa` persistent prior:
        - currently plausible
      - `tau` persistent prior:
        - currently problematic at the tested scale / parameterization

- Updated next step recommendation:
  - if this branch continues, prioritize:
    - multi-seed confirmation of `kappa-only`
    - then a narrower `tau` follow-up such as:
      - smaller `direction_logit_bias_scale`
      - or random direction residual init plus weaker persistent tau bias

### 2026-03-27 Multi-Seed Confirmation: `kappa`-Only Persistent Prior

- Objective:
  - extend the previous `seed=11` smoke into a small multi-seed confirmation
  - compare:
    - baseline:
      - `kappa_logit_bias_scale = 0.0`
    - support-only persistent prior:
      - `kappa_logit_bias_scale = 0.3`
  - keep the rest of the branch fixed so the comparison isolates the support
    prior

- Runner:
  - `GraphExp/run_cross_pred_v1_final_only_compare.py`

- Command:
  - `python .\run_cross_pred_v1_final_only_compare.py --csv_path ..\fMRI_dataset\sim3.csv --gt_path ..\fMRI_dataset\h3.txt --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 18 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.0,0.3 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag residual_kappa_only_confirm`

- Artifacts:
  - shell log:
    - `GraphExp/results/residual_patel_kappa_only_confirm_runner_shell.log`
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_172812_residual_kappa_only_confirm.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_172812_residual_kappa_only_confirm_aggregate.csv`

### 2026-03-27 Multi-Seed Aggregate Summary

- Baseline (`kappa_bias = 0.0`)
  - `diff = 1.1880 +/- 0.0028`
  - `best_strict = 0.8519 +/- 0.0262`
  - `strict@0 = 0.8148 +/- 0.0262`
  - `gap@0 = 0.0370`
  - `gt_margin_gap = 6.6343e-02`
  - `p90 = 7.0161e-02 +/- 6.3921e-03`
  - `eff_par = 1.85`
  - `near0(<1e-2) = 83.17%`
  - failure mode:
    - `mixed_or_partial` in `3/3`

- `kappa`-only persistent prior (`kappa_bias = 0.3`)
  - `diff = 1.1876 +/- 0.0032`
  - `best_strict = 0.8148 +/- 0.0693`
  - `strict@0 = 0.7963 +/- 0.0524`
  - `gap@0 = 0.0185`
  - `gt_margin_gap = 7.0573e-02`
  - `p90 = 7.4711e-02 +/- 1.5634e-03`
  - `eff_par = 1.86`
  - `near0(<1e-2) = 83.81%`
  - failure mode:
    - `mixed_or_partial` in `3/3`

### 2026-03-27 Multi-Seed Interpretation

- The earlier single-seed read was slightly optimistic.
- Across `3` seeds, the `kappa`-only branch is best described as:
  - **close to neutral**
  - not clearly harmful
  - but also not a consistent improvement
- Relative to the baseline:
  - `best_strict` and `strict@0` are slightly lower on average
  - but `gap@0` is smaller
    - `0.0370 -> 0.0185`
  - and the upper-margin tail is slightly stronger / more stable
    - `p90: 0.0702 -> 0.0747`
- Working interpretation:
  - support-side persistent `kappa` prior looks more like a mild
    stabilization trade-off than a headline gain
  - this does **not** support dropping the old baseline in favor of
    `kappa_bias=0.3`
  - but it also does not read like the main failure source

### 2026-03-27 Updated Direction For This Branch

- Current split after the multi-seed check:
  - `kappa` persistent prior:
    - approximately neutral / mildly stabilizing
    - not the main blocker
  - `tau` persistent prior:
    - still the main unresolved issue
- Practical next step:
  - if the residual-Patel branch continues, the next efficient test should be a
    **small-`tau` sweep** while keeping:
    - `kappa_logit_bias_scale = 0.0` or optionally the neutral `0.3`
    - `direction_init_mode = random`
  - preferred first sweep:
    - `direction_logit_bias_scale in {0.05, 0.10, 0.20}`

### 2026-03-27 Follow-Up: Support-Only Persistent Prior (`kappa` on, `tau` off)

- Objective:
  - isolate whether the first-round treatment underperformance came from:
    - the support-side persistent Patel prior
    - or specifically from the new persistent direction-prior path

- Setup:
  - same smoke setup as the baseline above:
    - dataset:
      - `sim3.csv`
    - GT:
      - `h3.txt`
    - seed:
      - `11`
    - epochs:
      - `30`
    - device:
      - `cuda`
  - config change relative to baseline:
    - `kappa_logit_bias_scale = 0.3`
    - `direction_logit_bias_scale = 0.0`
    - keep `direction_init_mode = random`
  - command:
    - `python .\main_structure_learning.py --csv_path ..\fMRI_dataset\sim3.csv --device cuda --epochs 30 --pretrain_epochs 50 --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --structure_parameterization support_direction --fixed_support_mask_mode maxgap_kappa --direction_init_mode random --structure_init_mode patel_kappa --structure_init_scale 0.5 --adj_activation sigmoid --directional_prior_mode patel --directional_schedule plateau --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratio 0.01 --lambda_l1 0.02 --optimizer_step_mode subject --main_loss_weight 1.0 --selection_agreement_weight 0.0 --kappa_logit_bias_scale 0.3 --direction_logit_bias_scale 0.0 --seed 11 --log_interval 10`

- Artifacts:
  - run dir:
    - `GraphExp/results/run_20260327_162130`
  - shell log:
    - `GraphExp/results/residual_patel_kappa_only_sim3_seed11_smoke_shell.log`

### 2026-03-27 Support-Only Result Summary

- Exported / best checkpoint:
  - exported epoch:
    - `11`
  - training proxy:
    - `best score_only quality = 0.4480`
  - evaluator:
    - `precision = 0.8889`
    - `recall = 0.8889`
    - `F1 = 0.8889`
    - `TP/FP/FN = 16/2/2`

- Final epoch:
  - evaluator:
    - `precision = 0.8333`
    - `recall = 0.8333`
    - `F1 = 0.8333`
    - `TP/FP/FN = 15/3/3`

### 2026-03-27 Support-Only Interpretation

- This follow-up is effectively **baseline-equivalent** on the first smoke:
  - baseline exported best:
    - `F1 = 0.8889`
    - `best quality = 0.4460`
  - support-only persistent prior:
    - `F1 = 0.8889`
    - `best quality = 0.4480`
- So the first-round degradation is **not** explained by the support-side
  persistent Patel prior alone.
- The strongest current reading is:
  - `kappa` persistent prior is safe, or at least close to neutral, in this
    branch
  - the underperformance is concentrated in the persistent `tau` direction-prior
    path

### 2026-03-27 Updated Branch Status

- Current evidence split:
  - persistent `kappa` support prior:
    - `PASS / neutral-to-safe` on the first `sim3` smoke
  - persistent `tau` direction prior:
    - `not yet competitive`
    - still pushes the branch into a low-margin regime under the tested setting
- Practical next step if this branch continues:
  - keep `kappa` persistent prior as an allowed option
  - treat the `tau` persistent prior as the actual unresolved subproblem
  - if revisiting `tau`, prefer:
    - random direction residual init instead of zeros
    - or a smaller, more weakly anchoring direction bias sweep

### 2026-03-27 Follow-Up: Small Persistent-`tau` Sweep

- Objective:
  - re-test the persistent Patel-`tau` path after moving away from the earlier
    `direction_init_mode=zeros` failure mode
  - isolate whether a **small** direction-logit bias can help when the residual
    direction field still starts from `random`

- Setup:
  - dataset / GT:
    - `sim3.csv`
    - `h3.txt`
  - seeds:
    - `11,22,33`
  - shared config:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `structure_init_mode = patel_kappa`
    - `direction_init_mode = random`
    - `directional_prior_mode = patel`
    - `directional_schedule = plateau`
    - `directional_kappa_gate = on`
    - `directional_target_ratio = 0.01`
    - `lambda_l1 = 0.02`
    - `kappa_logit_bias_scale = 0.0`
  - sweep axis:
    - `direction_logit_bias_scale in {0.0, 0.05, 0.10, 0.20}`
  - runner command:
    - `python .\run_cross_pred_v1_final_only_compare.py --csv_path ..\fMRI_dataset\sim3.csv --gt_path ..\fMRI_dataset\h3.txt --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 18 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.0 --direction_logit_bias_scales 0.0,0.05,0.1,0.2 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag residual_tau_small_sweep`

- Artifacts:
  - raw runner csv:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_181414_residual_tau_small_sweep.csv`
  - shell log:
    - `GraphExp/results/residual_patel_tau_small_sweep_runner_shell.log`
  - parsed summary:
    - `GraphExp/results/residual_patel_tau_small_sweep_parsed_summary.csv`
  - parsed aggregate:
    - `GraphExp/results/residual_patel_tau_small_sweep_parsed_aggregate.csv`

### 2026-03-27 Tau Sweep Execution Note

- All `12` train/eval runs finished.
- The runner then failed during post-run aggregation because the new
  `direction_logit_bias_scale` field shifted the aggregate sort indices:
  - traceback key line:
    - `ValueError: could not convert string to float: 'patel_kappa'`
- The aggregation bug was fixed afterward in
  `GraphExp/run_cross_pred_v1_final_only_compare.py`.
- To avoid re-running finished jobs, the completed shell log was parsed into the
  summary / aggregate csv files listed above.

### 2026-03-27 Tau Sweep Result Summary

- `tau_bias = 0.00`
  - `best_strict = 0.8518 +/- 0.0262`
  - `strict@0 = 0.8148 +/- 0.0262`
  - `gap@0 = 0.0371`
  - `gt_margin_gap = 6.6344e-02`
  - `p90 = 7.0161e-02 +/- 6.3923e-03`

- `tau_bias = 0.05`
  - `best_strict = 0.8333 +/- 0.0454`
  - `strict@0 = 0.8148 +/- 0.0693`
  - `gap@0 = 0.0185`
  - `gt_margin_gap = 6.4392e-02`
  - `p90 = 7.1702e-02 +/- 7.6890e-03`

- `tau_bias = 0.10`
  - `best_strict = 0.8148 +/- 0.0262`
  - `strict@0 = 0.7778 +/- 0.0454`
  - `gap@0 = 0.0371`
  - `gt_margin_gap = 6.9426e-02`
  - `p90 = 6.9319e-02 +/- 2.1457e-03`

- `tau_bias = 0.20`
  - `best_strict = 0.8518 +/- 0.0262`
  - `strict@0 = 0.7963 +/- 0.0524`
  - `gap@0 = 0.0556`
  - `gt_margin_gap = 6.3080e-02`
  - `p90 = 7.5963e-02 +/- 3.8547e-03`

### 2026-03-27 Tau Sweep Interpretation

- This sweep does **not** show a clean winner over the `tau_bias=0` baseline.
- What changed:
  - `tau_bias=0.05` reduces `gap@0`
  - `tau_bias=0.20` improves the upper-margin tail (`p90`)
- What did **not** improve cleanly:
  - no tested `tau_bias` beats baseline on both `best_strict` and `strict@0`
  - the direction field still stays in the same broad `mixed_or_partial`
    regime across all seeds
  - `near0` remains around `83%`, so the model is still not leaving the
    low-margin basin in a decisive way
- Working conclusion:
  - persistent Patel-`tau` bias is still **not a clear gain** under the current
    parameterization
  - the residual-Patel branch remains plausible in concept, but the support-side
    `kappa` prior is currently the only part that looks safe
  - if this branch continues, the next efficient test should be an
    **ultra-small `tau` bias** on top of the safe `kappa` prior rather than a
    stronger persistent direction anchor

### 2026-03-27 Follow-Up: Full Residual Patel Fusion With Tiny `tau`

- Objective:
  - test the **actual combined branch** instead of the isolated `tau` sweep
  - keep the support-side persistent Patel prior at the previously safe setting
    (`kappa_bias=0.3`)
  - only probe **ultra-small** direction persistence to see whether the
    full residual-Patel formulation wants a very weak directional anchor rather
    than none or a stronger one

- Setup:
  - dataset / GT:
    - `sim3.csv`
    - `h3.txt`
  - seeds:
    - `11,22,33`
  - shared config:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `structure_init_mode = patel_kappa`
    - `direction_init_mode = random`
    - `directional_prior_mode = patel`
    - `directional_schedule = plateau`
    - `directional_kappa_gate = on`
    - `directional_target_ratio = 0.01`
    - `lambda_l1 = 0.02`
    - `kappa_logit_bias_scale = 0.3`
  - sweep axis:
    - `direction_logit_bias_scale in {0.0, 0.02, 0.05}`
  - runner command:
    - `python .\run_cross_pred_v1_final_only_compare.py --csv_path ..\fMRI_dataset\sim3.csv --gt_path ..\fMRI_dataset\h3.txt --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 18 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.3 --direction_logit_bias_scales 0.0,0.02,0.05 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag residual_kappa03_tau_tiny`

- Artifacts:
  - shell log:
    - `GraphExp/results/residual_patel_kappa03_tau_tiny_runner_shell.log`
  - summary csv:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_190641_residual_kappa03_tau_tiny.csv`
  - aggregate csv:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_190641_residual_kappa03_tau_tiny_aggregate.csv`

### 2026-03-27 Tiny-`tau` Result Summary

- `kappa_bias = 0.3`, `tau_bias = 0.00`
  - `best_strict = 0.8148 +/- 0.0693`
  - `strict@0 = 0.7963 +/- 0.0524`
  - `gap@0 = 0.0185`
  - `gt_margin_gap = 7.0573e-02`
  - `p90 = 7.4711e-02 +/- 1.5634e-03`
  - `near0 = 83.81%`

- `kappa_bias = 0.3`, `tau_bias = 0.02`
  - `best_strict = 0.8333 +/- 0.0454`
  - `strict@0 = 0.8148 +/- 0.0693`
  - `gap@0 = 0.0185`
  - `gt_margin_gap = 6.3791e-02`
  - `p90 = 7.7669e-02 +/- 2.5931e-03`
  - `near0 = 83.17%`

- `kappa_bias = 0.3`, `tau_bias = 0.05`
  - `best_strict = 0.8333 +/- 0.0454`
  - `strict@0 = 0.8148 +/- 0.0693`
  - `gap@0 = 0.0185`
  - `gt_margin_gap = 6.8243e-02`
  - `p90 = 7.5558e-02 +/- 6.8727e-04`
  - `near0 = 83.81%`

### 2026-03-27 Tiny-`tau` Interpretation

- This is the first **non-negative** signal for the full residual-Patel branch:
  - both `tau_bias=0.02` and `tau_bias=0.05` outperform the local
    `kappa_bias=0.3, tau_bias=0.0` baseline on `best_strict`
  - both also recover `strict@0` from `0.7963` back to `0.8148`
- The best point in this narrow follow-up is:
  - `kappa_bias=0.3`, `tau_bias=0.02`
  - it gives the strongest `p90`, the lowest `near0`, and the best
    `gt_margin_gap` among the three tested settings
- Important restraint:
  - this is **not** a decisive global win yet
  - compared with the earlier no-persistent-`tau` / no-persistent-`kappa`
    baseline, the branch is now roughly competitive on `strict@0`, but still
    not clearly ahead on `best_strict`
- Updated branch conclusion:
  - the combined idea seems viable only when the Patel direction prior is used
    as a **very weak residual bias**, not a moderate anchor
  - practical recommendation for future runs:
    - keep `kappa_logit_bias_scale` as the safe persistent prior
    - if enabling persistent `tau`, keep it in the **tiny** range
      near `0.02`
    - do **not** use the earlier moderate `tau` settings (`0.1` to `0.3`) as
      the default
- Extra check completed:
  - the updated runner now completes summary and aggregate export successfully,
    so the earlier aggregate-sort bug is resolved in the exercised path

### 2026-03-27 Next Formal Check: Three-Way `sim3` Confirmation

- Objective:
  - test whether the currently best residual-Patel candidate
    (`kappa_bias=0.3`, `tau_bias=0.02`) is **stably** competitive once the
    comparison is tightened to the three most relevant settings
  - avoid wasting budget on already-disfavored moderate-`tau` settings

- Comparison set:
  - baseline:
    - `kappa_logit_bias_scale = 0.0`
    - `direction_logit_bias_scale = 0.0`
  - support-only persistent prior:
    - `kappa_logit_bias_scale = 0.3`
    - `direction_logit_bias_scale = 0.0`
  - full residual-Patel candidate:
    - `kappa_logit_bias_scale = 0.3`
    - `direction_logit_bias_scale = 0.02`

- Setup:
  - dataset / GT:
    - `sim3.csv`
    - `h3.txt`
  - seeds:
    - `11,22,33,44,55`
  - shared config:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `structure_init_mode = patel_kappa`
    - `direction_init_mode = random`
    - `directional_prior_mode = patel`
    - `directional_schedule = plateau`
    - `directional_kappa_gate = on`
    - `directional_target_ratio = 0.01`
    - `lambda_l1 = 0.02`
    - `optimizer_step_mode = subject`
    - `main_loss_weight = 1.0`
    - `selection_agreement_weight = 0.0`

- Decision targets:
  - `best_strict`
  - `strict@0`
  - `gap@0`
  - `gt_margin_gap`
  - `near0_pct`

- Planned commands:
  - baseline-only:
    - `python .\run_cross_pred_v1_final_only_compare.py --csv_path ..\fMRI_dataset\sim3.csv --gt_path ..\fMRI_dataset\h3.txt --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 18 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.0 --direction_logit_bias_scales 0.0 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33,44,55 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag residual_threeway_baseline5`
  - residual branch check:
    - `python .\run_cross_pred_v1_final_only_compare.py --csv_path ..\fMRI_dataset\sim3.csv --gt_path ..\fMRI_dataset\h3.txt --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 18 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.3 --direction_logit_bias_scales 0.0,0.02 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33,44,55 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag residual_threeway_kappa03_taucheck`

### 2026-03-27 Three-Way `sim3` Confirmation Results

- Artifacts:
  - baseline shell log:
    - `GraphExp/results/residual_threeway_baseline5_runner_shell.log`
  - baseline summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260327_211907_residual_threeway_baseline5.csv`
  - baseline aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260327_211907_residual_threeway_baseline5_aggregate.csv`
  - residual shell log:
    - `GraphExp/results/residual_threeway_kappa03_taucheck_runner_shell.log`
  - residual summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260327_213615_residual_threeway_kappa03_taucheck.csv`
  - residual aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260327_213615_residual_threeway_kappa03_taucheck_aggregate.csv`

- Aggregate summary:
  - old baseline:
    - `kappa=0.0`, `tau=0.0`
    - `best_strict = 0.8111 +/- 0.0567`
    - `strict@0 = 0.8000 +/- 0.0444`
    - `gap@0 = 0.0111`
    - `gt_margin_gap = 6.6777e-02`
    - `p90 = 7.6280e-02 +/- 1.2457e-02`
    - `near0 = 83.05%`
  - support-only persistent prior:
    - `kappa=0.3`, `tau=0.0`
    - `best_strict = 0.7889 +/- 0.0648`
    - `strict@0 = 0.7889 +/- 0.0544`
    - `gap@0 = 0.0000`
    - `gt_margin_gap = 7.0657e-02`
    - `p90 = 8.0193e-02 +/- 1.3384e-02`
    - `near0 = 83.43%`
  - full residual-Patel candidate:
    - `kappa=0.3`, `tau=0.02`
    - `best_strict = 0.8111 +/- 0.0444`
    - `strict@0 = 0.8000 +/- 0.0667`
    - `gap@0 = 0.0111`
    - `gt_margin_gap = 6.4566e-02`
    - `p90 = 8.3753e-02 +/- 1.2863e-02`
    - `near0 = 83.05%`

- Seed-level read on the core metrics:
  - `seed 11`:
    - baseline:
      - `best = 0.8889`
      - `strict@0 = 0.8333`
    - `kappa=0.3`, `tau=0.0`:
      - `best = 0.8889`
      - `strict@0 = 0.8333`
    - `kappa=0.3`, `tau=0.02`:
      - `best = 0.8889`
      - `strict@0 = 0.8889`
  - `seed 22`:
    - all three settings are identical on `best` and `strict@0`
      - `0.8333 / 0.8333`
  - `seed 33`:
    - baseline:
      - `best = 0.8333`
      - `strict@0 = 0.7778`
    - `kappa=0.3`, `tau=0.0`:
      - `best = 0.7222`
      - `strict@0 = 0.7222`
    - `kappa=0.3`, `tau=0.02`:
      - `best = 0.7778`
      - `strict@0 = 0.7222`
  - `seed 44`:
    - baseline:
      - `best = 0.7778`
      - `strict@0 = 0.8333`
    - `kappa=0.3`, `tau=0.0`:
      - `best = 0.7222`
      - `strict@0 = 0.8333`
    - `kappa=0.3`, `tau=0.02`:
      - `best = 0.7778`
      - `strict@0 = 0.8333`
  - `seed 55`:
    - baseline:
      - `best = 0.7222`
      - `strict@0 = 0.7222`
    - `kappa=0.3`, `tau=0.0`:
      - `best = 0.7778`
      - `strict@0 = 0.7222`
    - `kappa=0.3`, `tau=0.02`:
      - `best = 0.7778`
      - `strict@0 = 0.7222`

### 2026-03-27 Three-Way Interpretation

- The key negative result is now clear:
  - `kappa=0.3`, `tau=0.0` does **not** hold up as a default
  - on `5` seeds it is below the old baseline on both `best_strict` and
    `strict@0`
- The key positive result is also clearer now:
  - adding the tiny persistent direction prior (`tau=0.02`) largely repairs the
    support-only degradation
  - relative to `kappa=0.3`, `tau=0.0`, it recovers:
    - `best_strict: 0.7889 -> 0.8111`
    - `strict@0: 0.7889 -> 0.8000`
    - `gap@0: 0.0000 -> 0.0111`
- But compared with the old baseline, the branch is still best described as:
  - **competitive, not superior**
  - the mean `best_strict` and `strict@0` are exactly tied with the old
    baseline at this operating point
  - the residual candidate does show a somewhat stronger upper-margin tail:
    - `p90: 0.0763 -> 0.0838`
  - and a slightly better `gt_margin_gap`:
    - `0.0668 -> 0.0646`
  - but those gains are not yet converting into a higher average strict F1
- Updated practical conclusion:
  - if we insist on a residual-Patel formulation, the preferred setting is now:
    - `kappa_logit_bias_scale = 0.3`
    - `direction_logit_bias_scale = 0.02`
  - however, on current `sim3` evidence, this branch is **not yet justified as a
    replacement** for the old `kappa=0.0`, `tau=0.0` baseline
  - what it has earned is a narrower claim:
    - strong / moderate persistent `tau` is wrong
    - support-only persistent `kappa` is not enough
    - **tiny `tau` + persistent `kappa` is the only residual-Patel variant that
      stays competitive**

### 2026-03-27 Next Transfer Check: `sim4` Pilot

- Objective:
  - test whether the only still-viable residual-Patel setting from `sim3`
    transfers at all to the larger synthetic graph
  - decide whether the branch deserves a later full-budget `sim4` formal run
    or should be closed here

- Why a pilot instead of a full formal:
  - on `sim3`, the candidate is only **competitive / tied**, not clearly better
  - so the next rational step is a cheap transfer check before committing to a
    `5`-seed / `100`-epoch `sim4` sweep

- Comparison set:
  - old baseline:
    - `kappa_logit_bias_scale = 0.0`
    - `direction_logit_bias_scale = 0.0`
  - residual-Patel candidate:
    - `kappa_logit_bias_scale = 0.3`
    - `direction_logit_bias_scale = 0.02`

- Pilot setup:
  - dataset / GT:
    - `sim4.csv`
    - `h4.txt`
  - seeds:
    - `11,22,33`
  - epochs:
    - `30`
  - evaluation size:
    - `top_k_edges = 61`
  - shared config:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `structure_init_mode = patel_kappa`
    - `direction_init_mode = random`
    - `directional_prior_mode = patel`
    - `directional_schedule = plateau`
    - `directional_kappa_gate = on`
    - `directional_target_ratio = 0.01`
    - `lambda_l1 = 0.02`
    - `optimizer_step_mode = subject`
    - `main_loss_weight = 1.0`
    - `selection_agreement_weight = 0.0`

- Planned commands:
  - baseline:
    - `python .\run_cross_pred_v1_final_only_compare.py --csv_path ..\fMRI_dataset\sim4.csv --gt_path ..\fMRI_dataset\h4.txt --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 61 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.0 --direction_logit_bias_scales 0.0 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag residual_sim4_transfer_baseline3`
  - residual candidate:
    - `python .\run_cross_pred_v1_final_only_compare.py --csv_path ..\fMRI_dataset\sim4.csv --gt_path ..\fMRI_dataset\h4.txt --pretrain_checkpoint .\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 61 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.3 --direction_logit_bias_scales 0.02 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag residual_sim4_transfer_tau002_3`

### 2026-03-27 `sim4` Transfer Pilot Results

- Artifacts:
  - baseline shell log:
    - `GraphExp/results/residual_sim4_transfer_baseline3_runner_shell.log`
  - baseline summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_221657_residual_sim4_transfer_baseline3.csv`
  - baseline aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_221657_residual_sim4_transfer_baseline3_aggregate.csv`
  - residual shell log:
    - `GraphExp/results/residual_sim4_transfer_tau002_3_runner_shell.log`
  - residual summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_224930_residual_sim4_transfer_tau002_3.csv`
  - residual aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_224930_residual_sim4_transfer_tau002_3_aggregate.csv`

- Aggregate summary:
  - old baseline:
    - `kappa=0.0`, `tau=0.0`
    - `best_strict = 0.8197 +/- 0.0000`
    - `strict@0 = 0.7978 +/- 0.0204`
    - `gap@0 = 0.0219`
    - `gt_margin_gap = 2.3904e-02`
    - `p90 = 0.0000e+00`
    - `near0 = 96.24%`
    - failure mode:
      - `symmetric_collapse: 3/3`
  - residual-Patel candidate:
    - `kappa=0.3`, `tau=0.02`
    - `best_strict = 0.8142 +/- 0.0204`
    - `strict@0 = 0.7814 +/- 0.0155`
    - `gap@0 = 0.0328`
    - `gt_margin_gap = 2.7083e-02`
    - `p90 = 0.0000e+00`
    - `near0 = 96.11%`
    - failure mode:
      - `symmetric_collapse: 3/3`

- Seed-level read:
  - `seed 11`
    - baseline:
      - `best = 0.8197`
      - `strict@0 = 0.7705`
    - residual candidate:
      - `best = 0.8197`
      - `strict@0 = 0.7705`
  - `seed 22`
    - baseline:
      - `best = 0.8197`
      - `strict@0 = 0.8197`
    - residual candidate:
      - `best = 0.8361`
      - `strict@0 = 0.7705`
  - `seed 33`
    - baseline:
      - `best = 0.8197`
      - `strict@0 = 0.8033`
    - residual candidate:
      - `best = 0.7869`
      - `strict@0 = 0.8033`

### 2026-03-27 `sim4` Transfer Interpretation

- This is a **negative transfer** result for the residual-Patel branch.
- The essential observation is not just that the candidate failed to beat the
  baseline, but that both settings remained trapped in the same failure regime:
  - `symmetric_collapse` in all `3/3` seeds
  - `p90 = 0`
  - `near0` around `96%`
- Relative to the baseline, the residual candidate does **not** improve the
  metrics we actually care about on `sim4`:
  - `best_strict` is slightly lower
    - `0.8197 -> 0.8142`
  - `strict@0` is lower
    - `0.7978 -> 0.7814`
  - the model still does not escape the zero-margin basin
- Updated branch conclusion after `sim4`:
  - the residual-Patel idea remains conceptually coherent
  - but empirically, under the current implementation, it behaves like a
    **local sim3-compatible tweak**, not a robust scaling improvement
  - it is therefore **not justified as the next mainline branch**
- Practical decision:
  - close the current residual-Patel branch as a primary optimization target
  - keep the code path and the documented best setting
    (`kappa=0.3`, `tau=0.02`) as a reference variant
  - do not spend a larger `sim4` formal budget on this branch unless a new
    mechanism is added that directly targets `symmetric_collapse`

## 2026-03-28 Next Mainline: Time-Supervised Anti-Collapse Branch

### Why This Is A New Branch

- This branch is **not** a revival of the earlier Option B / DiffuGC-style
  idea.
- Phase 0 / 0B / 0C already established:
  - diffusion-side learning by itself does **not** expose a strong endogenous
    direction signal
  - offline multi-lag prediction is weak under the earlier simple predictor
    test
  - timestep-gradient and edge-ablation probes also do not show a hidden
    denoiser-side direction mechanism worth promoting
- The new question is narrower and more practical:
  - if we keep the current structure-learning framework, freeze the temporal
    encoder, and preserve `cross_prediction`'s `no_grad` source path,
    can **explicit time-order supervision plus a direct anti-collapse
    constraint** pull the direction branch out of the `D - D^T ~= 0` basin?

### Mainline Decision

- Do **not** continue Patel-strength tuning.
- Do **not** treat residual-Patel as the optimization target.
- Move the mainline to a new branch whose purpose is to break
  `symmetric_collapse` directly.

### Stage-1 Mechanism

- Keep the encoder frozen.
- Keep `cross_prediction` source features under `torch.no_grad()`.
- Keep the current `support_direction + maxgap_kappa` factorization so support
  search space stays controlled.
- Reassign the direction branch to three mechanisms:
  1. Promote future-prediction supervision from weak auxiliary usage toward the
     main direction-learning signal.
  2. Replace single-lag time supervision with **multi-lag** supervision.
  3. Add a direct anti-collapse / margin-floor term on high-support directional
     pairs so the directional contrast is explicitly discouraged from returning
     to `0`.

### Attribution Rules

- Stage 1 should keep the direction sign source as clean as possible.
- In particular:
  - do not rely on persistent Patel-`tau` bias in this branch
  - keep Patel `kappa` only in its support / skeleton role
  - avoid introducing encoder unfreezing at the same time
- This is required so the answer to the first pilot is interpretable:
  - did **time supervision + anti-collapse** break the symmetry basin, or not?

### First Pilot

- Dataset / GT:
  - `sim3.csv`
  - `h3.txt`
- Seeds:
  - `11,22,33`
- Comparison:
  - current old baseline
  - new time-supervised anti-collapse branch
- Primary decision metrics:
  - `near0_pct`
  - `p90`
  - `failure_mode`
- Secondary guardrail metrics:
  - `strict@0`
  - `best_strict`
- Promotion rule:
  - only if the new branch clearly reduces `near0_pct` / lifts `p90` /
    stops ending in `symmetric_collapse` on `sim3`, and does not obviously
    degrade strict F1, should it be promoted to `sim4`

### Stage-2 Boundary

- Only after Stage 1 shows a real anti-collapse effect should we consider
  controlled partial encoder unfreezing.
- Full joint optimization remains explicitly out of scope.

### Dataset Naming Reminder

- `fMRI.csv` is a **synthetic** dataset, not real-data fMRI.
- Its paired GT is `fMRI_dataset/h1.txt`.
- Future notes in this file should continue to treat `fMRI.csv` as synthetic.

## 2026-03-28 Runner Recovery And Smoke For Stage-1 Anti-Collapse

### Objective

- Verify that the new Stage-1 branch can run end-to-end through:
  - runner CLI parsing
  - multi-lag direction prior wiring
  - fixed-weight cross-prediction wiring
  - gated anti-collapse wiring
  - aggregate / comparison / paired CSV export
- This was a **plumbing smoke**, not a model-quality decision experiment.

### Smoke Setup

- Dataset / GT:
  - `fMRI.csv` (synthetic)
  - `h1.txt`
- Scope:
  - `epochs = 1`
  - `subject_limit = 2`
  - `time_limit = 20`
  - `seed = 11`
- Stage-1 smoke knobs:
  - `directional_prior_lags = 1,2`
  - `cross_pred_lags = 1,2`
  - `cross_pred_fixed_weight = 0.1`
  - `anti_collapse_lambda = 0.1`
  - `anti_collapse_margin = 0.02`

### Recovery Notes

- `GraphExp/run_cross_pred_v1_final_only_compare.py` had a broken nested-loop
  indentation after adding the new
  `cross_pred_fixed_weight / anti_collapse` sweep block.
- That syntax issue is now fixed.
- `py_compile` now passes for:
  - `GraphExp/run_cross_pred_v1_final_only_compare.py`
  - `GraphExp/main_structure_learning.py`
- The runner also now preserves the new Stage-1 fields through:
  - per-run rows
  - aggregate rows
  - comparison rows
  - paired rows

### Latest Smoke Artifacts

- Summary:
  - `GraphExp/results/cross_pred_v1_final_only_compare_random_cross_direction_compare_1seeds_20260328_201241_smoke_time_anti_collapse_v2.csv`
- Aggregate:
  - `GraphExp/results/cross_pred_v1_final_only_compare_random_cross_direction_compare_1seeds_20260328_201241_smoke_time_anti_collapse_v2_aggregate.csv`
- Comparison:
  - `GraphExp/results/cross_pred_v1_final_only_compare_random_cross_direction_compare_1seeds_20260328_201241_smoke_time_anti_collapse_v2_comparison.csv`
- Paired:
  - `GraphExp/results/cross_pred_v1_final_only_compare_random_cross_direction_compare_1seeds_20260328_201241_smoke_time_anti_collapse_v2_paired.csv`

### Smoke Result

- Pass/fail answer:
  - **PASS on plumbing**
- All tiny-smoke conditions still show:
  - `symmetric_collapse`
  - `p90 = 0`
  - `near0(<1e-2) = 100%`
- This should **not** be promoted into a mechanism conclusion because:
  - it is only a 1-epoch smoke
  - the purpose was to verify runner recovery, not directional learning quality

### Recovered Old-Baseline Config

- The current old baseline was re-read from the existing formal `sim3`
  baseline aggregate instead of being reconstructed from memory.
- Old baseline config:
  - `structure_init_mode = patel_kappa`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `structure_init_scale = 0.5`
  - `lambda_l1 = 0.02`
  - `kappa_logit_bias_scale = 0.0`
  - `direction_logit_bias_scale = 0.0`
  - `adj_activation = sigmoid`
  - `optimizer_step_mode = subject`
  - `main_loss_weight = 1.0`
  - `selection_agreement_weight = 0.0`
  - `direction_lr_multiplier = 1.0`
  - `freeze_direction_after_epoch = -1`

### Next Formal Pilot

- First real Stage-1 check remains:
  - dataset / GT:
    - `sim3.csv`
    - `h3.txt`
  - seeds:
    - `11,22,33`
  - comparison:
    - current old baseline
    - one single-point time-supervised anti-collapse treatment
- To stay interpretable, the new treatment should inherit the same structural
  backbone as the old baseline:
  - `patel_kappa` init
  - `support_direction`
  - `maxgap_kappa`
  - `direction_init_mode = random`
  - no persistent `tau` bias
- Proposed first single-point treatment:
  - `cross_prediction = on`
  - `directional_prior_mode = lag_corr`
  - `lag_direction_source = raw`
  - `directional_prior_lags = 1,2,3`
  - `cross_pred_lags = 1,2,3`
  - `cross_pred_fixed_weight = 0.1`
  - `directional_target_ratio = 0.01`
  - `anti_collapse_lambda = 0.1`
  - `anti_collapse_margin = 0.02`
- Primary readouts remain:
  - `near0_pct`
  - `p90`
  - `failure_mode`
- Guardrails remain:
  - `strict@0`
  - `best_strict`

### Budget Note

- The current runner still expands
  `cross_pred_conditions x directional_conditions` as a cartesian product.
- For the formal Stage-1 question, the most budget-efficient path is likely:
  - run the old baseline as one 3-seed sweep
  - run the new treatment as one 3-seed sweep
  - compare the two aggregate files directly
- This avoids paying for two extra hybrid conditions that are not part of the
  current experimental question.

### 2026-04-04 - Selector Formalization Update For `random structure init + kappa gate on`

- Fixed training trajectory under discussion:
  - `run_20260403_222030`
  - best/export/final strict:
    - `0.885246 / 0.770492 / 0.852459`
- New selector-only read:
  - the remaining gap on this branch is now specifically selector-side
  - within the current code, `causal_lag_composite` is the cleanest next
    formalization target because it already exposes:
    - `selection_soft_agreement_weight`
    - `selection_causal_lag_weight`
    - `selection_margin_penalty_weight`
- Important correction to the earlier Phase-B starting point:
  - do not assume that forcing Patel-shaped agreement weights to `0.0` is by
    itself the best next move on this branch
  - the current offline replay indicates a better first validation point is:
    - `selection_score_mode = causal_lag_composite`
    - `selection_soft_agreement_weight = 0.03`
    - `selection_causal_lag_weight = 1.0`
    - `selection_margin_penalty_weight = 0.10`
- Reason:
  - on the fixed audited trajectory, this existing-CLI selector setting moves
    the chosen epoch to `34` and matches the GT-best strict score `0.885246`
  - the current weakness is better described as:
    - `soft_agreement` overweight
    - plus insufficient `dir_margin` penalty
  - not as:
    - causal-lag signal being unusable
- Discipline:
  - keep GT audit-only
  - first validate the selector on a real rerun with training held fixed
  - only after that decide whether a new `primary + margin` style selector is
    worth adding to the code

### 2026-04-04 - Result Of The First Real Selector Validation Rerun

- Executed rerun:
  - `run_20260404_111017`
  - selector setting:
    - `selection_score_mode = causal_lag_composite`
    - `selection_soft_agreement_weight = 0.03`
    - `selection_causal_lag_weight = 1.0`
    - `selection_margin_penalty_weight = 0.10`
- Result:
  - best/export/final strict:
    - `0.885246 / 0.852459 / 0.868852`
- Interpretation:
  - the selector-only direction is still validated
  - but the exact `0.03 / 0.10` point is not yet stable enough to freeze as a
    default
  - the stronger conclusion is now:
    - lower `soft_agreement`
    - plus higher `dir_margin` penalty
    helps this branch
  - the weaker conclusion is:
    - one exact weight pair has already been solved
- Updated next step:
  - if continuing selector-only controls on this branch, prioritize a narrower
    low-soft / high-margin window before adding a new selector mode:
    - `selection_soft_agreement_weight = 0.00` to `0.01`
    - `selection_margin_penalty_weight = 0.12` to `0.15`
  - keep training backbone fixed while doing that check

### Current Active References

- After the 2026-04-03 four-dataset replay of the two retained models:
  - keep two explicit references only
  - do not reopen removed `lag_corr` / `directed_noise` / `post_detach_direction`
    branches in the main program
- Score-leading reference:
  - `patel_assisted`
  - empirical result:
    - strongest GT best score on all four datasets:
      - `fMRI`
      - `sim2`
      - `sim3`
      - `sim4`
  - limitation:
    - best/export mismatch still persists on multiple datasets
- Cleaner mechanism reference:
  - `patel_free`
  - empirical result:
    - lower ceiling than `patel_assisted`
    - but cleaner late-phase interpretation because `direction` is no longer
      taught by Patel tau
    - on `sim4`, best/export/final are aligned at `0.770492`
- Mainline planning consequence:
  - if the immediate goal is highest current score:
    - compare against `patel_assisted`
  - if the immediate goal is cleaner architectural redesign:
    - build on `patel_free`
  - do not claim that the current Patel-free branch has already replaced the
    Patel-assisted branch across the dataset suite

## 2026-04-02 - Next-stage plan after routing refactor

### Goal Reset

- The next stage should stop optimizing around a partially Patel-shaped answer.
- The main question is now:
  - can `support / direction` routing stay useful after Patel stops acting as
    the direction teacher and after checkpoint selection stops depending on
    Patel agreement?

### Fixed Anchors To Keep

- Retention anchor:
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - reason:
    - best and final stay aligned on the recent `sim4` timing window
- Ceiling anchor:
  - same routing with `detach_direction_from_main_after_epoch = 24`
  - reason:
    - this is the current strongest observed GT-best point in the same window

### Phase A - Remove Patel As Direction Teacher

- Keep the routing refactor fixed.
- Keep the current support backbone fixed for this phase so only the direction
  target changes.
- Proposed delta from the current anchor:
  - switch `directional_prior_mode` from `patel` to `lag_corr`
  - keep `lag_direction_source = raw`
  - keep `causal_lag_main` on
  - keep `fixed_support_mask_mode = maxgap_kappa`
- Purpose:
  - test whether the explicit routing design still holds when direction is
    supervised by lag evidence instead of Patel tau
- Readouts:
  - final GT audit
  - best GT audit
  - export/final gap
  - collapse mode

### Phase A Status

- First anchor check completed on `sim4`, `seed = 11`, `detach = 23`.
- Outcome:
  - `directional_prior_mode = lag_corr` under the current backbone is not yet a
    viable replacement for Patel direction supervision
  - `disable_directional_loss` performs materially better than `lag_corr`, but
    still below the Patel-teacher anchor
- Current conclusion:
  - do **not** move the mainline to `lag_corr`
  - do **not** prioritize selector ablations on top of this weakened branch
  - treat `lag_corr` as experimental-only, not mainline
- Immediate next mechanism question:
  - why is the current `lag_corr` teacher worse than having no explicit
    direction teacher at all?
  - completed diagnostics:
    - `directional_prior_scope = global_dataset`
      - no meaningful recovery
    - lower `directional_target_ratio`
      - slight best-epoch recovery, weak final effect
    - removing `directional_kappa_gate`
      - strongest single-variable improvement
  - updated read:
    - the largest current incompatibility is likely the mix of
      `lag_corr teacher + Patel-kappa supervision gate`
    - after combining:
      - `directional_kappa_gate = off`
      - `directional_target_ratio = 0.003`
      the best `lag_corr` rescue point only ties the
      `disable_directional_loss` reference instead of beating it
    - this means the current `lag_corr` implementation can be made non-harmful,
      but not beneficial
  - next likely mechanism checks:
    - no further mainline work on the current `lag_corr` formulation
    - if revisited later, it should be treated as a redesign problem rather than
      a tuning problem
    - Phase B selector work should continue only on branches that are at least
      as strong as the current preferred Patel-free reference:
      - `disable_directional_loss`

### Architecture Boundary

- Current shared conclusion before any deeper redesign:
  - the routing refactor solved a stability problem, not a theory problem
  - the current denoising objective does not appear to be a reliable source of
    directional learning
  - the practical late-phase winner is:
    - denoising updates `support`
    - another objective updates `direction`
- Therefore:
  - do not frame the current system as if the diffusion loss itself were now
    learning directed causal structure cleanly
  - future redesign discussion should begin from this explicit boundary rather
    than from the assumption that only hyperparameters remain to be fixed
- Preferred temporary interpretation:
  - treat the current mainline as:
    - diffusion-for-support
    - auxiliary temporal objective-for-direction
- Before new architectural implementation work:
  - first finish the conceptual analysis of whether the direction objective
    should become:
    - a principled lagged predictive / Granger-like target
    - or remain an explicitly pragmatic auxiliary term

### Terminology Boundary

- Use the following distinction consistently in later analysis:
  - `lag_corr`:
    - directional teacher / prior
  - `causal_lag_main`:
    - graph-conditioned lagged prediction task
- Do not describe Patel-free runs with active `causal_lag_main` as if they had
  no directional signal at all.
- The more accurate phrasing is:
  - no explicit direction teacher
  - but still a task-driven temporal direction objective

### Phase B - Remove Patel From Checkpoint Selection

- After Phase A produces a **competitive** Patel-free branch, stop using
  Patel-shaped selection as the export decision rule.
- Proposed selector starting point:
  - `selection_score_mode = causal_lag_primary`
  - `selection_soft_agreement_weight = 0.0`
  - `selection_primary_soft_tiebreak_weight = 0.0`
  - `selection_primary_skeleton_tiebreak_weight = 0.0`
  - `selection_primary_density_tiebreak_weight = 0.0`
- Purpose:
  - make checkpoint choice depend on causal-lag behavior instead of Patel
    agreement / skeleton overlap
- Important rule:
  - GT remains audit-only and must not be fed back into export selection

### Phase C - Ablate Patel Support Constraints

- Only after direction target and selector are cleaned up:
  - test whether `fixed_support_mask_mode = maxgap_kappa` is still needed
  - then separately test whether the Patel-based noise guide should remain
- Purpose:
  - separate "Patel as weak support prior" from "Patel as answer key"
- Reason for ordering:
  - removing every Patel component at once would make failures uninterpretable

### Experiment Discipline

- Do not resume detach-window tuning as the main activity unless a later phase
  explicitly shows the routing split is still sensitive to switch timing.
- For each new run, record:
  - exact config delta
  - whether Patel enters:
    - direction target
    - direction gate
    - support mask
    - selector
  - result:
    - exported / best / final
    - failure mode
    - whether the conclusion is about mechanism, selector, or support prior

## 2026-03-29 Readout Recheck And Follow-Up Probes

### `sim4` Readout Recheck On Existing `signed_gate + global_dataset`

- Re-read existing `sim4` transfer runs:
  - `GraphExp/results/run_20260329_000653`
  - `GraphExp/results/run_20260329_000744`
  - `GraphExp/results/run_20260329_000835`
- Important correction:
  - the aggregate-level `p90 = 0` / `near0 ~= 95%` on `sim4` is dominated by
    `all off-diagonal` evaluation over a sparse fixed support mask
  - it is **not** evidence that the internal direction branch re-collapsed to
    `D - D^T = 0`
- Direct offline check on the saved exported causal adjacencies shows:
  - nonzero support-pair margins still exist on all 3 seeds
  - support-only `p90` is about:
    - `0.0586`
    - `0.0919`
    - `0.0781`
  - support-only `near0(<1e-2)` is much lower than the all-offdiag aggregate
- Additional metric caveat:
  - current `final_f1` is also contaminated in sparse-mask settings because
    `evaluate_directional()` emits one direction for every unordered pair, so
    zero-zero ties become default `i->j` predictions
- Therefore for sparse fixed-support experiments:
  - trust `strict@0` / `best_strict` more than all-offdiag `p90`, `near0`,
    and `final_f1`
  - do **not** read `sim4 p90=0` as automatic proof of internal
    `symmetric_collapse`
- Even after this correction, the current `signed_gate + global_dataset`
  transfer point is still not good enough as a new mainline:
  - old baseline `sim4 strict_f1_mean ~= 0.7978`
  - `signed_gate + global_dataset sim4 strict_f1_mean ~= 0.5956`
  - so the remaining issue is no longer pure internal tie collapse; it is that
    the current time-supervised branch still hurts overall directional quality
    on `sim4`

### Probe A: `global_dataset + unsigned_raw + no signed teacher`

- Mechanism tested:
  - keep `cross_prediction` as the main time supervision
  - keep multi-lag (`1,2,3`)
  - remove signed directional teacher by setting:
    - `directional_target_ratio = 0.0`
    - `anti_collapse_mode = unsigned_raw`
  - interpretation:
    - `unsigned_raw` = only push `|D - D^T|` away from `0`
    - no longer force the sign to match the lag prior
- Artifact:
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260329_122202_stage1_unsigned_globalprior_nosign_seed11_probe_aggregate.csv`
  - run dir:
    - `GraphExp/results/run_20260329_122205`
- `sim3` seed11 result:
  - `strict@0 = 0.6667`
  - `best_strict = 0.7778`
  - `final_f1 = 0.1951`
  - `p90 = 0.0033`
  - `near0 = 93.33%`
  - `failure_mode = weak_asymmetry`
- Final-epoch internal diagnostics:
  - `cross_loss_weighted ~= 0.0733`
  - `dir_loss_weighted = 0.0`
  - `anti_collapse_weighted ~= 1e-6`
  - `dir_active_abs_margin_mean ~= 0.166`
  - `dir_active_signed_raw_frac_pos ~= 58.3%`
- Conclusion:
  - this point is **not** competitive
  - removing the signed teacher entirely is too weak
  - `cross_prediction` by itself is not yet enough to reliably determine sign

### Probe B: hard selective-signed gate via cross-subject consistency floor

- Code change:
  - `GraphExp/main_structure_learning.py` now accepts:
    - `--directional_prior_consistency_floor`
  - it only applies to cached `global_dataset` `lag_corr` priors
  - implementation:
    - compute the mean dataset lag prior
    - compute per-pair cross-subject sign consistency
    - zero out prior entries below the requested floor
- Mechanism tested:
  - keep `signed_gate`
  - keep `global_dataset`
  - use `directional_prior_consistency_floor = 0.55`
  - interpretation:
    - `global_dataset` = one fixed lag prior averaged across all subjects
    - `signed_gate` = signed anti-collapse on `tanh(0.5 * sign(prior) * (D-D^T))`
    - `consistency_floor=0.55` = only keep signed teacher on pairs whose lag
      sign agrees across at least 55% of subjects
- Why `0.55` was chosen:
  - offline support-pair analysis showed it is the first usable floor that
    materially improves sign purity without collapsing coverage to near-zero
  - on GT support pairs, the kept signed-teacher subset reaches:
    - `sim3`: `100%` sign accuracy at about `27.8%` ordered-pair coverage
    - `sim4`: `85.7%` sign accuracy at about `22.95%` ordered-pair coverage
- Artifact:
  - direct run dir:
    - `results/run_20260329_123405`
- Offline GT evaluation on the saved adjacencies:
  - best (`learned_adjacency_causal.npy`):
    - `strict_f1 = 0.6667`
    - `final_f1 = 0.1951`
    - `p90 = 0.0503`
    - `near0 = 82.86%`
    - `failure_mode = mixed_or_partial`
  - final (`final_epoch_adjacency_causal.npy`):
    - `strict_f1 = 0.6111`
    - `final_f1 = 0.1789`
    - `p90 = 0.0130`
    - `near0 = 89.52%`
    - `failure_mode = weak_asymmetry`
- Best/final internal diagnostics:
  - both best epoch 17 and final epoch 100 show:
    - `dir_active_abs_margin_mean = 12.0`
    - `dir_active_signed_raw_frac_pos = 1.0`
    - `dir_active_signed_gate_frac_pos = 1.0`
  - but the active signed-teacher subset becomes too small:
    - `dir_active_pair_frac = 0.044444`
  - `cross_loss_weighted` stays around `0.073`
  - `dir_loss_weighted` and `anti_collapse_weighted` quickly go to `0`
- Conclusion:
  - this hard floor **over-prunes** the signed teacher
  - the kept subset is very clean, but too sparse to improve whole-graph
    direction quality
  - this is not a good next mainline point

### Updated Branching Conclusion

- Already rejected:
  - continue Patel-strength tuning
  - `global_dataset + unsigned_raw + no signed teacher`
  - hard `directional_prior_consistency_floor = 0.55`
- Still worth trying next:
  - keep time-supervised mainline
  - keep encoder frozen and keep `cross_prediction` source under `no_grad`
  - move from hard consistency gating to **soft consistency weighting**
    on signed supervision
  - goal:
    - keep signed teacher coverage
    - downweight cross-subject unstable lag-sign pairs
    - avoid both extremes already observed:
    - no sign teacher -> too weak
    - hard consistency floor -> too sparse

### Soft Consistency Weighting Probes (`sim3`, seed11 only)

- New code path added in:
  - `GraphExp/main_structure_learning.py`
- New CLI:
  - `--directional_prior_consistency_power`
- Mechanism:
  - keep the same `global_dataset` lag-sign teacher
  - do **not** hard-prune active pairs
  - instead multiply the directional supervision weights by
    `consistency^power`
  - interpretation:
    - more cross-subject sign-stable pairs get larger signed-teacher weight
    - unstable pairs are downweighted, not removed

#### Probe C: soft weighting with `consistency_power = 2.0`

- Run dir:
  - `results/run_20260329_155609`
- Best-epoch readout (`learned_adjacency_causal.npy`):
  - `strict_f1 = 0.8333`
  - `final_f1 = 0.2439`
  - `p90 = 0.1057`
  - `near0 = 82.86%`
  - `failure_mode = mixed_or_partial`
- Final-epoch readout (`final_epoch_adjacency_causal.npy`):
  - `strict_f1 = 0.7222`
  - `final_f1 = 0.2114`
  - `p90 = 0.0212`
  - `near0 = 85.71%`
  - `failure_mode = weak_asymmetry`
- Internal diagnostics:
  - best epoch 13:
    - `cross_loss_weighted ~= 0.0734`
    - `dir_loss_weighted ~= 3.6e-4`
    - `dir_active_pair_frac ~= 0.1067`
    - `dir_active_reliability_mean ~= 0.3283`
    - `dir_active_abs_margin_mean ~= 6.74`
    - `dir_active_signed_raw_frac_pos = 1.0`
  - final epoch 100:
    - `dir_active_abs_margin_mean = 12.0`
    - `dir_active_signed_raw_frac_pos = 1.0`
- Comparison against current best time-supervision point
  (`GraphExp/results/run_20260329_000210`):
  - `best_strict` ties at `0.8333`
  - best-epoch `near0` is slightly better (`82.86%` vs `83.81%`)
  - but best-epoch `p90` is lower (`0.1057` vs `0.1274`)
  - final metrics are clearly worse than the current best point
- Conclusion:
  - `power=2` does **not** justify expanding to 3 seeds

#### Probe D: soft weighting with `consistency_power = 4.0`

- Run dir:
  - `results/run_20260329_160206`
- Best-epoch readout (`learned_adjacency_causal.npy`):
  - `strict_f1 = 0.8333`
  - `final_f1 = 0.2439`
  - `p90 = 0.1041`
  - `near0 = 82.86%`
  - `failure_mode = mixed_or_partial`
- Final-epoch readout (`final_epoch_adjacency_causal.npy`):
  - `strict_f1 = 0.7778`
  - `final_f1 = 0.2276`
  - `p90 = 0.0195`
  - `near0 = 85.71%`
  - `failure_mode = weak_asymmetry`
- Internal diagnostics:
  - best epoch 13:
    - `cross_loss_weighted ~= 0.0734`
    - `dir_loss_weighted ~= 3.2e-4`
    - `dir_active_pair_frac ~= 0.1067`
    - `dir_active_reliability_mean ~= 0.1098`
    - `dir_active_abs_margin_mean ~= 6.30`
    - `dir_active_signed_raw_frac_pos = 1.0`
  - final epoch 100:
    - `dir_active_abs_margin_mean = 12.0`
    - `dir_active_signed_raw_frac_pos = 1.0`
- Comparison against current best time-supervision point:
  - `best_strict` still only ties at `0.8333`
  - final `strict_f1` also ties at `0.7778`
  - but final `p90` is much lower (`0.0195` vs `0.0605`)
  - final `near0` is worse (`85.71%` vs `82.86%`)
- Conclusion:
  - `power=4` also does **not** beat the current best time-supervision point
  - stronger soft weighting does not recover the desired improvement

### Updated Decision After Soft Weighting

- The planned order was:
  1. add soft consistency weighting
  2. run `sim3` seed11
  3. only expand to `sim3` 3-seed if the single-seed point improves
- Result:
  - both `power=2` and `power=4` fail that gate
  - neither improves the key final readouts (`near0`, `p90`, `strict@0`)
    over the current best `signed_gate + global_dataset` point
- Therefore:
  - **do not** expand soft consistency weighting to `sim3` 3-seed
  - **do not** move this branch to `sim4`
- Mechanistic takeaway:
  - the signed lag teacher can already fully separate the active directional
    subset (`dir_active_signed_raw_frac_pos = 1.0`)
  - reweighting that teacher by cross-subject sign stability alone is not
    enough to improve whole-graph exported direction quality
  - the next mechanism should not be “same teacher, better weighting” only;
    it likely needs a stronger change in how time supervision couples to the
    exported graph

## 2026-03-28 Signed-Gate Anti-Collapse With Online-Subject Lag Prior

### Seed11 Probe

- Objective:
  - keep the Stage-1 time-supervised branch unchanged except for one mechanism
    replacement:
    - replace unsigned raw-logit anti-collapse with a sign-aware gate-space
      floor aligned to the lag prior sign
- Treatment config:
  - `cross_prediction = on`
  - `directional_prior_mode = lag_corr`
  - `lag_direction_source = raw`
  - `directional_prior_lags = 1,2,3`
  - `cross_pred_lags = 1,2,3`
  - `cross_pred_fixed_weight = 0.1`
  - `directional_target_ratio = 0.01`
  - `anti_collapse_lambda = 0.1`
  - `anti_collapse_mode = signed_gate`
  - `anti_collapse_margin = 0.2`
  - `directional_prior_scope = online_subject`
- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_234350_stage1_signed_gate_seed11_probe.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_234350_stage1_signed_gate_seed11_probe_aggregate.csv`
- Seed `11` result:
  - `p90 = 0.01412`
  - `near0(<1e-2) = 85.71%`
  - `strict@0 = 0.6111`
  - `best_strict = 0.8333`
  - `final_f1 = 0.1789`
  - `gt_signed_margin_median = 0.01294`
  - `failure_mode = weak_asymmetry`

### Mechanism Read

- This run is important because the new anti-collapse term is **not** inert:
  - `cross_loss_weighted = 0.07338`
  - `dir_loss_weighted = 0.00001`
  - `anti_collapse_weighted = 0.01627`
- Final-epoch active-pair diagnostics show the wrong-sign regime directly:
  - `dir_active_abs_margin_mean = 0.32468`
  - `dir_active_abs_margin_median = 0.39167`
  - `dir_active_signed_raw_median = -0.26365`
  - `dir_active_signed_gate_median = -0.13107`
  - `dir_active_signed_gate_frac_pos = 30.77%`
- Interpretation:
  - the problem is **not** that signed gate-space anti-collapse cannot produce
    gradients
  - the problem is that with `online_subject` lag supervision, the active-pair
    sign target is still conflicting enough that the branch is pushed into a
    wrong-sign asymmetry regime

### Subject-Level Lag Sign Consistency Read On `sim3`

- To check whether the sign target itself is unstable, a quick raw multi-lag
  sign-consistency read was computed across all `50` subjects on `sim3`.
- Across all ordered pairs:
  - consistency mean = `0.6293`
  - consistency median = `0.58`
  - consistency p10 = `0.52`
  - fraction with consistency `< 0.6` = `50.48%`
  - fraction with consistency `< 0.7` = `73.33%`
- On the strongest quartile of ordered pairs ranked by `|mean lag score|`:
  - consistency mean = `0.7940`
  - consistency median = `0.80`
  - consistency p10 = `0.70`
  - fraction with consistency `< 0.7` = `9.43%`
- Interpretation:
  - a per-subject online lag prior injects substantial sign noise on many pairs
  - averaging the lag prior over the full dataset is a justified next minimal
    mechanism test

## 2026-03-28 Fixed Global-Dataset Lag Prior Branch

### Mechanism Change

- Added `directional_prior_scope` with:
  - `online_subject`:
    - old behavior; recompute the lag prior from the current subject only
  - `global_dataset`:
    - new behavior; average the multi-lag lag-corr prior over all subjects once
      and reuse that fixed matrix throughout training
- This change preserves the current Stage-1 constraints:
  - encoder remains frozen
  - `cross_prediction` still uses `prepare_clean_target(x)` under `torch.no_grad()`
  - no joint encoder optimization is introduced

### `sim3` Seed11 Probe

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_235900_stage1_signed_gate_globalprior_seed11_probe.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_235900_stage1_signed_gate_globalprior_seed11_probe_aggregate.csv`
- Seed `11` result:
  - `p90 = 0.06048`
  - `near0(<1e-2) = 82.86%`
  - `strict@0 = 0.7778`
  - `best_strict = 0.8333`
  - `final_f1 = 0.2276`
  - `gt_signed_margin_median = 0.05194`
  - `failure_mode = mixed_or_partial`
- Relative to the `online_subject` signed-gate probe on the same seed:
  - `p90: 0.01412 -> 0.06048`
  - `near0: 85.71% -> 82.86%`
  - `strict@0: 0.6111 -> 0.7778`
  - `final_f1: 0.1789 -> 0.2276`
  - `gt_signed_margin_median: 0.01294 -> 0.05194`
  - `failure_mode: weak_asymmetry -> mixed_or_partial`

### `sim3` Seed11 Internal Read

- The internal direction branch no longer shows any sign conflict:
  - `dir_active_abs_margin_mean = 12.0`
  - `dir_active_signed_raw_mean = 12.0`
  - `dir_active_signed_raw_frac_pos = 1.0`
  - `dir_active_signed_gate_mean = 0.999988`
  - `dir_active_signed_gate_frac_pos = 1.0`
  - `cross_loss_weighted = 0.07336`
  - `dir_loss_weighted = 0`
  - `anti_collapse_weighted = 0`
- Interpretation:
  - once the lag sign target is fixed globally, the direction branch saturates
    in the correct signed direction instead of fighting the prior

### `sim3` Three-Seed Confirm

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000207_stage1_signed_gate_globalprior_confirm3.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000207_stage1_signed_gate_globalprior_confirm3_aggregate.csv`
- Aggregate:
  - `final_f1 = 0.22764`
  - `best_strict = 0.77778`
  - `strict@0 = 0.77778`
  - `p90 = 0.05610`
  - `near0(<1e-2) = 83.81%`
  - `failure_mode = mixed_or_partial:3/3`
- Comparison:
  - vs failed `online_subject` signed-gate branch:
    - `final_f1: 0.18428 -> 0.22764`
    - `best_strict: 0.55556 -> 0.77778`
    - `strict@0: 0.62963 -> 0.77778`
    - `p90: 0.00935 -> 0.05610`
    - `near0: 90.48% -> 83.81%`
    - `failure_mode: weak_asymmetry:3/3 -> mixed_or_partial:3/3`
  - vs old baseline:
    - old baseline still remains slightly stronger on `sim3`
    - old baseline aggregate:
      - `final_f1 = 0.23848`
      - `best_strict = 0.85185`
      - `strict@0 = 0.81481`
      - `p90 = 0.07016`
      - `near0(<1e-2) = 83.17%`
- Final-epoch internal diagnostics for all three seeds:
  - `dir_active_abs_margin_mean ≈ 12.0`
  - `dir_active_signed_raw_frac_pos = 1.0`
  - `dir_active_signed_gate_frac_pos = 1.0`
- Conclusion:
  - `directional_prior_scope = global_dataset` is the first time-supervised
    anti-collapse variant that consistently escapes the weak-asymmetry basin on
    `sim3`
  - it supersedes the `online_subject` signed-gate branch
  - but it does **not** yet beat the old baseline on `sim3`

## 2026-03-29 `sim4` Transfer Pilot For The Global-Dataset Signed-Gate Branch

### Setup

- Dataset / GT:
  - `sim4.csv`
  - `h4.txt`
- Branch:
  - same Stage-1 treatment as the successful `sim3` confirm
  - only dataset / GT changed
- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000650_stage1_signed_gate_globalprior_sim4transfer3.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000650_stage1_signed_gate_globalprior_sim4transfer3_aggregate.csv`

### Aggregate Result

- `final_f1 = 0.05651`
- `best_strict = 0.52459`
- `strict@0 = 0.59563`
- `p90 = 0.0`
- `near0(<1e-2) = 95.51%`
- `failure_mode = symmetric_collapse:3/3`

### Transfer Comparison

- Existing old baseline `sim4` transfer aggregate:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_221657_residual_sim4_transfer_baseline3_aggregate.csv`
- Old baseline on `sim4`:
  - `final_f1 = 0.07569`
  - `best_strict = 0.81967`
  - `strict@0 = 0.79781`
  - `p90 = 0.0`
  - `near0(<1e-2) = 96.24%`
  - `failure_mode = symmetric_collapse:3/3`
- New branch vs old baseline on `sim4`:
  - both remain `symmetric_collapse`
  - new branch reduces `near0` slightly
  - but it does not recover transfer and remains below the old baseline on
    strict / F1 metrics

### Critical Internal Read

- Despite final exported collapse, all three `sim4` runs show the same
  saturated internal direction-branch state at epoch `30`:
  - `dir_active_abs_margin_mean = 12.0`
  - `dir_active_signed_raw_mean = 12.0`
  - `dir_active_signed_raw_frac_pos = 1.0`
  - `dir_active_signed_gate_mean = 0.999988`
  - `dir_active_signed_gate_frac_pos = 1.0`
  - `cross_loss_weighted ≈ 0.07328`
  - `dir_loss_weighted = 0`
  - `anti_collapse_weighted = 0`
- This is the most important new mechanism read:
  - on `sim4`, the **internal direction branch is no longer collapsing**
  - but the final exported adjacency still collapses into near-zero margins
- Therefore the remaining transfer bottleneck is not the old
  `D - D^T ≈ 0` direction-branch tie basin by itself
- The remaining failure is more likely downstream in:
  - support weighting
  - support-direction coupling
  - or export-time effective margin preservation

### Updated Mainline Conclusion

- Residual Patel remains closed.
- `signed_gate + online_subject` is also closed as a mainline branch because it
  reveals the subject-level sign-conflict problem directly.
- `signed_gate + global_dataset` is the current best time-supervised
  anti-collapse branch:
  - it materially fixes `sim3` weak asymmetry
  - it proves that fixed time-prior sign supervision can drive the internal
    direction branch into a fully non-collapsed regime without unfreezing the
    encoder
- However:
  - it still does not beat the old baseline on `sim3`
  - and it still fails to transfer on `sim4`
- Updated next step:
  - keep:
    - `directional_prior_scope = global_dataset`
    - `anti_collapse_mode = signed_gate`
  - do **not** return to Patel scale sweeps
  - shift the next mechanism target from pure direction anti-collapse to
    support/export preservation on the same kappa-gated directional pairs
  - a minimal next probe should add a support-retention / support-floor
    mechanism and re-test on `sim3` seed `11` before any broader sweep

## 2026-03-28 Stage-1 Signed-Gate Follow-Up

### Why This Branch Was Opened

- The previous `anti_collapse_margin=0.02` branch failed mainly because the
  floor was too small in raw-logit space and the anti-collapse term stayed
  nearly inert.
- The `margin=1.0` raw-logit probe then showed:
  - the scale mismatch was real
  - but simply increasing unsigned magnitude can also amplify wrong directions
- Therefore the next minimal mechanism change was:
  - keep encoder frozen
  - keep `cross_prediction` and multi-lag setup unchanged
  - change anti-collapse from unsigned raw contrast to a **sign-aware
    gate-space floor** aligned to the time prior

### Code Additions

- `GraphExp/main_structure_learning.py`
  - added `anti_collapse_mode` with:
    - `unsigned_raw`
    - `signed_raw`
    - `signed_gate`
  - added signed diagnostics on active directional pairs:
    - `dir_active_signed_raw_*`
    - `dir_active_signed_gate_*`
  - later added `directional_prior_scope` with:
    - `online_subject`
    - `global_dataset`
- `GraphExp/run_cross_pred_v1_final_only_compare.py`
  - runner now forwards and records both:
    - `anti_collapse_mode`
    - `directional_prior_scope`
  - aggregate / pairing keys were updated so new branches do not merge
    incorrectly

### `signed_gate` With `online_subject` Prior: `sim3` Seed-11 Probe

- Probe artifact:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_234350_stage1_signed_gate_seed11_probe_aggregate.csv`
- Config:
  - same failed Stage-1 treatment backbone as before
  - only change:
    - `anti_collapse_mode = signed_gate`
    - `anti_collapse_margin = 0.2`
- Result:
  - `p90 = 0.0141`
  - `near0(<1e-2) = 85.71%`
  - `strict@0 = 0.6111`
  - `best_strict = 0.8333`
  - `final_f1 = 0.1789`
  - `failure_mode = weak_asymmetry`
- Internal epoch-30 diagnostics:
  - run dir:
    - `GraphExp/results/run_20260328_234353/quality_history.csv`
  - `dir_active_abs_margin_mean = 0.3247`
  - `dir_active_signed_raw_mean = -0.1287`
  - `dir_active_signed_raw_median = -0.2637`
  - `dir_active_signed_gate_mean = -0.0637`
  - `dir_active_signed_gate_frac_pos = 30.77%`
  - `cross_loss_weighted = 0.07338`
  - `dir_loss_weighted = 0.00001`
  - `anti_collapse_weighted = 0.01627`
- Mechanism read:
  - unlike the earlier inert branch, anti-collapse is now **active**
  - but under `online_subject` prior it pushes many active pairs toward the
    **wrong sign** relative to the current lag prior
  - therefore the next question became:
    - is the remaining failure caused by noisy / conflicting subject-level sign
      targets rather than weak anti-collapse strength?

### Subject-Level Lag-Prior Consistency Read

- Quick analysis on `sim3.csv` with multi-lag raw lag-corr (`lags=1,2,3`) shows
  substantial subject-level sign variation.
- Across all ordered off-diagonal pairs:
  - sign-consistency mean = `0.6293`
  - median = `0.58`
  - `consistency < 0.6` on `50.48%` of pairs
  - `consistency < 0.7` on `73.33%` of pairs
- On the strongest quartile of pairs by absolute mean lag score:
  - sign-consistency mean = `0.7940`
  - median = `0.80`
  - `consistency < 0.7` only `9.43%`
- Practical read:
  - a subject-level online lag prior is noisy enough to create conflicting sign
    supervision on many pairs
  - this makes a fixed **global-dataset** lag prior a justified next test

### `signed_gate` With `global_dataset` Prior: `sim3` Seed-11 Probe

- Probe artifact:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_235900_stage1_signed_gate_globalprior_seed11_probe_aggregate.csv`
- Only change from the previous probe:
  - `directional_prior_scope = global_dataset`
- Result:
  - `p90 = 0.0605`
  - `near0(<1e-2) = 82.86%`
  - `strict@0 = 0.7778`
  - `best_strict = 0.8333`
  - `final_f1 = 0.2276`
  - `gt_signed_margin_median = 0.0519`
  - `failure_mode = mixed_or_partial`
- Internal epoch-30 diagnostics:
  - run dir:
    - `GraphExp/results/run_20260328_235903/quality_history.csv`
  - `dir_active_abs_margin_mean = 12.0`
  - `dir_active_signed_raw_mean = 12.0`
  - `dir_active_signed_raw_frac_pos = 100%`
  - `dir_active_signed_gate_mean = 0.999988`
  - `dir_active_signed_gate_frac_pos = 100%`
  - `cross_loss_weighted = 0.07336`
  - `dir_loss_weighted = 0`
  - `anti_collapse_weighted = 0`
- Mechanism read:
  - switching only the prior scope from `online_subject` to `global_dataset`
    flips the active signed margins from negative to fully positive
  - this is strong evidence that the previous failure was dominated by
    **subject-level sign conflict**, not by an inherently bad signed-gate loss

### `sim3` Three-Seed Confirmation For `signed_gate + global_dataset`

- Aggregate artifact:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000207_stage1_signed_gate_globalprior_confirm3_aggregate.csv`
- Aggregate result:
  - `final_f1 = 0.2276`
  - `best_strict = 0.7778`
  - `strict@0 = 0.7778`
  - `p90 = 0.0561`
  - `near0(<1e-2) = 83.81%`
  - `failure_mode = mixed_or_partial:3/3`
- Comparison against earlier Stage-1 variants:
  - old baseline:
    - `final_f1 = 0.2385`
    - `best_strict = 0.8519`
    - `strict@0 = 0.8148`
    - `p90 = 0.0702`
    - `near0 = 83.17%`
    - `failure = mixed_or_partial:3/3`
  - failed online-subject signed/anti-collapse branch:
    - `final_f1 = 0.1843`
    - `best_strict = 0.5556`
    - `strict@0 = 0.6296`
    - `p90 = 0.00935`
    - `near0 = 90.48%`
    - `failure = weak_asymmetry:3/3`
- Updated `sim3` conclusion:
  - `global_dataset` prior scope is a **real mechanism fix** for the previous
    online-sign-conflict failure
  - it restores the branch from `weak_asymmetry` back to
    `mixed_or_partial` on all three seeds
  - however it still does **not** surpass the old baseline on `sim3`
  - so this branch is worth keeping as the current best time-supervised
    anti-collapse variant, but not yet as a new default winner

### `sim4` Transfer Pilot For `signed_gate + global_dataset`

- Aggregate artifact:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000650_stage1_signed_gate_globalprior_sim4transfer3_aggregate.csv`
- Aggregate result:
  - `final_f1 = 0.0565`
  - `best_strict = 0.5246`
  - `strict@0 = 0.5956`
  - `p90 = 0`
  - `near0(<1e-2) = 95.51%`
  - `failure_mode = symmetric_collapse:3/3`
- Internal epoch-30 diagnostics for all `3` seeds:
  - run dirs:
    - `GraphExp/results/run_20260329_000653/quality_history.csv`
    - `GraphExp/results/run_20260329_000744/quality_history.csv`
    - `GraphExp/results/run_20260329_000835/quality_history.csv`
  - all three show:
    - `dir_active_abs_margin_mean = 12.0`
    - `dir_active_signed_raw_mean = 12.0`
    - `dir_active_signed_raw_frac_pos = 100%`
    - `dir_active_signed_gate_mean = 0.999988`
    - `dir_active_signed_gate_frac_pos = 100%`
    - `anti_collapse_weighted = 0`
- Important mechanism conclusion:
  - on `sim4`, the direction branch itself is **not** failing to form a strong,
    correctly signed internal asymmetry under the fixed global prior
  - but the exported final adjacency still falls back to
    `symmetric_collapse`
  - therefore the remaining transfer bottleneck is no longer just
    "direction branch tied at zero"
  - it is more specifically:
    - the current framework can learn an internal directional solution
    - yet that solution does not survive through the final
      support-direction export on `sim4`

### Updated Mainline Decision

- Do **not** spend more budget re-sweeping:
  - Patel fusion strengths
  - `online_subject` signed-gate variants
  - nearby anti-collapse margins around this branch
- Keep the following updated picture:
  - `online_subject` lag prior causes conflicting sign supervision and is now
    superseded by `global_dataset`
  - `global_dataset + signed_gate` materially improves `sim3`
  - but it does **not** solve `sim4` transfer
- Therefore the next mechanism branch should move from pure direction-floor work
  toward **support/export coupling**, for example:
  - keeping support on direction-supervised pairs from collapsing away
  - or making final exported asymmetry depend less on a support branch that can
    null out an otherwise-correct direction solution

## 2026-03-28 Signed-Gate Anti-Collapse Probe With Online-Subject Lag Prior

### Objective

- Test the next minimal anti-collapse variant without changing the failed
  Stage-1 treatment backbone:
  - keep encoder frozen
  - keep `cross_prediction` active with fixed weight
  - keep multi-lag raw time prior
  - change only the anti-collapse target from unsigned raw-logit contrast to a
    **signed gate-space floor**
- Hypothesis:
  - if the earlier failure was mainly because anti-collapse only enforced
    unsigned magnitude, then a sign-aware gate-space floor should increase
    exported asymmetry **and** improve direction correctness.

### Probe Setup

- Dataset / GT:
  - `sim3.csv`
  - `h3.txt`
- Seed:
  - `11`
- Structural backbone:
  - `structure_init_mode = patel_kappa`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `structure_init_scale = 0.5`
  - `lambda_l1 = 0.02`
  - no persistent `kappa` / `tau` bias
- Time-supervised branch:
  - `cross_prediction = on`
  - `directional_prior_mode = lag_corr`
  - `lag_direction_source = raw`
  - `directional_prior_lags = 1,2,3`
  - `cross_pred_lags = 1,2,3`
  - `cross_pred_fixed_weight = 0.1`
  - `directional_target_ratio = 0.01`
  - `anti_collapse_lambda = 0.1`
  - `anti_collapse_margin = 0.2`
  - `anti_collapse_mode = signed_gate`

### Artifacts

- Summary:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_234350_stage1_signed_gate_seed11_probe.csv`
- Aggregate:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_234350_stage1_signed_gate_seed11_probe_aggregate.csv`
- Run dir:
  - `GraphExp/results/run_20260328_234353`

### Result

- Seed `11` result:
  - `p90 = 1.4122e-02`
  - `near0(<1e-2) = 85.71%`
  - `strict@0 = 0.6111`
  - `best_strict = 0.8333`
  - `final_f1 = 0.1789`
  - `gt_signed_margin_median = 1.2944e-02`
  - `gt_signed_margin_frac_pos = 61.11%`
  - `failure_mode = weak_asymmetry`

### End-Epoch Internal Read

- From `GraphExp/results/run_20260328_234353/quality_history.csv` at epoch `30`:
  - `dir_active_abs_margin_mean = 0.324675`
  - `dir_active_abs_margin_median = 0.391671`
  - `dir_active_abs_margin_p90 = 0.410629`
  - `dir_active_signed_raw_mean = -0.128719`
  - `dir_active_signed_raw_median = -0.263653`
  - `dir_active_signed_raw_frac_pos = 30.77%`
  - `dir_active_signed_gate_mean = -0.063654`
  - `dir_active_signed_gate_median = -0.131068`
  - `dir_active_signed_gate_frac_pos = 30.77%`
  - `cross_loss_weighted = 0.073380`
  - `dir_loss_weighted = 1.0e-05`
  - `anti_collapse_weighted = 1.6273e-02`

### Interpretation

- This probe is important because it shows a **different failure mode** from the
  earlier inert anti-collapse term:
  - anti-collapse is no longer numerically dead
  - it is now strongly active
- But the active-pair signed diagnostics are mostly **negative**:
  - the branch is being pushed away from zero
  - yet it still lands on the **opposite sign** for most active pairs
- Therefore:
  - the bottleneck is not just "make anti-collapse stronger"
  - the more specific issue is likely that the **subject-level online lag prior**
    provides conflicting sign supervision over training.

### Quick Sign-Consistency Read For Subject-Level Lag Prior

- A quick offline check on `sim3.csv` using the same multi-lag raw prior
  (`lags = 1,2,3`) shows:
  - across **all ordered pairs** over `50` subjects:
    - sign-consistency mean = `0.6293`
    - median = `0.58`
    - p10 = `0.52`
    - fraction below `0.6` = `50.48%`
    - fraction below `0.7` = `73.33%`
  - on the strongest top-quartile pairs by mean absolute lag score:
    - sign-consistency mean = `0.7940`
    - median = `0.80`
    - p10 = `0.70`
- Practical read:
  - per-subject online lag sign is noisy enough to create real gradient conflict
  - the next minimal move should therefore be:
    - keep the same signed-gate loss
    - but replace `online_subject` prior scope with one fixed
      `global_dataset` lag prior.

## 2026-03-28 Fixed Global-Dataset Lag Prior For The Signed-Gate Branch

### Code Change

- Added `directional_prior_scope` with:
  - `online_subject` = previous behavior
  - `global_dataset` = compute the multi-lag lag-corr prior once by averaging
    subject-level priors over the full dataset, then reuse that fixed matrix
    through the whole run
- The default remains `online_subject`, so old runs stay reproducible.

### Seed-11 Probe

- Same config as the previous signed-gate probe except:
  - `directional_prior_scope = global_dataset`
- Artifacts:
  - Summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_235900_stage1_signed_gate_globalprior_seed11_probe.csv`
  - Aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_235900_stage1_signed_gate_globalprior_seed11_probe_aggregate.csv`
  - Run dir:
    - `GraphExp/results/run_20260328_235903`

### Seed-11 Result

- Seed `11`:
  - `p90 = 6.0483e-02`
  - `near0(<1e-2) = 82.86%`
  - `strict@0 = 0.7778`
  - `best_strict = 0.8333`
  - `final_f1 = 0.2276`
  - `gt_signed_margin_median = 5.1943e-02`
  - `gt_signed_margin_frac_pos = 77.78%`
  - `failure_mode = mixed_or_partial`

### Direct Comparison To The Online-Subject Probe

- `p90`:
  - `0.0141 -> 0.0605`
- `near0(<1e-2)`:
  - `85.71% -> 82.86%`
- `strict@0`:
  - `0.6111 -> 0.7778`
- `final_f1`:
  - `0.1789 -> 0.2276`
- `gt_signed_margin_median`:
  - `0.0129 -> 0.0519`
- `failure_mode`:
  - `weak_asymmetry -> mixed_or_partial`

### End-Epoch Internal Read

- From `GraphExp/results/run_20260328_235903/quality_history.csv` at epoch `30`:
  - `dir_active_abs_margin_mean = 12.000000`
  - `dir_active_signed_raw_mean = 12.000000`
  - `dir_active_signed_raw_frac_pos = 100%`
  - `dir_active_signed_gate_mean = 0.999988`
  - `dir_active_signed_gate_frac_pos = 100%`
  - `cross_loss_weighted = 0.073357`
  - `dir_loss_weighted = 0`
  - `anti_collapse_weighted = 0`

### Interpretation

- This is a strong mechanism confirmation:
  - once the lag-sign target is fixed globally, the direction branch no longer
    fights contradictory subject-level signs
  - the active directional pairs saturate to the correct signed side
- Therefore the main blocker on `sim3` was not simply "time supervision is too
  weak":
  - it was much more specifically:
    - **online sign conflict in the lag prior**

## 2026-03-29 Three-Seed Confirmation For The Global-Prior Signed-Gate Branch

### Artifacts

- Summary:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000207_stage1_signed_gate_globalprior_confirm3.csv`
- Aggregate:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000207_stage1_signed_gate_globalprior_confirm3_aggregate.csv`
- Run dirs:
  - `GraphExp/results/run_20260329_000210`
  - `GraphExp/results/run_20260329_000317`
  - `GraphExp/results/run_20260329_000425`

### Aggregate Result

- `final_f1 = 0.2276 ± 0.0000`
- `strict@0 = 0.7778 ± 0.0000`
- `best_strict = 0.7778 ± 0.0454`
- `p90 = 5.6099e-02 ± 4.0292e-03`
- `near0(<1e-2) = 83.81%`
- `failure_mode = mixed_or_partial:3/3`

### Comparison Against Existing References

- Old baseline (`GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260328_201524_stage1_old_baseline3_aggregate.csv`):
  - `final_f1 = 0.2385`
  - `strict@0 = 0.8148`
  - `best_strict = 0.8519`
  - `p90 = 7.0161e-02`
  - `near0(<1e-2) = 83.17%`
  - `failure_mode = mixed_or_partial:3/3`
- Failed online-subject signed branch
  (`GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260328_202432_stage1_time_anti_collapse3_aggregate.csv`):
  - `final_f1 = 0.1843`
  - `strict@0 = 0.6296`
  - `best_strict = 0.5556`
  - `p90 = 9.3542e-03`
  - `near0(<1e-2) = 90.48%`
  - `failure_mode = weak_asymmetry:3/3`

### Internal Read Across All Three Seeds

- All three run dirs end with essentially identical active-pair direction stats:
  - `dir_active_abs_margin_mean ≈ 12.0`
  - `dir_active_signed_raw_frac_pos = 100%`
  - `dir_active_signed_gate_frac_pos = 100%`
  - `anti_collapse_weighted = 0`
- Practical read:
  - the global-prior fix is stable across seeds
  - this branch is now the strongest time-supervised anti-collapse variant tried
    so far on `sim3`
  - however it still does **not** clearly surpass the old baseline on `sim3`

### Decision

- Keep this branch.
- Do **not** go back to:
  - Patel strength tuning
  - `online_subject` lag-prior scope
  - more anti-collapse margin sweeps for this branch
- The key win is the **prior-scope fix**, not another margin retune.

## 2026-03-29 `sim4` Transfer Pilot For The Global-Prior Signed-Gate Branch

### Artifacts

- Summary:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000650_stage1_signed_gate_globalprior_sim4transfer3.csv`
- Aggregate:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260329_000650_stage1_signed_gate_globalprior_sim4transfer3_aggregate.csv`
- Run dirs:
  - `GraphExp/results/run_20260329_000653`
  - `GraphExp/results/run_20260329_000744`
  - `GraphExp/results/run_20260329_000835`

### Aggregate Result

- `final_f1 = 0.0565 ± 0.0029`
- `strict@0 = 0.5956 ± 0.0309`
- `best_strict = 0.5246 ± 0.0134`
- `p90 = 0`
- `near0(<1e-2) = 95.51%`
- `failure_mode = symmetric_collapse:3/3`

### Comparison To Existing `sim4` Baseline Transfer Pilot

- Existing old baseline aggregate
  (`GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_221657_residual_sim4_transfer_baseline3_aggregate.csv`):
  - `final_f1 = 0.0757`
  - `strict@0 = 0.7978`
  - `best_strict = 0.8197`
  - `p90 = 0`
  - `near0(<1e-2) = 96.24%`
  - `failure_mode = symmetric_collapse:3/3`
- New global-prior signed-gate branch:
  - `final_f1 = 0.0565`
  - `strict@0 = 0.5956`
  - `best_strict = 0.5246`
  - `p90 = 0`
  - `near0(<1e-2) = 95.51%`
  - `failure_mode = symmetric_collapse:3/3`

### Critical Internal Observation

- Despite the transfer failure, all three `sim4` run dirs still show:
  - `dir_active_abs_margin_mean = 12.0`
  - `dir_active_signed_raw_frac_pos = 100%`
  - `dir_active_signed_gate_frac_pos = 100%`
  - `anti_collapse_weighted = 0`
- Therefore the `sim4` failure is **not**:
  - "the direction branch stayed symmetric internally"
- It is more specifically:
  - the internal direction branch is fully separated and aligned
  - but the exported adjacency still collapses at evaluation scale

### Updated Mechanism Read

- The `global_dataset` lag-prior fix successfully repairs the Stage-1
  direction-branch supervision problem on `sim3`.
- But it does **not** solve transfer to `sim4`.
- The remaining bottleneck is now likely downstream of internal direction-logit
  separation, for example:
  - support-side suppression
  - support-direction coupling at export time
  - insufficient retention of support mass on direction-supervised pairs

### Updated Mainline

- Current best time-supervised branch:
  - `lag_corr(raw)` with `directional_prior_scope = global_dataset`
  - `cross_pred_fixed_weight = 0.1`
  - `anti_collapse_mode = signed_gate`
  - `anti_collapse_margin = 0.2`
- But after the `sim4` pilot, the next mechanism should no longer be framed as
  only "break internal direction symmetric collapse".
- The next more relevant target is:
  - why a direction branch that is already internally separated and correctly
    signed still fails to produce non-collapsed exported margins on `sim4`
- Therefore the next minimal branch should move toward support/export coupling,
  not back toward Patel tuning or another sign-floor sweep.

## 2026-03-28 First Formal `sim3` 3-Seed Pilot For The Time-Supervised Anti-Collapse Branch

### Objective

- Run the first real Stage-1 comparison on:
  - `sim3.csv`
  - `h3.txt`
  - seeds `11,22,33`
- Keep the old baseline exactly as recovered from the prior formal aggregate.
- Compare it against one single-point new branch:
  - same structural backbone
  - no persistent `tau` bias
  - explicit raw-lag time supervision
  - multi-lag cross-prediction
  - gated anti-collapse margin floor

### Commands Used

- Old baseline:
  - `python GraphExp\run_cross_pred_v1_final_only_compare.py --csv_path D:\mockup\DDM-main\fMRI_dataset\sim3.csv --gt_path D:\mockup\DDM-main\fMRI_dataset\h3.txt --pretrain_checkpoint D:\mockup\DDM-main\GraphExp\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 18 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.0 --direction_logit_bias_scales 0.0 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag stage1_old_baseline3`
- New treatment:
  - `python GraphExp\run_cross_pred_v1_final_only_compare.py --csv_path D:\mockup\DDM-main\fMRI_dataset\sim3.csv --gt_path D:\mockup\DDM-main\fMRI_dataset\h3.txt --pretrain_checkpoint D:\mockup\DDM-main\GraphExp\results\run_20260310_185625\pretrained_encoder.pt --device cuda --epochs 30 --pretrain_epochs 50 --log_interval 10 --top_k_edges 18 --structure_init_mode patel_kappa --scales 0.5 --emb_dims 0 --structure_parameterizations support_direction --fixed_support_mask_modes maxgap_kappa --direction_init_modes random --optimizer_step_modes subject --adj_activations sigmoid --kappa_logit_bias_scales 0.0 --direction_logit_bias_scales 0.0 --main_loss_weights 1.0 --selection_agreement_weights 0.0 --direction_lr_multipliers 1.0 --freeze_direction_after_epochs -1 --lambda_l1_values 0.02 --seeds 11,22,33 --cross_pred_conditions on --directional_conditions lag_corr_raw --directional_schedule plateau --cross_pred_schedule plateau --structure_message_graph_mode raw --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --directional_target_ratios 0.01 --directional_loss_end_epochs -1 --directional_prior_lags 1,2,3 --cross_pred_lags 1,2,3 --cross_pred_fixed_weights 0.1 --cross_pred_fixed_warmup_epochs 0 --cross_pred_fixed_ramp_epochs 1 --anti_collapse_lambdas 0.1 --anti_collapse_margin_values 0.02 --anti_collapse_warmup_epochs 0 --anti_collapse_ramp_epochs 1 --parent_entropy_values 0.0 --parent_cap_values 0.0 --parent_cap_targets 0.0 --ungated_symmetry_values 0.0 --strict_margin_eps_values 0,3e-4 --experiment_tag stage1_time_anti_collapse3`

### Artifacts

- Old baseline:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260328_201524_stage1_old_baseline3.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260328_201524_stage1_old_baseline3_aggregate.csv`
- New treatment:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260328_202432_stage1_time_anti_collapse3.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_3seeds_20260328_202432_stage1_time_anti_collapse3_aggregate.csv`

### Aggregate Comparison

- Old baseline:
  - `near0(<1e-2) = 83.17%`
  - `p90 = 7.0161e-02`
  - `failure_mode = mixed_or_partial: 3/3`
  - `strict@0 = 0.8148`
  - `best_strict = 0.8519`
- New treatment:
  - `near0(<1e-2) = 90.48%`
  - `p90 = 9.3542e-03`
  - `failure_mode = weak_asymmetry: 3/3`
  - `strict@0 = 0.6296`
  - `best_strict = 0.5556`

### Interpretation

- This first single-point Stage-1 treatment **does not** satisfy the promotion
  rule.
- Relative to the old baseline:
  - `near0` got worse
    - `83.17% -> 90.48%`
  - `p90` collapsed sharply
    - `7.02e-02 -> 9.35e-03`
  - `strict@0` dropped materially
    - `0.8148 -> 0.6296`
  - `best_strict` also dropped materially
    - `0.8519 -> 0.5556`
- However, one mechanism signal is still worth noting:
  - the failure mode changed from `mixed_or_partial` to `weak_asymmetry`
  - this suggests the new time-supervised branch is **not** leaving the
    direction head in the exact old regime
  - but the asymmetry it induces is too weak and too poorly aligned to beat the
    old baseline

### Decision

- Do **not** promote this single-point treatment to `sim4`.
- Current read:
  - explicit time supervision plus anti-collapse can perturb the branch away
    from the old baseline regime
  - but this exact setting does **not** create a strong enough directional tail
    and is net-negative on the guardrail metrics
- If this branch continues, the next step should focus on why the treatment
  only reaches `weak_asymmetry` instead of producing a healthy high-margin tail,
  rather than treating this point itself as viable.

### Post-Pilot Mechanism Read

- The failure pattern is consistent across all three treatment seeds.
- Old baseline at epoch `30`:
  - all three seeds end with saturated active-pair directional contrast:
    - `dir_active_abs_margin_mean = 12.0`
    - `dir_active_abs_margin_median = 12.0`
    - `dir_active_abs_margin_p90 = 12.0`
  - `dir_loss_raw = 0`
- New treatment at epoch `30`:
  - active-pair raw directional contrast stays modest:
    - `dir_active_abs_margin_mean = 0.246 / 0.322 / 0.285`
    - `dir_active_abs_margin_median = 0.075 / 0.111 / 0.064`
  - weighted cross-prediction supervision dominates:
    - `cross_loss_weighted ≈ 0.073`
  - weighted lag-corr directional supervision is much smaller:
    - `dir_loss_weighted ≈ 0.012`
  - direct anti-collapse is effectively negligible:
    - `anti_collapse_weighted ≈ 1e-6 ~ 1e-5`

### Important Code-Level Interpretation

- The current anti-collapse term is applied in **raw direction-logit space**:
  - `abs(logits - logits.T)` with `relu(margin_floor - contrast)`
  - see `GraphExp/main_structure_learning.py`
- But the exported `support_direction` adjacency uses:
  - `support_weights * sigmoid(direction_logits - direction_logits.T)`
  - see `GraphExp/models/DDM.py`
- Therefore:
  - `anti_collapse_margin = 0.02` is a floor in raw logit space
  - it is **not** a floor on final `A_ij - A_ji`
  - the directional gate skew is `2 * sigmoid(delta) - 1 = tanh(delta / 2)`
  - so `delta = 0.02` only implies a gate skew of about `tanh(0.01) ~= 0.01`
    before support scaling
- This matches the observed failed pilot:
  - treatment `p90 = 9.35e-03`
  - treatment ends in `weak_asymmetry`

### Updated Mechanism Judgment

- The current Stage-1 treatment is not failing because the branch is completely
  inert.
- It is failing because the present formulation is effectively:
  - strong fixed cross-prediction pressure
  - weaker lag-corr directional pressure
  - almost no effective direct anti-collapse pressure in the exported margin
    space
- So the branch is pushed away from the old regime, but only into
  **low-amplitude weak asymmetry**.

### 2026-03-28 Minimal Confirmatory Probe: Raise The Raw-Logit Margin Floor

- Objective:
  - test whether the weak result above is mainly caused by the anti-collapse
    floor being too small in raw-logit space
- Setup:
  - keep the failed treatment fixed
  - same:
    - `sim3.csv`
    - `h3.txt`
    - `seed = 11`
    - `epochs = 30`
    - `cross_pred_fixed_weight = 0.1`
    - `directional_prior_lags = 1,2,3`
    - `cross_pred_lags = 1,2,3`
    - `anti_collapse_lambda = 0.1`
  - only change:
    - `anti_collapse_margin: 0.02 -> 1.0`

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_221235_stage1_margin1_seed11_probe.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_221235_stage1_margin1_seed11_probe_aggregate.csv`

- Result versus the failed `seed=11` treatment:
  - `p90: 0.0113 -> 0.0428`
  - `near0(<1e-2): 89.52% -> 82.86%`
  - `best_strict: 0.7222 -> 0.7778`
  - `strict@0: 0.7778 -> 0.7778`
  - `final_f1: 0.2276 -> 0.2276`
  - `failure_mode` remains:
    - `weak_asymmetry`

- Internal training read at epoch `30` for the `margin = 1.0` probe:
  - `dir_active_abs_margin_mean = 1.0178`
  - `dir_active_abs_margin_median = 1.0134`
  - `dir_active_abs_margin_p90 = 1.0457`
  - `dir_margin = 0.0877`
  - `cross_loss_weighted = 0.0734`
  - `dir_loss_weighted = 0.0129`
  - `anti_collapse_weighted = 3.1e-05`

- Probe interpretation:
  - this strongly supports the code-level diagnosis
  - simply increasing the raw-logit margin floor already moves:
    - `p90` upward
    - `near0` downward
    - `best_strict` upward
  - so the earlier failure was **not** just "time supervision is useless"
  - a large part of the failure came from the anti-collapse floor being
    calibrated far below the final directional margin scale we care about

### Updated Next Step

- Do **not** broaden a generic hyperparameter sweep yet.
- The next best mechanism step is now:
  - either move anti-collapse into gate-space / exported-adjacency-margin space
  - or continue with a very small number of margin-floor points that are
    meaningful in raw-logit space
- This is now a better target than revisiting Patel fusion or broad cross-pred
  weight sweeps.

### Post-Pilot Mechanism Read

- The new branch does **not** fail as a random one-off seed.
- The same pattern holds across all three treatment runs at the final epoch:
  - `dir_margin` stays low:
    - `0.017`
    - `0.024`
    - `0.019`
  - active-pair raw directional contrast stays modest rather than saturating:
    - `dir_active_abs_margin_mean = 0.246 / 0.322 / 0.285`
  - weighted cross-prediction supervision dominates:
    - `cross_loss_weighted ≈ 0.073` on all three seeds
  - weighted lag-corr directional supervision is much smaller:
    - `dir_loss_weighted ≈ 0.012`
  - direct anti-collapse is effectively negligible:
    - `anti_collapse_weighted ≈ 0` on all three seeds
- By contrast, the old baseline ends with fully saturated active-pair
  directional contrast on all three seeds:
  - `dir_active_abs_margin_mean = 12.0`
  - `dir_loss_raw = 0`

### Important Code-Level Interpretation

- The current anti-collapse term is applied in **raw direction-logit space**:
  - `abs(logits - logits.T)` with `relu(margin_floor - contrast)`
  - see `GraphExp/main_structure_learning.py`
- But the exported `support_direction` adjacency uses:
  - `support_weights * sigmoid(direction_logits - direction_logits.T)`
  - see `GraphExp/models/DDM.py`
- Therefore the current `anti_collapse_margin = 0.02` does **not** correspond
  to a final adjacency margin floor of `0.02`.
- Inference from the code:
  - the directional gate skew is `2 * sigmoid(delta) - 1 = tanh(delta / 2)`
  - so a raw-logit floor of `delta = 0.02` only implies a gate skew of about
    `tanh(0.01) ~= 0.01` before support scaling
  - after multiplying by support weights, the resulting exported adjacency
    margin can easily stay around `1e-2` or lower
- This matches the observed pilot result:
  - treatment `p90 = 9.35e-03`
  - treatment still lands in `weak_asymmetry`

### Updated Mechanism Judgment

- The current single-point Stage-1 treatment is not failing because the new
  branch is completely inert.
- It is failing because the present formulation is effectively:
  - strong fixed cross-prediction pressure
  - weak lag-corr directional pressure
  - almost no effective direct anti-collapse pressure in the exported margin
    space
- So the branch is pushed away from the old regime, but only into
  **low-amplitude weak asymmetry**, not into a strong directional tail.

### Next Minimal Step

- The next mechanism step should target the discovered mismatch directly:
  - either move anti-collapse from raw-logit contrast to
    gate-space / exported-adjacency-margin space
  - or, if staying in raw-logit space, use a much larger floor that is
    calibrated to the final margin we actually care about
- This is a better next step than broad parameter sweeping, because it follows
  directly from the current code path and the observed pilot dynamics.

### 2026-03-28 Immediate Confirmatory Probe: Larger Raw-Logit Anti-Collapse Floor

- Objective:
  - directly test the new mechanism diagnosis above
  - keep the failed treatment fixed
  - change only:
    - `anti_collapse_margin: 0.02 -> 1.0`
- Scope:
  - `sim3`
  - `seed = 11`
  - `30` epochs
  - this is a **single-seed mechanism probe**, not a new formal result

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_221235_stage1_margin1_seed11_probe.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_on_1seeds_20260328_221235_stage1_margin1_seed11_probe_aggregate.csv`
  - run dir:
    - `GraphExp/results/run_20260328_221238`

- Seed-11 comparison against the failed treatment point (`anti_collapse_margin = 0.02`):
  - `p90: 0.01130 -> 0.04285`
  - `near0(<1e-2): 89.52% -> 82.86%`
  - `best_strict: 0.7222 -> 0.7778`
  - `strict@0: 0.7778 -> 0.7778`
  - `final_f1: 0.2276 -> 0.2276`
  - `failure_mode: weak_asymmetry -> weak_asymmetry`

- Internal epoch-30 diagnostics on the new probe:
  - `dir_margin = 0.08766`
  - `dir_active_abs_margin_mean = 1.01776`
  - `dir_active_abs_margin_median = 1.01338`
  - `dir_active_abs_margin_p90 = 1.04574`
  - `cross_loss_weighted = 0.07339`
  - `dir_loss_weighted = 0.01289`
  - `anti_collapse_weighted = 3.1e-05`

- Interpretation:
  - this strongly supports the earlier code-level diagnosis
  - increasing only the raw-logit anti-collapse floor produces a clear upward
    move in the exported directional tail and reduces near-zero margins
  - therefore the previous failure was **not** mainly "time supervision has no
    effect"
  - it was much more specifically:
    - the current anti-collapse floor was too small in raw-logit space to
      matter for the exported adjacency margins

- Updated next step:
  - the next efficient move is **not** a broad sweep
  - it is to confirm this revised floor on `3` seeds, or move the anti-collapse
    constraint into gate / exported-margin space so the margin hyperparameter is
    directly interpretable

### Post-Pilot Mechanism Read

- Additional inspection of the per-seed `quality_history.csv` runs shows that
  the treatment failure is structurally consistent across all `3` seeds.
- Old baseline at epoch `30`:
  - all three seeds saturate the active directional contrast diagnostics:
    - `dir_active_abs_margin_mean = 12.0`
    - `dir_active_abs_margin_median = 12.0`
    - `dir_active_abs_margin_p90 = 12.0`
  - `dir_loss_raw = 0`
  - interpretation:
    - the baseline direction branch reaches a very strong internal pairwise
      separation regime
- New treatment at epoch `30`:
  - `dir_active_abs_margin_mean ≈ 0.246 / 0.322 / 0.285`
  - `dir_active_abs_margin_median ≈ 0.075 / 0.111 / 0.064`
  - `dir_active_abs_margin_p90 ≈ 0.592 / 0.942 / 0.988`
  - `dir_loss_raw ≈ 0.98 - 1.11`
  - `dir_loss_weighted ≈ 0.0118 - 0.0123`
  - `cross_loss_raw ≈ 0.739`
  - `cross_loss_weighted ≈ 0.0734`
  - `anti_collapse_raw ≈ 0 - 2.4e-4`
  - `anti_collapse_weighted ≈ 1e-6 - 1e-5`
- Practical read:
  - the current treatment is not being driven mainly by the anti-collapse term
  - it is effectively a:
    - cross-pred dominated branch
    - with a much smaller lag-corr directional term
    - and an almost inert anti-collapse floor

### Important Code-Level Interpretation

- In the current code, anti-collapse is applied to the **raw direction-logit
  contrast**:
  - `abs(logits - logits.T)` with
    `relu(margin_floor - contrast)`
  - see:
    - `GraphExp/main_structure_learning.py:1040`
- But in `support_direction`, the exported adjacency uses:
  - `support_weights * sigmoid(direction_logits - direction_logits.T)`
  - see:
    - `GraphExp/models/DDM.py:408`
- Therefore:
  - `anti_collapse_margin = 0.02` is a floor in raw logit space
  - it is **not** a floor on final `A_ij - A_ji`
  - this scale is small enough that anti-collapse can become nearly inactive
    while the final exported directional margins remain weak
- This matches the pilot logs:
  - anti-collapse stays numerically near zero
  - final exported `p90` still collapses to `~1e-2`

### Updated Next-Step Recommendation

- Do **not** broaden the current sweep yet.
- The next minimal mechanism test should target the discovered mismatch:
  - either move anti-collapse from raw direction-logit contrast to a quantity
    closer to the exported directional margin
  - or, before any code change, run a tiny confirmatory check with a much larger
    raw-logit margin floor to verify that the current `0.02` value is simply
    too small to matter in `support_direction`

## 2026-03-28 Runner Recovery And Smoke For The Time-Supervised Anti-Collapse Branch

### Objective

- Before spending real `sim3` budget, verify that the new Stage-1 branch can
  actually run end-to-end through:
  - runner CLI parsing
  - launch of multi-lag time supervision
  - fixed-weight cross-prediction supervision
  - gated anti-collapse loss
  - aggregate / comparison / paired CSV export
- This was a **plumbing smoke**, not a model-quality test.

### Smoke Setup

- Dataset / GT:
  - `fMRI.csv` (synthetic)
  - `h1.txt`
- Scope:
  - `epochs = 1`
  - `subject_limit = 2`
  - `time_limit = 20`
  - `seed = 11`
- Purposefully tiny so the question stays binary:
  - can the new branch run and export correctly, or not?
- Smoke command used the current runner with:
  - `directional_prior_lags = 1,2`
  - `cross_pred_lags = 1,2`
  - `cross_pred_fixed_weight = 0.1`
  - `anti_collapse_lambda = 0.1`
  - `anti_collapse_margin = 0.02`

### Smoke Artifacts

- Summary:
  - `GraphExp/results/cross_pred_v1_final_only_compare_random_cross_direction_compare_1seeds_20260328_200528_smoke_time_anti_collapse.csv`
- Aggregate:
  - `GraphExp/results/cross_pred_v1_final_only_compare_random_cross_direction_compare_1seeds_20260328_200528_smoke_time_anti_collapse_aggregate.csv`
- Comparison:
  - `GraphExp/results/cross_pred_v1_final_only_compare_random_cross_direction_compare_1seeds_20260328_200528_smoke_time_anti_collapse_comparison.csv`
- Paired:
  - `GraphExp/results/cross_pred_v1_final_only_compare_random_cross_direction_compare_1seeds_20260328_200528_smoke_time_anti_collapse_paired.csv`

### Smoke Result

- The important pass/fail answer is:
  - **PASS on plumbing**
- The runner now successfully:
  - accepts the new Stage-1 CLI
  - launches the new loss wiring
  - writes aggregate / comparison / paired outputs
  - pairs baseline and treatment rows instead of failing in the merge stage
- On this tiny 1-epoch smoke, all four cartesian-product conditions still show:
  - `symmetric_collapse`
  - `p90 = 0`
  - `near0(<1e-2) = 100%`
- This should **not** be interpreted as evidence against the branch because:
  - the smoke was deliberately undertrained
  - the purpose was runner recovery, not directional learning quality

### Additional Recovery Note

- While wiring the new branch, `GraphExp/run_cross_pred_v1_final_only_compare.py`
  had a broken nested-loop indentation in the new
  `cross_pred_fixed_weight / anti_collapse` sweep block.
- That syntax issue is now fixed.
- `py_compile` now passes for both:
  - `GraphExp/run_cross_pred_v1_final_only_compare.py`
  - `GraphExp/main_structure_learning.py`

### Recovered Old-Baseline Config

- The current old baseline was re-read from the existing formal `sim3`
  baseline aggregate instead of being reconstructed from memory.
- Old baseline config:
  - `structure_init_mode = patel_kappa`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `structure_init_scale = 0.5`
  - `lambda_l1 = 0.02`
  - `kappa_logit_bias_scale = 0.0`
  - `direction_logit_bias_scale = 0.0`
  - `adj_activation = sigmoid`
  - `optimizer_step_mode = subject`
  - `main_loss_weight = 1.0`
  - `selection_agreement_weight = 0.0`
  - `direction_lr_multiplier = 1.0`
  - `freeze_direction_after_epoch = -1`

### Next Formal Pilot

- The first real Stage-1 pilot should now move to:
  - dataset / GT:
    - `sim3.csv`
    - `h3.txt`
  - seeds:
    - `11,22,33`
  - comparison:
    - old baseline
    - one single-point new anti-collapse treatment
- To stay interpretable, the new treatment should inherit the same structural
  backbone as the old baseline:
  - `patel_kappa` init
  - `support_direction`
  - `maxgap_kappa`
  - `direction_init_mode = random`
  - no persistent `tau` bias
- Proposed first single-point treatment:
  - `cross_prediction = on`
  - `directional_prior_mode = lag_corr`
  - `lag_direction_source = raw`
  - `directional_prior_lags = 1,2,3`
  - `cross_pred_lags = 1,2,3`
  - `cross_pred_fixed_weight = 0.1`
  - `directional_target_ratio = 0.01`
  - `anti_collapse_lambda = 0.1`
  - `anti_collapse_margin = 0.02`
- Primary readouts remain:
  - `near0_pct`
  - `p90`
  - `failure_mode`
- Guardrails remain:
  - `strict@0`
  - `best_strict`

### Budget Note

- The current runner still expands `cross_pred_conditions x directional_conditions`
  as a cartesian product.
- For the formal Stage-1 check, that means the cleanest budget use is likely:
  - run the old baseline as one 3-seed sweep
  - run the new treatment as one 3-seed sweep
  - compare the two aggregate files directly
- This avoids paying for two extra hybrid conditions that are not part of the
  current experimental question.

## 2026-04-05 Direction-Side Patel Boundary

- On the current `sim4` random-support backbone, the direction-side Patel
  question is now better localized than the support-side question.
- Important implementation note:
  - the current code does **not** expose a meaningful literal `teacher off +
    gate on` branch
  - once `--disable_directional_loss` is used, the practical comparison
    collapses to a `3`-way control:
    - `teacher on + gate on`
    - `teacher on + gate off`
    - `teacher off`
- Formal `3`-seed result on:
  - `support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `support_prior_mode = patel_kappa`
  - `structure_init_mode = random`
  - `direction_init_mode = random`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `selection_score_mode = legacy`
  - dataset:
    - `sim4`
  - aggregate strict-F1:
    - `teacher on + gate on`
      - best/export/final:
        - `0.857924 / 0.819672 / 0.808743`
    - `teacher on + gate off`
      - best/export/final:
        - `0.857923 / 0.765027 / 0.825137`
    - `teacher off`
      - best/export/final:
        - `0.715847 / 0.704918 / 0.704918`
- Architecture read:
  - Patel tau teacher is still supplying real direction signal on the current
    branch
  - removing the teacher causes a clear drop in both:
    - strict-F1
    - GT signed-margin separation
  - Patel kappa gate is not the same thing:
    - it is not the primary source of directional signal
    - it changes export/final behavior more than GT-best ceiling
- Practical implication:
  - support-side Patel removal is no longer the main open question
  - the highest-value next direction-side question is:
    - what can replace the Patel tau teacher?
  - a secondary question after that is:
    - whether any gate is still needed once the teacher is replaced
- Artifacts:
  - summary:
    - `GraphExp/results/direction_patelside_randombackbone_sim4_3way_20260405_summary.csv`
  - aggregate:
    - `GraphExp/results/direction_patelside_randombackbone_sim4_3way_20260405_aggregate.csv`

## 2026-04-05 Support Learning Boundary Under Fixed Mask

- `fixed_support_mask_mode = maxgap_kappa` changes the meaning of "support
  learning" in the current architecture.
- Once that mode is enabled:
  - the admissible undirected skeleton is chosen before training from the
    support prior
  - pairs outside the mask are permanently zeroed by the model
  - the support branch cannot discover new undirected pairs outside that
    skeleton later
- Therefore the current support branch is **not** doing full support discovery.
- What it still learns is:
  - continuous support reweighting **within** the fixed skeleton
  - i.e. how strong each allowed pair should be for message passing / denoising
  - plus whether some allowed pairs get shrunk close to zero by the learned
    logits and L1 pressure
- Accurate wording for future notes:
  - do not say:
    - diffusion is learning the full support structure
  - say:
    - diffusion is learning support reweighting inside a prior-fixed hard
      support carrier
- Consequence for architecture judgment:
  - the current mainline is now better described as:
    - prior-fixed support skeleton
    - learned support weights within that skeleton
    - separate direction branch on top
  - not as:
    - unconstrained structure discovery from diffusion alone

## 2026-04-05 Scheduled-Blend Noise-Guide Smoke

- Status:
  - implemented a conservative training-time `scheduled_blend` branch
  - this upgrades dynamic noise-guide mixing from a probe-only idea to a real
    trainable code path
- Code boundary:
  - `DDM.forward(...)` now supports `noise_guide_adj_override`
  - `main_structure_learning.py` adds:
    - `build_training_noise_guide_override(...)`
    - CLI flags:
      - `--training_noise_guide_mode`
      - `--training_noise_guide_blend_target`
      - `--training_noise_guide_warmup_epochs`
      - `--training_noise_guide_ramp_epochs`
  - the override is used only during training
  - the model's stored/exported Patel noise guide remains unchanged
- Logging boundary:
  - `quality_history.csv` now records:
    - `training_noise_guide_mode`
    - `training_noise_guide_active`
    - `training_noise_guide_blend_weight`
    - `training_noise_guide_guide_l1_mean`
  - `config.npy` also records the scheduled-blend settings
- Smoke config:
  - run dir:
    - `GraphExp/results/run_20260405_202707`
  - dataset:
    - `sim4`
  - seed:
    - `11`
  - branch:
    - `support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `structure_init_mode = random`
    - `direction_init_mode = random`
    - `directional_kappa_gate = on`
    - `gradient_routing_mode = warmup_then_orthogonal`
    - `detach_direction_from_main_after_epoch = 23`
    - `causal_lag_main_weight = 0.25`
  - scheduled blend:
    - target `0.5`
    - warmup `5`
    - ramp `5`
  - smoke duration:
    - `10` epochs
- Smoke result:
  - branch ran end-to-end without crashes
  - `quality_history.csv` shows the intended activation pattern:
    - epochs `1-5`:
      - inactive
      - blend weight `0.0`
    - epochs `6-10`:
      - active
      - blend weight `0.1 -> 0.5`
  - selector audit:
    - best GT epoch `7`
      - strict `0.8525`
    - exported epoch `8`
      - strict `0.8361`
    - final epoch `10`
      - strict `0.8197`
- Current conclusion:
  - the scheduled-blend mechanism is now available for controlled experiments
  - this smoke is still **plumbing validation**, not evidence of gain
  - do not claim improvement until a paired `fixed_patel` control is run on the
    same branch
- Recommended next step:
  - run `fixed_patel` vs `scheduled_blend` on the same backbone
  - reuse:
    - `GraphExp/results/run_20260405_202707/pretrained_encoder.pt`
  - so the comparison isolates noise-guide dynamics rather than repeating full
    encoder pretraining

## 2026-04-05 Scheduled-Blend Paired Control

- Completed paired control on the same backbone using the same reused encoder
  checkpoint from:
  - `GraphExp/results/run_20260405_202707/pretrained_encoder.pt`
- Shared config:
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
- Runs:
  - `fixed_patel`
    - `GraphExp/results/run_20260405_205715`
  - `scheduled_blend`
    - `GraphExp/results/run_20260405_211525`
  - summary csv:
    - `GraphExp/results/noise_guide_paired_control_sim4_seed11_20260405_summary.csv`
- Result:
  - both variants reached the same selector-audit ceiling:
    - best GT strict F1:
      - `0.8197`
  - both exported/final models also matched:
    - exported strict F1:
      - `0.7377`
    - final strict F1:
      - `0.7377`
    - export/final gap vs GT-best:
      - `-0.0820`
- Structural side-effect:
  - `scheduled_blend` made the final adjacency notably less spiky:
    - offdiag max:
      - `0.6317 -> 0.1946`
    - offdiag std:
      - `0.0184 -> 0.0099`
    - parent entropy mean:
      - `0.1084 -> 0.1231`
- Updated judgment:
  - current evidence does **not** support promoting `scheduled_blend` as a
    performance-improving mainline change
  - it is better described as:
    - a support-side collapse-shape regularizer
  - not as:
    - a demonstrated fix for best/export mismatch
    - or a demonstrated GT-quality improvement

## 2026-04-07 Full Sweep Under The New Soft Prior

- Goal:
  - run the new soft-prior family once across all four datasets with one fixed
    backbone, then summarize the actual results and exact configuration
- Interpreted "new soft prior" configuration:
  - `support_prior_algorithm = soft_patel`
  - `direction_prior_algorithm = lag_gain`
  - `direction_init_mode = lag_gain`
  - keep the current stable non-prior backbone unchanged otherwise
- Shared config across all four datasets:
  - `seed = 11`
  - `epochs = 40`
  - `pretrain_checkpoint = .\results\run_20260405_202707\pretrained_encoder.pt`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `structure_init_mode = patel_score`
  - `support_prior_mode = patel_kappa`
  - `directional_kappa_gate = on`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `causal_lag_main_lags = 1,2`
  - `soft_patel_K = 5`
  - `soft_patel_beta = 10.0`
  - `lag_gain_ridge_lambda = 1e-3`
  - `lag_gain_score_alpha = 1.0`
  - `training_noise_guide_mode = fixed_patel`
    - note:
      - this flag name is historical
      - under this sweep it means "keep the fixed base guide built from the
        current soft support prior", not "force Patel guidance back in"
- Runs:
  - `fMRI`
    - `GraphExp/results/run_20260407_151924`
  - `sim2`
    - `GraphExp/results/run_20260407_152234`
  - `sim3`
    - `GraphExp/results/run_20260407_152704`
  - `sim4`
    - `GraphExp/results/run_20260407_153308`
  - compact CSV summary:
    - `GraphExp/results/soft_prior_full_sweep_20260407_summary.csv`
- Result summary:
  - `fMRI`
    - best GT:
      - epoch `8`
      - strict `0.8000`
      - mode `mixed_or_partial`
    - exported:
      - epoch `13`
      - strict `0.6000`
      - gap `-0.2000`
    - final:
      - epoch `40`
      - strict `0.6000`
      - gap `-0.2000`
  - `sim2`
    - best GT:
      - epoch `18`
      - strict `0.7273`
      - mode `mixed_or_partial`
    - exported:
      - epoch `12`
      - strict `0.7273`
      - gap `+0.0000`
    - final:
      - epoch `40`
      - strict `0.6364`
      - gap `-0.0909`
  - `sim3`
    - best GT:
      - epoch `8`
      - strict `0.6111`
      - mode `weak_asymmetry`
    - exported:
      - epoch `16`
      - strict `0.5556`
      - gap `-0.0556`
    - final:
      - epoch `40`
      - strict `0.5556`
      - gap `-0.0556`
  - `sim4`
    - best GT:
      - epoch `4`
      - strict `0.5000`
      - mode `symmetric_collapse`
    - exported:
      - epoch `40`
      - strict `0.3448`
      - gap `-0.1552`
    - final:
      - epoch `40`
      - strict `0.3448`
      - gap `-0.1552`
- Immediate read:
  - the new soft-prior family is fully wired and runnable across all datasets
  - but under the current backbone it does **not** look competitive as a
    drop-in replacement for the stronger Patel-based direction signal
  - degradation becomes more obvious as graph size grows:
    - `sim3`
      - weak asymmetry
    - `sim4`
      - clear symmetric collapse
- Practical conclusion:
  - do not promote this full `soft_patel + lag_gain` stack into the mainline
    yet
  - if revisited, it should be treated as:
    - a new prior family that is implemented and benchmarked
  - not as:
    - a validated replacement for the current direction teacher
