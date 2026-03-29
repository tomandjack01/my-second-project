# Cross-Prediction V1 Tracker

Last updated: 2026-03-21

## Goal

Add a small online directionality auxiliary loss so structure learning can get
non-Patel directional gradients from time-series data, while keeping diffusion
denoising as the main objective.

## Confirmed Design Decisions

- Use explicit causal semantics for the auxiliary loss.
  - Internal structure adjacency is raw convention: `A_raw[effect, cause]`.
  - The auxiliary loss will convert to causal convention first and aggregate
    sources with explicit einsum notation to avoid hidden transpose bugs.
- Do not use a learnable affine prediction head in v1.
  - Use per-node time-axis z-score normalization on both prediction and target.
  - Use `smooth_l1_loss` after z-score for robustness.
- Do not use residual/self-baseline targets in v1.
  - The encoder pretrain head is not a strictly causal per-step baseline.
- Do not feed the new loss into best-epoch selection in v1.
  - First validation must be final-only to avoid early-selection bias.

## Planned V1 Loss

Given one subject `x` with shape `[N, T]`:

1. Build frozen causal source features:
   - `h = model.prepare_clean_target(x)` under `torch.no_grad()`
   - With encoder enabled, this is the encoder's unnormalized causal state.
   - Without encoder, this falls back to raw `x`.
2. Use one-step cross prediction:
   - `h_past = h[:, :-1]`
   - `x_future = x[:, 1:]`
3. Use causal adjacency explicitly:
   - `adj_causal = to_causal_matrix_torch(model.get_structure_adj())`
   - `pred = einsum('ce,ct->et', adj_causal, h_past)`
   - `in_degree = adj_causal.sum(dim=0, keepdim=True).transpose(0, 1)`
   - `pred = pred / in_degree.clamp_min(1e-8)`
4. Normalize prediction and target along time:
   - z-score per node over the last dimension
5. Compute:
   - `raw_loss_cross = smooth_l1_loss(pred_z, x_future_z)`

## Integration Plan

1. Add CLI flags:
   - `--enable_cross_prediction`
   - `--cross_pred_target_ratio`
2. Add helper(s) in `main_structure_learning.py`:
   - per-node z-score
   - cross-prediction loss computation
3. Extend auxiliary lambda scaling:
   - keep current dir/ortho behavior
   - add third adaptive weight for cross prediction
4. Update training logs:
   - print raw/weighted cross loss
5. Save config fields so experiments are reproducible.

## Validation Plan

Smoke test:
- 1 epoch CPU run with `--enable_cross_prediction`
- Follow-up smoke test with `epochs >= 4` so cross-loss warmup ends and the
  weighted term becomes active

First proper experiment:
- final-only evaluation
- `selection_agreement_weight=0`
- `disable_directional_loss=True`
- non-Patel init such as `random`
- inspect:
  - final F1
  - margin median / p90
  - near-zero margin fraction

## Current Status

- [x] Design reviewed against current code
- [x] Decision to implement v1 confirmed
- [x] Tracking doc created
- [x] CLI flags added
- [x] Cross-prediction loss added
- [x] Adaptive lambda for cross loss added
- [x] Logging added
- [x] Smoke test passed
- [x] Final-only baseline/treatment comparison completed
- [x] Initial conclusion recorded
- [x] Plateau-schedule formal follow-up completed
- [x] Plateau ratio pilot completed
- [x] Pairwise lag-asymmetry offline signal diagnostic completed
- [x] Online directional-prior plumbing completed
- [x] Final-only Patel vs lag-corr prior comparison completed
- [x] Effective-parents cap regularizer plumbing completed
- [x] Effective-parents cap GPU smoke completed
- [x] Effective-parents cap lambda follow-up completed
- [x] Kappa-gated directional margin plumbing completed
- [x] Kappa-gated directional margin GPU smoke completed
- [x] Kappa-gated margin plus cap-lambda follow-up completed
- [x] Ungated-pair symmetry regularizer plumbing completed
- [x] Ungated-pair symmetry GPU smoke completed
- [x] Optimizer-step-mode architecture follow-up completed
- [x] Noise-guide probe diagnostic completed
- [x] Current-best `sim4` scaling smoke completed
- [x] Competitive adjacency activation plumbing completed
- [x] `sparsemax` adjacency pilot on `sim3` completed
- [x] `sparsemax` cap-ablation pilot on `sim3` completed
- [x] `sparsemax + no-cap` formal on `sim3` completed
- [x] `sim4` sparsemax scale / symmetry / message-direction follow-up completed
- [x] Goal 4B retention formal follow-up completed
- [x] Remaining-data extension on `sim2` and `fMRI.csv` completed

## Implemented Code Paths

- `main_structure_learning.py`
  - Added `zscore_per_node_time(...)`
  - Added `compute_cross_prediction_loss(model, x)`
  - Added `compute_pairwise_lagged_score_matrix(...)`
  - Added `compute_online_direction_prior_matrix(...)`
  - Added `compute_incoming_entropy_loss(...)`
  - Added `compute_excess_effective_parents_loss(...)`
  - Added `compute_incoming_parent_diagnostics(...)`
  - Added `build_kappa_gate_matrix(...)`
  - Added `compute_ungated_symmetry_loss(...)`
  - Added `compute_ungated_asymmetry_diagnostics(...)`
  - Added `compute_single_aux_lambda(...)`
  - Extended `compute_auxiliary_lambdas(...)` to return dir/ortho/cross weights
  - Added CLI flags:
    - `--directional_prior_mode`
    - `--lag_direction_source`
  - Added CLI flags:
    - `--enable_cross_prediction`
    - `--cross_pred_target_ratio`
    - `--cross_pred_schedule`
    - `--cross_pred_aggregation`
    - `--cross_pred_softmax_temp`
    - `--parent_entropy_lambda`
    - `--parent_cap_lambda`
    - `--parent_cap_target`
    - `--directional_kappa_gate`
    - `--directional_kappa_gate_quantile`
    - `--ungated_symmetry_lambda`
  - Integrated `raw_loss_cross`, `lambda_cross`, and weighted logging into
    `train_brain_connectivity(...)`
  - Integrated optional main-graph parent-entropy regularization with
    `parent_entropy_raw` / `parent_entropy_weighted` logging
  - Integrated hinge-style effective-parent cap regularization with
    `parent_cap_raw` / `parent_cap_weighted` logging
  - Integrated optional `kappa` gate on directional margin supervision
  - Integrated optional ungated-pair symmetry regularization with
    `ungated_symmetry_raw` / `ungated_symmetry_weighted` logging
  - Added cross-pred diagnostics:
    - shared-signal summaries for `h/pred/target`
    - `pred_z` vs `target_z` diagonal/off-diagonal alignment gap
    - aggregation concentration summaries:
      - `cross_agg_max_mean`
      - `cross_agg_max_p90`
      - `cross_agg_eff_parents_mean`
    - adjacency uniformity / in-degree summaries in `quality_history.csv`
    - incoming-parent concentration summaries:
      - `adj_parent_entropy_mean`
      - `adj_eff_parents_mean`
      - `adj_eff_parents_p90`
    - directional-gate summaries:
      - `directional_kappa_gate_enabled`
      - `directional_kappa_gate_threshold`
      - `directional_kappa_gate_pair_frac`
    - ungated-pair asymmetry summaries:
      - `adj_ungated_asym_mean`
      - `adj_ungated_asym_median`
      - `adj_ungated_asym_p90`
- `run_cross_pred_v1_final_only_compare.py`
  - Added passthrough / CSV fields for:
    - `cross_pred_aggregation`
    - `cross_pred_softmax_temp`
    - `lambda_l1`
    - `parent_cap_lambda`
    - `parent_cap_target`
    - `directional_kappa_gate`
    - `directional_kappa_gate_quantile`
  - Added directional-prior sweep support:
    - `--directional_conditions off,patel,lag_corr_raw,lag_corr_encoder`
    - CSV fields:
      - `enable_directional_loss`
      - `directional_prior_mode`
      - `lag_direction_source`
  - Added final-only diagnostics for:
    - `final_diff_loss`
    - `adj_offdiag_mean`
    - `adj_offdiag_cv`
    - `adj_in_degree_mean`
    - `adj_eff_parents_mean`
    - `adj_eff_parents_p90`
    - `adj_top1_share_mean`
    - `adj_top2_share_mean`
  - Added passthrough / grouping support for:
    - `optimizer_step_mode`
    - `adj_activation`
- `models/DDM.py`
  - Reused existing `prepare_clean_target(...)` as the frozen source-feature
    path for v1
  - `get_structure_adj(...)` already applies `diag_mask`, so self-loops are
    zeroed and the auxiliary loss cannot cheat through trivial self-history edges
  - Added competitive adjacency activations:
    - `sigmoid`
    - `sparsemax`
    - `entmax15`
  - Added `get_structure_message_adj(...)` so message-passing direction and
    adjacency parameterization can be controlled independently

## Completed Verification

- 1-epoch CPU smoke run completed successfully on 2026-03-11
- Command:
  - `python main_structure_learning.py --csv_path ..\\fMRI_dataset\\fMRI.csv --device cpu --epochs 1 --pretrain_epochs 50 --pretrain_checkpoint .\\results\\run_20260310_185625\\pretrained_encoder.pt --top_k_edges 5 --log_interval 1 --enable_cross_prediction --disable_directional_loss --selection_agreement_weight 0 --structure_init_mode random --structure_init_scale 0.05`
- Result directory:
  - `results/run_20260311_204226`
- Observed behavior:
  - Cross loss was computed without crashing
  - Weighted cross term stayed at `0.0000` in epoch 1, which is expected because
    `cross_warmup_epochs=3`
- 4-epoch CPU activation run completed successfully on 2026-03-11
- Command:
  - `python main_structure_learning.py --csv_path ..\\fMRI_dataset\\fMRI.csv --device cpu --epochs 4 --pretrain_epochs 50 --pretrain_checkpoint .\\results\\run_20260310_185625\\pretrained_encoder.pt --top_k_edges 5 --log_interval 1 --enable_cross_prediction --disable_directional_loss --selection_agreement_weight 0 --structure_init_mode random --structure_init_scale 0.05`
- Result directory:
  - `results/run_20260311_204615`
- Observed behavior:
  - `Cross Loss(raw/w)` stayed at `0.0000` through epochs 1-3, matching warmup
  - `Cross Loss(raw/w)` became `0.7666/0.0169` at epoch 4, confirming the
    auxiliary term became active
  - Training completed cleanly after activation, with no crash or obvious
    numerical instability

## Formal Final-Only Comparison

- Final baseline/treatment sweep completed on 2026-03-11
- Runner:
  - `run_cross_pred_v1_final_only_compare.py`
- Command:
  - `python run_cross_pred_v1_final_only_compare.py --structure_init_mode random --scales 0.01,0.05,0.1 --seeds 11,22,33 --cross_pred_conditions off,on --experiment_tag formal_cross_v1`
- Fixed controls:
  - `--disable_directional_loss`
  - `--selection_agreement_weight 0`
  - `--structure_init_mode random`
  - default `--cross_pred_target_ratio 0.02`
- Output CSVs:
  - `results/cross_pred_v1_final_only_compare_random_cross_compare_3seeds_20260311_211459_formal_cross_v1.csv`
  - `results/cross_pred_v1_final_only_compare_random_cross_compare_3seeds_20260311_211459_formal_cross_v1_aggregate.csv`
  - `results/cross_pred_v1_final_only_compare_random_cross_compare_3seeds_20260311_211459_formal_cross_v1_comparison.csv`
  - `results/cross_pred_v1_final_only_compare_random_cross_compare_3seeds_20260311_211459_formal_cross_v1_paired.csv`

## Formal Comparison Summary

Per-scale aggregate deltas, treatment minus baseline:

- `scale=0.01`
  - `delta margin_median = +1.31e-3`
  - `delta margin_p90 = +2.55e-3`
  - `delta margin_lt_1e2_frac = -6.67%`
  - `delta final_f1 = -0.0889`
  - failure mode changed from `1/3 symmetric_collapse + 2/3 weak_asymmetry`
    to `3/3 weak_asymmetry`
- `scale=0.05`
  - `delta margin_median = +1.00e-4`
  - `delta margin_p90 = +2.68e-3`
  - `delta margin_lt_1e2_frac = +3.33%`
  - `delta final_f1 = +0.0000`
  - failure mode changed from `3/3 weak_asymmetry` to
    `1/3 symmetric_collapse + 2/3 weak_asymmetry`
- `scale=0.1`
  - `delta margin_median = -3.52e-4`
  - `delta margin_p90 = -6.33e-4`
  - `delta margin_lt_1e2_frac = -3.33%`
  - `delta final_f1 = -0.0889`
  - failure mode changed from `3/3 weak_asymmetry` to
    `1/3 symmetric_collapse + 2/3 weak_asymmetry`

Overall across all 9 paired runs:

- mean `final_f1`: `0.1926 -> 0.1333`
- mean `margin_median`: `5.97e-3 -> 6.33e-3`
- mean `margin_p90`: `1.09e-2 -> 1.24e-2`
- mean `margin_lt_1e2_frac`: `77.78% -> 75.56%`
- failure modes:
  - baseline: `8 weak_asymmetry`, `1 symmetric_collapse`
  - treatment: `7 weak_asymmetry`, `2 symmetric_collapse`

Paired seed-level check:

- `margin_p90` improved in `6/9` pairs
- `margin_median` improved in `4/9` pairs and worsened in `5/9`
- `final_f1` improved in only `1/9` pairs, worsened in `5/9`, unchanged in `3/9`
- failure-mode transitions:
  - `symmetric_collapse -> weak_asymmetry`: `1`
  - `weak_asymmetry -> symmetric_collapse`: `2`
  - `weak_asymmetry -> weak_asymmetry`: `6`

## Initial Conclusion

- Cross-pred v1 is implemented correctly and enters optimization as intended.
- On this first formal final-only comparison, it does **not** reliably change the
  dominant failure mode.
- The dominant regime remains `weak_asymmetry`, with occasional
  `symmetric_collapse`.
- Relative to baseline, v1 gives only a small and inconsistent margin lift, while
  directional accuracy does not improve and mean final F1 is lower.
- Working interpretation:
  - v1 currently adds some asymmetry pressure in a subset of runs, but the
    auxiliary signal is still too weak or too indirect to consistently prevent
    final margin collapse / weak asymmetry.

## Tooling Follow-Up

- `run_cross_pred_v1_final_only_compare.py` now supports:
  - `--cross_pred_conditions off,on`
  - `--cross_pred_schedule cosine_anneal|plateau`
  - aggregate baseline/treatment comparison CSV output
  - paired seed-level baseline/treatment delta CSV output for future runs

## Plateau Schedule Follow-Up

- Motivation:
  - The original v1 formal comparison used `cross_pred_schedule=cosine_anneal`.
  - In that setting, the adaptive cross weight decayed to `0.0000` by epoch 100,
    so final-only evaluation was effectively judging the model after the
    auxiliary signal had already exited training.
- New formal treatment-only runs completed on 2026-03-13 for the same seeds and
  scales as the 2026-03-11 baseline:
  - `scale=0.05` pilot:
    - `results/cross_pred_v1_final_only_compare_random_cross_on_3seeds_20260313_090219_plateau_pilot_scale005.csv`
    - `results/cross_pred_v1_final_only_compare_random_cross_on_3seeds_20260313_090219_plateau_pilot_scale005_aggregate.csv`
  - remaining scales:
    - `results/cross_pred_v1_final_only_compare_random_cross_on_3seeds_20260313_093056_plateau_formal_remaining_scales.csv`
    - `results/cross_pred_v1_final_only_compare_random_cross_on_3seeds_20260313_093056_plateau_formal_remaining_scales_aggregate.csv`
  - merged formal plateau summary:
    - `results/cross_pred_v1_schedule_formal_random_3seeds_20260313_plateau_combined.csv`
  - merged 3-way schedule comparison:
    - `results/cross_pred_v1_schedule_formal_random_3seeds_20260313_aggregate.csv`
    - `results/cross_pred_v1_schedule_formal_random_3seeds_20260313_paired.csv`

### Plateau vs Cosine vs Baseline

Overall across all 9 paired runs:

- baseline:
  - `final_f1 = 0.1926`
  - `margin_median = 5.97e-3`
  - `margin_p90 = 1.09e-2`
  - `margin_lt_1e2_frac = 77.78%`
  - failure modes: `1 symmetric_collapse`, `8 weak_asymmetry`
- cosine treatment (`ratio=0.02`):
  - `final_f1 = 0.1333`
  - `margin_median = 6.33e-3`
  - `margin_p90 = 1.24e-2`
  - `margin_lt_1e2_frac = 75.56%`
  - failure modes: `2 symmetric_collapse`, `7 weak_asymmetry`
- plateau treatment (`ratio=0.02`):
  - `final_f1 = 0.1630`
  - `margin_median = 6.37e-3`
  - `margin_p90 = 1.21e-2`
  - `margin_lt_1e2_frac = 77.78%`
  - `final_same_dir_vs_tau = 3.11` vs baseline `2.78`
  - failure modes: `1 symmetric_collapse`, `8 weak_asymmetry`

Paired interpretation:

- Plateau clearly removes the strongest schedule confounder.
  - In new plateau logs, `weighted_cross` remains active at epoch 100.
  - Example `scale=0.05` runs:
    - old cosine epoch 100: `Cross Loss(raw/w) ≈ 0.72/0.0000`
    - new plateau epoch 100: `Cross Loss(raw/w) ≈ 0.77/0.021-0.022`
- Plateau partially recovers the harm introduced by cosine anneal.
  - `final_f1`: mean delta vs cosine `+0.0296`
  - `final_same_dir_vs_tau`: mean delta vs cosine `+0.3333`
  - one cosine `symmetric_collapse` run was recovered to `weak_asymmetry`
- Plateau does **not** yet produce a clean global failure-mode shift vs baseline.
  - vs baseline, `margin_p90` improved in `6/9` pairs and `margin_median`
    improved in `5/9`
  - but `final_f1` improved in only `1/9` pairs and worsened in `3/9`
  - failure-mode transitions vs baseline:
    - `symmetric_collapse -> weak_asymmetry`: `1`
    - `weak_asymmetry -> symmetric_collapse`: `1`
    - `weak_asymmetry -> weak_asymmetry`: `7`

Working conclusion after the plateau follow-up:

- Hypothesis A' is real:
  - the previous `cosine_anneal` schedule was masking long-horizon effects by
    turning the auxiliary term off before final evaluation.
- But fixing the schedule alone is not sufficient:
  - plateau recovers some lost margin/consistency relative to cosine
  - yet the dominant final regime is still overwhelmingly `weak_asymmetry`
  - cross-pred v1 still does not reliably prevent final low-margin behavior

## Plateau Ratio Pilot

- Goal:
  - test whether the remaining weakness is mainly just insufficient cross-loss
    weight after switching to plateau
- Fixed setup:
  - `structure_init_scale=0.05`
  - seeds `11,22,33`
  - `cross_pred_schedule=plateau`
- Output aggregate:
  - `results/cross_pred_v1_plateau_ratio_pilot_scale005_3seeds_20260313_aggregate.csv`

Aggregate summary at `scale=0.05`:

- baseline:
  - `final_f1 = 0.1333`
  - `margin_median = 6.73e-3`
  - `margin_p90 = 1.11e-2`
- cosine `ratio=0.02`:
  - `final_f1 = 0.1333`
  - `margin_median = 6.84e-3`
  - `margin_p90 = 1.38e-2`
  - failure modes: `1 symmetric_collapse`, `2 weak_asymmetry`
- plateau `ratio=0.02`:
  - `final_f1 = 0.1778`
  - `margin_median = 7.58e-3`
  - `margin_p90 = 1.34e-2`
  - failure modes: `3 weak_asymmetry`
- plateau `ratio=0.10`:
  - `final_f1 = 0.0889`
  - `margin_median = 5.17e-3`
  - `margin_p90 = 1.21e-2`
  - `final_same_dir_vs_tau = 1.33`
- plateau `ratio=0.20`:
  - `final_f1 = 0.1333`
  - `margin_median = 7.11e-3`
  - `margin_p90 = 1.51e-2`
  - `margin_lt_1e2_frac = 70.00%`

Interpretation:

- Increasing the plateau weight is **not** monotonically helpful.
- `ratio=0.10` clearly over-pushes in a bad way on this pilot:
  - lower F1
  - lower median margin
  - worse tau alignment
- `ratio=0.20` can enlarge tail margins (`p90`) for some seeds, but the gain is
  not stable enough to improve F1 or eliminate `weak_asymmetry`.
- So the remaining problem is not simply “2% is too small”.

## Diagnostics-Based Interpretation

The new `CrossDiag` / `AdjDiag` logs strongly support the ranking:
`A(schedule/weight) > C(dense uniform graph) > B(shared-signal collapse)`.

Observed on the 2026-03-13 plateau runs:

- Shared-signal hypothesis on `h` is not the main blocker right now.
  - `cross_source_mean_cos ≈ 0.148`
  - `cross_source_p90_cos ≈ 0.306`
  - `cross_source_gt_0p7_frac = 0`
  - so `prepare_clean_target(x)` is not collapsing all nodes to near-identical
    waveforms
- Predicted mixtures are still highly shared.
  - `cross_pred_mean_cos ≈ 0.85`
  - `cross_target_mean_cos ≈ 0.17`
  - `pred->target` diagonal gap is only about `+0.027` to `+0.031`
  - this implies adjacency-weighted aggregation is producing very similar
    predicted node trajectories even when the targets are not highly similar
- Final learned graphs remain dense and near-uniform.
  - final off-diagonal mean is typically around `0.074` to `0.080`
  - off-diagonal CV is only about `0.08` to `0.12`
  - mean in-degree stays around `0.30`
  - with 5 nodes, that means each target is still averaging over roughly 4
    similarly weighted incoming edges, so any single edge-direction flip has
    limited marginal effect on the cross-pred objective

Current best interpretation:

- A is confirmed, but only as a partial explanation:
  - cosine anneal was definitely hiding part of the signal
- C remains the strongest remaining bottleneck:
  - the graph stays too dense and too uniform, which washes out edge-specific
    directional gradients
- B remains worth monitoring, but the encoder output itself does not currently
  look shared enough to be the primary cause

## Updated Conclusion

- Cross-pred v1 is now validated under both cosine and plateau schedules.
- The schedule mattered materially:
  - plateau is a better control setting for any future cross-pred experiments
- However, even with plateau the v1 objective still does not reliably produce a
  decisive failure-mode change.
- Recommended next step:
  - keep plateau as the default control for future variants
  - do **not** assume larger ratio alone will solve the problem
  - focus the next design iteration on increasing edge-specific contrast
    (reducing dense averaging / uniform in-degree dilution) rather than only
    scaling the current v1 loss upward

## Softmax Aggregation Follow-Up

Claude's suggestion to try path 2 is useful.

- Why it is useful:
  - it isolates the change to the cross-pred auxiliary only
  - it directly targets the identified bottleneck: dense averaging and weak
    edge-specific sensitivity inside the auxiliary aggregation
- What it does **not** assume:
  - it does not require changing the main diffusion graph
  - it does not require immediately increasing global sparsity pressure

Implementation added on 2026-03-13:

- new cross-pred aggregation modes:
  - `mean` (existing behavior)
  - `softmax`
- new tuning flag:
  - `--cross_pred_softmax_temp`
- softmax uses masked causal logits inside the auxiliary loss, so self-loops stay
  excluded and stronger relative edges can dominate the prediction mixture

Artifacts:

- pilot runner output:
  - `results/cross_pred_v1_softmax_pilot_scale005_3seeds_20260313_aggregate.csv`
  - `results/cross_pred_v1_softmax_pilot_scale005_3seeds_20260313_paired.csv`

## Experiment Log

### Experiment: softmax smoke `temp=1.0`

- Objective:
  - test whether a default masked-softmax aggregator already increases
    edge-specific contrast relative to mean normalization
- Plan:
  - 4-epoch CPU smoke
  - fixed setup: `plateau`, `ratio=0.02`, `scale=0.05`, `random` init
- Result:
  - run: `results/run_20260313_100254`
  - `agg_max ≈ 0.26`
  - `agg_eff_parents ≈ 3.99`
  - `pred_cos ≈ 0.854`
  - `pred->target diag gap ≈ 0.030`
- Interpretation:
  - mechanically correct, but too weak
  - aggregation is still almost the same as uniform 4-way averaging
- Next step:
  - lower temperature and re-check whether concentration actually changes

### Experiment: softmax smoke `temp=0.25`

- Objective:
  - test whether a moderately sharper softmax materially changes aggregation
    concentration
- Plan:
  - same 4-epoch CPU setup, only change `softmax_temp=0.25`
- Result:
  - run: `results/run_20260313_100319`
  - `agg_max ≈ 0.30`
  - `agg_eff_parents ≈ 3.88`
  - `pred_cos ≈ 0.850`
  - `pred->target diag gap ≈ 0.035`
- Interpretation:
  - sharper than `1.0`, but still too close to dense averaging
  - not a strong enough perturbation for a meaningful full pilot
- Next step:
  - lower temperature further until the auxiliary is no longer effectively
    averaging almost all parents equally

### Experiment: softmax smoke `temp=0.1`

- Objective:
  - find the first temperature that truly changes the auxiliary aggregation
    regime
- Plan:
  - same 4-epoch CPU setup, only change `softmax_temp=0.1`
- Result:
  - run: `results/run_20260313_100358`
  - `agg_max ≈ 0.38`
  - `agg_eff_parents ≈ 3.46`
  - `pred_cos ≈ 0.811`
  - `pred->target diag gap ≈ 0.042`
- Interpretation:
  - this is the first setting that clearly reduces prediction sharing and
    increases edge-specific contrast
  - the change is still not extremely sparse, but it is now large enough to
    justify a multi-seed pilot
- Next step:
  - run a 3-seed formal pilot against the existing `plateau + mean` reference

### Experiment: softmax pilot `temp=0.1`

- Objective:
  - test whether stronger edge-specific contrast inside cross-pred alone
    improves final-only behavior, without touching main diffusion training
- Plan:
  - 100 epochs
  - seeds: `11,22,33`
  - `scale=0.05`
  - fixed controls:
    - `disable_directional_loss=True`
    - `selection_agreement_weight=0`
    - `cross_pred_schedule=plateau`
    - `cross_pred_target_ratio=0.02`
  - compare:
    - baseline
    - `plateau + mean`
    - `plateau + softmax(temp=0.1)`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_random_cross_on_3seeds_20260313_100420_softmax_t01_scale005.csv`
    - `results/cross_pred_v1_final_only_compare_random_cross_on_3seeds_20260313_100420_softmax_t01_scale005_aggregate.csv`
  - aggregate comparison:
    - baseline:
      - `final_f1 = 0.1333`
      - `margin_median = 6.73e-3`
      - `margin_p90 = 1.11e-2`
    - `plateau + mean`:
      - `final_f1 = 0.1778`
      - `margin_median = 7.58e-3`
      - `margin_p90 = 1.34e-2`
    - `plateau + softmax(temp=0.1)`:
      - `final_f1 = 0.1333`
      - `margin_median = 6.30e-3`
      - `margin_p90 = 1.48e-2`
      - `margin_lt_1e2_frac = 70.00%`
      - `final_same_dir_vs_tau = 2.33`
      - failure modes: `3 weak_asymmetry`
  - paired vs `plateau + mean`:
    - `margin_p90` improved in `3/3`
    - `margin_lt_1e2_frac` improved in `2/3`
    - `margin_median` worsened in `3/3`
    - `final_f1` improved in `0/3`
    - `final_same_dir_vs_tau` worsened sharply in one seed
  - end-of-training diagnostics confirm the mechanism changed:
    - `pred_cos` dropped from about `0.85` to about `0.73-0.76`
    - `agg_max` rose to about `0.52-0.70`
    - `agg_eff_parents` dropped to about `1.97-2.64`
- Interpretation:
  - path 2 is a valid mechanism probe: it **does** materially increase
    edge-specific contrast inside the auxiliary
  - but the first naive softmax variant does not improve final directional
    quality
  - it seems to push tail margins up while hurting median margin stability and
    not improving F1
  - likely interpretation:
    - the auxiliary is now concentrating on a few edges
    - but that concentration is not yet aligned enough with globally useful
      directionality to improve the final graph
- Next step:
  - do **not** expand this exact softmax setting to more seeds
  - if continuing path 2, prefer a small temperature sweep between `0.1` and
    `0.25` or a constrained top-k auxiliary mask
  - if prioritizing likelihood of success, path 2 should now be treated as a
    secondary branch, while the main branch should focus on reducing dense
    averaging in a more controlled way

## Main-Graph Sparsity Follow-Up

After the softmax pilot, priority shifted from auxiliary-internal aggregation
changes to the main graph itself.

Reason:

- `softmax(temp=0.1)` proved that auxiliary concentration can be changed
  mechanically
- but failure mode still stayed in `weak_asymmetry`
- this suggests the more fundamental bottleneck may be the main learned graph
  staying dense and near-uniform

Important caveat identified before running:

- stronger `L1` on `sigmoid(logits)` does **not** necessarily imply true
  sparsification
- because the model has a shared bias term, the easiest optimization path may be
  shrinking all edges together instead of separating a few strong edges from many
  weak edges
- so the right first test is an `L1-only` sweep with explicit concentration
  diagnostics, not immediately a hard top-k mask

## Experiment Log

### Experiment: `L1-only` smoke

- Objective:
  - verify that the runner correctly sweeps `lambda_l1` and records the new
    main-graph concentration metrics
- Plan:
  - 4-epoch smoke
  - fixed setup:
    - `cross_pred off`
    - `structure_init_mode=random`
    - `structure_init_scale=0.05`
    - `seed=11`
    - `lambda_l1=0.1`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_random_cross_off_1seeds_20260313_104815_l1_only_smoke.csv`
    - `results/cross_pred_v1_final_only_compare_random_cross_off_1seeds_20260313_104815_l1_only_smoke_aggregate.csv`
  - recorded fields were correct:
    - `final_diff_loss`
    - `adj_eff_parents_mean`
    - `adj_top1_share_mean`
- Interpretation:
  - instrumentation worked
  - safe to launch the longer diagnostic sweep
- Next step:
  - run the full 3-seed `L1-only` sweep at the representative `scale=0.05`

### Experiment: `L1-only` formal sweep at `scale=0.05`

- Objective:
  - distinguish between:
    - Hypothesis X: larger `L1` yields a genuinely sparser, more concentrated
      main graph while diffusion still works
    - Hypothesis Y: larger `L1` mostly shrinks all edges together without
      meaningfully reducing effective parent count
- Plan:
  - final-only baseline runs only, no cross-pred
  - fixed setup:
    - `cross_pred off`
    - `disable_directional_loss=True`
    - `selection_agreement_weight=0`
    - `structure_init_mode=random`
    - `structure_init_scale=0.05`
    - seeds `11,22,33`
  - sweep:
    - `lambda_l1 = 0.02, 0.05, 0.1, 0.2`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_random_cross_off_3seeds_20260313_104849_l1_only_scale005.csv`
    - `results/cross_pred_v1_final_only_compare_random_cross_off_3seeds_20260313_104849_l1_only_scale005_aggregate.csv`
  - convenience summaries:
    - `results/cross_pred_v1_l1_only_scale005_3seeds_20260313_summary.csv`
    - `results/cross_pred_v1_l1_only_scale005_3seeds_20260313_paired_vs_0p02.csv`

Aggregate summary:

- `lambda_l1=0.02`
  - `final_diff_loss = 1.0844`
  - `margin_median = 6.73e-3`
  - `margin_p90 = 1.11e-2`
  - `adj_offdiag_mean = 7.98e-2`
  - `adj_in_degree_mean = 0.319`
  - `adj_eff_parents_mean = 3.990`
  - `adj_top1_share_mean = 0.268`
  - failure: `3/3 weak_asymmetry`
- `lambda_l1=0.05`
  - `final_diff_loss = 1.0848`
  - `margin_median = 4.91e-3`
  - `margin_p90 = 9.35e-3`
  - `adj_offdiag_mean = 5.70e-2`
  - `adj_in_degree_mean = 0.228`
  - `adj_eff_parents_mean = 3.990`
  - `adj_top1_share_mean = 0.271`
  - failure: `2/3 symmetric_collapse + 1/3 weak_asymmetry`
- `lambda_l1=0.10`
  - `final_diff_loss = 1.1185`
  - `margin_median = 3.03e-3`
  - `margin_p90 = 5.46e-3`
  - `adj_offdiag_mean = 3.51e-2`
  - `adj_in_degree_mean = 0.140`
  - `adj_eff_parents_mean = 3.985`
  - `adj_top1_share_mean = 0.269`
  - failure: `3/3 symmetric_collapse`
- `lambda_l1=0.20`
  - `final_diff_loss = 1.1567`
  - `margin_median = 1.35e-3`
  - `margin_p90 = 2.90e-3`
  - `adj_offdiag_mean = 1.45e-2`
  - `adj_in_degree_mean = 0.058`
  - `adj_eff_parents_mean = 3.969`
  - `adj_top1_share_mean = 0.291`
  - failure: `3/3 symmetric_collapse`

Paired interpretation vs `lambda_l1=0.02`:

- `adj_offdiag_mean` drops strongly and monotonically
  - this confirms stronger `L1` does shrink the graph globally
- `adj_eff_parents_mean` barely moves
  - from `3.990` to `3.969` even at `lambda_l1=0.2`
- `adj_top1_share_mean` only rises slightly
  - from `0.268` to `0.291`
- `final_diff_loss` stays roughly flat at `0.05`, then worsens at `0.1` and
  `0.2`
- failure mode worsens well before any meaningful concentration gain appears
  - `0.05`: already drifting into `symmetric_collapse`
  - `0.1` and `0.2`: `3/3 symmetric_collapse`

Interpretation:

- Hypothesis Y is strongly supported.
- In the current sigmoid-parameterized setup, stronger `L1` mostly shrinks all
  edges together.
- It does **not** create meaningful main-graph sparsification in the sense that
  matters here:
  - effective parent count stays almost unchanged
  - top-1 mass share barely increases
- So larger `L1` alone is not a sufficient route to the kind of edge-specific
  contrast needed for direction-sensitive learning.

Current decision:

- Do **not** continue scaling `lambda_l1` upward as the main approach.
- Do **not** yet combine high-`L1` settings with cross-pred, because the
  `L1-only` diagnostic already shows those settings mainly induce shrinkage and
  collapse rather than useful concentration.

Next step:

- Move to a structural sparsity mechanism rather than plain stronger `L1`.
- Most likely next experiment:
  - keep `lambda_l1` near the current safe region
  - add an explicit main-graph sparsity operator or mask (for example top-k /
    gated sparsity in the main learned adjacency path)
  - first test it without cross-pred, exactly as was done for the `L1-only`
    sweep

## Scaling Follow-Up

After the `L1-only` sweep, there was a strategic question:

- should the next step be structural sparsity on the 5-node graph?
- or should cross-pred v1 first be tested on larger graphs, where the true graph
  is naturally sparser and edge-specific gradients might scale better?

Local dataset check on 2026-03-13:

- `fMRI.csv`: 5 nodes, `h1.txt`
- `sim2.csv`: 10 nodes, `h2.txt`
- `sim3.csv`: 15 nodes, `h3.txt`
- `sim4.csv`: 50 nodes, `h4.txt`
- all four datasets have 50 subjects × 200 time points
- the same temporal-encoder checkpoint is dimension-compatible across them

Decision:

- test scaling first on larger synthetic graphs
- start with `sim3` formal baseline/treatment
- use `sim4` only as a short diagnostic smoke before committing to a full sweep

## Experiment Log

### Experiment: `sim3` treatment smoke

- Objective:
  - verify that the existing v1 setup runs cleanly on a larger graph before
    launching a formal multi-seed comparison
- Plan:
  - 4 epochs
  - dataset: `sim3.csv`
  - GT companion: `h3.txt`
  - fixed setup:
    - `cross_pred on`
    - `plateau`
    - `mean` aggregation
    - `ratio=0.02`
    - `lambda_l1=0.02`
    - `structure_init_mode=random`
    - `structure_init_scale=0.05`
    - `top_k_edges=18`
- Result:
  - run: `results/run_20260313_130439`
  - key diagnostics at epoch 4:
    - `pred_cos ≈ 0.980`
    - `cross_target_mean_cos ≈ 0.150`
    - `agg_eff_par ≈ 13.96`
    - `pred->target diag gap ≈ 0.011`
- Interpretation:
  - the larger graph runs cleanly
  - but mean aggregation is already showing severe averaging even at 15 nodes
  - the auxiliary prediction is becoming much more shared than the targets
- Next step:
  - run a formal 3-seed baseline/treatment comparison on `sim3`

### Experiment: `sim3` formal baseline/treatment scaling test

- Objective:
  - test whether cross-pred v1 becomes more useful once the graph is larger and
    naturally less toy-like than the 5-node case
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seeds: `11,22,33`
  - compare:
    - baseline: `cross off`
    - treatment: `cross on + plateau + mean + ratio=0.02`
  - fixed setup:
    - `lambda_l1=0.02`
    - `structure_init_mode=random`
    - `structure_init_scale=0.05`
    - `top_k_edges=18`
    - pretrained encoder checkpoint reused from the 5-node experiments
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_random_cross_compare_3seeds_20260313_130545_sim3_scaling_v1.csv`
    - `results/cross_pred_v1_final_only_compare_random_cross_compare_3seeds_20260313_130545_sim3_scaling_v1_aggregate.csv`
    - `results/cross_pred_v1_final_only_compare_random_cross_compare_3seeds_20260313_130545_sim3_scaling_v1_comparison.csv`
    - `results/cross_pred_v1_final_only_compare_random_cross_compare_3seeds_20260313_130545_sim3_scaling_v1_paired.csv`

Aggregate summary:

- baseline:
  - `final_f1 = 0.1192`
  - `final_diff_loss = 1.1612`
  - `margin_median = 1.55e-3`
  - `margin_p90 = 5.83e-3`
  - `adj_eff_parents_mean = 13.38`
  - failure modes: `3/3 symmetric_collapse`
- treatment:
  - `final_f1 = 0.1355`
  - `final_diff_loss = 1.1591`
  - `margin_median = 1.71e-3`
  - `margin_p90 = 5.37e-3`
  - `adj_eff_parents_mean = 13.36`
  - failure modes: `3/3 symmetric_collapse`

Paired interpretation:

- `final_f1` improved in `2/3` seeds, but only slightly on average
  - mean delta: `+0.0163`
- `margin_median` improved slightly on average
  - mean delta: `+1.60e-4`
- `margin_p90` actually worsened on average
  - mean delta: `-4.55e-4`
- `near-zero(<1e-2)` stayed at `100%` for all runs
- failure mode did not change in any seed
  - always `symmetric_collapse`

Interpretation:

- moving from 5 nodes to 15 nodes does **not** rescue v1
- there is a tiny average F1/median-margin lift, but no qualitative regime shift
- the graph is still effectively dense from the auxiliary-loss perspective
  - `adj_eff_parents_mean ≈ 13.4` out of 14 possible parents
- so scaling to `sim3` does not support the idea that v1 becomes naturally
  strong enough once the graph is modestly larger

### Experiment: `sim4` treatment smoke

- Objective:
  - decide whether a full 50-node formal sweep is worth running immediately
    under the current v1 configuration
- Plan:
  - 4 epochs
  - dataset: `sim4.csv`
  - fixed setup:
    - `cross_pred on`
    - `plateau`
    - `mean` aggregation
    - `ratio=0.02`
    - `lambda_l1=0.02`
    - `structure_init_mode=random`
    - `structure_init_scale=0.05`
    - `top_k_edges=61`
- Result:
  - run: `results/run_20260313_132611`
  - key diagnostics at epoch 4:
    - `pred_cos ≈ 0.993`
    - `cross_target_mean_cos ≈ 0.043`
    - `agg_eff_par ≈ 48.77`
    - `agg_max ≈ 0.024`
    - `pred->target diag gap ≈ 0.008`
- Interpretation:
  - under current `mean` aggregation, the 50-node case is even more dominated by
    uniform averaging than the 15-node case
  - the auxiliary prediction is essentially a full-graph average
  - this is strong evidence that a full `sim4` formal sweep with the unchanged
    v1 setup is unlikely to reveal a qualitatively different success mode
- Next step:
  - do not immediately spend a full 6-run `sim4` budget on unchanged v1
  - if continuing the scaling branch, only do so after changing the auxiliary or
    structure dynamics enough to avoid near-uniform aggregation

## Updated Strategic Conclusion

- The 5-node graph was not the only reason v1 was weak.
- Scaling from 5 nodes to 15 nodes does not fix the problem:
  - collapse persists
  - effective parent count stays near the maximum
- The 50-node smoke suggests the unchanged `mean`-aggregation v1 may actually
  scale *worse* in terms of averaging, not better.

Current best interpretation:

- There is no strong evidence that simply moving to a larger graph will make the
  current v1 objective suddenly become sufficient.
- But there is also little value in introducing a hard main-graph top-k mask
  immediately, because that would change the main model much more radically than
  the original objective.

Most reasonable next step from here:

- stay on the larger-graph branch conceptually, but do **not** run more of the
  unchanged v1
- first design a minimally more edge-selective auxiliary or graph-dynamics
  variant that still preserves the spirit of “diffusion learns the graph, the
  auxiliary only nudges direction”

## Pairwise v1.5 Direction-Signal Diagnostic

After the scaling follow-up, a more radical but still auxiliary-only idea was
considered:

- instead of aggregating over the learned adjacency at all
- compute an independent pairwise lagged prediction-asymmetry score for every
  ordered pair `(i, j)`
- then use that pairwise asymmetry only as a direction prior on the learned
  adjacency

Key conceptual shift:

- this no longer asks the current graph to produce a useful prediction through
  aggregation
- it asks whether the data themselves contain a pairwise directional asymmetry
  signal that can supervise adjacency direction independently of graph density

### Experiment: offline pairwise asymmetry signal check

- Objective:
  - test whether the proposed v1.5 pairwise asymmetry signal has any useful GT
    direction information before integrating it into training
- Plan:
  - no training
  - datasets: `sim3`, `sim4`
  - sources:
    - raw `x`
    - encoder output `h = prepare_clean_target(x)` using the frozen temporal
      encoder checkpoint
  - method:
    - compute lagged pairwise score
    - `score[i,j] = corr(z(source_i[:-1]), z(target_j[1:]))`
    - `delta[i,j] = score[i,j] - score[j,i]`
    - average `delta` over all subjects
    - evaluate whether `sign(delta)` matches GT direction on each unordered pair
- Result:
  - saved summary:
    - `results/cross_pred_v15_pairwise_signal_offline_20260313.csv`

Summary:

- `sim3`, raw `x`
  - `same_dir_frac = 0.552`
  - `delta_abs_mean = 0.0168`
- `sim3`, encoder `h`
  - `same_dir_frac = 0.390`
  - `delta_abs_mean = 0.0131`
- `sim4`, raw `x`
  - `same_dir_frac = 0.523`
  - `delta_abs_mean = 0.0136`
- `sim4`, encoder `h`
  - `same_dir_frac = 0.533`
  - `delta_abs_mean = 0.0130`

Interpretation:

- the pairwise asymmetry idea is **not** obviously dead
  - on `sim3` raw `x` and `sim4` raw/encoder, it is modestly above chance
- but it is also clearly not a strong signal yet
  - direction accuracy is only around `52%` to `55%` in the better cases
- importantly, the signal source matters
  - on `sim3`, encoder `h` is actually worse than chance (`39%`)
  - so the proposed v1.5 should **not** assume that `prepare_clean_target(x)`
    is the best signal source

Working conclusion:

- this v1.5 direction is worth trying **before** any hard main-graph top-k mask
- but it should be framed correctly:
  - not as another aggregated prediction loss
  - as an online pairwise lag-asymmetry prior
- and the first implementation should likely use raw `x` as the default source,
  with `h` kept as an ablation

Recommended implementation shape:

- compute pairwise lagged similarity or error for all ordered pairs
- convert it into an antisymmetric `delta` matrix
- apply a confidence mask on high-`|delta|` pairs only
- reuse the current margin-style directional loss on adjacency logits instead of
  a raw sign-only agreement loss

Next step:

- implement a minimal v1.5 auxiliary based on pairwise lag asymmetry
- run the first training pilot on `sim3` rather than `sim4`
- compare:
  - source = raw `x`
  - optional ablation: source = encoder `h`

## Pairwise v1.5 Training Follow-Up

The offline diagnostic above motivated a minimal training-side integration:

- keep the existing directional margin-loss mechanism
- swap only the direction-prior source
  - baseline: directional loss off
  - reference: Patel tau prior
  - treatment: online lag-correlation prior from raw `x`

This was explicitly framed as an apples-to-apples test of signal strength, not
as proof that the model had become “fully autonomous” in discovering direction.

### Experiment: `sim3` directional-prior runner smoke (4 epochs)

- Objective:
  - verify that the updated runner can launch and log the three conditions
    cleanly:
    - `directional_off`
    - `directional_patel`
    - `directional_lag_corr_raw`
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seed: `11`
  - epochs: `4`
  - fixed setup:
    - `cross off`
    - `lambda_l1=0.02`
    - `structure_init_mode=random`
    - `structure_init_scale=0.05`
    - `top_k_edges=18`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_1seeds_20260313_140731_sim3_direction_smoke.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_1seeds_20260313_140731_sim3_direction_smoke_aggregate.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_1seeds_20260313_140731_sim3_direction_smoke_comparison.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_1seeds_20260313_140731_sim3_direction_smoke_paired.csv`
  - all three conditions produced identical final metrics
    - `final_f1 = 0.1951`
    - `margin_median = 4.23e-3`
    - `margin_p90 = 9.89e-3`
    - `adj_eff_parents_mean = 13.98`
- Interpretation:
  - the runner plumbing was correct
  - but the experiment was not informative about directional-prior quality,
    because directional auxiliary warmup is `5` epochs
  - so in a 4-epoch run, all three conditions still had effectively zero
    directional weight
- Next step:
  - rerun a short smoke past warmup so the directional prior is actually active

### Experiment: `sim3` directional-prior active smoke (8 epochs, 1 seed)

- Objective:
  - confirm that Patel and lag-corr prior modes begin to diverge once the
    directional auxiliary is active
- Plan:
  - same setup as the 4-epoch smoke, except `epochs=8`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_1seeds_20260313_140840_sim3_direction_smoke8.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_1seeds_20260313_140840_sim3_direction_smoke8_aggregate.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_1seeds_20260313_140840_sim3_direction_smoke8_comparison.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_1seeds_20260313_140840_sim3_direction_smoke8_paired.csv`

Per-condition snapshot:

- directional off:
  - `final_f1 = 0.1789`
  - `margin_median = 2.64e-3`
  - `margin_p90 = 6.62e-3`
  - `adj_eff_parents_mean = 13.98`
  - failure mode: `symmetric_collapse`
- Patel:
  - `final_f1 = 0.2276`
  - `margin_median = 6.33e-2`
  - `margin_p90 = 1.11e-1`
  - `adj_eff_parents_mean = 9.75`
  - failure mode: `mixed_or_partial`
- lag-corr raw:
  - `final_f1 = 0.1789`
  - `margin_median = 2.45e-4`
  - `margin_p90 = 1.09e-3`
  - `adj_eff_parents_mean = 14.00`
  - failure mode: `symmetric_collapse`

- Interpretation:
  - once active, the three conditions separate immediately
  - Patel behaves like a strong symmetry-breaking prior
  - lag-corr raw does **not** behave like Patel under the same margin-loss
    mechanism
    - it is weaker than Patel
    - and in this smoke it is even slightly worse than the no-direction-loss
      baseline on margins
- Next step:
  - run the full 3-seed final-only comparison before drawing a firm conclusion

### Experiment: `sim3` formal directional-prior comparison

- Objective:
  - test the user/Claude concern directly:
    - if lag-corr prior is weaker than Patel offline, does it also underperform
      Patel in the exact same final-only random-init training setup?
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seeds: `11,22,33`
  - compare:
    - baseline: `directional_off`
    - reference: `directional_patel`
    - treatment: `directional_lag_corr_raw`
  - fixed setup:
    - `cross off`
    - `lambda_l1=0.02`
    - `structure_init_mode=random`
    - `structure_init_scale=0.05`
    - `top_k_edges=18`
    - final-only evaluation
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_3seeds_20260313_140949_sim3_directional_prior_compare.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_3seeds_20260313_140949_sim3_directional_prior_compare_aggregate.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_3seeds_20260313_140949_sim3_directional_prior_compare_comparison.csv`
    - `results/cross_pred_v1_final_only_compare_random_direction_compare_3seeds_20260313_140949_sim3_directional_prior_compare_paired.csv`

Aggregate summary:

- directional off:
  - `final_f1 = 0.1192 +/- 0.0153`
  - `final_diff_loss = 1.1612 +/- 0.0122`
  - `margin_median = 1.55e-3 +/- 4.27e-4`
  - `margin_p90 = 5.83e-3 +/- 4.81e-4`
  - `adj_eff_parents_mean = 13.38`
  - failure modes: `3/3 symmetric_collapse`
- Patel:
  - `final_f1 = 0.2439 +/- 0.0000`
  - `final_diff_loss = 1.2121 +/- 0.0010`
  - `margin_median = 5.95e-1 +/- 2.23e-1`
  - `margin_p90 = 9.95e-1 +/- 0.00`
  - `adj_eff_parents_mean = 4.83`
  - failure modes: `3/3 mixed_or_partial`
- lag-corr raw:
  - `final_f1 = 0.1518 +/- 0.0153`
  - `final_diff_loss = 1.1533 +/- 0.0064`
  - `margin_median = 3.06e-4 +/- 3.68e-5`
  - `margin_p90 = 7.77e-4 +/- 1.78e-4`
  - `adj_eff_parents_mean = 13.90`
  - failure modes: `3/3 symmetric_collapse`

Direct deltas vs directional-off baseline:

- lag-corr raw minus baseline:
  - `delta final_f1 = +0.0325`
  - `delta margin_median = -1.24e-3`
  - `delta margin_p90 = -5.05e-3`
  - `delta near-zero(<1e-2) = +0.00%`
  - failure mode: no change, still `3/3 symmetric_collapse`
- Patel minus baseline:
  - `delta final_f1 = +0.1247`
  - `delta margin_median = +5.94e-1`
  - `delta margin_p90 = +9.89e-1`
  - `delta near-zero(<1e-2) = -85.40%`
  - failure mode: `3/3 symmetric_collapse -> 3/3 mixed_or_partial`

- Interpretation:
  - the user concern was correct
  - under the same margin-style training mechanism, raw lag-corr prior is much
    weaker than Patel tau
  - lag-corr raw does **not** rescue asymmetry collapse on `sim3`
    - margins are actually smaller than the no-direction-loss baseline
    - the graph remains almost maximally diffuse
  - Patel, by contrast, is strong enough to create large margins and a visibly
    more selective graph, even though its final F1 is still far from ideal
  - this means the proposed v1.5, in its current “same loss, weaker prior”
    form, should not be expected to outperform Patel or to establish a stronger
    claim of autonomous direction discovery
- Next step:
  - do **not** spend more budget treating lag-corr raw as a near-term Patel
    replacement
  - if continuing the autonomy branch, change the mechanism rather than only
    swapping in another weak pairwise statistic under the same margin loss
  - immediate practical direction:
    - keep Patel as the stronger reference prior
    - only revisit online/data-derived priors if they can provide a materially
      stronger or qualitatively different signal than the current lag-corr raw

## Main-Graph Competition Probe

The current strongest mechanism hypothesis is now:

- the core blocker is not only weak direction statistics
- it is also that the learned main graph stays near-uniform and diffuse, so
  candidate parents barely compete
- that means weak asymmetry signals get diluted before they can produce stable
  directional margins

Implementation added on 2026-03-13:

- new optional main-graph regularizer:
  - `compute_incoming_entropy_loss(adj_causal)`
  - minimize normalized incoming-parent entropy per target node
  - this targets parent competition directly and is intentionally different from
    plain `L1`
- new CLI flag:
  - `--parent_entropy_lambda`
- new diagnostics:
  - `parent_entropy_raw`
  - `parent_entropy_weighted`

### Experiment: `sim3` parent-entropy smoke

- Objective:
  - test whether a minimal main-graph competition mechanism can reduce the
    diffuse near-uniform parent mixture without introducing an external
    directional prior
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seed: `11`
  - epochs: `8`
  - compare:
    - baseline: `parent_entropy_lambda=0.0`
    - mild treatment: `parent_entropy_lambda=0.05`
    - stress treatment: `parent_entropy_lambda=0.2`
  - fixed setup:
    - `cross off`
    - `disable_directional_loss=True`
    - `lambda_l1=0.02`
    - `structure_init_mode=random`
    - `structure_init_scale=0.05`
    - `top_k_edges=18`
- Result:
  - baseline:
    - run: `results/run_20260313_150052`
    - `final_f1 = 0.1789`
    - `margin_median = 2.64e-3`
    - `margin_p90 = 6.62e-3`
    - `adj_eff_parents_mean = 13.98`
    - `adj_top1_share_mean = 0.081`
    - `adj_offdiag_cv = 0.062`
    - `parent_entropy_raw = 0.9994`
    - failure mode: `symmetric_collapse`
  - `parent_entropy_lambda=0.05`:
    - run: `results/run_20260313_150129`
    - `final_f1 = 0.1951`
    - `margin_median = 3.82e-3`
    - `margin_p90 = 1.14e-2`
    - `adj_eff_parents_mean = 13.94`
    - `adj_top1_share_mean = 0.090`
    - `adj_offdiag_cv = 0.099`
    - `parent_entropy_raw = 0.9983`
    - failure mode: `weak_asymmetry`
  - `parent_entropy_lambda=0.2`:
    - run: `results/run_20260313_150241`
    - `final_f1 = 0.1951`
    - `margin_median = 1.94e-3`
    - `margin_p90 = 4.82e-1`
    - `adj_eff_parents_mean = 2.73`
    - `adj_top1_share_mean = 0.695`
    - `adj_top2_share_mean = 0.911`
    - `adj_offdiag_cv = 2.625`
    - `parent_entropy_raw = 0.3704`
    - failure mode: `mixed_or_partial`
- Interpretation / Thoughts:
  - this mechanism is materially different from stronger `L1`
    - `L1-only` mostly caused uniform shrinkage
    - parent-entropy regularization can sharply reduce effective parent count
      and create real concentration contrast
  - the `0.05` setting is only a mild perturbation
    - it nudges concentration and margins a bit, but does not yet create a
      strong regime shift
  - the `0.2` setting shows real headroom
    - `adj_eff_parents_mean` drops from about `14` to about `2.7`
    - `top1` mass rises from about `0.08` to about `0.69`
    - this means the main graph can be pushed out of the near-uniform regime
      without relying on Patel or lag-corr priors
  - however, this is still only a 1-seed 8-epoch smoke
    - `margin_p90` becomes very large while `margin_median` stays small
    - so the current evidence is “mechanism can change the graph”, not yet
      “mechanism yields stable correct direction learning”
- Next step:
  - run a minimal multi-seed final-only control on `sim3` for
    `parent_entropy_lambda=0.0` vs `0.2`
  - keep `cross off` and `directional_off` first, to isolate whether the main
    graph competition mechanism alone reproducibly breaks collapse
  - only if that concentration shift is repeatable should it be combined with
    `cross_pred v1`

## Patel-Kappa + Competition Branch

Implementation update on 2026-03-14:

- `main_structure_learning.py`
  - added `--directional_schedule {cosine_anneal,plateau}`
  - directional margin loss can now stay active through final epoch via
    `plateau`
  - added fixed-weight parent-entropy warmup/ramp controls:
    - `--parent_entropy_warmup_epochs`
    - `--parent_entropy_ramp_epochs`
  - added epoch logging field:
    - `parent_entropy_lambda_current`
- `run_cross_pred_v1_final_only_compare.py`
  - added passthrough / CSV fields for:
    - `directional_schedule`
    - `parent_entropy_lambda`
    - `parent_entropy_warmup_epochs`
    - `parent_entropy_ramp_epochs`
  - baseline/treatment matching now treats `parent_entropy_lambda=0` as the
    true no-aux baseline, so `directional_off + entropy_on` is no longer
    misclassified as a baseline row

### Experiment: `patel_kappa` 2x2 runner smoke

- Objective:
  - verify that the revised runner and training flags correctly launch the new
    mechanism-control design:
    - symmetric skeleton init via `patel_kappa`
    - directional margin on/off
    - parent-entropy on/off
    - directional schedule fixed to `plateau`
    - parent-entropy delayed by warmup/ramp
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seed: `11`
  - epochs: `8`
  - conditions:
    - `directional_off + parent_entropy=0.0`
    - `directional_off + parent_entropy=0.2`
    - `directional_patel + parent_entropy=0.0`
    - `directional_patel + parent_entropy=0.2`
  - fixed setup:
    - `cross off`
    - `lambda_l1=0.02`
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.05`
    - `directional_schedule=plateau`
    - `parent_entropy_warmup_epochs=10`
    - `parent_entropy_ramp_epochs=10`
    - `top_k_edges=18`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_1seeds_20260314_150941_patel_kappa_2x2_smoke.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_1seeds_20260314_150941_patel_kappa_2x2_smoke_aggregate.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_1seeds_20260314_150941_patel_kappa_2x2_smoke_comparison.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_1seeds_20260314_150941_patel_kappa_2x2_smoke_paired.csv`

Per-condition snapshot:

- `directional_off + parent_entropy=0.0`
  - `final_f1 = 0.0976`
  - `margin_median = 1.50e-3`
  - `margin_p90 = 3.11e-3`
  - `adj_eff_parents_mean = 13.99`
  - failure mode: `symmetric_collapse`
- `directional_off + parent_entropy=0.2`
  - identical to the no-entropy baseline in this smoke
- `directional_patel + parent_entropy=0.0`
  - `final_f1 = 0.2439`
  - `margin_median = 8.25e-2`
  - `margin_p90 = 1.71e-1`
  - `adj_eff_parents_mean = 8.97`
  - failure mode: `mixed_or_partial`
- `directional_patel + parent_entropy=0.2`
  - identical to `directional_patel + parent_entropy=0.0` in this smoke
- Interpretation / Thoughts:
  - the revised runner and training flags are working correctly
  - `patel_kappa` + Patel margin + `directional_schedule=plateau` already
    separates cleanly from the symmetric baseline after only 8 epochs
  - the entropy branch was intentionally inactive here
    - because `parent_entropy_warmup_epochs=10`
    - and the smoke only ran for `8` epochs
  - so the equality of `parent_entropy=0.0` and `0.2` is expected and confirms
    that the warmup gate is functioning as designed
- Next step:
  - launch the corresponding 5-seed final-only formal run so entropy actually
    activates and can be evaluated

### Experiment: `patel_kappa` 2x2 formal final-only comparison

- Objective:
  - test the core mechanism claim directly:
    - under symmetric skeleton initialization, does main-graph parent
      competition amplify the usefulness of Patel margin supervision?
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seeds: `11,22,33,44,55`
  - grid:
    - `direction ∈ {off, patel}`
    - `parent_entropy_lambda ∈ {0.0, 0.05, 0.1, 0.2}`
  - fixed setup:
    - `cross off`
    - `lambda_l1=0.02`
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.05`
    - `directional_schedule=plateau`
    - `parent_entropy_warmup_epochs=10`
    - `parent_entropy_ramp_epochs=10`
    - `top_k_edges=18`
    - final-only evaluation
  - diagnostics to inspect in addition to global F1/margins:
    - GT-edge signed margin stats
    - GT forward vs reverse weight means
    - non-GT directed weight mean
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_5seeds_20260314_200057_patel_kappa_2x4_formal_gtdiag.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_5seeds_20260314_200057_patel_kappa_2x4_formal_gtdiag_aggregate.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_5seeds_20260314_200057_patel_kappa_2x4_formal_gtdiag_comparison.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_5seeds_20260314_200057_patel_kappa_2x4_formal_gtdiag_paired.csv`

Legacy aggregate summary:

- `directional_off + parent_entropy=0.0`
  - `final_f1 = 0.1008 +/- 0.0216`
  - `margin_median = 1.98e-3 +/- 4.74e-4`
  - `adj_eff_parents_mean = 13.30`
  - failure: `5/5 symmetric_collapse`
- `directional_off + parent_entropy=0.05`
  - `final_f1 = 0.2439 +/- 0.0000`
  - `margin_median = 0.0000`
  - `margin_p90 = 6.44e-1`
  - `adj_eff_parents_mean = 2.15`
  - failure: `5/5 mixed_or_partial`
- `directional_off + parent_entropy=0.10`
  - `final_f1 = 0.2341 +/- 0.0130`
  - `margin_median = 0.0000`
  - `margin_p90 = 9.48e-1`
  - `adj_eff_parents_mean = 2.02`
  - failure: `5/5 mixed_or_partial`
- `directional_off + parent_entropy=0.20`
  - `final_f1 = 0.2439 +/- 0.0000`
  - `margin_median = 0.0000`
  - `margin_p90 = 9.82e-1`
  - `adj_eff_parents_mean = 1.66`
  - failure: `5/5 mixed_or_partial`
- `directional_patel + parent_entropy=0.0`
  - `final_f1 = 0.2276 +/- 0.0000`
  - `margin_median = 5.03e-1 +/- 1.28e-1`
  - `adj_eff_parents_mean = 5.08`
  - `gt_signed_margin_median = 9.95e-1`
  - `gt_signed_margin_frac_pos = 66.67%`
  - failure: `5/5 mixed_or_partial`
- `directional_patel + parent_entropy=0.05`
  - `final_f1 = 0.2341 +/- 0.0080`
  - `margin_median = 2.80e-1 +/- 8.62e-2`
  - `adj_eff_parents_mean = 4.37`
  - `gt_signed_margin_median = 9.95e-1`
  - `gt_signed_margin_frac_pos = 68.89%`
  - failure: `5/5 mixed_or_partial`
- `directional_patel + parent_entropy=0.10`
  - `final_f1 = 0.2309 +/- 0.0065`
  - `margin_median = 1.29e-1 +/- 6.19e-2`
  - `adj_eff_parents_mean = 4.24`
  - `gt_signed_margin_median = 9.95e-1`
  - `gt_signed_margin_frac_pos = 67.78%`
  - failure: `5/5 mixed_or_partial`
- `directional_patel + parent_entropy=0.20`
  - `final_f1 = 0.2537 +/- 0.0080`
  - `margin_median = 2.79e-2 +/- 1.87e-2`
  - `adj_eff_parents_mean = 4.07`
  - `gt_signed_margin_median = 9.95e-1`
  - `gt_signed_margin_frac_pos = 64.44%`
  - failure: `5/5 mixed_or_partial`
- Interpretation / Thoughts:
  - the main pragmatic claim is supported:
    - with symmetric `patel_kappa` init, Patel margin can still learn direction
    - direction is therefore not being inherited from asymmetric initialization
  - legacy `final_f1` alone is misleading for the entropy-only branch
    - entropy-only settings look strong under the legacy metric
    - but their `margin_median = 0` and GT-edge signed margins are mostly zero
      or negative
  - among the legacy metrics, the best combined setting is
    `directional_patel + parent_entropy=0.2`
    - but the median-margin drop already suggested the interaction was not a
      simple “more synergy is always better” story
- Next step:
  - run a tie-aware post-hoc diagnostic on the completed formal outputs before
    choosing the next training sweep

### Experiment: post-hoc tie-aware diagnostic for the `2x4` formal run

- Objective:
  - determine whether the entropy-only gains are real directional improvement or
    an evaluation artifact caused by zero-margin ties being deterministically
    resolved by index order in the legacy metric
- Plan:
  - no retraining
  - reuse the completed result directories from:
    - `cross_pred_v1_final_only_compare_patel_kappa_direction_compare_5seeds_20260314_200057_patel_kappa_2x4_formal_gtdiag.csv`
  - compute supplementary strict metrics:
    - `strict_precision`
    - `strict_recall`
    - `strict_f1`
    - `strict_pred_count`
  - compare these against:
    - legacy `final_f1`
    - GT-edge signed margin stats
    - GT forward / reverse weight means
- Result:
  - entropy-only branch:
    - `parent_entropy=0.00`
      - `strict_f1 = 0.1008 +/- 0.0216`
      - `strict_recall = 34.44%`
      - `strict_pred_count = 105.0`
    - `parent_entropy=0.05`
      - `strict_f1 = 0.0943 +/- 0.0339`
      - `strict_recall = 10.00%`
      - `strict_pred_count = 19.6`
    - `parent_entropy=0.10`
      - `strict_f1 = 0.0701 +/- 0.0223`
      - `strict_recall = 6.67%`
      - `strict_pred_count = 16.2`
    - `parent_entropy=0.20`
      - `strict_f1 = 0.0625 +/- 0.0000`
      - `strict_recall = 5.56%`
      - `strict_pred_count = 14.0`
  - Patel branch:
    - `parent_entropy=0.00`
      - `strict_f1 = 0.2210 +/- 0.0024`
      - `strict_precision = 0.1325`
      - `strict_recall = 66.67%`
      - `strict_pred_count = 90.6`
    - `parent_entropy=0.05`
      - `strict_f1 = 0.2318 +/- 0.0092`
      - `strict_precision = 0.1393`
      - `strict_recall = 68.89%`
      - `strict_pred_count = 89.0`
    - `parent_entropy=0.10`
      - `strict_f1 = 0.2555 +/- 0.0119`
      - `strict_precision = 0.1575`
      - `strict_recall = 67.78%`
      - `strict_pred_count = 77.6`
    - `parent_entropy=0.20`
      - `strict_f1 = 0.2945 +/- 0.0111`
      - `strict_precision = 0.1910`
      - `strict_recall = 64.44%`
      - `strict_pred_count = 60.8`
  - GT-edge weight summary on the Patel branch:
    - `parent_entropy=0.00`
      - `gt_forward_weight_mean = 0.6425`
      - `gt_reverse_weight_mean = 0.1051`
      - `non_gt_weight_mean = 0.2483`
    - `parent_entropy=0.20`
      - `gt_forward_weight_mean = 0.6125`
      - `gt_reverse_weight_mean = 0.0607`
      - `non_gt_weight_mean = 0.2460`
- Interpretation / Thoughts:
  - the entropy-only “improvement” under legacy `final_f1` is mostly a tie
    artifact
    - once zero-margin pairs are excluded, entropy-only gets worse, not better
    - this matches the GT-edge signed-margin picture
  - the Patel branch is the real signal
    - strict `F1` improves monotonically from `0.2210 -> 0.2945` as entropy
      increases from `0.0 -> 0.2`
    - the mechanism is precision gain via pruning
      - `strict_pred_count` falls from `90.6 -> 60.8`
      - `strict_precision` rises from `0.1325 -> 0.1910`
      - recall slips only modestly (`66.67% -> 64.44%`)
  - importantly, entropy is **not** destroying true directed edges on the Patel
    branch
    - GT signed-margin median stays pegged near `0.995`
    - GT reverse weights shrink substantially
  - revised conclusion:
    - entropy alone is not a viable direction-learning mechanism
    - but entropy is a useful *precision-improving structural companion* to
      Patel margin under symmetric initialization
- Next step:
  - continue on the Patel branch only
  - run a focused entropy-extension sweep beyond `0.2` to find where precision
    gains stop and recall collapse begins

### Experiment: Patel-branch entropy extension sweep

- Objective:
  - test whether the monotonic strict-F1 improvement seen up to
    `parent_entropy=0.2` continues, plateaus, or reverses at stronger entropy
    values
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seeds: `11,22,33,44,55`
  - conditions:
    - `directional_patel + parent_entropy=0.2`
    - `directional_patel + parent_entropy=0.3`
    - `directional_patel + parent_entropy=0.4`
  - fixed setup:
    - `cross off`
    - `lambda_l1=0.02`
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.05`
    - `directional_schedule=plateau`
    - `parent_entropy_warmup_epochs=10`
    - `parent_entropy_ramp_epochs=10`
    - `top_k_edges=18`
    - `epochs=100`
  - primary diagnostics:
    - `strict_f1`
    - `strict_precision`
    - `strict_recall`
    - `strict_pred_count`
    - GT-edge signed-margin stats
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_083955_patel_entropy_extension_strict.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_083955_patel_entropy_extension_strict_aggregate.csv`

Aggregate summary:

- `parent_entropy=0.20`
  - `final_f1 = 0.2537 +/- 0.0080`
  - `strict_precision = 0.1910 +/- 0.0080`
  - `strict_recall = 64.44% +/- 2.72%`
  - `strict_f1 = 0.2945 +/- 0.0111`
  - `strict_pred_count = 60.8 +/- 2.7`
  - `margin_median = 2.79e-2 +/- 1.87e-2`
  - `gt_signed_margin_median = 9.95e-1`
  - `gt_signed_margin_frac_pos = 64.44%`
  - `final_diff_loss = 1.2157 +/- 0.0008`
- `parent_entropy=0.30`
  - `final_f1 = 0.2667 +/- 0.0080`
  - `strict_precision = 0.1974 +/- 0.0106`
  - `strict_recall = 63.33% +/- 2.72%`
  - `strict_f1 = 0.3009 +/- 0.0151`
  - `strict_pred_count = 57.8 +/- 1.5`
  - `margin_median = 1.08e-2 +/- 7.45e-3`
  - `gt_signed_margin_median = 9.95e-1`
  - `gt_signed_margin_frac_pos = 63.33%`
  - `final_diff_loss = 1.2160 +/- 0.0037`
- `parent_entropy=0.40`
  - `final_f1 = 0.2634 +/- 0.0065`
  - `strict_precision = 0.1958 +/- 0.0078`
  - `strict_recall = 62.22% +/- 2.22%`
  - `strict_f1 = 0.2979 +/- 0.0114`
  - `strict_pred_count = 57.2 +/- 0.7`
  - `margin_median = 5.40e-3 +/- 3.33e-3`
  - `gt_signed_margin_median = 9.95e-1`
  - `gt_signed_margin_frac_pos = 62.22%`
  - `final_diff_loss = 1.2140 +/- 0.0016`
- Interpretation / Thoughts:
  - the precision-improving Patel+entropy story continues beyond `0.2`, but only
    slightly
  - `0.3` is the best operating point among the tested values
    - highest legacy `final_f1`
    - highest strict `f1`
    - highest strict `precision`
  - the gain from `0.2 -> 0.3` is real but modest
    - `strict_f1: 0.2945 -> 0.3009`
    - `strict_precision: 0.1910 -> 0.1974`
    - recall falls slightly (`64.44% -> 63.33%`)
  - pushing to `0.4` does not improve further
    - strict precision and strict F1 both flatten or dip slightly
    - recall continues to drift downward
  - importantly, stronger entropy still does **not** erase true directed edges
    - GT signed-margin median stays near `0.995` across all three settings
    - diffusion loss stays essentially flat
  - current best interpretation:
    - on the Patel branch, parent entropy acts mainly as a precision-oriented
      pruning operator
    - there is a shallow optimum around `0.3`
- Next step:
  - carry forward `directional_patel + parent_entropy=0.3` as the current best
    pragmatic configuration on `sim3`
  - if continuing this branch, compare that best setting directly against:
    - the existing Patel-only reference
    - the original random-init Patel reference
  - only after locking the best operating point should any new auxiliary branch
    be revisited

### Experiment: best-vs-legacy Patel reference comparison

- Objective:
  - test the key practical question directly:
    - can the best symmetric-training configuration approach or match the old
      asymmetric `patel_score` initialization reference?
- Plan:
  - new legacy-favorable reference run:
    - dataset: `sim3.csv`
    - GT: `h3.txt`
    - seeds: `11,22,33,44,55`
    - conditions:
      - `structure_init_mode=patel_score, scale=1.0, directional_off, parent_entropy=0.0`
      - `structure_init_mode=patel_score, scale=1.0, directional_patel, parent_entropy=0.0`
    - fixed setup:
      - `cross off`
      - `lambda_l1=0.02`
      - `epochs=100`
      - final-only evaluation
  - compare those legacy references against the existing best current setting:
    - `structure_init_mode=patel_kappa`
    - `scale=0.05`
    - `directional_patel`
    - `directional_schedule=plateau`
    - `parent_entropy=0.3`
    - `parent_entropy_warmup=10`
    - `parent_entropy_ramp=10`
  - primary metrics:
    - `strict_f1`
    - `strict_precision`
    - `strict_recall`
    - `final_f1`
    - `final_diff_loss`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_patel_score_direction_compare_5seeds_20260315_120445_patel_score_legacy_reference.csv`
    - `results/cross_pred_v1_final_only_compare_patel_score_direction_compare_5seeds_20260315_120445_patel_score_legacy_reference_aggregate.csv`
    - `results/cross_pred_v1_final_only_compare_patel_score_direction_compare_5seeds_20260315_120445_patel_score_legacy_reference_comparison.csv`
    - `results/cross_pred_v1_final_only_compare_patel_score_direction_compare_5seeds_20260315_120445_patel_score_legacy_reference_paired.csv`

Aggregate summary for the new legacy references:

- `patel_score init + directional_off`
  - `final_f1 = 0.2049 +/- 0.0130`
  - `strict_f1 = 0.2033 +/- 0.0139`
  - `strict_precision = 0.1192`
  - `strict_recall = 68.89%`
  - `final_diff_loss = 1.1625 +/- 0.0089`
  - `adj_eff_parents_mean = 10.45`
  - failure: `5/5 weak_asymmetry`
- `patel_score init + directional_patel`
  - `final_f1 = 0.2309 +/- 0.0065`
  - `strict_f1 = 0.2309 +/- 0.0065`
  - `strict_precision = 0.1352`
  - `strict_recall = 78.89%`
  - `final_diff_loss = 1.2137 +/- 0.0024`
  - `adj_eff_parents_mean = 5.77`
  - failure: `5/5 mixed_or_partial`

Direct comparison against the current best symmetric-training setting:

- current best:
  - `patel_kappa init + directional_patel + parent_entropy=0.3`
  - `final_f1 = 0.2667 +/- 0.0080`
  - `strict_f1 = 0.3009 +/- 0.0151`
  - `strict_precision = 0.1974`
  - `strict_recall = 63.33%`
  - `final_diff_loss = 1.2160 +/- 0.0037`
  - `adj_eff_parents_mean = 4.06`
- Interpretation / Thoughts:
  - the best symmetric-training branch now clearly beats the old asymmetric
    `patel_score` reference on both legacy and tie-aware precision-oriented
    metrics
  - relative to `patel_score init + directional_patel`, the current best gives:
    - `final_f1: 0.2309 -> 0.2667`
    - `strict_f1: 0.2309 -> 0.3009`
    - `strict_precision: 0.1352 -> 0.1974`
    - `adj_eff_parents_mean: 5.77 -> 4.06`
  - the tradeoff is recall:
    - `strict_recall: 78.89% -> 63.33%`
  - but this recall drop is exactly the pruning effect already identified for
    parent entropy
    - the model is making far fewer directional claims
    - and a much larger fraction of those claims are correct
  - practical conclusion:
    - the current branch has already surpassed the legacy asymmetric-init
      reference as a precision-oriented operating point
    - it no longer looks like `patel_score` init contains indispensable extra
      directional information that training-time supervision cannot recover
- Next step:
  - treat `patel_kappa + directional_patel + parent_entropy=0.3` as the current
    best overall reference on `sim3`
  - only revisit asymmetric init if a later branch explicitly targets higher
    recall rather than higher precision

### Experiment: `patel_kappa` 2x2 activation smoke (`30` epochs)

- Objective:
  - run a short but actually active `2x2` check where
    `parent_entropy_warmup_epochs=10` no longer masks the entropy branch
  - verify whether entropy begins to change graph concentration before launching
    a long formal run
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seed: `11`
  - epochs: `30`
  - conditions:
    - `directional_off + parent_entropy=0.0`
    - `directional_off + parent_entropy=0.2`
    - `directional_patel + parent_entropy=0.0`
    - `directional_patel + parent_entropy=0.2`
  - fixed setup:
    - `cross off`
    - `lambda_l1=0.02`
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.05`
    - `directional_schedule=plateau`
    - `parent_entropy_warmup_epochs=10`
    - `parent_entropy_ramp_epochs=10`
    - `top_k_edges=18`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_1seeds_20260314_184403_patel_kappa_2x2_smoke30.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_1seeds_20260314_184403_patel_kappa_2x2_smoke30_aggregate.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_1seeds_20260314_184403_patel_kappa_2x2_smoke30_comparison.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_direction_compare_1seeds_20260314_184403_patel_kappa_2x2_smoke30_paired.csv`

Per-condition snapshot:

- `directional_off + parent_entropy=0.0`
  - `final_f1 = 0.0813`
  - `margin_median = 3.86e-3`
  - `margin_p90 = 7.75e-3`
  - `adj_eff_parents_mean = 13.84`
  - failure mode: `weak_asymmetry`
- `directional_off + parent_entropy=0.2`
  - `final_f1 = 0.2114`
  - `margin_median = 0.0000`
  - `margin_p90 = 9.61e-1`
  - `adj_eff_parents_mean = 1.43`
  - failure mode: `mixed_or_partial`
- `directional_patel + parent_entropy=0.0`
  - `final_f1 = 0.2114`
  - `margin_median = 8.58e-1`
  - `margin_p90 = 9.95e-1`
  - `adj_eff_parents_mean = 6.47`
  - failure mode: `mixed_or_partial`
- `directional_patel + parent_entropy=0.2`
  - `final_f1 = 0.2276`
  - `margin_median = 7.48e-1`
  - `margin_p90 = 9.95e-1`
  - `adj_eff_parents_mean = 5.78`
  - failure mode: `mixed_or_partial`
- Interpretation / Thoughts:
  - this is the first short run where all three mechanism pieces are visibly
    active at the same time:
    - symmetric `patel_kappa` init
    - sustained Patel margin supervision
    - delayed parent-entropy competition
  - parent entropy is clearly not a no-op once it activates
    - even without Patel margin, it drives `adj_eff_parents_mean` from `13.84`
      to `1.43`
    - this is a much stronger concentration shift than the no-entropy baseline
  - Patel margin remains the cleaner way to create broad directional margins
    - `directional_patel + entropy=0.0` still has the strongest median margin
  - the combined setting is plausible but not yet clearly dominant
    - `directional_patel + entropy=0.2` improves `final_f1` slightly over
      Patel-only (`0.2276` vs `0.2114`)
    - but its `margin_median` is lower than Patel-only
  - the entropy-only branch is especially interesting
    - it creates a highly concentrated graph and very high tail margins even
      without an explicit directional prior
    - but the zero median margin means its asymmetry is still unstable / very
      unevenly distributed
  - working interpretation:
    - the main-graph competition mechanism is real
    - the Patel supervision mechanism is also real
    - whether they truly *synergize* is still unresolved from one seed
- Next step:
  - proceed to the matching multi-seed formal run
  - keep the same `patel_kappa` / `plateau` / delayed-entropy setup
  - use the 2x2 design to test whether the combined branch wins on average, not
    just in this one seed

### Experiment: delayed parent-entropy timing sweep

- Objective:
  - test whether the current best branch can raise `strict_f1` further by
    delaying parent-entropy pruning
  - concrete mechanism question:
    - does giving Patel margin more time before entropy activates preserve more
      useful recall, then let entropy prune false edges later?
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seeds: `11,22,33,44,55`
  - fixed setup:
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.05`
    - `directional_patel`
    - `directional_schedule=plateau`
    - `device=cuda` when available
    - `parent_entropy_lambda=0.3`
    - `lambda_l1=0.02`
    - `cross off`
    - `top_k_edges=18`
    - `epochs=100`
  - timing settings to compare:
    - reference: existing `warmup=10, ramp=10`
    - new runs: `warmup=20, ramp=10`
    - new runs: `warmup=20, ramp=20`
    - new runs: `warmup=30, ramp=20`
  - primary diagnostics:
    - `strict_f1`
    - `strict_precision`
    - `strict_recall`
    - `strict_pred_count`
    - `gt_signed_margin_*`
- Result:
  - actual execution used the real experiment path:
    - `run_cross_pred_v1_final_only_compare.py`
    - `main_structure_learning.py`
    - device: `cuda`
  - started the first delayed-timing condition:
    - `parent_entropy_warmup_epochs=20`
    - `parent_entropy_ramp_epochs=10`
    - `5` seeds planned
  - shell batch timed out before the runner finished aggregating all `5` seeds,
    but `3` seeds completed successfully on GPU:
    - `results/run_20260315_144439` (`seed=11`)
    - `results/run_20260315_145232` (`seed=22`)
    - `results/run_20260315_150117` (`seed=33`)
  - no experiment process remained alive after timeout; seeds `44,55` were not
    completed for this condition
  - partial `3/5` result, compared against the existing `warmup=10, ramp=10`
    reference on the same seeds:
    - reference (`10,10`) mean over seeds `11,22,33`
      - `strict_f1 ~= 0.3072`
      - `strict_precision ~= 0.2014`
      - `strict_recall ~= 64.81%`
      - `final_f1 ~= 0.2710`
    - delayed entropy (`20,10`) mean over seeds `11,22,33`
      - `strict_f1 ~= 0.2882`
      - `strict_precision ~= 0.1886`
      - `strict_recall ~= 61.11%`
      - `final_f1 ~= 0.2493`
    - paired per-seed `strict_f1` deltas (`20,10 - 10,10`)
      - seed `11`: `-0.0349`
      - seed `22`: `-0.0222`
      - seed `33`: `+0.0000`
- Interpretation / Thoughts:
  - the real GPU run answered the mechanism question early enough:
    delaying parent entropy to `20,10` does **not** look like a good way to
    recover recall while preserving precision
  - on the first `3` seeds, the branch is consistently worse or equal to the
    existing `10,10` reference
    - precision falls
    - recall also falls
    - this is the opposite of the hoped-for "let Patel build recall first, then
      prune later" story
  - importantly, the graph concentration metric barely changes
    - effective parents stay near the same level (`~4.05`)
    - so the timing shift is not producing a new structural regime
  - under the "minimal, diagnostic, controlled" rule, finishing the remaining
    `2` seeds for `20,10` does not look like a good use of budget
- Next step:
  - stop the delayed-entropy timing branch unless there is a specific reason to
    fully close out the `5`-seed average for reporting
  - treat entropy timing as close to saturated around the current
    `warmup=10, ramp=10` operating point
  - move to the next precision-oriented mechanism rather than keep tuning the
    same schedule

### Experiment: narrow `parent_entropy` sweep around the current best

- Objective:
  - test whether the current best operating point can be improved with a narrow
    sweep around `parent_entropy=0.3`
  - remove the CPU-vs-GPU confound by rerunning the `0.3` anchor on GPU before
    comparing `0.25` and `0.35`
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seeds: `11,22,33,44,55`
  - fixed setup:
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.05`
    - `directional_patel`
    - `directional_schedule=plateau`
    - `parent_entropy_warmup_epochs=10`
    - `parent_entropy_ramp_epochs=10`
    - `lambda_l1=0.02`
    - `cross off`
    - `top_k_edges=18`
    - `epochs=100`
    - `device=cuda`
  - entropy values:
    - anchor: `0.30`
    - treatment: `0.25`
    - treatment: `0.35`
  - primary diagnostics:
    - `strict_f1`
    - `strict_precision`
    - `strict_recall`
    - `strict_pred_count`
    - `final_f1`
    - `gt_signed_margin_*`
- Result:
  - actual GPU runner outputs:
    - anchor `0.30`
      - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_153018_patel_entropy_narrow_gpu_anchor030.csv`
      - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_153018_patel_entropy_narrow_gpu_anchor030_aggregate.csv`
    - treatment `0.25`
      - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_161232_patel_entropy_narrow_gpu_025.csv`
      - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_161232_patel_entropy_narrow_gpu_025_aggregate.csv`
    - treatment `0.35`
      - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_171059_patel_entropy_narrow_gpu_035.csv`
      - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_171059_patel_entropy_narrow_gpu_035_aggregate.csv`

Aggregate summary on GPU:

- `parent_entropy=0.25`
  - `final_f1 = 0.2602 +/- 0.0000`
  - `strict_precision = 0.1952 +/- 0.0069`
  - `strict_recall = 63.33% +/- 2.72%`
  - `strict_f1 = 0.2984 +/- 0.0106`
  - `strict_pred_count = 58.4 +/- 1.6`
  - `final_diff_loss = 1.2150 +/- 0.0022`
  - `margin_median = 2.61e-2 +/- 2.04e-2`
- `parent_entropy=0.30` (GPU anchor)
  - `final_f1 = 0.2602 +/- 0.0000`
  - `strict_precision = 0.1952 +/- 0.0084`
  - `strict_recall = 62.22% +/- 2.22%`
  - `strict_f1 = 0.2972 +/- 0.0121`
  - `strict_pred_count = 57.4 +/- 1.0`
  - `final_diff_loss = 1.2156 +/- 0.0011`
  - `margin_median = 9.33e-3 +/- 6.28e-3`
- `parent_entropy=0.35`
  - `final_f1 = 0.2634 +/- 0.0065`
  - `strict_precision = 0.1985 +/- 0.0115`
  - `strict_recall = 63.33% +/- 4.44%`
  - `strict_f1 = 0.3023 +/- 0.0182`
  - `strict_pred_count = 57.4 +/- 1.4`
  - `final_diff_loss = 1.2134 +/- 0.0019`
  - `margin_median = 3.22e-2 +/- 3.39e-2`

Paired comparison against the GPU `0.30` anchor:

- `0.25 - 0.30`
  - mean `strict_f1` delta: `+0.00125`
  - mean `strict_precision` delta: `+0.00002`
  - mean `strict_recall` delta: `+1.11%`
- `0.35 - 0.30`
  - mean `strict_f1` delta: `+0.00510`
  - mean `strict_precision` delta: `+0.00332`
  - mean `strict_recall` delta: `+1.11%`
- Interpretation / Thoughts:
  - the narrow sweep did what it was supposed to do:
    - it checked whether the shallow optimum around `0.3` was real
    - it did so under a clean same-hardware GPU comparison
  - result:
    - `0.25` is effectively tied with `0.30`
    - `0.35` is the best point in this GPU-only sweep, but only slightly
  - the gain from `0.30 -> 0.35` is real enough to notice, but still shallow:
    - `strict_f1: 0.2972 -> 0.3023`
    - `strict_precision: 0.1952 -> 0.1985`
    - `strict_recall: 62.22% -> 63.33%`
  - importantly, this is **not** a new regime change
    - effective parents stay near `4.06`
    - failure mode stays `mixed_or_partial` for all seeds
    - GT signed-margin median stays saturated near `0.995`
  - interpretation:
    - plain entropy magnitude still behaves like a fine pruning knob
    - but by this point it is only moving the operating point a little
    - it is not solving the precision bottleneck in a qualitatively new way
  - practical reading of this sweep:
    - if a single reference value must be carried forward on GPU, use `0.35`
    - but this branch now looks close to exhausted
- Next step:
  - carry forward `parent_entropy=0.35` as the provisional GPU reference
  - stop tuning plain entropy magnitude after this narrow sweep
  - move to a new precision-oriented mechanism rather than continuing the same
    entropy-only branch

### Experiment: false-positive composition diagnostic on the current best run

- Objective:
  - determine what the current false positives are actually made of before
    changing the regularizer shape
  - specifically distinguish:
    - direction flips on true GT skeleton pairs
    - hallucinated edges on pairs absent from the GT skeleton
    - near-reciprocal / small-margin false positives that might benefit from a
      reciprocal penalty
- Plan:
  - use the existing best GPU reference:
    - `patel_kappa + directional_patel + parent_entropy=0.35`
    - `warmup=10`
    - `ramp=10`
    - `5` seeds
  - no retraining; post-hoc analysis only
  - analyze each seed's `final_epoch_adjacency_causal.csv`
  - note an evaluation semantic constraint up front:
    - under the current `strict` directional evaluation, a node pair can
      contribute at most one predicted directed edge
    - so "AB and BA both in strict prediction set" cannot literally happen
  - instead compute:
    - strict false positives split into:
      - orientation flips: predicted `B->A` where GT contains `A->B`
      - non-GT skeleton hallucinations: neither direction exists in GT
    - reciprocal-leakage proxies on strict false positives:
      - reverse/forward weight ratio
      - small-margin fraction
- Result:
  - analyzed the existing best GPU reference:
    - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260315_171059_patel_entropy_narrow_gpu_035.csv`
    - per-seed `result_dir` entries from that summary were used to load each
      `final_epoch_adjacency_causal.csv`
  - strict-evaluation semantic confirmation:
    - under the current `strict` directional evaluation, an unordered node pair
      can contribute at most one predicted directed edge
    - so literal "AB and BA both in the strict prediction set" cannot happen
  - strict false-positive composition over the `5` seeds:
    - mean strict predicted edges: `57.4`
    - mean TP: `11.4`
    - mean FP: `46.0`
    - mean direction-flip FP: `1.8`
    - mean non-GT-skeleton hallucination FP: `44.2`
    - FP composition:
      - direction flips: `3.9%` of strict FPs
      - non-GT skeleton hallucinations: `96.1%` of strict FPs
  - GT-edge miss composition:
    - mean GT correct: `11.4 / 18`
    - mean GT flipped: `1.8 / 18`
    - mean GT tied / unresolved: `4.8 / 18`
  - non-GT pair activation:
    - non-GT unordered pairs total: `87`
    - mean non-GT unordered pairs with nonzero strict direction prediction:
      `44.2 / 87 = 50.8%`
  - reciprocal-leakage proxy on strict false positives:
    - median reverse/forward weight ratio on strict FPs: `0.0025`
    - median FP margin: `0.9951`
    - fraction of strict FPs with reverse/forward ratio `> 0.9`: `9.1%`
    - fraction of strict FPs with margin `< 1e-2`: `10.4%`
- Interpretation / Thoughts:
  - this diagnostic is decisive enough to change the priority order
  - the dominant error mode is **not** reciprocal ambiguity
    - strict false positives are almost never GT-direction flips
    - they are overwhelmingly hallucinated edges on node pairs absent from the
      GT skeleton
  - reciprocal-style fixes may still help a small tail of cases, but they do
    not target the main bottleneck
    - only about `9%` of strict FPs even look strongly near-reciprocal by the
      reverse/forward ratio proxy
  - another important nuance:
    - the model is not mainly failing by confidently picking the wrong
      direction on true pairs
    - it is instead doing two things at once:
      - confidently activating many non-GT pairs
      - leaving a nontrivial chunk of true GT pairs in tie/unresolved state
  - mechanism implication:
    - the next regularizer should primarily target **non-GT pair
      hallucinations**, i.e. pair-level over-activation / excess skeleton mass
    - this is much closer to a parent-cap / sparsity-shape problem than a
      reciprocal-penalty problem
- Next step:
  - prioritize a minimal parent-cap style regularizer over a reciprocal penalty
  - a good next implementation target is:
    - penalize only the excess of effective parents above a target value
    - rather than continuing to minimize entropy uniformly

### Experiment: hinge-style `effective parents` cap smoke on GPU (`30` epochs)

- Objective:
  - implement the prioritized parent-cap regularizer suggested by the
    false-positive composition diagnostic
  - test whether a hinge on excess effective parents can reduce
    non-GT-pair over-activation more directly than unconstrained entropy
    minimization
  - keep this first pass intentionally small:
    - smoke only, not formal
    - compare targets `2.0`, `2.5`, `3.0`
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seed: `11`
  - fixed setup:
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.05`
    - `directional_patel`
    - `directional_schedule=plateau`
    - `lambda_l1=0.02`
    - `cross off`
    - `parent_entropy=0.0`
    - `top_k_edges=18`
    - `epochs=30`
    - `device=cuda`
  - new cap settings:
    - `parent_cap_lambda=0.15`
    - `parent_cap_warmup_epochs=10`
    - `parent_cap_ramp_epochs=10`
    - targets:
      - `2.0`
      - `2.5`
      - `3.0`
  - include a no-cap reference in the same smoke:
    - `parent_cap_lambda=0.0`
    - `parent_cap_target=0.0`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260315_200424_parent_cap_gpu_smoke30.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260315_200424_parent_cap_gpu_smoke30_aggregate.csv`
  - per-condition summary:
    - no cap
      - `final_f1 = 0.2276`
      - `strict_precision = 0.1333`
      - `strict_recall = 77.78%`
      - `strict_f1 = 0.2276`
      - `strict_pred_count = 105`
      - `adj_eff_parents_mean = 6.47`
      - `margin_median = 9.15e-1`
      - `margin_lt_1e2_frac = 10.48%`
    - cap `target=2.0`
      - `final_f1 = 0.2439`
      - `strict_precision = 0.1692`
      - `strict_recall = 61.11%`
      - `strict_f1 = 0.2651`
      - `strict_pred_count = 65`
      - `adj_eff_parents_mean = 4.06`
      - `margin_median = 3.96e-2`
      - `margin_lt_1e2_frac = 47.62%`
    - cap `target=2.5`
      - `final_f1 = 0.2439`
      - `strict_precision = 0.1818`
      - `strict_recall = 66.67%`
      - `strict_f1 = 0.2857`
      - `strict_pred_count = 66`
      - `adj_eff_parents_mean = 4.07`
      - `margin_median = 9.01e-2`
      - `margin_lt_1e2_frac = 47.62%`
    - cap `target=3.0`
      - `final_f1 = 0.2439`
      - `strict_precision = 0.1719`
      - `strict_recall = 61.11%`
      - `strict_f1 = 0.2683`
      - `strict_pred_count = 64`
      - `adj_eff_parents_mean = 4.07`
      - `margin_median = 1.22e-1`
      - `margin_lt_1e2_frac = 47.62%`
  - training-time cap activation snapshots from `quality_history.csv`:
    - no cap:
      - epoch `30`: `adj_eff_parents_mean = 6.4699`
    - cap `target=2.0`:
      - epoch `20`: `adj_eff_parents_mean = 4.2682`
      - epoch `30`: `adj_eff_parents_mean = 4.0561`
      - epoch `30`: `parent_cap_raw = 2.0561`
      - epoch `30`: `parent_cap_weighted = 0.3086`
    - cap `target=2.5`:
      - epoch `20`: `adj_eff_parents_mean = 4.2817`
      - epoch `30`: `adj_eff_parents_mean = 4.0657`
      - epoch `30`: `parent_cap_raw = 1.6092`
      - epoch `30`: `parent_cap_weighted = 0.2415`
    - cap `target=3.0`:
      - epoch `20`: `adj_eff_parents_mean = 4.3004`
      - epoch `30`: `adj_eff_parents_mean = 4.0712`
      - epoch `30`: `parent_cap_raw = 1.2092`
      - epoch `30`: `parent_cap_weighted = 0.1815`
- Interpretation / Thoughts:
  - the cap is **real** and not a no-op
    - compared with the no-cap reference, it sharply reduces predicted edges:
      `105 -> 64~66`
    - it also improves strict precision and strict F1 in all three target
      settings
  - the best point in this first smoke is `target=2.5`
    - among the three targets it gives the best strict precision / recall
      tradeoff on this seed:
      - `strict_precision = 0.1818`
      - `strict_recall = 66.67%`
      - `strict_f1 = 0.2857`
  - but the cap is **not yet target-tracking**
    - all three targets converge to nearly the same final
      `adj_eff_parents_mean ~= 4.06`
    - so the target value is changing the residual cap penalty magnitude
      (`2.06 -> 1.61 -> 1.21`) much more than it is changing the final graph
      regime
  - the smoke therefore supports a more precise interpretation:
    - hinge-style excess regularization is pointing in the right direction
    - but with `lambda=0.15`, `30` epochs, and `warmup/ramp=10/10`, the model
      still falls into roughly the same concentration basin
    - in other words:
      - the mechanism activates
      - it improves the operating point
      - it does not yet let the chosen target directly control the final
        effective-parent level
  - another useful contrast with the no-cap baseline:
    - no-cap Patel margin produces very large median margins (`~0.915`) but
      does so on an overactive graph
    - cap prunes many claims and improves strict precision, but median margin
      drops substantially
    - so this branch is currently trading broad confident asymmetry for cleaner
      pair activation
- Practical conclusion:
  - the smoke is positive enough to justify one more focused pass on the cap
    branch before switching priorities
  - the right follow-up is **not** yet `kappa-gated directional margin`
  - first, try to determine whether the missing piece is mostly:
    - insufficient cap strength / horizon
    - or a genuine floor around `~4` effective parents under the current
      Patel-margin + DDM setup
- Next step:
  - stay on the cap branch for one controlled follow-up
  - use `target=2.5` as the provisional center point
  - keep the rest of the setup fixed and test one minimal axis at a time:
    - either a slightly stronger `parent_cap_lambda`
    - or a longer run / earlier activation
  - only move to `kappa-gated directional margin` if the next cap follow-up
    still cannot break the `~4 effective parents` floor

### Experiment: `parent_cap` lambda follow-up at fixed `target=2.5` on GPU (`30` epochs)

- Objective:
  - test whether the `~4 effective parents` floor from the first cap smoke was
    just a weak-`lambda` effect
  - verify Claude's specific risk hypothesis:
    - stronger cap may keep deleting hallucinated edges
    - but after the weaker hallucinations are gone, it may start eroding true
      GT edges and GT margins
- Plan:
  - reuse the same `sim3` / `h3` / `seed=11` / `30`-epoch smoke setup as the
    earlier cap experiment
  - keep `target=2.5` fixed
  - compare stronger cap weights:
    - `parent_cap_lambda=0.25`
    - `parent_cap_lambda=0.35`
    - `parent_cap_lambda=0.50`
  - compare all of them against the existing references:
    - no cap
    - `parent_cap_lambda=0.15, target=2.5`
- Result:
  - runner output:
    - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260315_204634_parent_cap_lambda_followup_gpu_smoke30.csv`
    - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260315_204634_parent_cap_lambda_followup_gpu_smoke30_aggregate.csv`
  - strict metrics:
    - no cap
      - `strict_precision = 0.1333`
      - `strict_recall = 77.78%`
      - `strict_f1 = 0.2276`
      - `pred_count = 105`
    - `lambda=0.15, target=2.5`
      - `strict_precision = 0.1818`
      - `strict_recall = 66.67%`
      - `strict_f1 = 0.2857`
      - `pred_count = 66`
      - `adj_eff_parents_mean = 4.07`
      - `gt_signed_margin_median = 0.9951`
    - `lambda=0.25, target=2.5`
      - `strict_precision = 0.1964`
      - `strict_recall = 61.11%`
      - `strict_f1 = 0.2973`
      - `pred_count = 56`
      - `adj_eff_parents_mean = 4.06`
      - `gt_signed_margin_median = 0.9951`
    - `lambda=0.35, target=2.5`
      - `strict_precision = 0.1833`
      - `strict_recall = 61.11%`
      - `strict_f1 = 0.2821`
      - `pred_count = 60`
      - `adj_eff_parents_mean = 2.73`
      - `gt_signed_margin_median = 0.0613`
    - `lambda=0.50, target=2.5`
      - `strict_precision = 0.1667`
      - `strict_recall = 61.11%`
      - `strict_f1 = 0.2619`
      - `pred_count = 66`
      - `adj_eff_parents_mean = 2.40`
      - `gt_signed_margin_median = 0.0296`
  - deletion breakdown vs no-cap:
    - `lambda=0.15`
      - removed `39` edges
      - `37` hallucinated edges
      - `2` GT edges
      - remaining FP median margin: `0.9951`
    - `lambda=0.25`
      - removed `49` edges
      - `46` hallucinated edges
      - `3` GT edges
      - remaining FP median margin: `0.9951`
    - `lambda=0.35`
      - removed `48` edges
      - `45` hallucinated edges
      - `3` GT edges
      - added back `3` new hallucinated edges
      - remaining FP median margin: `0.0649`
    - `lambda=0.50`
      - removed `48` edges
      - `45` hallucinated edges
      - `3` GT edges
      - added back `9` new hallucinated edges
      - remaining FP median margin: `0.0298`
  - cap-state diagnostics at epoch `30`:
    - `lambda=0.15`
      - `parent_cap_raw = 1.6092`
      - `parent_cap_weighted = 0.2415`
      - `adj_eff_parents_mean = 4.0657`
    - `lambda=0.25`
      - `parent_cap_raw = 1.6045`
      - `parent_cap_weighted = 0.4012`
      - `adj_eff_parents_mean = 4.0575`
    - `lambda=0.35`
      - `parent_cap_raw = 0.3633`
      - `parent_cap_weighted = 0.1271`
      - `adj_eff_parents_mean = 2.7273`
    - `lambda=0.50`
      - `parent_cap_raw = 0.1052`
      - `parent_cap_weighted = 0.0525`
      - `adj_eff_parents_mean = 2.4029`
- Interpretation / Thoughts:
  - this follow-up resolved the earlier ambiguity:
    - the `~4 effective parents` floor was **not** fundamental
    - `lambda=0.15` was simply too weak to move the equilibrium
  - but Claude's mechanism warning also turned out to be right
    - `lambda=0.25` is still in the good regime:
      - it improves strict precision / F1
      - it keeps GT signed margins intact
    - `lambda=0.35` and `0.50` cross into a new regime:
      - effective parents drop into the target range
      - but GT signed-margin median collapses from `~0.995` to `0.061` / `0.030`
      - strict F1 worsens again
  - this is exactly the pattern expected if cap is starting to fight the
    directional mechanism itself rather than just pruning weak hallucinations
  - the key reading:
    - cap alone can prune well
    - cap alone can also force the graph into the target parent count
    - but once it gets strong enough to do that, it starts eroding the useful
      direction signal too
- Practical conclusion:
  - `parent_cap_lambda=0.25, target=2.5` is the best operating point inside
    this narrow smoke
  - there is no longer a strong reason to keep tuning cap magnitude in isolation
  - this is the point where the next mechanism should change **which pairs**
    receive direction supervision, not just how hard the graph is globally
    pruned

### Experiment: first `kappa-gated directional margin` smoke on GPU (`30` epochs)

- Objective:
  - test whether the current high-confidence hallucination problem can be
    attacked at the source by restricting directional margin supervision to
    high-`kappa` pairs only
  - keep the current best cap operating point fixed:
    - `parent_cap_lambda=0.25`
    - `parent_cap_target=2.5`
  - change only the directional loss:
    - ungated Patel margin
    - `kappa` gate at positive-`kappa` median (`quantile=0.50`)
    - `kappa` gate at positive-`kappa` upper quartile (`quantile=0.75`)
- Implementation:
  - `compute_directional_margin_loss(...)` now accepts an optional pair gate
  - new helper:
    - `build_kappa_gate_matrix(...)`
  - new CLI flags:
    - `--directional_kappa_gate`
    - `--directional_kappa_gate_quantile`
- Runs:
  - ungated reference:
    - `results/run_20260315_210656`
  - gate `quantile=0.50`
    - `results/run_20260315_211047`
    - gate stats:
      - threshold `0.0631`
      - pair fraction `37.33%`
  - gate `quantile=0.75`
    - `results/run_20260315_211447`
    - gate stats:
      - threshold `0.1458`
      - pair fraction `18.67%`
  - evaluation note:
    - to stay consistent with the previous cap analysis, the comparison below
      uses `final_epoch_adjacency_causal.csv` (final-only), not the
      best-epoch export
- Final-only strict metrics:
  - ungated (`lambda=0.25, target=2.5`)
    - `pred_count = 56`
    - `TP = 11`
    - `FP = 45`
    - `strict_precision = 0.1964`
    - `strict_recall = 61.11%`
    - `strict_f1 = 0.2973`
    - FP median margin = `0.9951`
    - GT median margin = `0.9951`
  - gate `quantile=0.50`
    - `pred_count = 80`
    - `TP = 12`
    - `FP = 68`
    - `strict_precision = 0.1500`
    - `strict_recall = 66.67%`
    - `strict_f1 = 0.2449`
    - FP median margin = `0.0375`
    - GT median margin = `0.9951`
  - gate `quantile=0.75`
    - `pred_count = 73`
    - `TP = 11`
    - `FP = 62`
    - `strict_precision = 0.1507`
    - `strict_recall = 61.11%`
    - `strict_f1 = 0.2418`
    - FP median margin = `0.0541`
    - GT median margin = `0.9951`
- FP confidence-shape diagnostic:
  - ungated:
    - FP margin median = `0.9951`
    - FP margin `> 0.9`: `91.11%`
    - FP margin `< 1e-2`: `2.22%`
  - gate `quantile=0.50`:
    - FP margin median = `0.0375`
    - FP margin `> 0.9`: `23.53%`
    - FP margin `< 1e-2`: `36.76%`
  - gate `quantile=0.75`:
    - FP margin median = `0.0541`
    - FP margin `> 0.9`: `30.65%`
    - FP margin `< 1e-2`: `27.42%`
- Edge-set diff vs ungated reference:
  - gate `quantile=0.50`
    - removed `4` edges, all `4` are hallucinated
    - added `28` edges:
      - `1` GT edge
      - `27` hallucinated edges
  - gate `quantile=0.75`
    - removed `4` edges, all `4` are hallucinated
    - added `21` edges:
      - `0` GT edges
      - `21` hallucinated edges
- Interpretation / Thoughts:
  - this smoke gives a very specific result:
    - `kappa` gating does **not** improve strict F1 yet
    - but it clearly changes the *kind* of false positives the model makes
  - the good news:
    - the very-high-confidence hallucination regime is strongly weakened
    - GT margins stay intact at the current moderate cap level
    - so the gate is doing something mechanistically meaningful
  - the bad news:
    - under the current strict evaluation (`margin_eps ~= 0`), many low-margin
      non-GT pairs are still counted as directional predictions
    - that means the gate converts:
      - fewer ultra-confident hallucinations
      - into more weak hallucinations
    - and strict precision drops because all those weak asymmetries still count
  - practical reading:
    - `kappa` gating looks promising as a **source-side** fix for the
      `margin≈0.995` hallucination problem
    - but by itself it is not enough; it needs a companion mechanism that
      suppresses the resulting weak residual asymmetries on non-GT pairs
- Next step:
  - keep the moderate cap setting (`lambda=0.25, target=2.5`)
  - continue on the `kappa-gated` branch rather than stronger cap-only tuning
  - the next controlled test should be:
    - moderate cap
    - `kappa`-gated directional margin
    - and one minimal mechanism to prevent weak low-margin residual false
      positives from counting as edges

### Experiment: `kappa-gated margin + cap lambda` follow-up on GPU (`30` epochs)

- Objective:
  - test Claude's concrete synergy hypothesis:
    - once `kappa` gating removes directional supervision from many non-GT
      pairs, stronger cap may be able to prune those pairs without crushing GT
      margins
  - hold the gate fixed at the better first-smoke setting:
    - `directional_kappa_gate_quantile=0.50`
  - sweep cap strength again at fixed target:
    - `parent_cap_target=2.5`
    - `parent_cap_lambda = 0.25, 0.35, 0.50`
- Plan:
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seed: `11`
  - fixed setup:
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.05`
    - `directional_patel`
    - `directional_schedule=plateau`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `lambda_l1=0.02`
    - `cross off`
    - `top_k_edges=18`
    - `epochs=30`
    - `device=cuda`
  - outputs:
    - runner summary:
      - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260315_214011_gated_cap_lambda_followup_gpu_smoke30.csv`
      - `results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260315_214011_gated_cap_lambda_followup_gpu_smoke30_aggregate.csv`
- Aggregate summary from the smoke:
  - `gate q=0.50 + cap lambda=0.25`
    - `strict_precision = 0.1500`
    - `strict_recall = 66.67%`
    - `strict_f1 = 0.2449`
    - `final_f1 = 0.2276`
    - `adj_eff_parents_mean = 3.14`
    - `gt_signed_margin_median = 0.9951`
    - `margin_median = 1.96e-2`
  - `gate q=0.50 + cap lambda=0.35`
    - `strict_precision = 0.1500`
    - `strict_recall = 66.67%`
    - `strict_f1 = 0.2449`
    - `final_f1 = 0.2276`
    - `adj_eff_parents_mean = 2.87`
    - `gt_signed_margin_median = 0.9951`
    - `margin_median = 6.50e-3`
  - `gate q=0.50 + cap lambda=0.50`
    - `strict_precision = 0.1733`
    - `strict_recall = 72.22%`
    - `strict_f1 = 0.2796`
    - `final_f1 = 0.2439`
    - `adj_eff_parents_mean = 2.85`
    - `gt_signed_margin_median = 0.9876`
    - `margin_median = 4.35e-3`
- Main comparison against ungated runs at the same cap lambda:
  - `lambda=0.25`
    - ungated:
      - `strict_f1 = 0.2973`
      - GT margin median = `0.9951`
      - FP margin median = `0.9951`
    - gated:
      - `strict_f1 = 0.2449`
      - GT margin median = `0.9951`
      - FP margin median = `0.0375`
  - `lambda=0.35`
    - ungated:
      - `strict_f1 = 0.2821`
      - GT margin median = `0.0614`
      - FP margin median = `0.0649`
    - gated:
      - `strict_f1 = 0.2449`
      - GT margin median = `0.9951`
      - FP margin median = `0.0131`
  - `lambda=0.50`
    - ungated:
      - `strict_f1 = 0.2619`
      - GT margin median = `0.0298`
      - FP margin median = `0.0298`
    - gated:
      - `strict_f1 = 0.2796`
      - GT margin median = `0.9876`
      - FP margin median = `0.0244`
- Edge-set diff, gated vs ungated at the same lambda:
  - `lambda=0.25`
    - removed `4` edges, all `4` are hallucinated
    - added `28` edges:
      - `1` GT edge
      - `27` hallucinated edges
  - `lambda=0.35`
    - removed `6` edges, all `6` are hallucinated
    - added `26` edges:
      - `1` GT edge
      - `25` hallucinated edges
  - `lambda=0.50`
    - removed `18` edges, all `18` are hallucinated
    - added `27` edges:
      - `2` GT edges
      - `25` hallucinated edges
- Within the gated branch, stronger cap vs `lambda=0.25`:
  - `lambda=0.35`
    - removed `7` edges, all `7` are hallucinated
    - added `7` edges, all `7` are hallucinated
  - `lambda=0.50`
    - removed `15` edges, all `15` are hallucinated
    - added `10` edges:
      - `1` GT edge
      - `9` hallucinated edges
  - key point:
    - inside the gated branch, stronger cap no longer shows the clear
      GT-margin destruction that appeared in the ungated branch
- Margin-threshold diagnostic on the gated branch:
  - because current `strict` counts any nonzero directional margin as a
    prediction, low-margin residual FPs still hurt the headline metric
  - gated `lambda=0.50` final-only:
    - `eps = 1e-12`
      - `strict_f1 = 0.2796`
    - `eps = 1e-2`
      - `strict_f1 = 0.3582`
    - `eps = 5e-2`
      - `strict_f1 = 0.4211`
    - `eps = 1e-1`
      - `strict_f1 = 0.4444`
  - for comparison, ungated `lambda=0.25`:
    - `eps = 1e-12`
      - `strict_f1 = 0.2973`
    - `eps = 1e-2`
      - `strict_f1 = 0.3014`
    - `eps = 5e-2`
      - `strict_f1 = 0.3056`
    - `eps = 1e-1`
      - `strict_f1 = 0.3143`
- Interpretation / Thoughts:
  - Claude's mechanism story was substantially correct
  - the new synergy result is:
    - under gating, stronger cap can push `eff_parents` into the target range
      (`~2.85`) **without** collapsing GT margins
    - this is exactly what failed in the ungated branch
  - the remaining issue is now much narrower:
    - gating + stronger cap converts the error regime into many weak residual
      FPs rather than a smaller set of ultra-confident FPs
    - under the current strict metric (`margin_eps ~= 0`), those weak residual
      asymmetries still count as full directional predictions
  - so the branch has become much cleaner mechanistically:
    - source-side hallucination confidence is reduced by the gate
    - pair-count pressure is supplied by the cap
    - GT margins largely survive
  - this is the first branch where the two mechanisms look genuinely
    complementary rather than adversarial
- Practical conclusion:
  - `gate q=0.50 + cap lambda=0.50 + target=2.5` is the best result of this
    combined smoke under the current strict metric
    - it beats gated `0.25/0.35`
    - and it avoids the GT-margin collapse of ungated `0.35/0.50`
  - however, the branch still underperforms ungated `lambda=0.25` on the raw
    `strict_f1` headline because the current metric counts many weak residual
    asymmetries as full edges
  - the threshold-scan result strongly suggests the next improvement lever is
    not "more cap" but a minimal way to suppress or ignore very low-margin
    residual false positives
- Next step:
  - carry forward:
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
  - and test one minimal residual-FP mechanism rather than changing the main
    losses again

### Experiment: ungated-pair symmetry regularizer on top of gated+cap branch

- Objective:
  - test a minimal residual-FP mechanism that does not touch the gated
    directional pairs:
    - explicitly suppress asymmetry on pairs outside the directional
      `kappa` gate
  - target the exact failure mode left by the previous branch:
    - many weak residual false positives that survive because current `strict`
      counts any nonzero directional margin
- Mechanism:
  - new loss on causal adjacency:
    - mean `|A_ij - A_ji|` over pairs outside the directional `kappa` gate
  - intent:
    - keep gated high-`kappa` directional pairs free
    - push low-`kappa` residual asymmetries toward ties
- Implementation:
  - new helper:
    - `compute_ungated_symmetry_loss(...)`
  - new diagnostics:
    - `compute_ungated_asymmetry_diagnostics(...)`
  - new CLI:
    - `--ungated_symmetry_lambda`
    - `--ungated_symmetry_warmup_epochs`
    - `--ungated_symmetry_ramp_epochs`
- Plan:
  - start from the current best combined branch:
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
  - dataset: `sim3.csv`
  - GT: `h3.txt`
  - seed: `11`
  - epochs: `30`
  - sweep:
    - `ungated_symmetry_lambda = 0.10`
    - `ungated_symmetry_lambda = 0.20`
    - `ungated_symmetry_lambda = 0.50`
  - timing:
    - `warmup=10`
    - `ramp=10`
  - baseline for comparison:
    - existing gated+cap run with no symmetry:
      - `results/run_20260315_214810`
- Runs:
  - `lambda=0.10`
    - `results/run_20260315_220146`
  - `lambda=0.20`
    - `results/run_20260315_220438`
  - `lambda=0.50`
    - `results/run_20260315_220842`
- Final-only strict metrics:
  - baseline gated+cap (`sym=0.0`)
    - `strict_precision = 0.1733`
    - `strict_recall = 72.22%`
    - `strict_f1 = 0.2796`
    - `pred_count = 75`
    - FP margin median = `0.0244`
    - GT margin median = `0.9876`
  - `sym=0.10`
    - `strict_precision = 0.1625`
    - `strict_recall = 72.22%`
    - `strict_f1 = 0.2653`
    - `pred_count = 80`
    - FP margin median = `0.0127`
    - GT margin median = `0.9942`
  - `sym=0.20`
    - `strict_precision = 0.1646`
    - `strict_recall = 72.22%`
    - `strict_f1 = 0.2680`
    - `pred_count = 79`
    - FP margin median = `0.0117`
    - GT margin median = `0.9951`
  - `sym=0.50`
    - `strict_precision = 0.1625`
    - `strict_recall = 72.22%`
    - `strict_f1 = 0.2653`
    - `pred_count = 80`
    - FP margin median = `0.0104`
    - GT margin median = `0.9951`
- Ungated-asymmetry diagnostics at epoch `30`:
  - baseline gated+cap (`sym=0.0`)
    - offline diagnostic on ungated pairs:
      - mean asymmetry `~0.151`
      - median asymmetry `~0.0030`
      - p90 asymmetry `~0.798`
  - `sym=0.10`
    - `adj_ungated_asym_mean = 0.1239`
    - `adj_ungated_asym_median = 0.0025`
    - `adj_ungated_asym_p90 = 0.5945`
  - `sym=0.20`
    - `adj_ungated_asym_mean = 0.0993`
    - `adj_ungated_asym_median = 0.0023`
    - `adj_ungated_asym_p90 = 0.4233`
  - `sym=0.50`
    - `adj_ungated_asym_mean = 0.0429`
    - `adj_ungated_asym_median = 0.0016`
    - `adj_ungated_asym_p90 = 0.1114`
- Edge-set diff vs baseline gated+cap (`sym=0.0`):
  - `sym=0.10`
    - removed `4` edges, all hallucinated
    - added `9` edges, all hallucinated
  - `sym=0.20`
    - removed `7` edges, all hallucinated
    - added `11` edges, all hallucinated
  - `sym=0.50`
    - removed `8` edges, all hallucinated
    - added `13` edges, all hallucinated
  - importantly:
    - no GT edges were removed by the symmetry regularizer in this smoke
    - no GT edges were added either
- Margin-threshold diagnostic:
  - baseline gated+cap (`sym=0.0`)
    - `eps = 1e-12`: `strict_f1 = 0.2796`
    - `eps = 1e-2`: `strict_f1 = 0.3582`
    - `eps = 5e-2`: `strict_f1 = 0.4211`
    - `eps = 1e-1`: `strict_f1 = 0.4444`
  - `sym=0.20`
    - `eps = 1e-12`: `strict_f1 = 0.2680`
    - `eps = 1e-2`: `strict_f1 = 0.3750`
    - `eps = 5e-2`: `strict_f1 = 0.4528`
    - `eps = 1e-1`: `strict_f1 = 0.4706`
  - `sym=0.50`
    - `eps = 1e-12`: `strict_f1 = 0.2653`
    - `eps = 1e-2`: `strict_f1 = 0.3750`
    - `eps = 5e-2`: `strict_f1 = 0.4800`
    - `eps = 1e-1`: `strict_f1 = 0.5106`
- Interpretation / Thoughts:
  - the symmetry regularizer is mechanistically real
    - it steadily suppresses ungated-pair asymmetry
    - it lowers FP margins further
    - it does so without damaging GT margins
  - but under the current raw `strict` metric (`margin_eps ~= 0`), it does not
    improve headline strict F1
    - the branch still produces many residual nonzero asymmetries
    - once those tiny asymmetries are counted as full predictions, strict
      precision stays limited
  - so this smoke sharpened the diagnosis rather than changing the winner:
    - training can now separate GT and FP confidence quite well
    - the remaining bottleneck is how the evaluation / prediction rule treats
      very small directional margins
  - another important nuance:
    - stronger symmetry does **not** look harmful in the way stronger cap was
    - it mainly redistributes / shrinks hallucinated predictions
    - but because the current strict metric has effectively no deadzone, the
      benefit only becomes visible once a small margin threshold is applied
- Practical conclusion:
  - the current best raw-strict operating point still remains:
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `ungated_symmetry_lambda=0.0`
  - but if the prediction rule is allowed even a modest tie deadzone
    (`eps ~ 0.05`), then the symmetry branch overtakes it clearly
  - this strongly suggests the next minimal mechanism should not be another
    training loss; it should be an explicit low-margin tie / deadzone rule at
    export or evaluation time

### Experiment: systematic post-hoc deadzone sweep on existing gated+cap(+sym) runs

- Objective:
  - stop changing training losses and test the actual remaining bottleneck:
    - how much strict evaluation improves once tiny residual margins are
      treated as ties
  - include GT safety diagnostics so `eps` is not chosen by F1 alone
- Runs compared:
  - baseline gated+cap:
    - `results/run_20260315_214810`
  - gated+cap+ungated-symmetry `lambda=0.20`:
    - `results/run_20260315_220438`
  - gated+cap+ungated-symmetry `lambda=0.50`:
    - `results/run_20260315_220842`
- Analysis script / artifact:
  - script:
    - `GraphExp/deadzone_sweep_existing_runs.py`
  - CSV:
    - `GraphExp/results/deadzone_sweep_existing_runs_20260316.csv`
- Sweep:
  - `eps in {1e-12, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1}`
- Reported diagnostics for each `eps`:
  - `strict_precision`, `strict_recall`, `strict_f1`, `strict_pred_count`
  - `tp_lost_by_deadzone`, `tp_lost_by_deadzone_frac`
  - `fp_removed_by_deadzone`, `fp_removed_by_deadzone_frac`
  - `all_gt_fragile_count`, `all_gt_fragile_frac`
- Result summary:
  - baseline gated+cap (`sym=0.0`)
    - raw strict (`eps=1e-12`):
      - `strict_f1 = 0.2796`
      - `precision = 0.1733`
      - `recall = 72.22%`
      - `pred_count = 75`
    - best zero-extra-TP-loss point on this grid:
      - `eps = 3e-4`
      - `strict_f1 = 0.2955`
      - `precision = 0.1857`
      - `recall = 72.22%`
      - `fp_removed = 5`
      - `tp_lost = 0`
    - best F1 point on this grid:
      - `eps = 1e-1`
      - `strict_f1 = 0.4444`
      - `precision = 0.3333`
      - `recall = 66.67%`
      - `fp_removed = 38`
      - `tp_lost = 1`
  - gated+cap+sym `lambda=0.20`
    - raw strict (`eps=1e-12`):
      - `strict_f1 = 0.2680`
      - `precision = 0.1646`
      - `recall = 72.22%`
      - `pred_count = 79`
    - best zero-extra-TP-loss point on this grid:
      - `eps = 3e-4`
      - `strict_f1 = 0.2796`
      - `precision = 0.1733`
      - `recall = 72.22%`
      - `fp_removed = 4`
      - `tp_lost = 0`
    - best F1 point on this grid:
      - `eps = 1e-1`
      - `strict_f1 = 0.4706`
      - `precision = 0.3636`
      - `recall = 66.67%`
      - `fp_removed = 45`
      - `tp_lost = 1`
  - gated+cap+sym `lambda=0.50`
    - raw strict (`eps=1e-12`):
      - `strict_f1 = 0.2653`
      - `precision = 0.1625`
      - `recall = 72.22%`
      - `pred_count = 80`
    - best zero-extra-TP-loss point on this grid:
      - `eps = 3e-4`
      - `strict_f1 = 0.2737`
      - `precision = 0.1688`
      - `recall = 72.22%`
      - `fp_removed = 3`
      - `tp_lost = 0`
    - best F1 point on this grid:
      - `eps = 1e-1`
      - `strict_f1 = 0.5106`
      - `precision = 0.4138`
      - `recall = 66.67%`
      - `fp_removed = 50`
      - `tp_lost = 1`
- GT safety readout:
  - all three runs have `18` GT edges total
  - already at raw strict, only `13/18` GT edges are recovered
    - the other `5` GT edges already have non-positive signed margin and are
      not caused by the deadzone
  - the first extra TP loss appears at `eps = 1e-3`
  - weakest currently recovered GT edge margins:
    - baseline gated+cap: `0.000714`
    - gated+cap+sym `0.20`: `0.000892`
    - gated+cap+sym `0.50`: `0.000621`
  - so if we require **zero additional TP loss relative to current raw strict**,
    the safe region is roughly:
    - `eps < 6e-4`
    - on the tested grid, `3e-4` is the best safe point
- Interpretation:
  - this confirms the training-side diagnosis:
    - GT and FP have already been separated enough that a deadzone materially
      improves strict precision/F1 without any retraining
  - the symmetry branch benefits the most once a deadzone is allowed
    - under raw strict it looks worse
    - under `eps = 0.05 ~ 0.1` it becomes the clear winner
  - the tradeoff is now explicit and simple:
    - conservative operating point:
      - `eps = 3e-4`
      - preserves current recall exactly
      - only modestly improves F1
    - F1-oriented operating point on current grid:
      - `eps = 1e-1`
      - removes a large block of residual hallucinated edges
      - costs exactly `1` currently recovered GT edge
- Practical conclusion:
  - the next experiment should not be more loss engineering
  - it should be one of:
    - formalize a deadzone in export/eval and compare branches under the chosen
      `eps`
    - or, if recall preservation is mandatory, do a denser sweep below
      `1e-3` to pick the best zero-extra-TP-loss threshold

### Experiment: 5-seed / 100-epoch formal with multi-eps strict evaluation

- Objective:
  - move `margin_eps` into the formal runner so the same trained model is
    evaluated at multiple strict deadzones without retraining
  - test the two most relevant training branches only:
    - baseline gated+cap
    - gated+cap+ungated-symmetry `lambda=0.50`
- Runner update:
  - `run_cross_pred_v1_final_only_compare.py` now supports:
    - `--strict_margin_eps_values`
    - `--ungated_symmetry_values`
    - `--ungated_symmetry_warmup_epochs`
    - `--ungated_symmetry_ramp_epochs`
  - for each run it now writes:
    - legacy primary `strict_*` metrics for the first `eps`
    - plus per-eps fields such as:
      - `strict_f1_eps_0`
      - `strict_f1_eps_0p0003`
      - `strict_f1_eps_0p1`
- Formal command:
  - `python GraphExp\run_cross_pred_v1_final_only_compare.py --csv_path fMRI_dataset\sim3.csv --gt_path fMRI_dataset\h3.txt --device cuda --epochs 100 --pretrain_epochs 50 --structure_init_mode patel_kappa --scales 0.05 --lambda_l1_values 0.02 --seeds 11,22,33,44,55 --cross_pred_conditions off --directional_conditions patel --directional_schedule plateau --directional_kappa_gate --directional_kappa_gate_quantile 0.5 --parent_cap_values 0.5 --parent_cap_targets 2.5 --parent_cap_warmup_epochs 10 --parent_cap_ramp_epochs 10 --ungated_symmetry_values 0.0,0.5 --ungated_symmetry_warmup_epochs 10 --ungated_symmetry_ramp_epochs 10 --strict_margin_eps_values 0,3e-4,0.1 --experiment_tag formal_deadzone_multi_eps`
- Artifacts:
  - run summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260316_090716_formal_deadzone_multi_eps.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260316_090716_formal_deadzone_multi_eps_aggregate.csv`
- Branch definitions:
  - baseline gated+cap:
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `ungated_symmetry_lambda=0.0`
  - symmetry branch:
    - same as above
    - plus `ungated_symmetry_lambda=0.50`
- Aggregate result summary:
  - baseline gated+cap (`sym=0.0`)
    - `eps = 0`
      - `strict_precision = 0.2006 +/- 0.0117`
      - `strict_recall = 66.67% +/- 4.97%`
      - `strict_f1 = 0.3079 +/- 0.0151`
      - `pred_count = 60.0 +/- 5.3`
    - `eps = 3e-4`
      - `strict_precision = 0.2068 +/- 0.0137`
      - `strict_recall = 66.67% +/- 4.97%`
      - `strict_f1 = 0.3153 +/- 0.0182`
      - `pred_count = 58.2 +/- 4.8`
    - `eps = 0.1`
      - `strict_precision = 0.2604 +/- 0.0260`
      - `strict_recall = 56.67% +/- 5.44%`
      - `strict_f1 = 0.3568 +/- 0.0346`
      - `pred_count = 39.2 +/- 1.3`
    - structural diagnostics:
      - `adj_eff_parents_mean = 3.17`
      - `margin_p90 = 0.9888`
      - `gt_signed_margin_median = 0.8037`
  - gated+cap+sym `lambda=0.50`
    - `eps = 0`
      - `strict_precision = 0.2527 +/- 0.0218`
      - `strict_recall = 62.22% +/- 2.22%`
      - `strict_f1 = 0.3590 +/- 0.0249`
      - `pred_count = 44.6 +/- 3.4`
    - `eps = 3e-4`
      - `strict_precision = 0.2807 +/- 0.0278`
      - `strict_recall = 62.22% +/- 2.22%`
      - `strict_f1 = 0.3864 +/- 0.0301`
      - `pred_count = 40.2 +/- 3.2`
    - `eps = 0.1`
      - `strict_precision = 0.4099 +/- 0.0650`
      - `strict_recall = 35.56% +/- 12.96%`
      - `strict_f1 = 0.3748 +/- 0.0877`
      - `pred_count = 15.4 +/- 3.9`
    - structural diagnostics:
      - `adj_eff_parents_mean = 2.18`
      - `margin_p90 = 0.1673`
      - `gt_signed_margin_median = 0.0595`
- Seed-level winner check:
  - at `eps = 0`:
    - `sym=0.50` beats `sym=0.0` on `strict_f1` in `5/5` seeds
  - at `eps = 3e-4`:
    - `sym=0.50` beats `sym=0.0` on `strict_f1` in `5/5` seeds
  - at `eps = 0.1`:
    - `sym=0.50` beats `sym=0.0` on `strict_f1` in only `2/5` seeds
- Interpretation / Thoughts:
  - formal confirms that adding a small deadzone is worthwhile:
    - both branches improve from `eps=0` to `eps=3e-4`
  - but it also changes the earlier smoke-era conclusion about the symmetry
    branch under a large deadzone:
    - on 100 epochs, `sym=0.50` does **not** preserve GT margins the way it did
      in the 30-epoch smoke
    - it compresses the whole margin distribution much harder
      - `margin_p90` drops from `0.9888` to `0.1673`
      - `gt_signed_margin_median` drops from `0.8037` to `0.0595`
  - as a result:
    - `sym=0.50` is clearly better at `eps=0` and `eps=3e-4`
    - but `eps=0.1` is now too aggressive for that branch, because it starts
      filtering true edges as well
- Practical conclusion:
  - if the project chooses a conservative deadzone near `3e-4`, the current
    best formal branch is:
    - gated+cap+ungated-symmetry `lambda=0.50`
  - if the project insists on a very aggressive deadzone near `0.1`, the
    current formal winner is instead:
    - gated+cap with `ungated_symmetry_lambda=0.0`
  - so the formal result points to:
    - adopt a small deadzone in formal evaluation/export
    - do **not** hard-commit to `eps=0.1` based on the earlier 30-epoch smoke

### Experiment: low-rank `emb_dim` and message-graph orientation smoke

- Objective:
  - test two architecture-level levers outside the previous loss sweep:
    - lower-rank structure parameterization via `emb_dim`
    - explicit GraphConv message orientation via `structure_message_graph_mode`
- Implementation:
  - `main_structure_learning.py` now exposes:
    - `--emb_dim`
    - `--structure_message_graph_mode {raw, causal}`
  - `run_cross_pred_v1_final_only_compare.py` now exposes:
    - `--emb_dims`
    - `--structure_message_graph_mode`
  - new epoch diagnostic in `quality_history.csv`:
    - `msg_dir_raw_diag_gap`
    - `msg_dir_causal_diag_gap`
    - `msg_dir_gap_delta_causal_minus_raw`
    - `msg_dir_prefers_causal`
- Smoke setup:
  - fixed branch:
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `ungated_symmetry_lambda=0.50`
  - dataset:
    - `sim3.csv`
  - seed:
    - `11`
  - epochs:
    - `30`
  - sweep:
    - `emb_dim in {0(full), 8, 4}`
    - `message_graph_mode in {raw, causal}`
  - strict reporting:
    - `eps in {0, 3e-4, 0.1}`
- Artifacts:
  - raw-mode summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260316_143016_embdim_msgmode_smoke30_raw.csv`
  - causal-mode summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260316_143650_embdim_msgmode_smoke30_causal.csv`
- Result summary:
  - `raw`, `emb_dim=0` (full rank)
    - `strict_f1 eps=0 = 0.2500`
    - `strict_f1 eps=3e-4 = 0.2549`
    - `strict_f1 eps=0.1 = 0.3333`
    - `gt_margin_median = 0.0815`
    - `eff_parents_mean = 2.83`
  - `raw`, `emb_dim=8`
    - `strict_f1 eps=0 = 0.2523`
    - `strict_f1 eps=3e-4 = 0.2718`
    - `strict_f1 eps=0.1 = 0.2917`
    - `gt_margin_median = 0.0661`
    - `eff_parents_mean = 2.83`
  - `raw`, `emb_dim=4`
    - `strict_f1 eps=0 = 0.2182`
    - `strict_f1 eps=3e-4 = 0.2286`
    - `strict_f1 eps=0.1 = 0.2667`
    - `gt_margin_median = 0.0718`
    - `eff_parents_mean = 3.48`
    - failure mode flagged as `wrong_direction_asymmetry`
  - `causal`, `emb_dim=0` (full rank)
    - `strict_f1 eps=0 = 0.2364`
    - `strict_f1 eps=3e-4 = 0.2500`
    - `strict_f1 eps=0.1 = 0.4167`
    - `gt_margin_median = 0.2347`
    - `eff_parents_mean = 2.84`
  - `causal`, `emb_dim=8`
    - `strict_f1 eps=0 = 0.2321`
    - `strict_f1 eps=3e-4 = 0.2524`
    - `strict_f1 eps=0.1 = 0.3729`
    - `gt_margin_median = 0.6539`
    - `eff_parents_mean = 3.49`
  - `causal`, `emb_dim=4`
    - `strict_f1 eps=0 = 0.2523`
    - `strict_f1 eps=3e-4 = 0.2523`
    - `strict_f1 eps=0.1 = 0.3077`
    - `gt_margin_median = 0.1179`
    - `eff_parents_mean = 3.53`
- Message-direction diagnostic:
  - `raw`, `emb_dim=0`
    - `raw_gap = -0.0076`
    - `causal_gap = 0.0186`
    - diagnostic prefers `causal`
  - `raw`, `emb_dim=8`
    - `raw_gap = -0.0089`
    - `causal_gap = 0.0158`
    - diagnostic prefers `causal`
  - `raw`, `emb_dim=4`
    - `raw_gap = 0.0201`
    - `causal_gap = -0.0108`
    - diagnostic prefers `raw`
  - `causal`, `emb_dim=0`
    - `raw_gap = -0.0094`
    - `causal_gap = 0.0236`
    - diagnostic prefers `causal`
  - `causal`, `emb_dim=8`
    - `raw_gap = -0.0130`
    - `causal_gap = 0.0203`
    - diagnostic prefers `causal`
  - `causal`, `emb_dim=4`
    - `raw_gap = -0.0098`
    - `causal_gap = 0.0078`
    - diagnostic prefers `causal`
- Interpretation / Thoughts:
  - low-rank does **not** automatically help
    - `emb_dim=4` is not promising in this smoke
    - it either underperforms directly or raises effective parents / direction instability
  - `emb_dim=8` is the only low-rank candidate worth carrying forward
    - under `raw` it produced the best `eps=3e-4` strict F1 in this 1-seed smoke
  - the new message-direction diagnostic is informative
    - for full rank and `emb_dim=8`, it consistently prefers `causal`
    - this supports the semantic audit that GraphConv should likely use the
      transposed causal orientation
  - but 1-seed headline strict is still mixed
    - `causal` is not yet a universal winner at `eps=0` / `eps=3e-4`
    - it helps much more at aggressive `eps=0.1`, largely by improving GT
      margins for some cells
- Practical conclusion:
  - drop `emb_dim=4` for now
  - carry forward the 2x2 pilot:
    - `emb_dim in {0, 8}`
    - `message_graph_mode in {raw, causal}`
  - run that as a multi-seed smoke before any new formal

### Experiment: 2x2 pilot on `emb_dim in {0,8}` and `message_graph_mode in {raw, causal}`

- Objective:
  - verify whether the promising 1-seed smoke signals survive a small
    multi-seed pilot
  - focus on:
    - whether `causal + emb_dim=8` keeps its GT-margin advantage
    - whether `eps = 3e-4` strict ranking converges
- Setup:
  - fixed branch:
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `ungated_symmetry_lambda=0.50`
  - dataset:
    - `sim3.csv`
  - seeds:
    - `11, 22, 33`
  - epochs:
    - `30`
  - sweep:
    - `emb_dim in {0(full), 8}`
    - `message_graph_mode in {raw, causal}`
  - strict reporting:
    - `eps in {0, 3e-4, 0.1}`
- Artifacts:
  - raw-mode summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_145305_embdim_msgmode_pilot3_raw.csv`
  - raw-mode aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_145305_embdim_msgmode_pilot3_raw_aggregate.csv`
  - causal-mode summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_150547_embdim_msgmode_pilot3_causal.csv`
  - causal-mode aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_150547_embdim_msgmode_pilot3_causal_aggregate.csv`
- Aggregate result summary:
  - `raw`, `emb_dim=0`
    - `strict_f1 eps=0 = 0.2621 +/- 0.0100`
    - `strict_f1 eps=3e-4 = 0.2696 +/- 0.0141`
    - `strict_f1 eps=0.1 = 0.3956 +/- 0.0567`
    - `gt_margin_median = 0.3330 +/- 0.2035`
    - `strict_recall eps=3e-4 = 74.07%`
    - `pred_count eps=3e-4 = 81.0`
  - `raw`, `emb_dim=8`
    - `strict_f1 eps=0 = 0.2526 +/- 0.0134`
    - `strict_f1 eps=3e-4 = 0.2688 +/- 0.0200`
    - `strict_f1 eps=0.1 = 0.3536 +/- 0.0456`
    - `gt_margin_median = 0.3628 +/- 0.3401`
    - `strict_recall eps=3e-4 = 75.93%`
    - `pred_count eps=3e-4 = 84.0`
  - `causal`, `emb_dim=0`
    - `strict_f1 eps=0 = 0.2442 +/- 0.0142`
    - `strict_f1 eps=3e-4 = 0.2525 +/- 0.0035`
    - `strict_f1 eps=0.1 = 0.4028 +/- 0.0196`
    - `gt_margin_median = 0.2507 +/- 0.0522`
    - `strict_recall eps=3e-4 = 72.22%`
    - `pred_count eps=3e-4 = 85.0`
  - `causal`, `emb_dim=8`
    - `strict_f1 eps=0 = 0.2442 +/- 0.0104`
    - `strict_f1 eps=3e-4 = 0.2551 +/- 0.0020`
    - `strict_f1 eps=0.1 = 0.3733 +/- 0.0111`
    - `gt_margin_median = 0.5237 +/- 0.2411`
    - `strict_recall eps=3e-4 = 70.37%`
    - `pred_count eps=3e-4 = 81.3`
- GT / FP margin post-hoc readout from final causal adjacency:
  - `raw`, `emb_dim=0`
    - TP margin median mean `0.7087`
    - FP margin median mean `0.0145`
    - FP count mean `72.7`
  - `raw`, `emb_dim=8`
    - TP margin median mean `0.6154`
    - FP margin median mean `0.0245`
    - FP count mean `76.7`
  - `causal`, `emb_dim=0`
    - TP margin median mean `0.7293`
    - FP margin median mean `0.0136`
    - FP count mean `78.0`
  - `causal`, `emb_dim=8`
    - TP margin median mean `0.6652`
    - FP margin median mean `0.0393`
    - FP count mean `75.7`
- Message-direction diagnostic:
  - across all `12/12` seed-config cells in this pilot:
    - `msg_dir_prefers_causal = 1`
  - i.e. the online proxy diagnostic consistently says that transposed
    causal message flow aligns better than raw message flow
- Per-seed `eps=3e-4` ranking:
  - seed `11`:
    - best = `raw + emb_dim=8` (`0.2718`)
  - seed `22`:
    - best = `raw + emb_dim=0` (`0.2653`)
  - seed `33`:
    - best = `raw + emb_dim=8` (`0.2917`)
  - none of the three seeds had a causal-mode winner at `eps=3e-4`
- Interpretation / Thoughts:
  - Claude's key observation partially holds:
    - `causal + emb_dim=8` does keep the strongest GT-margin median on average
      (`0.5237`)
  - but the pilot does **not** support the stronger follow-up claim that this
    should already win under the current `eps=3e-4` operating point
    - it does not
    - all three seeds still favor `raw` mode at `eps=3e-4`
  - the reason looks important:
    - `causal + emb_dim=8` is not creating a cleaner GT/FP separation
    - its FP median margin is actually the highest of the four pilot cells
      (`0.0393`)
    - so the extra GT-margin strength is not being bought selectively
  - this means:
    - "causal message flow is semantically better" and
    - "causal message flow is currently the best strict operating point"
    are not the same statement
  - right now the pilot supports the first statement much more strongly than
    the second
- Practical conclusion:
  - do **not** jump to a 5-seed formal on `causal + emb_dim=8`
  - keep `emb_dim=8` as a secondary candidate, but not as the new default
  - for the current `sim3`, `30-epoch`, `eps=3e-4` operating point, the best
    3-seed pilot result remains:
    - `raw + emb_dim=0` (very slightly ahead overall)
    - with `raw + emb_dim=8` competitive but not clearly better
  - the most defensible next architecture step is:
    - if we want to pursue `causal`, do it because the message-direction
      diagnostic is consistently in its favor, not because pilot strict F1 has
      already won
    - that means the next test should be a **targeted causal-only follow-up**
      (for example adjusting deadzone or training length), not a large formal
      sweep pretending the winner is already clear

### Experiment: causal-only `ungated_symmetry` follow-up (`sym in {0.5, 1.0, 2.0}`)

- Objective:
  - test Claude's narrower causal-only hypothesis directly:
    - keep `message_graph_mode=causal`
    - sweep stronger `ungated_symmetry_lambda`
    - check whether causal mode can reduce residual FP margins enough to become
      the best branch at the current `eps=3e-4` operating point
- Setup:
  - fixed branch:
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `structure_message_graph_mode=causal`
  - sweep:
    - `emb_dim in {0(full), 8}`
    - `ungated_symmetry_lambda in {0.5, 1.0, 2.0}`
  - dataset:
    - `sim3.csv`
  - seeds:
    - `11, 22, 33`
  - epochs:
    - `30`
  - strict reporting:
    - `eps in {0, 3e-4, 0.1}`
- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_154203_causal_sym_followup_pilot3.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_154203_causal_sym_followup_pilot3_aggregate.csv`
- Aggregate result summary:
  - `causal`, `emb_dim=0`, `sym=0.5`
    - `strict_f1 eps=0 = 0.2442 +/- 0.0142`
    - `strict_f1 eps=3e-4 = 0.2525 +/- 0.0035`
    - `strict_f1 eps=0.1 = 0.4028 +/- 0.0196`
    - `gt_margin_median = 0.2507 +/- 0.0522`
    - `strict_recall eps=3e-4 = 72.22%`
  - `causal`, `emb_dim=0`, `sym=1.0`
    - `strict_f1 eps=0 = 0.2541 +/- 0.0043`
    - `strict_f1 eps=3e-4 = 0.2636 +/- 0.0055`
    - `strict_f1 eps=0.1 = 0.3268 +/- 0.0245`
    - `gt_margin_median = 0.0416 +/- 0.0015`
    - `strict_recall eps=3e-4 = 72.22%`
  - `causal`, `emb_dim=0`, `sym=2.0`
    - `strict_f1 eps=0 = 0.2552 +/- 0.0184`
    - `strict_f1 eps=3e-4 = 0.2823 +/- 0.0123`
    - `strict_f1 eps=0.1 = 0.3235 +/- 0.0585`
    - `gt_margin_median = 0.0215 +/- 0.0033`
    - `strict_recall eps=3e-4 = 64.81%`
  - `causal`, `emb_dim=8`, `sym=0.5`
    - `strict_f1 eps=0 = 0.2442 +/- 0.0104`
    - `strict_f1 eps=3e-4 = 0.2551 +/- 0.0020`
    - `strict_f1 eps=0.1 = 0.3733 +/- 0.0111`
    - `gt_margin_median = 0.5237 +/- 0.2411`
    - `strict_recall eps=3e-4 = 70.37%`
  - `causal`, `emb_dim=8`, `sym=1.0`
    - `strict_f1 eps=0 = 0.2456 +/- 0.0071`
    - `strict_f1 eps=3e-4 = 0.2578 +/- 0.0128`
    - `strict_f1 eps=0.1 = 0.3887 +/- 0.0080`
    - `gt_margin_median = 0.2229 +/- 0.0563`
    - `strict_recall eps=3e-4 = 70.37%`
  - `causal`, `emb_dim=8`, `sym=2.0`
    - `strict_f1 eps=0 = 0.2625 +/- 0.0146`
    - `strict_f1 eps=3e-4 = 0.2660 +/- 0.0083`
    - `strict_f1 eps=0.1 = 0.3640 +/- 0.0844`
    - `gt_margin_median = 0.0437 +/- 0.0281`
    - `strict_recall eps=3e-4 = 68.52%`
- GT / FP margin post-hoc readout from final causal adjacency:
  - `causal`, `emb_dim=0`, `sym=0.5`
    - TP margin median mean `0.7293`
    - FP margin median mean `0.0136`
    - TP count mean `13.3`
    - FP count mean `78.0`
  - `causal`, `emb_dim=0`, `sym=1.0`
    - TP margin median mean `0.0703`
    - FP margin median mean `0.0054`
    - TP count mean `13.0`
    - FP count mean `71.3`
  - `causal`, `emb_dim=0`, `sym=2.0`
    - TP margin median mean `0.1533`
    - FP margin median mean `0.0033`
    - TP count mean `11.7`
    - FP count mean `62.0`
  - `causal`, `emb_dim=8`, `sym=0.5`
    - TP margin median mean `0.6652`
    - FP margin median mean `0.0393`
    - TP count mean `13.0`
    - FP count mean `75.7`
  - `causal`, `emb_dim=8`, `sym=1.0`
    - TP margin median mean `0.6495`
    - FP margin median mean `0.0209`
    - TP count mean `12.7`
    - FP count mean `72.7`
  - `causal`, `emb_dim=8`, `sym=2.0`
    - TP margin median mean `0.1838`
    - FP margin median mean `0.0095`
    - TP count mean `12.7`
    - FP count mean `66.3`
- Interpretation / Thoughts:
  - stronger `sym` **does** help causal mode numerically at `eps=3e-4`
    - the best cell becomes `causal + emb_dim=0 + sym=2.0`
    - its `strict_f1 eps=3e-4 = 0.2823`
    - that is above the earlier causal cells and slightly above the earlier
      raw 3-seed pilot headline (`0.2696`)
  - but the way it improves is not the clean win we would want before a formal
    promotion
    - `sym=2.0` sharply compresses GT margins:
      - `gt_margin_median: 0.2507 -> 0.0215` for `emb_dim=0`
      - TP median margin also drops from `0.7293 -> 0.1533`
    - at the same time recall falls:
      - `72.22% -> 64.81%` for `emb_dim=0`
    - so the gain is not "better selective separation"
    - it is closer to "make many edges near-tie so the deadzone can prune them"
  - the FP story is therefore mixed:
    - yes, stronger `sym` lowers FP median margin substantially
    - but it lowers GT margins too, and not by a small amount
  - given the earlier 100-epoch experience where symmetry-heavy branches could
    collapse GT margins over longer training, this 30-epoch improvement is not
    enough evidence to justify a direct formal jump
- Practical conclusion:
  - do **not** promote `causal + stronger sym` to a 100-epoch formal yet
  - treat this follow-up as evidence that causal mode can be made competitive
    under a deadzone, but not yet robust
  - the remaining gap is no longer best attacked by more `causal/sym` loss
    tuning
  - the more important untested direction is now **data utilization during
    optimization**, not another loss sweep

### Next Experiment Design: true multi-subject shared-graph batch updates

- Rationale from code inspection:
  - current training exposes `batch_size`, but the optimizer still does:
    - `optimizer.zero_grad()`
    - forward/backward on **one subject**
    - `optimizer.step()`
  - so "multiple subjects per batch" is currently only a data-ordering device
  - it is **not** yet true multi-subject gradient averaging on the shared graph
  - this means the architecture/data-use idea
    - "treat subjects as a batch so the shared structure gets more stable
      gradients"
    has not actually been tested yet
- Why this is the right next step:
  - it directly targets a real implementation gap, not a hypothetical one
  - it preserves the current best raw branch instead of opening a new loss
    dimension
  - it is cheaper and cleaner than jumping straight to:
    - learned `noise_guide_adj`
    - new pretraining objectives
    - larger architectural rewrites
  - it also addresses one of the clearest current risks:
    - all conclusions are still mainly on `sim3`
    - before bigger architecture work, we should stabilize how the shared graph
      uses subject evidence
- Proposed implementation knob:
  - add an optimizer-step mode switch, e.g.
    - `subject`:
      - current behavior, one optimizer step per subject
    - `batch_mean`:
      - accumulate losses across the subject minibatch
      - divide by batch size
      - do one optimizer step per subject batch
- Smoke experiment:
  - fixed branch:
    - `structure_message_graph_mode=raw`
    - `emb_dim=0`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `ungated_symmetry_lambda=0.50`
  - dataset:
    - `sim3.csv`
  - seeds:
    - `11, 22, 33`
  - epochs:
    - `30`
  - strict reporting:
    - `eps in {0, 3e-4, 0.1}`
  - sweep:
    - `optimizer_step_mode in {subject, batch_mean}`
  - optional follow-up only if `batch_mean` underfits:
    - small LR correction for `batch_mean`, e.g. `lr in {1e-3, 2e-3}`
- Promotion criteria:
  - primary:
    - improve `strict_f1 eps=3e-4` over the current raw pilot baseline
  - safety:
    - no more than about `3` percentage points recall drop at `eps=3e-4`
    - `gt_margin_median` should not collapse toward the `~0.02` regime seen in
      the aggressive symmetry cells
  - secondary:
    - lower FP median margin and/or lower FP count without sacrificing TP count
- If the smoke is positive:
  - run a small scaling check on `sim4`
    - first `1 seed x 30 epochs`
    - then a multi-seed formal if the signal survives
- If the smoke is negative:
  - drop this direction quickly
  - the next architecture candidate should then be a scheduled
    learned-structure `noise_guide_adj` experiment, not more `causal/sym`
    tuning

### Experiment: `optimizer_step_mode` smoke (`subject` vs `batch_mean`)

- Objective:
  - test whether true multi-subject gradient averaging on the shared graph
    improves the current raw mainline
- Setup:
  - fixed branch:
    - `structure_message_graph_mode=raw`
    - `emb_dim=0`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `ungated_symmetry_lambda=0.50`
  - sweep:
    - `optimizer_step_mode in {subject, batch_mean}`
  - dataset:
    - `sim3.csv`
  - seeds:
    - `11, 22, 33`
  - epochs:
    - `30`
  - strict reporting:
    - `eps in {0, 3e-4, 0.1}`
- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_182002_optimizer_step_mode_pilot3.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_182002_optimizer_step_mode_pilot3_aggregate.csv`
- Aggregate result summary:
  - `subject`:
    - `strict_f1 eps=0 = 0.2621 +/- 0.0100`
    - `strict_f1 eps=3e-4 = 0.2696 +/- 0.0141`
    - `strict_f1 eps=0.1 = 0.3956 +/- 0.0567`
    - `strict_recall eps=3e-4 = 74.07%`
    - `pred_count eps=3e-4 = 81.0`
    - `gt_margin_median = 0.3330`
    - `eff_parents_mean = 2.92`
  - `batch_mean`:
    - `strict_f1 eps=0 = 0.2114 +/- 0.0000`
    - `strict_f1 eps=3e-4 = 0.2149 +/- 0.0015`
    - `strict_f1 eps=0.1 = 0.3181 +/- 0.0096`
    - `strict_recall eps=3e-4 = 72.22%`
    - `pred_count eps=3e-4 = 103.0`
    - `gt_margin_median = 0.5145`
    - `eff_parents_mean = 6.19`
- Interpretation / Thoughts:
  - naive `batch_mean` is clearly worse under the same epoch count
    - more predicted edges
    - much weaker precision
    - effective parents explode from `2.92` to `6.19`
  - but this comparison is confounded:
    - with `batch_size=4`, `batch_mean` takes about `1/4` as many optimizer
      steps per epoch as `subject`
    - its warmup/ramp schedules are also effectively `1/4` as long in update
      count
  - so this first smoke is enough to reject the naive "same epochs, same
    schedule" version, but not enough to reject the underlying idea

### Experiment: fairer `batch_mean` step-scaled follow-up

- Objective:
  - remove the obvious optimizer-step confound from the first smoke
  - approximately match total optimizer updates by scaling:
    - `epochs: 30 -> 120`
    - `parent_cap warmup/ramp: 10/10 -> 40/40`
    - `ungated_symmetry warmup/ramp: 10/10 -> 40/40`
- Artifacts:
  - 1-seed check:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260316_184247_optimizer_step_mode_batchmean_stepscaled_seed22.csv`
  - 3-seed summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_185738_optimizer_step_mode_batchmean_stepscaled_pilot3.csv`
  - 3-seed aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_185738_optimizer_step_mode_batchmean_stepscaled_pilot3_aggregate.csv`
- Aggregate result summary (`batch_mean`, step-scaled):
  - `strict_f1 eps=0 = 0.2331 +/- 0.0050`
  - `strict_f1 eps=3e-4 = 0.2443 +/- 0.0051`
  - `strict_f1 eps=0.1 = 0.4566 +/- 0.0402`
  - `strict_recall eps=3e-4 = 70.37%`
  - `pred_count eps=3e-4 = 85.7`
  - `gt_margin_median = 0.9752`
  - `eff_parents_mean = 3.12`
- GT / FP margin post-hoc readout:
  - `subject`, 30 epochs
    - TP margin median mean `0.7087`
    - FP margin median mean `0.0145`
    - TP count mean `13.7`
    - FP count mean `72.7`
  - `batch_mean`, step-scaled
    - TP margin median mean `0.9753`
    - FP margin median mean `0.0157`
    - TP count mean `12.7`
    - FP count mean `78.0`
- Interpretation / Thoughts:
  - the confound was real
    - after step-scaling, `batch_mean` no longer collapses into the dense
      `eff_par ~6.2` regime
    - it recovers to `eff_par ~3.1`, close to the subject baseline
  - but even after that correction, it still does **not** beat the current raw
    mainline at the operating point we actually care about
    - `strict_f1 eps=3e-4` stays below `subject`:
      - `0.2443` vs `0.2696`
    - recall is also lower:
      - `70.37%` vs `74.07%`
  - what it really changes is the margin distribution:
    - TP margins become extremely strong (`gt_margin_median ~0.98`)
    - FP median margins do **not** improve
      - `0.0157` vs `0.0145`
    - FP count is actually higher
      - `78.0` vs `72.7`
  - so the mechanism is not "cleaner true/false separation"
    - it is closer to "push true edges harder without reducing false edges"
    - under current strict evaluation, that is not the right trade
- Practical conclusion:
  - do **not** promote `optimizer_step_mode=batch_mean` into the new default
  - the stronger conclusion is:
    - true multi-subject averaging is not useless
      - it can train a sharp graph once total optimizer steps are matched
    - but it does not solve the current bottleneck
      - FP existence remains the problem
  - this means the current architecture/data-use follow-up should move on
  - the most defensible next candidate is now:
    - a scheduled / partial learned-structure `noise_guide_adj` experiment
    - not more `optimizer_step_mode` tuning

### Diagnostic: in-training `noise-guide probe` (`patel` vs `blend50` vs `learned(detach)`)

- Objective:
  - test Claude's proposed mechanism before any risky diffusion-process change:
    - does replacing the fixed Patel noise guide with the current learned
      adjacency actually reduce denoising loss on a fixed probe?
  - diagnostic design:
    - same `probe_x`
    - same `t=500`
    - same fixed Gaussian `eps`
    - compare probe denoising loss under:
      - current `patel` noise guide
      - `blend50 = 0.5 * patel + 0.5 * learned(detach)`
      - `learned(detach)` noise guide
  - implementation note:
    - this is logging-only
    - it does **not** change training gradients
- Code changes:
  - `GraphExp/models/DDM.py`
    - `build_noise()` and `sample_q()` now accept optional noise-guide override
  - `GraphExp/main_structure_learning.py`
    - added `compute_noise_guide_probe_diagnostics()`
    - logs probe metrics into `quality_history.csv`
- 1-seed smoke artifact:
  - `GraphExp/results/run_20260316_210746/quality_history.csv`
- 1-seed readout (`seed=22`, current best raw branch):
  - visible log checkpoints:
    - epoch `10`
      - `patel=1.7475`
      - `blend50=1.7402` (`-0.0073`)
      - `learned=1.7501` (`+0.0026`)
    - epoch `20`
      - `patel=1.7367`
      - `blend50=1.7317` (`-0.0050`)
      - `learned=1.7362` (`-0.0005`)
    - epoch `30`
      - `patel=1.7661`
      - `blend50=1.7738` (`+0.0077`)
      - `learned=1.7689` (`+0.0028`)
  - full-epoch summary from `quality_history.csv`:
    - after warmup (`epoch >= 11`)
      - `blend50` mean delta vs `patel` = `-9.30e-4`
      - `learned` mean delta vs `patel` = `-8.94e-4`
      - both were better than `patel` on a majority of epochs
    - best-score epoch `22`
      - `patel=1.744239`
      - `blend50=1.740132` (`-0.004107`)
      - `learned=1.740404` (`-0.003835`)
  - tentative interpretation from 1 seed:
    - there is a real but **small** coupling signal
    - learned-guided noise can slightly reduce denoising loss
    - but the effect size is tiny (roughly `0.2%` scale)
- 3-seed confirmation run:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_211721_noise_probe_pilot3.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_211721_noise_probe_pilot3_aggregate.csv`
- 3-seed probe summary from each run's `quality_history.csv`:
  - after warmup (`epoch >= 11`)
    - `blend50` mean delta vs `patel`
      - aggregate mean `-1.38e-4`
      - better-epoch fraction `55%`
    - `learned` mean delta vs `patel`
      - aggregate mean `+1.62e-4`
      - better-epoch fraction `45%`
  - at each seed's best-score epoch:
    - average `blend50` delta = `+0.00464`
    - average `learned` delta = `+0.00310`
    - i.e. at the epochs we would most want to export, the probe is not
      reliably better than Patel
  - final-epoch deltas are slightly negative on average, but still extremely
    small:
    - `blend50`: `-0.00164`
    - `learned`: `-0.00043`
- Interpretation / Thoughts:
  - Claude's proposed diagnostic was worth doing
    - it exposed a subtle but important point:
      - the mechanism is **not** clearly zero
      - but it is also **not** strong or stable enough to justify a risky
        diffusion-process rewrite on its own
  - the 1-seed looked mildly promising
  - the 3-seed view is much more cautious:
    - any denoising-loss advantage is tiny
    - seed-to-seed sign flips happen
    - the advantage is not concentrated at best-score epochs
  - this is exactly the pattern of a weak secondary effect, not a clean new
    lever
- Practical conclusion:
  - do **not** jump straight to a learned-noise-guide training branch yet
  - if we still want to exhaust this architecture idea, the only defensible
    version would be:
    - a very conservative `blend50` / scheduled-blend smoke
    - explicitly framed as a weak-signal test, not a promoted mainline
  - otherwise, the stronger conclusion is:
    - the learned-noise-guide mechanism does **not** currently have enough
      evidence to outrank other future architecture changes

### Experiment: current-best raw branch scaling smoke on `sim4`

- Objective:
  - check whether the current loss-side best branch scales at all to the 50-node
    setting before spending budget on more local tuning
- Setup:
  - dataset:
    - `sim4.csv`
  - branch:
    - `structure_message_graph_mode=raw`
    - `emb_dim=0`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `ungated_symmetry_lambda=0.50`
    - conservative eval deadzones `eps in {0, 3e-4, 0.1}`
  - seeds:
    - `11`
  - epochs:
    - `30`
- Artifact:
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260316_214801_sim4_current_best_smoke1_aggregate.csv`
- Result summary:
  - `strict_f1 eps=0 = 0.1801`
  - `strict_f1 eps=3e-4 = 0.1865`
  - `strict_f1 eps=0.1 = 0.2044`
  - `pred_count eps=3e-4 = 250`
  - `gt_margin_median = 0.0000`
  - `eff_parents_mean = 5.25`
- Interpretation / Thoughts:
  - the current loss-side best branch does **not** scale cleanly to `sim4`
  - margins on GT edges collapse back to zero, while predicted edge count stays
    extremely high
  - this is exactly the pattern expected if the underlying adjacency
    parameterization still lets many candidate parents coexist without enough
    competition
- Practical conclusion:
  - this smoke materially strengthens the "sigmoid coexistence" diagnosis
  - it does **not** justify more loss tuning on top of the unchanged sigmoid
    parameterization

### Plumbing: competitive adjacency activations

- Objective:
  - add a parameterization-level alternative to independent sigmoid edges so
    candidate parents for the same target must compete
- Code changes:
  - `GraphExp/models/DDM.py`
    - added `adj_activation in {sigmoid, sparsemax, entmax15}`
    - `get_structure_adj()` now applies:
      - `sigmoid` as before
      - `sparsemax` / `entmax15` row-wise on `A_raw[effect, cause]`
    - this means parents compete within each target/effect row, which is the
      intended structural constraint
  - `GraphExp/main_structure_learning.py`
    - added CLI flag:
      - `--adj_activation`
    - plumbed activation choice into `ddm_kwargs` and saved config
  - `GraphExp/run_cross_pred_v1_final_only_compare.py`
    - added sweep support:
      - `--adj_activations`
    - added aggregation/grouping/reporting fields for `adj_activation`
- Verification:
  - `python -m py_compile GraphExp\\run_cross_pred_v1_final_only_compare.py GraphExp\\main_structure_learning.py GraphExp\\models\\DDM.py`
  - `python GraphExp\\run_cross_pred_v1_final_only_compare.py --help`
  - both passed after runner repair

### Experiment: `adj_activation` pilot on `sim3` (`sigmoid` vs `sparsemax`)

- Objective:
  - test whether parent competition fixes the main bottleneck more directly than
    another loss-side patch
- Setup:
  - fixed branch:
    - `structure_message_graph_mode=raw`
    - `emb_dim=0`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `parent_cap_lambda=0.50`
    - `parent_cap_target=2.5`
    - `ungated_symmetry_lambda=0.50`
    - `lambda_l1=0.02`
  - sweep:
    - `adj_activation in {sigmoid, sparsemax}`
  - dataset:
    - `sim3.csv`
  - seeds:
    - smoke `11`
    - pilot `11, 22, 33`
  - epochs:
    - `30`
  - strict reporting:
    - `eps in {0, 3e-4, 0.1}`
- Artifacts:
  - 1-seed smoke:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260316_221829_adj_activation_smoke1_aggregate.csv`
  - 3-seed pilot:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_222325_adj_activation_pilot3_aggregate.csv`
- 3-seed aggregate result summary:
  - `sigmoid`:
    - `strict_f1 eps=0 = 0.2545 +/- 0.0165`
    - `strict_f1 eps=3e-4 = 0.2663 +/- 0.0170`
    - `strict_f1 eps=0.1 = 0.3889 +/- 0.0167`
    - `strict_pred_count eps=3e-4 = 82.3`
    - `strict_recall eps=3e-4 = 74.07%`
    - `eff_parents_mean = 2.88`
    - `gt_margin_median = 0.2887`
  - `sparsemax`:
    - `strict_f1 eps=0 = 0.4843 +/- 0.0057`
    - `strict_f1 eps=3e-4 = 0.4919 +/- 0.0148`
    - `strict_f1 eps=0.1 = 0.5185 +/- 0.0262`
    - `strict_pred_count eps=3e-4 = 24.0`
    - `strict_recall eps=3e-4 = 57.41%`
    - `eff_parents_mean = 1.40`
    - `gt_margin_median = 0.3307`
- Interpretation / Thoughts:
  - this is the first architecture-side change that directly moves the actual
    bottleneck:
    - predicted edge count collapses from `~82` to `24`
    - effective parents collapse from `2.88` to `1.40`
    - strict F1 improves massively at every reported deadzone
  - the margin distribution changes in the expected sparse way:
    - overall median margin becomes exact `0`
    - `eps=0` and `eps=3e-4` become almost identical
    - this is consistent with sparsemax producing many exact zeros instead of a
      large cloud of weak-but-positive edges
  - the gain is not coming from better recall
    - recall actually drops
    - precision and FP count improve much more, which is exactly what we wanted
      from a competitive parent parameterization
- Practical conclusion:
  - the "sigmoid coexistence" diagnosis is strongly supported
  - `sparsemax` is immediately more promising than any of the recent
    `causal/batch_mean/noise-guide` architecture follow-ups

### Experiment: `sparsemax` cap-ablation follow-up on `sim3`

- Objective:
  - check whether a competitive parent parameterization already supplies most of
    the sparsity pressure, making the old `parent_cap` unnecessary or harmful
- Setup:
  - fixed branch:
    - `adj_activation=sparsemax`
    - `structure_message_graph_mode=raw`
    - `emb_dim=0`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `ungated_symmetry_lambda=0.50`
    - `lambda_l1=0.02`
  - sweep:
    - `parent_cap_lambda in {0.0, 0.5}`
    - `parent_cap_target=2.5` for the capped branch
  - dataset:
    - `sim3.csv`
  - seeds:
    - smoke `11`
    - pilot `11, 22, 33`
  - epochs:
    - `30`
- Artifacts:
  - 1-seed smoke:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260316_223654_sparsemax_cap_ablation_smoke1_aggregate.csv`
  - 3-seed pilot:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_224144_sparsemax_cap_ablation_pilot3_aggregate.csv`
  - 100-epoch stability smoke (`seed=11`, no-cap):
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260316_225955_sparsemax_nocap_100e_smoke1_aggregate.csv`
- 3-seed aggregate result summary:
  - `sparsemax + no-cap`:
    - `strict_f1 eps=0 = 0.5342 +/- 0.0089`
    - `strict_f1 eps=3e-4 = 0.5342 +/- 0.0089`
    - `strict_f1 eps=0.1 = 0.5382 +/- 0.0102`
    - `strict_pred_count eps=3e-4 = 25.7`
    - `strict_recall eps=3e-4 = 64.81%`
    - `strict_precision eps=3e-4 = 45.48%`
    - `eff_parents_mean = 1.88`
  - `sparsemax + cap(0.5@2.5)`:
    - `strict_f1 eps=0 = 0.4843 +/- 0.0057`
    - `strict_f1 eps=3e-4 = 0.4919 +/- 0.0148`
    - `strict_f1 eps=0.1 = 0.5185 +/- 0.0262`
    - `strict_pred_count eps=3e-4 = 24.0`
    - `strict_recall eps=3e-4 = 57.41%`
    - `strict_precision eps=3e-4 = 43.05%`
    - `eff_parents_mean = 1.40`
- 100-epoch single-seed check (`sparsemax + no-cap`):
  - `strict_f1 eps=0 = 0.5581`
  - `strict_f1 eps=3e-4 = 0.5581`
  - `strict_f1 eps=0.1 = 0.5581`
  - `strict_pred_count = 25`
  - `eff_parents_mean = 1.80`
- Interpretation / Thoughts:
  - once parent competition is built into the adjacency parameterization, the old
    hinge cap no longer looks essential
  - `cap` still pushes the graph even sparser (`1.88 -> 1.40` effective parents),
    but that extra pressure slightly hurts the operating point we care about
    most
  - importantly, the no-cap branch did **not** explode back into a dense graph
    even at `100` epochs
- Practical conclusion:
  - the current leading candidate is now:
    - `adj_activation=sparsemax`
    - `parent_cap_lambda=0.0`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `ungated_symmetry_lambda=0.50`
  - the most defensible next experiment is a `5-seed x 100-epoch` formal on
    this no-cap sparsemax branch

### Experiment: `sparsemax + no-cap` formal (`5 seeds x 100 epochs`)

- Objective:
  - verify that the strong `30`-epoch pilot survives long training and seed
    averaging
- Setup:
  - branch:
    - `adj_activation=sparsemax`
    - `parent_cap_lambda=0.0`
    - `structure_message_graph_mode=raw`
    - `emb_dim=0`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `ungated_symmetry_lambda=0.50`
    - `lambda_l1=0.02`
  - dataset:
    - `sim3.csv`
  - seeds:
    - `11, 22, 33, 44, 55`
  - epochs:
    - `100`
  - strict reporting:
    - `eps in {0, 3e-4, 0.1}`
- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260316_231015_sparsemax_nocap_formal5.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260316_231015_sparsemax_nocap_formal5_aggregate.csv`
- Aggregate result summary:
  - `strict_f1 eps=0 = 0.5376 +/- 0.0248`
  - `strict_f1 eps=3e-4 = 0.5376 +/- 0.0248`
  - `strict_f1 eps=0.1 = 0.5423 +/- 0.0217`
  - `strict_pred_count eps=3e-4 = 25.2`
  - `strict_precision eps=3e-4 = 46.21%`
  - `strict_recall eps=3e-4 = 64.44%`
  - `eff_parents_mean = 1.84`
  - `gt_margin_median = 0.1886`
  - `final_diff_loss = 1.1937 +/- 0.0156`
- Comparison against the previous sigmoid formal winner:
  - previous best reported branch:
    - `sigmoid + cap(0.5@2.5) + kappa_gate + ungated_sym=0.5`
    - `strict_f1 eps=3e-4 = 0.3864 +/- 0.0301`
  - new branch:
    - `sparsemax + no-cap + kappa_gate + ungated_sym=0.5`
    - `strict_f1 eps=3e-4 = 0.5376 +/- 0.0248`
  - absolute gain:
    - `+0.1512`
- Interpretation / Thoughts:
  - the competitive adjacency parameterization survives formal training cleanly
  - the improvement is not being propped up by deadzone tuning:
    - `eps=0` and `eps=3e-4` are identical
    - `eps=0.1` adds only a very small extra lift
  - this is strong evidence that the core issue really was parameterization-level
    parent coexistence, not mainly the lack of one more auxiliary loss
- Updated strategic conclusion:
  - within the current training framework, the most important improvement so far
    is **not** another loss term
  - it is replacing independent sigmoid edges with a competitive parent
    parameterization
  - the next architecture follow-up should therefore be modest and local:
    - optional `entmax15` comparison
    - `sim4` smoke under `sparsemax + no-cap`
  - not a return to more sigmoid-side loss stacking

### Follow-up: `sim4` with `sparsemax`

- Objective:
  - test whether the strong `sim3` sparsemax result transfers to the 50-node
    regime, and if not, identify which knob becomes the new bottleneck

#### Smoke: direct transfer of `sim3` sparsemax winner to `sim4`

- Setup:
  - branch:
    - `adj_activation=sparsemax`
    - `parent_cap_lambda=0.0`
    - `structure_message_graph_mode=raw`
    - `emb_dim=0`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `ungated_symmetry_lambda=0.50`
    - `lambda_l1=0.02`
    - `structure_init_scale=0.05`
  - dataset:
    - `sim4.csv`
  - seeds:
    - `11`
  - epochs:
    - `30`
- Artifact:
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260317_081503_sim4_sparsemax_nocap_smoke1_aggregate.csv`
- Result summary:
  - `strict_f1 eps=0 = 0.1149`
  - `strict_f1 eps=3e-4 = 0.1336`
  - `strict_f1 eps=0.1 = 0.2396`
  - `strict_pred_count eps=3e-4 = 373`
  - `gt_margin_median = 0.0000`
  - `eff_parents_mean = 8.04`
- Interpretation / Thoughts:
  - direct transfer fails
  - compared with the old sigmoid `sim4` baseline, sparsemax at the inherited
    `scale=0.05` is actually worse
  - the likely reason is structural:
    - under sparsemax, the scalar `adj_bias` no longer controls sparsity
    - the effective sharpness knob becomes logit scale / temperature
    - with `scale=0.05`, the 50-node logits are too flat, so sparsemax still
      keeps a very wide support

#### Scale sweep: `sim4 sparsemax` is highly sensitive to `structure_init_scale`

- Setup:
  - same branch as above, but sweep:
    - `structure_init_scale in {0.05, 0.1, 0.2, 0.5}`
  - seeds:
    - `11`
  - epochs:
    - `30`
- Artifact:
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260317_082129_sim4_sparsemax_scale_sweep_smoke1_aggregate.csv`
- Result summary:
  - `scale=0.05`
    - `strict_f1 eps=3e-4 = 0.1429`
    - `eff_parents_mean = 8.87`
    - failure mode `mixed_or_partial`
  - `scale=0.1`
    - `strict_f1 eps=3e-4 = 0.2599`
    - `eff_parents_mean = 2.47`
    - failure mode `weak_asymmetry`
  - `scale=0.2`
    - `strict_f1 eps=3e-4 = 0.2981`
    - `eff_parents_mean = 1.51`
    - failure mode `symmetric_collapse`
  - `scale=0.5`
    - `strict_f1 eps=3e-4 = 0.3184`
    - `eff_parents_mean = 1.77`
    - failure mode `weak_asymmetry`
- Interpretation / Thoughts:
  - this confirms the expected sparsemax behavior:
    - `structure_init_scale` is now the real sparsity / support-width knob
  - once scale is increased, sparsemax immediately beats the old sigmoid
    `sim4` baseline (`0.1865`) on strict F1
  - but the bottleneck changes:
    - existence gets much better
    - direction margins become weak and can collapse if the scale is pushed too
      aggressively

#### 3-seed pilots: best `sim4 sparsemax` branch beats sigmoid, but direction is now the bottleneck

- Artifacts:
  - `scale=0.5, ungated_sym=0.5`
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260317_084617_sim4_sparsemax_scale0p5_pilot3_aggregate.csv`
  - `scale=0.1, ungated_sym=0.5`
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260317_090627_sim4_sparsemax_scale0p1_pilot3_aggregate.csv`
  - `scale=0.5, ungated_sym=0.0`
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260317_093526_sim4_sparsemax_scale0p5_sym0_pilot3_aggregate.csv`
- Aggregate result summary:
  - `scale=0.5, ungated_sym=0.5`
    - `strict_f1 eps=0 = 0.2971 +/- 0.0113`
    - `strict_f1 eps=3e-4 = 0.2982 +/- 0.0113`
    - `strict_f1 eps=0.1 = 0.3319 +/- 0.0314`
    - `strict_pred_count eps=3e-4 = 112.7`
    - `eff_parents_mean = 1.52`
    - failure:
      - `2/3 symmetric_collapse`
      - `1/3 weak_asymmetry`
  - `scale=0.1, ungated_sym=0.5`
    - `strict_f1 eps=0 = 0.2786 +/- 0.0188`
    - `strict_f1 eps=3e-4 = 0.2794 +/- 0.0188`
    - `strict_f1 eps=0.1 = 0.3282 +/- 0.0169`
    - `strict_pred_count eps=3e-4 = 164.0`
    - `eff_parents_mean = 2.20`
    - failure:
      - `2/3 weak_asymmetry`
      - `1/3 mixed_or_partial`
  - `scale=0.5, ungated_sym=0.0`
    - `strict_f1 eps=0 = 0.3086 +/- 0.0097`
    - `strict_f1 eps=3e-4 = 0.3097 +/- 0.0101`
    - `strict_f1 eps=0.1 = 0.3524 +/- 0.0462`
    - `strict_pred_count eps=3e-4 = 122.3`
    - `eff_parents_mean = 1.58`
    - failure:
      - `2/3 symmetric_collapse`
      - `1/3 weak_asymmetry`
- Interpretation / Thoughts:
  - the best `3`-seed `sim4` sparsemax branch so far is:
    - `scale=0.5`
    - `ungated_symmetry_lambda=0.0`
    - `strict_f1 eps=3e-4 = 0.3097 +/- 0.0101`
  - that is a large lift over the old sigmoid `sim4` smoke:
    - `0.3097` vs `0.1865`
  - but the qualitative failure mode is now very different from the old
    sigmoid story:
    - edge existence is much less of a problem
    - directional asymmetry is too weak, often bordering on collapse
  - removing `ungated_symmetry` helps a bit
    - which is exactly what we would expect if symmetry regularization became
      redundant or too strong once sparsemax already controls false edge
      existence

#### One-seed causal-message check on the current best `sim4 sparsemax` candidate

- Objective:
  - test whether the remaining `sim4` weakness is primarily a message-direction
    issue once edge existence is under control
- Setup:
  - branch:
    - `adj_activation=sparsemax`
    - `parent_cap_lambda=0.0`
    - `ungated_symmetry_lambda=0.0`
    - `structure_init_scale=0.5`
  - compare:
    - `structure_message_graph_mode=causal`
  - dataset:
    - `sim4.csv`
  - seed:
    - `11`
  - epochs:
    - `30`
- Artifact:
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260317_094953_sim4_sparsemax_scale0p5_sym0_causal_smoke1_aggregate.csv`
- Result summary:
  - `strict_f1 eps=0 = 0.2578`
  - `strict_f1 eps=3e-4 = 0.2578`
  - `strict_f1 eps=0.1 = 0.3111`
  - `eff_parents_mean = 2.70`
  - `gt_margin_median = 0.0262`
  - `margin_p90 = 0.0565`
- Interpretation / Thoughts:
  - causal message flow does raise the margin scale relative to the raw
    `scale=0.5, sym=0` seed-11 run
  - but it does so by making the graph less sparse and hurting strict precision,
    so the overall strict F1 gets worse
  - on current evidence, this is not the right `sim4` follow-up to promote

### Updated `sim4` conclusion after sparsemax follow-up

- Sparsemax **does** matter on `sim4`, but only after retuning scale.
- The main diagnosis still holds:
  - parameterization-level parent competition is valuable
  - the old sigmoid branch was bottlenecked by FP existence
- But after that bottleneck is reduced on `sim4`, the active problem changes:
  - now the model tends to under-separate directions
  - i.e. `weak_asymmetry / symmetric_collapse`, not pair hallucination, is the
    dominant failure mode
- Practical implication:
  - do not treat `sim4` as "sparsemax failed"
  - the more accurate conclusion is:
    - sparsemax fixes the first problem
    - `sim4` then exposes the next problem
- Most defensible current `sim4` branch:
  - `adj_activation=sparsemax`
  - `parent_cap_lambda=0.0`
  - `ungated_symmetry_lambda=0.0`
  - `structure_init_scale=0.5`
  - `strict_f1 eps=3e-4 ≈ 0.31` in `3`-seed pilot
- Most defensible next step from here:
  - if staying on `sim4`, focus on restoring directional asymmetry without
    reopening the dense-edge problem
  - that points more toward a direction-strength or margin-separation follow-up
    than another existence penalty

#### `sim4 sparsemax` directional-target-ratio follow-up

- Objective:
  - test whether the remaining `sim4 sparsemax` weakness can be fixed by making
    the Patel directional term explicitly stronger
- Setup:
  - fixed branch:
    - `adj_activation=sparsemax`
    - `parent_cap_lambda=0.0`
    - `ungated_symmetry_lambda=0.0`
    - `structure_init_scale=0.5`
    - `structure_message_graph_mode=raw`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `lambda_l1=0.02`
  - dataset:
    - `sim4.csv`
  - seeds:
    - `11,22,33`
  - epochs:
    - `30`
  - sweep:
    - `directional_target_ratio in {0.01, 0.03, 0.05}`
- Artifacts:
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260317_105014_sim4_sparsemax_dirratio_pilot3_aggregate.csv`
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260317_105014_sim4_sparsemax_dirratio_pilot3.csv`
- Aggregate result summary:
  - `dir_ratio=0.01`
    - `strict_f1 eps=0 = 0.3186 +/- 0.0158`
    - `strict_f1 eps=3e-4 = 0.3212 +/- 0.0146`
    - `strict_f1 eps=0.1 = 0.3389 +/- 0.0499`
    - `strict_pred_count eps=3e-4 = 120.0`
    - `eff_parents_mean = 1.60`
    - `gt_margin_median = 0.0063`
    - failure:
      - `2/3 symmetric_collapse`
      - `1/3 weak_asymmetry`
  - `dir_ratio=0.03`
    - `strict_f1 eps=0 = 0.1272 +/- 0.0025`
    - `strict_f1 eps=3e-4 = 0.1282 +/- 0.0033`
    - `strict_f1 eps=0.1 = 0.3021 +/- 0.0088`
    - `strict_pred_count eps=3e-4 = 506.0`
    - `eff_parents_mean = 4.96`
    - `gt_margin_median = 0.0316`
    - failure:
      - `3/3 mixed_or_partial`
  - `dir_ratio=0.05`
    - `strict_f1 eps=0 = 0.1118 +/- 0.0049`
    - `strict_f1 eps=3e-4 = 0.1128 +/- 0.0044`
    - `strict_f1 eps=0.1 = 0.3030 +/- 0.0240`
    - `strict_pred_count eps=3e-4 = 542.0`
    - `eff_parents_mean = 6.09`
    - `gt_margin_median = 0.0282`
    - failure:
      - `3/3 mixed_or_partial`
- Training-log cross-check:
  - `quality_history.csv` does not expose a separate `lambda_dir` column
  - but the run logs do expose `Dir Loss(raw/w)`, and those values confirm that
    the higher-ratio branches really are receiving a stronger directional term:
    - `dir_ratio=0.01`, epoch 30:
      - `Dir Loss(raw/w): 0.0843 / 0.0107`
      - `eff_par=1.85`
    - `dir_ratio=0.03`, epoch 30:
      - `Dir Loss(raw/w): 0.0438 / 0.0227`
      - `eff_par=5.72`
    - `dir_ratio=0.05`, epoch 30:
      - `Dir Loss(raw/w): 0.0420 / 0.0217`
      - `eff_par=6.62`
  - so this is not a case where the extra ratio failed to activate
  - the stronger directional term is active, but it widens support instead of
    restoring clean directional separation
- Interpretation / Thoughts:
  - this follow-up is clearly negative
  - increasing `directional_target_ratio` does raise median GT margin somewhat,
    but it does so by re-densifying the graph
  - the failure mode shifts from `weak_asymmetry / symmetric_collapse` to a much
    worse `mixed_or_partial` regime with hundreds of predicted edges
  - this means the next `sim4 sparsemax` lever should **not** be "stronger Patel
    directional weighting"
  - keep `directional_target_ratio=0.01` as the current best setting on this
    branch

#### Patel-only upper-bound diagnostic on `sim3` / `sim4`

- Objective:
  - test whether the current `sim4` bottleneck is really a `Patel` ceiling, or
    whether the training pipeline is simply failing to realize the signal that
    `Patel` already contains
- Method:
  - no training
  - use dataset-native `patel_kappa.csv` / `patel_tau.csv` directly
  - compare three heuristics on each dataset:
    - `oracle_skeleton + tau_sign`
      - use GT undirected skeleton, direct each pair by `tau_ij >= tau_ji`
    - `top-k kappa + tau_sign`
      - rank unordered pairs by `max(kappa_ij, kappa_ji)`
      - keep top-`k`, where `k = |GT edges|`
      - direct each kept pair by `tau_sign`
    - `q50 kappa_gate + tau_sign`
      - keep all unordered pairs above the positive-`kappa` median threshold
      - direct each kept pair by `tau_sign`
- Result summary:
  - `sim3`
    - `oracle_skeleton + tau_sign`
      - `strict_precision = 0.7778`
      - `strict_recall = 0.7778`
      - `strict_f1 = 0.7778`
    - `top-k kappa + tau_sign`
      - skeleton:
        - `precision = 1.0000`
        - `recall = 1.0000`
        - `f1 = 1.0000`
      - strict:
        - `precision = 0.7778`
        - `recall = 0.7778`
        - `f1 = 0.7778`
    - `q50 kappa_gate + tau_sign`
      - `pred_count = 42`
      - `strict_precision = 0.3333`
      - `strict_recall = 0.7778`
      - `strict_f1 = 0.4667`
  - `sim4`
    - `oracle_skeleton + tau_sign`
      - `strict_precision = 0.7705`
      - `strict_recall = 0.7705`
      - `strict_f1 = 0.7705`
    - `top-k kappa + tau_sign`
      - skeleton:
        - `precision = 1.0000`
        - `recall = 1.0000`
        - `f1 = 1.0000`
      - strict:
        - `precision = 0.7705`
        - `recall = 0.7705`
        - `f1 = 0.7705`
    - `q50 kappa_gate + tau_sign`
      - `pred_count = 409`
      - `strict_precision = 0.1149`
      - `strict_recall = 0.7705`
      - `strict_f1 = 0.2000`
- Interpretation / Thoughts:
  - this diagnostic is the strongest evidence so far **against** the idea that
    `sim4` is mainly blocked by a low-`Patel` ceiling
  - on both datasets, `top-k kappa + tau_sign` reaches essentially the same
    result as `oracle_skeleton + tau_sign`
    - i.e. `kappa` already identifies the exact GT skeleton at the correct edge
      budget
    - and `tau_sign` already gives a ~`0.77` directional ceiling on that
      skeleton
  - in other words:
    - `Patel` is not the bottleneck at the benchmark level
    - the bottleneck is that the current training setup does **not** translate
      `kappa` into a hard skeleton choice
      and does **not** translate `tau` into stable pairwise direction
  - the `q50 kappa_gate + tau_sign` result is especially informative:
    - `sim3`: `42` predicted pairs
    - `sim4`: `409` predicted pairs
    - this mirrors the training story almost exactly
      - `kappa` is currently being used as a **broad gate**
      - not as an actual skeleton selector
  - updated strategic conclusion:
    - the next useful mechanism is unlikely to be "stronger Patel loss"
    - it should be a structure that can:
      - enforce pair/skeleton selection much more explicitly
      - then resolve direction inside the selected support
    - said differently:
      - the missing piece is not more signal
      - it is a better way to **use** the signal already present in
        `kappa/tau`

#### `sim4 sparsemax` persistent `kappa`-logit-bias smoke

- Objective:
  - test the smallest possible upgrade from "kappa only as init / q50 gate" to
    "kappa as a persistent skeleton prior"
  - mechanism:
    - add a symmetric `kappa` prior directly into structure logits:
      - `logits = sender @ receiver.T + adj_bias + alpha * kappa_prior`
- Implementation:
  - added `--kappa_logit_bias_scale` to `main_structure_learning.py`
  - added `--kappa_logit_bias_scales` sweep to
    `run_cross_pred_v1_final_only_compare.py`
  - `kappa_prior` is symmetric, non-negative, zero-diagonal, and persistent for
    the whole training run
- Setup:
  - fixed branch:
    - `sim4.csv`
    - `adj_activation=sparsemax`
    - `structure_init_mode=patel_kappa`
    - `structure_init_scale=0.5`
    - `directional_kappa_gate=True`
    - `directional_kappa_gate_quantile=0.50`
    - `directional_target_ratio=0.01`
    - `parent_cap_lambda=0.0`
    - `ungated_symmetry_lambda=0.0`
    - `lambda_l1=0.02`
    - `optimizer_step_mode=subject`
    - `structure_message_graph_mode=raw`
  - seed:
    - `11`
  - epochs:
    - `30`
  - sweep:
    - `kappa_logit_bias_scale in {0.0, 0.5, 1.0, 2.0, 4.0}`
- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260317_141900_sim4_sparsemax_kappabias_smoke1.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260317_141900_sim4_sparsemax_kappabias_smoke1_aggregate.csv`
- Result summary:
  - `kappa_bias=0.0`
    - `strict_f1 eps=3e-4 = 0.2941`
    - `strict_pred_count eps=3e-4 = 143`
    - `eff_parents_mean = 1.70`
    - `gt_margin_median = 0.0000`
    - `gt_pos = 49.18%`
    - `margin_p90 = 8.47e-3`
  - `kappa_bias=0.5`
    - `strict_f1 eps=3e-4 = 0.2911`
    - `strict_pred_count eps=3e-4 = 152`
    - `eff_parents_mean = 1.88`
    - `gt_margin_median = 1.01e-3`
    - `gt_pos = 50.82%`
    - `margin_p90 = 1.44e-2`
  - `kappa_bias=1.0`
    - `strict_f1 eps=3e-4 = 0.2936`
    - `strict_pred_count eps=3e-4 = 157`
    - `eff_parents_mean = 1.94`
    - `gt_margin_median = 6.10e-3`
    - `gt_pos = 52.46%`
    - `margin_p90 = 2.17e-2`
  - `kappa_bias=2.0`
    - `strict_f1 eps=3e-4 = 0.3004`
    - `strict_pred_count eps=3e-4 = 172`
    - `eff_parents_mean = 1.99`
    - `gt_margin_median = 2.97e-2`
    - `gt_pos = 57.38%`
    - `margin_p90 = 2.64e-2`
  - `kappa_bias=4.0`
    - `strict_f1 eps=3e-4 = 0.2982`
    - `strict_pred_count eps=3e-4 = 167`
    - `eff_parents_mean = 2.04`
    - `gt_margin_median = 1.97e-2`
    - `gt_pos = 55.74%`
    - `margin_p90 = 2.18e-2`
    - `strict_f1 eps=0.1 = 0.4144`
- Interpretation / Thoughts:
  - this mechanism is **not** the missing skeleton-selection lever
  - increasing `kappa_logit_bias_scale` does not pull the graph toward the
    Patel top-`k` oracle
    - `strict_pred_count` rises from `143` to `167-172`
    - `eff_parents_mean` rises from `1.70` to `~2.0`
  - so the persistent `kappa` bias does **not** make support more selective
  - what it does do is raise margin scale somewhat:
    - `gt_margin_median` moves from `0.0000` to `~0.02-0.03`
    - `margin_p90` roughly triples
    - `eps=0.1` strict F1 improves substantially (`0.3469 -> 0.4144`)
  - that means this bias mostly strengthens confidence on the pairs already
    being selected, but it does not solve the core mismatch diagnosed by the
    Patel oracle experiment
- updated conclusion:
  - continuous `kappa` bias is at best a small margin-shaping helper
  - it is **not** a substitute for explicit skeleton selection
  - the next mechanism should therefore be more structural:
      - either explicit support selection from `kappa`
      - or a support/direction factorization where pair existence and direction
        no longer interfere

#### `sim4` support/direction factorization smoke with fixed top-`k` support

- Objective:
  - directly test the hypothesis that the current bottleneck is the coupled
    parameterization itself
  - force support selection to be solved by `top-k kappa`, then ask whether a
    separate direction branch can realize the remaining `tau` signal cleanly
- Mechanism:
  - new `structure_parameterization=support_direction`
  - support branch:
    - symmetric support logits
    - exported adjacency is `support * direction_gate`
  - direction branch:
    - separate bilinear logits
    - pairwise direction gate `sigmoid(D_ij - D_ji)`
  - optional `fixed_support_mask_mode=topk_kappa`
    - keep only the top-`k` undirected `kappa` pairs
    - here `k=61` on `sim4`
- Implementation notes:
  - added:
    - `--structure_parameterization {coupled,support_direction}`
    - `--fixed_support_mask_mode {none,topk_kappa}`
    - `--direction_init_mode {patel_tau,zeros,random}`
  - directional supervision was switched to use the separate direction logits
    when `support_direction` is enabled
  - during this follow-up, two training pathologies were found and clarified:
    - `adaptive_margin` could collapse to `0` when all directional logits tie,
      which makes the directional loss turn off exactly at the symmetric
      solution
      - fixed by lower-bounding adaptive margin with the base margin
    - `direction_init=zeros` is a dead initialization for the bilinear
      direction branch
      - both sender and receiver factors are zero, so the branch gets no usable
        gradient and remains perfectly symmetric
- Common setup:
  - dataset:
    - `sim4.csv`
  - support:
    - fixed `top-k kappa`, `k=61`
  - structure init:
    - `patel_kappa`, `scale=0.5`
  - direction prior:
    - `patel`
  - direction gate:
    - `kappa quantile = 0.50`
  - epochs:
    - `30`
  - seed:
    - `11`
  - message graph mode:
    - `raw`
  - adjacency activation:
    - `sigmoid`
- Result A: `direction_init=patel_tau`
  - artifact:
    - `results/run_20260317_150847`
  - best exported causal adjacency:
    - `strict_pred_count @ eps=0 = 61`
    - `strict_f1 @ eps=0 = 0.8033`
    - `strict_f1 @ eps=3e-4 = 0.8033`
    - `strict_f1 @ eps=0.1 = 0.4359`
    - `gt_margin_median = 0.0481`
    - `gt_frac_pos = 0.8033`
  - final epoch causal adjacency:
    - `strict_f1 @ eps=0 = 0.7705`
    - `strict_f1 @ eps=3e-4 = 0.7603`
    - `strict_f1 @ eps=0.1 = 0.0923`
  - interpretation:
    - once support is fixed to the correct `kappa` skeleton, the factorized
      form immediately jumps to the `~0.77-0.80` regime predicted by the Patel
      oracle
    - this is far above the best coupled `sim4 sparsemax` pilot (`~0.31`)
- Result B: `direction_init=zeros`
  - artifact:
    - `GraphExp/results/run_20260317_152819`
  - best/final causal adjacency:
    - `strict_pred_count = 0`
    - `strict_f1 @ eps=0 = 0.0000`
    - `strict_f1 @ eps=3e-4 = 0.0000`
    - `gt_margin_median = 0.0000`
  - interpretation:
    - this is a negative control, but not evidence against factorization
    - it confirms the bilinear direction branch cannot be initialized at exact
      zero
- Result C: `direction_init=random`
  - artifact:
    - `GraphExp/results/run_20260317_153438`
  - best exported causal adjacency:
    - `strict_pred_count @ eps=0 = 61`
    - `strict_f1 @ eps=0 = 0.8033`
    - `strict_f1 @ eps=3e-4 = 0.8033`
    - `strict_f1 @ eps=0.1 = 0.3896`
    - `gt_margin_median = 0.0444`
    - `gt_frac_pos = 0.8033`
  - final epoch causal adjacency:
    - `strict_f1 @ eps=0 = 0.7705`
    - `strict_f1 @ eps=3e-4 = 0.7705`
    - `strict_f1 @ eps=0.1 = 0.0635`
  - training-side evidence that direction is actually learned:
    - quality history shows proxy agreement climbing from `0.40` at epoch `1`
      to `1.00` by epochs `8-9`
    - best epoch selected at epoch `9`, not at initialization
  - interpretation:
    - this is the key confirmation
    - with the coupled support problem removed, a separate direction branch can
      learn to realize the `tau` signal even without `tau` initialization
- Interpretation / Thoughts:
  - this is the strongest support yet for the original diagnosis:
    - the main `sim4` bottleneck is **not** missing directional signal
    - it is the coupled `N x N` parameterization that entangles support
      existence and direction in one matrix
  - fixed-support factorization reaches `strict_f1 ≈ 0.80`, essentially the
    Patel top-`k` ceiling, while the coupled sparsemax branch stalls near
    `0.31`
  - therefore:
    - support/direction factorization is a valid next mainline architecture
    - but the present smoke still uses a fixed `top-k kappa` support mask
  - updated strategic conclusion:
    - factorization itself is validated
    - the next real problem is to replace fixed top-`k` support with a
      learnable or data-adaptive support selector
      without collapsing back to the old coupled failure mode

#### `maxgap_kappa` support replacement for factorized branch

- Objective:
  - test the cheapest replacement for fixed `top-k`
  - instead of giving the model the GT edge count, infer support size directly
    from the `kappa` ranking by cutting at the largest adjacent score gap
- Mechanism:
  - for each unordered pair, score = `max(kappa_ij, kappa_ji)`
  - sort scores descending
  - choose the split at the largest gap `score[m] - score[m+1]`
  - keep the top `m` pairs as the undirected support
- Implementation:
  - added `build_undirected_kappa_skeleton(..., selection_mode={topk,maxgap})`
  - added `--fixed_support_mask_mode maxgap_kappa`
  - when `maxgap_kappa` is used, the same inferred skeleton is used for:
    - `noise_guide_adj`
    - `fixed_support_mask`
    - `target_edge_count` / selection target density
- Offline validation:
  - `sim3`
    - inferred `k = 18`
    - threshold `= 0.213051`
    - max gap `= 0.048385`
  - `sim4`
    - inferred `k = 61`
    - threshold `= 0.206475`
    - max gap `= 0.030850`
  - importantly:
    - on both datasets, the largest `kappa` gap lands exactly at the GT edge
      count
    - so this is not just a loose elbow heuristic on these benchmarks

#### `support_direction + maxgap_kappa + random direction init` smoke

- Common setup:
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `structure_init_mode = patel_kappa`
  - `structure_init_scale = 0.5`
  - `adj_activation = sigmoid`
  - `directional_prior_mode = patel`
  - `directional_kappa_gate = True`
  - `directional_kappa_gate_quantile = 0.50`
  - `directional_target_ratio = 0.01`
  - `optimizer_step_mode = subject`
  - `epochs = 30`
  - `seed = 11`
- `sim3`
  - artifact:
    - `GraphExp/results/run_20260317_160215`
  - best exported causal adjacency:
    - `strict_pred_count @ eps=0 = 18`
    - `strict_f1 @ eps=0 = 0.8889`
    - `strict_f1 @ eps=3e-4 = 0.8889`
    - `strict_f1 @ eps=0.1 = 0.7097`
    - `gt_margin_median = 0.1411`
    - `gt_frac_pos = 0.8889`
  - final epoch causal adjacency:
    - `strict_f1 @ eps=0 = 0.8333`
    - `strict_f1 @ eps=3e-4 = 0.8333`
    - `strict_f1 @ eps=0.1 = 0.3478`
- `sim4`
  - artifact:
    - `GraphExp/results/run_20260317_160408`
  - best exported causal adjacency:
    - `strict_pred_count @ eps=0 = 61`
    - `strict_f1 @ eps=0 = 0.8033`
    - `strict_f1 @ eps=3e-4 = 0.8033`
    - `strict_f1 @ eps=0.1 = 0.3896`
    - `gt_margin_median = 0.0457`
    - `gt_frac_pos = 0.8033`
  - final epoch causal adjacency:
    - `strict_f1 @ eps=0 = 0.7705`
    - `strict_f1 @ eps=3e-4 = 0.7705`
    - `strict_f1 @ eps=0.1 = 0.0635`
- Interpretation / Thoughts:
  - this follow-up is strongly positive
  - on both benchmarks, `maxgap_kappa` reproduces the hand-specified top-`k`
    support exactly
  - therefore, on the current synthetic suite:
    - we do **not** need a learned support selector yet
    - `Patel kappa` already provides a data-adaptive hard skeleton rule
    - the missing ingredient was the factorized direction branch
  - updated strategic conclusion:
    - the current best architecture is now:
      - `hard kappa skeleton selection` via `maxgap_kappa`
      - plus `support/direction` factorization
    - the next useful step is no longer "invent a learnable support selector"
      by default
    - it is to formalize this branch first:
      - multi-seed
      - 100-epoch
      - compare `sim3` / `sim4`

#### Formal: `support_direction + maxgap_kappa + random` on `sim3` / `sim4`

- Objective:
  - formalize the now-leading branch with:
    - hard support from `maxgap_kappa`
    - separate direction branch
    - random direction init
  - and explicitly report both:
    - best exported epoch
    - final epoch
- Common setup:
  - seeds:
    - `11,22,33,44,55`
  - epochs:
    - `100`
  - config:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `direction_init_mode = random`
    - `structure_init_mode = patel_kappa`
    - `structure_init_scale = 0.5`
    - `adj_activation = sigmoid`
    - `directional_prior_mode = patel`
    - `directional_kappa_gate = True`
    - `directional_kappa_gate_quantile = 0.50`
    - `directional_target_ratio = 0.01`
    - `lambda_l1 = 0.02`
    - `optimizer_step_mode = subject`
    - `strict_margin_eps in {0, 3e-4, 0.1}`
- Artifacts:
  - `sim3`
    - summary:
      - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260317_164608_sim3_support_direction_maxgap_formal.csv`
    - aggregate:
      - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260317_164608_sim3_support_direction_maxgap_formal_aggregate.csv`
  - `sim4`
    - summary:
      - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260317_172003_sim4_support_direction_maxgap_formal.csv`
    - aggregate:
      - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260317_172003_sim4_support_direction_maxgap_formal_aggregate.csv`

#### Formal result summary

- `sim3`
  - best epoch:
    - `best_strict_f1 @ eps=0 = 0.8222 ± 0.0544`
    - `best_strict_f1 @ eps=3e-4 = 0.8222 ± 0.0544`
    - `best_strict_f1 @ eps=0.1 = 0.7039 ± 0.0276`
    - `best_gt_margin_median = 0.1378 ± 0.0105`
    - `best_gt_frac_pos = 82.22% ± 5.44%`
  - final epoch:
    - `strict_f1 @ eps=0 = 0.7778 ± 0.0497`
    - `strict_f1 @ eps=3e-4 = 0.7778 ± 0.0497`
    - `strict_f1 @ eps=0.1 = 0.0000`
    - `gt_margin_median = 0.0272 ± 0.0037`
  - failure mode:
    - `weak_asymmetry` in all `5/5` seeds
- `sim4`
  - best epoch:
    - `best_strict_f1 @ eps=0 = 0.7902 ± 0.0217`
    - `best_strict_f1 @ eps=3e-4 = 0.7915 ± 0.0217`
    - `best_strict_f1 @ eps=0.1 = 0.3779 ± 0.0121`
    - `best_gt_margin_median = 0.0416 ± 0.0053`
    - `best_gt_frac_pos = 79.02% ± 2.17%`
  - final epoch:
    - `strict_f1 @ eps=0 = 0.7344 ± 0.0502`
    - `strict_f1 @ eps=3e-4 = 0.7272 ± 0.0481`
    - `strict_f1 @ eps=0.1 = 0.0000`
    - `gt_margin_median = 0.0097 ± 0.0011`
  - failure mode:
    - `symmetric_collapse` in all `5/5` seeds

- Interpretation / Thoughts:
  - this formal confirms both of the key hypotheses behind the branch:
    - `best` epoch remains very strong on both datasets
    - but `final` epoch degrades substantially over long training
  - so the branch is **real**, but its current stability depends on epoch
    selection
  - the best/final gap is now a first-order result, not a smoke artifact:
    - `sim3`: `0.8222 -> 0.7778`
    - `sim4`: `0.7915 -> 0.7272` at `eps=3e-4`
  - the second hypothesis is also supported, but more modestly than the 1-seed
    smoke first suggested:
    - `sim3` best mean (`0.8222`) is still above the Patel top-`k + tau-sign`
      ceiling (`0.7778`)
    - `sim4` best mean (`0.7915`) is also slightly above the Patel ceiling
      (`0.7705`)
    - so the direction branch does appear to extract some additional usable
      directional information from training
    - but the gain is not huge, and it is partially lost by the final epoch
  - updated strategic conclusion:
    - support selection is solved well enough for the current synthetic suite
      by `maxgap_kappa`
    - the next bottleneck is **direction retention over long training**
    - that means the next follow-up should target training dynamics:
      - early stopping based on the current best-epoch proxy
      - and/or direction-branch-specific learning-rate decay / freeze schedule

### Strategic decision: choose Option A (`diffusion as structured carrier`)

- Objective:
  - record the current interpretation of the project after the latest ablation
    review
  - decide whether the mainline should continue chasing "diffusion discovers
    direction by itself" or instead formalize the already-supported
    `prior -> parameterization -> export` story

- Evidence summary driving this decision:
  - diffusion-only direction discovery is not working in the current framework
    - `ablation 1`: with Patel supervision removed, `100`-epoch training does
      not recover direction
      - `95.8%` of pair margins stay `< 0.01`
      - `5/5` seeds end in `symmetric_collapse`
  - the denoising objective does not appear to rescue direction after the prior
    is removed
    - `ablation 3`: turning Patel supervision off after epoch `15` produces a
      degradation trajectory close to "Patel on for the whole run"
    - working interpretation:
      - denoising is largely direction-insensitive in the later phase
  - gradient evidence points to active tension rather than hidden cooperation
    - `ablation 4`: cosine between denoising gradients and directional gradients
      on direction parameters is often negative
      - `sim4` mean cosine `= -0.307`
    - so the current denoising path is not a credible source of autonomous
      directional improvement
  - the current best branch remains close to the Patel-oracle ceiling
    - Patel top-`k` / oracle-style references on `sim3` and `sim4` are around
      `0.77`
    - the best `support_direction + maxgap_kappa + random` formal means are:
      - `sim3`: `best_strict_f1 @ eps=3e-4 = 0.8222`
      - `sim4`: `best_strict_f1 @ eps=3e-4 = 0.7915`
    - practical interpretation:
      - the current framework is mostly learning how to realize a strong Patel
        prior inside a differentiable adjacency parameterization
      - it is **not** yet justified to claim that diffusion itself discovers
        causal direction

- Practical conclusion:
  - choose **Option A** as the mainline interpretation
  - from this point onward, the most defensible project statement is:
    - given a strong pairwise causal prior, design a parameterization and
      training/export pipeline that uses that prior as effectively as possible
  - the current method contribution is therefore centered on:
    - `support/direction` factorization
    - hard skeleton selection via `maxgap_kappa`
    - competitive / structure-aware parameterization choices
    - stable export / best-epoch selection under this factorized structure
  - do **not** frame the present architecture as "diffusion autonomously learns
    direction from denoising"

### Next experiment design: minimum required validation for Option A

- Objective:
  - strengthen the Option A story at the points where the current evidence is
    still most vulnerable
  - answer the reviewer-style question:
    - if Patel already contains most of the signal, what exactly is the
      diffusion framework adding?

- Highest-priority required comparisons:
  - direct Patel heuristic baseline
    - `maxgap_kappa + tau_sign`
    - purpose:
      - quantify the strongest non-learned baseline under the same inferred
        skeleton rule
  - factorized no-diffusion baseline
    - keep `support_direction + maxgap_kappa`, but remove the denoising loss
      from the training objective
    - purpose:
      - test whether the gain over direct Patel comes from the factorized
        learnable direction branch itself, from diffusion training, or both
  - full current model
    - `support_direction + maxgap_kappa + diffusion`
    - purpose:
      - measure the actual incremental value of the denoising backbone after the
        support/direction decomposition is fixed

- Second-priority required stress tests:
  - prior-quality degradation experiments
    - reduce subject count
    - reduce time length
    - perturb or partially corrupt `kappa/tau`
    - compare which method degrades more gracefully:
      - direct Patel heuristic
      - factorized no-diffusion
      - full diffusion model
    - purpose:
      - if the full model is only matching Patel when Patel is already nearly
        oracle-level, the Option A story stays weak
      - if the full model is more robust under degraded priors, the story
        becomes much stronger

- Claim-boundary follow-up:
  - decide explicitly whether the paper/story is:
    - Patel-specific
    - or a generic external-prior utilization framework
  - if the claim remains generic, add at least one additional prior family
  - if not, narrow the claim deliberately to:
    - structured utilization of Patel-style pairwise causal priors

- Stability follow-up:
  - formalize the `best` vs `final` selection protocol
    - current best/final gaps are too large to leave as an informal detail
  - likely useful directions:
    - lock in a best-epoch proxy and report it consistently
    - compare against final-epoch export
    - if needed, test direction-branch LR decay / freeze only as a stability
      aid, not as a new main claim

- Deprioritized directions under Option A:
  - more `cross-pred v1/v1.5` variants
  - more loss stacking intended to make diffusion "discover" direction inside
    the current architecture
  - small schedule / margin / cap tweaks whose only purpose is to rescue the
    old coupled story

### Next experiment design: four-goal Option A validation suite

- Objective:
  - convert the high-level Option A decision into a compact but defensible
    experiment roadmap
  - answer four concrete questions:
    - what does diffusion add beyond direct Patel heuristics?
    - is the framework more robust when the prior is imperfect?
    - should the claim stay Patel-specific or be widened toward generic priors?
    - how should best-vs-final selection be stabilized and reported?

- Common base config for all learned branches unless explicitly overridden:
  - dataset:
    - `sim3.csv`
    - `sim4.csv`
  - seeds:
    - `11,22,33,44,55`
  - epochs:
    - `100`
  - structure:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `direction_init_mode = random`
    - `structure_init_mode = patel_kappa`
    - `structure_init_scale = 0.5`
    - `adj_activation = sigmoid`
  - training:
    - `directional_prior_mode = patel`
    - `directional_kappa_gate = True`
    - `directional_kappa_gate_quantile = 0.50`
    - `directional_target_ratio = 0.01`
    - `lambda_l1 = 0.02`
    - `optimizer_step_mode = subject`
  - reporting:
    - `strict_margin_eps in {0, 3e-4, 0.1}`
    - always report both:
      - best exported epoch
      - final epoch
    - always report:
      - `strict_f1`
      - `strict_precision`
      - `strict_recall`
      - `gt_margin_median`
      - `best_final_gap_strict_f1`
      - `best_final_gap_gt_signed_margin_median`

#### Goal 1: isolate what diffusion is actually contributing

- Core question:
  - after support is already fixed by `maxgap_kappa`, does diffusion add any
    directional value beyond a direct prior heuristic or a learnable
    direction-only branch?

- Required branches:
  - branch A: direct heuristic reference
    - deterministic offline baseline:
      - `maxgap_kappa + tau_sign`
    - purpose:
      - strongest non-learned Patel reference under the same skeleton rule
  - branch B: diffusion-only negative control
    - use the common base config
    - set:
      - `disable_directional_loss = True`
    - purpose:
      - test whether denoising can orient edges once support is solved
  - branch C: factorized no-diffusion baseline
    - use the common base config
    - add a small plumbing flag:
      - `main_loss_weight = 0`
      - or equivalently `disable_main_loss = True`
    - keep directional supervision on
    - purpose:
      - isolate the contribution of the learnable factorized direction branch
      - separate it from any benefit of diffusion training
  - branch D: full current model
    - use the common base config exactly as-is
    - purpose:
      - measure the incremental value of diffusion on top of the factorized
        prior-utilization design

- Interpretation logic:
  - if branch B still collapses:
    - diffusion remains direction-insensitive even with correct support
  - if branch C > branch A:
    - the learnable factorized direction branch adds value over hard `tau_sign`
  - if branch D > branch C:
    - diffusion contributes additional usable signal or regularization
  - if branch D ~= branch C:
    - the real gain is mostly from parameterization, not diffusion itself

- Practical notes:
  - branch A is deterministic and does not need seed averaging
  - branches B/C/D should be run as `5`-seed formal on both `sim3` and `sim4`
  - this is the highest-priority block because it determines the central claim

#### Goal 2: test whether the framework is more robust under imperfect priors

- Core question:
  - if the prior is no longer near-oracle, does the learned factorized pipeline
    degrade more gracefully than direct heuristics?

- Stress family 1: data-starvation prior degradation
  - mechanism:
    - recompute priors from reduced observations rather than perturbing priors
      synthetically
  - recommended reduced-data grid:
    - clean:
      - `subjects=50, time_points=200`
    - subject-starved:
      - `subjects=10, time_points=200`
    - time-starved:
      - `subjects=50, time_points=50`
    - balanced-mid:
      - `subjects=20, time_points=100`
    - severe:
      - `subjects=10, time_points=50`
  - compared methods:
    - branch A: direct heuristic
    - branch C: factorized no-diffusion
    - branch D: full current model
  - staging:
    - `sim3`, `3`-seed pilot first to verify the grid is informative
    - promote the clean / mid / severe conditions to `5`-seed formal on
      `sim3` and `sim4`

- Stress family 2: explicit prior corruption
  - mechanism:
    - perturb the prior directly while keeping the raw data unchanged
  - minimal required new plumbing:
    - symmetric support corruption:
      - `kappa_pair_swap_ratio` or `kappa_noise_std`
    - directional corruption:
      - `tau_flip_ratio`
  - recommended one-dimensional sweeps:
    - support corruption only:
      - `kappa corruption in {0.0, 0.1, 0.2}`
      - `tau` kept clean
    - direction corruption only:
      - `tau flip ratio in {0.0, 0.1, 0.2}`
      - `kappa` kept clean
  - compared methods:
    - branch A
    - branch C
    - branch D

- Decision rule:
  - the Option A story becomes substantially stronger only if branch D loses
    less performance than branch A/C as priors worsen
  - primary summary should be:
    - absolute `strict_f1`
    - relative drop from the clean condition
    - best/final gap under stress

#### Goal 3: decide whether the claim is Patel-specific or genuinely generic

- Core question:
  - does the factorized framework exploit only Patel priors well, or can it
    also use other pairwise direction/support signals?

- Stage 3A: direction-prior genericity check with existing plumbing
  - keep support fixed to:
    - `maxgap_kappa`
  - sweep:
    - `directional_prior_mode in {patel, lag_corr_raw, lag_corr_encoder}`
  - compare:
    - direct sign heuristic from the same direction source when available
    - branch C: factorized no-diffusion
    - branch D: full model
  - staging:
    - `sim3`, `3`-seed pilot first
    - promote to `sim4` only if non-Patel direction priors are at least
      qualitatively viable
  - purpose:
    - test whether the direction branch only works with Patel `tau`, or can
      exploit weaker directional priors too

- Stage 3B: support-prior genericity check with small new plumbing
  - add a generalized support-prior interface, e.g.:
    - `support_prior_mode in {patel_kappa, pearson_abs}`
    - `support_selection_mode in {topk, maxgap}`
  - recommended comparison:
    - Patel support + Patel direction
    - Pearson support + Patel direction
  - compared methods:
    - direct heuristic reference under the same support rule
    - branch C
    - branch D
  - purpose:
    - test whether the framework can also improve over non-Patel support priors,
      not just consume Patel's nearly oracle skeleton

- Claim decision rule:
  - if off-Patel priors stay weak and the learned branches do not materially
    improve them:
    - keep the claim deliberately Patel-specific
  - if off-Patel priors are consistently improved by the factorized learned
    pipeline:
    - widen the story to "structured utilization of external pairwise priors"

#### Goal 4: lock down the best-vs-final stability and selection protocol

- Core question:
  - the current branch is strong at the best epoch but decays by the final
    epoch; what reporting rule is defensible, and can the gap be reduced?

- Step 4A: selection audit on the current full model
  - use branch D on `sim3` and `sim4`
  - compare:
    - guarded best epoch
    - score-only fallback best epoch
    - final epoch
  - additionally sweep:
    - `selection_agreement_weight in {0.0, 0.25}`
  - purpose:
    - verify whether the current guarded proxy is genuinely helping, and how
      dependent the reported result is on Patel agreement during selection

- Step 4B: stability-intervention sweep
  - use branch D on `sim4` first, because it has the larger best/final gap
  - supported knobs already exposed by the runner:
    - `direction_lr_multiplier`
    - `freeze_direction_after_epoch`
    - `directional_loss_end_epoch`
  - recommended compact pilot grid:
    - baseline:
      - `dir_lr_mult=1.0, freeze=-1, dir_end=-1`
    - lower direction LR:
      - `dir_lr_mult=0.3, freeze=-1, dir_end=-1`
    - early direction freeze:
      - `dir_lr_mult=1.0, freeze=30, dir_end=-1`
    - lower LR + freeze:
      - `dir_lr_mult=0.3, freeze=30, dir_end=-1`
    - stop directional supervision early:
      - `dir_lr_mult=1.0, freeze=-1, dir_end=30`
    - lower LR + early supervision stop:
      - `dir_lr_mult=0.3, freeze=-1, dir_end=30`
  - staging:
    - `sim4`, `3`-seed pilot first
    - promote the best `1-2` settings to `5`-seed formal on `sim3` and `sim4`

- Stability success criterion:
  - primary objective:
    - reduce `best_final_gap_strict_f1` and `best_final_gap_gt_signed_margin`
  - with constraint:
    - do not materially reduce best-epoch performance
  - the winning protocol should be the simplest one that keeps the gap smaller
    while preserving the current best-epoch regime

- Final reporting rule to lock in after this block:
  - primary metric:
    - guarded best-epoch export at `eps=3e-4`
  - secondary metric:
    - final epoch at the same `eps`
  - mandatory disclosure:
    - best/final gap
    - selection mode
    - whether the reported branch used any direction freeze / LR reduction

#### Minimal plumbing required before running the full four-goal suite

- Needed immediately:
  - add `main_loss_weight` or `disable_main_loss`
    - required for the factorized no-diffusion baseline in Goal 1 / Goal 2 /
      Goal 3
  - add a lightweight reduced-data interface
    - either:
      - derived CSV generation scripts
      - or runner flags such as `subject_limit` / `time_limit`
    - required for Goal 2 data-starvation tests

- Needed only if the generic-claim path is pursued:
  - generalized support-prior plumbing
    - e.g. `support_prior_mode`
    - needed for Goal 3B
  - explicit prior-perturbation flags
    - `kappa corruption`
    - `tau corruption`
    - needed for Goal 2 explicit prior-corruption tests

- Practical ordering:
  - first:
    - Goal 1
    - Goal 4A
  - second:
    - Goal 4B
    - Goal 2 pilot
  - third:
    - Goal 3A
  - only then decide whether Goal 3B is worth the extra plumbing

### Plumbing: Option A suite execution support

- Objective:
  - implement the minimum code-path changes needed to actually run the
    highest-priority Option A experiments
  - specifically:
    - Goal 1 branch C (`factorized no-diffusion`)
    - Goal 2 reduced-data stress tests
    - Goal 4A selection-protocol sweep

- Code changes:
  - `GraphExp/main_structure_learning.py`
    - added:
      - `--main_loss_weight`
      - `--subject_limit`
      - `--time_limit`
    - `load_fmri_data(...)` now supports:
      - subject truncation after reshape
      - time-axis truncation after reshape
      - recomputing `data_2d` from the truncated tensor so Pearson / Patel are
        derived from the same reduced dataset
    - training now applies:
      - `total_loss = main_loss_weight * loss_ddm_main + auxiliary_terms`
    - config / quality logging now records:
      - `main_loss_weight`
      - `effective_time_points`
  - `GraphExp/run_cross_pred_v1_final_only_compare.py`
    - added runner arguments:
      - `--main_loss_weights`
      - `--selection_agreement_weights`
      - `--subject_limit`
      - `--time_limit`
    - runner now forwards:
      - reduced-data limits
      - main-loss weight
      - best-epoch Patel-agreement selection weight
    - run / aggregate / paired CSV metadata now includes:
      - `main_loss_weight`
      - `selection_agreement_weight`
      - `subject_limit`
      - `time_limit`
    - important repair:
      - fixed the baseline/treatment key mapping used by aggregate and paired
        comparisons
      - previous code was slicing treatment keys by position in a way that did
        not reliably reconstruct the corresponding baseline key
      - this repair is required so Goal 1 / Goal 4A comparison CSVs remain
        trustworthy once new sweep dimensions are introduced

- Verification:
  - passed:
    - `python -m py_compile GraphExp\main_structure_learning.py GraphExp\run_cross_pred_v1_final_only_compare.py`
    - `python GraphExp\main_structure_learning.py --help`
    - `python GraphExp\run_cross_pred_v1_final_only_compare.py --help`
  - not run yet:
    - no training experiments were launched in this plumbing step

- Practical conclusion:
  - the codebase can now express the first required Option A comparisons
    without ad-hoc one-off scripts:
    - full model vs no-diffusion
    - reduced-data stress tests
    - guarded-selection weight sweeps
  - the next step should be to run:
    - Goal 1
    - Goal 4A
    - before spending time on broader prior-genericity plumbing

### Experiment: Goal 1 diffusion-only negative control on fixed `maxgap_kappa` support (`sim3`, 3-seed pilot)

- Objective:
  - test whether denoising alone can recover direction once support selection is
    already solved by `maxgap_kappa`
  - this is the direct negative control for the Option A claim boundary

- Setup:
  - dataset:
    - `sim3.csv`
  - seeds:
    - `11,22,33`
  - branch:
    - `structure_parameterization = support_direction`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `direction_init_mode = random`
    - `structure_init_mode = patel_kappa`
    - `structure_init_scale = 0.5`
    - `adj_activation = sigmoid`
    - `main_loss_weight = 1.0`
    - `disable_directional_loss = True`
    - `selection_agreement_weight = 0.0`
  - epochs:
    - `100`

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_off_3seeds_20260320_175537_goal1_diff_only_sim3_pilot.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_cross_off_3seeds_20260320_175537_goal1_diff_only_sim3_pilot_aggregate.csv`

- Aggregate result summary:
  - failure mode:
    - `symmetric_collapse` in `3/3` seeds
  - best epoch:
    - `best_strict_f1 @ eps=0 = 0.6296 +/- 0.0524`
    - `best_gt_margin_median = 0.0263 +/- 0.0231`
  - final epoch:
    - `strict_f1 @ eps=0 = 0.5000 +/- 0.0454`
    - `strict_f1 @ eps=3e-4 = 0.4706 +/- 0.0615`
    - `strict_f1 @ eps=0.1 = 0.0000`
    - `gt_margin_median = 0.0003 +/- 0.0019`
    - `margin_median = 0.0000`
    - `margin_lt_1e2_frac = 95.87% +/- 0.45%`

- Interpretation / Thoughts:
  - fixing support is **not** enough to make denoising discover direction
  - final margins collapse back to near-zero even though the skeleton is fixed
  - the final `strict_f1 ~ 0.50` regime is consistent with near-random
    direction assignment on the correct edge budget, not with learned
    asymmetry

- Practical conclusion:
  - this pilot supports the same high-level diagnosis as the earlier ablations:
    - the current denoising path remains direction-insensitive
  - therefore, any gain in the strong branch must come from:
    - the factorized directional machinery
    - and possibly how diffusion interacts with it
    - not from diffusion-only direction discovery

### Experiment: Goal 1 `main_loss_weight` ablation (`no-diffusion` vs `full`) on `sim3` (`3`-seed pilot)

- Objective:
  - isolate the incremental effect of diffusion after support selection and the
    factorized direction branch are already fixed
  - compare:
    - `main_loss_weight = 0.0` (`no-diffusion`)
    - `main_loss_weight = 1.0` (`full`)

- Setup:
  - dataset:
    - `sim3.csv`
  - seeds:
    - `11,22,33`
  - common branch:
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
    - `selection_agreement_weight = 0.0`
  - sweep:
    - `main_loss_weight in {0.0, 1.0}`

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260320_183526_goal1_nodiff_vs_full_sim3_pilot.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260320_183526_goal1_nodiff_vs_full_sim3_pilot_aggregate.csv`
  - note:
    - no paired comparison CSV was produced for this run family
    - the current comparison helper treats "all auxiliary losses off" as the
      baseline key, so this `main_loss_weight` ablation is best read directly
      from the summary / aggregate tables

- Aggregate result summary:
  - `main_loss_weight = 0.0`
    - failure mode:
      - `mixed_or_partial` in `3/3` seeds
    - best epoch:
      - `best_strict_f1 @ eps=0 = 0.7778 +/- 0.0000`
      - `best_gt_margin_median = 0.1703 +/- 0.0030`
    - final epoch:
      - `strict_f1 @ eps=0 = 0.7778 +/- 0.0000`
      - `best_final_gap_strict_f1 = 0.0000`
      - `gt_margin_median = 0.1705 +/- 0.0027`
      - `final_diff_loss = 1.2422 +/- 0.0004`
  - `main_loss_weight = 1.0`
    - failure mode:
      - `weak_asymmetry` in `3/3` seeds
    - best epoch:
      - `best_strict_f1 @ eps=0 = 0.8519 +/- 0.0314`
      - `best_gt_margin_median = 0.1367 +/- 0.0072`
    - final epoch:
      - `strict_f1 @ eps=0 = 0.7778 +/- 0.0454`
      - `best_final_gap_strict_f1 = 0.0741 +/- 0.0262`
      - `gt_margin_median = 0.0277 +/- 0.0039`
      - `final_diff_loss = 1.1498 +/- 0.0117`

- Interpretation / Thoughts:
  - this pilot cleanly separates two effects:
    - diffusion improves the **best** reachable operating point
      - `0.7778 -> 0.8519`
    - but diffusion also reintroduces the known long-training retention problem
      - `best_final_gap: 0.0000 -> 0.0741`
      - `final gt_margin_median: 0.1705 -> 0.0277`
  - the no-diffusion branch is remarkably stable:
    - best and final are effectively identical
  - the full branch optimizes the denoising objective as expected:
    - lower `final_diff_loss`
    - but direction retention degrades by the final epoch

- Practical conclusion:
  - on current evidence, diffusion is **not** useless
    - it improves the best-epoch ceiling
  - but its contribution is not "autonomous direction discovery"
    - it is better described as a training signal that can temporarily sharpen
      the factorized direction branch
      while also destabilizing long-horizon retention
  - this strongly supports keeping Goal 4 as a first-order follow-up

### Experiment: Goal 4A selection-agreement audit on `sim4`

- Objective:
  - test whether adding Patel-agreement weight to guarded best-epoch selection
    changes the exported epoch or improves the reported best model

- Initial formal attempt:
  - target setup:
    - `sim4.csv`
    - seeds `11,22,33`
    - `selection_agreement_weight in {0.0, 0.25}`
  - result:
    - the `3`-seed run did not finish within the current `2h` execution window
    - no complete summary CSV was produced in that timed-out attempt

- Completed fallback run:
  - dataset:
    - `sim4.csv`
  - seeds:
    - `11`
  - branch:
    - same current-best factorized branch
    - `main_loss_weight = 1.0`
    - `selection_agreement_weight in {0.0, 0.25}`
  - epochs:
    - `100`

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260320_215347_goal4a_selection_audit_sim4_smoke1.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_1seeds_20260320_215347_goal4a_selection_audit_sim4_smoke1_aggregate.csv`

- Result summary (`seed=11`):
  - `selection_agreement_weight = 0.0`
    - `best_strict_f1 @ eps=0 = 0.8033`
    - `strict_f1 @ eps=0 = 0.7213`
    - `best_final_gap_strict_f1 = 0.0820`
    - `best_gt_margin_median = 0.0439`
    - `gt_margin_median = 0.0103`
    - exported epoch:
      - `9`
    - selection mode:
      - `score_only_fallback`
  - `selection_agreement_weight = 0.25`
    - `best_strict_f1 @ eps=0 = 0.7869`
    - `strict_f1 @ eps=0 = 0.7213`
    - `best_final_gap_strict_f1 = 0.0656`
    - `best_gt_margin_median = 0.0446`
    - `gt_margin_median = 0.0096`
    - exported epoch:
      - `9`
    - selection mode:
      - `score_only_fallback`

- Interpretation / Thoughts:
  - on this seed, changing `selection_agreement_weight` does **not** activate
    the guarded selector
    - both runs fall back to `score_only`
    - both export the same epoch (`9`)
  - increasing agreement weight changes the proxy score value, but not the
    selected checkpoint
  - the small reduction in best/final gap at `0.25` is offset by a slightly
    lower best strict F1

- Practical conclusion:
  - current evidence does **not** support Patel-agreement weight as the main
    fix for best-epoch stability
  - the more important follow-up still appears to be:
    - direction retention interventions
    - rather than simply reweighting the selection proxy
  - if Goal 4A is revisited formally, it should likely be:
    - shorter pilot first
    - or paired with lighter runtime settings

### Experiment: Goal 4B direction-retention formal follow-up on `sim4` (`3` seeds)

- Objective:
  - test whether simple retention interventions can preserve the strong
    best-epoch direction quality of the current full branch through the final
    epoch
  - compare the two most promising settings from the `seed=11` smoke run:
    - freeze direction updates after epoch `30`
    - freeze after epoch `30` plus lower direction learning-rate multiplier

- Setup:
  - dataset:
    - `sim4.csv`
  - seeds:
    - `11,22,33`
  - common branch:
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
    - `main_loss_weight = 1.0`
    - `selection_agreement_weight = 0.0`
  - treatments:
    - `direction_lr_multiplier = 1.0`, `freeze_direction_after_epoch = 30`
    - `direction_lr_multiplier = 0.3`, `freeze_direction_after_epoch = 30`
  - comparison baseline:
    - same-seed slice (`11,22,33`) from the earlier formal current-best run
    - baseline config:
      - `direction_lr_multiplier = 1.0`
      - `freeze_direction_after_epoch = -1`

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260321_140850_goal4b_retention_sim4_formal3_freeze30.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260321_140850_goal4b_retention_sim4_formal3_freeze30_aggregate.csv`
  - shell log:
    - `GraphExp/results/goal4b_retention_sim4_formal3_freeze30_shell.log`
  - historical same-seed baseline source:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260317_172003_sim4_support_direction_maxgap_formal.csv`

- Aggregate result summary:
  - historical same-seed baseline (`dir_lr_mult=1.0`, `freeze=-1`)
    - failure mode:
      - `symmetric_collapse` in `3/3` seeds
    - best epoch:
      - `best_strict_f1 @ eps=0 = 0.8033`
      - `best_strict_f1 @ eps=3e-4 = 0.8054`
      - `best_gt_margin_median = 0.0410`
    - final epoch:
      - `strict_f1 @ eps=0 = 0.7322`
      - `strict_f1 @ eps=3e-4 = 0.7272`
      - `best_final_gap_strict_f1 = 0.0710`
      - `best_final_gap_strict_f1 @ eps=3e-4 = 0.0783`
      - `gt_margin_median = 0.0099`
      - `best_final_gap_gt_margin_median = 0.0311`
      - `final_diff_loss = 1.0974`
  - freeze only (`dir_lr_mult=1.0`, `freeze=30`)
    - failure mode:
      - `symmetric_collapse` in `3/3` seeds
    - best epoch:
      - `best_strict_f1 @ eps=0 = 0.8142`
      - `best_strict_f1 @ eps=3e-4 = 0.8142`
      - `best_gt_margin_median = 0.0410`
    - final epoch:
      - `strict_f1 @ eps=0 = 0.7978`
      - `strict_f1 @ eps=3e-4 = 0.8011`
      - `best_final_gap_strict_f1 = 0.0164`
      - `best_final_gap_strict_f1 @ eps=3e-4 = 0.0131`
      - `gt_margin_median = 0.0124`
      - `best_final_gap_gt_margin_median = 0.0286`
      - `final_diff_loss = 1.0980`
  - low-LR plus freeze (`dir_lr_mult=0.3`, `freeze=30`)
    - failure mode:
      - `symmetric_collapse` in `3/3` seeds
    - best epoch:
      - `best_strict_f1 @ eps=0 = 0.7978`
      - `best_strict_f1 @ eps=3e-4 = 0.7879`
      - `best_gt_margin_median = 0.0193`
    - final epoch:
      - `strict_f1 @ eps=0 = 0.7869`
      - `strict_f1 @ eps=3e-4 = 0.7912`
      - `best_final_gap_strict_f1 = 0.0109`
      - `best_final_gap_strict_f1 @ eps=3e-4 = -0.0033`
      - `gt_margin_median = 0.0124`
      - `best_final_gap_gt_margin_median = 0.0068`
      - `final_diff_loss = 1.1143`

- Interpretation / Thoughts:
  - freezing direction updates after epoch `30` is the first intervention that
    clearly improves final retention on `sim4` without sacrificing best-epoch
    quality
    - versus the historical same-seed baseline:
      - `strict_f1 @ eps=0: 0.7322 -> 0.7978`
      - `best_final_gap_strict_f1: 0.0710 -> 0.0164`
  - adding lower direction LR on top of the freeze shrinks the best-final gap
    even further, but it does so by compressing the whole margin scale
    - `best_gt_margin_median: 0.0410 -> 0.0193`
    - `best_strict_f1 @ eps=0: 0.8142 -> 0.7978`
  - neither treatment resolves the deeper failure mode
    - all seeds still end in `symmetric_collapse`
    - final signed margins remain very small (`~0.0124`)
  - this means Goal 4B is a retention repair, not a mechanism repair
    - it keeps the factorized Patel-guided direction signal alive longer
    - it does not make diffusion become direction-discovering

- Practical conclusion:
  - under the current Option A framing, `freeze_direction_after_epoch = 30`
    should be treated as the default retention fix
  - among the tested settings, the best trade-off is:
    - keep `direction_lr_multiplier = 1.0`
    - freeze direction updates after epoch `30`
  - the lower-LR variant is useful if the sole target is minimizing best-final
    drift, but it is not the best headline setting because it also lowers the
    peak margin / strict-F1 operating point

### Experiment: current recommended retention-fix branch on `sim2` (`5` seeds)

- Objective:
  - verify whether the current Option A headline configuration also transfers to
    the remaining synthetic dataset
  - use the now-recommended retention setting directly:
    - `direction_lr_multiplier = 1.0`
    - `freeze_direction_after_epoch = 30`

- Setup:
  - dataset:
    - `sim2.csv`
  - seeds:
    - `11,22,33,44,55`
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
    - `strict_margin_eps in {0, 3e-4, 0.1}`

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260321_180424_sim2_support_direction_maxgap_retentionfreeze30_formal.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260321_180424_sim2_support_direction_maxgap_retentionfreeze30_formal_aggregate.csv`
  - shell log:
    - `GraphExp/results/goal_all_data_sim2_formal_shell.log`

- Aggregate result summary:
  - failure mode:
    - `mixed_or_partial` in `5/5` seeds
  - best epoch:
    - `best_strict_f1 @ eps=0 = 0.7818`
    - `best_strict_f1 @ eps=3e-4 = 0.7818`
    - `best_strict_f1 @ eps=0.1 = 0.7205`
    - `best_gt_margin_median = 0.1763`
  - final epoch:
    - `strict_f1 @ eps=0 = 0.8364`
    - `strict_f1 @ eps=3e-4 = 0.8364`
    - `strict_f1 @ eps=0.1 = 0.1441`
    - `best_final_gap_strict_f1 = -0.0545`
    - `gt_margin_median = 0.0496`
    - `best_final_gap_gt_margin_median = 0.1267`
    - `final_diff_loss = 1.1489`

- Interpretation / Thoughts:
  - `sim2` behaves differently from `sim3` / `sim4`
    - on the primary strict metric (`eps=0`), the final epoch is actually
      slightly better than the exported best checkpoint
  - this suggests the retention problem is not universal across the synthetic
    suite
    - it is concentrated on the harder benchmarks
  - however, `sim2` is not "solved perfectly"
    - the signed margin scale still shrinks substantially:
      - `0.1763 -> 0.0496`
    - the large deadzone metric also drops sharply:
      - `strict_f1 @ eps=0.1: 0.7205 -> 0.1441`
  - so the correct reading is:
    - the recommended branch transfers cleanly to `sim2`
    - but the final solution is still mostly low-margin rather than strongly
      separated

- Practical conclusion:
  - the current recommended Option A configuration is now verified on all three
    synthetic datasets:
    - `sim2`
    - `sim3`
    - `sim4`
  - the retention bottleneck remains a `sim3` / `sim4` story, not a full-suite
    story

### Experiment: full `fMRI.csv` formal evaluation with the current retention-fix branch (`5` seeds)

- Objective:
  - evaluate the current recommended Option A branch on the full
    synthetic `fMRI.csv` benchmark using the available directed reference graph
    - dataset: `fMRI.csv`
    - GT: `h1.txt`
  - determine whether the current retention fix also solves checkpoint
    selection on this small synthetic `fMRI.csv` setting

- Setup:
  - dataset:
    - `fMRI.csv`
    - `50` subjects
    - `5` nodes
    - `200` time points per subject
  - GT:
    - `h1.txt`
    - `5` directed edges / `5` undirected pairs
  - seeds:
    - `11,22,33,44,55`
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
    - `strict_margin_eps in {0, 3e-4, 0.1}`

- Artifacts:
  - summary:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260321_202949_fmri_support_direction_maxgap_retentionfreeze30_formal.csv`
  - aggregate:
    - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_5seeds_20260321_202949_fmri_support_direction_maxgap_retentionfreeze30_formal_aggregate.csv`
  - shell log:
    - `GraphExp/results/goal_all_data_fmri_formal_shell.log`
  - preliminary single-run diagnostics retained for inspection:
    - `GraphExp/results/goal_all_data_fmri_seed11_shell.log`
    - `GraphExp/results/goal_all_data_fmri_seed22_shell.log`
    - `GraphExp/results/goal_all_data_fmri_seed33_shell.log`

- Aggregate result summary:
  - failure mode:
    - `mixed_or_partial` in `5/5` seeds
  - exported / "best" checkpoint:
    - `best_strict_f1 @ eps=0 = 0.4000`
    - `best_strict_f1 @ eps=3e-4 = 0.4000`
    - `best_strict_f1 @ eps=0.1 = 0.2960`
    - `best_gt_margin_median = -0.0669`
  - final epoch:
    - `strict_f1 @ eps=0 = 0.7200`
    - `strict_f1 @ eps=3e-4 = 0.7200`
    - `strict_f1 @ eps=0.1 = 0.2286`
    - `best_final_gap_strict_f1 = -0.3200`
    - `gt_margin_median = 0.0375`
    - `best_final_gap_gt_margin_median = -0.1044`
    - `final_diff_loss = 1.1017`

- Interpretation / Thoughts:
  - `fMRI.csv` does have a usable GT, so the correct conclusion is stronger than
    the earlier diagnostic-only reading:
    - the branch reaches a **reasonable final direction score**
      - `strict_f1 @ eps=0 = 0.7200`
  - however, the real failure on this dataset is now very clear:
    - the exported checkpoint is dramatically worse than the final epoch
      - `best_strict_f1 @ eps=0 = 0.4000`
      - `final_strict_f1 @ eps=0 = 0.7200`
  - this is the opposite of the `sim3` / `sim4` retention story
    - on `fMRI.csv`, the problem is not that the final epoch collapses below a
      good early checkpoint
    - it is that the current checkpoint-selection proxy picks a bad early
      checkpoint and misses the later improvement
  - the negative exported margin median (`-0.0669`) reinforces that diagnosis:
    - the selected checkpoint can be directionally wrong even when the final
      epoch is directionally useful

- Practical conclusion:
  - the current recommended Option A branch now has full-benchmark evidence:
    - `sim2`, `sim3`, `sim4`, and `fMRI.csv` have all been run
  - but the bottleneck is now dataset-dependent:
    - `sim3` / `sim4`:
      - retention to the final epoch is the main issue
    - `fMRI.csv`:
      - checkpoint selection is the main issue
  - for the synthetic `fMRI.csv` benchmark, the defensible statement is:
    - the branch can reach `~0.72` strict F1 on `fMRI.csv`
    - but the current best-epoch selector is not yet trustworthy there
