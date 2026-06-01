# DDM-main Current Project Handoff

## Scope

This file is the current project handoff for future sessions. It replaces the older generic repository guidance.

Repository root:

- `D:\mockup\DDM-main`

Primary focus of current work:

- `GraphExp/main_structure_learning.py`
- `GraphExp/run_replay_saved_config.py`
- `GraphExp/models/DDM.py`
- `GraphExp/results/`
- `parameter_tuning.md`
- `GraphExp/results/strict_audit_report_20260511_093104.md`

The user is studying a temporal causal graph learning model and cares about strict, evidence-backed conclusions. Do not answer from memory when local files can be read.

## Core rules for future sessions

- Respond in Chinese unless the user explicitly asks otherwise.
- Style should be direct, engineering-focused, and evidence-based.
- Read local files before concluding anything about results, parameters, or past experiments.
- Use PowerShell commands and prefer `rg` for search.
- Manual file edits must use `apply_patch`.
- Never delete or reset user files unless explicitly asked.
- `GraphExp/results/` is generated output. Read it freely; do not treat it as source code.

## Project entrypoints

Main structure-learning entry:

- `GraphExp/main_structure_learning.py`

Replay existing configs across seeds:

- `GraphExp/run_replay_saved_config.py`

Key model file:

- `GraphExp/models/DDM.py`

Important datasets:

- `fMRI_dataset/fMRI.csv`
- `fMRI_dataset/h1.txt`
- `fMRI_dataset/sim2.csv`, `sim3.csv`, `sim4.csv`
- newly added `fMRI_dataset/sim8.csv`, `sim10.csv`, `sim11.csv`, `sim12.csv`, etc.

## Current understanding of the model

- The graph used by message passing is learned by the model, not fixed GT.
- In `support_direction` parameterization:
  - `node_emb_sender @ node_emb_receiver.T` produces support-like edge existence scores.
  - `direction_emb_sender / direction_emb_receiver` produce direction logits or gate asymmetry.
- Final learned adjacency is typically derived from support, direction gate, masks, priors, and selection/export policy.
- `support` is mainly about whether an edge exists.
- `direction` is mainly about choosing `i -> j` vs `j -> i`.

Patel-related understanding:

- Raw Patel is the default path: computed from raw `data_2d`.
- Encoded Patel exists via `--patel_input_source encoded` and requires `--prior_encoder_checkpoint`.
- Whether Patel should be computed before or after the temporal encoder is an empirical question, not something to answer theoretically without experiments.

Loss understanding:

- Main denoising loss is the core DDM reconstruction path.
- Causal-lag main loss reconstructs future from lagged parents using learned causal adjacency.
- Directional loss is an auxiliary direction-margin loss against a Patel-derived directional prior.
- L1 regularization continuously shrinks learned adjacency mass and can materially affect late-epoch degradation.

## Critical output semantics

When discussing a run, always distinguish:

- `best`: best GT-audited epoch within the run
- `exported`: what `learned_adjacency*.csv/.npy` currently export
- `final`: last training epoch

Never collapse these into one number.

Important CLI now supported:

- `--export_epoch_policy {selector,final}`
- `--diffusion_noise_mode {guided,gaussian_iid}`

Behavior:

- `selector`: `learned_adjacency*.csv/.npy` export selector-chosen epoch
- `final`: `learned_adjacency*.csv/.npy` export final epoch
- `guided`: use the existing neighbor-guided diffusion noise path
- `gaussian_iid`: use pure `eps ~ N(0,I)` diffusion noise; do not use `noise_guide_adj`, neighbor statistics, global statistics, signal-related mean bias, or extra layer/global normalization

Regardless of export policy, these files are also written:

- `selector_epoch_adjacency.*`
- `selector_epoch_adjacency_causal.*`
- `final_epoch_adjacency.*`
- `final_epoch_adjacency_causal.*`

GT audit semantics:

- `--selector_audit_gt_path` is only for auditing and diagnostics
- GT is never used for training or model selection logic except audit reporting

## Strict audit baseline

Primary strict audit files:

- `GraphExp/results/strict_all_run_audit_20260511_093104.csv`
- `GraphExp/results/strict_family_audit_20260511_093104.csv`
- `GraphExp/results/strict_audit_report_20260511_093104.md`

Audit coverage from the report:

- scanned run dirs: `1021`
- `selector_audit_summary.csv` present: `393`
- missing GT audit: `585`
- missing or bad config: `43`

Strict comparison policy:

- Prefer full-data runs with GT audit
- Prefer multi-seed family aggregates
- Prefer `run_count >= 5`
- Do not present single-seed maxima as stable conclusions

## Old strict family bests from the audit

These are the pre-tuning strict family baselines from the 2026-05-11 audit.

### fMRI

- final mean: `0.9200`
- exported mean: `0.8400`
- best mean: `0.9200`
- final strict F1 @ eps=0.1: `0.3976`
- family summary:
  - `structure_init_mode=random`
  - `support_prior_mode=patel_kappa`
  - `gradient_routing_mode=warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch=23`
  - `causal_lag_main_weight=0.25`
  - `directional_kappa_gate=True`
  - `selection_score_mode=causal_lag_composite`
  - `selection_soft_agreement_weight=0.03`
  - `selection_margin_penalty_weight=0.1`
  - `epochs=100`
  - `top_k_edges=5`

Representative run family:

- `run_20260412_203901`
- `run_20260412_204541`
- `run_20260412_205206`
- `run_20260412_205836`
- `run_20260412_210457`

### sim2

- final mean: `0.8545`
- exported mean: `0.8000`
- best mean: `0.8727`
- family summary:
  - `structure_init_mode=random`
  - `support_prior_mode=pearson_abs`
  - `gradient_routing_mode=warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch=23`
  - `causal_lag_main_weight=0.25`
  - `directional_kappa_gate=True`
  - `message_self_loop_weight=0.01`
  - `selection_score_mode=legacy`
  - `epochs=40`
  - `top_k_edges=11`

Representative base run:

- `run_20260420_090231`

### sim3

- final mean: `0.9111`
- exported mean: `0.8444`
- best mean: `0.9222`
- family summary:
  - `structure_init_mode=random`
  - `support_prior_mode=patel_kappa`
  - `gradient_routing_mode=warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch=23`
  - `causal_lag_main_weight=0.25`
  - `directional_kappa_gate=True`
  - `directional_loss_end_epoch=10`
  - `selection_score_mode=causal_lag_composite`
  - `epochs=100`
  - `top_k_edges=18`

Representative base run:

- `run_20260412_140034`

### sim4

- final mean: `0.8492`
- exported mean: `0.8492`
- best mean: `0.8820`
- family summary:
  - `structure_init_mode=random`
  - `support_prior_mode=patel_kappa`
  - `gradient_routing_mode=main_detach_only`
  - `detach_direction_from_main_after_epoch=0`
  - `causal_lag_main_weight=0.25`
  - `directional_kappa_gate=True`
  - `message_self_loop_weight=0.01`
  - `selection_score_mode=causal_lag_composite`
  - `epochs=100`
  - `top_k_edges=61`

Representative base run:

- `run_20260410_002543`

## Important single-seed fact that must not be overstated

There is a single-seed fMRI run with perfect scores:

- `GraphExp/results/run_20260330_155759`

This run has:

- best/exported/final strict F1 = `1.0 / 1.0 / 1.0`
- seed = `11`

Do not present this as stable multi-seed performance.

## Directional Precision/Recall/F1/SHD snapshot

These numbers are a separate audit view and are not the same as `strict_f1 @ eps=0.1`.

Evaluation convention:

- use saved causal adjacency snapshots (`learned_adjacency_causal.npy` / `final_epoch_adjacency_causal.npy`)
- predicted directed edges = top `|GT|` unordered pairs ranked by `abs(A[i, j] - A[j, i])`
- direction is chosen by the sign of `A[i, j] - A[j, i]`
- SHD is directed SHD: add/delete/reverse each cost `1`
- edge accuracy is computed over all directed off-diagonal entries
- for the older audit families, exact best-epoch SHD is unavailable because best-epoch adjacency snapshots were not exported

Final-epoch summary:

| dataset | chosen family | precision | recall | F1 | SHD | edge accuracy |
|---|---|---:|---:|---:|---:|---:|
| fMRI | old strict 5-seed best family | 0.9200 +/- 0.0980 | 0.9200 +/- 0.0980 | 0.9200 +/- 0.0980 | 0.40 +/- 0.49 | 0.9600 +/- 0.0490 |
| sim2 | old strict 5-seed best family | 0.8545 +/- 0.0445 | 0.8545 +/- 0.0445 | 0.8545 +/- 0.0445 | 1.60 +/- 0.49 | 0.9644 +/- 0.0109 |
| sim3 | old strict 5-seed best family | 0.9111 +/- 0.0444 | 0.9111 +/- 0.0444 | 0.9111 +/- 0.0444 | 1.60 +/- 0.80 | 0.9848 +/- 0.0076 |
| sim4 | old strict 5-seed best family | 0.8492 +/- 0.0066 | 0.8492 +/- 0.0066 | 0.8492 +/- 0.0066 | 9.20 +/- 0.40 | 0.9925 +/- 0.0003 |
| sim8 | `sim8_gt_5seed_repretrain` family | 0.8400 +/- 0.0800 | 0.8400 +/- 0.0800 | 0.8400 +/- 0.0800 | 0.8000 +/- 0.4000 | 0.9200 +/- 0.0400 |
| sim10 | `sim10_gt_5seed_repretrain` family | 0.8000 +/- 0.0000 | 0.8000 +/- 0.0000 | 0.8000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.9000 +/- 0.0000 |
| sim11 | current recommended `sim11_D3_lag035_5seed` family | 0.5818 +/- 0.0927 | 0.5818 +/- 0.0927 | 0.5818 +/- 0.0927 | 7.6000 +/- 1.7436 | 0.8978 +/- 0.0227 |
| sim12 | `sim12_gt_5seed_repretrain` family | 0.7273 +/- 0.0813 | 0.7273 +/- 0.0813 | 0.7273 +/- 0.0813 | 3.0000 +/- 0.8944 | 0.9333 +/- 0.0199 |

Source notes:

- `fMRI/sim2/sim3/sim4`: from `GraphExp/results/best_family_precision_recall_shd_20260512.md` and `GraphExp/results/best_family_precision_recall_shd_20260512.csv`
- `sim8/sim10/sim12`: recomputed on `2026-05-13` from the corresponding `unify_replay_*_5seed_repretrain.csv` run lists plus saved adjacency snapshots
- `sim11`: use `GraphExp/results/unify_replay_sim11_20260512_184811_sim11_D3_lag035_5seed.csv`, not the earlier collapsed `sim11_gt_5seed_repretrain` family
- `sim8/sim10/sim11/sim12`: exported metrics equal final metrics for the chosen families
- `fMRI/sim2/sim3`: in the older audit baseline, exported is worse than final; check `GraphExp/results/best_family_precision_recall_shd_20260512.md` when export-vs-final matters

## 2026-05-16 multi-seed best config index

This index was built by scanning local multi-seed aggregate files under `GraphExp/results/` plus `strict_family_audit_20260511_093104.csv`.

Selection rule:

- require `run_count >= 5`
- rank primarily by `final_primary_strict_f1_mean`
- tie-break by exported F1, best F1, then final strict F1 @ eps=0.1
- exclude 2-seed tuning probes from this formal index
- treat these as the best currently found multi-seed configurations, not proof of global optimum

Directional SHD / edge accuracy computation for this index:

- load each run's final causal adjacency snapshot, preferring `final_epoch_adjacency_causal.npy/.csv`; if absent, fall back to `learned_adjacency_causal.npy/.csv`
- load GT directed edges from `fMRI_dataset/h*.txt` and convert 1-based node ids to 0-based ids
- predicted directed edge set has exactly `|GT|` edges
- for every unordered pair `{i,j}`, score it by `abs(A[i,j] - A[j,i])`
- choose the top `|GT|` unordered pairs by that score
- orient each chosen pair by the sign of `A[i,j] - A[j,i]`; non-negative means `i -> j`, negative means `j -> i`
- TP/FP/FN are computed by exact directed-edge set comparison against GT
- directed SHD counts one operation for each reversal, addition, or deletion: first match predicted reversed edges to GT reversed edges as reversals, then add remaining FP plus remaining FN
- edge accuracy is the fraction of all directed off-diagonal entries `(i,j), i != j` whose predicted present/absent label matches GT
- values below are mean/std over the 5 seeds

Mainline / non-ablation best found configurations:

| dataset | aggregate/source file | representative config file | seeds | best | exported | final | final eps=0.1 | final SHD | final edge acc | key override / family |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| fMRI | `GraphExp/results/unify_replay_fMRI_20260511_124620_fmri_F3_lag035_final_export_5seed_aggregate.csv` | `GraphExp/results/run_20260511_124622/config.npy` | `11,22,33,44,55` | 0.9600 | 0.9600 | 0.9600 | 0.4452 | 0.2000 +/- 0.4000 | 0.9800 +/- 0.0400 | `export_epoch_policy=final; causal_lag_main_weight=0.35` |
| sim2 | `GraphExp/results/unify_replay_sim2_20260420_090228_sim2_2x2_incumbent_selfloop_alpha001_aggregate.csv` | `GraphExp/results/run_20260420_090231/config.npy` | `11,22,33,44,55` | 0.8727 | 0.8000 | 0.8545 | 0.1256 | 1.6000 +/- 0.4899 | 0.9644 +/- 0.0109 | `structure_message_edge_mode=full; message_self_loop_weight=0.01`; tied with strict audit row 22 |
| sim3 | `GraphExp/results/unify_replay_sim3_20260420_sim3_dirend10_epochs100_aggregate.csv` | `GraphExp/results/run_20260420_152306/config.npy` | `11,22,33,44,55` | 0.9222 | 0.8444 | 0.9111 | 0.4100 | 1.6000 +/- 0.8000 | 0.9848 +/- 0.0076 | `epochs=100`; strict audit row 34 equivalent |
| sim4 | `GraphExp/results/unify_replay_sim4_20260420_sim4_l15a_low_epochs100_aggregate.csv` | `GraphExp/results/run_20260420_175556/config.npy` | `11,22,33,44,55` | 0.8820 | 0.8492 | 0.8492 | 0.0000 | 9.2000 +/- 0.4000 | 0.9925 +/- 0.0003 | `epochs=100`; strict audit row 66 equivalent |
| sim8 | `GraphExp/results/unify_replay_sim8_20260512_091903_sim8_gt_5seed_repretrain_aggregate.csv` | `GraphExp/results/run_20260512_091906/config.npy` | `11,22,33,44,55` | 0.8800 | 0.8400 | 0.8400 | 0.1500 | 0.8000 +/- 0.4000 | 0.9200 +/- 0.0400 | `top_k_edges=5; selection_top_k=5; pretrain_epochs=50; export_epoch_policy=final` |
| sim10 | `GraphExp/results/unify_replay_sim10_20260512_095726_sim10_gt_5seed_repretrain_aggregate.csv` | `GraphExp/results/run_20260512_095729/config.npy` | `11,22,33,44,55` | 0.8400 | 0.8000 | 0.8000 | 0.1333 | 1.0000 +/- 0.0000 | 0.9000 +/- 0.0000 | `top_k_edges=5; selection_top_k=5; pretrain_epochs=50; export_epoch_policy=final` |
| sim11 | `GraphExp/results/unify_replay_sim11_20260512_184811_sim11_D3_lag035_5seed_aggregate.csv` | `GraphExp/results/run_20260512_184814/config.npy` | `11,22,33,44,55` | 0.6370 | 0.5926 | 0.5926 | 0.0750 | 7.6000 +/- 1.7436 | 0.8978 +/- 0.0227 | `fixed_support_mask_mode=topk_kappa; top_k_edges=16; selection_top_k=11; causal_lag_main_weight=0.35; export_epoch_policy=final` |
| sim12 | `GraphExp/results/unify_replay_sim12_20260512_111301_sim12_gt_5seed_repretrain_aggregate.csv` | `GraphExp/results/run_20260512_111304/config.npy` | `11,22,33,44,55` | 0.7818 | 0.7273 | 0.7273 | 0.2056 | 3.0000 +/- 0.8944 | 0.9333 +/- 0.0199 | `top_k_edges=11; selection_top_k=11; pretrain_epochs=50; export_epoch_policy=final` |

If ablation branches are allowed in the search space, the best-by-final-F1 file changes for some datasets:

| dataset | best including ablations | representative config file | best | exported | final | final eps=0.1 | final SHD | final edge acc | note |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| fMRI | `GraphExp/results/unify_replay_fMRI_20260513_164419_fmri_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/run_20260513_164422/config.npy` | 0.9600 | 0.9600 | 0.9600 | 0.4643 | 0.2000 +/- 0.4000 | 0.9800 +/- 0.0400 | ties mainline final F1; ablation branch, not default recommendation |
| sim3 | `GraphExp/results/unify_replay_sim3_20260513_sim3_ablation_gaussian_iid_5seed_combined_aggregate.csv` | `GraphExp/results/run_20260513_193512/config.npy` | 0.9556 | 0.9222 | 0.9222 | 0.5331 | 1.4000 +/- 1.0198 | 0.9867 +/- 0.0097 | gaussian IID ablation exceeds current mainline on this dataset |
| sim8 | `GraphExp/results/unify_replay_sim8_20260514_022611_sim8_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/run_20260514_022614/config.npy` | 0.8800 | 0.8400 | 0.8400 | 0.2643 | 0.8000 +/- 0.4000 | 0.9200 +/- 0.0400 | ties mainline final F1; ablation branch |
| sim10 | `GraphExp/results/unify_replay_sim10_20260514_032112_sim10_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/run_20260514_032115/config.npy` | 0.8800 | 0.8400 | 0.8400 | 0.1143 | 0.8000 +/- 0.4000 | 0.9200 +/- 0.0400 | disable-encoder ablation exceeds current mainline final F1 |
| sim11 | `GraphExp/results/unify_replay_sim11_20260514_050543_sim11_ablation_gaussian_iid_5seed_aggregate.csv` | `GraphExp/results/run_20260514_050546/config.npy` | 0.6370 | 0.5926 | 0.5926 | 0.0800 | 9.2000 +/- 1.1662 | 0.8756 +/- 0.0109 | ties mainline final F1 but worse SHD/edge accuracy; ablation branch |
| sim12 | `GraphExp/results/unify_replay_sim12_20260514_042044_sim12_ablation_gaussian_iid_5seed_aggregate.csv` | `GraphExp/results/run_20260514_042047/config.npy` | 0.8000 | 0.7455 | 0.7455 | 0.1749 | 2.8000 +/- 0.7483 | 0.9378 +/- 0.0166 | gaussian IID ablation exceeds current mainline final F1 |

Interpretation:

- Use the mainline / non-ablation table when discussing recommended model configurations.
- Use the ablation-inclusive table only when explicitly allowing mechanism-ablation branches into model selection.
- `sim3` and `sim12` gaussian IID, and `sim10` disable-encoder, are positive follow-up candidates, but they should not be silently promoted to default settings without a targeted confirmation run.

## 2026-05-14 ablation session: encoder and diffusion noise

This session completed the requested single-factor ablations on:

- `fMRI`, `sim2`, `sim3`, `sim4`, `sim8`, `sim10`, `sim11`, `sim12`

Common protocol:

- seeds: `11,22,33,44,55`
- use each dataset's current recommended baseline family
- keep `export_epoch_policy=final`
- do not retune other training-family parameters inside each ablation
- evaluate directional Precision/Recall/F1/SHD on saved final causal adjacency snapshots using the same top `|GT|` pair convention as the directional audit
- for these ablation runs, `exported` and `final` metrics are identical because `export_epoch_policy=final`

Implemented code / analysis files from this session:

- `GraphExp/models/DDM.py`
- `GraphExp/main_structure_learning.py`
- `GraphExp/run_replay_saved_config.py`
- `GraphExp/aggregate_existing_replay_runs.py`
- `GraphExp/evaluate_ablation_precision_recall_shd.py`

Key result files:

- `GraphExp/results/ablation_precision_recall_shd_20260514.csv`
- `GraphExp/results/ablation_precision_recall_shd_20260514_aggregate.csv`
- `GraphExp/results/ablation_precision_recall_shd_20260514.md`
- `GraphExp/results/ablation_comparison_tables_20260514.tex`

Formal ablation aggregate files:

- fMRI disable encoder: `GraphExp/results/unify_replay_fMRI_20260513_173101_fmri_ablation_disable_encoder_5seed_v2_aggregate.csv`
- fMRI gaussian IID noise: `GraphExp/results/unify_replay_fMRI_20260513_165348_fmri_ablation_gaussian_iid_5seed_aggregate.csv`
- sim2 disable encoder: `GraphExp/results/unify_replay_sim2_20260513_185047_sim2_ablation_disable_encoder_5seed_aggregate.csv`
- sim2 gaussian IID noise: `GraphExp/results/unify_replay_sim2_20260513_185725_sim2_ablation_gaussian_iid_5seed_aggregate.csv`
- sim3 disable encoder: `GraphExp/results/unify_replay_sim3_20260513_192354_sim3_ablation_disable_encoder_5seed_aggregate.csv`
- sim3 gaussian IID noise: `GraphExp/results/unify_replay_sim3_20260513_sim3_ablation_gaussian_iid_5seed_combined_aggregate.csv`
- sim4 disable encoder: `GraphExp/results/unify_replay_sim4_20260513_211344_sim4_ablation_disable_encoder_5seed_aggregate.csv`
- sim4 gaussian IID noise: `GraphExp/results/unify_replay_sim4_20260514_sim4_ablation_gaussian_iid_5seed_combined_aggregate.csv`
- sim8 disable encoder: `GraphExp/results/unify_replay_sim8_20260514_022611_sim8_ablation_disable_encoder_5seed_aggregate.csv`
- sim8 gaussian IID noise: `GraphExp/results/unify_replay_sim8_20260514_023604_sim8_ablation_gaussian_iid_5seed_aggregate.csv`
- sim10 disable encoder: `GraphExp/results/unify_replay_sim10_20260514_032112_sim10_ablation_disable_encoder_5seed_aggregate.csv`
- sim10 gaussian IID noise: `GraphExp/results/unify_replay_sim10_20260514_033111_sim10_ablation_gaussian_iid_5seed_aggregate.csv`
- sim11 disable encoder: `GraphExp/results/unify_replay_sim11_20260514_050123_sim11_ablation_disable_encoder_5seed_aggregate.csv`
- sim11 gaussian IID noise: `GraphExp/results/unify_replay_sim11_20260514_050543_sim11_ablation_gaussian_iid_5seed_aggregate.csv`
- sim12 disable encoder: `GraphExp/results/unify_replay_sim12_20260514_041620_sim12_ablation_disable_encoder_5seed_aggregate.csv`
- sim12 gaussian IID noise: `GraphExp/results/unify_replay_sim12_20260514_042044_sim12_ablation_gaussian_iid_5seed_aggregate.csv`

Final directional F1 and SHD deltas versus baseline:

| dataset | disable encoder Delta F1 | disable encoder Delta SHD | gaussian IID Delta F1 | gaussian IID Delta SHD |
|---|---:|---:|---:|---:|
| fMRI | +0.0000 | +0.00 | -0.1200 | +0.60 |
| sim2 | -0.0909 | +1.00 | -0.0182 | +0.20 |
| sim3 | -0.0778 | +1.40 | +0.0111 | -0.20 |
| sim4 | -0.0525 | +3.20 | -0.0197 | +1.20 |
| sim8 | +0.0000 | +0.00 | -0.0800 | +0.40 |
| sim10 | +0.0400 | -0.20 | +0.0000 | +0.00 |
| sim11 | -0.0727 | +1.60 | -0.0909 | +1.60 |
| sim12 | -0.0364 | +0.40 | +0.0182 | -0.20 |

Important interpretation:

- The fMRI conclusion does not generalize globally. On fMRI, disabling the encoder is neutral under the current final-export recommendation, while gaussian IID noise is clearly worse.
- Disabling the temporal encoder is harmful on `sim2/sim3/sim4/sim11/sim12`, neutral on `fMRI/sim8`, and slightly positive on `sim10`.
- Replacing guided diffusion noise with pure `N(0,I)` is harmful on `fMRI/sim8/sim11`, slightly harmful on `sim2/sim4`, neutral on `sim10`, and positive on `sim3/sim12`.
- `sim3` with gaussian IID noise is the strongest positive follow-up candidate, but it should be treated as dataset-specific rather than a global default.
- `sim12` with gaussian IID noise is mildly positive on final directional F1/SHD, but the evidence is weaker and should be followed up cautiously.
- `sim4` gaussian IID may improve some best-epoch behavior, but final directional F1/SHD is worse than baseline; do not recommend it without a best-vs-final degradation analysis.
- Full LaTeX comparison tables for report/paper use are in `GraphExp/results/ablation_comparison_tables_20260514.tex`; they contain two tables: baseline vs disable encoder and baseline vs gaussian IID noise.

## Current tuned recommendations

These are newer than the strict audit report and should be treated as the current best known practical settings.

### fMRI current recommendation

Source of record:

- `parameter_tuning.md`
- `GraphExp/results/unify_replay_fMRI_20260511_124620_fmri_F3_lag035_final_export_5seed_aggregate.csv`

Recommended family:

- base family from `run_20260412_203901`
- `export_epoch_policy=final`
- `causal_lag_main_weight=0.35`
- all other major settings follow the old best family

Observed 5-seed result:

- best mean: `0.9600`
- exported mean: `0.9600`
- final mean: `0.9600`
- final strict F1 @ eps=0.1: `0.4452`

Interpretation:

- better than old strict family on main strict F1 and exported/final alignment
- margin median is not uniformly stronger, so do not overclaim

### Newly added GT datasets already audited

5-seed replay with re-pretraining was completed for:

- `sim8`
- `sim10`
- `sim11`
- `sim12`

Key aggregate files:

- `GraphExp/results/unify_replay_sim8_20260512_091903_sim8_gt_5seed_repretrain_aggregate.csv`
- `GraphExp/results/unify_replay_sim10_20260512_095726_sim10_gt_5seed_repretrain_aggregate.csv`
- `GraphExp/results/unify_replay_sim11_20260512_103852_sim11_gt_5seed_repretrain_aggregate.csv`
- `GraphExp/results/unify_replay_sim12_20260512_111301_sim12_gt_5seed_repretrain_aggregate.csv`

Summary:

- `sim8`: workable, final mean `0.8400`, high-margin metric weak
- `sim10`: workable, final mean `0.8000`, high-margin metric weak
- `sim12`: workable, final mean `0.7273`, but late-epoch degradation exists
- `sim11`: required extra debugging and retuning; see separate section below

## sim11: current state and latest conclusion

This is the dataset that required the most work. Read this section before continuing any `sim11` experiment.

### What went wrong initially

Initial 5-seed re-pretrained run:

- `GraphExp/results/unify_replay_sim11_20260512_103852_sim11_gt_5seed_repretrain_aggregate.csv`

Observed:

- final mean: `0.0333`
- failure mode: `symmetric_collapse`

Root cause:

- with `support_prior_mode=pearson_abs`
- and `fixed_support_mask_mode=maxgap_kappa`
- the max-gap skeleton on `sim11` collapsed to only `1` undirected pair

This was confirmed from logs:

- `Noise guide adj: 1 undirected pairs`
- `Fixed support mask: ... undirected_pairs=1`

This is not an encoder failure. Re-pretraining worked correctly. The support space was simply too narrow.

### The key repair

The support-collapse fix was:

- `fixed_support_mask_mode=topk_kappa`
- `top_k_edges=16`
- `selection_top_k=11`

Reason:

- both `pearson_abs` and `patel_kappa` top-16 sets cover all `11` GT undirected edges on `sim11`
- the previous `maxgap` setting did not

Important result:

- under `topk16`, `pearson_abs` and `patel_kappa` produce the exact same top-16 undirected pair set on `sim11`
- therefore changing `support_prior_mode` between those two does not matter under the current `topk16` setup

### Current sim11 recommended configuration

Current best practical recommendation for `sim11`:

- `support_prior_mode=pearson_abs`
- `fixed_support_mask_mode=topk_kappa`
- `top_k_edges=16`
- `selection_top_k=11`
- `causal_lag_main_weight=0.35`
- `export_epoch_policy=final`
- `pretrain_checkpoint=`
- `pretrain_epochs=50`

Supporting result file:

- `GraphExp/results/unify_replay_sim11_20260512_184811_sim11_D3_lag035_5seed_aggregate.csv`

Current aggregate:

- best mean: `0.6370`
- exported mean: `0.5926`
- final mean: `0.5926`
- final strict F1 @ eps=0.1: `0.0750`
- final vs best gap mean: `-0.0444`

Interpretation:

- this is better than the earlier `topk16` baseline on exported/final mean and on best-final gap
- it does not improve best mean, but it reduces late-epoch degradation
- this is currently the best-known `sim11` training-dynamics direction

### sim11 experiments already done and what they mean

Support-collapse repair:

- `GraphExp/results/unify_replay_sim11_20260512_142013_sim11_topk16_repretrain_aggregate.csv`
- fixed the catastrophic collapse

Selector export comparison:

- `GraphExp/results/unify_replay_sim11_20260512_153711_sim11_topk16_selector_export_aggregate.csv`
- conclusion: `selector` export is worse than `final` here
- reason: selector-chosen epochs did not align with GT-best epochs

Lower L1:

- `GraphExp/results/unify_replay_sim11_20260512_165805_sim11_D1_l1_001_5seed_aggregate.csv`
- conclusion: weak positive on final margin/support, but no main F1 gain

Later detach:

- probe only
- conclusion: not worth continuing

Causal-lag weight scan:

- `lag=0.30`: worse than `0.35`
- `lag=0.35`: current best tradeoff
- `lag=0.40`: roughly tied with `0.35` on 2-seed main outcome, no clear advantage

Patel support-prior comparison under topk16:

- `GraphExp/results/unify_replay_sim11_20260512_211716_sim11_D7_patel_support_probe_aggregate.csv`
- conclusion: no difference from `pearson_abs`, because top-16 pair sets are identical

### sim11 next-step guidance

If future work continues on `sim11`, do not repeat already-settled paths:

- do not go back to `maxgap_kappa`
- do not re-run `support_prior_mode=patel_kappa` under the same `topk16` setup
- do not assume `selector` export is better

More useful remaining directions would be:

- change `selection_score_mode`
- change `selection_*` weights
- change `top_k_edges` itself, such as `14` or `18`

## Re-pretraining policy for new datasets

When replaying onto a newly added dataset, do not silently reuse the old encoder checkpoint unless the purpose is explicitly transfer testing.

For strict comparisons, use:

- `--set pretrain_checkpoint=`
- `--set pretrain_epochs=50`

This forces per-dataset temporal encoder pretraining.

Confirmed from current experiments:

- new dataset strict replays used `pretrain_checkpoint=None`
- logs showed `开始时间因果编码器的自回归预训练 (50 Epochs)`

## Known common parameters across old best families

Across the older best families for fMRI, sim2, sim3, sim4, these parameters were common:

- `pretrain_epochs=50`
- `skip_pretrain=False`
- `disable_temporal_encoder=False`
- `structure_parameterization=support_direction`
- `structure_init_mode=random`
- `structure_init_scale=0.5`
- `fixed_support_mask_mode=maxgap_kappa`
- `direction_init_mode=random`
- `optimizer_step_mode=subject`
- `support_prior_algorithm=patel`
- `direction_prior_algorithm=patel`
- `directional_kappa_gate=True`
- `disable_directional_loss=False`
- `causal_lag_main_lags=1,2`

These should not be treated as universally optimal for new datasets. They are just common among the old winners.

## Historical document digest

The following historical documents were reviewed and should be treated as experiment lineage, not as the sole source of current best settings:

- `unify.md`
- `plan.md`
- `encoded_patel_ablation.md`
- `constrict.md`
- `ablation.md`
- `GraphExp/TEST_RESULTS.md`
- `GraphExp/README_DISABLE_ENCODER.md`

How to use them:

- use them to understand why branches were opened, kept, or dropped
- do not let older single-seed or pre-strict conclusions override May 2026 strict multi-seed audit results
- for current best settings, trust:
  - latest relevant aggregate CSV under `GraphExp/results/`
  - `parameter_tuning.md`
  - `strict_audit_report_20260511_093104.md`

### Historical methodology that still stands

- training factors, direction-teacher factors, and selector factors must be separated; do not bundle many changes and then claim a mechanism conclusion
- always distinguish `best`, `exported`, and `final`
- selector-only rescoring can materially change exported quality; do not confuse selector mismatch with training ceiling
- keep the terminology boundary:
  - `causal_lag_main` = task-level lagged predictive objective
  - `lag_corr` / `lag_gain`-style branch = teacher / prior proposal, not the same thing

### Historical mechanism conclusions still worth keeping

From `plan.md` and `constrict.md`:

- the denoising/diffusion path is not a reliable direction learner by itself
- late denoising gradients often hurt direction retention; this is why routed training variants such as `warmup_then_orthogonal` and later `main_detach_only` were explored
- old Patel-free direction-teacher replacements did not beat the Patel-assisted branch on the original four datasets
- the most honest architecture read from that period is:
  - diffusion/support path mainly stabilizes support or support reweighting
  - temporal / auxiliary objectives carry most of the direction-learning burden

From `unify.md`:

- `main_detach_only` and `message_self_loop_weight` were historically useful mainly on harder synthetic regimes like `sim3` / `sim4`
- the same family did not transfer cleanly to `sim2` and `fMRI`
- this regime split is still compatible with the later strict audit:
  - current strict best family uses `main_detach_only` only on `sim4`
  - `fMRI/sim2/sim3` still prefer other routed families in the strict audit
- therefore do not assume one routing family is globally best across datasets

From `ablation.md`:

- under the old factorized support branch, `fixed_support_mask_mode = maxgap_kappa` was doing real work; removing the hard support mask on `sim4` collapsed support badly
- `structure_init_mode = patel_kappa` looked largely redundant relative to `random` on that line
- Patel vs Pearson support-prior comparisons were often empty under `maxgap` or nearby support-selection rules because both priors were projected to the same selected skeleton
- the bottleneck there was often the support-selection operator, not necessarily the prior source itself

### Encoded Patel line

From `encoded_patel_ablation.md`:

- this was a focused `sim4` 3-seed pilot, not a full strict multi-dataset confirmation
- `patel_input_source=encoded` with `encoded_patel_scope=support_only` was mildly positive relative to raw Patel on that pilot
- replacing direction tau with encoded tau (`encoded_patel_scope=support_and_direction`) was clearly harmful
- current safe interpretation:
  - encoded support + raw direction is an allowed experimental branch
  - encoded direction prior is not supported as a default

### Temporal encoder / disable-encoder feature status

From `GraphExp/TEST_RESULTS.md` and `GraphExp/README_DISABLE_ENCODER.md`:

- `--disable_temporal_encoder` is a real, working feature
- those documents mainly establish feature behavior and early engineering smoke coverage
- they are not authoritative for current model-selection decisions because:
  - the comparison there is short-horizon (`10` epochs) and early-stage
  - later strict winners across the main datasets all keep the temporal encoder enabled with pretraining on
- current practical read:
  - treat `disable_temporal_encoder` as a feature flag for controlled comparison or debugging
  - do not treat it as the current default recommendation

### Old branches that are now low priority or superseded

- old single-seed maxima in `unify.md` / `constrict.md` must not override strict family aggregates
- old "unified family" ambitions were superseded by later evidence that different datasets prefer different routing/support settings
- old `lag_corr` teacher line was removed from the main program as a retained mainline path
- current code still contains `lag_gain` / soft-prior options, but historical 4-dataset evidence in `constrict.md` says `lag_gain` was falsified as a useful direction teacher under the present `T=200` setting
- `training_noise_guide_mode=scheduled_blend` remains an available experiment knob, but historical evidence did not justify promoting it to the default mainline

### Current code-status cross-check

Verified in current code:

- still available:
  - `--disable_temporal_encoder`
  - `--export_epoch_policy`
  - `--patel_input_source`
  - `--encoded_patel_scope`
  - `--prior_encoder_checkpoint`
  - `--gradient_routing_mode`
  - `--structure_message_edge_mode`
  - `--message_self_loop_weight`
  - `--training_noise_guide_mode`
  - `--diffusion_noise_mode`
  - `--support_prior_mode`
  - `--fixed_support_mask_mode`
  - `--direction_prior_algorithm lag_gain`
- no longer present as active main-program branches:
  - `lag_corr`
  - `directed_noise`
  - `post_detach_direction`

## Files that matter most for current project state

Read these first in any new session:

- `AGENTS.md`
- `parameter_tuning.md`
- `GraphExp/results/strict_audit_report_20260511_093104.md`
- `GraphExp/results/strict_family_audit_20260511_093104.csv`
- `GraphExp/results/ablation_precision_recall_shd_20260514_aggregate.csv`
- `GraphExp/results/ablation_comparison_tables_20260514.tex`
- `GraphExp/main_structure_learning.py`

Read next if you need historical experiment lineage or branch rationale:

- `unify.md`
- `ablation.md`
- `encoded_patel_ablation.md`
- `constrict.md`
- `plan.md`
- `GraphExp/TEST_RESULTS.md`
- `GraphExp/README_DISABLE_ENCODER.md`

When the user asks about current best settings, also read the most recent relevant aggregate CSV under `GraphExp/results/`.

## Safe command patterns

Useful replay pattern:

```powershell
cd D:\mockup\DDM-main\GraphExp
python .\run_replay_saved_config.py --base_run_dir <run_dir> --seeds 11,22,33,44,55 --tag <tag> --set key=value ...
```

Typical strict replay overrides for new GT datasets:

```powershell
--set csv_path=..\fMRI_dataset\<dataset>.csv
--set selector_audit_gt_path=..\fMRI_dataset\<gt>.txt
--set pretrain_checkpoint=
--set pretrain_epochs=50
```

## Documentation status

Current session-level experiment history has been consolidated into:

- `parameter_tuning.md`

This file contains:

- fMRI export-policy and causal-lag tuning
- new GT dataset replays
- sim11 failure analysis
- sim11 support-collapse repair
- sim11 selector-export comparison
- sim11 L1 / detach / causal-lag tuning
- sim11 patel-support comparison
- fMRI single-factor ablations: disable temporal encoder and gaussian IID diffusion noise
- other-dataset single-factor ablations on `sim2/sim3/sim4/sim8/sim10/sim11/sim12`
- LaTeX comparison tables are stored separately in `GraphExp/results/ablation_comparison_tables_20260514.tex`

When in doubt, trust the latest aggregate CSV plus `parameter_tuning.md` over older ad hoc notes.
