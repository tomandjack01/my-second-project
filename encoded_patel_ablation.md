# Encoded Patel Ablation

## 2026-05-04 Setup

Purpose: compare raw Patel against frozen-encoder Patel on `sim4` while keeping the incumbent training backbone, selector, GT audit, and seeds fixed.

Base run:

- `GraphExp/results/run_20260404_111017`
- dataset: `..\fMRI_dataset\sim4.csv`
- GT: `..\fMRI_dataset\h4.txt`
- prior encoder checkpoint: `GraphExp/results/run_20260310_185625/pretrained_encoder.pt`

Variants:

| variant | `patel_input_source` | `encoded_patel_scope` | intended prior use |
|---|---|---|---|
| `V0_raw_patel` | `raw` | `support_only` | raw score/kappa/tau |
| `V1_encoded_support_raw_direction` | `encoded` | `support_only` | encoded score/kappa, raw tau |
| `V2_encoded_support_encoded_direction` | `encoded` | `support_and_direction` | encoded score/kappa/tau |

Smoke command:

```powershell
cd GraphExp
python .\run_replay_saved_config.py `
  --base_run_dir .\results\run_20260404_111017 `
  --seeds 11 `
  --device cpu `
  --tag encoded_patel_smoke_v1 `
  --set epochs=1 `
  --set subject_limit=2 `
  --set time_limit=20 `
  --set skip_pretrain=true `
  --set pretrain_checkpoint= `
  --set patel_input_source=encoded `
  --set encoded_patel_scope=support_only `
  --set prior_encoder_checkpoint=.\results\run_20260310_185625\pretrained_encoder.pt
```

Formal pilot commands:

```powershell
cd GraphExp
python .\run_replay_saved_config.py --base_run_dir .\results\run_20260404_111017 --seeds 11,22,33 --tag V0_raw_patel --set patel_input_source=raw
python .\run_replay_saved_config.py --base_run_dir .\results\run_20260404_111017 --seeds 11,22,33 --tag V1_encoded_support_raw_direction --set patel_input_source=encoded --set encoded_patel_scope=support_only --set prior_encoder_checkpoint=.\results\run_20260310_185625\pretrained_encoder.pt
python .\run_replay_saved_config.py --base_run_dir .\results\run_20260404_111017 --seeds 11,22,33 --tag V2_encoded_support_encoded_direction --set patel_input_source=encoded --set encoded_patel_scope=support_and_direction --set prior_encoder_checkpoint=.\results\run_20260310_185625\pretrained_encoder.pt
```

Metrics to record:

- `best/exported/final_primary_strict_f1`
- `strict_f1@eps=0.1`
- `failure_mode`
- `gt_signed_margin_median`
- support/gate/exported margin
- exported/final retention gaps

## 2026-05-04 Smoke

Smoke artifacts:

- `GraphExp/results/unify_replay_sim4_20260504_151119_encoded_patel_smoke_v1.csv`
- `GraphExp/results/unify_replay_sim4_20260504_151119_encoded_patel_smoke_v1_aggregate.csv`
- `GraphExp/results/unify_replay_sim4_20260504_151206_encoded_patel_smoke_v2.csv`
- `GraphExp/results/unify_replay_sim4_20260504_151206_encoded_patel_smoke_v2_aggregate.csv`

Result: both encoded scopes loaded `GraphExp/results/run_20260310_185625/pretrained_encoder.pt`, generated `encoded_patel_score/kappa/tau` artifacts, saved prior-source config fields, and completed 1 epoch end to end on CPU with `subject_limit=2`, `time_limit=20`.

The smoke intentionally used `skip_pretrain=true` and an empty `pretrain_checkpoint` override because the training encoder checkpoint is 200-step, while this smoke truncates to 20 time points. The prior encoder still loaded for encoded Patel.

## 2026-05-04 Formal 3-Seed Pilot

Summary artifacts:

- `V0_raw_patel`
  - `GraphExp/results/unify_replay_sim4_20260504_164907_V0_raw_patel.csv`
  - `GraphExp/results/unify_replay_sim4_20260504_164907_V0_raw_patel_aggregate.csv`
- `V1_encoded_support_raw_direction`
  - `GraphExp/results/unify_replay_sim4_20260504_175105_V1_encoded_support_raw_direction.csv`
  - `GraphExp/results/unify_replay_sim4_20260504_175105_V1_encoded_support_raw_direction_aggregate.csv`
- `V2_encoded_support_encoded_direction`
  - `GraphExp/results/unify_replay_sim4_20260504_185531_V2_encoded_support_encoded_direction.csv`
  - `GraphExp/results/unify_replay_sim4_20260504_185531_V2_encoded_support_encoded_direction_aggregate.csv`

Aggregate results, mean over seeds `11,22,33`:

| variant | best F1 | exported F1 | final F1 | exported-best gap | final-best gap | final strict@0.1 | final signed margin | final gate margin | final support med | final exported margin | final failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `V0_raw_patel` | 0.8470 | 0.8087 | 0.8142 | -0.0383 | -0.0328 | 0.0317 | 0.00746 | 0.67631 | 0.01593 | 0.00746 | `symmetric_collapse=3/3` |
| `V1_encoded_support_raw_direction` | 0.8525 | 0.8251 | 0.8361 | -0.0273 | -0.0164 | 0.0418 | 0.00724 | 0.66023 | 0.01290 | 0.00724 | `symmetric_collapse=3/3` |
| `V2_encoded_support_encoded_direction` | 0.7978 | 0.7541 | 0.7650 | -0.0437 | -0.0328 | 0.0108 | 0.00724 | 0.71736 | 0.01356 | 0.00724 | `symmetric_collapse=3/3` |

Per-seed primary F1:

| variant | seed | best | exported | final |
|---|---:|---:|---:|---:|
| `V0_raw_patel` | 11 | 0.8852 | 0.8852 | 0.8361 |
| `V0_raw_patel` | 22 | 0.8361 | 0.7541 | 0.8361 |
| `V0_raw_patel` | 33 | 0.8197 | 0.7869 | 0.7705 |
| `V1_encoded_support_raw_direction` | 11 | 0.8361 | 0.8361 | 0.8033 |
| `V1_encoded_support_raw_direction` | 22 | 0.8525 | 0.8033 | 0.8525 |
| `V1_encoded_support_raw_direction` | 33 | 0.8689 | 0.8361 | 0.8525 |
| `V2_encoded_support_encoded_direction` | 11 | 0.8033 | 0.7705 | 0.7705 |
| `V2_encoded_support_encoded_direction` | 22 | 0.7705 | 0.7213 | 0.7705 |
| `V2_encoded_support_encoded_direction` | 33 | 0.8197 | 0.7705 | 0.7541 |

Conclusion:

- `V1` is the only positive branch in this pilot.
  - It slightly improves best F1 over raw (`0.8525` vs `0.8470`).
  - It improves exported F1 (`0.8251` vs `0.8087`) and final F1 (`0.8361` vs `0.8142`).
  - It narrows both retention gaps, especially final-best (`-0.0164` vs `-0.0328`).
- `V1` does not visibly solve the low-margin taxonomy:
  - final failure remains `symmetric_collapse=3/3`.
  - final support median is lower than raw (`0.01290` vs `0.01593`).
  - final exported margin is effectively unchanged/slightly lower (`0.00724` vs `0.00746`).
- `V2` validates the risk hypothesis for encoded direction:
  - replacing raw tau with encoded tau substantially hurts best/exported/final F1.
  - The encoded direction branch has a higher final gate margin, but that does not translate into better primary F1 or strict@0.1.

Next decision: if expanding this line, use `V1` only. The evidence supports testing encoded support with raw direction on `sim3+sim4` with 5 seeds; it does not support encoded tau as a default direction prior.
