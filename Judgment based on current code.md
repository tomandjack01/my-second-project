# Judgment Based on Current Code

Date: 2026-03-09

## Purpose

This note records judgments that are currently based on code inspection, not on a fresh full training run of the current snapshot.

It should be updated whenever either:

- the code path in `GraphExp/main_structure_learning.py`, `GraphExp/models/DDM.py`, or `GraphExp/utils/patel_util.py` changes
- new training outputs from the current snapshot become available

## Scope

The current judgments are based on static reading of:

- `GraphExp/main_structure_learning.py`
- `GraphExp/models/DDM.py`
- `GraphExp/utils/patel_util.py`

They now also include targeted post-fix runs on `sim4.csv -> h4.txt`, so this note contains both code-based judgment and current-snapshot empirical updates.

## Related Working Docs

- `Structure learning priority plan.md`
- `Structure learning change log.md`

## What Can Be Judged With High Confidence From Code

1. Training/export structure semantics are now more consistent.
   - Structure logits and adjacency are routed through the same logic in `DDM.get_structure_logits()` and `DDM.get_structure_adj()`.
   - This reduces the earlier risk that training diagnostics, best-epoch selection, and final export were looking at slightly different graph objects.

2. Patel semantics are cleaner than before.
   - `score = -kappa * tau` is used for asymmetric initialization.
   - `kappa` is used as skeleton-strength prior for noise guidance and proxy skeleton overlap.
   - `tau` is used as the weak directional prior.
   - This is a meaningful semantic improvement over treating one Patel matrix as if it served all roles equally well.

3. The main DDM objective is still the dominant training signal.
   - The actual base optimization target remains diffusion denoising plus sparsity and hub regularization.
   - Direction and orthogonality losses are auxiliary terms with warmup, ratio-adaptive scaling, annealing, and step caps.
   - Therefore the code is intentionally biased toward protecting the main task.

4. The current code can plausibly learn direction, not just an undirected skeleton.
   - The learned structure is asymmetric because it is parameterized through sender/receiver embeddings.
   - The directional margin loss explicitly acts on pairwise direction differences.
   - The model is therefore not structurally restricted to symmetric edges.

5. Best-epoch export semantics are clearer.
   - `learned_adjacency.*` is intended to represent the selected best epoch.
   - `final_epoch_adjacency.*` is saved separately for comparison.
   - This reduces evaluation confusion caused by silently exporting the last epoch while discussing the best epoch.

6. A practical "Patel ceiling" risk still exists.
   - Even if the model is not theoretically bounded by Patel, Patel still influences four critical places:
     - initialization
     - noise guidance
     - directional prior
     - proxy-based model selection
   - That means the training process is still strongly biased toward Patel-consistent solutions in practice.

7. Skeleton and direction are still coupled at the parameter level.
   - The same sender/receiver parameters affect both whether a pair has a strong edge and which direction wins.
   - So even a direction-only auxiliary term can still perturb skeleton quality indirectly.
   - This is the main reason the next stage should focus on "preserve skeleton first, then assign direction".

## Existing Empirical Evidence Already Known

The following observations are available from previous experiments and should be treated as empirical evidence, but not necessarily evidence for the exact latest snapshot:

1. Early versions had almost no effective directional learning.
   - Weighted direction/orthogonality terms were close to zero.
   - Many symmetric ties appeared.
   - F1 was low.

2. After tuning, h4 improved substantially.
   - F1 reportedly rose to around 0.59 to 0.61.
   - Ties dropped sharply.
   - This supports the claim that directional learning can be induced.

3. sim3 remained weak.
   - Reported F1 stayed low.
   - This suggests clear dataset sensitivity and limited robustness.

4. Removing warmup caused obvious damage.
   - Orthogonality loss exploded.
   - Adjacency mean rose sharply.
   - Skeleton overlap broke down.
   - This strongly supports the judgment that auxiliary losses can corrupt the main task if not scheduled carefully.

5. Reintroducing short warmup and normalization improved stability relative to the aggressive version.
   - This supports the current scheduling philosophy.

## New Empirical Record Added On 2026-03-09

The following result was provided after this note was first created and is important because the evaluation protocol is explicit:

1. Historical baseline run:
   - Run directory: `GraphExp/results/run_20260308_183908`
   - Prediction file: `learned_adjacency.csv`
   - Dataset mapping used by the user: `sim4.csv -> h4.txt`
   - Evaluation command:
     - `python test_eval.py --pred results\\run_20260308_183908\\learned_adjacency.csv --gt ..\\fMRI_dataset\\h4.txt --top_k 61`
   - Reported result:
     - Precision = 0.6066
     - Recall = 0.6066
     - F1 = 0.6066
     - TP = 37
     - FP = 24
     - FN = 24
     - Ties = 2

2. Important evaluation-semantic note:
   - The historical baseline corresponds to evaluating the raw export under the causal GT convention.
   - In the current evaluator, this should be written explicitly as `--adj_convention raw` for `learned_adjacency.csv`, or `--adj_convention causal` for `learned_adjacency_causal.csv`.
   - Future comparisons should use the same evaluator or explicitly state if a different convention is used.

3. Immediate implication:
   - The historical strong result on the `sim4 -> h4` setting should currently be treated as the main local baseline for comparison with the latest code snapshot.
   - It is more informative than comparing against older untransposed or differently truncated evaluations.

## Current-Snapshot Results Added On 2026-03-09

The following runs were executed from the current snapshot after the latest code changes.

### Evaluation Protocol Used For Comparison

- Dataset: `sim4.csv -> h4.txt`
- Evaluator: `GraphExp/test_eval.py`
- Direction convention:
  - `learned_adjacency.csv` with `--adj_convention raw`
  - `learned_adjacency_causal.csv` with `--adj_convention causal`
- Edge truncation: `--top_k 61`

### Run A: 6-Epoch Current Snapshot (pre-P0, for reference)

- Run directory: `GraphExp/results/run_20260309_192310`
- Training setting:
  - `--csv_path ..\\fMRI_dataset\\sim4.csv`
  - `--epochs 6`
  - default pretraining ran for 50 epochs

Observed results:

1. Exported best epoch:
   - Exported epoch = 1
   - Proxy score = `0.0000`
   - F1 = `0.5738`
   - TP = 35, FP = 26, FN = 26
   - Ties = 11

2. Final epoch:
   - Final epoch = 6
   - F1 = `0.5902`
   - TP = 36, FP = 25, FN = 25
   - Ties = 7

3. Comparison against historical baseline:
   - Historical baseline F1 = `0.6066`
   - Current 6-epoch best export is worse than baseline
   - Current 6-epoch final epoch is also worse than baseline, but closer

4. Immediate judgment:
   - The new proxy score failed in this short run because agreement stayed at zero, causing all epoch scores to collapse to zero.
   - In this run, proxy-selected best epoch was worse than the final epoch under the ground-truth evaluator.

### Run B: 100-Epoch Current Snapshot (pre-P0, for reference)

- Run directory: `GraphExp/results/run_20260309_193851`
- Training setting:
  - `--csv_path ..\\fMRI_dataset\\sim4.csv`
  - `--epochs 100`
  - pretraining weights loaded from `run_20260309_192310\\pretrained_encoder.pt`

Observed results:

1. Proxy-selected best epoch:
   - Exported epoch = 27
   - Best proxy score = `0.0296`
   - F1 = `0.2787`
   - TP = 17, FP = 44, FN = 44
   - Ties = 13

2. Final epoch:
   - Final epoch = 100
   - F1 = `0.1148`
   - TP = 7, FP = 54, FN = 54
   - Ties = 987

3. Proxy score behavior:
   - Scores were highest around epochs 23 to 32
   - Top score region had only moderate Patel agreement and low skeleton overlap
   - The final epoch had much lower proxy score than epoch 27, so the proxy did detect later degeneration

4. Collapse / stability signal:
   - Encoder collapse diagnostics remained constant across logged epochs
   - This is expected because the temporal encoder was frozen after pretraining
   - Therefore these diagnostics are not informative about graph-structure degradation in the frozen-encoder regime

5. Immediate judgment:
   - Long training under the current snapshot severely degraded task-level directional F1 on `sim4 -> h4`
   - The proxy score helped avoid the catastrophic final epoch, but the proxy-selected epoch was still far worse than the historical baseline
   - This means the current proxy is directionally useful only in a weak sense: it can detect some late-stage collapse, but it is not yet aligned strongly enough with real evaluation quality

### Run C: 6-Epoch P0 Validation Snapshot

- Run directory: `GraphExp/results/run_20260309_205705`
- Training setting:
  - `--csv_path ..\\fMRI_dataset\\sim4.csv`
  - `--epochs 6`
  - encoder initialized from `run_20260309_192310\\pretrained_encoder.pt`

Observed results:

1. Export semantics:
   - `learned_adjacency.csv` evaluated with raw convention and `learned_adjacency_causal.csv` evaluated with causal convention produced the same result
   - This confirms that raw export and causal export are now semantically aligned after conversion

2. Best epoch selection:
   - Exported best epoch = `6`
   - F1 = `0.5902`
   - This is no longer the obviously wrong epoch-1 behavior seen before P0

3. Immediate judgment:
   - The semantic fix and delayed selection gate solved the short-run "wrong file / wrong epoch" style confusion
   - Short-run P0 behavior is more trustworthy, but still below the historical baseline `0.6066`

### Run D: 100-Epoch P0 Validation Snapshot

- Run directory: `GraphExp/results/run_20260309_210058`
- Training setting:
  - `--csv_path ..\\fMRI_dataset\\sim4.csv`
  - `--epochs 100`
  - encoder initialized from `run_20260309_192310\\pretrained_encoder.pt`

Observed results:

1. Proxy-selected best epoch:
   - Exported epoch = `10`
   - Best proxy score = `0.829277`
   - `learned_adjacency.csv` with `--adj_convention raw`:
     - F1 = `0.5410`
     - TP = `33`, FP = `28`, FN = `28`
     - Ties = `0`
   - `learned_adjacency_causal.csv` with `--adj_convention causal` gives the same result:
     - F1 = `0.5410`
     - TP = `33`, FP = `28`, FN = `28`
     - Ties = `0`

2. Final epoch:
   - Final epoch = `100`
   - `final_epoch_adjacency.csv` and `final_epoch_adjacency_causal.csv` both evaluate to:
     - F1 = `0.0328`
     - TP = `2`, FP = `59`, FN = `59`
     - Ties = `3`

3. Proxy details at selected epoch versus final epoch:
   - Epoch `10`:
     - `skeleton_overlap = 0.56`
     - `agreement_score = 0.98`
     - `agreement_coverage = 0.96`
     - `margin_score = 0.9958`
     - `density_factor = 0.9965`
     - `actual_pair_density = 0.037551`
     - `target_pair_density = 0.040816`
   - Epoch `100`:
     - `skeleton_overlap = 0.04`
     - `agreement_score = 1.00`
     - `agreement_coverage = 1.00`
     - `margin_score = 0.9993`
     - `density_factor = 0.0272`
     - `actual_pair_density = 0.598367`
     - `target_pair_density = 0.040816`

4. Immediate judgment:
   - P0 clearly improved long-run best-epoch selection relative to the pre-P0 100-epoch run:
     - pre-P0 exported best F1 = `0.2787`
     - post-P0 exported best F1 = `0.5410`
   - Export semantics are no longer the main confounder because raw and causal exports now agree under explicit evaluator conventions
   - However, the proxy still over-rewards Patel agreement and directional confidence
   - Epoch `10` received a very high proxy score even though its skeleton overlap was only `56%`, and its true F1 still lagged behind the historical baseline `0.6066`

### Run E: 6-Epoch P0-3 Guarded-Selection Validation

- Run directory: `GraphExp/results/run_20260309_215251`
- Training setting:
  - `--csv_path ..\\fMRI_dataset\\sim4.csv`
  - `--epochs 6`
  - encoder initialized from `run_20260309_192310\\pretrained_encoder.pt`
  - guarded selection defaults:
    - `selection_start_epoch = 6`
    - `selection_min_skeleton_overlap = 0.50`
    - `selection_min_skeleton_retention = 0.85`
    - `selection_min_density_factor = 0.65`
    - `selection_max_density_ratio = 2.50`

Observed results:

1. Best epoch selection:
   - Exported epoch = `6`
   - Selection mode = `guarded`
   - `learned_adjacency.csv` with `--adj_convention raw`:
     - F1 = `0.5902`
     - TP = `36`, FP = `25`, FN = `25`
     - Ties = `0`
   - `learned_adjacency_causal.csv` with `--adj_convention causal` gives the same result

2. Immediate judgment:
   - P0-3 did not damage the short-run regime
   - The guarded path remained usable in the 6-epoch case and matched the previous short-run best behavior

### Run F: 100-Epoch P0-3 Guarded-Selection Validation

- Run directory: `GraphExp/results/run_20260309_215355`
- Training setting:
  - `--csv_path ..\\fMRI_dataset\\sim4.csv`
  - `--epochs 100`
  - encoder initialized from `run_20260309_192310\\pretrained_encoder.pt`
  - guarded selection defaults:
    - `selection_start_epoch = 6`
    - `selection_min_skeleton_overlap = 0.50`
    - `selection_min_skeleton_retention = 0.85`
    - `selection_min_density_factor = 0.65`
    - `selection_max_density_ratio = 2.50`

Observed results:

1. Guarded best epoch:
   - Exported epoch = `9`
   - Selection mode = `guarded`
   - `learned_adjacency.csv` with `--adj_convention raw`:
     - F1 = `0.5738`
     - TP = `35`, FP = `26`, FN = `26`
     - Ties = `0`
   - `learned_adjacency_causal.csv` with `--adj_convention causal` gives the same result

2. Final epoch:
   - Final epoch = `100`
   - `final_epoch_adjacency.csv` and `final_epoch_adjacency_causal.csv` both evaluate to:
     - F1 = `0.0328`
     - TP = `2`, FP = `59`, FN = `59`
     - Ties = `10`

3. Why epoch `10` was blocked:
   - Epoch `9`:
     - `score = 0.773034`
     - `skeleton_overlap = 0.62`
     - `density_factor = 0.786450`
     - `guardrail_density_ratio = 0.50`
     - `guardrail_required_skeleton_overlap = 0.578`
     - `guardrail_pass = 1`
   - Epoch `10`:
     - `score = 0.820602`
     - `skeleton_overlap = 0.56`
     - `density_factor = 0.980501`
     - `guardrail_density_ratio = 0.82`
     - `guardrail_required_skeleton_overlap = 0.578`
     - `guardrail_pass = 0`
     - `guardrail_reason = low_skeleton`
   - Epoch `100`:
     - `score = 0.468353`
     - `skeleton_overlap = 0.04`
     - `density_factor = 0.022408`
     - `guardrail_density_ratio = 15.74`
     - `guardrail_pass = 0`
     - `guardrail_reason = low_skeleton|low_density_factor|density_ratio_out_of_range`

4. Immediate judgment:
   - P0-3 improved long-run best-epoch selection further:
     - post-P0 exported best F1 = `0.5410`
     - post-P0-3 exported best F1 = `0.5738`
   - The guardrails did what they were designed to do:
     - they rejected epoch `10` despite its higher proxy score because skeleton retention had already slipped below the required floor
     - they blocked all later density-exploded epochs from being selected
   - This is still below the historical baseline `0.6066`, so the selection fix is helpful but not sufficient by itself

### Run G: 6-Epoch P0-4 Selection-TopK Alignment Validation

- Run directory: `GraphExp/results/run_20260309_220359`
- Training setting:
  - same as Run E
  - plus `selection_top_k = 61`

Observed results:

1. Best epoch selection:
   - Exported epoch = `6`
   - Selection mode = `score_only_fallback`
   - `learned_adjacency.csv` with `--adj_convention raw`:
     - F1 = `0.5902`
     - TP = `36`, FP = `25`, FN = `25`
     - Ties = `0`

2. Guardrail detail:
   - Epoch `6` had:
     - `skeleton_overlap = 0.688525`
     - `density_factor = 0.621461`
     - `guardrail_density_ratio = 0.377049`
     - `guardrail_reason = low_density_factor|density_ratio_out_of_range`

3. Immediate judgment:
   - Aligning proxy `top_k` to `61` did not improve short-run F1
   - Under the stricter `top_k = 61` density target, the current guardrail thresholds become too strict for the short-run guarded path, so fallback selection is used instead

### Run H: 100-Epoch P0-4 Selection-TopK Alignment Validation

- Run directory: `GraphExp/results/run_20260309_220535`
- Training setting:
  - same as Run F
  - plus `selection_top_k = 61`

Observed results:

1. Guarded best epoch:
   - Exported epoch = `9`
   - Selection mode = `guarded`
   - `learned_adjacency.csv` with `--adj_convention raw`:
     - F1 = `0.5738`
     - TP = `35`, FP = `26`, FN = `26`
     - Ties = `0`

2. Guardrail detail:
   - Epoch `9` had:
     - `score = 0.747192`
     - `skeleton_overlap = 0.622951`
     - `guardrail_required_skeleton_overlap = 0.599180`
     - `guardrail_pass = 1`
   - Epoch `10` had:
     - `score = 0.801427`
     - `skeleton_overlap = 0.540984`
     - `guardrail_required_skeleton_overlap = 0.599180`
     - `guardrail_pass = 0`
     - `guardrail_reason = low_skeleton`

3. Immediate judgment:
   - Aligning proxy `top_k` to `61` did not change the selected best epoch or the exported long-run F1 on `sim4 -> h4`
   - Therefore the remaining gap to the historical baseline is not mainly caused by the proxy `k` mismatch alone
   - The `selection_top_k` option is still useful as a semantic-control knob, but it is not a sufficient improvement by itself

### Strongest Empirical Conclusions So Far

1. The current snapshot still has not matched the historical `sim4 -> h4` baseline `0.6066`.

2. P0 and P0-3 fixed the semantic confusion and materially improved best-epoch selection.
   - Short-run selection no longer collapses to the obviously wrong epoch-1 export
   - Long-run exported best F1 improved from `0.2787` to `0.5410`, then to `0.5738`
   - Raw and causal exports now agree under explicit evaluator conventions

3. P0-4 improved protocol alignment but did not improve F1 further on `sim4 -> h4`.
   - `selection_top_k = 61` kept short-run F1 at `0.5902`
   - `selection_top_k = 61` kept long-run best F1 at `0.5738`
   - This suggests selection cleanup is approaching diminishing returns under the current training dynamics

4. The long-training regime still allows severe late-stage structural drift.
   - Evidence:
     - post-P0 and post-P0-3 final epoch both fell to `0.0328`
     - epoch `100` had `actual_pair_density = 0.598367` against target `0.040816`
     - epoch `100` kept perfect Patel-style agreement metrics while skeleton overlap collapsed to `0.04`

5. The current best-epoch proxy is useful but still not conservative enough.
   - The guarded selection layer now rejects catastrophic late collapse and can reject some Patel-clean but skeleton-weakened epochs
   - But the underlying proxy still scores Patel-clean, high-margin states too generously before the guardrail layer intervenes

6. The main next bottleneck is now clearer than before.
   - Selection/evaluation consistency cleanup is mostly sufficient for now
   - The next likely gain must come from training dynamics that preserve skeleton quality better, not from more proxy-threshold tuning alone

## What Still Requires Current-Snapshot Training Output

The following points cannot be judged precisely from code alone and should be validated with new runs from the current snapshot:

1. Whether a stronger "preserve skeleton first, then assign direction" design can recover the remaining gap to the historical baseline on `sim4 -> h4`.

2. Whether the current directional prior is still too strong, too weak, or properly balanced on each dataset.

3. Whether the practical Patel ceiling remains severe after the current cleanup, or only moderate.

## What Data Would Make Future Judgment More Accurate

To make future judgments more precise, the following outputs from the current snapshot are needed:

1. Per-epoch training signals:
   - diffusion loss
   - sparsity loss
   - raw and weighted directional loss
   - raw and weighted orthogonality loss

2. Per-epoch structure statistics:
   - adjacency mean
   - pair density
   - skeleton overlap
   - direction margin
   - Patel agreement
   - proxy score

3. Evaluation outputs:
   - precision
   - recall
   - F1
   - tie count
   - best epoch versus final epoch comparison

4. Robustness outputs:
   - multi-seed variance
   - cross-dataset comparison such as h1-h4 and sim1-sim4

## Current Working Judgment

Based on current code plus the updated empirical record, the strongest current judgment is:

- the direction-learning path is valid
- semantic mismatch is no longer the main blocker
- the system is still sensitive
- guarded best-epoch selection is now meaningfully better than score-only selection in long runs
- proxy `top_k` alignment alone did not improve F1 further on `sim4 -> h4`
- the next major training change should prioritize skeleton preservation before stronger direction assignment
- Patel should gradually act more like an early teacher and less like the final judge

## Update Template

When new evidence appears, update this file with:

1. Code change summary
2. Dataset and run setting
3. New empirical result
4. Which judgment was confirmed, weakened, or overturned
5. Remaining open question
