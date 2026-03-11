# Structure Learning Priority Plan

Date: 2026-03-09

## Goal

Turn the current directional-structure-learning work into a controlled improvement loop with explicit priorities, explicit evaluation protocol, and explicit logging.

## Locked Evaluation Protocol

Until explicitly changed, all core comparisons should use the same protocol:

- dataset mapping: `sim4.csv -> h4.txt`
- evaluator: `GraphExp/test_eval.py`
- raw export evaluation: `python test_eval.py --pred ...\\learned_adjacency.csv --gt ..\\fMRI_dataset\\h4.txt --top_k 61 --adj_convention raw`
- causal export evaluation: `python test_eval.py --pred ...\\learned_adjacency_causal.csv --gt ..\\fMRI_dataset\\h4.txt --top_k 61 --adj_convention causal`
- truncation: `--top_k 61`

This is the current baseline protocol because:

- the historical baseline `run_20260308_183908` was judged under this protocol
- recent current-snapshot runs were also compared under this protocol

## Current Baseline Facts

1. Historical baseline:
   - run: `GraphExp/results/run_20260308_183908`
   - F1 = `0.6066`
   - Ties = `2`

2. Current snapshot, 6 epochs:
   - run: `GraphExp/results/run_20260309_192310`
   - exported best F1 = `0.5738`
   - final F1 = `0.5902`
   - proxy score collapsed to zero across the run

3. Current snapshot, 100 epochs:
   - run: `GraphExp/results/run_20260309_193851`
   - exported best F1 = `0.2787`
   - final F1 = `0.1148`
   - final ties = `987`

4. Current snapshot after P0, 6 epochs:
   - run: `GraphExp/results/run_20260309_205705`
   - exported best F1 = `0.5902`
   - raw and causal exports agree under explicit conventions

5. Current snapshot after P0, 100 epochs:
   - run: `GraphExp/results/run_20260309_210058`
   - exported best F1 = `0.5410`
   - final F1 = `0.0328`
   - raw and causal exports agree under explicit conventions

6. Current snapshot after P0-3, 6 epochs:
   - run: `GraphExp/results/run_20260309_215251`
   - exported best F1 = `0.5902`
   - selection mode = `guarded`

7. Current snapshot after P0-3, 100 epochs:
   - run: `GraphExp/results/run_20260309_215355`
   - exported best F1 = `0.5738`
   - final F1 = `0.0328`
   - selection mode = `guarded`

8. Current snapshot after P0-4, 6 epochs:
   - run: `GraphExp/results/run_20260309_220359`
   - exported best F1 = `0.5902`
   - selection mode = `score_only_fallback`

9. Current snapshot after P0-4, 100 epochs:
   - run: `GraphExp/results/run_20260309_220535`
   - exported best F1 = `0.5738`
   - final F1 = `0.0328`
   - selection mode = `guarded`

## Priority Order

## P0

### P0-1. Unify direction semantics end to end

Problem:

- training losses, proxy scoring, export, and evaluator may still be using different direction conventions
- this can make proxy selection look wrong even when the model is not the only problem

Target:

- define one causal convention explicitly
- make directional loss, proxy score, saved adjacency, and evaluation all compare under that same convention

Expected effect:

- short-run proxy score should stop collapsing just because sign conventions disagree
- best-epoch selection should become interpretable

### P0-2. Redesign best-epoch scoring to remove zero-collapse behavior

Problem:

- the current proxy can collapse to all zeros when one factor becomes zero
- this happened in the 6-epoch current-snapshot run

Target:

- replace brittle multiplicative scoring with a stabilized score
- use Patel-derived terms only as soft diagnostics, not as hard veto terms

Expected effect:

- best epoch should no longer get stuck at epoch 1 in short runs
- proxy ordering should correlate better with true F1

### P0-3. Add selection guardrails on skeleton and density

Problem:

- post-P0 long-run selection improved, but the proxy still gave epoch `10` a very high score (`0.8293`) with only `56%` skeleton overlap
- late epochs can keep perfect Patel agreement and margin while the learned graph density and skeleton are already broken

Target:

- keep the weighted score, but only allow best-epoch updates when basic structure conditions are met
- candidate guardrails should focus on:
  - minimum `skeleton_overlap`
  - minimum `density_factor`
  - maximum allowed deviation between `actual_pair_density` and `target_pair_density`
- if no epoch passes the guardrails, fall back to the old score-only rule instead of exporting nothing

Expected effect:

- prevent Patel-clean but skeleton-weak epochs from being selected too early
- make long-run best-epoch selection track real F1 more closely before touching the training objective again

### P0-4. Align proxy `top_k` with the locked evaluation protocol

Problem:

- the locked evaluator uses `--top_k 61`
- the current proxy still derives `quality_top_k` from `target_edge_count/top_k_edges`, which defaults to `50`
- this means model selection is still not optimizing under the same sparsity budget used for the headline F1 comparison

Target:

- decouple proxy-selection `top_k` from Patel noise-guide `top_k_edges`
- allow selection to use the locked evaluation budget directly

Expected effect:

- reduce remaining selection/evaluation mismatch
- make the proxy and guardrail metrics better aligned with the comparison protocol before changing the training objective again

Observed result:

- protocol alignment is now available through `selection_top_k`
- on `sim4 -> h4`, using `selection_top_k = 61` did not improve best F1 beyond the P0-3 result
- therefore P0-level selection cleanup is no longer the main expected source of gains

## P1

### P1-1. Move to three-stage training

Stages:

1. skeleton-first stage
2. direction-assignment stage
3. late stabilization stage

Problem addressed:

- long training currently drifts badly
- directional pressure and structure drift are not well separated

Expected effect:

- less late-stage degeneration
- lower tie explosion
- better best-epoch region in long runs

### P1-2. Add skeleton anchor loss

Problem:

- sender/receiver parameters currently control both skeleton and direction
- direction updates can damage skeleton quality indirectly

Target:

- derive an undirected skeleton quantity from the learned adjacency
- anchor it to a teacher skeleton obtained from an earlier skeleton-first stage

Expected effect:

- preserve useful structure while still allowing directional refinement

## P2

### P2-1. Gate direction loss by model-supported skeleton

Target:

- only apply directional prior where the model already supports the pair as a plausible edge

Expected effect:

- fewer false positive directed edges on implausible pairs

### P2-2. Reduce Patel's role to early guidance only

Target:

- Patel should remain an initializer and weak early teacher
- Patel should stop acting like the dominant judge of final model quality

### P2-3. Consider reliability-weighted tau later

Target:

- bootstrap or stability-weight Patel tau before using it as directional guidance

Use case:

- especially relevant for unstable datasets such as `sim3`

## Execution Rule

From this point forward, every implementation batch should do both:

1. update `Structure learning change log.md` before or alongside the code change
2. record any new empirical result in `Judgment based on current code.md`

## Next Concrete Step

The next implementation batch should move to P1-1:

1. separate training into:
   - skeleton-first stage
   - direction-assignment stage
   - late stabilization stage
2. keep the current guarded best-epoch selection in place as the export mechanism
3. re-run the locked `sim4 -> h4` protocol for a short run and a 100-epoch run
4. compare against:
   - historical baseline `0.6066`
   - post-P0-3 best `0.5738`

Further P0-only threshold tuning is not the preferred next move unless new evidence shows a specific remaining selection bug.
