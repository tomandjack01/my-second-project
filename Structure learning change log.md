# Structure Learning Change Log

Date started: 2026-03-09

## Logging Rule

Every meaningful change to the structure-learning pipeline should be recorded here with:

- date
- change batch name
- files changed
- purpose
- expected effect
- actual observed effect, if any

## Entries

### 2026-03-09 - Logging setup and agreed priority order

Files:

- `Judgment based on current code.md`
- `Structure learning priority plan.md`
- `Structure learning change log.md`

Purpose:

- formalize the currently agreed implementation priorities
- separate judgment notes from implementation planning
- establish a persistent place to record every later structure-learning change

Expected effect:

- reduce context drift while iterating on the model
- make it easier to compare code changes against empirical outcomes

Observed effect:

- documentation only
- no model behavior change yet

### 2026-03-09 - P0 batch start: direction convention and proxy stabilization

Files:

- `GraphExp/main_structure_learning.py`
- `GraphExp/test_eval.py`
- `GraphExp/evaluate_directional_prf1.py`
- `Structure learning priority plan.md`
- `Structure learning change log.md`

Purpose:

- unify direction semantics between training, proxy scoring, export, and evaluation-facing outputs
- remove brittle proxy zero-collapse behavior

Expected effect:

- proxy scores should remain informative in short runs
- exported "causal" adjacency should match the evaluator convention directly
- best-epoch selection should become easier to interpret against real F1

Observed effect:

- implementation in progress

### 2026-03-09 - P0 batch result: semantics fixed, proxy improved, long-run drift still unresolved

Files:

- `GraphExp/main_structure_learning.py`
- `GraphExp/test_eval.py`
- `GraphExp/evaluate_directional_prf1.py`
- `Judgment based on current code.md`
- `Structure learning priority plan.md`
- `Structure learning change log.md`

Purpose:

- validate whether the P0 semantic fixes and proxy redesign actually improved model selection under the locked `sim4.csv -> h4.txt` protocol
- update the working priorities based on current-snapshot evidence rather than code reading alone

Expected effect:

- raw and causal exports should become directly comparable without hidden transpose ambiguity
- short-run selection should stop exporting the obviously wrong early epoch
- long-run best-epoch selection should improve materially, even if not yet optimal

Observed effect:

- post-P0 6-epoch validation (`run_20260309_205705`) exported epoch `6` with `F1 = 0.5902`
- post-P0 100-epoch validation (`run_20260309_210058`) exported epoch `10` with `F1 = 0.5410`
- in the same 100-epoch run, the final epoch collapsed to `F1 = 0.0328`
- raw and causal exports now match under explicit evaluator conventions, so export semantics are no longer the main confounder
- the proxy is now useful enough to avoid catastrophic late collapse, but it still overvalues Patel-clean, high-margin states before checking skeleton quality strictly enough
- next agreed priority is to add selection guardrails on skeleton and density before changing the training objective again

### 2026-03-09 - P0-3 implementation: guarded best-epoch selection with fallback

Files:

- `GraphExp/main_structure_learning.py`
- `Structure learning change log.md`

Purpose:

- keep the current weighted proxy score
- add conservative selection guardrails on skeleton retention and density sanity
- preserve robustness by falling back to score-only best-epoch selection when no epoch passes guardrails

Expected effect:

- long-run selection should stop preferring Patel-clean but skeleton-weakened epochs too early
- catastrophic density drift should be filtered before export selection
- short runs should remain safe because score-only fallback still exists

Observed effect:

- implementation complete
- short-run validation (`run_20260309_215251`) kept best F1 at `0.5902` with selection mode `guarded`
- 100-epoch validation (`run_20260309_215355`) improved exported best F1 from `0.5410` to `0.5738`
- epoch `10` was explicitly blocked by the new skeleton-retention guardrail:
  - score `0.8206`
  - required skeleton overlap `0.578`
  - actual skeleton overlap `0.560`
- final epoch collapse was still present (`F1 = 0.0328`), but all late broken epochs were filtered from export selection
- next agreed priority is to align proxy `top_k` with the locked evaluation protocol before changing the training objective

### 2026-03-09 - P0-4 implementation: decouple selection `top_k` from noise-guide `top_k`

Files:

- `GraphExp/main_structure_learning.py`
- `Structure learning change log.md`

Purpose:

- let best-epoch proxy selection use the locked evaluation edge budget directly
- avoid forcing proxy selection to inherit the Patel noise-guide edge budget

Expected effect:

- selection metrics should become more comparable to the headline `--top_k 61` evaluator
- if the remaining gap to baseline is partly due to `k` mismatch, best-epoch selection should improve without touching the training loss

Observed effect:

- implementation complete
- short-run validation with `selection_top_k = 61` (`run_20260309_220359`) kept F1 at `0.5902`, but the guarded path became too strict and fell back to score-only selection
- 100-epoch validation with `selection_top_k = 61` (`run_20260309_220535`) kept best epoch at `9` and best F1 at `0.5738`
- this means P0-4 improved protocol alignment, but did not improve the best observed F1 on `sim4 -> h4`
- next agreed priority is to move from proxy cleanup to skeleton-preserving training design

### 2026-03-09 - Added new-conversation handoff file

Files:

- `Structure learning handoff.md`
- `Structure learning change log.md`

Purpose:

- preserve the current modification logic, evaluation protocol, and empirical conclusions across context resets or fresh conversations

Expected effect:

- a new assistant can resume work with much less ambiguity
- reduces the risk of repeating already-finished P0 cleanup work

Observed effect:

- handoff file created with:
  - locked evaluation protocol
  - current best runs
  - agreed next priority
  - exact starter prompt for a fresh conversation
