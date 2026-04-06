# Support Learning Ablation

## Scope

This note isolates the **support-learning** side of the current
`main_structure_learning.py` pipeline.

The current support path is mainly composed of:

1. `fixed_support_mask_mode = maxgap_kappa`
2. `structure_init_mode = patel_kappa` or `random`
3. optional persistent support prior:
   - `kappa_logit_bias_scale`
4. Patel-based symmetric noise guide for diffusion corruption

The question here is not "what gives the best total score overall", but:

- which support-side components are still necessary
- which ones look redundant
- which ones are still unproven and should not be removed yet

## Current Support Ablation Plan

Use a staged plan, from lowest-risk removal to highest-risk removal:

1. Support init ablation
   - compare `structure_init_mode = patel_kappa` vs `random`
   - keep the hard support mask fixed
   - purpose:
     - test whether Patel support initialization is still needed once the
       factorized support/direction branch and routing are already in place

2. Persistent support-prior ablation
   - compare `kappa_logit_bias_scale = 0.0` vs `0.3`
   - keep the rest of the branch fixed
   - purpose:
     - test whether a persistent Patel-kappa support bias is a real mechanism
       gain or just extra complexity

3. Hard support-mask ablation
   - compare `fixed_support_mask_mode = maxgap_kappa` vs `none`
   - keep the same routing, same init family, same directional settings
   - purpose:
     - test whether the model can now learn a usable support skeleton by itself
       without the externally fixed kappa mask

4. Noise-guide ablation
   - compare current Patel noise guide vs learned or mixed noise guide
   - purpose:
     - test whether the diffusion corruption process still needs the Patel
       support prior
   - status:
     - only diagnostic evidence exists so far; no training-branch replacement
       is validated yet

## Executed Evidence

### A. Support init ablation

- Branch:
  - `sim4`
  - `seed = 11`
  - `support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - Patel direction teacher on
  - Patel kappa gate on

- Baseline:
  - run:
    - `GraphExp/results/run_20260403_193344`
  - config:
    - `structure_init_mode = patel_kappa`
  - result:
    - best/export/final:
      - `0.868852 / 0.803279 / 0.868852`

- Ablation:
  - run:
    - `GraphExp/results/run_20260403_222030`
  - config:
    - `structure_init_mode = random`
  - result:
    - best/export/final:
      - `0.885246 / 0.770492 / 0.852459`

- Read:
  - replacing Patel support init with random init did **not** hurt the GT best
    ceiling
  - it actually improved the GT best on this branch:
    - `0.868852 -> 0.885246`
  - export mismatch still remained
  - current interpretation:
    - Patel support initialization looks **non-essential**
    - it is a plausible redundancy candidate
    - but this is still a local result, not yet a suite-wide proof

### B. Persistent support-prior ablation

- Branch:
  - `sim3`
  - `3` seeds
  - same factorized support branch with fixed `maxgap_kappa` support

- Baseline:
  - `kappa_logit_bias_scale = 0.0`
  - aggregate:
    - best strict:
      - `0.8519 +/- 0.0262`
    - strict at export/final audit:
      - `0.8148 +/- 0.0262`

- Ablation:
  - `kappa_logit_bias_scale = 0.3`
  - aggregate:
    - best strict:
      - `0.8148 +/- 0.0693`
    - strict at export/final audit:
      - `0.7963 +/- 0.0524`

- Source:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260327_172812_residual_kappa_only_confirm_aggregate.csv`

- Read:
  - the persistent kappa support bias is approximately neutral to mildly
    stabilizing
  - it is **not** the main blocker
  - it is also **not** a clear gain
  - current interpretation:
    - this looks **redundant as a default-on support mechanism**
    - keep it out of the mainline unless a later branch specifically needs it

### C. Hard support-mask ablation

- Fixed reference branch:
  - run:
    - `GraphExp/results/run_20260403_222030`
  - config:
    - `structure_init_mode = random`
    - `fixed_support_mask_mode = maxgap_kappa`
    - `directional_kappa_gate = on`
    - `gradient_routing_mode = warmup_then_orthogonal`
    - `detach_direction_from_main_after_epoch = 23`
    - `causal_lag_main_weight = 0.25`
  - result:
    - best/export/final:
      - `0.885246 / 0.770492 / 0.852459`

- New ablation:
  - run:
    - `GraphExp/results/run_20260404_212848`
  - single delta:
    - `fixed_support_mask_mode: maxgap_kappa -> none`
  - result:
    - best/export/final:
      - `0.0824 / 0.0793 / 0.0778`
  - training-side support diagnostics:
    - epoch 10:
      - `adj_parent_entropy_mean = 0.908`
      - `adj_eff_parents_mean = 34.850`
    - final epoch:
      - `adj_parent_entropy_mean = 0.572`
      - `adj_eff_parents_mean = 12.081`
    - guardrails:
      - no epoch passed the guarded selector
      - export fell back to `score_only`

- Read:
  - removing the hard support mask did **not** produce a mild degradation
  - it caused a near-total collapse of support selectivity
  - the learned graph became far too wide and diffuse
  - current interpretation:
    - `fixed_support_mask_mode = maxgap_kappa` is **not redundant**
    - under the current backbone, it is still a required support-space
      constraint

### D. Noise-guide diagnostic

- Source:
  - `GraphExp/results/cross_pred_v1_final_only_compare_patel_kappa_dir_patel_3seeds_20260316_211721_noise_probe_pilot3_aggregate.csv`

- Existing diagnostic read:
  - `blend50` and `learned(detach)` noise guides only changed probe denoising
    loss at a tiny scale
  - seed-to-seed sign flips occurred
  - best-score epochs did not show a stable advantage over the Patel noise
    guide

- Read:
  - there is no strong evidence yet that the current Patel noise guide should
    be replaced
  - there is also no strong evidence that it is the main performance bottleneck
  - current interpretation:
    - **unproven**
    - do not call it redundant yet
    - do not spend mainline budget on rewriting the diffusion corruption path
      before more important support tests are settled

### E. Support-prior source ablation under fixed hard support

- Purpose:
  - test whether the current branch needs **Patel specifically** as the
    support-prior source
  - keep the hard support constraint itself
  - replace only the support-prior source used by:
    - fixed support mask
    - diffusion noise guide

- Code support:
  - `main_structure_learning.py` now exposes:
    - `--support_prior_mode {patel_kappa, pearson_abs}`
  - this changes only:
    - the symmetric support prior used to build the hard support mask
    - the symmetric support prior used to build the diffusion noise guide
  - it does **not** change:
    - the Patel tau direction teacher
    - the Patel-kappa directional gate
    - the current routing split

- Fixed branch:
  - `sim4`
  - `seed = 11`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `structure_init_mode = random`
  - `direction_init_mode = random`
  - `directional_kappa_gate = on`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `selection_score_mode = legacy`

- Patel support prior reference:
  - run:
    - `GraphExp/results/run_20260403_222030`
  - config:
    - `support_prior_mode = patel_kappa`
  - result:
    - best/export/final:
      - `0.885246 / 0.770492 / 0.852459`

- Pearson support prior pilot:
  - run:
    - `GraphExp/results/run_20260404_222539`
  - config delta:
    - `support_prior_mode: patel_kappa -> pearson_abs`
  - result:
    - best/export/final:
      - `0.901639 / 0.803279 / 0.885246`
  - selector audit:
    - best GT epoch:
      - `38`
    - exported epoch:
      - `8`
    - final epoch:
      - `40`

- Read:
  - replacing the Patel support prior with `pearson_abs` did **not** damage the
    branch
  - on this pilot, it improved:
    - GT best:
      - `0.885246 -> 0.901639`
    - final:
      - `0.852459 -> 0.885246`
    - export:
      - `0.770492 -> 0.803279`
  - therefore the current evidence moves beyond:
    - "Patel support init may be redundant"
  - and toward:
    - "Patel may not be required as the support-prior source at all, as long as
      a hard support constraint remains"
  - caution:
    - this is still a `1`-seed pilot
    - selector mismatch is still present, so export should not be over-read

- 3-seed confirmation:
  - summary:
    - `GraphExp/results/support_prior_sim4_3seed_summary_20260405_105844.csv`
  - aggregate:
    - `GraphExp/results/support_prior_sim4_3seed_summary_20260405_105844_aggregate.csv`
  - fixed backbone:
    - same as the pilot above
  - compared:
    - `support_prior_mode = patel_kappa`
    - `support_prior_mode = pearson_abs`
  - aggregate result:
    - `patel_kappa`
      - best mean:
        - `0.857924 +/- 0.025041`
      - export mean:
        - `0.819672 +/- 0.032787`
      - final mean:
        - `0.808743 +/- 0.050083`
    - `pearson_abs`
      - best mean:
        - `0.852459 +/- 0.028394`
      - export mean:
        - `0.808743 +/- 0.025042`
      - final mean:
        - `0.819672 +/- 0.043373`
  - paired seed-level read:
    - best:
      - tie on seeds `11` and `22`
      - `pearson_abs` is lower on seed `33`
    - final:
      - tie on seed `11`
      - `pearson_abs` is slightly higher on seeds `22` and `33`
  - updated interpretation:
    - the earlier 1-seed Pearson result was somewhat optimistic
    - `pearson_abs` is **competitive**, but not a clear across-the-board win
    - the defensible conclusion is now:
      - Patel is not uniquely required as the support-prior source
      - but `pearson_abs` is not yet a proven replacement default either

- Cross-dataset transfer check (`seed = 11`):
  - summary:
    - `GraphExp/results/support_prior_cross_dataset_seed11_20260405_114142.csv`
  - datasets:
    - `fMRI`
    - `sim2`
    - `sim3`
  - observed result:
    - all three datasets produced **identical** best/export/final metrics under:
      - `support_prior_mode = patel_kappa`
      - `support_prior_mode = pearson_abs`
  - direct skeleton audit:
    - a follow-up check compared the hard support masks produced by the current
      `maxgap` rule under both sources
    - artifact:
      - `GraphExp/results/support_prior_skeleton_overlap_20260405.csv`
    - on:
      - `fMRI`
      - `sim2`
      - `sim3`
      - `sim4`
    - the selected undirected support skeletons were exactly identical
  - implication:
    - under the current `maxgap` hard-support rule, this source comparison is
      largely a **no-op** on the current dataset suite
    - the earlier small metric differences on `sim4` should therefore be read
      as run noise, not as clean evidence that Pearson is better
  - corrected interpretation:
    - the present code now supports non-Patel support-prior sources
    - but the current `maxgap` support-selection rule collapses Patel and
      Pearson into the same chosen skeleton on the tested datasets
    - so the real question "does the framework need Patel specifically as the
      support source?" is still **not actually resolved** by this ablation
      family alone

### F. Support-selection rule sensitivity scan

- Purpose:
  - test whether the Patel-vs-Pearson support-source comparison is empty because:
    - the two priors are globally almost the same
  - or because:
    - the current selection operator / cutoff happens to project them to the
      same skeleton

- Script:
  - `GraphExp/scan_support_selection_rules.py`

- Artifacts:
  - detailed rule scan:
    - `GraphExp/results/support_selection_rule_scan_20260405_115654.csv`
  - rank diagnostics:
    - `GraphExp/results/support_selection_rule_scan_rank_20260405_115654.csv`

- Datasets:
  - `fMRI`
  - `sim2`
  - `sim3`
  - `sim4`

- Rules scanned:
  - current `maxgap`
  - current reference `topk`
  - `topk` at multiple fractions of the full pair set:
    - `5%`
    - `10%`
    - `20%`
    - `30%`
    - `50%`
  - quantile thresholds:
    - `0.50`
    - `0.70`
    - `0.80`
    - `0.90`
    - `0.95`

- Rank diagnostic read:
  - Patel and Pearson are **not** globally identical on any dataset
  - tiebreak rank Kendall-like agreement:
    - `fMRI`: `0.7778`
    - `sim2`: `0.8869`
    - `sim3`: `0.7692`
    - `sim4`: `0.5139`
  - so the earlier empty-ablation read must be narrowed:
    - it is **not** that the two support priors are the same object
    - it is that the current mainline selection operating point often maps them
      to the same hard support set

- Mainline operating-point check:
  - on all four datasets:
    - `maxgap` produced the same Patel/Pearson skeleton
    - current reference `topk` also produced the same Patel/Pearson skeleton
  - this preserves the earlier practical conclusion:
    - under the current training branch, Patel-vs-Pearson remains an empty
      ablation at the actual operating point we are using

- But alternative rules do trigger differences:
  - `fMRI`
    - first differing scanned rule:
      - `topk_frac_0.30`
    - Patel vs Pearson:
      - `3` pairs vs `3` pairs
      - Jaccard:
        - `0.50`
  - `sim2`
    - first differing scanned rule:
      - `topk_frac_0.05`
    - Patel vs Pearson:
      - `2` pairs vs `2` pairs
      - Jaccard:
        - `0.3333`
  - `sim3`
    - first differing scanned rule:
      - `topk_frac_0.05`
    - Patel vs Pearson:
      - `5` pairs vs `5` pairs
      - Jaccard:
        - `0.4286`
  - `sim4`
    - current reference `topk = 61` is still identical
    - but `topk_frac_0.10` already differentiates them:
      - `122` pairs vs `122` pairs
      - Jaccard:
        - `0.8626`
    - quantile thresholds separate them more strongly:
      - `quantile_0.90`
        - Patel `82` pairs
        - Pearson `123` pairs
        - Jaccard `0.6667`

- Updated interpretation:
  - the bottleneck is now localized more precisely:
    - not "support prior source has no distinguishable information"
    - but:
      - "the current selection operator plus cutoff does not expose that
        difference at the mainline operating point"
  - therefore support-source genericity is still unresolved for the current
    training path
  - but it is no longer accurate to say Patel and Pearson are globally
    indistinguishable on the current datasets
  - the next useful support experiment is not "more datasets with the same
    `maxgap` rule"
  - it is:
    - redesign or recalibrate the support-selection operator near the current
      cut region, then re-run the Patel-vs-Pearson comparison


## Current Conclusions

### Likely redundant support-side items

1. `structure_init_mode = patel_kappa`
   - current best local support-side reduction on `sim4` was achieved with
     `structure_init_mode = random`
   - current read:
     - likely redundant
     - removable candidate

2. `kappa_logit_bias_scale`
   - multi-seed evidence says it is near-neutral, not a main gain
   - current read:
     - likely redundant as a default mechanism

3. Patel as the support-prior source itself
   - the current code can swap Patel for `pearson_abs`
   - Patel and Pearson are not globally identical priors on the tested datasets
   - but under the present mainline operating point:
     - `maxgap`
     - and current reference `topk`
     - they still collapse to the same selected support skeleton
   - current read:
     - source genericity remains unresolved for the current training path
     - the immediate bottleneck is now better localized to the support-selection
       operator / cutoff, not to a proven lack of alternative prior signal

### Not redundant under the current architecture

1. `fixed_support_mask_mode = maxgap_kappa`
   - removing it on the current `sim4` factorized/routed branch collapsed the
     support structure almost completely
   - current read:
     - required
     - do not remove from the mainline yet

### Still unresolved

1. Patel-based noise guide
   - only weak diagnostic evidence exists
   - current read:
     - unresolved
     - low priority

## Practical Mainline Read

For the current support side, the safest interpretation is:

- keep:
  - `fixed_support_mask_mode = maxgap_kappa`
  - the current diffusion backbone
- allow removal first:
  - `structure_init_mode = patel_kappa`
  - persistent `kappa_logit_bias_scale`
- postpone:
  - noise-guide rewrites

In short:

- hard support masking is still doing real work
- Patel support initialization does not currently look necessary
- persistent kappa support bias does not currently justify its complexity
- support-prior source genericity is still unresolved in the mainline because
  the current selection operating point (`maxgap` / current reference `topk`)
  projects Patel and Pearson to the same skeleton
- offline rule scanning shows the two priors are still distinguishable under
  other selection rules, so the current bottleneck is operator resolution, not
  a proven absence of source difference
- noise-guide replacement is still unproven

## Recommended Next Support Follow-Up

The next support-side experiment should **not** remove the mask entirely again.
That question is answered well enough for now.

The next useful support follow-up is:

1. keep `fixed_support_mask_mode = maxgap_kappa`
2. keep `structure_init_mode = random`
3. if support-side work continues, do **not** spend budget on more Patel-vs-
   Pearson training runs under the same `maxgap` operator first
4. first redesign or recalibrate the support-selection operator so that Patel
   and Pearson can differ near the current mainline cut region:
   - a local `topk` sweep around the current reference density
   - or a calibrated threshold rule
   - or a two-stage rule that preserves comparable support size while changing
     the cutoff geometry
5. only after a rule yields distinct Patel/Pearson skeletons at a comparable
   operating density should the training ablation be re-run

That would test whether the framework needs:

- a hard support constraint in general

or

- Patel specifically
