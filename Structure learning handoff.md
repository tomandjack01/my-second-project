# Structure Learning Handoff

Date: 2026-03-09

## Purpose

This file is the shortest reliable handoff for a new conversation.

Use it when starting a fresh chat so the next assistant can inherit:

- the current modification logic
- the evaluation protocol
- the empirical results already obtained
- the agreed next priority

## First Files To Read

In a new conversation, the assistant should read these files first:

1. `Structure learning handoff.md`
2. `Judgment based on current code.md`
3. `Structure learning priority plan.md`
4. `Structure learning change log.md`
5. `GraphExp/main_structure_learning.py`
6. `GraphExp/test_eval.py`

## Locked Evaluation Protocol

Do not change this protocol unless explicitly requested.

- dataset mapping: `sim4.csv -> h4.txt`
- evaluator: `GraphExp/test_eval.py`
- raw export evaluation:
  - `python .\test_eval.py --pred .\results\RUN_DIR\learned_adjacency.csv --gt ..\fMRI_dataset\h4.txt --top_k 61 --adj_convention raw`
- causal export evaluation:
  - `python .\test_eval.py --pred .\results\RUN_DIR\learned_adjacency_causal.csv --gt ..\fMRI_dataset\h4.txt --top_k 61 --adj_convention causal`

## Current Best Known Comparison Points

Historical baseline:

- run: `GraphExp/results/run_20260308_183908`
- F1 = `0.6066`

Current best after recent changes:

- run: `GraphExp/results/run_20260309_215355`
- change stage: `P0-3`
- best epoch = `9`
- best F1 = `0.5738`
- final F1 = `0.0328`

P0-4 alignment result:

- run: `GraphExp/results/run_20260309_220535`
- `selection_top_k = 61`
- best F1 still = `0.5738`
- conclusion: proxy `top_k` alignment improved protocol consistency but did not improve F1 further

## What Has Already Been Changed

The current snapshot already includes:

1. explicit raw vs causal adjacency semantics
2. raw and causal export files saved separately
3. evaluator support for `--adj_convention raw|causal|auto`
4. stabilized weighted proxy score instead of zero-collapsing multiplicative score
5. delayed selection start epoch
6. guarded best-epoch selection with:
   - skeleton retention check
   - density sanity check
   - score-only fallback if no epoch passes guardrails
7. optional `selection_top_k` decoupled from Patel noise-guide `top_k_edges`

## What The Evidence Currently Says

1. The semantic/export/evaluation mismatch is no longer the main blocker.
2. Guarded best-epoch selection is materially better than the old score-only selection in long runs.
3. Long training still causes severe late-stage structural drift.
4. Further P0-style proxy cleanup is now showing diminishing returns.
5. The next likely gain should come from training design that preserves skeleton quality better.

## Agreed Next Priority

Move to `P1-1`: skeleton-first training.

The current recommended direction is:

1. skeleton-first stage
2. direction-assignment stage
3. late stabilization stage

Do not start by strengthening directional losses further.

## Working Rules

The next assistant should keep these rules:

1. update `Structure learning change log.md` immediately when making a meaningful change
2. update `Judgment based on current code.md` immediately after any new empirical result
3. keep using the locked `sim4 -> h4` evaluation protocol for headline comparisons
4. avoid changing both training objective and evaluation protocol in the same batch
5. prefer narrow batches with short-run and 100-epoch validation

## Recommended Prompt For A New Conversation

Paste the following as the first user message in a new chat:

```text
请先阅读以下文件，再继续这个结构学习改造，不要重新发明流程：

1. D:\mockup\DDM-main\Structure learning handoff.md
2. D:\mockup\DDM-main\Judgment based on current code.md
3. D:\mockup\DDM-main\Structure learning priority plan.md
4. D:\mockup\DDM-main\Structure learning change log.md
5. D:\mockup\DDM-main\GraphExp\main_structure_learning.py
6. D:\mockup\DDM-main\GraphExp\test_eval.py

要求：
- 继承已有修改逻辑、评估协议、记录方式
- headline 比较一律使用 sim4.csv -> h4.txt 和 top_k 61
- 每次改动都要立即更新 change log
- 每次新实验结果都要立即更新 judgment 文档
- 先按 handoff 里的“Agreed Next Priority”继续做，不要回到已经验证收益不大的 P0 微调

先告诉我你从这些文件里读到了什么当前结论，以及你准备执行的下一步。
```

## Minimal Alternative If You Do Not Want To Paste A Long Prompt

If you want a shorter start message, use:

```text
请先阅读 D:\mockup\DDM-main\Structure learning handoff.md，并严格按其中列出的文件、评估协议、记录规则继续结构学习改造。先总结当前结论，再执行下一步。
```
