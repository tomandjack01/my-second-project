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

### 2026-04-01 - `main_structure_learning.py` 冗余清理（低风险）

Files:

- `GraphExp/main_structure_learning.py`
- `Structure learning change log.md`

Purpose:

- 删除已确认未引用的辅助函数与未生效 CLI 参数，降低主程序冗余度
- 在不改变当前训练主路径行为的前提下，提升后续维护可读性

Expected effect:

- 不影响现有实验命令（当前仓库内命令不依赖被删参数）
- 代码结构更简洁，减少误读和后续修改冲突面

Observed effect:

- 已删除 2 个未引用函数：
  - `get_current_structure_logits(...)`
  - `compute_dataset_direction_prior_matrix(...)`
- 已删除 3 个未使用/不生效参数定义：
  - `--save_path`
  - `--uniform_timestep`
  - `--noise_zero_mean`
- 已保留实际生效的替代开关：
  - `--per_node_timestep`
  - `--noise_with_mean`
- VS Code 问题诊断（`get_errors`）对主文件无新增报错

Rollback note:

- 建议以本条日志作为清理批次锚点进行回退
- 若后续发现兼容性问题，可基于该批次对 `GraphExp/main_structure_learning.py` 做单文件回滚

### 2026-04-01 - `main_structure_learning.py` 冗余清理（第二轮：废弃注释代码）

Files:

- `GraphExp/main_structure_learning.py`
- `Structure learning change log.md`

Purpose:

- 清理主参数区中注释形式保留的历史废弃参数定义
- 进一步减少阅读噪音，保持参数区与实际可用 CLI 一致

Expected effect:

- 不改变任何运行逻辑（仅删除注释死代码）
- `--help` 输出、训练路径与第一轮清理后保持一致

Observed effect:

- 已删除 4 行注释死代码（均为历史废弃参数定义）：
  - `--pretrain_split_ratio` 的注释定义行
  - `--skip_pretrain` 的旧注释定义行
  - 以及对应两行“已废弃”说明注释
- 保留了真实仍在使用的 `--skip_pretrain` 参数定义

Rollback note:

- 本轮为纯注释清理，可按本条记录直接回退到第一轮状态
- 若需要最小回滚，仅恢复 `GraphExp/main_structure_learning.py` 本轮删除区块即可

### 2026-04-01 - 废弃损失分支清理收口（第三轮：runner 兼容）

Files:

- `GraphExp/main_structure_learning.py`
- `GraphExp/run_cross_pred_v1_final_only_compare.py`
- `Structure learning change log.md`

Purpose:

- 收口已放弃损失分支后的脚本兼容问题，避免历史 sweep runner 继续向主脚本传递已删除参数
- 保持旧 runner 的可用性（用于历史对比 CSV 流程），同时确保它调用当前主训练入口不会触发参数报错

Expected effect:

- `run_cross_pred_v1_final_only_compare.py` 不再传递 `cross_pred_*` / `anti_collapse_*` 到 `main_structure_learning.py`
- 旧 runner 的开关语义通过兼容映射落到现有 `causal_lag_main_*` 参数上
- 若用户尝试启用已删除的 anti-collapse（`lambda>0`），会在 runner 侧直接报错并给出原因

Observed effect:

- `build_command(...)` 已改为仅传递当前存在的参数：
  - `--causal_lag_main_weight`
  - `--causal_lag_main_aggregation`
  - `--causal_lag_main_softmax_temp`
  - `--causal_lag_main_lags`
  - `--causal_lag_main_lag_weights`
- 新增 `resolve_causal_lag_main_weight(...)`：
  - `cross_pred_fixed_weight > 0` 时优先映射为主权重
  - 否则退回 `cross_pred_target_ratio`
- anti-collapse 参数保留在 runner parser 中仅作历史兼容元数据；`anti_collapse_lambdas>0` 现在会直接拒绝执行
- 修复 `main_structure_learning.py` 中一处注释污染：`TIME_Prtyu677OINTS -> TIME_POINTS`

Rollback note:

- 若需回滚兼容映射，可仅恢复 `GraphExp/run_cross_pred_v1_final_only_compare.py` 的 `build_command(...)` 改动块
- 若需保持主脚本最小改动，可独立回滚 `main_structure_learning.py` 的注释修复行

### 2026-04-01 - 废弃损失分支清理收口（第四轮：主脚本核心函数去壳）

Files:

- `GraphExp/main_structure_learning.py`
- `Structure learning change log.md`

Purpose:

- 把主脚本中残留的 cross-pred 核心函数入口（旧命名与旧语义提示）彻底去壳
- 避免后续维护误判为“cross-pred 分支仍在主训练入口中活跃”

Expected effect:

- `main_structure_learning.py` 不再出现 `build_cross_prediction_aggregation_weights(...)` 等 cross-pred 核心函数名
- causal-lag 主路径保持不变，仅命名与提示语义改为当前机制

Observed effect:

- 旧函数 `build_cross_prediction_aggregation_weights(...)` 已替换为 `build_causal_lag_aggregation_weights(...)`
- `compute_causal_lag_main_loss(...)` 已切到新函数调用
- 同步清理了相关 cross-pred 文案：
  - 函数 docstring
  - aggregation 错误提示语

Rollback note:

- 若需回退，本轮只涉及 `GraphExp/main_structure_learning.py` 一处函数重命名与文案替换，可独立回滚

### 2026-04-01 - 废弃损失分支清理收口（第五轮：runner anti-collapse 物理删除）

Files:

- `GraphExp/run_cross_pred_v1_final_only_compare.py`
- `Structure learning change log.md`

Purpose:

- 将 runner 中 anti-collapse 从“兼容保留+运行时拒绝”升级为“完全物理删除”
- 清除 sweep 维度、CSV 字段与对比键中的历史 anti-collapse 噪音，降低维护成本

Expected effect:

- runner CLI 不再提供 `--anti_collapse_*` 参数
- 运行循环、`run_single_experiment(...)` 参数链、聚合与对比输出均不再包含 anti-collapse 字段
- 文件内不再残留 `anti_collapse` 相关字符串与注释壳

Observed effect:

- 已删除 runner parser 中全部 anti-collapse 参数定义：
  - `--anti_collapse_lambdas`
  - `--anti_collapse_margin_values`
  - `--anti_collapse_modes`
  - `--anti_collapse_warmup_epochs`
  - `--anti_collapse_ramp_epochs`
- 已删除 `build_command(...)` / `run_single_experiment(...)` 的 anti-collapse 参数链及对应 row 字段
- 已删除 aggregate/comparison/paired 关键键与输出行中的 anti-collapse 列
- 已删除主循环中的 anti-collapse 嵌套 sweep 维度，并清理运行日志中的相关打印
- 已移除 runner 内部一整段历史注释实现块，避免后续误读
- 验证通过：
  - `python -m py_compile GraphExp/run_cross_pred_v1_final_only_compare.py`
  - `python GraphExp/run_cross_pred_v1_final_only_compare.py --help`（确认无 `--anti_collapse_*`）
  - 最小实跑 smoke（cross on, 1 seed, 1 epoch）成功生成 summary/aggregate 输出

Rollback note:

- 若需回滚本轮，建议仅回滚 `GraphExp/run_cross_pred_v1_final_only_compare.py` 本次提交区块
- 本轮未改动主训练脚本逻辑，回滚影响范围仅限 runner 参数面与汇总 schema

### 2026-04-01 - 方向A选择器重构（第六轮：`causal_lag_primary`）

Files:

- `GraphExp/main_structure_learning.py`
- `Structure learning change log.md`

Purpose:

- 按方向A将 best-epoch 选择从 legacy 启发式偏置中解耦，降低对早期高 margin/asymmetry 的过度偏好
- 增加一个“causal-lag 主导，Patel/骨架/密度弱 tie-break”的可选评分模式，且保持旧模式完全可回归

Expected effect:

- 新增 `selection_score_mode=causal_lag_primary` 后，checkpoint 评分可由单主体 `causal_lag_reverse_minus_forward` 主导
- 旧模式（`legacy`、`causal_lag_composite`、`causal_lag_entropy_composite`）行为保持不变
- CLI、训练调用、配置导出和 quality 记录对新模式全链路贯通

Observed effect:

- `compute_epoch_quality(...)` 已新增：
  - `score_mode='causal_lag_primary'`
  - primary 分量：
    - `primary_causal_lag_weight`
    - `primary_soft_tiebreak_weight`
    - `primary_skeleton_tiebreak_weight`
    - `primary_density_tiebreak_weight`
  - 返回详情新增 `score_primary_total` 及对应 term 字段
- `train_brain_connectivity(...)` 已新增上述四个 `selection_primary_*` 参数、参数合法性检查、打印与调用透传
- CLI 已新增：
  - `--selection_score_mode` choice 包含 `causal_lag_primary`
  - 四个 `--selection_primary_*` 参数
- `config.npy` 导出已包含四个 `selection_primary_*` 配置键
- 验证通过：
  - `python -m py_compile GraphExp/main_structure_learning.py`
  - `python GraphExp/main_structure_learning.py --help`（确认新 mode 与四个 primary 参数均显示）
  - 最小实跑 smoke（CPU, 1 epoch, subject_limit=1, time_limit=20）成功，日志中出现：
    - `Selection score mode: causal_lag_primary`
    - `Selection primary weights: lag=... / soft_tiebreak=... / skeleton_tiebreak=... / density_tiebreak=...`

Rollback note:

- 若需回滚本轮，优先回滚 `GraphExp/main_structure_learning.py` 中 selector 相关增量（`causal_lag_primary` 与 `selection_primary_*` 参数链）
- 本轮为“新增可选模式”而非替换旧模式，回滚后不会影响既有 legacy/composite 路径
