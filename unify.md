# Unified Best-Config Experiment Log

更新日期：2026-04-12

## 1. 目标与边界

- 目标：验证是否存在一套足够统一的配置，可以在 `fMRI`、`sim2`、`sim3`、`sim4` 四个数据集上替代当前各自的最佳 run，而不显著损害性能和稳定性。
- 本文档的目标不是“最少 run 数”，而是“结论可归因、可复核、可持续追加”。
- 统一的定义：
  - 尽量统一训练主干、方向监督策略、checkpoint 选择策略。
  - 允许保留与数据规模直接绑定的参数：`epochs`、`top_k_edges`。
  - `detach_direction_from_main_after_epoch` 的公式化自适应不在主方案决策阶段引入，只有在统一主干确定后才单独验证。
- 核心原则：
  - `sim3` 是第一关键验证点，因为它是当前四个最佳 run 中唯一明显走了不同路径的数据集。
  - `sim4` 必须作为第二关键验证点；不能把它仅仅放在最后“顺带确认”。
  - `sim2` 与 `fMRI` 主要用于确认统一方案的可转移性，不用于决定机制。

## 2. 当前最佳 Run 基线

来源：

- `GraphExp/results/best_run_summary_20260406_211725.csv`
- `GraphExp/results/best_run_config_table_20260406_211725.md`

| Dataset | Best Run | Best / Exported / Final `primary_strict_f1` | Epochs | Seed | 当前最佳配置中的关键路径 |
| --- | --- | --- | ---: | ---: | --- |
| `fMRI` | `run_20260330_155759` | `1.0000 / 1.0000 / 1.0000` | 100 | 11 | `patel_kappa` init, Patel direction, `subject`, 无 detach, 有 pretrain |
| `sim2` | `run_20260405_112520` | `0.8182 / 0.7273 / 0.8182` | 40 | 11 | `random` init, 无 Patel direction, `causal_lag_main_weight=0.25`, `subject`, 有 pretrain |
| `sim3` | `run_20260331_191723` | `0.9444 / 0.9444 / 0.9444` | 30 | 11 | `patel_kappa` init, Patel direction, `batch_mean`, 无 causal-lag, 有 pretrain |
| `sim4` | `run_20260404_111017` | `0.8852 / 0.8525 / 0.8689` | 40 | 11 | `random` init, Patel direction, `causal_lag_main_weight=0.25`, `subject`, 有 pretrain |

结论：

- `sim3` 是当前唯一明显偏离统一方向的数据点。
- 但 `sim4` 仍然是必须共同验证的 gate，因为历史 evidence 表明真正的稳定性瓶颈主要在 `sim3/sim4`，而不是 `sim2/fMRI`。

## 3. 已有证据与硬约束

### 3.1 已有历史结论

- `optimizer_step_mode=batch_mean` 不应直接提升为默认配置。
  - 证据：`GraphExp/CROSS_PRED_V1_TRACKER.md:3758`
- `selection_agreement_weight=0.0` 已经是当前主线常见选择。
  - 证据：`GraphExp/CROSS_PRED_V1_TRACKER.md:5422`, `:5487`, `:5582`, `:5651`, `:5775`, `:5867`
- “固定 support + diffusion-only” 不足以自动学出方向。
  - 证据：`GraphExp/CROSS_PRED_V1_TRACKER.md:5447`
- 当前推荐 Option A 分支已经有全 benchmark evidence，但瓶颈是数据集相关的：
  - `sim3/sim4`：direction retention / final gap
  - `fMRI`：checkpoint selection
  - 证据：`GraphExp/CROSS_PRED_V1_TRACKER.md:5920`

### 3.2 代码层面的关键耦合点

- `warmup_then_orthogonal` 不是普通超参，而是显式改梯度路由。
  - 定义与行为：`GraphExp/main_structure_learning.py:793`, `:833`, `:4176`
- `causal_lag_main_weight` 会改变训练目标，而不是仅仅改变评估。
  - 参数定义与使用：`GraphExp/main_structure_learning.py:2425`, `:2804`, `:3289`
- `selection_score_mode` 属于 checkpoint 选择逻辑，不应与训练因素混为一谈。
  - 约束与使用：`GraphExp/main_structure_learning.py:2557`, `:3521`
- `selection_agreement_weight` 进入的是 epoch quality / selector，而不是训练主损失。
  - 使用点：`GraphExp/main_structure_learning.py:3521`
- `fixed_support_mask_mode=maxgap_kappa` 仍然依赖 `support_prior_mode` 来构造 support skeleton。
  - 参数与构造：`GraphExp/main_structure_learning.py:4021`, `:4055`, `:4526`, `:4589`
- `build_structure_init_matrix` 与 `build_support_prior_matrix` 是两个不同入口。
  - 定义：`GraphExp/main_structure_learning.py:343`, `:378`

推论：

- “去掉 Patel direction”不等于“整个框架已经去 Patel”。
- `1B` 式的一次性大包改动会同时改训练目标、梯度流和 selector，结果不可归因。

## 4. 统一实验的评估协议

### 4.1 固定实验协议

- Seeds：`11,22,33,44,55`
- 所有比较必须做 paired comparison，同一组 seed 对同一组候选进行对照。
- 每个数据集先沿用当前最佳 run 对应的原始 `epochs`：
  - `fMRI = 100`
  - `sim2 = 40`
  - `sim3 = 30`
  - `sim4 = 40`
- `top_k_edges = GT edge count`
- 在主方案未确定前，不引入新的经验公式，例如：
  - `detach_epoch = max(15, int(epochs * 0.55))`

### 4.2 报告指标

主指标：

- `best_primary_strict_f1`
- `exported_primary_strict_f1`
- `final_primary_strict_f1`

稳定性指标：

- `best_final_gap = best - final`
- `exported_best_gap = exported - best`
- `strict_f1 @ eps=0.1`
- `failure_mode`
- `gt_signed_margin_median`

补充指标：

- `strict_precision`
- `strict_recall`
- `strict_pred_count`
- `adj_eff_parents_mean`

### 4.3 判定规则

候选配置要被提升为“统一候选”，至少要满足以下条件：

- 在 `sim3` 和 `sim4` 上都通过；任何一个失败都不提升。
- 相对 incumbent 的 paired mean delta，满足：
  - `delta(best_primary_strict_f1) >= -0.03`
  - `delta(final_primary_strict_f1) >= -0.03`
- `best_final_gap` 不能比 incumbent 恶化超过 `0.02`
- `strict_f1 @ eps=0.1` 不能下降超过 `0.05`
- `failure_mode` 不得从较轻问题退化到更严重问题，例如：
  - `weak_asymmetry -> symmetric_collapse`

建议同时报告：

- paired bootstrap CI
- paired signed-rank test
- seed-wise delta 列表

说明：

- 不再使用诸如“`>= 0.90` 就通过”的绝对阈值。
- 统一配置的判断必须是“相对当前强基线非劣效”，而不是“达到某个孤立数值”。

## 5. 实验阶段设计

## 5.1 Phase 0：控制组与 Selector-only 复核

目标：

- 固定比较基线。
- 先把训练因素和 selector 因素拆开。

### Phase 0A：控制组重跑

数据集：

- `sim3`
- `sim4`

运行：

- 重跑当前各自 incumbent 配置，seeds=`11,22,33,44,55`

目的：

- 建立和后续实验同 seed、同批次、同日志格式的正式 control。

### Phase 0B：Selector-only post-hoc rescoring

在不重训的前提下，对同一批训练轨迹做 checkpoint 复核。优先使用现有 `quality_history.csv` 和 selector 相关日志离线完成；如果现有脚本不足，再补离线脚本，但不混入训练因素实验。

比较项：

- `selection_score_mode in {legacy, causal_lag_composite, causal_lag_entropy_composite}`
- `selection_agreement_weight in {0.0, 0.25}`

目的：

- 回答“问题在训练，还是在 checkpoint 选择”。

输出：

- `sim3` 与 `sim4` 的 selector 基线
- 后续训练实验以本阶段胜出的 selector 基线为准；如果不同数据集的胜者不一致，则在对应数据集上分别固定，并在 cross-dataset gate 中额外做 selector-only replay 对照

## 5.2 Phase 1：sim3 低风险训练因素隔离

控制组：

- `S0 = sim3 incumbent`

本阶段默认固定 selector：

- `selection_score_mode = causal_lag_entropy_composite`

在本阶段，以下因素一次只改一个，其余保持与 `S0` 完全一致：

| ID | 只改什么 | 不改什么 | 要回答的问题 |
| --- | --- | --- | --- |
| `S1` | `optimizer_step_mode: batch_mean -> subject` | 保留 Patel direction、无 causal-lag、原 selector | `subject` 是否能安全替代 `batch_mean` |
| `S2` | `pretrain: on -> off` | 其他全不改 | 移除 pretrain 是否会伤害/改善 `sim3` |
| `S3` | `structure_init_mode: patel_kappa -> random` | 其他全不改 | random init 是否能替代 Patel init |
| `S4` | `directional_kappa_gate: False -> True` | 其他全不改 | gate 是否是统一化所需且不伤 sim3 |
| `S5` | 合并所有单因素通过项 | 仍保留 Patel direction、无 causal-lag、固定 selector | 低风险统一主干是否可成立 |

通过条件：

- 与 `S0` 比较，满足第 4 节的判定规则。

停止条件：

- 如果某单因素明显失败，则不进入 `S5` 的组合。

## 5.3 Phase 2：sim3 方向监督替代链

前提：

- 以 `S5` 为基础；如果 `S5` 未通过，则以 Phase 1 中表现最好的版本作为新 control。

原则：

- 一次只回答一个问题。
- 不把 training objective、gradient routing、selector 改动打包。

| ID | 只改什么 | 目的 |
| --- | --- | --- |
| `D0` | Phase 1 胜出 control | 作为方向监督替代链的零点 |
| `D1` | `causal_lag_main_weight: 0.0 -> 0.25` | causal-lag 加进来本身是否有益 |
| `D2` | 在 `D1` 基础上关闭 Patel directional supervision | causal-lag 能否在不改 routing 的前提下替代 Patel direction |
| `D3` | 在 `D2` 基础上 `gradient_routing_mode -> warmup_then_orthogonal` | 改梯度路由后是否进一步稳定 direction branch |
| `D4` | 可选：`support_prior_mode: patel_kappa -> pearson_abs` | 最大程度减少 Patel 先验接触点是否仍可接受 |

说明：

- `D2` 回答“能否去掉 Patel direction”。
- `D3` 回答“是否需要新的梯度路由来支撑 causal-lag 替代 Patel direction”。
- `D4` 是更激进的去先验化，不参与第一轮统一主方案判断。

## 5.4 Phase 3：sim4 Gate

目标：

- 验证 `sim3` 胜出方案是否能在第二关键难点数据集上成立。

运行候选：

- `G0 = sim4 incumbent control`
- `G1 = Phase 1 中最强的 Patel-family 统一主干候选`
- `G2 = Phase 2 中最强的“去 Patel direction”候选`

固定：

- seeds=`11,22,33,44,55`
- `epochs=40`
- 训练侧 selector 沿用 `sim3` 阶段固定配置；同时必须补充 `legacy` 的 offline replay 结果，避免把 `sim4` 的 selector miss 误判成 training 失败

判定：

- 只有在 `sim4` 也满足第 4 节非劣效标准的候选，才有资格进入全数据集确认。

## 5.5 Phase 4：全数据集确认

目标：

- 验证最终统一候选是否能在全套 benchmark 上成立。

比较：

- `U0 = 各数据集 incumbent`
- `U1 = 最终统一候选`

数据集：

- `fMRI`
- `sim2`
- `sim3`
- `sim4`

说明：

- `sim2` 与 `fMRI` 在这一阶段是 confirmatory，而不是 mechanism gate。
- `fMRI` 的结果解释要特别关注：
  - 是 training 失败
  - 还是 selector 失败

## 5.6 Phase 5：公式化自适应参数验证（可选）

仅在 Phase 4 通过后执行。

候选项：

- `detach_direction_from_main_after_epoch = max(15, int(epochs * 0.55))`

比较：

- 与最终统一候选中的显式固定值版本比较

目的：

- 验证“统一主干 + 少量公式化数据集自适应”是否能够进一步简化配置，而不损害稳定性。

## 6. 立即执行顺序

当前推荐顺序如下：

1. Phase 0A：重跑 `sim3` / `sim4` control
2. Phase 0B：selector-only rescoring
3. Phase 1：`S1` 到 `S4` 单因素
4. Phase 1：`S5` 组合
5. Phase 2：`D1` 到 `D3`
6. Phase 3：`sim4` gate
7. Phase 4：全数据集确认
8. Phase 5：公式化自适应参数

## 7. 当前状态

- [x] 统一实验方法学冻结
- [x] Phase 0A 控制组重跑
- [x] Phase 0B selector-only rescoring
- [x] Phase 1 sim3 低风险训练因素隔离
- [x] Phase 2 sim3 方向监督替代链
- [x] Phase 3 sim4 gate
- [ ] Phase 4 全数据集确认
- [ ] Phase 5 公式化自适应参数验证

当前阻塞：

- `Phase 3` 没有产生通过 sim4 gate 的统一候选，因此 `Phase 4` 暂不启动。

## 8. 已知历史结果摘要

以下结果来自既有 tracker，用于后续判断，不替代本方案下的新 control。

| ID | 数据集 | 历史结论 |
| --- | --- | --- |
| `H-sim3-formal` | `sim3` | `support_direction + maxgap_kappa + random + Patel direction` 的 formal 结果显示：best 强，但 final 有 retention gap；`best_strict_f1@eps=0 = 0.8222 +- 0.0544`，`final = 0.7778 +- 0.0497`。来源：`GraphExp/CROSS_PRED_V1_TRACKER.md:4872`, `:4878` |
| `H-sim4-formal` | `sim4` | 同类分支在 `sim4` 上也存在明显 final gap；`best_strict_f1@eps=3e-4 = 0.7915 +- 0.0217`，`final = 0.7272 +- 0.0481`。来源：`GraphExp/CROSS_PRED_V1_TRACKER.md:4887`, `:4893` |
| `H-diff-only-sim3` | `sim3` | 固定 support 的 diffusion-only 无法学出方向。来源：`GraphExp/CROSS_PRED_V1_TRACKER.md:5447` |
| `H-main-loss-ablation-sim3` | `sim3` | diffusion 能提高 best ceiling，但会重新引入 retention 问题。来源：`GraphExp/CROSS_PRED_V1_TRACKER.md:5507`, `:5521` |
| `H-sim2-formal` | `sim2` | 当前推荐 retention-fix 分支能稳定转移到 `sim2`，但不是强 margin 解。来源：`GraphExp/CROSS_PRED_V1_TRACKER.md:5823` |
| `H-fMRI-formal` | `fMRI` | 当前推荐分支在 `fMRI` 上更像 selector 问题而非训练上限问题；`strict_f1@eps=0 = 0.7200`。来源：`GraphExp/CROSS_PRED_V1_TRACKER.md:5893`, `:5909` |

## 9. 后续实验结果记录区

记录规则：

- 每完成一个实验，就追加一个条目。
- 结果只追加，不覆盖；如果需要修正，新增“Correction”条目说明。
- 必须同时写：配置变化、数据集、seeds、原始 artifact 路径、核心指标、结论、是否推进下一阶段。

### 9.1 结果记录模板

```md
### [实验ID] 标题

- 日期：
- 阶段：
- 数据集：
- Seeds：
- 对照组：
- 仅改动的参数：
- 固定不变的参数：
- 训练相关 artifact：
- 汇总 artifact：

结果：

- best_primary_strict_f1：
- exported_primary_strict_f1：
- final_primary_strict_f1：
- best_final_gap：
- exported_best_gap：
- strict_f1@eps=0.1：
- failure_mode：
- gt_signed_margin_median：

判定：

- 是否通过：
- 相对 control 的 paired delta：
- 是否推进下一步：

结论：

- 
```

### 9.2 待填写结果

### [P0A-sim3-control] sim3 incumbent 5-seed control

- 日期：2026-04-09 至 2026-04-10
- 阶段：Phase 0A
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：当前 `sim3` incumbent，基于 `run_20260331_191723`
- 仅改动的参数：
  - 无；按保存的最佳 run 配置回放，仅更换 seed
- 固定不变的参数：
  - `structure_init_mode = patel_kappa`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `optimizer_step_mode = batch_mean`
  - `directional_loss_end_epoch = 15`
  - `causal_lag_main_weight = 0.0`
  - `selection_score_mode = causal_lag_entropy_composite`
  - `selection_agreement_weight = 0.25`
  - `epochs = 30`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260409_221238`
  - `GraphExp/results/run_20260409_222253`
  - `GraphExp/results/run_20260409_223057`
  - `GraphExp/results/run_20260409_223851`
  - `GraphExp/results/run_20260409_224727`
- 汇总 artifact：
  - `GraphExp/results/unify_phase0_control_sim3_20260410_003500_phase0_sim3_control_full.csv`
  - `GraphExp/results/unify_phase0_control_sim3_20260410_003500_phase0_sim3_control_full_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim3_20260410_003900_rich.csv`
  - `GraphExp/results/unify_phase0_control_sim3_20260410_003900_rich_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.6889 +- 0.1474`
- `exported_primary_strict_f1 = 0.6889 +- 0.1474`
- `final_primary_strict_f1 = 0.6889 +- 0.1474`
- `best_final_gap = 0.0000 +- 0.0000`
- `exported_best_gap = 0.0000 +- 0.0000`
- `best_strict_f1@eps=0.1 = 0.2213 +- 0.1202`
- `exported_strict_f1@eps=0.1 = 0.2215 +- 0.1401`
- `final_strict_f1@eps=0.1 = 0.2215 +- 0.1401`
- `best_signed_margin_median = 0.0299 +- 0.0205`
- `exported_signed_margin_median = 0.0217 +- 0.0144`
- `final_signed_margin_median = 0.0217 +- 0.0144`
- `best failure_mode counts = {'mixed_or_partial': 2, 'weak_asymmetry': 3}`
- `final failure_mode counts = {'weak_asymmetry': 5}`

Seed 级摘要：

- `seed=11`：`best/exported/final = 0.9444 / 0.9444 / 0.9444`
- `seed=22`：`0.7222 / 0.7222 / 0.7222`
- `seed=33`：`0.6111 / 0.6111 / 0.6111`
- `seed=44`：`0.6667 / 0.6667 / 0.6667`
- `seed=55`：`0.5000 / 0.5000 / 0.5000`

判定：

- 是否通过：作为 control 已完成，不涉及通过/不通过
- 是否推进下一步：是，推进到 `Phase 0B`

结论：

- 当前 `sim3` incumbent 的单 seed 最优值可以复现，但多种子稳定性很差。
- `seed=11` 明显是高值离群点；以 5 seeds 看，当前 `sim3` incumbent 不能被视为稳健统一基线。
- 该结论强化了 `sim3` 作为第一关键 gate 的必要性。

### [P0A-sim4-control] sim4 incumbent 5-seed control

- 日期：2026-04-09 至 2026-04-10
- 阶段：Phase 0A
- 数据集：`sim4`
- Seeds：`11,22,33,44,55`
- 对照组：当前 `sim4` incumbent，基于 `run_20260404_111017`
- 仅改动的参数：
  - 无；按保存的最佳 run 配置回放，仅更换 seed
- 固定不变的参数：
  - `structure_init_mode = random`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `optimizer_step_mode = subject`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `selection_score_mode = causal_lag_composite`
  - `selection_agreement_weight = 0.0`
  - `epochs = 40`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260409_225715`
  - `GraphExp/results/run_20260409_231925`
  - `GraphExp/results/run_20260409_234129`
  - `GraphExp/results/run_20260410_000340`
  - `GraphExp/results/run_20260410_002543`
- 汇总 artifact：
  - `GraphExp/results/unify_phase0_control_sim4_20260409_225712_phase0_sim4_control.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260409_225712_phase0_sim4_control_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.8623 +- 0.0131`
- `exported_primary_strict_f1 = 0.8098 +- 0.0459`
- `final_primary_strict_f1 = 0.8295 +- 0.0368`
- `best_final_gap = 0.0328 +- 0.0344`
- `exported_best_gap = 0.0525 +- 0.0334`
- `best_strict_f1@eps=0.1 = 0.0245 +- 0.0354`
- `exported_strict_f1@eps=0.1 = 0.0000 +- 0.0000`
- `final_strict_f1@eps=0.1 = 0.0372 +- 0.0295`
- `best_signed_margin_median = 0.0085 +- 0.0013`
- `exported_signed_margin_median = 0.0093 +- 0.0007`
- `final_signed_margin_median = 0.0076 +- 0.0005`
- `best failure_mode counts = {'symmetric_collapse': 5}`
- `final failure_mode counts = {'symmetric_collapse': 5}`

Seed 级摘要：

- `seed=11`：`best/exported/final = 0.8852 / 0.8852 / 0.8689`
- `seed=22`：`0.8525 / 0.7541 / 0.8525`
- `seed=33`：`0.8525 / 0.7869 / 0.7705`
- `seed=44`：`0.8689 / 0.8361 / 0.8033`
- `seed=55`：`0.8525 / 0.7869 / 0.8525`

判定：

- 是否通过：作为 control 已完成，不涉及通过/不通过
- 是否推进下一步：是，推进到 `Phase 0B`

结论：

- 当前 `sim4` incumbent 的 primary strict F1 明显比 `sim3` 稳定，seed 方差较小。
- 但它的 `exported` 与 `final` 均明显低于 `best`，说明 selector / retention 依然是第一类问题。
- `strict_f1@eps=0.1` 几乎为零，说明该分支主要依赖低 margin 解。
- 所有 seeds 都落在 `symmetric_collapse`，说明它并不是“高置信方向解”。

### [P0B-sim3-selector] sim3 selector-only rescoring

- 日期：2026-04-10
- 阶段：Phase 0B
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：`P0A-sim3-control`
- 仅改动的参数：
  - 不重训，只离线改 selector
  - `selection_score_mode in {legacy, causal_lag_composite, causal_lag_entropy_composite}`
  - `selection_agreement_weight in {0.0, 0.25}`
- 固定不变的参数：
  - 所有训练轨迹、模型权重、guardrail、`selection_start_epoch`
  - 仍按 `quality_history.csv` 中记录的 `selection_eligible -> fallback_best -> best_guarded -> final choice` 逻辑离线复现
- 训练相关 artifact：
  - `GraphExp/replay_selector_modes.py`
  - `GraphExp/results/run_20260409_221238`
  - `GraphExp/results/run_20260409_222253`
  - `GraphExp/results/run_20260409_223057`
  - `GraphExp/results/run_20260409_223851`
  - `GraphExp/results/run_20260409_224727`
- 汇总 artifact：
  - `GraphExp/results/unify_phase0_selector_sim3_20260410_082446_phase0b.csv`
  - `GraphExp/results/unify_phase0_selector_sim3_20260410_082446_phase0b_aggregate.csv`

结果：

| Selector 设置 | 选中 epoch | `chosen_primary_strict_f1` | `strict_f1@eps=0.1` | failure_mode | 相对当前 exported delta |
| --- | ---: | ---: | ---: | --- | ---: |
| `legacy, agreement=0.0` | `6.0 +- 0.0` | `0.5333 +- 0.1879` | `0.2326 +- 0.1431` | `mixed_or_partial:2, weak_asymmetry:3` | `-0.1556 +- 0.0737` |
| `legacy, agreement=0.25` | `15.2 +- 6.5` | `0.6444 +- 0.2155` | `0.2520 +- 0.0680` | `mixed_or_partial:3, weak_asymmetry:2` | `-0.0444 +- 0.0889` |
| `causal_lag_composite, agreement=0.0/0.25` | `6.0 +- 0.0` | `0.5333 +- 0.1879` | `0.2326 +- 0.1431` | `mixed_or_partial:2, weak_asymmetry:3` | `-0.1556 +- 0.0737` |
| `causal_lag_entropy_composite, agreement=0.0/0.25` | `30.0 +- 0.0` | `0.6889 +- 0.1474` | `0.2215 +- 0.1401` | `weak_asymmetry:5` | `0.0000 +- 0.0000` |

补充检查：

- 原始线上设置 `causal_lag_entropy_composite + agreement=0.25` 被离线 replay `5/5` 精确复现：
  - `matches_trained_exported_epoch = 5/5`
  - `matches_trained_exported_primary_strict_f1 = 5/5`
- 对 `causal_lag_composite` 与 `causal_lag_entropy_composite` 来说，`selection_agreement_weight` 不参与当前 score 计算，所以 `0.0` 与 `0.25` 的 replay 结果完全相同。

判定：

- 是否通过：作为 selector 基线复核已完成
- `sim3` 胜出 selector：`causal_lag_entropy_composite`
- 是否推进下一步：是，推进到 `Phase 1`

结论：

- `sim3` 的问题不只是 training；selector 也会显著改变结论。
- 但在当前 `sim3` control 轨迹上，`causal_lag_entropy_composite` 仍然是主指标最优的 selector。
- `legacy` 尤其是 `agreement=0.0` 会把选择明显拉回过早 epoch（平均 `epoch=6`），造成主指标大幅下降。
- `legacy, agreement=0.25` 虽然提高了 `strict_f1@eps=0.1`，但主指标仍低于当前 exported，不能作为 `sim3` 的新 selector baseline。

保留意见：

- 不能把 `sim3` 上的 entropy 行为直接表述为“完全没有区分力”。
- 目前更准确的说法是：在这 5 条 `sim3 control` 轨迹上，`causal_lag_entropy_composite` 的 argmax 都稳定落在最后一个 eligible epoch（`epoch 30`）。
- 但它并不是在所有 seed 上都严格单调上升；因此只能说“当前最优点稳定落在最后一个 eligible epoch”，还不能说“score 完全退化成固定选最后 epoch”。 
- 后续 `Phase 1` 需要把这一点作为附带观测项：如果训练改动后 entropy 开始在不同 epoch 之间做出稳定选择，才说明 selector 获得了更有信息量的排序空间。

### [P0B-sim4-selector] sim4 selector-only rescoring

- 日期：2026-04-10
- 阶段：Phase 0B
- 数据集：`sim4`
- Seeds：`11,22,33,44,55`
- 对照组：`P0A-sim4-control`
- 仅改动的参数：
  - 不重训，只离线改 selector
  - `selection_score_mode in {legacy, causal_lag_composite, causal_lag_entropy_composite}`
  - `selection_agreement_weight in {0.0, 0.25}`
- 固定不变的参数：
  - 所有训练轨迹、模型权重、guardrail、`selection_start_epoch`
  - 仍按 `quality_history.csv` 中记录的 `selection_eligible -> fallback_best -> best_guarded -> final choice` 逻辑离线复现
- 训练相关 artifact：
  - `GraphExp/replay_selector_modes.py`
  - `GraphExp/results/run_20260409_225715`
  - `GraphExp/results/run_20260409_231925`
  - `GraphExp/results/run_20260409_234129`
  - `GraphExp/results/run_20260410_000340`
  - `GraphExp/results/run_20260410_002543`
- 汇总 artifact：
  - `GraphExp/results/unify_phase0_selector_sim4_20260410_082446_phase0b.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260410_082446_phase0b_aggregate.csv`

结果：

| Selector 设置 | 选中 epoch | `chosen_primary_strict_f1` | `strict_f1@eps=0.1` | failure_mode | 相对当前 exported delta |
| --- | ---: | ---: | ---: | --- | ---: |
| `causal_lag_composite, agreement=0.0/0.25` | `21.0 +- 7.4` | `0.8098 +- 0.0459` | `0.0000 +- 0.0000` | `symmetric_collapse:5` | `0.0000 +- 0.0000` |
| `causal_lag_entropy_composite, agreement=0.0/0.25` | `18.8 +- 4.1` | `0.8000 +- 0.0241` | `0.0000 +- 0.0000` | `symmetric_collapse:5` | `-0.0098 +- 0.0422` |
| `legacy, agreement=0.0/0.25` | `33.8 +- 12.4` | `0.8361 +- 0.0274` | `0.0372 +- 0.0295` | `symmetric_collapse:5` | `+0.0262 +- 0.0493` |

补充检查：

- 原始线上设置 `causal_lag_composite + agreement=0.0` 被离线 replay `5/5` 精确复现：
  - `matches_trained_exported_epoch = 5/5`
  - `matches_trained_exported_primary_strict_f1 = 5/5`
- 在当前 `sim4` control 轨迹上，`legacy` 的 `agreement=0.0` 与 `0.25` 选中了完全相同的 epochs，因此两者结果一致。
- `legacy` 相对当前 `final` 仍有小幅提升：
  - `chosen_vs_trained_final_delta_primary_strict_f1 = +0.0066 +- 0.0131`
- 但相对每个 seed 的 `gt_best` 仍有差距：
  - `chosen_vs_gt_best_delta_primary_strict_f1 = -0.0262 +- 0.0266`

判定：

- 是否通过：作为 selector 基线复核已完成
- `sim4` 胜出 selector：`legacy`
- 是否推进下一步：是，但要把 `sim4` 的 selector 敏感性单独记录进后续 gate 解释

结论：

- `sim4` 当前 control 的主要问题更偏 selector，而不是 ceiling 不足。
- 仅做 selector-only replay，不重训，就能把 `exported_primary_strict_f1` 从 `0.8098` 提到 `0.8361`。
- 但 failure mode 没有变化，所有 seeds 仍是 `symmetric_collapse`，说明它不是“方向质量已经解决，只差 checkpoint”的完整解。
- `causal_lag_entropy_composite` 在 `sim4` 上反而略差于当前 exported，不适合作为 `sim4` baseline selector。

保留意见：

- 不能把 `sim4` 的结论简化成“直接选最后一个 epoch 就行”。
- 当前 `legacy` replay 的确明显偏向更晚的 epoch，但 5 个 seeds 中并不是全部都选到最后一个 epoch。
- 因而现阶段只能说：`sim4` 存在晚期偏好，且当前 `causal_lag_composite` 对后期优质 checkpoint 的排序不足；还不能把 `last-feasible-epoch` 当作可替代 selector 的规则。
- 同样地，`agreement_weight` 在这批 `sim4 control` 轨迹上没有改变最终选中的 epoch/F1，但这不等于它对 score 数值本身完全没有影响。

### Phase 1/2 固定 selector 说明

- `sim3` 的后续训练因素隔离（`S1-S5`, `D1-D4`）默认固定使用 `causal_lag_entropy_composite`。
- `sim4` 在后续 gate 中必须保留 `legacy` 这个 selector-only 上界参考；否则会把 selector miss 误判成 training 失败。
- 目前还不存在一个在 `sim3` 与 `sim4` control 轨迹上同时胜出的单一 selector，因此“统一 selector”不能先验假定成立，必须等训练主干变化后再复核。

### [Correction-2026-04-10] sim3 incumbent pretrain state

- 经核实，`run_20260331_191723` 的保存配置为：
  - `skip_pretrain = False`
  - `pretrain_epochs = 50`
- 原始 run 目录中也存在 `pretrained_encoder.pt`。
- 因此，先前文档里把 `sim3` incumbent 记为“无 pretrain”是错误的。
- 本文档已修正：
  - 基线表中的 `sim3` 关键路径改为“有 pretrain”
  - `S2` 由原先误写的 `pretrain: off -> on` 改为真实可检验的 `pretrain: on -> off`

### [S1] sim3 optimizer `batch_mean -> subject`

- 日期：2026-04-10
- 阶段：Phase 1
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：`P0A-sim3-control`
- 仅改动的参数：
  - `optimizer_step_mode: batch_mean -> subject`
- 固定不变的参数：
  - 保留 Patel direction
  - `causal_lag_main_weight = 0.0`
  - `structure_init_mode = patel_kappa`
  - `directional_kappa_gate = False`
  - selector 固定为 `causal_lag_entropy_composite`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260410_085121`
  - `GraphExp/results/run_20260410_085912`
  - `GraphExp/results/run_20260410_090707`
  - `GraphExp/results/run_20260410_091503`
  - `GraphExp/results/run_20260410_092250`
- 汇总 artifact：
  - `GraphExp/results/unify_replay_sim3_20260410_085118_phase1_s1_subject.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_085118_phase1_s1_subject_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.8222 +- 0.0544`
- `exported_primary_strict_f1 = 0.6778 +- 0.1333`
- `final_primary_strict_f1 = 0.7778 +- 0.0351`
- `best_final_gap = 0.0444 +- 0.0416`
- `exported_best_gap = 0.1444 +- 0.1343`
- `best_strict_f1@eps=0.1 = 0.5181 +- 0.0983`
- `exported_strict_f1@eps=0.1 = 0.0811 +- 0.0751`
- `final_strict_f1@eps=0.1 = 0.5146 +- 0.0541`
- `best_signed_margin_median = 0.0821 +- 0.0483`
- `exported_signed_margin_median = 0.0180 +- 0.0160`
- `final_signed_margin_median = 0.0627 +- 0.0269`
- `best failure_mode counts = {'mixed_or_partial': 4, 'weak_asymmetry': 1}`
- `final failure_mode counts = {'mixed_or_partial': 3, 'weak_asymmetry': 2}`

附带观测：

- entropy selector 的 exported 选点不再固定在最后一个 epoch：
  - `exported epochs = [9, 11, 10, 12, 10]`
  - `exported_eq_final_frac = 0.0`
- 说明在 `S1` 轨迹上，selector 已经开始有区分力，但当前 exported 选点仍明显偏早。

判定：

- 是否通过：否，未通过正式提升标准
- 相对 control 的 paired delta：
  - `delta(best_primary_strict_f1) = +0.1333`
  - `delta(exported_primary_strict_f1) = -0.0111`
  - `delta(final_primary_strict_f1) = +0.0889`
  - `delta(best_final_gap) = +0.0444`
  - `delta(final_strict_f1@eps=0.1) = +0.2931`
- 是否推进下一步：不进入 `S5` 的正式通过项集合，但保留为后续 side-branch 候选

结论：

- `subject` optimizer 明显提高了 `sim3` 的 best / final ceiling 和高 margin 指标。
- 但它也明显加剧了 `best -> exported` 与 `best -> final` 的 gap，未满足 Phase 1 的 retention 约束。
- 因而 `S1` 不能直接作为统一主干通过项，但值得在后续和更强 routing / direction-control 机制一起复查。

### [S2] sim3 pretrain `on -> off`

- 日期：2026-04-10
- 阶段：Phase 1
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：`P0A-sim3-control`
- 仅改动的参数：
  - `skip_pretrain: False -> True`
- 固定不变的参数：
  - 保留 Patel direction
  - `causal_lag_main_weight = 0.0`
  - `structure_init_mode = patel_kappa`
  - `directional_kappa_gate = False`
  - selector 固定为 `causal_lag_entropy_composite`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260410_093127`
  - `GraphExp/results/run_20260410_093606`
  - `GraphExp/results/run_20260410_094106`
  - `GraphExp/results/run_20260410_094619`
  - `GraphExp/results/run_20260410_095143`
- 汇总 artifact：
  - `GraphExp/results/unify_replay_sim3_20260410_093125_phase1_s2_no_pretrain.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_093125_phase1_s2_no_pretrain_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.5556 +- 0.1685`
- `exported_primary_strict_f1 = 0.4556 +- 0.1625`
- `final_primary_strict_f1 = 0.4556 +- 0.1625`
- `best_final_gap = 0.1000 +- 0.0956`
- `exported_best_gap = 0.1000 +- 0.0956`
- `best_strict_f1@eps=0.1 = 0.2194 +- 0.1292`
- `exported_strict_f1@eps=0.1 = 0.1139 +- 0.0302`
- `final_strict_f1@eps=0.1 = 0.0992 +- 0.0049`
- `best_signed_margin_median = 0.0068 +- 0.0271`
- `exported_signed_margin_median = -0.0072 +- 0.0198`
- `final_signed_margin_median = -0.0055 +- 0.0192`
- `best failure_mode counts = {'weak_asymmetry': 3, 'mixed_or_partial': 2}`
- `final failure_mode counts = {'weak_asymmetry': 4, 'mixed_or_partial': 1}`

附带观测：

- exported epochs 明显不稳定：
  - `exported epochs = [30, 7, 8, 21, 30]`
  - `exported_eq_final_frac = 0.4`

判定：

- 是否通过：否
- 相对 control 的 paired delta：
  - `delta(best_primary_strict_f1) = -0.1333`
  - `delta(exported_primary_strict_f1) = -0.2333`
  - `delta(final_primary_strict_f1) = -0.2333`
  - `delta(best_final_gap) = +0.1000`
  - `delta(final_strict_f1@eps=0.1) = -0.1223`
- 是否推进下一步：否，淘汰

结论：

- 移除 pretrain 会让 `sim3` 的 best / exported / final 全面退化。
- 因此 pretrain 对当前 `sim3` 主线不是可有可无项，而是应保留的组成部分。

### [S3] sim3 init `patel_kappa -> random`

- 日期：2026-04-10
- 阶段：Phase 1
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：`P0A-sim3-control`
- 仅改动的参数：
  - `structure_init_mode: patel_kappa -> random`
- 固定不变的参数：
  - 保留 Patel direction
  - `causal_lag_main_weight = 0.0`
  - `optimizer_step_mode = batch_mean`
  - `directional_kappa_gate = False`
  - selector 固定为 `causal_lag_entropy_composite`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260410_095716`
  - `GraphExp/results/run_20260410_100606`
  - `GraphExp/results/run_20260410_101424`
  - `GraphExp/results/run_20260410_102218`
  - `GraphExp/results/run_20260410_103032`
- 汇总 artifact：
  - `GraphExp/results/unify_replay_sim3_20260410_095713_phase1_s3_random_init.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_095713_phase1_s3_random_init_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.7000 +- 0.0969`
- `exported_primary_strict_f1 = 0.6889 +- 0.1030`
- `final_primary_strict_f1 = 0.6889 +- 0.1030`
- `best_final_gap = 0.0111 +- 0.0222`
- `exported_best_gap = 0.0111 +- 0.0222`
- `best_strict_f1@eps=0.1 = 0.0211 +- 0.0421`
- `exported_strict_f1@eps=0.1 = 0.0421 +- 0.0516`
- `final_strict_f1@eps=0.1 = 0.0421 +- 0.0516`
- `best_signed_margin_median = 0.0229 +- 0.0086`
- `exported_signed_margin_median = 0.0197 +- 0.0052`
- `final_signed_margin_median = 0.0214 +- 0.0083`
- `best failure_mode counts = {'weak_asymmetry': 5}`
- `final failure_mode counts = {'weak_asymmetry': 5}`

附带观测：

- entropy selector 大多数时候又回到了最后一个 epoch：
  - `exported epochs = [12, 30, 30, 30, 30]`
  - `exported_eq_final_frac = 0.8`

判定：

- 是否通过：否
- 相对 control 的 paired delta：
  - `delta(best_primary_strict_f1) = +0.0111`
  - `delta(exported_primary_strict_f1) = +0.0000`
  - `delta(final_primary_strict_f1) = +0.0000`
  - `delta(best_final_gap) = +0.0111`
  - `delta(final_strict_f1@eps=0.1) = -0.1794`
- 是否推进下一步：否，淘汰

结论：

- `random` init 在主 strict F1 上基本做到与 control 持平。
- 但它会把高 margin 指标大幅拉低，说明当前 `sim3` 轨迹会退化为更弱的方向解。
- 因而 `random init` 不能作为 `sim3` 的低风险统一化通过项。

### [S4] sim3 kappa gate `False -> True`

- 日期：2026-04-10
- 阶段：Phase 1
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：`P0A-sim3-control`
- 仅改动的参数：
  - `directional_kappa_gate: False -> True`
- 固定不变的参数：
  - 保留 Patel direction
  - `causal_lag_main_weight = 0.0`
  - `optimizer_step_mode = batch_mean`
  - `structure_init_mode = patel_kappa`
  - selector 固定为 `causal_lag_entropy_composite`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260410_103924`
  - `GraphExp/results/run_20260410_104826`
  - `GraphExp/results/run_20260410_105716`
  - `GraphExp/results/run_20260410_110635`
  - `GraphExp/results/run_20260410_111555`
- 汇总 artifact：
  - `GraphExp/results/unify_replay_sim3_20260410_103921_phase1_s4_kappa_gate.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_103921_phase1_s4_kappa_gate_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.8222 +- 0.0816`
- `exported_primary_strict_f1 = 0.7667 +- 0.1237`
- `final_primary_strict_f1 = 0.8111 +- 0.0667`
- `best_final_gap = 0.0111 +- 0.0222`
- `exported_best_gap = 0.0556 +- 0.0609`
- `best_strict_f1@eps=0.1 = 0.6513 +- 0.0795`
- `exported_strict_f1@eps=0.1 = 0.4597 +- 0.1646`
- `final_strict_f1@eps=0.1 = 0.5980 +- 0.0786`
- `best_signed_margin_median = 0.1150 +- 0.0195`
- `exported_signed_margin_median = 0.0630 +- 0.0325`
- `final_signed_margin_median = 0.0854 +- 0.0253`
- `best failure_mode counts = {'mixed_or_partial': 5}`
- `final failure_mode counts = {'mixed_or_partial': 5}`

附带观测：

- entropy selector 已不再固定选最后一个 epoch，但比 `S1` 更接近晚期 checkpoint：
  - `exported epochs = [30, 9, 9, 30, 30]`
  - `exported_eq_final_frac = 0.6`

判定：

- 是否通过：是，当前唯一明确通过的单因素
- 相对 control 的 paired delta：
  - `delta(best_primary_strict_f1) = +0.1333`
  - `delta(exported_primary_strict_f1) = +0.0778`
  - `delta(final_primary_strict_f1) = +0.1222`
  - `delta(best_final_gap) = +0.0111`
  - `delta(final_strict_f1@eps=0.1) = +0.3765`
- 是否推进下一步：是，保留进入后续方向监督替代链

结论：

- `directional_kappa_gate` 是当前 `sim3` 低风险统一化中最强、也最完整的单因素改动。
- 它同时提高了 best / exported / final strict F1，并显著提高了高 margin 指标。
- 从 Phase 1 的正式标准看，`S4` 是唯一清晰通过项。

### [S5] sim3 通过项组合

- 状态：无需额外运行
- 说明：
  - 本轮 `S1-S4` 中，只有 `S4` 明确通过正式提升标准。
  - 因而“组合所有通过项”的 `S5` 与 `S4` 在配置上等价，不产生新的训练候选。
  - 后续 `Phase 2` 直接以 `S4` 作为新的 control（`D0`）。

### [D1] sim3 加入 causal-lag

- 日期：2026-04-10
- 阶段：Phase 2
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：`S4`
- 仅改动的参数：
  - `causal_lag_main_weight: 0.0 -> 0.25`
- 固定不变的参数：
  - `directional_kappa_gate = True`
  - 保留 Patel directional supervision
  - `optimizer_step_mode = batch_mean`
  - `structure_init_mode = patel_kappa`
  - `gradient_routing_mode = legacy`
  - selector 固定为 `causal_lag_entropy_composite`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260410_153131`
  - `GraphExp/results/run_20260410_154155`
  - `GraphExp/results/run_20260410_155148`
  - `GraphExp/results/run_20260410_160148`
  - `GraphExp/results/run_20260410_161242`
- 汇总 artifact：
  - `GraphExp/results/unify_replay_sim3_20260410_153127_phase2_d1_add_causallag.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_153127_phase2_d1_add_causallag_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.8222 +- 0.0889`
- `exported_primary_strict_f1 = 0.7667 +- 0.0648`
- `final_primary_strict_f1 = 0.8000 +- 0.0754`
- `best_final_gap = 0.0222 +- 0.0272`
- `exported_best_gap = 0.0556 +- 0.0497`
- `best_strict_f1@eps=0.1 = 0.6668 +- 0.0331`
- `exported_strict_f1@eps=0.1 = 0.4889 +- 0.0817`
- `final_strict_f1@eps=0.1 = 0.5789 +- 0.0527`
- `best_signed_margin_median = 0.1162 +- 0.0124`
- `exported_signed_margin_median = 0.0615 +- 0.0209`
- `final_signed_margin_median = 0.0823 +- 0.0155`
- `best failure_mode counts = {'mixed_or_partial': 5}`
- `final failure_mode counts = {'mixed_or_partial': 5}`

附带观测：

- 相对 `S4`，D1 的净变化非常小：
  - `delta(best_primary_strict_f1) = +0.0000`
  - `delta(exported_primary_strict_f1) = +0.0000`
  - `delta(final_primary_strict_f1) = -0.0111`
- entropy selector 仍然不是固定选最后一个 epoch：
  - `exported epochs = [7, 10, 30, 10, 30]`

判定：

- 是否通过：作为链路中间检查完成，但不构成对 `S4` 的明确提升
- 相对 control / `S4` 的 paired delta：
  - `vs control: delta(final_primary_strict_f1) = +0.1111`
  - `vs S4: delta(final_primary_strict_f1) = -0.0111`
- 是否推进下一步：是，进入 `D2`

结论：

- 在保留 Patel directional supervision 的前提下，引入 `causal_lag_main_weight=0.25` 对 `S4` 没有带来明确净收益。
- 因而 D1 只说明 causal-lag 可以共存；它本身并没有证明 causal-lag 是必要增益项。

### [D2] sim3 去掉 Patel directional supervision

- 日期：2026-04-10
- 阶段：Phase 2
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：`D1`
- 仅改动的参数：
  - 关闭 Patel directional supervision：
    - `disable_directional_loss = True`
    - `directional_loss_end_epoch = null`（仅为避免 CLI 冲突，不属于机制改动）
- 固定不变的参数：
  - `directional_kappa_gate = True`
  - `causal_lag_main_weight = 0.25`
  - `optimizer_step_mode = batch_mean`
  - `structure_init_mode = patel_kappa`
  - `gradient_routing_mode = legacy`
  - selector 固定为 `causal_lag_entropy_composite`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260410_190724`
  - `GraphExp/results/run_20260410_192015`
  - `GraphExp/results/run_20260410_193150`
  - `GraphExp/results/run_20260410_194233`
  - `GraphExp/results/run_20260410_195314`
- 汇总 artifact：
  - `GraphExp/results/unify_replay_sim3_20260410_190720_phase2_d2_disable_directional_loss.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_190720_phase2_d2_disable_directional_loss_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.5778 +- 0.1633`
- `exported_primary_strict_f1 = 0.5000 +- 0.1449`
- `final_primary_strict_f1 = 0.5000 +- 0.1449`
- `best_final_gap = 0.0778 +- 0.0831`
- `exported_best_gap = 0.0778 +- 0.0831`
- `best_strict_f1@eps=0.1 = 0.2344 +- 0.1603`
- `exported_strict_f1@eps=0.1 = 0.0211 +- 0.0421`
- `final_strict_f1@eps=0.1 = 0.0211 +- 0.0421`
- `best_signed_margin_median = 0.0165 +- 0.0226`
- `exported_signed_margin_median = 0.0002 +- 0.0103`
- `final_signed_margin_median = 0.0002 +- 0.0103`
- `best failure_mode counts = {'weak_asymmetry': 4, 'mixed_or_partial': 1}`
- `final failure_mode counts = {'weak_asymmetry': 5}`

附带观测：

- exported epoch 几乎全部又回到最后一个 epoch：
  - `exported epochs = [30, 30, 30, 30, 30]`
- best epoch 明显前移：
  - `best epochs = [4, 5, 1, 3, 28]`

判定：

- 是否通过：否，明确失败
- 相对 `D1` 的 paired delta：
  - `delta(best_primary_strict_f1) = -0.2444`
  - `delta(exported_primary_strict_f1) = -0.2667`
  - `delta(final_primary_strict_f1) = -0.3000`
  - `delta(final_strict_f1@eps=0.1) = -0.5579`
- 是否推进下一步：是，继续执行 `D3` 验证 routing 是否能补救

结论：

- 在当前 `sim3` 路径上，`causal-lag + kappa gate` 不能直接替代 Patel directional supervision。
- 一旦关闭 Patel directional supervision，主指标、margin、failure mode 都系统性恶化。

### [D3] sim3 启用 `warmup_then_orthogonal`

- 日期：2026-04-10
- 阶段：Phase 2
- 数据集：`sim3`
- Seeds：`11,22,33,44,55`
- 对照组：`D2`
- 仅改动的参数：
  - `gradient_routing_mode: legacy -> warmup_then_orthogonal`
- 固定不变的参数：
  - `directional_kappa_gate = True`
  - `causal_lag_main_weight = 0.25`
  - `disable_directional_loss = True`
  - `optimizer_step_mode = batch_mean`
  - `structure_init_mode = patel_kappa`
  - selector 固定为 `causal_lag_entropy_composite`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260410_200508`
  - `GraphExp/results/run_20260410_201549`
  - `GraphExp/results/run_20260410_202630`
  - `GraphExp/results/run_20260410_203708`
  - `GraphExp/results/run_20260410_204744`
- 汇总 artifact：
  - `GraphExp/results/unify_replay_sim3_20260410_200505_phase2_d3_warmup_then_orthogonal.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_200505_phase2_d3_warmup_then_orthogonal_aggregate.csv`

结果：

- `best_primary_strict_f1 = 0.6000 +- 0.1333`
- `exported_primary_strict_f1 = 0.5222 +- 0.1030`
- `final_primary_strict_f1 = 0.5333 +- 0.1197`
- `best_final_gap = 0.0667 +- 0.0416`
- `exported_best_gap = 0.0778 +- 0.0567`
- `best_strict_f1@eps=0.1 = 0.1864 +- 0.1854`
- `exported_strict_f1@eps=0.1 = 0.0612 +- 0.0501`
- `final_strict_f1@eps=0.1 = 0.0612 +- 0.0501`
- `best_signed_margin_median = 0.0171 +- 0.0229`
- `exported_signed_margin_median = -0.0030 +- 0.0074`
- `final_signed_margin_median = -0.0017 +- 0.0091`
- `best failure_mode counts = {'weak_asymmetry': 4, 'mixed_or_partial': 1}`
- `final failure_mode counts = {'weak_asymmetry': 5}`

附带观测：

- routing 改动并没有把 exported checkpoint 拉回高质量区域：
  - `exported epochs = [26, 30, 30, 30, 30]`

判定：

- 是否通过：否
- 相对 `D2` 的 paired delta：
  - `delta(best_primary_strict_f1) = +0.0222`
  - `delta(exported_primary_strict_f1) = +0.0222`
  - `delta(final_primary_strict_f1) = +0.0333`
  - 但相对 `S4` 仍有大幅退化：
    - `delta(final_primary_strict_f1 vs S4) = -0.2778`
    - `delta(final_strict_f1@eps=0.1 vs S4) = -0.5369`
- 是否推进下一步：否，不保留为正式候选

结论：

- `warmup_then_orthogonal` 对去-Patel 路径只有极弱补救，远不足以恢复到 `S4` 或 even control 水平。
- 因而在当前框架和 `sim3` 条件下，去掉 Patel directional supervision 的路径不成立。

### [D4] sim3 `support_prior_mode -> pearson_abs`（可选）

- 状态：未执行
- 说明：
  - 在 `D2/D3` 已经明确失败的前提下，继续做更激进的去先验化（`D4`）不再具有主方案决策价值。
  - 如后续需要探索“最大程度去 Patel 接触点”的研究性方向，可作为独立 side branch 重开。

### [G1/G2] sim4 gate

- 日期：2026-04-11
- 阶段：Phase 3
- 数据集：`sim4`
- Seeds：`11,22,33,44,55`
- 对照组：
  - `G0 = P0A-sim4-control`
- 候选：
  - `G1 = sim3 S4 Patel-family 统一主干候选迁移到 sim4`
  - `G2 = 未执行；Phase 2 已证明无-Patel-direction 路径在 sim3 上不成立`
- `G1` 的数据集自适应项：
  - `csv_path = ..\fMRI_dataset\sim4.csv`
  - `selector_audit_gt_path = ..\fMRI_dataset\h4.txt`
  - `top_k_edges = 61`
  - `selection_top_k = 61`
  - `epochs = 40`
- `G1` 的训练主干保持为 Phase 1 胜出 Patel-family 候选：
  - `optimizer_step_mode = batch_mean`
  - `structure_init_mode = patel_kappa`
  - `directional_kappa_gate = True`
  - `causal_lag_main_weight = 0.0`
  - `gradient_routing_mode = legacy`
  - `directional_loss_end_epoch = 15`
  - selector 固定为 `causal_lag_entropy_composite`
- 训练相关 artifact：
  - `GraphExp/run_replay_saved_config.py`
  - `GraphExp/results/run_20260411_103411`
  - `GraphExp/results/run_20260411_110117`
  - `GraphExp/results/run_20260411_112600`
  - `GraphExp/results/run_20260411_114951`
  - `GraphExp/results/run_20260411_123548`
- 汇总 artifact：
  - `GraphExp/results/unify_replay_sim4_20260411_130205_phase3_g1_sim4_patelfamily_full.csv`
  - `GraphExp/results/unify_replay_sim4_20260411_130205_phase3_g1_sim4_patelfamily_full_aggregate.csv`
- selector replay artifact：
  - `GraphExp/replay_selector_modes.py`
  - `GraphExp/results/unify_phase0_selector_sim4_20260411_130306_phase3_g1_selector.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260411_130306_phase3_g1_selector_aggregate.csv`

结果：

`G1` 训练导出结果（training-side selector=`causal_lag_entropy_composite`）：

- `best_primary_strict_f1 = 0.8033 +- 0.0104`
- `exported_primary_strict_f1 = 0.7443 +- 0.0222`
- `final_primary_strict_f1 = 0.7443 +- 0.0222`
- `best_final_gap = 0.0590 +- 0.0222`
- `exported_best_gap = 0.0590 +- 0.0222`
- `best_strict_f1@eps=0.1 = 0.4782 +- 0.0377`
- `exported_strict_f1@eps=0.1 = 0.4158 +- 0.0528`
- `final_strict_f1@eps=0.1 = 0.4158 +- 0.0528`
- `best_signed_margin_median = 0.0621 +- 0.0044`
- `exported_signed_margin_median = 0.0143 +- 0.0003`
- `final_signed_margin_median = 0.0143 +- 0.0003`
- `best failure_mode counts = {'symmetric_collapse': 5}`
- `final failure_mode counts = {'symmetric_collapse': 5}`

相对 `G0` 的 paired delta：

- `delta(best_primary_strict_f1) = -0.0590`
- `delta(exported_primary_strict_f1) = -0.0656`
- `delta(final_primary_strict_f1) = -0.0852`
- `delta(best_final_gap) = +0.0262`
- `delta(final_strict_f1@eps=0.1) = +0.3786`

selector replay（同一批 `G1` 轨迹上的离线复核）：

- `causal_lag_entropy_composite`：
  - `chosen_epoch_mean = 40.0`
  - `chosen_primary_strict_f1_mean = 0.7443`
  - 与训练导出完全一致：`matches_trained_exported_primary_strict_f1 = 5/5`
- `legacy`：
  - `chosen_epoch_mean = 18.2`
  - `chosen_primary_strict_f1_mean = 0.7443`
  - `chosen_strict_f1@eps=0.1_mean = 0.4494`
  - 对主 strict F1 没有任何补救：`chosen_vs_trained_exported_delta_primary_strict_f1 = 0.0000`
- `causal_lag_composite`：
  - `chosen_epoch_mean = 6.0`
  - `chosen_primary_strict_f1_mean = 0.5836`
  - 明显更差

判定：

- 是否通过：否，`G1` 未通过 sim4 gate
- `G2` 是否执行：否；Phase 2 已经把无-Patel-direction 路径淘汰，继续 gate 不再改变主方案决策
- 是否推进下一步：否，不进入 `Phase 4`

结论：

- `G1` 在 sim4 上是训练性失败，而不是 selector miss。
- 证据是：即使在同一批 `G1` 轨迹上做 `legacy` offline replay，主 strict F1 也无法超过训练导出的 `0.7443`。
- 因而当前 Patel-family 统一主干候选虽然在 sim3 上通过，但不能跨过 sim4 gate。
- 到此为止，本轮统一化主线没有候选能同时通过 `sim3` 与 `sim4`。

### [U1] 全数据集统一候选确认

- 状态：未开始
- 说明：
  - `Phase 3` 未产生 gate-passed 候选，因此当前不进入 `Phase 4`。

### [F1] `detach_epoch` 公式验证（可选）

- 状态：未开始

## 10. 结论日志

每完成一个阶段，必须追加一条结论日志。格式如下：

```md
### [日期] 结论

- 阶段：
- 结论：
- 证据：
- 被淘汰的候选：
- 保留的候选：
- 下一步：
```

### [2026-04-09] 初始方法学结论

- 阶段：方案冻结
- 结论：
  - `sim3` 必须是第一关键验证点，但不是唯一关键验证点。
  - `sim4` 必须作为第二 gate。
  - 训练因素、方向监督因素、selector 因素必须拆开验证。
  - 不再接受“多个大改动打包一次性 run 后直接得出机制结论”的做法。
- 证据：
  - `GraphExp/results/best_run_summary_20260406_211725.csv`
  - `GraphExp/CROSS_PRED_V1_TRACKER.md`
  - `GraphExp/main_structure_learning.py`
- 被淘汰的候选：
  - “两次 run 直接决定统一方案”的快速方案
  - “用绝对阈值 0.90 作为统一标准”的判定方式
- 保留的候选：
  - 分阶段、可归因、paired comparison 的统一实验设计
- 下一步：
  - 先执行 Phase 0A 与 Phase 0B

### [2026-04-10] Phase 0A 结论

- 阶段：Phase 0A 控制组重跑
- 结论：
  - `sim3` 当前 incumbent 在 5 seeds 上不稳健，不能再用单 seed 最佳值代表它的真实基线。
  - `sim4` 当前 incumbent 在主 strict F1 上相对稳定，但明显存在 `best -> exported/final` 的退化，而且仍然是低 margin / `symmetric_collapse` 解。
  - 因此，后续统一实验必须保留：
    - `sim3` 作为第一机制 gate
    - `sim4` 作为第二稳定性 gate
  - 在进入任何新的训练 ablation 之前，必须先做 `Phase 0B` 的 selector-only rescoring，把“训练问题”和“选模问题”拆开。
- 证据：
  - `GraphExp/results/unify_phase0_control_sim3_20260410_003500_phase0_sim3_control_full_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim3_20260410_003900_rich_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260409_225712_phase0_sim4_control_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich_aggregate.csv`
- 被淘汰的候选：
  - “把当前 `sim3` 单 seed 最佳 run 直接当作稳健 incumbent”的假设
  - “不做 selector-only 复核，直接进入训练 ablation”的顺序
- 保留的候选：
  - `Phase 0B` selector-only rescoring
  - 之后再进入 `Phase 1` 的 `sim3` 单因素训练因素隔离
- 下一步：
  - 先执行 `Phase 0B-sim3-selector`
  - 再执行 `Phase 0B-sim4-selector`

### [2026-04-10] Phase 0B 结论

- 阶段：Phase 0B selector-only rescoring
- 结论：
  - `sim3` 与 `sim4` 的 control 轨迹不存在同一个自然胜出的 selector。
  - `sim3` 明显需要 `causal_lag_entropy_composite`；改成 `legacy` 或 `causal_lag_composite` 都会拉向更早 epoch，并降低主指标。
  - `sim4` 则相反：`legacy` 能在不重训的情况下把 exported 从 `0.8098` 提到 `0.8361`，说明它对 selector 明显敏感。
  - 因此，后续训练 ablation 必须把 `sim3` 的 selector 固定住，且在 `sim4` gate 中显式区分“training 问题”和“selector 问题”。
  - 但当前证据只支持“晚期偏好”和“现有打分公式存在错排”，还不支持把 selector 简化成“直接选最后一个可行 epoch”。
- 证据：
  - `GraphExp/replay_selector_modes.py`
  - `GraphExp/results/unify_phase0_selector_sim3_20260410_082446_phase0b.csv`
  - `GraphExp/results/unify_phase0_selector_sim3_20260410_082446_phase0b_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260410_082446_phase0b.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260410_082446_phase0b_aggregate.csv`
- 被淘汰的候选：
  - “先统一 selector，再做 training ablation”的假设
  - `sim3` 上的 `legacy` / `causal_lag_composite` selector 作为正式 baseline 的方案
  - `sim4` 上继续只看当前 `causal_lag_composite` exported 结果就判断 training ceiling 的做法
- 保留的候选：
  - `sim3`：`causal_lag_entropy_composite`
  - `sim4`：`legacy` 作为 selector-only 参考上界
- 下一步：
  - 执行 `Phase 1` 的 `S1-S4`
  - 在 `sim3` 上固定 selector 为 `causal_lag_entropy_composite`
  - 同时记录 `sim3 entropy` 的 argmax 是否仍然钉在最后一个 eligible epoch，作为训练改动是否提升 selector 可分辨性的附带观测项

### [2026-04-10] Phase 1 结论

- 阶段：Phase 1 sim3 低风险训练因素隔离
- 结论：
  - 经 correction 核实，`sim3` incumbent 实际上是“有 pretrain”，因此 `S2` 应解释为 `pretrain on -> off`。
  - `S2` 明确失败；移除 pretrain 会让 `sim3` 的 best / exported / final 全面退化。
  - `S3` 虽然在主 strict F1 上基本持平，但会显著损伤高 margin 指标，因此也不能通过。
  - `S1` 能显著提高 best / final ceiling 和高 margin 指标，但 retention gap 明显恶化，未通过正式提升标准。
  - `S4` 是当前唯一清晰通过的单因素：它同时改善 best / exported / final strict F1，并显著提高高 margin 指标，且 gap 仍在可接受范围内。
  - 额外的重要现象是：`S1/S4` 的训练轨迹让 entropy selector 不再总是选最后一个 epoch，说明训练改动本身可以提高 selector 的可分辨性。
- 证据：
  - `GraphExp/results/unify_replay_sim3_20260410_085118_phase1_s1_subject_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_093125_phase1_s2_no_pretrain_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_095713_phase1_s3_random_init_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_103921_phase1_s4_kappa_gate_aggregate.csv`
- 被淘汰的候选：
  - `S2` (`skip_pretrain=True`)
  - `S3` (`structure_init_mode=random`)
  - `S1` 作为正式通过项的直接提升
- 保留的候选：
  - `S4` (`directional_kappa_gate=True`)
  - `S1` 作为后续 side-branch 观察对象
- 下一步：
  - 以 `S4` 作为 `D0` 进入 `Phase 2`
  - 先执行 `D1`：在 `S4` 基础上加入 `causal_lag_main_weight=0.25`

### [2026-04-10] Phase 2 结论

- 阶段：Phase 2 sim3 方向监督替代链
- 结论：
  - `D1` 表明：在保留 Patel directional supervision 的情况下，加入 `causal-lag` 基本是中性的，并没有对 `S4` 带来明确净收益。
  - `D2` 表明：一旦去掉 Patel directional supervision，`sim3` 会系统性退化；因此 `causal-lag + kappa gate` 不能直接替代 Patel direction。
  - `D3` 表明：`warmup_then_orthogonal` routing 只能带来极弱补救，远不足以把无-Patel-direction 路径拉回可接受范围。
  - 因而当前统一主方案必须保留 Patel directional supervision；`sim3` 还不支持去掉这一项。
- 证据：
  - `GraphExp/results/unify_replay_sim3_20260410_153127_phase2_d1_add_causallag_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_190720_phase2_d2_disable_directional_loss_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_200505_phase2_d3_warmup_then_orthogonal_aggregate.csv`
- 被淘汰的候选：
  - `D2`：`causal-lag + kappa gate + no Patel direction`
  - `D3`：在 `D2` 基础上再加 `warmup_then_orthogonal`
  - `D4` 作为主方案决策候选
- 保留的候选：
  - `S4` 作为当前最强的 Patel-family 统一主干候选
  - `D1` 仅作为“causal-lag 可共存但非必要”的旁证
- 下一步：
  - 进入 `Phase 3 sim4 gate`
  - 正式 gate 候选以 `G1 = S4` 为主

### [2026-04-11] Phase 3 结论

- 阶段：Phase 3 sim4 gate
- 结论：
  - `G1`（sim3 `S4` Patel-family 统一主干候选迁移到 sim4）未通过 sim4 gate。
  - 它相对 `sim4` incumbent control 在主 strict F1 上全面退化：
    - `delta(best_primary_strict_f1) = -0.0590`
    - `delta(exported_primary_strict_f1) = -0.0656`
    - `delta(final_primary_strict_f1) = -0.0852`
    - `delta(best_final_gap) = +0.0262`
  - 虽然 `strict_f1@eps=0.1` 大幅上升，但 failure mode 仍是 `symmetric_collapse`，且主 strict F1 已明显越过非劣效边界。
  - 同一批轨迹上的 `legacy` selector replay 也无法提升主 strict F1，说明这不是单纯 selector miss，而是训练主干本身不适配 sim4。
  - `G2` 未执行，因为 Phase 2 已证明无-Patel-direction 路径在 sim3 上不成立，继续 gate 不会改变统一主方案判断。
- 证据：
  - `GraphExp/results/unify_replay_sim4_20260411_130205_phase3_g1_sim4_patelfamily_full_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260411_130306_phase3_g1_selector_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich_aggregate.csv`
- 被淘汰的候选：
  - `G1`：Patel-family 统一主干候选
  - `G2`：无-Patel-direction 路径（已在 Phase 2 实质淘汰）
- 保留的候选：
  - 当前没有通过 `sim3 + sim4` 双 gate 的统一候选
  - `sim4 incumbent + legacy selector replay` 仍是 sim4 的最强参考上界
- 下一步：
  - 暂不进入 `Phase 4`
  - 如果继续统一化探索，需要重新设计候选族，而不是沿当前 `S4/D1` 主线外推

### [2026-04-11] Exploratory H1 结论

- 阶段：`sim4 incumbent -> sim3` 反向迁移验证
- 更正：
  - 经 replay 语义核查，`sim4 incumbent`（`run_20260404_111017`）实际包含 Patel directional prior。
  - 因而先前草拟的 `H2 = 在 H1 上恢复 Patel direction` 没有信息增量，已取消；本分支只保留 `H1` 与 `H1-selector`。
- `H1` 定义：
  - 以 `run_20260404_111017` 为 base run，保留其训练主干与 selector 设置，只把数据集切到 `sim3`。
  - 关键有效配置为：`optimizer_step_mode=subject`、`structure_init_mode=random`、`directional_kappa_gate=True`、`causal_lag_main_weight=0.25`、`gradient_routing_mode=warmup_then_orthogonal`、`detach_direction_from_main_after_epoch=23`、`skip_pretrain=False`、`fixed_support_mask_mode=maxgap_kappa`、`selection_score_mode=causal_lag_composite`。
  - `sim3` 数据集覆写：`csv_path=..\fMRI_dataset\sim3.csv`、`selector_audit_gt_path=..\fMRI_dataset\h3.txt`、`top_k_edges=18`、`selection_top_k=18`、`epochs=30`。
- 结论：
  - `H1` 已经实测完成，不存在“还没试过 sim4 配置给 sim3”的空白。
  - 训练侧迁移结果明显强于 `sim3 control`，也略强于当前 `S4`：
    - `H1`: `best/exported/final = 0.8778 / 0.7889 / 0.8222`
    - `vs sim3 control`: `delta(best/exported/final) = +0.1889 / +0.1000 / +0.1333`
    - `vs S4`: `delta(best/exported/final) = +0.0556 / +0.0222 / +0.0111`
  - 但 `H1` 作为“完整统一包”仍不能直接提升，因为 retention gap 明显偏大：
    - `exported_vs_best_gap = -0.0889`
    - `final_vs_best_gap = -0.0556`
    - 相比 `sim3 control` 的 `0.0000 / 0.0000` 和 `S4` 的 `-0.0556 / -0.0111`，都更差。
  - `H1-selector` 说明主要问题不是训练 ceiling 不够，而是 selector 错排：
    - `causal_lag_composite` replay 与训练导出完全一致，均值 `0.7889`，说明当前导出就是被 `sim4` 默认 selector 锁住了。
    - `legacy` replay 可升到 `0.8222`。
    - `causal_lag_entropy_composite` replay 进一步升到 `0.8556`，较训练导出再增 `+0.0667`，并在 `5` 个 seed 里有 `4` 个精确追回 GT-best。
  - `H1` 轨迹上的 entropy selector 会选 `epoch 7-9`（均值 `7.8`），不再像 `sim3 control` 那样钉在最后一个 epoch。这说明：
    - `sim4` 风格训练主干在 `sim3` 上形成了不同的质量轨迹；
    - 当前问题不是“sim3 必须选晚”，而是“sim4 风格轨迹的好 checkpoint 更早，但 `causal_lag_composite` 没排对顺序”。
  - 与 `Phase 3` 的反方向结果合起来看，迁移是明显非对称的：
    - `sim3 S4 -> sim4` 失败；
    - `sim4 incumbent -> sim3` 则在训练 ceiling 上成立，只是 selector / retention 还不够统一。
  - 因而如果继续统一化探索，优先级应转向：
    - 以 `sim4 incumbent` 的训练主干为起点；
    - 单独攻克跨数据集 selector / checkpoint ranking，而不是继续沿 `S4/D1` 主线外推。
- 证据：
  - `GraphExp/results/unify_replay_sim3_20260411_153925_h1_sim4_full_to_sim3.csv`
  - `GraphExp/results/unify_replay_sim3_20260411_153925_h1_sim4_full_to_sim3_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim3_20260411_160252_h1_selector.csv`
  - `GraphExp/results/unify_phase0_selector_sim3_20260411_160252_h1_selector_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim3_20260410_003500_phase0_sim3_control_full_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260410_103921_phase1_s4_kappa_gate_aggregate.csv`
  - `GraphExp/results/unify_replay_sim4_20260411_130205_phase3_g1_sim4_patelfamily_full_aggregate.csv`
- 被淘汰的候选：
  - `H2`：在 `H1` 上“恢复 Patel direction”的分支
  - “sim4 配置迁移到 sim3 会直接失败”的假设
  - “这次迁移的主要问题是训练本身完全跑坏”的解释
- 保留的候选：
  - `H1` 作为新的统一化 side-branch 起点
  - `H1 + entropy replay` 作为 `sim3` 侧的 selector 上界
- 下一步：
  - 如果继续实验，应围绕 `H1` 做 selector / ranking 机制拆解
  - 优先验证能否在不破坏 `sim4` 的前提下，让 `sim3` 也抓到 H1 轨迹里的早期优质 checkpoint

### [2026-04-12] Late-Window Selector Validation

- 阶段：离线验证 “训练晚期稳定化是否足以让 selector 简化”
- 背景问题：
  - Claude 提出的核心判断是：如果训练轨迹满足 `final ≈ best ± ε`，那么复杂 selector 的价值会大幅下降，`final` 或 `last-N` 这类简单规则就足够。
  - 本轮验证要回答的不是“复杂 selector 是否永远必要”，而是“当前已观察到的失败，是否已经可以主要归结为 late drift”。
- 协议：
  - 不重训，只对现有 `quality_history.csv` 做离线 replay。
  - 新增脚本：`GraphExp/replay_window_selectors.py`
  - 比较对象：
    - `final_epoch`
    - `last-k oracle`：仅用于诊断，表示“如果只允许在最后 `k` 个 epoch 里挑，GT 上限是多少”
    - `legacy / causal_lag_composite / causal_lag_entropy_composite` 在 `last-k` 窗口内的实际选点
  - 代表性轨迹族：
    - `sim3 control`
    - `sim3 S4`
    - `sim4 control`
    - `H1 = sim4 incumbent -> sim3`
- 结论：
  - Claude 的方向只部分成立：
    - 在 `sim3 control` 与 `sim3 S4` 这两类轨迹上，late-window 近乎充分，selector 可以大幅简化。
    - 但在 `sim4 control` 与 `H1 -> sim3` 上，仅靠 late-window 不能统一解决问题。
  - `sim3 control`：
    - `final_epoch` 与所有测试的 `last-k oracle`（`k=3/5/8/10`）都精确命中 global GT best。
    - 说明当前 `sim3 control` 轨迹对 selector 的要求很低；其主问题不是 late-window miss。
  - `sim3 S4`：
    - `final_epoch` 相对 global GT best 的均值差仅 `-0.0111`。
    - `last-k oracle`（`k=3/5/8/10`）与 `final` 基本等价。
    - 这表明 training stabilization 确实能让 selector 问题显著降级。
  - `sim4 control`：
    - `final_epoch` 相对 global GT best 的均值差为 `-0.0328`。
    - `last10_oracle` 改善到 `-0.0131`，`last20_oracle` / `last24_oracle` 改善到 `-0.0066`。
    - 因而 `sim4` 的一部分问题确实是“好 checkpoint 更偏晚，简单 final 会错过”。
    - 但更关键的是：即使把窗口放宽到 `last20` / `last24`，当前 selector 仍然排不好窗内候选：
      - `legacy_last20 = -0.0328`
      - `causal_lag_composite_last20 = -0.0459`
      - `causal_lag_entropy_composite_last20 = -0.0459`
    - 这说明 `sim4` 不只是 late drift；它同时存在明显的 in-window ranking miss。
  - `H1 -> sim3`：
    - `final_epoch` 相对 global GT best 的均值差为 `-0.0556`。
    - `last10_oracle` 仍是 `-0.0556`，`last20_oracle` 也只有 `-0.0444`。
    - 只有 `last24_oracle` 才能精确追回 global GT best；对 `30` epoch 训练来说，这已经等于“最后 `80%` 的轨迹”，不再是有意义的晚窗简化。
    - 具体到 seed，global GT best epoch 为 `7, 7, 9, 8, 11`；要覆盖全部 best，需要最小 `last-N = 24, 24, 22, 23, 20`。
    - 这证明 `H1` 属于明确的 early/mid-peak 轨迹族，不能用一个合理的小型 late-window 规则解决。
    - 即便把窗口放大到 `last24`，selector 本身仍有差异：
      - `causal_lag_entropy_composite_last24 = -0.0222`（`4/5` exact hits）
      - `legacy_last24 = -0.0556`
      - `causal_lag_composite_last24 = -0.0889`
    - 因而 `H1` 上的问题是“晚窗不够 + 排序也不能乱”，而不是单纯 final drift。
  - 结合以上四组轨迹，可把当前现象拆成三类：
    - `late-stable`：`sim3 control`, `sim3 S4`
    - `late-window helps but ranking still matters`：`sim4 control`
    - `peak too early for reasonable late-window`：`H1 -> sim3`
  - 因此，当前不能把“解决 late drift”直接等价为“selector 问题自动消失”。
  - 更准确的结论是：
    - training stabilization 是高杠杆主方向；
    - 但 selector 仍需保留，至少要能区分“晚窗内排序失败”和“最佳点根本不在晚窗”。
- 证据：
  - `GraphExp/replay_window_selectors.py`
  - `GraphExp/results/window_replay_sim3_20260412_124614_phase_window_sim3_control_aggregate.csv`
  - `GraphExp/results/window_replay_sim3_20260412_124615_phase_window_sim3_s4_aggregate.csv`
  - `GraphExp/results/window_replay_sim4_20260412_124615_phase_window_sim4_control_aggregate.csv`
  - `GraphExp/results/window_replay_sim4_20260412_124815_phase_window_sim4_control_wide_aggregate.csv`
  - `GraphExp/results/window_replay_sim3_20260412_124615_phase_window_h1_sim3_aggregate.csv`
  - `GraphExp/results/window_replay_sim3_20260412_124815_phase_window_h1_sim3_wide_aggregate.csv`
  - `GraphExp/CROSS_PRED_V1_TRACKER.md:5907`
  - `GraphExp/CROSS_PRED_V1_TRACKER.md:5911`
- 限制：
  - 历史 `fMRI` retention-fix 分支在本地只保留了 aggregate / shell log，没有完整 5-seed `run_dir` 清单，因此本轮未对其逐 epoch 轨迹重放 late-window。
  - 但历史记录已经明确指出：该分支在 `fMRI` 上是“exported checkpoint 选错、final 反而更好”的 selector miss，而不是典型的 final drift。
- 被淘汰的候选：
  - “只要把 selector 改成 final / last10，就能统一解决当前问题”的假设
  - “当前 selector 问题本质上只有 late drift 一种”的解释
- 保留的候选：
  - 把 retention stabilization 作为更高杠杆的训练主线
  - 同时保留简单 late-window baseline，作为后续 selector 方案的必过对照
  - 对 `H1` 继续做 ranking 机制拆解，而不是仅扩大 late-window
- 下一步：
  - 训练侧：优先探索能让 `H1` 轨迹更晚期稳定的改动，而不是默认接受其 early-peak 形状
  - selector 侧：后续所有新方案都必须至少对比 `final_epoch`、`last10` / `last20` 控制，以及当前数据集最强 replay 上界

### [2026-04-12] H1 Retention / Stability Experiments

- 阶段：`H1 = sim4 incumbent -> sim3` 的 retention/stability 训练修复
- 要回答的问题：
  - `H1` 的 early/mid peak 是否只是晚期 direction drift 造成的？
  - 如果做最小化 retention fix，能否把 `H1` 的高 ceiling 保留到 exported / final，而不是继续依赖更复杂 selector？
  - 漂移的主因更像：
    - 持续的 direction branch 参数更新
    - 持续的 Patel directional supervision
    - 还是二者都要同时关
- 设计：
  - 基线：`H1` 原始配置
    - `best/exported/final = 0.8778 / 0.7889 / 0.8222`
    - `exported_vs_best_gap = -0.0889`
    - `final_vs_best_gap = -0.0556`
  - `R1`: `freeze_direction_after_epoch = 10`
  - `R2`: `directional_loss_end_epoch = 10`
  - `R3`: `freeze_direction_after_epoch = 10` + `directional_loss_end_epoch = 10`
  - 其余配置保持 `H1` 完全不变；seeds=`11,22,33,44,55`
- 结果汇总：
  - `H1`：
    - `best/exported/final = 0.8778 / 0.7889 / 0.8222`
    - `best_strict_f1@eps=0.1 final = 0.1011`
  - `R1 = freeze@10`：
    - `best/exported/final = 0.8667 / 0.8556 / 0.8556`
    - `exported_vs_best_gap = -0.0111`
    - `final_vs_best_gap = -0.0111`
    - `final_strict_f1@eps=0.1 = 0.2652`
  - `R2 = dir_end@10`：
    - `best/exported/final = 0.9000 / 0.8444 / 0.8556`
    - `exported_vs_best_gap = -0.0556`
    - `final_vs_best_gap = -0.0444`
    - `final_strict_f1@eps=0.1 = 0.2842`
  - `R3 = freeze@10 + dir_end@10`：
    - aggregate 与 `R1` 基本完全一致：
      - `0.8667 / 0.8556 / 0.8556`
      - gaps 同为 `-0.0111 / -0.0111`
- 关键 paired 比较：
  - `R1 vs H1`：
    - `delta(best) = -0.0111`
    - `delta(exported) = +0.0667`
    - `delta(final) = +0.0333`
  - `R2 vs H1`：
    - `delta(best) = +0.0222`
    - `delta(exported) = +0.0556`
    - `delta(final) = +0.0333`
  - `R1 vs S4`：
    - `delta(best/exported/final) = +0.0444 / +0.0889 / +0.0444`
  - `R2 vs S4`：
    - `delta(best/exported/final) = +0.0778 / +0.0778 / +0.0444`
- seed 级现象：
  - 原始 `H1`：
    - best epoch 分布为 `10, 12, 14, 10, 13`
    - exported/final 在 `seed 11,44` 上明显掉队
  - `R1`：
    - best epoch 收缩到 `10, 10, 9, 11, 10`
    - `seed 11,44` 的 exported/final 被直接抬到 `0.9444`
    - `seed 33` 与 `55` 仍然较弱，但不再出现大 gap
  - `R2`：
    - best epoch 仍集中在 `10-13`，但 gap 没有像 `R1` 那样被压平
    - 说明“只关晚期 Patel 监督”能改善，但不够彻底
- 结论：
  - `H1` 不是“天生只能 early-peak、无法保留到 final”的 backbone。
  - 仅加一个简单的 retention fix，`R1 = freeze_direction_after_epoch=10`，就能把 `H1` 的强 early solution 基本保留到 exported / final。
  - 这意味着：
    - `H1` 的核心问题更像 **late direction drift**
    - 而不是“training backbone 本身错误”
  - `R2` 也能改善，但明显弱于 `R1`：
    - 持续的 Patel directional supervision 是 drift 的一部分来源
    - 但不是主要来源；更大的问题是 direction branch 参数本身继续被更新
  - `R3` 与 `R1` 几乎完全一致，说明：
    - 一旦 direction branch 在 epoch 10 后被冻结，继续/停止 Patel supervision 已经几乎不再改变结果
    - 因而对 `H1` 这条线，真正关键的 retention 修复是 `freeze_direction_after_epoch=10`
  - 重要附加发现：
    - `R1` 的 exported / final 均值 `0.8556`，与原始 `H1` 上 `causal_lag_entropy_composite` 的 selector replay 上界完全对齐
    - 这说明 `R1` 实际上把“原本需要复杂 selector 才能追回的好 checkpoint”烘焙进了训练 final / export
    - 换句话说，`R1` 不是 selector 修补，而是真正的 training-side retention repair
  - 但仍有保留意见：
    - `R1` / `R2` / `R3` 的 final failure mode 仍是 `weak_asymmetry:5`
    - `R1` 的 `final_strict_f1@eps=0.1 = 0.2652`，仍显著低于 `S4` 的 `0.5980`
    - 因而 `H1` retention repair 虽然提升了 primary strict F1 与 gap，但还没有把方向 margin 质量提升到 `S4` 水平
- 当前判定：
  - 可以把 `H1 + freeze_direction_after_epoch=10` 视为新的 **高优先级统一训练主干候选**
  - 但还不能直接宣布它是最终统一配置，因为：
    - 还未重新 gate 到 `sim4`
    - 高 margin 质量仍弱于 `S4`
- 证据：
  - `GraphExp/results/unify_replay_sim3_20260412_133714_h1_retention_r1_freeze10.csv`
  - `GraphExp/results/unify_replay_sim3_20260412_133714_h1_retention_r1_freeze10_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260412_140031_h1_retention_r2_dirend10.csv`
  - `GraphExp/results/unify_replay_sim3_20260412_140031_h1_retention_r2_dirend10_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260412_142544_h1_retention_r3_freeze10_dirend10.csv`
  - `GraphExp/results/unify_replay_sim3_20260412_142544_h1_retention_r3_freeze10_dirend10_aggregate.csv`
  - `GraphExp/results/unify_replay_sim3_20260411_153925_h1_sim4_full_to_sim3_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim3_20260411_160252_h1_selector_aggregate.csv`
- 被淘汰的候选：
  - “`H1` 的 early peak 只能靠更复杂 selector 处理”的解释
  - “只停 Patel directional supervision 就足够修复 H1 retention”的解释
- 保留的候选：
  - `H1 + freeze_direction_after_epoch=10`
  - `R2` 仅作为次优旁证，不作为首选 retention fix
- 下一步：
  - 把 `H1 + freeze@10` 迁回 `sim4` 做 gate，验证它是否在原生 `sim4` 数据集上也保持非劣效
  - 若 `sim4` gate 通过，再决定是否把它提升为新的统一主线 backbone

### [H1-R1->sim4 gate] `H1 + freeze_direction_after_epoch=10` 迁回 `sim4`

- 配置：
  - 阶段：`Phase 3` 补充 gate
  - 数据集：`sim4`
  - seeds：`11,22,33,44,55`
  - base run dir：`GraphExp/results/run_20260404_111017`
  - 唯一改动：`freeze_direction_after_epoch=10`
  - 其余配置保持 `H1 / sim4 control` 主干不变
- 训练结果：
  - `sim4 control`：
    - `best/exported/final = 0.8623 / 0.8098 / 0.8295`
    - `exported_vs_best_gap = -0.0525`
    - `final_vs_best_gap = -0.0328`
    - `final_strict_f1@eps=0.1 = 0.0372`
  - `H1 + freeze@10`：
    - `best/exported/final = 0.8131 / 0.8000 / 0.8000`
    - `exported_vs_best_gap = -0.0131`
    - `final_vs_best_gap = -0.0131`
    - `final_strict_f1@eps=0.1 = 0.0194`
  - paired delta（`freeze@10 - control`）：
    - `delta(best/exported/final) = -0.0492 / -0.0098 / -0.0295`
    - `delta(exported_vs_best_gap / final_vs_best_gap) = +0.0393 / +0.0197`
  - failure mode：
    - control 与 `freeze@10` 都是 `symmetric_collapse:5`
- seed 级现象：
  - `best_primary_strict_f1` 在 5 个 seeds 上全部低于 control，没有任何一个 seed 保持原 ceiling。
  - `freeze@10` 的确压平了 gap：
    - `seed 11/22/44/55` 的 exported 与 final 都等于各自 best
    - 只有 `seed 33` 仍保留 `-0.0656` gap
  - 但这个“更稳”是以更低的上界换来的：
    - `seed 11/22/33/44/55` 的 best delta 分别是 `-0.0820 / -0.0656 / -0.0328 / -0.0492 / -0.0164`
  - 也就是说，`freeze@10` 在 `sim4` 上不是“保留原有好解”，而是“把更低的轨迹更早锁住”。
- selector replay：
  - 新轨迹上的 selector 结果：
    - `causal_lag_composite = 0.8000`
    - `causal_lag_entropy_composite = 0.7836`
    - `legacy = 0.7803`
  - 对照旧 `sim4 control` selector baseline：
    - `legacy(control) = 0.8361`
    - `causal_lag_composite(control) = 0.8098`
    - `causal_lag_entropy_composite(control) = 0.8000`
  - 关键判断：
    - 新轨迹上最好的 selector 仍只是 `causal_lag_composite = 0.8000`，与训练导出的 `exported/final` 完全一致
    - `legacy` 不仅没有像 control 那样把结果抬高，反而降到 `0.7803`
    - selector 选中的 epoch 也整体前移：
      - `legacy: 33.8 -> 8.4`
      - `causal_lag_composite: 21.0 -> 13.8`
  - 因而这不是“checkpoint 没挑对”，而是训练轨迹本身已经失去 control 上那部分高质量后期解。
- 结论：
  - `H1 + freeze_direction_after_epoch=10` 未通过 `sim4 gate`。
  - 它在 `sim3` 上是有效的 retention repair，但不能作为跨 `sim3/sim4` 的统一训练主干。
  - 这次失败的性质是：
    - 稳定性改善为真
    - 但绝对性能和高 margin 质量都下降
    - selector replay 无法追回损失
  - 因而此前“`H1 + freeze@10` 是高优先级统一训练主干候选”的临时判定需要撤销；它现在只保留为：
    - `sim3` 上有效的训练侧 retention fix
    - 不是新的统一 backbone
- 证据：
  - `GraphExp/results/unify_replay_sim4_20260412_160323_h1_retention_gate_sim4_freeze10_fullrerun.csv`
  - `GraphExp/results/unify_replay_sim4_20260412_160323_h1_retention_gate_sim4_freeze10_fullrerun_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260412_174117_h1_retention_gate_sim4_freeze10_fullrerun_selector.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260412_174117_h1_retention_gate_sim4_freeze10_fullrerun_selector_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260410_082446_phase0b_aggregate.csv`

### [Exploratory] `sim4` 基线主干迁移到 `sim2 / fMRI`

- 目的：
  - 回答“如果四个数据集都强行使用 `sim4` 基线主干，其余数据集会得到什么 F1”。
  - 这一步是补充性 transfer 检查，不改变此前“统一候选必须先过 `sim3 + sim4 gate`”的主流程判定。
- 统一 base：
  - base run dir：`GraphExp/results/run_20260404_111017`
  - 保留 `sim4` 主干：
    - `support_direction + random init + patel_kappa support prior + maxgap_kappa mask`
    - `directional_kappa_gate = true`
    - `causal_lag_main_weight = 0.25`
    - `warmup_then_orthogonal`
    - `detach_direction_from_main_after_epoch = 23`
    - `selection_score_mode = causal_lag_composite`
  - 仅做数据集自适应 override：
    - `sim2`: `csv_path=..\fMRI_dataset\sim2.csv`, `selector_audit_gt_path=..\fMRI_dataset\h2.txt`, `epochs=40`, `top_k_edges=11`, `selection_top_k=11`
    - `fMRI`: `csv_path=..\fMRI_dataset\fMRI.csv`, `selector_audit_gt_path=..\fMRI_dataset\h1.txt`, `epochs=100`, `top_k_edges=5`, `selection_top_k=5`
  - seeds：`11,22,33,44,55`
- 结果汇总：
  - `sim2`：
    - `best/exported/final = 0.8545 / 0.8182 / 0.8182`
    - `exported_vs_best_gap = -0.0364`
    - `final_vs_best_gap = -0.0364`
    - failure mode：
      - best `mixed_or_partial:5`
      - final `mixed_or_partial:4, weak_asymmetry:1`
  - `fMRI`：
    - `best/exported/final = 0.9200 / 0.8400 / 0.9200`
    - `exported_vs_best_gap = -0.0800`
    - `final_vs_best_gap = 0.0000`
    - failure mode：
      - best/final/exported 均为 `mixed_or_partial:5`
- selector replay：
  - `sim2`：
    - `causal_lag_composite = 0.8182`
    - `legacy = 0.7818`
    - `causal_lag_entropy_composite = 0.7636`
    - 结论：
      - `sim2` 上当前训练导出 selector 已经是最好结果；没有额外 selector 红利。
      - 因而 `sim4 -> sim2` 的表现主要由 training 主干决定，不是 selector miss。
  - `fMRI`：
    - `causal_lag_composite = 0.8400`
    - `causal_lag_entropy_composite = 0.8800`
    - `legacy(0.25) = 0.8000`
    - 结论：
      - `fMRI` 上确实存在明显的 exported selector 问题。
      - `entropy` replay 能把 `0.8400` 提到 `0.8800`，但仍低于训练 final 的 `0.9200`。
      - 这说明 `sim4 -> fMRI` 轨迹后期质量是好的，只是当前 `causal_lag_composite` 倾向更早 checkpoint。
- 解释：
  - `sim4` 基线主干并不是“对所有其他数据集都不适配”。
  - 从这次补跑看：
    - `sim2`：迁移效果相当稳，`best/final` 都不低，问题很小
    - `fMRI`：training ceiling 也不低，核心更像 selector / exported checkpoint 问题
  - 因而统一化真正的硬冲突仍然集中在：
    - `sim3`
    - 以及候选重新迁回 `sim4` 时是否还能守住原生 `sim4` ceiling
  - 换句话说：
    - `sim2 / fMRI` 不是当前统一失败的主因
    - 当前统一主线的决定性矛盾仍然在 `sim3 <-> sim4`
- 对用户问题的直接回答：
  - 如果都用 `sim4` 基线主干，当前 5-seed 实测均值是：
    - `sim2`: `0.8545 / 0.8182 / 0.8182`
    - `sim3`: `0.8778 / 0.7889 / 0.8222`
    - `sim4`: `0.8623 / 0.8098 / 0.8295`
    - `fMRI`: `0.9200 / 0.8400 / 0.9200`
  - 其中：
    - `sim2` 与 `fMRI` 基本都能接受
    - `sim3` 有 retention / selector 问题
    - `sim4` 自身仍是必须守住的 gate
- 证据：
  - `GraphExp/results/unify_replay_sim2_20260412_201731_h1_sim4_full_to_sim2.csv`
  - `GraphExp/results/unify_replay_sim2_20260412_201731_h1_sim4_full_to_sim2_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim2_20260412_211235_h1_sim4_full_to_sim2_selector.csv`
  - `GraphExp/results/unify_phase0_selector_sim2_20260412_211235_h1_sim4_full_to_sim2_selector_aggregate.csv`
  - `GraphExp/results/unify_replay_fMRI_20260412_203859_h1_sim4_full_to_fmri.csv`
  - `GraphExp/results/unify_replay_fMRI_20260412_203859_h1_sim4_full_to_fmri_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_fMRI_20260412_211235_h1_sim4_full_to_fmri_selector.csv`
  - `GraphExp/results/unify_phase0_selector_fMRI_20260412_211235_h1_sim4_full_to_fmri_selector_aggregate.csv`

### [Planned-2026-04-12] 结构部件定位实验

- 背景：
  - 截至当前结果，最像问题根因的不是时序卷积容量，而是：
    - `support_direction` 参数化下的 support / direction 分解
    - 主扩散去噪路径对方向的弱敏感性
    - 方向分支主要依赖 Patel directional supervision / causal-lag auxiliary 信号存活
  - 因而后续实验目标从“继续找统一配置”切换为“定位到底是哪个结构部件在失效”。
- 目标：
  - 把问题尽量定位到以下三类之一：
    - `support` 分支本身不行
    - `direction` 分支本身不行
    - 主扩散损失与 `support_direction` 参数化之间存在目标错位，导致方向分支长期拿不到有效学习信号
- 当前已知限制：
  - 现有 `run_dir` 只保存了：
    - `best` / `final` 导出的整张邻接矩阵
    - `quality_history.csv`
    - `model_final.pt`
  - 但没有保存每个 epoch 的 `support` / `direction` 分支参数或中间矩阵。
  - 因而严格的离线 `branch-swap` / `epoch-swap` 目前不能直接在历史 artifact 上完成；如果要做，必须新增保存或重跑带额外日志的实验。
- 执行顺序：
  1. `L1 = gradient-alignment probe`
  2. `L2 = branch-swap rerun`（需要补保存）
  3. `L3 = coupled vs support_direction`
  4. `L4 = support/direction epoch-swap rerun`（需要补保存）
- 各实验定义：
  - `L1 = gradient-alignment probe`
    - 目的：
      - 直接测量 direction branch 上：
        - 主扩散路径梯度 `grad_probe_diff_norm`
        - directional supervision 梯度 `grad_probe_dir_norm_weighted`
        - 两者夹角 `grad_probe_cosine`
      - 判断主路径到底是在支持方向学习、无视方向学习，还是在和方向监督互相打架。
    - 优先级最高的原因：
      - 代码已内置 probe，不需要先改模型结构。
      - 这是当前最接近“定位部件归因”的低成本实验。
    - 主要判定：
      - 若 `diff_norm` 很小、`dir_norm_weighted` 主导方向更新：
        - 说明方向学习主要靠辅助损失存活，问题更偏目标函数接口
      - 若 `cosine < 0` 频繁出现：
        - 说明主路径与方向监督在方向分支上存在系统性冲突
      - 若 `sim4` 比 `sim3` 更频繁出现低比值或负夹角：
        - 说明 `sim4` 的方向崩塌更像结构头/梯度路由问题，不像 selector 问题
  - `L2 = branch-swap rerun`
    - 目的：
      - 直接分离 `support` 与 `direction` 两个分支各自的上限
    - 设计：
      - `GT support + learned direction`
      - `learned support + GT direction`
    - 主要判定：
      - 若 `GT support + learned direction` 仍低，而 `learned support + GT direction` 很高：
        - 问题主在 direction 分支
      - 反之则 support 分支更可疑
    - 依赖：
      - 需要重跑并额外保存可重构的 branch-level 状态
  - `L3 = coupled vs support_direction`
    - 目的：
      - 判断问题是 `support_direction` 这套拆分本身引入的，还是任何参数化都会遇到
    - 主要判定：
      - 若 `coupled` 在 `sim4` 显著缓解 `symmetric_collapse` 或提升高 margin 指标：
        - 则“拆成 support × direction_gate”本身就是高风险设计
  - `L4 = support/direction epoch-swap rerun`
    - 目的：
      - 判断 late drift 到底来自 support 漂移还是 direction 漂移
    - 设计：
      - 固定早期优质 epoch 的 support，只换晚期 direction
      - 固定早期优质 epoch 的 direction，只换晚期 support
    - 主要判定：
      - 哪一边一换就坏，哪一边就是 late drift 的主体
- 当前决定：
  - 先执行 `L1 = gradient-alignment probe`。
  - `L2 / L4` 在没有 branch-level 中间状态保存之前，不做口头推断，等补齐保存后再正式启动。

### [L1] gradient-alignment probe：`sim3 control` vs `sim4 control`

- 日期：`2026-04-13`
- 阶段：结构部件定位实验 `L1`
- 数据集：
  - `sim3 control`：base `run_20260331_191723`
  - `sim4 control`：base `run_20260404_111017`
- Seeds：`11,22,33,44,55`
- 对照目的：
  - 比较 direction branch 在两条代表性主线上的梯度来源差异：
    - 主扩散路径梯度 `grad_probe_diff_norm`
    - directional supervision 梯度 `grad_probe_dir_norm_weighted`
    - 两者夹角 `grad_probe_cosine`
- 仅改动的参数：
  - `enable_gradient_alignment_probe = true`
- 固定不变的参数：
  - 各数据集沿用各自 control 配置，不改 training backbone
- 新增工具：
  - `GraphExp/summarize_grad_probe_runs.py`
  - 用于从 `quality_history.csv` 汇总 `grad_probe_*` 指标

结果：

- `sim3 control`：
  - 训练主指标与原 control 一致：`best/exported/final = 0.6889 / 0.6889 / 0.6889`
  - probe 有效 epoch 数：`15`
  - `probe_ratio_mean = 0.3669 +- 0.2050`
  - `probe_diff_norm_mean = 0.00539`
  - `probe_dir_norm_weighted_mean = 0.00231`
  - `probe_cosine_mean = +0.0319 +- 0.0429`
  - `probe_negative_frac = 0.2133 +- 0.1857`
  - 含义：
    - 在 `sim3` 的 joint 阶段，directional supervision 并没有压倒主扩散梯度
    - 主扩散路径对 direction branch 仍提供了同量级、略偏正向的更新信号
- `sim4 control`：
  - 训练主指标与原 control 接近：`best/exported/final = 0.8557 / 0.8098 / 0.8230`
  - probe 有效 epoch 数：`23`
  - `probe_ratio_mean = 7.0086 +- 1.7396`
  - `probe_diff_norm_mean = 0.00468`
  - `probe_dir_norm_weighted_mean = 0.02784`
  - `probe_cosine_mean = -0.0143 +- 0.0071`
  - `probe_negative_frac = 0.1652 +- 0.0325`
  - 含义：
    - 在 `sim4` 的 joint 阶段，directional supervision 对 direction branch 的更新量级约为主扩散梯度的 `7x`
    - 而且平均夹角略为负，说明主扩散路径并没有稳定支持方向学习，反而存在轻度冲突
- 额外结构性现象：
  - `sim3` 的 probe 只在前 `15` 个 epoch 有效，后段记录为 `legacy_detached`
  - `sim4` 的 probe 只在前 `23` 个 epoch 有效，后段记录为 `orthogonal_after_warmup`
  - 这不是脚本错误，而是说明：
    - 一旦进入 detach / orthogonal 阶段，主路径到 direction branch 的梯度按设计就被切断
    - 后期 direction branch 不再可能靠主扩散路径自行纠偏

判定：

- 是否通过：是，`L1` 已足够提供部件级定位信号
- 相对 control 的 paired delta：
  - `L1` 不以 F1 改进为目标；重点是梯度结构诊断
- 是否推进下一步：是，推进到 `L2/L3` 设计收敛

结论：

- `L1` 明确支持“目标函数接口 / 结构头分解失配”这一方向，而不支持“主要是时序卷积容量不够”的解释。
- 更具体地：
  - 在 `sim4` 上，direction branch 的学习几乎是被 directional supervision 单独拖着走
  - 主扩散路径对 direction branch 的直接学习信号非常弱，而且平均并非正对齐
  - 这与 `sim4` 的 `symmetric_collapse`、低 margin、对 Patel-family 的强依赖是同一条机制链
- 因而当前最可能的问题层级是：
  - `support_direction` 分解
  - 主扩散损失对 direction branch 的弱约束
  - 以及后期 detach / orthogonal 后 direction branch 无法再被主任务纠偏
- 基于 `L1`，后续优先级调整为：
  1. `L3 = coupled vs support_direction`
  2. 为 `L2 / L4` 补 branch-level 保存，再做 branch-swap / epoch-swap
  3. 不优先投入“加 epochs”或“加时序卷积容量”

- 证据：
  - `GraphExp/results/unify_replay_sim3_20260412_222139_l1_grad_probe_sim3_control.csv`
  - `GraphExp/results/unify_replay_sim3_20260412_222139_l1_grad_probe_sim3_control_aggregate.csv`
  - `GraphExp/results/unify_grad_probe_sim3_20260413_011536_l1_grad_probe_sim3_control.csv`
  - `GraphExp/results/unify_grad_probe_sim3_20260413_011536_l1_grad_probe_sim3_control_aggregate.csv`
  - `GraphExp/results/unify_replay_sim4_20260412_231055_l1_grad_probe_sim4_control.csv`
  - `GraphExp/results/unify_replay_sim4_20260412_231055_l1_grad_probe_sim4_control_aggregate.csv`
  - `GraphExp/results/unify_grad_probe_sim4_20260413_011551_l1_grad_probe_sim4_control.csv`
  - `GraphExp/results/unify_grad_probe_sim4_20260413_011551_l1_grad_probe_sim4_control_aggregate.csv`

### [Planned-2026-04-13] L3：`coupled` vs `support_direction`

- 目的：
  - 判断当前问题是否主要来自 `support_direction = symmetric support × direction_gate` 这套分解本身。
  - 也就是回答：
    - 如果不再把 support 和 direction 硬拆开，`sim4` 的 `symmetric_collapse` / 低 margin / 强 Patel 依赖会不会明显缓解？
- 数据集：
  - `sim3 control`
  - `sim4 control`
- 基线：
  - `sim3 control = run_20260331_191723`
  - `sim4 control = run_20260404_111017`
- 设计：
  - 在各自 control 主干上，切换：
    - `structure_parameterization: support_direction -> coupled`
  - 其余参数尽量保持不变
  - 但由于 CLI/实现约束，`coupled` 比较不能做到严格“只改一个参数”：
    - 必须关闭 `fixed_support_mask_mode`
    - 必须清空 direction-branch 专用 routing / detach 设置
    - 即：
      - `fixed_support_mask_mode = none`
      - `gradient_routing_mode = legacy`
      - `detach_direction_from_main_after_epoch = -1`
      - `freeze_direction_after_epoch = -1`
  - 仍保留的数据集主干因素：
    - `sim3`：原 control 的 `batch_mean + Patel directional supervision + no causal-lag`
    - `sim4`：原 control 的 `subject + causal_lag_main_weight=0.25 + directional_kappa_gate`
  - selector 处理：
    - 训练导出结果直接比较
    - 同时补做 selector replay，避免把 selector miss 误判为 `coupled` 训练失败
- 主要判定：
  - 若 `coupled` 在 `sim4` 上显著改善：
    - `best/exported/final`
    - `failure_mode`
    - `gt_signed_margin_median`
    - 或 selector replay 上界
    - 则支持“`support_direction` 分解本身就是高风险设计”
  - 若 `coupled` 无改善甚至更差：
    - 则问题更可能在主扩散目标本身对方向不敏感，而不只是参数化拆分

### [L3] `coupled` vs `support_direction`：`sim3 control` / `sim4 control`

- 日期：`2026-04-13`
- 阶段：结构部件定位实验 `L3`
- 数据集：
  - `sim3 control`
  - `sim4 control`
- Seeds：`11,22,33,44,55`
- 对照组：
  - `sim3 control = run_20260331_191723`
  - `sim4 control = run_20260404_111017`
- 仅改动的参数：
  - `structure_parameterization = coupled`
  - `fixed_support_mask_mode = none`
  - `gradient_routing_mode = legacy`
  - `detach_direction_from_main_after_epoch = -1`
  - `freeze_direction_after_epoch = -1`
- 固定不变的参数：
  - 各自 control 的数据集、epochs、pretrain、optimizer、directional supervision、selector 训练配置尽量保持不变
  - `sim4` 仍保留 `directional_kappa_gate = true` 与 `causal_lag_main_weight = 0.25`
- 重要比较限制：
  - 由于当前实现约束，`coupled` 不支持：
    - `fixed_support_mask_mode != none`
    - direction-branch 专用 routing / detach
  - 因而 `L3` 不是“只改一个参数”的纯净比较。
  - 它回答的是：
    - “去掉 `support_direction` 这一整套 factorization + scaffold 以后，会不会更好？”
    - 而不是严格回答“只有 factorization 公式本身是不是唯一问题”。

结果：

- 训练主指标：
  - `sim3 control`：
    - `best/exported/final = 0.6889 / 0.6889 / 0.6889`
  - `sim3 coupled`：
    - `0.2146 / 0.1691 / 0.2016`
    - `delta(best/exported/final) = -0.4743 / -0.5198 / -0.4873`
    - `final_failure_mode = weak_asymmetry:3, wrong_direction_asymmetry:2`
  - `sim4 control`：
    - `0.8623 / 0.8098 / 0.8295`
  - `sim4 coupled`：
    - `0.0802 / 0.0743 / 0.0782`
    - `delta(best/exported/final) = -0.7821 / -0.7355 / -0.7513`
    - `failure_mode = wrong_direction_asymmetry:5`
- selector replay：
  - `sim3 coupled`：
    - `causal_lag_composite = 0.1691`
    - `causal_lag_entropy_composite = 0.1691`
    - `legacy = 0.1886`
    - 对照 `sim3 control` 最强 selector `entropy = 0.6889`
  - `sim4 coupled`：
    - `causal_lag_composite = 0.0743`
    - `causal_lag_entropy_composite = 0.0743`
    - `legacy = 0.0764`
    - 对照 `sim4 control` 最强 selector `legacy = 0.8361`
- 关键观察：
  - `sim3` 上，`coupled` 不是“轻度退化”，而是主 strict F1 直接塌到 `0.17-0.21` 区间。
  - `sim4` 上更极端，`best` 均值只剩 `0.0802`，而且 failure mode 从 `symmetric_collapse` 变成了 `wrong_direction_asymmetry`。
  - selector replay 只能带来极小改善，完全救不回性能：
    - 说明这不是导出 epoch 选错，而是训练主干本身失效。
  - `sim4 coupled` 的 `best_signed_margin_median_mean = 0.5962` 很高，但 F1 极低：
    - 这意味着它不是“没学出方向”，而是学出了大量高置信但错误的方向。
    - 因而问题不是简单的“direction gate 太弱”，而是 `coupled` 下方向约束失控。

判定：

- 是否通过：否，`coupled` 明确失败
- 相对 control 的 paired delta：
  - `sim3`: `-0.4743 / -0.5198 / -0.4873`
  - `sim4`: `-0.7821 / -0.7355 / -0.7513`
- 是否推进下一步：是，但不再沿“直接改成 coupled”这条线继续

结论：

- `L3` 强烈反对“当前主要问题就是 `support_direction` factorization，本应直接改成 `coupled`”这一解释。
- 更准确地说：
  - `support_direction` 这套 factorization family 至少在当前框架里是 **load-bearing** 的
  - 直接去掉它，会让模型从：
    - `sim3` 的弱但可用方向解
    - `sim4` 的低 margin / collapse 解
    - 进一步退化成几乎不可用的错误方向解
- 因而从 `L1 + L3` 合并看：
  - `sim4` 的问题不是“factorization 多余”
  - 而是“在 factorization 存在的前提下，主扩散损失对 direction branch 的支持仍然太弱”
  - 也就是说：
    - `support_direction` 不是当前该删除的部件
    - 更可能需要改的是：
      - factorization 内部的方向约束方式
      - 或主扩散路径与方向分支之间的接口
- 仍保留的技术保留意见：
  - 由于 `coupled` 比较同时失去了 `fixed_support_mask` 与 direction-specific routing，`L3` 不能单独证明“factorization 公式本身完全无责”
  - 但它已经足够证明：
    - “直接换成 coupled”不是可行方向
    - 后续不应优先把精力投入纯参数化删除，而应转向 branch-level 归因与接口改造
- 基于 `L3` 的下一步优先级：
  1. 为 `L2 / L4` 补 branch-level 保存
  2. 做 `branch-swap / epoch-swap`
  3. 暂不继续 `coupled` 变体搜索

- 证据：
  - `GraphExp/results/unify_replay_sim3_20260413_080137_l3_coupled_sim3_control.csv`
  - `GraphExp/results/unify_replay_sim3_20260413_080137_l3_coupled_sim3_control_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim3_20260413_100538_l3_coupled_sim3_control_selector.csv`
  - `GraphExp/results/unify_phase0_selector_sim3_20260413_100538_l3_coupled_sim3_control_selector_aggregate.csv`
  - `GraphExp/results/unify_replay_sim4_20260413_083639_l3_coupled_sim4_control.csv`
  - `GraphExp/results/unify_replay_sim4_20260413_083639_l3_coupled_sim4_control_aggregate.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260413_100539_l3_coupled_sim4_control_selector.csv`
  - `GraphExp/results/unify_phase0_selector_sim4_20260413_100539_l3_coupled_sim4_control_selector_aggregate.csv`

### [Planned-2026-04-13] L4：`support@epoch_a + direction@epoch_b` 的 epoch-swap 定位

- 目的：
  - 在保留 `support_direction` factorization 的前提下，定位 `sim4` 的 retention / collapse 更像是：
    - support branch 退化
    - direction branch 退化
    - 还是两支单独看都不算坏，但后期组合接口失配
- 背景：
  - `L1` 已证明 `sim4` 上 direction branch 受到的 directional supervision 梯度约为主扩散梯度的 `7x`，且平均夹角略负。
  - `L3` 又证明不能简单切回 `coupled`。
  - 因而下一步不该继续发明 selector，而应直接做 branch-level 归因。
- 设计：
  - 先为 `support_direction` 训练补充逐 epoch branch snapshot：
    - `support_logits`
    - `support_weights`
    - `direction_logits`
    - `direction_gate`
    - 以及按真实导出公式重建的 `adj_raw / adj_causal`
  - 训练不改 backbone，只新增保存：
    - `save_support_direction_snapshots = true`
  - 先在 `sim4 control` 上执行，因为它是当前最关键、最难的数据集。
  - 离线分析时做全网格：
    - `support@epoch_a + direction@epoch_b`
    - 用现有 `selector_audit` 的 `strict_f1` 逻辑评估，而不是新造指标
- 主判定逻辑：
  - 若 `max_a F1(support@a, direction@final)` 能明显救回 final，而 `max_b F1(support@final, direction@b)` 不能：
    - 更支持 support branch 后期退化
  - 若相反：
    - 更支持 direction branch 后期退化
  - 若两边单独替换都救不回，但某些混合组合仍优于 `final_final`：
    - 更支持 branch interface / co-adaptation 失配
  - 若对角线最佳已经接近全局最佳：
    - 说明问题主要是 checkpoint retention，而不是跨支路错配
- 当前新增工具：
  - `GraphExp/analyze_support_direction_epoch_swap.py`
  - 以及训练侧 `support_direction_snapshots.npz`

### [L4] `sim4 control` epoch-swap：direction branch 是否解释 retention gap

- 日期：`2026-04-13`
- 阶段：结构部件定位实验 `L4`
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- Seeds：
  - `11,22,33,44`
- 训练改动：
  - 仅新增 `save_support_direction_snapshots = true`
  - 不改 backbone、不改 selector、不改 loss
- 离线分析：
  - 对每个 run 扫描全网格 `support@epoch_a + direction@epoch_b`
  - 用 `strict_f1@eps=0` 作为主判定，和 `selector_audit` 主指标一致

结果：

| seed | best_epoch | final_epoch | best | final | final gap | best support + final direction | final support + best direction | 结论 |
|------|-----------:|------------:|-----:|------:|----------:|-------------------------------:|-------------------------------:|------|
| 11 | 36 | 40 | 0.8852 | 0.8689 | 0.0164 | 0.8689 | 0.8852 | 只换 direction 即完全恢复 |
| 22 | 27 | 40 | 0.8361 | 0.8361 | 0.0000 | 0.8361 | 0.8361 | 本来就无 retention gap |
| 33 | 8 | 40 | 0.8197 | 0.7705 | 0.0492 | 0.7705 | 0.8197 | 只换 direction 即完全恢复 |
| 44 | 17 | 40 | 0.8689 | 0.8525 | 0.0164 | 0.8525 | 0.8689 | 只换 direction 即完全恢复 |

核心观察：

- 四个 seed 上都成立：
  - `best_support_with_final_direction_strict_f1 == final_strict_f1`
  - `best_direction_with_final_support_strict_f1 == best_strict_f1`
- 也就是说：
  - 换 support branch 不能把 final 拉高
  - 换 direction branch 可以把 final 精确拉回 best
- 对有 retention gap 的 `3/4` 个 seed（`11,33,44`）：
  - `direction_swap_gain_over_final` 分别为 `+0.0164, +0.0492, +0.0164`
  - 且都恰好等于各自的 `best-final gap`
- `seed22` 是控制例：
  - 该 seed 本来 `best=final`
  - epoch-swap 也相应显示两支都没有可恢复增益

判定：

- 是否通过：是，`L4` 已给出明确的 branch-level 归因
- 结论：
  - 在 `sim4 control` 当前配置下，`best-final retention gap` 的直接责任部件是 `direction branch`，不是 `support branch`
  - 更精确地说：
    - **在固定 skeleton 的前提下**，final checkpoint 比 best checkpoint 差，原因是 direction 分支后期方向排序变差
    - 不是 support 分支漂移导致的

重要限制：

- 这个结论的作用域是：
  - `sim4 control`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `strict_f1@eps=0`
- 因为 `sim4 control` 使用固定 support mask，且主评估本质上主要看每个保留 pair 的方向是否正确：
  - `L4` 更像是在定位“固定 skeleton 内部，谁决定 directed F1 的 retention”
  - 它**不能**单独证明“support branch 在所有意义上都不重要”
  - 它只证明：在这个 setting 下，当前 observed retention gap 不是 support epoch 选错造成的

与前面实验的合并解释：

- `L1`：`sim4` 上 direction branch 的主扩散梯度弱且略冲突
- `L3`：不能直接改成 `coupled`
- `L4`：最终掉点发生在 direction branch 的 checkpoint retention 上

因此当前最可信的结构性结论是：

- `support_direction` 这条路线仍然必须保留
- 但 `sim4` 的后期质量问题，核心不是 selector，也不是 support branch，而是：
  - direction branch 在 best epoch 之后无法被稳定保留
  - 这与 `warmup_then_orthogonal / detach` 后主路径不再纠偏的机制是一致的

- 证据：
  - `GraphExp/results/unify_replay_sim4_20260413_141742_l4_sim4_control_snapshots_smoke.csv`
  - `GraphExp/results/unify_replay_sim4_20260413_143925_l4_sim4_control_snapshots_pair.csv`
  - `GraphExp/results/unify_replay_sim4_20260413_151733_l4_sim4_control_seed33.csv`
  - `GraphExp/results/unify_epoch_swap_sim4_20260413_143820_l4_sim4_control_seed11_summary.csv`
  - `GraphExp/results/unify_epoch_swap_sim4_20260413_151708_l4_sim4_control_seed22_summary.csv`
  - `GraphExp/results/unify_epoch_swap_sim4_20260413_153711_l4_sim4_control_seed33_summary.csv`
  - `GraphExp/results/unify_epoch_swap_sim4_20260413_151708_l4_sim4_control_seed44_summary.csv`
  - `GraphExp/results/unify_epoch_swap_sim4_20260413_l4_sim4_control_4seed_summary.csv`

### [Planned-2026-04-13] L5：`sim4 control` direction-retention repair

- 目的：
  - 在 `L4` 已确认 retention gap 发生在 direction branch 之后，进一步回答：
    - 问题主要来自 direction branch 参数在 late stage 持续更新
    - 还是主要来自 Patel directional supervision 在 late stage 持续施压
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- seeds：
  - `11,22,33,44,55`
- 设计：
  - `L5-A = freeze@30`
    - `freeze_direction_after_epoch = 30`
  - `L5-B = dir_end@30`
    - `directional_loss_end_epoch = 30`
  - 其余配置保持 `sim4 control` 不变
- 选择 `30` 的原因：
  - 历史 tracker 中，`sim4` 的 `freeze@30` 曾表现为有效 retention repair
  - 同时 `L4` 说明当前问题是 late direction retention，而不是 support / selector
  - 因而 `epoch 30` 是一个合适的“保留成熟 direction、避免最后阶段漂移”的 first test
- 判定：
  - 如果 `freeze@30` 明显优于 `dir_end@30`：
    - 更支持“问题主要是 direction branch 参数继续漂”
  - 如果两者接近：
    - 更支持“主要是晚期 directional supervision 造成 drift”
  - 如果两者都无效：
    - 则需要继续测更靠近 routing 接口的改动，而不是简单 retention fix

### [L5] `sim4 control`：`freeze_direction_after_epoch=30` vs `directional_loss_end_epoch=30`

- 日期：`2026-04-13`
- 阶段：结构部件定位实验 `L5`
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- seeds：
  - `11,22,33,44,55`
- 改动：
  - `L5-A = freeze@30`
    - `freeze_direction_after_epoch = 30`
  - `L5-B = dir_end@30`
    - `directional_loss_end_epoch = 30`
  - 其余配置保持 `sim4 control` 不变

结果汇总：

| 变体 | best | exported | final | exported-best gap | final-best gap | best strict@0.1 | final strict@0.1 |
|------|-----:|---------:|------:|------------------:|---------------:|----------------:|-----------------:|
| control | 0.8623 | 0.8098 | 0.8295 | -0.0525 | -0.0328 | 0.0245 | 0.0372 |
| freeze@30 | 0.8525 | 0.8000 | 0.8262 | -0.0525 | -0.0262 | 0.0000 | 0.0065 |
| dir_end@30 | 0.8492 | 0.8000 | 0.8197 | -0.0492 | -0.0295 | 0.0000 | 0.0372 |

paired delta（相对 control）：

- `freeze@30`：
  - `delta(best/exported/final) = -0.0098 / -0.0098 / -0.0033`
  - `delta(exported_vs_best_gap / final_vs_best_gap) = +0.0000 / +0.0066`
- `dir_end@30`：
  - `delta(best/exported/final) = -0.0131 / -0.0098 / -0.0098`
  - `delta(exported_vs_best_gap / final_vs_best_gap) = +0.0033 / +0.0033`

seed 级现象：

- `freeze@30`：
  - 对 `seed33,44` 的 final retention 有改善
  - 但对 `seed11,22,55` 没有形成统一收益，且 `seed11/22/55` 的 best 或 final 均下降
- `dir_end@30`：
  - 整体更像“轻微改 gap，但不改 ceiling”
  - `seed55` 保持 `final=best`
  - 但 `seed11/22/44` 的 final 不升反降
- 两个 treatment 的最终 failure mode 都仍是：
  - `symmetric_collapse:5`

结论：

- `L5` 更支持“late-stage 问题不只是 Patel supervision 继续施压，而是 direction branch 参数本身继续更新/漂移”：
  - 因为 `freeze@30` 对 primary strict F1 的 final gap 修复幅度大于 `dir_end@30`
- 但这个支持是 **弱支持**，不是强阳性：
  - `freeze@30` 只带来很小的 `final-best gap` 改善（`-0.0328 -> -0.0262`）
  - 而且 `best` 与 `final` headline 指标都没有超过当前 control
- 更重要的是：
  - 两个固定 `epoch=30` 的 retention fix 都没有解决机制问题
  - 所有 seeds 仍是 `symmetric_collapse`
  - `freeze@30` 甚至显著压低了高置信方向质量：
    - `final strict@0.1 = 0.0372 -> 0.0065`
- 因而这次实验的实际判定是：
  - **“简单晚期截断”不是当前 `sim4 control` 的充分修复**
  - 它最多说明：
    - 继续更新 direction branch 参数，比继续保留 Patel supervision 更像是 drift 来源
    - 但把它在固定 epoch 硬冻结，并不能稳定保留当前 control 的高质量方向解

与历史 tracker 的关系：

- 旧 tracker 中 `freeze@30` 曾是更强的 `sim4` retention fix。
- 但当前 replay 基线与旧实验链并不完全同构：
  - 当前 control 使用的是今天的 `run_20260404_111017`
  - 当前基线 headline 已高于旧 tracker 中那条历史基线
- 因而对“现在这条 control 主干”，应以本轮 replay 结论为准，而不是直接沿用旧 freeze30 判断。

下一步含义：

- 不建议把 `freeze@30` 或 `dir_end@30` 直接升为默认配置。
- 下一层更有价值的问题不是“再试一个固定 freeze epoch”，而是：
  - 为什么 `direction branch` 在后期仍会偏离 best epoch 的方向排序
  - 以及这种偏离是否来自：
    - `warmup_then_orthogonal` 切换后的无纠偏状态
    - causal-lag 主路径对 direction 的后期更新
    - 或 selector proxy 与方向 margin 漂移之间的错位

- 证据：
  - `GraphExp/results/unify_replay_sim4_20260413_173351_l5_sim4_control_freeze30_seq.csv`
  - `GraphExp/results/unify_replay_sim4_20260413_173351_l5_sim4_control_freeze30_seq_aggregate.csv`
  - `GraphExp/results/unify_replay_sim4_20260413_190820_l5_sim4_control_dirend30_seq.csv`
  - `GraphExp/results/unify_replay_sim4_20260413_190820_l5_sim4_control_dirend30_seq_aggregate.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich.csv`
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich_aggregate.csv`
  - `GraphExp/results/unify_l5_sim4_control_retention_compare_aggregate.csv`

### [Planned-2026-04-13] L6：切开 `warmup_then_orthogonal` 之后的 late direction update 来源

- 目的：
  - 在 `L4/L5` 基础上继续定位：
    - `epoch > 23` 之后导致 direction drift 的主要来源
    - 更像是 `causal_lag_main` 在 orthogonal 阶段继续单独推 direction branch
    - 还是 late directional supervision 本身也在继续推偏
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- seeds：
  - `11,22,33,44,55`
- 已知代码机制：
  - `warmup_then_orthogonal` 在 `epoch > detach_direction_from_main_after_epoch` 后变成：
    - `main = support_only`
    - `structure_regularizers = support_only`
    - `causal_lag_main = direction_only`
  - 因而 `epoch > 23` 时，仍沿图路径更新 direction branch 的核心项就是 `causal_lag_main`
- 设计：
  - `L6-A = legacy_detach23`
    - `gradient_routing_mode = legacy`
    - 保留 `detach_direction_from_main_after_epoch = 23`
    - 含义：
      - `epoch > 23` 后，`causal_lag_main` 也不再更新 direction branch
      - 但 Patel directional supervision 仍持续到训练结束
  - `L6-B = legacy_detach23 + dir_end23`
    - `gradient_routing_mode = legacy`
    - `directional_loss_end_epoch = 23`
    - 保留 `detach_direction_from_main_after_epoch = 23`
    - 含义：
      - `epoch > 23` 后，graph-path 与 directional supervision 都不再更新 direction branch
      - 近似于“在 routing switch 点把 direction branch 定格”
- 判定：
  - 若 `L6-A` 已明显改善：
    - 更支持“late causal-lag direction-only update”是主漂移源
  - 若 `L6-A` 改善有限，但 `L6-B` 进一步明显改善：
    - 更支持“late directional supervision 也是重要漂移源”
  - 若两者都无明显改善：
    - 则问题更像发生在 `epoch 23` 前，或更像来自 switch 本身导致的无纠偏状态，而不是某个 late loss 单独作祟

### [L6] `sim4 control`：`legacy_detach23` / `legacy_detach23 + dir_end23`

- 日期：`2026-04-14`
- 阶段：结构部件定位实验 `L6`
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- seeds：
  - `11,22,33,44,55`
- 改动：
  - `L6-A = legacy_detach23`
    - `gradient_routing_mode = legacy`
    - `detach_direction_from_main_after_epoch = 23` 保持不变
  - `L6-B = legacy_detach23 + dir_end23`
    - `gradient_routing_mode = legacy`
    - `directional_loss_end_epoch = 23`
    - `detach_direction_from_main_after_epoch = 23` 保持不变

结果汇总：

| 变体 | best | exported | final | exported-best gap | final-best gap | final strict@0.1 |
|------|-----:|---------:|------:|------------------:|---------------:|-----------------:|
| control | 0.8623 | 0.8098 | 0.8295 | -0.0525 | -0.0328 | 0.0372 |
| legacy_detach23 | 0.8361 | 0.7869 | 0.8000 | -0.0492 | -0.0361 | 0.0000 |
| legacy_detach23 + dir_end23 | 0.8426 | 0.8033 | 0.8000 | -0.0393 | -0.0426 | 0.0065 |

paired delta（相对 control）：

- `legacy_detach23`：
  - `delta(best/exported/final) = -0.0262 / -0.0229 / -0.0295`
  - `delta(exported_vs_best_gap / final_vs_best_gap) = +0.0033 / -0.0033`
- `legacy_detach23 + dir_end23`：
  - `delta(best/exported/final) = -0.0197 / -0.0066 / -0.0295`
  - `delta(exported_vs_best_gap / final_vs_best_gap) = +0.0131 / -0.0098`

seed 级现象：

- `legacy_detach23`：
  - `seed11` exported 恢复到 best，但 final 仍掉
  - `seed55` final 直接掉到 `0.7541`
  - aggregate 没有形成净收益
- `legacy_detach23 + dir_end23`：
  - 相比 `legacy_detach23`，best/exported 略有回升
  - 但 final 均值没有回升，仍只有 `0.8000`
  - `seed55` 依然是明显掉点：`best=0.8525 -> final=0.7705`
- 两个 treatment 的最终 failure mode 都仍是：
  - `symmetric_collapse:5`

结论：

- `L6` 不支持“late causal-lag direction-only update 是当前 sim4 drift 主因”这个解释。
- 更具体地：
  - 只把 `warmup_then_orthogonal` 改成 `legacy_detach23`，也就是切掉 `epoch>23` 的 `causal_lag_main -> direction` 通路后：
    - 结果不是改善，而是 `best/exported/final` 全面低于 control
  - 进一步再关掉 late directional supervision（`dir_end23`）后：
    - `best/exported` 相对 `legacy_detach23` 有轻微回升
    - 但 `final` 仍没有回到 control 水平
- 因而当前最可信的机制判断变成：
  - `late causal-lag direction-only update` 不是主要作恶项，甚至更像是在帮助维持一部分方向质量
  - `late directional supervision` 可能有一定副作用，但也不是单独的主因
  - 真正的问题更像是：
    - 训练轨迹在 `epoch 23` 之前已经埋下了后期弱 margin / collapse 的趋势
    - 或 `warmup_then_orthogonal` 进入“主路径不再纠偏”的状态后，direction branch 缺少足够稳定的保真机制
    - 但这个保真问题不能靠简单移除某一条 late loss 通路来修复

与 `L4/L5` 的合并结论：

- `L4`：掉点直接发生在 `direction branch`
- `L5`：简单晚期 freeze / dir_end 不是充分修复
- `L6`：切掉 late `causal_lag -> direction` 更新也不是修复

所以当前更强的结论是：

- 问题层级已经从“哪个 loss 在后期推偏了 direction”收缩到：
  - `direction branch` 的表示/保真机制本身不稳
  - 以及 `warmup_then_orthogonal` 后缺乏能持续校正它的正向主任务信号
- 下一步不该继续做更多 late-loss on/off，而应优先考虑：
  - 在 switch 前后增加 branch-level 质量跟踪
  - 或直接设计能给 direction branch 提供持续校正信号的结构改动

- 证据：
  - `GraphExp/results/unify_replay_sim4_20260413_205943_l6_sim4_control_legacy_detach23.csv`
  - `GraphExp/results/unify_replay_sim4_20260413_205943_l6_sim4_control_legacy_detach23_aggregate.csv`
  - `GraphExp/results/unify_replay_sim4_20260414_080000_l6_sim4_control_legacy_detach23_dirend23.csv`
  - `GraphExp/results/unify_replay_sim4_20260414_080000_l6_sim4_control_legacy_detach23_dirend23_aggregate.csv`
  - `GraphExp/results/unify_sim4_control_retention_routing_compare_aggregate.csv`

### [L7] `sim4 control` pair-level direction margin drift

- 日期：`2026-04-14`
- 阶段：结构部件定位实验 `L7`
- 目的：
  - 把前面“direction branch 保真不稳”“后期缺少持续纠偏信号”拆成可观测的 pair-level 现象。
  - 具体判定：
    - `best -> final` 掉点主要来自少数 GT pair 翻错，还是大量 GT pair 虽未翻错但 margin 被压到 near-tie。
    - `epoch 23` 之后是否出现明显的 branch-level margin 退化。
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- seeds：
  - `11,22,33,44,55`
- runs：
  - `run_20260413_141746`
  - `run_20260413_143928`
  - `run_20260413_151736`
  - `run_20260413_145716`
  - `run_20260414_183827`
- 工具：
  - `GraphExp/analyze_direction_margin_drift.py`

输出文件：

- `GraphExp/results/unify_margin_drift_sim4_20260414_185633_l7_sim4_control_5seed.csv`
- `GraphExp/results/unify_margin_drift_sim4_20260414_185633_l7_sim4_control_5seed_aggregate.csv`
- `GraphExp/results/unify_margin_drift_sim4_20260414_185633_l7_sim4_control_5seed_epochs.csv`
- `GraphExp/results/unify_margin_drift_sim4_20260414_185633_l7_sim4_control_5seed_epochs_aggregate.csv`
- `GraphExp/results/unify_margin_drift_sim4_20260414_185633_l7_sim4_control_5seed_pairs.csv`
- `GraphExp/results/unify_margin_drift_sim4_20260414_185633_l7_sim4_control_5seed_pair_transitions.csv`

关键 aggregate：

| 指标 | 数值 |
|------|-----:|
| best GT strict F1 mean | 0.8525 |
| exported GT strict F1 mean | 0.8000 |
| final GT strict F1 mean | 0.8361 |
| best-correct -> final-correct | 0.8098 |
| best-correct -> final-wrong-or-zero | 0.0426 |
| best-correct -> final-low-margin | 0.1836 |
| best-correct -> final-strong-margin | 0.6262 |
| best -> final same-sign | 0.9311 |
| best -> final sign-flip | 0.0689 |
| mean GT margin delta (final-best) | -0.0020 |
| mean abs-margin drop (final-best) | -0.0039 |

pair transition 计数（5 seeds 全部 GT pair 合并）：

| transition | count |
|-----------|------:|
| `correct_to_correct` | 191 |
| `correct_to_near_zero` | 56 |
| `correct_to_wrong` | 13 |
| `wrong_to_correct` | 8 |
| `wrong_to_wrong` | 37 |

epoch-level margin 轨迹：

| epoch | GT margin median mean | GT positive frac mean | GT margin min mean |
|------:|----------------------:|----------------------:|-------------------:|
| 23 | 0.0089 | 0.7902 | -0.0289 |
| 30 | 0.0082 | 0.8197 | -0.0370 |
| 40 | 0.0071 | 0.8361 | -0.1473 |

核心观察：

- `best -> final` 的主模式不是“大面积翻错”，而是“多数 pair 仍保持原符号，但不少正确 pair 被压到 near-tie”。
  - 直接看 transition count：`correct_to_near_zero = 56`，明显多于 `correct_to_wrong = 13`。
  - 直接看 aggregate fraction：`best_correct_final_low_margin = 0.1836`，而 `best_correct_final_wrong_or_zero = 0.0426`。
  - 也就是说，在本轮 drift 里，“还对但很虚”的 pair 比“直接翻错”的 pair 更常见。
- 但这也不是“只有统一的轻微软化”，因为少数 pair 的 final 退化非常剧烈。
  - `correct_to_wrong` 的平均 `margin_delta_final_minus_best = -0.0290`，中位数 `-0.0210`。
  - 代表性样本：
    - `seed33: 7->8`，`best_margin = 0.0011 -> final_margin = -0.1145`
    - `seed44: 9->10`，`0.0020 -> -0.0494`
    - `seed44: 49->50`，`0.0026 -> -0.0366`
    - `seed33: 29->30`，`0.0223 -> -0.0137`
- 脆弱 pair 不是完全随机散落，存在跨 seed 重复出现的局部难点。
  - `correct_to_wrong` 重复出现的 pair 包括：
    - `(9,10)`、`(29,30)`、`(39,40)`、`(49,50)`
  - `correct_to_near_zero` 重复出现更多：
    - `(3,8)` 与 `(18,19)` 在 5 个 seed 里都出现
    - `(3,4)`、`(8,13)`、`(33,34)`、`(33,38)` 在 4 个 seed 里出现
  - 这说明问题不是单纯 seed 噪声，而是某些局部方向关系长期开不出稳定 margin。
- `epoch 23 -> 40` 期间，不是“整体符号突然反转”，而是“低-margin 状态延续，同时 worst-case pair 继续恶化”。
  - `GT positive frac mean` 从 `0.7902 -> 0.8361` 并没有全面变差。
  - 但 `GT margin median mean` 从 `0.0089 -> 0.0071` 继续缩小。
  - 更关键的是 `GT margin min mean` 从 `-0.0289` 掉到 `-0.1473`，说明后期出现了少数明显更坏的负向 outlier。
  - 结合 `pair_abs_margin_mean` 全程都只有 `1e-3` 量级，可以看出 direction branch 一直工作在非常薄的 margin 带上。

结论：

- `direction branch` 的“保真机制不稳”现在可以更精确定义为：
  - 它不是在 best 之后把大多数 GT pair 全部重新排错；
  - 而是把大量本就不厚实的正确方向维持在极薄 margin 上，后期很容易继续被压向 near-tie；
  - 同时少数局部 pair 会从这个薄 margin 带中直接穿过 0，变成明显翻错。
- “进入 `warmup_then_orthogonal` 后缺少持续纠偏信号”现在也有了更具体的含义：
  - 后期训练没有表现出“持续把弱正确 pair 往更安全的正 margin 推开”的趋势；
  - 相反，median margin 继续收缩，反复出现的脆弱 pair 也没有被系统性修复；
  - 因此后期更像是在维持一个低-margin、易漂移的方向解，而不是在逐步巩固它。
- 这与 `L4/L5/L6` 是一致的：
  - `L4` 说明掉点发生在 `direction branch`
  - `L5/L6` 说明简单切掉某个 late loss 不是修复
  - `L7` 进一步说明根因更像：
    - branch-level margin 本身太薄
    - 后期缺少持续把正确方向“拉离决策边界”的正向机制
    - 所以最终表现为“near-tie 压缩 + 少数 recurrent pair 翻错”，而不是单一一次性的全局反号

对后续实验的指向：

- 下一步不应继续堆叠 selector 公式，优先级更高的是直接验证如何提高 late-stage direction margin retention。
- 更具体的训练目标应该是：
  - 让 `final` 更接近 `best`
  - 让 recurrent fragile pairs 在后期持续增大正 margin
  - 而不是只提高某个单点 epoch 的 lucky F1

### [L8] `sim4 control` switch-timing retention test

- 日期：`2026-04-14`
- 阶段：结构部件定位实验 `L8`
- 目的：
  - 直接验证当前 retention gap 是否主要来自 `warmup_then_orthogonal` 切换过早。
  - 若把 `detach_direction_from_main_after_epoch` 往后推，检查 `exported/final`、gap 与 margin 是否系统改善。
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- seeds：
  - `11,22,33,44,55`
- 设计：
  - `L8-A`：`detach_direction_from_main_after_epoch = 30`
  - `L8-B`：`detach_direction_from_main_after_epoch = 35`
  - `L8-C`：`detach_direction_from_main_after_epoch = 40`
    - 在 `40`-epoch 训练内等价于整段保持 joint routing，不进入 orthogonal split
- 不改：
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `causal_lag_main_weight = 0.25`
  - `directional_schedule = plateau`
  - `directional_target_ratio = 0.01`

结果汇总：

| 变体 | best | exported | final | exported-best gap | final-best gap | best margin med | final margin med |
|------|-----:|---------:|------:|------------------:|---------------:|----------------:|-----------------:|
| control (`switch=23`) | 0.8623 | 0.8098 | 0.8295 | -0.0525 | -0.0328 | 0.00854 | 0.00764 |
| `switch=30` | 0.8590 | 0.7902 | 0.8230 | -0.0689 | -0.0361 | 0.00873 | 0.00676 |
| `switch=35` | 0.8590 | 0.7803 | 0.8197 | -0.0787 | -0.0393 | 0.00982 | 0.00705 |
| `switch=40` | 0.8492 | 0.8000 | 0.8000 | -0.0492 | -0.0492 | 0.00953 | 0.00715 |

seed 级现象：

- `switch=30`：
  - 只有 `seed11` 的 `final-best gap` 改善到 `-0.0164`
  - 但 `seed33` 掉到 `best=0.8689 -> final=0.7705`
  - aggregate `final` 低于 control
- `switch=35`：
  - `seed11` 达到 `best=final=0.9016`
  - 但 `seed22/33/44` 全都低于 control final
  - exported gap 进一步恶化到 `-0.0787`
- `switch=40`：
  - 没有进入 orthogonal split 也没有修复问题
  - `final` 均值掉到 `0.8000`
  - `seed33` 进一步掉到 `0.7213`

failure mode：

- 三个变体在 `best/exported/final` 上仍然全部是：
  - `symmetric_collapse: 5/5`

结论：

- `L8` 不支持“当前 sim4 retention gap 的主因只是 orthogonal switch 太早”。
- 更具体地：
  - 把 switch 从 `23` 推迟到 `30` 或 `35`，并没有形成更好的 aggregate `final`
  - 直接取消整段 orthogonal split（`switch=40`）也没有修复，反而把 `final-best gap` 扩大到 `-0.0492`
- 这说明：
  - 问题不是一个简单的“23 太早，晚一点就好”的时点错误
  - `post-switch 缺少持续纠偏信号` 这个判断仍成立，但它不是通过“单纯把 switch 往后推”就能修复的
  - 更像是 direction branch 在整个训练轨迹里已经形成了脆弱的低-margin 表示；切换只是暴露了它，不是唯一根因
- 与 `L7` 合并后，更强的结论是：
  - `L7`：后期表现为 `near-tie` 压缩 + 少数 recurrent pair 翻错
  - `L8`：这不是简单由 switch 时点过早触发，因为晚切换或不切换都没有把它修好
  - 因而下一步优先级应从“调 switch 时点”转向：
    - 直接增强 direction branch 的 late-stage margin retention 机制
    - 或改造 direction supervision / parameterization，使其能稳定拉开正确方向 margin

证据：

- `GraphExp/results/unify_replay_sim4_20260414_190835_l8_sim4_control_switch30.csv`
- `GraphExp/results/unify_replay_sim4_20260414_190835_l8_sim4_control_switch30_aggregate.csv`
- `GraphExp/results/unify_replay_sim4_20260414_204711_l8_sim4_control_switch35.csv`
- `GraphExp/results/unify_replay_sim4_20260414_204711_l8_sim4_control_switch35_aggregate.csv`
- `GraphExp/results/unify_replay_sim4_20260414_222512_l8_sim4_control_switch40.csv`
- `GraphExp/results/unify_replay_sim4_20260414_222512_l8_sim4_control_switch40_aggregate.csv`
- `GraphExp/results/unify_l8_sim4_switch_timing_compare_aggregate.csv`

### [L9] `sim4 control` persistent direction-prior anchor

- 日期：`2026-04-15`
- 阶段：结构部件定位实验 `L9`
- 目的：
  - 检验 `direction branch` 是否主要缺少“持续的方向锚点”。
  - 不增加新的监督梯度，而是通过 `direction_logit_bias_scale` 把 Patel tau 方向先验作为 persistent bias 加到 direction logits。
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- seeds：
  - `11,22,33,44,55`
- 设计：
  - `L9-A`：`direction_logit_bias_scale = 0.25`
  - `L9-B`：`direction_logit_bias_scale = 0.5`
  - `L9-C`：`direction_logit_bias_scale = 1.0`
- 不改：
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `directional_target_ratio = 0.01`
  - `direction_lr_multiplier = 1.0`

结果汇总：

| 变体 | best | exported | final | exported-best gap | final-best gap | best margin med | final margin med |
|------|-----:|---------:|------:|------------------:|---------------:|----------------:|-----------------:|
| control | 0.8623 | 0.8098 | 0.8295 | -0.0525 | -0.0328 | 0.00854 | 0.00764 |
| `dirbias=0.25` | 0.8557 | 0.8000 | 0.8328 | -0.0557 | -0.0230 | 0.00899 | 0.00751 |
| `dirbias=0.5` | 0.8525 | 0.7803 | 0.8262 | -0.0721 | -0.0262 | 0.00859 | 0.00735 |
| `dirbias=1.0` | 0.8623 | 0.8000 | 0.8262 | -0.0623 | -0.0361 | 0.00844 | 0.00761 |

seed 级现象：

- `dirbias=0.25`：
  - `seed11/22/44` 都达到 `best=final`
  - `seed55` 也只剩 `-0.0164` 的小 gap
  - 但 `seed33` 仍掉到 `0.7541`
  - aggregate `final-best gap` 从 control 的 `-0.0328` 收窄到 `-0.0230`
- `dirbias=0.5`：
  - `seed11/22` 仍能保住 `best=final`
  - 但 exported 明显变差，`seed11` exported 掉到 `0.7541`
  - aggregate `final` 反而低于 control
- `dirbias=1.0`：
  - `seed11` best/final 提到 `0.9016/0.8852`
  - `seed55` 达到 `best=final=0.8689`
  - 但 `seed33/44` 仍明显掉点
  - aggregate `final` 仍只有 `0.8262`

failure mode：

- 三个 bias 变体在 `best/exported/final` 上仍然全部是：
  - `symmetric_collapse: 5/5`

结论：

- `L9` 支持“当前 direction branch 确实从 persistent anchor 中获得了一些 retention 收益”，但这个收益是有限的。
- 最有价值的档位是 `direction_logit_bias_scale = 0.25`：
  - 它把 aggregate `final` 从 `0.8295` 提到 `0.8328`
  - 同时把 `final-best gap` 从 `-0.0328` 收窄到 `-0.0230`
  - 说明轻度 persistent anchor 确实能帮助部分 seed 保住后期方向质量
- 但更强 bias（`0.5` / `1.0`）没有继续带来 aggregate 改善：
  - `0.5` 和 `1.0` 的 `final` 都低于 `0.25`
  - `0.5` 尤其伤 exported，`exported-best gap` 恶化到 `-0.0721`
- 更关键的是：
  - 所有档位仍然是 `symmetric_collapse: 5/5`
  - `final_signed_margin_median` 也没有系统性高于 control
  - 这说明 persistent Patel anchor 只能帮助“少数关键 pair 不掉出正确侧”，但没有把整个 direction branch 变成真正的高-margin 解
- 因而 `L9` 与 `L7/L8` 合并后的结论是：
  - 仅靠调 switch 时点不行
  - 仅靠加 persistent direction prior 也不行
  - 但轻度 persistent anchor 的确有正向信号，说明问题里有一部分确实是“后期缺锚点”
  - 真正未解的部分仍是：
    - branch-level margin 普遍太薄
    - 导出空间仍落在 `symmetric_collapse`
    - 需要比 simple bias 更强的 margin-retention / parameterization 修复

对后续实验的指向：

- 下一步不应再继续单纯加大 Patel 强度，而应优先测试：
  - 能否直接把正确方向 pair 从 near-tie 区域推开
  - 或能否在导出公式 / parameterization 层面减少 `support * gate` 的低-margin 压缩

证据：

- `GraphExp/results/unify_replay_sim4_20260415_081320_l9_sim4_control_dirbias025.csv`
- `GraphExp/results/unify_replay_sim4_20260415_081320_l9_sim4_control_dirbias025_aggregate.csv`
- `GraphExp/results/unify_replay_sim4_20260415_094100_l9_sim4_control_dirbias05.csv`
- `GraphExp/results/unify_replay_sim4_20260415_094100_l9_sim4_control_dirbias05_aggregate.csv`
- `GraphExp/results/unify_replay_sim4_20260415_110852_l9_sim4_control_dirbias10.csv`
- `GraphExp/results/unify_replay_sim4_20260415_110852_l9_sim4_control_dirbias10_aggregate.csv`
- `GraphExp/results/unify_l9_sim4_dirbias_compare_aggregate.csv`

### [L10] `sim4 control` self-distilled anti-tie retention

- 日期：`2026-04-15`
- 阶段：结构部件定位实验 `L10`
- 目的：
  - 用更论文友好的方式直接测试：是否能在不把 Patel 先验注入 forward inference 的前提下，减少 `best -> final` 的 late-stage direction drift。
  - 针对 `L7/L8/L9` 暴露的核心问题：`sim4` 后期主要不是大面积彻底翻边，而是大量正确 pair 的 direction margin 被压回 `near-tie` 区域，最终在 top-k 导出时表现为 retention gap。
- 核心机制：
  - 新增 `self-distilled anti-tie retention loss`。
  - teacher 不是 Patel，而是模型自身 earlier / EMA 的 causal direction logits。
  - 仅在高 teacher-margin 的 confident pairs 上生效，并要求 student 至少保住一部分 teacher margin。
  - 这样做的目标是保留模型已经学出来的方向结构，而不是继续向外部先验对齐。
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- 实验设计：
  - `L10-A`:
    - `self_distill_direction_retention_lambda = 0.05`
    - `self_distill_direction_retention_start_epoch = 24`
    - `self_distill_direction_retention_ema = 0.9`
    - `self_distill_direction_retention_active_quantile = 0.5`
    - `self_distill_direction_retention_margin_scale = 0.5`
    - `self_distill_direction_retention_margin_floor = 0.02`
  - `L10-B`:
    - 与 `L10-A` 相同，仅把 `lambda` 提到 `0.1`
- 不改：
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `direction_logit_bias_scale = 0.0`
  - `selection_score_mode = causal_lag_composite`
- 判定重点：
  - aggregate `final_primary_strict_f1` 是否高于 control
  - `final-best gap` 是否系统收窄
  - `final_signed_margin_median` 是否上升
  - failure mode 是否仍然全部落在 `symmetric_collapse`
- 记录状态：
  - 代码改动已完成，包含：
    - 新 loss
    - 本地训练日志与 `quality_history` 字段
    - `config.npy` / replay override 支持
  - `dry-run` 已通过

结果汇总：

| 变体 | best | exported | final | exported-best gap | final-best gap | best margin med | final margin med |
|------|-----:|---------:|------:|------------------:|---------------:|----------------:|-----------------:|
| control | 0.8623 | 0.8098 | 0.8295 | -0.0525 | -0.0328 | 0.00854 | 0.00764 |
| `selfdistill_0.05` | 0.8492 | 0.7902 | 0.8197 | -0.0590 | -0.0295 | 0.00915 | 0.00720 |
| `selfdistill_0.1` | 0.8525 | 0.7902 | 0.8164 | -0.0623 | -0.0361 | 0.00911 | 0.00748 |

seed 级现象：

- `selfdistill_0.05`：
  - `seed22/55` 达到 `best=final`
  - 但 `seed44` 变成 `best=0.8525 -> final=0.7869`
  - aggregate `final` 比 control 低 `-0.0098`
- `selfdistill_0.1`：
  - `seed55` 也达到 `best=final`
  - 但 `seed11` exported 明显掉到 `0.7705`
  - `seed33/44` 的 final 仍然偏低
  - aggregate `final` 比 control 低 `-0.0131`

failure mode：

- `control / selfdistill_0.05 / selfdistill_0.1` 三组在 `best/final` 上仍然全部是：
  - `symmetric_collapse: 5/5`

结论：

- `L10` 不支持“self-distilled anti-tie retention 可以作为当前 sim4 retention 问题的主修复方向”。
- 更具体地：
  - `lambda=0.05` 只带来了很轻微的 `final-best gap` 收窄：`-0.0328 -> -0.0295`
  - 但这个改善是以更低的 `best`、更低的 `final` 和更低的 `final signed margin median` 为代价
  - `lambda=0.1` 进一步证明单纯加大这条 loss 并不能解决问题；它让 `final-best gap` 回到比 control 更差的 `-0.0361`
- 因而 `L10` 的信号是：
  - “保住 earlier teacher margin” 这个想法方向上不完全错，因为它确实能让少数 seed 保住 `best=final`
  - 但当前实现没有把正确 pair 系统性地推出 `near-tie`
  - 它也没有改变导出层面的 `symmetric_collapse`
  - 所以它更像是对 late-stage drift 的弱约束，而不是能解释主问题的结构性修复

对后续实验的指向：

- 不应继续沿着“更大 self-distill 权重”这条线做简单加法。
- 现在更值得优先测试的是：
  - 直接改 direction parameterization / export 机制，减少 `support * gate` 的低-margin 压缩
  - 或者让 late-stage direction signal 不是只在“保留旧 margin”，而是能持续产生新的正确方向分离

证据：

- `GraphExp/results/unify_replay_sim4_20260415_195320_l10_sim4_control_selfdistill005.csv`
- `GraphExp/results/unify_replay_sim4_20260415_195320_l10_sim4_control_selfdistill005_aggregate.csv`
- `GraphExp/results/unify_replay_sim4_20260415_213418_l10_sim4_control_selfdistill010.csv`
- `GraphExp/results/unify_replay_sim4_20260415_213418_l10_sim4_control_selfdistill010_aggregate.csv`
- `GraphExp/results/unify_l10_sim4_selfdistill_compare_aggregate.csv`

### [L11] `sim4 control` direction export-only sweep

- 日期：`2026-04-16`
- 阶段：direction parameterization / export 机制定位实验 `L11`
- 目的：
  - 在不重训模型的前提下，直接测试当前 `support_direction` 的问题是否主要来自导出公式。
  - 核心问题：如果同一组 learned support / direction logits 换一种导出规则就能恢复 final F1，那么问题主要是 export compression；如果恢复不了，则说明 direction branch 的符号排序本身已经错了。
- 数据：
  - `sim4 control`
  - 使用已有 `support_direction_snapshots.npz`
  - seeds：`11,22,33,44,55`
- 导出变体：
  - `current_soft`：当前公式 `support_weights * sigmoid(direction_logits - direction_logits.T)`
  - `mask_soft`：去掉 learned support 幅值，只用 fixed support mask 与 soft direction gate
  - `mask_hard`：fixed support mask 内按 direction contrast 符号做 hard direction export
  - `mask_hard_top75/top50/top25`：只保留 direction contrast 绝对值最高的 confident pairs，其余 pair abstain
- 判定重点：
  - full-keep export 是否提高 `strict_f1@eps=0`
  - confidence-aware abstention 是否能提高 final F1
  - GT edge margin 被放大后，F1 是否同步上升

结果汇总，final anchor：

| export | keep_frac | strict F1 | vs current | pred edges | GT signed margin median |
|--------|----------:|----------:|-----------:|-----------:|------------------------:|
| `current_soft` | 1.00 | 0.8361 | 0.0000 | 61.0 | 0.0071 |
| `mask_soft` | 1.00 | 0.8361 | 0.0000 | 61.0 | 0.6503 |
| `mask_hard` | 1.00 | 0.8361 | 0.0000 | 61.0 | 1.0000 |
| `mask_hard_top75` | 0.75 | 0.7776 | -0.0585 | 46.0 | 1.0000 |
| `mask_hard_top50` | 0.50 | 0.6391 | -0.1969 | 31.0 | 0.0000 |
| `mask_hard_top25` | 0.25 | 0.4156 | -0.4205 | 16.0 | 0.0000 |

结果汇总，best anchor：

| export | keep_frac | strict F1 | vs current | pred edges | GT signed margin median |
|--------|----------:|----------:|-----------:|-----------:|------------------------:|
| `current_soft` | 1.00 | 0.8525 | 0.0000 | 61.0 | 0.0090 |
| `mask_soft` | 1.00 | 0.8525 | 0.0000 | 61.0 | 0.5152 |
| `mask_hard` | 1.00 | 0.8525 | 0.0000 | 61.0 | 1.0000 |
| `mask_hard_top75` | 0.75 | 0.7776 | -0.0749 | 46.0 | 1.0000 |
| `mask_hard_top50` | 0.50 | 0.6478 | -0.2046 | 31.0 | 0.0000 |
| `mask_hard_top25` | 0.25 | 0.4104 | -0.4421 | 16.0 | 0.0000 |

结论：

- `L11` 明确否定“只改 export 就能恢复 F1”的解释。
- full-keep export 的三个版本 `current_soft / mask_soft / mask_hard` 的 `strict F1` 完全一致：
  - final anchor 都是 `0.8361`
  - best anchor 都是 `0.8525`
- 但是 `mask_soft / mask_hard` 把 GT signed margin median 从 `0.0071` 放大到 `0.6503 / 1.0000`，F1 仍然不变。
- 这说明：
  - 当前 `symmetric_collapse` 标签里确实有很大一部分是“幅值/导出尺度”现象
  - 但 F1 gap 不是单纯的幅值问题
  - 因为单调 hardening 不改变 direction contrast 的符号，所以错误方向仍然错误
- confidence-aware abstention 也失败：
  - `top75/top50/top25` 都低于 current
  - 说明错误 pair 并没有稳定集中在低 confidence 区域；简单按 `|direction contrast|` 截断会伤 recall，不能作为导出修复
- 因而下一步必须转向 direction branch 的参数化 / 学习动态本身，而不是继续调导出后处理。

证据：

- `GraphExp/analyze_direction_export_variants.py`
- `GraphExp/results/unify_export_variants_sim4_20260416_143614_l11_sim4_control_export_sweep.csv`
- `GraphExp/results/unify_export_variants_sim4_20260416_143614_l11_sim4_control_export_sweep_aggregate.csv`

### [L12] `sim4 control` direct skew-matrix direction parameterization

- 日期：`2026-04-16`
- 阶段：direction parameterization / export 机制定位实验 `L12`
- 目的：
  - 测试当前 direction branch 的问题是否来自 `direction_emb_sender @ direction_emb_receiver.T` 再做 `D - D.T` 的冗余 factorized 参数化。
  - 新增 direct skew-matrix direction parameterization：直接学习一个 unconstrained matrix，并在读取时使用 `0.5 * (P - P.T)` 作为 direction logits，使 gate 接收到一个直接的 skew contrast。
- 改动：
  - 新增 CLI / config / replay 字段：
    - `direction_parameterization = factorized | skew_matrix`
  - 默认仍是 `factorized`，不改变历史配置行为。
  - `skew_matrix` 只在 `structure_parameterization = support_direction` 下有效。
- 数据集：
  - `sim4 control`
- 基线：
  - `run_20260404_111017`
- seeds：
  - `11,22,33,44,55`
- 不改：
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `direction_logit_bias_scale = 0.0`
  - `selection_score_mode = causal_lag_composite`

结果汇总：

| variant | best | exported | final | exported-best gap | final-best gap | best margin med | final margin med |
|---------|-----:|---------:|------:|------------------:|---------------:|----------------:|-----------------:|
| `factorized_control` | 0.8623 | 0.8098 | 0.8295 | -0.0525 | -0.0328 | 0.00854 | 0.00764 |
| `skew_matrix` | 0.7967 | 0.7836 | 0.7902 | -0.0131 | -0.0066 | 0.00576 | 0.00345 |

seed 级现象：

- `seed11`：
  - `best=final=0.7377`
  - 相比 control 的 `best=0.8852 / final=0.8689` 是大幅退化
- `seed33`：
  - formal run 中 `best=0.8197 / final=0.7869`
  - 相比 control 的 `best=0.8197 / final=0.7705`，retention 有改善，但幅度不足以抵消其他 seed 的损失
- `seed22/44/55`：
  - 都没有超过 control final

结论：

- `skew_matrix` 没有通过。
- 它确实把 `final-best gap` 从 `-0.0328` 收窄到 `-0.0066`，但这是通过压低 `best` 和 `final` 达成的，不是有效修复：
  - best 从 `0.8623` 掉到 `0.7967`
  - final 从 `0.8295` 掉到 `0.7902`
  - final margin median 也从 `0.00764` 掉到 `0.00345`
- 这说明：
  - 单纯去掉 factorized sender/receiver 冗余并不能解决方向学习
  - direct skew contrast 可能减少 late drift，但同时削弱了可达到的方向质量上限
  - 当前问题不是“参数化表达能力不够”这么简单，而更像是 direction signal 本身和训练路由仍然不足

对后续实验的指向：

- 不应直接采用 `skew_matrix` 替代现有 factorized direction branch。
- 下一步如果继续参数化方向，更合理的是测试“受约束的 residual parameterization”，而不是完全替代：
  - 保留 factorized direction branch 的可学习动态
  - 额外加入一个小幅 residual skew contrast 或温度/scale 控制
  - 目标是改善 `seed33` 这类 retention gap，同时避免 `seed11` 的方向质量上限崩掉

证据：

- `GraphExp/models/DDM.py`
- `GraphExp/main_structure_learning.py`
- `GraphExp/run_replay_saved_config.py`
- `GraphExp/results/unify_replay_sim4_20260416_151636_l12_sim4_control_skewparam.csv`
- `GraphExp/results/unify_replay_sim4_20260416_151636_l12_sim4_control_skewparam_aggregate.csv`
- `GraphExp/results/unify_l12_sim4_direction_param_compare_aggregate.csv`

### [2026-04-17] 方向学习机制综合结论（Codex + Claude + 代码复核）

结论升级：

- 当前 `support_direction` 主线的问题，不应再仅表述为宽泛的 `objective-architecture mismatch`，而应更精确地表述为：
  - **direction gate 被放进了主去噪消息图，承担了本不应由方向分支承担的“消息预算分配”功能。**
- 这意味着：
  - 参数上虽然已经做了 `support / direction` factorization；
  - 但功能上并没有 factorize；
  - 主去噪仍然直接消费 `support_weights * direction_gate`，因此 direction 仍被主任务功能性挟持。

代码层关键事实：

- `support_direction` 下，support logits 在读取时被对称化：
  - `0.5 * (adj_logits + adj_logits.T)`
- direction gate 由：
  - `sigmoid(direction_logits - direction_logits.T)`
  - 构成零和门控
- 因而对于任意 pair `(i, j)`：
  - `gate_ij + gate_ji = 1`
  - 若 `support_ij = support_ji = s`，则
    - `A_ij = s * gate_ij`
    - `A_ji = s * (1 - gate_ij)`
    - `A_ij + A_ji = s`
- 方向分支不创造预算，只在固定 support budget 内做再分配。

由此得到的机制结论：

- 对主去噪来说，direction 不是“额外信号”，而是“如何分配同一份 support 流量”。
- 在高相关时序数据上，如果两个节点都能从对方获得重建收益，则：
  - 去噪对 direction gate 的梯度天然接近抵消；
  - 且在边际收益递减的局部图景下，`gate ~= 0.5` 的均分解最稳定。
- 因此：
  - `margin` 薄不是简单因为方向信号弱；
  - 而是因为主去噪的稳定点天然靠近对称解。

`no-self-loop` 的机制意义需要单独强调：

- 当前 GraphConv 没有显式 self path / residual self message；
- temporal encoder 又是逐节点独立的，不能提供跨节点补偿；
- 因而 GraphConv 是唯一跨节点信息通道。
- 在稀疏图上，若某节点只有极少数 incoming parent，错误方向边就不再只是“有功能价值的错误边”，而会升级为：
  - **维持该节点在 GraphConv 中不断粮的生存通道。**
- 这解释了为什么在 `sim4` 这类低 parent-count 数据集上，反向边会被主去噪强烈保护。

对已有实验链的统一再解释：

- `L1`：
  - `ratio ~= 7.0` 更合理的解读不是“主路径不在乎方向”，而是：
  - **主路径对 support 很敏感，但对 direction 的敏感度被零和门控天然抑制。**
- `D2`：
  - 去掉 Patel direction 后崩溃，不是因为系统没有任何方向来源，而是因为默认稳定点本来就在近对称一侧。
- `L4 / L7`：
  - retention gap、near-tie、pair flip 不是偶然晚期漂移，而是长期被压在薄 margin 带上的结果。
- `L5 / L6 / L8`：
  - 调 late loss / freeze / switch timing 更多是在修 late-stage retention，不是在修主冲突机制本身。
- `L9`：
  - persistent bias 有帮助，但本质上只是给近零 margin 加偏置，不改变“方向参与主消息预算分配”这一根因。
- `L10`：
  - self-distill 失败，与其说是 teacher 不稳定，不如说是 teacher 本身就来自这个冲突系统里的脆弱平衡。
- `L11`：
  - export-only hardening 不能恢复 F1，说明问题不在导出公式本身，而在训练时 direction 已经被学成了错误或脆弱排序。
- `L12`：
  - 改 `skew_matrix` 只压低了 drift，却同时压低了 best/final 上限，说明问题不只是 direction 参数化冗余，而是 direction 在训练中承担了错误功能角色。

对 `warmup_then_orthogonal` 的更新理解：

- 切换前：
  - direction branch 在主去噪与方向监督的直接拉锯中形成脆弱均衡。
- 切换后：
  - `detach_direction_gate` 只切断梯度，并不改变消息图本身仍然是非对称的事实；
  - 节点仍然会因为小 gate 而在 GraphConv 中断粮；
  - 因而 late-stage causal-lag 接手时，面对的是已经被主去噪塑形成“重建友好 + 近对称”的方向表示。
- 这解释了为什么纯粹依赖 detach / late routing 并不能把 margin 推厚。

当前最核心的统一判断：

- **真正的问题不是“方向监督太弱”，而是“direction gate 被错误地放进了主去噪消息图”。**
- 只要主去噪继续直接吃 `support_weights * direction_gate`：
  - 方向学习就仍在逆风局；
  - retention 修补大概率仍只是 repair，而不是 mechanism repair。

### [Planned-2026-04-17] L13：`sim4 control` support-only message graph 诊断实验

- 目的：
  - 以最小代码改动，直接验证“主冲突是否来自 direction gate 参与主去噪消息传递”。
- 核心假设：
  - 若主去噪消息图改为只使用对称 `support_weights`，而 `direction_gate` 仅用于：
    - Patel direction margin supervision
    - causal-lag main
    - 导出最终有向图
  - 则：
    - 主去噪不再对 direction 施加预算分配压力；
    - 反向边不再被当作 GraphConv 生存通道；
    - `gate ~= 0.5` 的对称吸引子应明显减弱；
    - `sim4` 上的 margin 应增厚，`symmetric_collapse` 比例应下降。

最小实现改动：

- 在结构消息图读取路径中新增一个实验开关，例如：
  - `structure_message_edge_mode = full | support_only`
- 当前 control：
  - `full`
  - 消息图边权 = `support_weights * direction_gate`
- 诊断变体：
  - `support_only`
  - 消息图边权 = `support_weights`
  - 不乘 `direction_gate`
- 注意：
  - `direction_gate` 仍保留在方向辅助损失、causal-lag、导出路径中；
  - 这不是“去掉 direction branch”，而是“把 direction 从主去噪消息图里功能性剥离”。

实验配置：

- 数据集：
  - 首先只做 `sim4 control`
- seeds：
  - `11,22,33,44,55`
- 基线：
  - 当前 `sim4 control`
- 固定不改：
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `selection_score_mode = causal_lag_composite`
  - 其余与 control 保持一致

实验分组：

- `L13-A`：
  - 当前 control，作为对照
- `L13-B`：
  - `structure_message_edge_mode = support_only`

主要判定指标：

- direction 质量：
  - `best / exported / final primary_strict_f1`
  - `best / final signed_margin_median`
  - `adj_offdiag_cv`
  - `dir_active_signed_raw_frac_pos`
  - `dir_active_abs_margin_mean`
- retention：
  - `exported-best gap`
  - `final-best gap`
- failure taxonomy：
  - `symmetric_collapse` 占比是否下降
  - `recurrent pair flip` / near-tie 现象是否减少（若可沿用 L7 分析脚本）
- task tradeoff：
  - `final_diff_loss`
  - `causal_lag_diag_reverse_minus_forward`
  - selector 导出是否更接近 best epoch

支持该理论链的预期结果：

- 若 `L13-B` 相比 control 出现以下现象，则强支持当前机制解释：
  - `best / final margin median` 明显上升
  - `final-best gap` 收窄
  - `symmetric_collapse` 不再是绝对主导 failure mode
  - `strict_f1` 至少不下降，甚至上升
  - `L7` 式 near-tie pair 数量下降

可能的两类结果与解释：

- 结果 A：
  - margin 显著增厚，collapse 下降
  - 说明主问题的确是 direction 被主去噪消息图功能性挟持
- 结果 B：
  - worst-case 节点改善，但平均 margin 仍薄
  - 说明 `support-only` 切断了“生存依赖”问题，但零和门控/辅助信号强度不足仍然限制平均方向分离
  - 此时下一步更合理的是：
    - `support-only + explicit self residual`
    - 或增强 direction-only auxiliary chain

若 `L13` 阳性，后续建议：

- `L14`：
  - 在 `support-only` 消息图基础上，加显式 self path / residual
  - 用于拆分：
    - 零和门控问题
    - no-self-loop 生存依赖问题
- `L15`：
  - 将 `support-only` 分支迁移到 `sim3`
  - 验证该机制是否是 `sim4` 特有，还是整个 retention story 的共因

当前优先级判断：

- `L13` 应优先于继续堆叠新的 direction regularizer / self-distill / export-only patch。
- 原因是：
  - 它直接测试主因果链；
  - 代码改动小；
  - 证伪/证实价值都很高；
  - 若阴性，也能快速收缩后续假设空间。

### [L13] `sim4 control` support-only message graph 诊断实验

- 日期：`2026-04-17`
- 阶段：direction / message graph 机制定位实验 `L13`
- 目的：
  - 检查“direction gate 参与主去噪消息图”是否就是 `sim4` retention story 的主因。
  - 以最小改动验证：若主 GraphConv 只吃对称 `support_weights`，direction 是否会明显脱离 `gate ~= 0.5` 的近对称吸引子。
- 改动：
  - 在 `DDM` 中新增：
    - `structure_message_edge_mode = full | support_only`
  - `full`：
    - 保持当前行为，消息图边权 = `support_weights * direction_gate`
  - `support_only`：
    - 主消息图边权 = `support_weights`
    - `direction_gate` 仍仅用于：
      - directional supervision
      - causal-lag main
      - 最终导出
- 数据集：
  - `sim4 control`
- 基线：
  - `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich_aggregate.csv`
- seeds：
  - `11,22,33,44,55`
- 固定不改：
  - `structure_init_mode = random`
  - `structure_parameterization = support_direction`
  - `fixed_support_mask_mode = maxgap_kappa`
  - `direction_init_mode = random`
  - `direction_parameterization = factorized`
  - `gradient_routing_mode = warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch = 23`
  - `causal_lag_main_weight = 0.25`
  - `selection_score_mode = causal_lag_composite`
  - 其余保持 `sim4 control` replay 一致

结果汇总：

| variant | best | exported | final | exported-best gap | final-best gap | best margin med | exported margin med | final margin med |
|---------|-----:|---------:|------:|------------------:|---------------:|----------------:|--------------------:|-----------------:|
| `sim4_control` | 0.8623 | 0.8098 | 0.8295 | -0.0525 | -0.0328 | 0.00854 | 0.00934 | 0.00764 |
| `support_only_message_graph` | 0.8426 | 0.7967 | 0.8033 | -0.0459 | -0.0393 | 0.01048 | 0.00991 | 0.00571 |

seed 级现象：

- `seed11`：
  - `best=0.8197 / exported=0.7705 / final=0.8197`
  - final 回到 best，但 best 本身低于 control
- `seed22`：
  - `best=0.8033 / final=0.7869`
  - final 仍低于 best，且整体不超过 control
- `seed33`：
  - `best=0.8689` 仍然较高，但 `exported=final=0.8033`
  - 说明 selector / late-stage 仍然会把方向压回 near-tie 区
- `seed44`：
  - `best=0.8525 / final=0.7869`
  - `exported` 与 `seed33` 类似卡在 `0.8033`
- `seed55`：
  - `best=0.8689 / exported=0.8525 / final=0.8197`
  - 是这组里 retention 相对最好的一条，但整体均值仍未超过 control

failure mode：

- `best`：
  - `symmetric_collapse = 5/5`
- `final`：
  - `symmetric_collapse = 5/5`
- 也就是说：
  - `support_only` 并没有把失败类型从“对称塌缩”改写成别的主导模式

结论：

- `L13` 没有给出强阳性结果。
- 它只带来了一个有限变化：
  - `best margin median` 从 `0.00854` 升到 `0.01048`
  - `exported-best gap` 从 `-0.0525` 收窄到 `-0.0459`
- 但更关键的主指标没有改善，反而退化：
  - `best` 从 `0.8623` 降到 `0.8426`
  - `exported` 从 `0.8098` 降到 `0.7967`
  - `final` 从 `0.8295` 降到 `0.8033`
  - `final margin median` 从 `0.00764` 降到 `0.00571`
  - `final-best gap` 还从 `-0.0328` 变差到 `-0.0393`
- 更重要的是：
  - `symmetric_collapse` 仍然是 `5/5` 的统一 failure mode
  - 这意味着“把 direction 从主消息图里拿掉”并不足以单独修复 `sim4`

对机制解释的更新：

- `L13` 支持一个更细的判断：
  - direction 参与主消息图，确实不是一个理想设计；
  - 但它不是当前 `sim4` 崩塌现象的唯一主因。
- 更合理的解释变成：
  - `support_only` 切断了部分“反向边作为生存通道”的压力；
  - 所以 `best margin` 有一定增厚；
  - 但零和 gate、本身很薄的 direction auxiliary chain、以及无 self path 的信息瓶颈仍然存在；
  - 结果就是：
    - 平均 failure taxonomy 不变；
    - late-stage 仍然会回到 near-symmetric 区；
    - 同时还损失了原本 `full` 消息图带来的部分重建/结构协同收益。

对后续实验的指向：

- `L14` 应优先于直接推广 `support_only`。
- `L14` 更合理的形态是：
  - `support_only + explicit self path / residual`
  - 目标是把两个问题拆开：
    - `direction gate` 的零和预算分配问题
    - `no-self-loop` 导致的节点断粮问题
- 若 `L14` 仍然不能改变 `symmetric_collapse` 主导格局，则说明下一步不应再停留在消息图局部修补，而需要回到：
  - direction-only auxiliary strength
  - selector / export consistency
  - 或更彻底的 support/direction 功能分离设计

证据：

- `GraphExp/models/DDM.py`
- `GraphExp/main_structure_learning.py`
- `GraphExp/run_replay_saved_config.py`
- `GraphExp/results/unify_replay_sim4_20260417_203516_l13_support_only_sim4.csv`
- `GraphExp/results/unify_replay_sim4_20260417_203516_l13_support_only_sim4_aggregate.csv`
- `GraphExp/results/unify_phase0_control_sim4_20260410_003901_rich_aggregate.csv`
