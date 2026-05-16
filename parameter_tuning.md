# Parameter Tuning Log

更新日期：2026-05-11

## 1. 当前问题

目标不是继续追单个 seed 的最高分，而是在严格审计口径下提高多 seed 稳定性，尤其是让 `exported` 更接近 `final`。

当前重点问题：

- `exported` 由训练过程中的 selector 选出，不一定等于最后一个 epoch。
- `final_epoch_adjacency.csv/.npy` 已经保存，但默认导出的 `learned_adjacency.csv/.npy` 仍使用 selector 选中的 epoch。
- 因此“直接用 final”本质是一个 export policy / selector policy 实验，不是训练机制改动。

## 2. 本地代码确认

来源：

- `GraphExp/main_structure_learning.py`

当前导出逻辑：

- 优先使用 guarded best epoch。
- 如果没有 guarded epoch，则使用 score-only fallback。
- 如果完全没有候选，才 fallback 到 final epoch。
- `learned_adjacency.csv/.npy` 保存的是 selector 选中的 `adj_matrix`。
- `final_epoch_adjacency.csv/.npy` 和 `final_epoch_adjacency_causal.csv/.npy` 已经单独保存。

结论：

- 现有结果目录里已经有 final adjacency，可用于离线复核。
- 如果要让正式 `exported` 直接等于 final，需要新增或调整导出策略，而不是重训模型本身。

## 3. fMRI 当前最优 family 的证据

来源：

- `GraphExp/results/strict_family_audit_20260511_093104.csv`
- `GraphExp/results/strict_all_run_audit_20260511_093104.csv`

family：

- dataset: `fMRI`
- seeds: `11,22,33,44,55`
- runs:
  - `run_20260412_203901`
  - `run_20260412_204541`
  - `run_20260412_205206`
  - `run_20260412_205836`
  - `run_20260412_210457`
- config summary:
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

5 seed aggregate:

| metric | exported | final | best |
| --- | ---: | ---: | ---: |
| primary strict F1 | 0.8400 | 0.9200 | 0.9200 |
| strict F1 @ eps=0.1 | 0.2476 | 0.3976 | 0.2167 |

per-seed primary strict F1:

| seed | best epoch | exported epoch | final epoch | best | exported | final |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 97 | 13 | 100 | 1.0 | 1.0 | 1.0 |
| 22 | 69 | 22 | 100 | 1.0 | 0.8 | 1.0 |
| 33 | 94 | 23 | 100 | 0.8 | 0.6 | 0.8 |
| 44 | 92 | 24 | 100 | 1.0 | 1.0 | 1.0 |
| 55 | 78 | 77 | 100 | 0.8 | 0.8 | 0.8 |

读数：

- 在该 fMRI family 中，final 对每个 seed 都不低于 exported。
- final 的均值等于 best 均值，且 `final_vs_best_gap_mean = 0.0`。
- exported 的均值低于 final，主要是 seed 22 和 seed 33 被较早 epoch 拉低。
- 因此对该 fMRI family，`export_policy=final` 是有证据支持的候选。

## 4. 不能直接全局默认的原因

严格审计也显示其他数据集存在 final 低于 best 的情况：

- `sim2` 当前最优 family:
  - best mean = 0.8727
  - exported mean = 0.8000
  - final mean = 0.8545
- `sim3` 当前最优 family:
  - best mean = 0.9222
  - exported mean = 0.8444
  - final mean = 0.9111
- `sim4` 当前最优 family:
  - best mean = 0.8820
  - exported mean = 0.8492
  - final mean = 0.8492

结论：

- final-export 对 fMRI 很可能是正收益。
- 对 sim2/sim3，final 比 exported 高，但仍低于 best，需要看是否接受“稳定保守导出”。
- 对 sim4，当前最优 family 中 exported 已经等于 final，直接用 final 没有额外收益。
- 因此不要把 final 直接设为所有数据集的无条件默认。

## 5. 建议的下一步

优先做 selector/export policy 分支，而不是改训练损失：

1. 离线复核：
   - 对已有 run 直接比较 `learned_adjacency_causal` 与 `final_epoch_adjacency_causal`。
   - 输出每个 dataset/family 的 `exported -> final` delta。

2. 代码参数化：
   - 新增类似 `--export_epoch_policy {selector,final}` 的参数。
   - 默认保持 `selector`，避免破坏旧实验语义。
   - fMRI 调参分支显式使用 `--export_epoch_policy final`。

3. 正式复跑：
   - 先只跑 fMRI 当前最优 family 的 5 seed。
   - 如果 `exported=final` 后均值稳定到 0.92，再记录为 fMRI 推荐 export policy。

## 6. 当前判断

对用户问题“能不能直接用 final”：

- 对 fMRI 当前严格多 seed 最优 family：可以，且应该优先验证。
- 对全项目默认策略：暂不建议。
- 工程上最稳妥的实现方式是新增 export policy 参数，而不是删除现有 selector。

## 7. 实现记录

2026-05-11 已实现：

- `GraphExp/main_structure_learning.py`
  - 新增 `--export_epoch_policy {selector,final}`。
  - 默认值为 `selector`，保持旧实验语义。
  - `selector`：`learned_adjacency.csv/.npy` 仍写 selector 选中的 epoch。
  - `final`：`learned_adjacency.csv/.npy` 改写 final epoch adjacency。
  - 无论策略如何，都会额外保存：
    - `selector_epoch_adjacency.csv/.npy`
    - `selector_epoch_adjacency_causal.csv/.npy`
    - `final_epoch_adjacency.csv/.npy`
    - `final_epoch_adjacency_causal.csv/.npy`
  - `config.npy` 记录：
    - `export_epoch_policy`
    - `selector_epoch`
    - `exported_epoch`
    - `selector_epoch_selection_mode`
    - `exported_epoch_selection_mode`

- `GraphExp/run_replay_saved_config.py`
  - 增加 `export_epoch_policy` 的默认值和透传。
  - 汇总结果增加 `selector_epoch` 和 `export_epoch_policy`。

验证：

- 已运行：
  - `python -m py_compile GraphExp/main_structure_learning.py GraphExp/run_replay_saved_config.py`
- 结果：
  - 通过。

## 8. 下一轮 fMRI 参数筛选计划

目标：

- 基准使用当前严格多 seed 最优 fMRI family。
- 固定 `export_epoch_policy=final`，避免 selector 误选早期 epoch 干扰训练参数判断。
- 每次只改一个训练参数。
- 先做 2 seed 筛选，看到正信号后再扩到 5 seed。

固定基准：

- base run: `GraphExp/results/run_20260412_203901`
- seeds: `11,22,33,44,55`
- 核心配置：
  - `structure_init_mode=random`
  - `support_prior_mode=patel_kappa`
  - `fixed_support_mask_mode=maxgap_kappa`
  - `direction_init_mode=random`
  - `gradient_routing_mode=warmup_then_orthogonal`
  - `detach_direction_from_main_after_epoch=23`
  - `causal_lag_main_weight=0.25`
  - `directional_kappa_gate=True`
  - `selection_score_mode=causal_lag_composite`
  - `epochs=100`
  - `top_k_edges=5`

筛选候选：

| ID | 单一改动 | 理由 | 筛选 seeds |
| --- | --- | --- | --- |
| `F0` | `export_epoch_policy=final` | 验证新 export policy，不改训练 | `11,22` |
| `F1` | `freeze_direction_after_epoch=15` + `export_epoch_policy=final` | fMRI 单 seed F1=1 路径里 freeze 有过强线索；该参数只影响方向分支后期漂移 | `11,22` |
| `F2` | `causal_lag_main_weight=0.15` + `export_epoch_policy=final` | 检查当前 `0.25` 是否过强，可能压制 final margin | `11,22` |
| `F3` | `causal_lag_main_weight=0.35` + `export_epoch_policy=final` | 检查更强 causal-lag 是否提高 final margin | `11,22` |

暂不优先动：

- `fixed_support_mask_mode`
  - 历史 ablation 显示去掉后 support selectivity 崩掉。
- `pretrain`
  - sim3 历史记录显示关 pretrain 明显伤害。
- `disable_directional_loss`
  - Patel-free direction learning 还没有解决。
- `structure_init_mode`
  - 当前 fMRI 最优 family 已经是 `random`。
- 大范围 selector 权重
  - 本轮固定 `export_epoch_policy=final` 后，selector 不再是主要变量。

筛选提升标准：

- 2 seed 筛选阶段：
  - `final/exported primary strict F1` 不能低于 F0。
  - `strict F1 @ eps=0.1` 或 signed margin 有改善优先。
- 5 seed 确认阶段：
  - 相对当前严格 family，`final_primary_strict_f1_mean` 目标超过 `0.92`。
  - 如果主 F1 持平，则要求 `final_strict_f1_eps_0p1_mean` 高于 `0.3976`。

## 9. fMRI 2-seed 筛选结果

执行时间：

- 2026-05-11

共同设置：

- base run: `GraphExp/results/run_20260412_203901`
- seeds: `11,22`
- 均使用 `export_epoch_policy=final`

结果文件：

- `F0`: `GraphExp/results/unify_replay_fMRI_20260511_113929_fmri_F0_final_export_probe.csv`
- `F1`: `GraphExp/results/unify_replay_fMRI_20260511_115553_fmri_F1_freeze15_final_export_probe.csv`
- `F2`: `GraphExp/results/unify_replay_fMRI_20260511_121255_fmri_F2_lag015_final_export_probe.csv`
- `F3`: `GraphExp/results/unify_replay_fMRI_20260511_123003_fmri_F3_lag035_final_export_probe.csv`

aggregate:

| ID | override | exported/final primary strict F1 mean | final strict F1 @ eps=0.1 mean | final signed margin median mean | read |
| --- | --- | ---: | ---: | ---: | --- |
| `F0` | `export_epoch_policy=final` | 1.0000 | 0.4524 | 0.0736 | baseline policy 正常 |
| `F1` | `freeze_direction_after_epoch=15` | 0.8000 | 0.0000 | 0.0240 | 淘汰；seed 22 掉到 0.6 |
| `F2` | `causal_lag_main_weight=0.15` | 1.0000 | 0.1667 | 0.0677 | 主 F1 持平，但 margin 指标弱于 F0 |
| `F3` | `causal_lag_main_weight=0.35` | 1.0000 | 0.5714 | 0.0681 | 主 F1 持平，eps=0.1 最好 |

读数：

- `export_epoch_policy=final` 已经按预期工作：`exported_epoch=100`，`selector_epoch` 仍记录原 selector 选中轮次。
- `freeze_direction_after_epoch=15` 对当前 fMRI family 是负向，不推进。
- 降低 causal-lag 权重到 `0.15` 没有收益。
- 提高 causal-lag 权重到 `0.35` 是当前唯一值得扩到 5 seed 的候选。

下一步：

- 扩跑 `F3` 到 seeds `11,22,33,44,55`。
- 如果 5 seed `final_primary_strict_f1_mean > 0.92`，则记为主指标提升。
- 如果主 F1 仍为 `0.92`，但 `final_strict_f1_eps_0p1_mean > 0.3976`，则记为 margin 稳定性提升。

## 10. fMRI F3 5-seed 严格确认

执行时间：

- 2026-05-11

配置：

- base run: `GraphExp/results/run_20260412_203901`
- seeds: `11,22,33,44,55`
- overrides:
  - `export_epoch_policy=final`
  - `causal_lag_main_weight=0.35`

结果文件：

- `GraphExp/results/unify_replay_fMRI_20260511_124620_fmri_F3_lag035_final_export_5seed.csv`
- `GraphExp/results/unify_replay_fMRI_20260511_124620_fmri_F3_lag035_final_export_5seed_aggregate.csv`

逐 seed：

| seed | selector epoch | exported epoch | best | exported | final | final strict F1 @ eps=0.1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 13 | 100 | 1.0 | 1.0 | 1.0 | 0.5714 |
| 22 | 23 | 100 | 1.0 | 1.0 | 1.0 | 0.5714 |
| 33 | 24 | 100 | 1.0 | 1.0 | 1.0 | 0.0000 |
| 44 | 24 | 100 | 1.0 | 1.0 | 1.0 | 0.7500 |
| 55 | 65 | 100 | 0.8 | 0.8 | 0.8 | 0.3333 |

aggregate:

| metric | previous strict family | F3 5-seed |
| --- | ---: | ---: |
| best primary strict F1 mean | 0.9200 | 0.9600 |
| exported primary strict F1 mean | 0.8400 | 0.9600 |
| final primary strict F1 mean | 0.9200 | 0.9600 |
| final strict F1 @ eps=0.1 mean | 0.3976 | 0.4452 |
| final signed margin median mean | 0.0797 | 0.0765 |
| final failure mode counts | `{'mixed_or_partial': 5}` | `{'mixed_or_partial': 5}` |

读数：

- `F3` 是当前 fMRI 严格 5-seed 下的新最好候选。
- 主指标从 `0.92` 提升到 `0.96`。
- `exported` 通过 `export_epoch_policy=final` 同步提升到 `0.96`。
- 高 margin 指标 `final_strict_f1@eps=0.1` 也高于旧 family。
- signed margin median 均值略低于旧 family，因此不能说方向 margin 全面变强；更准确的结论是：主 strict F1 和 eps=0.1 指标改善，但 margin 中位数未同步改善。
- seed 33 虽然 primary strict F1 为 `1.0`，但 eps=0.1 为 `0.0`，说明仍有低 margin 解。

当前 fMRI 推荐：

- `export_epoch_policy=final`
- `causal_lag_main_weight=0.35`
- 其余沿用 `run_20260412_203901` family。

## 11. 新增 GT 数据集重新预训练 replay

执行时间：

- 2026-05-12

目的：

- 对新增且有 ground truth 的 `sim8/sim10/sim11/sim12` 做严格 5-seed 跑法。
- 不复用旧 encoder checkpoint；通过 `pretrain_checkpoint=` 触发每个数据集自己的 `pretrain_epochs=50` 自回归预训练。
- 使用 `export_epoch_policy=final`，使 `learned_adjacency*.csv` 对应 final epoch。

配置：

| dataset | base run | csv | gt | top_k_edges / selection_top_k | seeds |
| --- | --- | --- | --- | ---: | --- |
| `sim8` | `GraphExp/results/run_20260511_124622` | `../fMRI_dataset/sim8.csv` | `../fMRI_dataset/h8.txt` | 5 | `11,22,33,44,55` |
| `sim10` | `GraphExp/results/run_20260511_124622` | `../fMRI_dataset/sim10.csv` | `../fMRI_dataset/h10.txt` | 5 | `11,22,33,44,55` |
| `sim11` | `GraphExp/results/run_20260420_090231` | `../fMRI_dataset/sim11.csv` | `../fMRI_dataset/h11.txt` | 11 | `11,22,33,44,55` |
| `sim12` | `GraphExp/results/run_20260420_090231` | `../fMRI_dataset/sim12.csv` | `../fMRI_dataset/h12.txt` | 11 | `11,22,33,44,55` |

结果文件：

- `GraphExp/results/unify_replay_sim8_20260512_091903_sim8_gt_5seed_repretrain.csv`
- `GraphExp/results/unify_replay_sim8_20260512_091903_sim8_gt_5seed_repretrain_aggregate.csv`
- `GraphExp/results/unify_replay_sim10_20260512_095726_sim10_gt_5seed_repretrain.csv`
- `GraphExp/results/unify_replay_sim10_20260512_095726_sim10_gt_5seed_repretrain_aggregate.csv`
- `GraphExp/results/unify_replay_sim11_20260512_103852_sim11_gt_5seed_repretrain.csv`
- `GraphExp/results/unify_replay_sim11_20260512_103852_sim11_gt_5seed_repretrain_aggregate.csv`
- `GraphExp/results/unify_replay_sim12_20260512_111301_sim12_gt_5seed_repretrain.csv`
- `GraphExp/results/unify_replay_sim12_20260512_111301_sim12_gt_5seed_repretrain_aggregate.csv`

aggregate：

| dataset | best F1 mean | exported F1 mean | final F1 mean | final eps=0.1 F1 mean | final signed margin median mean | final failure modes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sim8` | 0.8800 | 0.8400 | 0.8400 | 0.1500 | 0.0790 | `{'mixed_or_partial': 5}` |
| `sim10` | 0.8400 | 0.8000 | 0.8000 | 0.1333 | 0.0665 | `{'mixed_or_partial': 5}` |
| `sim11` | 0.0667 | 0.0333 | 0.0333 | 0.0000 | 0.0000 | `{'symmetric_collapse': 5}` |
| `sim12` | 0.7818 | 0.7273 | 0.7273 | 0.2056 | 0.0340 | `{'mixed_or_partial': 2, 'weak_asymmetry': 3}` |

逐 seed 主指标：

| dataset | seed 11 | seed 22 | seed 33 | seed 44 | seed 55 | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sim8` final | 0.8000 | 0.8000 | 0.8000 | 1.0000 | 0.8000 | seed 22 best 到 1.0 但 final 为 0.8 |
| `sim10` final | 0.8000 | 0.8000 | 0.8000 | 0.8000 | 0.8000 | seed 44 best 到 1.0 但 final 为 0.8 |
| `sim11` final | 0.0000 | 0.0000 | 0.0000 | 0.1667 | 0.0000 | 全部 symmetric collapse |
| `sim12` final | 0.8182 | 0.6364 | 0.6364 | 0.7273 | 0.8182 | final 低于 best，存在后期退化 |

重新预训练确认：

- 四组代表 run 的 `config.npy` 均为 `pretrain_checkpoint=None`、`pretrain_epochs=50`。
- 代表 run 日志均出现 `开始时间因果编码器的自回归预训练 (50 Epochs)`，没有出现加载旧 checkpoint 的路径。

读数：

- `sim8` 和 `sim10` 在 5 节点配置上可用，但 final/exported 明显低于 fMRI F3；主要问题是高 margin 指标弱。
- `sim11` 不能接受当前 sim2 参数迁移，失败模式是稳定的 `symmetric_collapse`，优先调方向/支撑初始化或 selector/causal-lag 配置，而不是继续加 seed。
- `sim12` 可用但 final 低于 best，说明后期训练存在退化；如果后续优化，应优先比较 `export_epoch_policy=selector` 或调整 final 稳定性，而不是直接认为 final 更好。

## 12. sim11 支撑修复：`topk_kappa` 替代 `maxgap_kappa`

执行时间：

- 2026-05-12

背景：

- `sim11` 与 `sim12` 使用相同 GT，但 `sim11` 在原始 sim2 family 配置下 5/5 seed 都是 `symmetric_collapse`。
- 代码与日志确认，失败的直接原因不是 encoder，而是：
  - `fixed_support_mask_mode=maxgap_kappa`
  - `support_prior_mode=pearson_abs`
  - `sim11` 的 `pearson_abs` 最大断点发生在第 1 条和第 2 条 pair 之间
  - 最终 `Noise guide adj: 1 undirected pairs`
  - `Fixed support mask: ... undirected_pairs=1`
- 这会把可学习支撑空间压缩到几乎空集，导致大多数 GT 边的 `support=0`，严格方向 F1 接近 0。

修复思路：

- 保留 sim2 family 其余参数不变。
- 只修改 fixed support 选择方式，强制给模型足够的候选支撑边：
  - `fixed_support_mask_mode=topk_kappa`
  - `top_k_edges=16`
  - `selection_top_k=11`
- 这里 `top_k_edges=16` 的依据是：`sim11` 的 `pearson_abs` 前 16 个无向 pair 已覆盖全部 11 条 GT 边；前 11 个 pair 只覆盖 6 条 GT 边。

配置：

- base run: `GraphExp/results/run_20260420_090231`
- seeds: `11,22,33,44,55`
- overrides:
  - `csv_path=../fMRI_dataset/sim11.csv`
  - `selector_audit_gt_path=../fMRI_dataset/h11.txt`
  - `fixed_support_mask_mode=topk_kappa`
  - `top_k_edges=16`
  - `selection_top_k=11`
  - `pretrain_checkpoint=`
  - `pretrain_epochs=50`
  - `export_epoch_policy=final`

结果文件：

- `GraphExp/results/unify_replay_sim11_20260512_142013_sim11_topk16_repretrain.csv`
- `GraphExp/results/unify_replay_sim11_20260512_142013_sim11_topk16_repretrain_aggregate.csv`

逐 seed：

| seed | best | exported | final | best epoch | exported/final epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.6667 | 0.5926 | 0.5926 | 18 | 40 |
| 22 | 0.5926 | 0.5926 | 0.5926 | 36 | 40 |
| 33 | 0.5926 | 0.5185 | 0.5185 | 18 | 40 |
| 44 | 0.7407 | 0.5926 | 0.5926 | 17 | 40 |
| 55 | 0.6667 | 0.5926 | 0.5926 | 23 | 40 |

aggregate：

| metric | 原始 sim11 5-seed | `topk16` 修复后 |
| --- | ---: | ---: |
| best primary strict F1 mean | 0.0667 | 0.6519 |
| exported primary strict F1 mean | 0.0333 | 0.5778 |
| final primary strict F1 mean | 0.0333 | 0.5778 |
| best strict F1 @ eps=0.1 mean | 0.0000 | 0.3466 |
| final strict F1 @ eps=0.1 mean | 0.0000 | 0.0286 |
| best signed margin median mean | 0.0000 | 0.0668 |
| final signed margin median mean | 0.0000 | 0.0435 |
| final failure modes | `{'symmetric_collapse': 5}` | `{'mixed_or_partial': 5}` |

日志确认：

- 5/5 seed 均为：
  - `Noise guide adj: 16 undirected pairs + 10 self-loops`
  - `Fixed support mask: top-k pearson_abs | undirected_pairs=16`
  - `开始时间因果编码器的自回归预训练 (50 Epochs)`

读数：

- 修复是有效且稳定的。`sim11` 的主问题确实是 fixed support mask 过窄，不是预训练失败。
- 但当前配置仍有明显缺口：
  - final/exported 只有 `0.5778`
  - best 到 final 平均掉 `0.0741`
  - `final strict F1 @ eps=0.1` 仍接近 0，说明 final 方向 margin 很弱
- 因此当前 `topk16` 应被视为“解除支撑坍缩”的中间修复，而不是 `sim11` 的最终最优配置。

下一步优先级：

- 对 `sim11` 比较 `export_epoch_policy=selector`，因为当前 best 明显好于 final。
- 如果要继续优化 final，再试：
  - 增大 `causal_lag_main_weight`
  - 调整 `selection_score_mode` / selector 权重
  - 比较 `support_prior_mode=patel_kappa` 与 `pearson_abs`

## 13. sim11 `topk16`：`selector` 导出策略对照

执行时间：

- 2026-05-12

目的：

- 在 `sim11 topk16` 修复配置下，只切换 `export_epoch_policy=selector`，验证当前差距是否主要来自 final 导出策略。

配置：

- base run: `GraphExp/results/run_20260420_090231`
- seeds: `11,22,33,44,55`
- overrides:
  - `csv_path=../fMRI_dataset/sim11.csv`
  - `selector_audit_gt_path=../fMRI_dataset/h11.txt`
  - `fixed_support_mask_mode=topk_kappa`
  - `top_k_edges=16`
  - `selection_top_k=11`
  - `pretrain_checkpoint=`
  - `pretrain_epochs=50`
  - `export_epoch_policy=selector`

结果文件：

- `GraphExp/results/unify_replay_sim11_20260512_153711_sim11_topk16_selector_export.csv`
- `GraphExp/results/unify_replay_sim11_20260512_153711_sim11_topk16_selector_export_aggregate.csv`

aggregate：

| metric | `topk16 + final export` | `topk16 + selector export` |
| --- | ---: | ---: |
| best primary strict F1 mean | 0.6519 | 0.6519 |
| exported primary strict F1 mean | 0.5778 | 0.5185 |
| final primary strict F1 mean | 0.5778 | 0.5778 |
| best strict F1 @ eps=0.1 mean | 0.3466 | 0.3466 |
| exported strict F1 @ eps=0.1 mean | 0.0286 | 0.3467 |
| final strict F1 @ eps=0.1 mean | 0.0286 | 0.0286 |
| exported signed margin median mean | 0.0435 | 0.0605 |
| exported vs best gap mean | -0.0741 | -0.1333 |

逐 seed：

| seed | best epoch / F1 | selector epoch / exported F1 | final epoch / final F1 |
| ---: | --- | --- | --- |
| 11 | `18 / 0.6667` | `11 / 0.5185` | `40 / 0.5926` |
| 22 | `36 / 0.5926` | `13 / 0.4444` | `40 / 0.5926` |
| 33 | `18 / 0.5926` | `11 / 0.5185` | `40 / 0.5185` |
| 44 | `17 / 0.7407` | `13 / 0.5926` | `40 / 0.5926` |
| 55 | `23 / 0.6667` | `13 / 0.5185` | `40 / 0.5926` |

日志读数：

- 5/5 seed 的 selector 都选到了较早的 guarded epoch（11 或 13）。
- 这些 epoch 的 exported margin 更强，但主 strict F1 反而低于 final。
- GT 最优 epoch 明显更晚（17/18/23/36），说明当前 selector proxy 在 `sim11` 上更偏向 margin/guard 条件，而没有选到 GT 最优轮次。

结论：

- `export_epoch_policy=selector` 不是 `sim11 topk16` 的修复方向。
- 对 `sim11` 而言，当前主要问题已经不是“final 导出拖坏结果”，而是 selector 本身没有对齐 GT 最优 epoch。
- 如果继续做 `sim11`，下一步优先应该是：
  - 调 `selection_score_mode` 或其权重
  - 或直接调训练动力学（如 `causal_lag_main_weight`），把 GT 最优轮次往后推近 final

## 14. sim11 训练动力学尝试：降低 L1 与延后正交切换

执行时间：

- 2026-05-12

目的：

- 沿“把 GT 最优 epoch 往后推，尽量靠近 final”这条路线做小规模筛选。
- 先试两个最直接影响后期退化的动力学参数：
  - `D1`: 降低 `lambda_l1`
  - `D2`: 推迟 `detach_direction_from_main_after_epoch`

背景读数：

- `sim11 topk16` 基线里，best 通常出现在 `15-23` epoch，之后 final 会退化。
- 同时 GT support median 会从约 `0.15` 掉到约 `0.09`，说明后期持续稀疏化。
- `warmup_then_orthogonal` 在 `epoch > 23` 后会让 causal-lag 只更新 direction，不再帮助 support。

### 14.1 D1: `lambda_l1=0.01`

2-seed probe：

- 文件：
  - `GraphExp/results/unify_replay_sim11_20260512_162628_sim11_D1_l1_001_probe.csv`
  - `GraphExp/results/unify_replay_sim11_20260512_162628_sim11_D1_l1_001_probe_aggregate.csv`
- 读数：
  - 主 final F1 与基线持平
  - 但 2-seed 下 `final_vs_best_gap_mean` 从 `-0.0741` 缩到 `-0.0370`
  - `final strict F1 @ eps=0.1` 有改善

因此扩到 5-seed：

- 文件：
  - `GraphExp/results/unify_replay_sim11_20260512_165805_sim11_D1_l1_001_5seed.csv`
  - `GraphExp/results/unify_replay_sim11_20260512_165805_sim11_D1_l1_001_5seed_aggregate.csv`

5-seed 对比：

| metric | `topk16` 基线 | `lambda_l1=0.01` |
| --- | ---: | ---: |
| best primary strict F1 mean | 0.6519 | 0.6519 |
| final primary strict F1 mean | 0.5778 | 0.5778 |
| final strict F1 @ eps=0.1 mean | 0.0286 | 0.1108 |
| final signed margin median mean | 0.0435 | 0.0476 |
| final support median mean | 0.0810 | 0.0847 |
| final vs best gap mean | -0.0741 | -0.0741 |

读数：

- 降低 L1 没有把 GT 最优 epoch 往后推，也没有提升主 final F1。
- 但它确实缓解了后期 margin / support 的塌缩，属于弱正向。
- 这更像“保住一点 final 质量”，不是主突破。

### 14.2 D2: `detach_direction_from_main_after_epoch=31`

2-seed probe：

- 文件：
  - `GraphExp/results/unify_replay_sim11_20260512_164039_sim11_D2_detach31_probe.csv`
  - `GraphExp/results/unify_replay_sim11_20260512_164039_sim11_D2_detach31_probe_aggregate.csv`

2-seed 读数：

| metric | `topk16` 基线 2-seed | `detach=31` |
| --- | ---: | ---: |
| final primary strict F1 mean | 0.5926 | 0.5926 |
| best strict F1 @ eps=0.1 mean | 0.2857 | 0.4732 |
| final strict F1 @ eps=0.1 mean | 0.2000 | 0.0000 |
| final vs best gap mean | -0.0370 | -0.0370 |

读数：

- 推迟正交切换没有改善 final 主指标。
- 更糟的是 final 的高-margin strict F1 反而掉到 0。
- 这条路不值得扩到 5-seed。

结论：

- 在当前 `sim11 topk16` 上，单独改 `lambda_l1` 或 `detach_direction_from_main_after_epoch` 都不能把 GT 最优 epoch 实质性推近 final。
- `lambda_l1=0.01` 只能算弱正向辅助项：主 F1 不变，但 final margin/support 略有改善。
- 如果继续走“训练动力学”路线，下一优先级应该转到 `causal_lag_main_weight`，因为前两类修改都没有改变 best/final 的主指标关系。

## 15. sim11 训练动力学：提高 `causal_lag_main_weight`

执行时间：

- 2026-05-12

目的：

- 继续沿“把 GT 最优 epoch 往后推近 final”这条路线，测试更强的 causal-lag 主约束是否能减少后期退化。

测试组：

- `D3`: `causal_lag_main_weight=0.35`
- `D4`: `causal_lag_main_weight=0.35 + lambda_l1=0.01`

### 15.1 D3 2-seed probe

文件：

- `GraphExp/results/unify_replay_sim11_20260512_180852_sim11_D3_lag035_probe.csv`
- `GraphExp/results/unify_replay_sim11_20260512_180852_sim11_D3_lag035_probe_aggregate.csv`

2-seed 结果：

| metric | `topk16` 基线 2-seed | `D3` |
| --- | ---: | ---: |
| final primary strict F1 mean | 0.5926 | 0.6296 |
| final vs best gap mean | -0.0370 | 0.0000 |
| final strict F1 @ eps=0.1 mean | 0.2000 | 0.1875 |

读数：

- seed 11 达到 `best = exported = final = 0.6667`
- seed 22 达到 `best = exported = final = 0.5926`
- 这是第一组明确把 `best/final` 对齐的动力学改动，值得扩到 5-seed

### 15.2 D4 2-seed probe

文件：

- `GraphExp/results/unify_replay_sim11_20260512_182754_sim11_D4_lag035_l1001_probe.csv`
- `GraphExp/results/unify_replay_sim11_20260512_182754_sim11_D4_lag035_l1001_probe_aggregate.csv`

2-seed 结果：

| metric | `D3` | `D4` |
| --- | ---: | ---: |
| final primary strict F1 mean | 0.6296 | 0.5926 |
| final vs best gap mean | 0.0000 | -0.0370 |
| final strict F1 @ eps=0.1 mean | 0.1875 | 0.2353 |

读数：

- `D4` 没有保住 `D3` 的 best/final 对齐
- 虽然 `eps=0.1` 更高，但主 final F1 更差
- 因此优先扩的是 `D3`，不是 `D4`

### 15.3 D3 5-seed 严格确认

文件：

- `GraphExp/results/unify_replay_sim11_20260512_184811_sim11_D3_lag035_5seed.csv`
- `GraphExp/results/unify_replay_sim11_20260512_184811_sim11_D3_lag035_5seed_aggregate.csv`

逐 seed：

| seed | best | exported | final | best epoch |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 0.6667 | 0.6667 | 0.6667 | 19 |
| 22 | 0.5926 | 0.5926 | 0.5926 | 34 |
| 33 | 0.5185 | 0.5185 | 0.5185 | 12 |
| 44 | 0.7407 | 0.5926 | 0.5926 | 18 |
| 55 | 0.6667 | 0.5926 | 0.5926 | 22 |

与 `sim11 topk16` 基线对比：

| metric | `topk16` 基线 | `D3 lag0.35` |
| --- | ---: | ---: |
| best primary strict F1 mean | 0.6519 | 0.6370 |
| exported primary strict F1 mean | 0.5778 | 0.5926 |
| final primary strict F1 mean | 0.5778 | 0.5926 |
| best strict F1 @ eps=0.1 mean | 0.3466 | 0.3758 |
| final strict F1 @ eps=0.1 mean | 0.0286 | 0.0750 |
| best signed margin median mean | 0.0668 | 0.0779 |
| final signed margin median mean | 0.0435 | 0.0487 |
| best support median mean | 0.1238 | 0.1251 |
| final support median mean | 0.0810 | 0.0852 |
| final vs best gap mean | -0.0741 | -0.0444 |

结论：

- `causal_lag_main_weight=0.35` 是目前 `sim11` 最有效的训练动力学改动。
- 它没有把平均 best 提高，但：
  - final/exported 主 F1 从 `0.5778` 提到 `0.5926`
  - final vs best gap 从 `-0.0741` 缩到 `-0.0444`
  - final 的高-margin strict F1、signed margin、support median 都有改善
- 这说明更强的 causal-lag 主约束确实在一定程度上把“好解”往后推近了 final。

当前 `sim11` 推荐顺序：

1. `topk16` 修复 support collapse
2. 在此基础上将 `causal_lag_main_weight` 提到 `0.35`
3. 若继续优化，可再比较：
  - `lag0.35 + lambda_l1=0.01`
  - 更细的 `lag` 扫描，如 `0.30 / 0.40`

## 16. sim11 `causal_lag_main_weight` 细扫：0.30 / 0.40

执行时间：

- 2026-05-12

目的：

- 在 `sim11 topk16 + final export + 重新预训练` 基础上，判断 `lag=0.35` 是否接近局部最佳，还是还值得继续往 `0.30` 或 `0.40` 移动。

测试组：

- `D5`: `causal_lag_main_weight=0.30`
- `D6`: `causal_lag_main_weight=0.40`

结果文件：

- `GraphExp/results/unify_replay_sim11_20260512_200934_sim11_D5_lag030_probe.csv`
- `GraphExp/results/unify_replay_sim11_20260512_200934_sim11_D5_lag030_probe_aggregate.csv`
- `GraphExp/results/unify_replay_sim11_20260512_202807_sim11_D6_lag040_probe.csv`
- `GraphExp/results/unify_replay_sim11_20260512_202807_sim11_D6_lag040_probe_aggregate.csv`

2-seed 对比：

| metric | `lag=0.30` | `lag=0.35` | `lag=0.40` |
| --- | ---: | ---: | ---: |
| final primary strict F1 mean | 0.5926 | 0.6296 | 0.6296 |
| final vs best gap mean | -0.0370 | 0.0000 | 0.0000 |
| best strict F1 @ eps=0.1 mean | 0.3000 | 0.3500 | 0.2857 |
| final strict F1 @ eps=0.1 mean | 0.2000 | 0.1875 | 0.1875 |
| best signed margin median mean | 0.0778 | 0.0832 | 0.0825 |
| final signed margin median mean | 0.0528 | 0.0567 | 0.0564 |
| best support median mean | 0.1180 | 0.1136 | 0.1106 |
| final support median mean | 0.0791 | 0.0838 | 0.0794 |

读数：

- `lag=0.30` 不如 `0.35`，因为它没有把 best/final 对齐，主 final F1 也更低。
- `lag=0.40` 与 `0.35` 在主目标上基本打平，都把 2-seed 的 best/final gap 拉到 0。
- 但 `lag=0.35` 的 best eps=0.1、best margin、final support 更好一些。
- `lag=0.40` 没有展示出足够清晰的增益，暂时不值得再扩到 5-seed。

结论：

- 当前 `sim11` 的推荐权重仍保持在 `causal_lag_main_weight=0.35`。
- 在当前局部扫描里，`0.35` 比 `0.30` 明显更好；相对 `0.40` 没有吃亏，且读数更均衡。

## 17. sim11 对照：`support_prior_mode=patel_kappa`

执行时间：

- 2026-05-12

目的：

- 在当前推荐的 `sim11 topk16 + lag0.35` 基线下，验证把 `support_prior_mode` 从 `pearson_abs` 改成 `patel_kappa` 是否能进一步改善。

先验检查：

- 对 `sim11` 的已有 run 直接分析 `pearson_matrix.npy` 与 `patel_kappa.npy`：
  - `top16` 下两者都覆盖 `11/11` GT 边
  - 更关键的是，两者的 `top16` 无向 pair 集合完全相同
- 这意味着在当前 `fixed_support_mask_mode=topk_kappa`、`top_k_edges=16` 设定下，support prior 改成 `patel_kappa` 后，固定支撑 mask 实际不会变化。

2-seed 对照文件：

- `GraphExp/results/unify_replay_sim11_20260512_211716_sim11_D7_patel_support_probe.csv`
- `GraphExp/results/unify_replay_sim11_20260512_211716_sim11_D7_patel_support_probe_aggregate.csv`

对照配置：

- 在 `sim11 D3 lag0.35` 基线上，只修改：
  - `support_prior_mode=patel_kappa`

结果：

| metric | `pearson_abs + lag0.35` | `patel_kappa + lag0.35` |
| --- | ---: | ---: |
| best primary strict F1 mean | 0.6296 | 0.6296 |
| final primary strict F1 mean | 0.6296 | 0.6296 |
| best strict F1 @ eps=0.1 mean | 0.3500 | 0.3500 |
| final strict F1 @ eps=0.1 mean | 0.1875 | 0.1875 |
| best signed margin median mean | 0.0832 | 0.0832 |
| final signed margin median mean | 0.0567 | 0.0567 |
| best support median mean | 0.1136 | 0.1136 |
| final support median mean | 0.0838 | 0.0838 |
| final vs best gap mean | 0.0000 | 0.0000 |

结论：

- 在当前 `topk16` 设定下，`support_prior_mode=patel_kappa` 与 `pearson_abs` 没有实际差异。
- 原因不是“偶然跑成一样”，而是这两种 prior 在 `sim11` 上的 `top16` fixed support mask 完全相同。
- 因此这条对照不需要扩到 5-seed，也不值得在当前 `topk16` 框架下继续投入。

## 18. fMRI 单因素消融：禁用编码器与纯 `N(0,I)` 扩散噪声

执行时间：

- 2026-05-13

目的：

- 在当前 fMRI 推荐线 `F3` 上做单因素消融，判断：
  - 时序编码器是否是当前 5-seed 主指标的必要组件
  - 邻居/Patel 引导噪声是否是当前 5-seed 主指标的必要组件
- 本轮只回答 fMRI，不外推到 `sim4/sim11`。
- 本轮不做 2x2 全因子，只做两个单因素分支。

实现记录：

- `GraphExp/models/DDM.py`
  - 新增 `diffusion_noise_mode={guided,gaussian_iid}`。
  - 默认 `guided` 保持旧行为。
  - `gaussian_iid` 直接使用 `eps ~ N(0,I)`，不使用 `noise_guide_adj`、邻居统计、全局 mean/std 或额外 noise normalization。
- `GraphExp/main_structure_learning.py`
  - 新增 CLI `--diffusion_noise_mode`。
  - `gaussian_iid` 下训练期 `training_noise_guide_mode` 记为 inactive。
  - `config.npy` 记录 `diffusion_noise_mode` 与 `use_temporal_encoder`。
- `GraphExp/run_replay_saved_config.py`
  - 新增 `diffusion_noise_mode` 的默认值、透传和 summary 字段。

验证：

- `python -m py_compile GraphExp/main_structure_learning.py GraphExp/run_replay_saved_config.py GraphExp/models/DDM.py` 通过。
- `gaussian_iid` 直接 `DDM.build_noise(x, eps=eps)` 检查通过：返回噪声与给定 `eps` 完全一致。
- 1-epoch smoke 跑通过：
  - `GraphExp/results/unify_replay_fMRI_20260513_164220_fmri_ablation_disable_encoder_smoke.csv`
  - `GraphExp/results/unify_replay_fMRI_20260513_164220_fmri_ablation_gaussian_iid_smoke.csv`

基线：

- `GraphExp/results/unify_replay_fMRI_20260511_124620_fmri_F3_lag035_final_export_5seed_aggregate.csv`
- 配置：
  - base run: `GraphExp/results/run_20260412_203901`
  - seeds: `11,22,33,44,55`
  - `export_epoch_policy=final`
  - `causal_lag_main_weight=0.35`

正式消融文件：

- 禁用编码器：
  - `GraphExp/results/unify_replay_fMRI_20260513_173101_fmri_ablation_disable_encoder_5seed_v2.csv`
  - `GraphExp/results/unify_replay_fMRI_20260513_173101_fmri_ablation_disable_encoder_5seed_v2_aggregate.csv`
- 纯 `N(0,I)` 扩散噪声：
  - `GraphExp/results/unify_replay_fMRI_20260513_165348_fmri_ablation_gaussian_iid_5seed.csv`
  - `GraphExp/results/unify_replay_fMRI_20260513_165348_fmri_ablation_gaussian_iid_5seed_aggregate.csv`

配置核验：

- 禁用编码器代表 run `GraphExp/results/run_20260513_173104`：
  - `disable_temporal_encoder=True`
  - `use_temporal_encoder=False`
  - `skip_pretrain=True`
  - `pretrain_epochs=0`
  - `diffusion_noise_mode=guided`
- 纯 `N(0,I)` 代表 run `GraphExp/results/run_20260513_165351`：
  - `disable_temporal_encoder=False`
  - `pretrain_epochs=50`
  - `pretrain_checkpoint=.\results\run_20260310_185625\pretrained_encoder.pt`
  - `diffusion_noise_mode=gaussian_iid`
  - 日志显示 `Training noise guide: inactive because diffusion_noise_mode=gaussian_iid`

aggregate 对比：

| variant | best primary F1 | exported primary F1 | final primary F1 | final strict F1 @ eps=0.1 | final signed margin median | final failure modes | final-best gap |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `F3 baseline` | 0.9600 | 0.9600 | 0.9600 | 0.4452 | 0.0765 | `{'mixed_or_partial': 5}` | 0.0000 |
| `disable_encoder` | 0.9600 | 0.9600 | 0.9600 | 0.4643 | 0.0831 | `{'mixed_or_partial': 5}` | 0.0000 |
| `gaussian_iid_noise` | 0.8800 | 0.8400 | 0.8400 | 0.0667 | 0.0513 | `{'mixed_or_partial': 5}` | -0.0400 |

逐 seed 主指标：

| variant | seed 11 | seed 22 | seed 33 | seed 44 | seed 55 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `F3 baseline` final | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 |
| `disable_encoder` final | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 |
| `gaussian_iid_noise` final | 1.0 | 0.8 | 0.8 | 0.8 | 0.8 |

读数：

- 在当前 fMRI F3 线下，禁用时序编码器没有降低 5-seed 主 strict F1：
  - best/exported/final 都保持 `0.9600`
  - `final strict F1 @ eps=0.1` 从 `0.4452` 小幅升到 `0.4643`
  - `final signed margin median` 从 `0.0765` 小幅升到 `0.0831`
- 因此，至少在这个 fMRI 配置和 5-seed 口径下，不能声称时序编码器是主指标的必要组件。
- 但这不等价于“编码器没用”：
  - 本结果只覆盖 fMRI 当前 F3 family
  - 未覆盖 `sim4/sim11`
  - 未覆盖 2x2 联合消融
- 纯 `N(0,I)` 扩散噪声明显伤害当前 fMRI F3 线：
  - final primary strict F1 从 `0.9600` 降到 `0.8400`
  - best primary strict F1 从 `0.9600` 降到 `0.8800`
  - final strict F1 @ eps=0.1 从 `0.4452` 降到 `0.0667`
  - final signed margin median 从 `0.0765` 降到 `0.0513`
  - seed 44 出现 best `1.0` 但 final `0.8`，说明 final 保持能力也变差

结论：

- 当前 fMRI 推荐线中，邻居/Patel 引导噪声比时序编码器更关键。
- 对 fMRI 当前 F3 family，`diffusion_noise_mode=gaussian_iid` 不应作为推荐设置。
- `disable_temporal_encoder` 在 fMRI 上值得作为后续对照或效率/稳健性分支保留，但不能据此推广到其他数据集。

## 19. 其它数据集单因素消融：禁用编码器与纯 `N(0,I)` 扩散噪声

执行时间：

- 2026-05-13 至 2026-05-14

目的：

- 将第 18 节 fMRI 的两个单因素消融扩展到已有严格 GT 5-seed 数据集：
  - `sim2/sim3/sim4`
  - `sim8/sim10/sim11/sim12`
- 每个数据集只做两个分支：
  - `disable_temporal_encoder=true`
  - `diffusion_noise_mode=gaussian_iid`
- 不做 2x2 联合消融。

基线口径：

- `sim2`: strict audit 中当前 final 最高 5-seed family，base run `GraphExp/results/run_20260420_090231`
- `sim3`: strict audit 中当前 final 最高 5-seed family，base run `GraphExp/results/run_20260420_152306`
- `sim4`: strict audit 中当前 final 最高 5-seed family，base run `GraphExp/results/run_20260420_175556`
- `sim8/sim10/sim12`: 使用第 11 节重新预训练 5-seed 基线
- `sim11`: 使用第 15.3 节当前推荐 `topk16 + lag0.35` 5-seed 基线

配置核验：

- 所有 `disable_encoder` 分支代表 run 均为：
  - `disable_temporal_encoder=True`
  - `use_temporal_encoder=False`
  - `skip_pretrain=True`
  - `pretrain_epochs=0`
  - `diffusion_noise_mode=guided`
- 所有 `gaussian_iid` 分支代表 run 均为：
  - `disable_temporal_encoder=False`
  - `use_temporal_encoder=True`
  - `pretrain_epochs=50`
  - `diffusion_noise_mode=gaussian_iid`
- `sim8/sim10/sim11/sim12` 的 `gaussian_iid` 分支继续使用 `pretrain_checkpoint=`，即每个数据集重新预训练 encoder。

正式消融文件：

| dataset | disable encoder aggregate | gaussian iid aggregate |
| --- | --- | --- |
| `sim2` | `GraphExp/results/unify_replay_sim2_20260513_185047_sim2_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/unify_replay_sim2_20260513_185725_sim2_ablation_gaussian_iid_5seed_aggregate.csv` |
| `sim3` | `GraphExp/results/unify_replay_sim3_20260513_192354_sim3_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/unify_replay_sim3_20260513_sim3_ablation_gaussian_iid_5seed_combined_aggregate.csv` |
| `sim4` | `GraphExp/results/unify_replay_sim4_20260513_211344_sim4_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/unify_replay_sim4_20260514_sim4_ablation_gaussian_iid_5seed_combined_aggregate.csv` |
| `sim8` | `GraphExp/results/unify_replay_sim8_20260514_022611_sim8_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/unify_replay_sim8_20260514_023604_sim8_ablation_gaussian_iid_5seed_aggregate.csv` |
| `sim10` | `GraphExp/results/unify_replay_sim10_20260514_032112_sim10_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/unify_replay_sim10_20260514_033111_sim10_ablation_gaussian_iid_5seed_aggregate.csv` |
| `sim11` | `GraphExp/results/unify_replay_sim11_20260514_050123_sim11_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/unify_replay_sim11_20260514_050543_sim11_ablation_gaussian_iid_5seed_aggregate.csv` |
| `sim12` | `GraphExp/results/unify_replay_sim12_20260514_041620_sim12_ablation_disable_encoder_5seed_aggregate.csv` | `GraphExp/results/unify_replay_sim12_20260514_042044_sim12_ablation_gaussian_iid_5seed_aggregate.csv` |

aggregate 对比：

| dataset | variant | best | exported | final | final eps=0.1 | final margin | final-best gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sim2` | baseline | 0.8727 | 0.8000 | 0.8545 | 0.1256 | 0.0440 | -0.0182 |
| `sim2` | disable encoder | 0.8182 | 0.7636 | 0.7636 | 0.1498 | 0.0407 | -0.0545 |
| `sim2` | gaussian iid | 0.8545 | 0.8364 | 0.8364 | 0.1212 | 0.0451 | -0.0182 |
| `sim3` | baseline | 0.9222 | 0.8444 | 0.9111 | 0.4100 | 0.0167 | -0.0111 |
| `sim3` | disable encoder | 0.8667 | 0.8333 | 0.8333 | 0.3103 | 0.0143 | -0.0333 |
| `sim3` | gaussian iid | 0.9556 | 0.9222 | 0.9222 | 0.5331 | 0.0175 | -0.0333 |
| `sim4` | baseline | 0.8820 | 0.8492 | 0.8492 | 0.0000 | 0.0023 | -0.0328 |
| `sim4` | disable encoder | 0.8098 | 0.7967 | 0.7967 | 0.0000 | 0.0023 | -0.0131 |
| `sim4` | gaussian iid | 0.9311 | 0.8295 | 0.8295 | 0.0000 | 0.0022 | -0.1016 |
| `sim8` | baseline | 0.8800 | 0.8400 | 0.8400 | 0.1500 | 0.0790 | -0.0400 |
| `sim8` | disable encoder | 0.8800 | 0.8400 | 0.8400 | 0.2643 | 0.0864 | -0.0400 |
| `sim8` | gaussian iid | 0.8400 | 0.7600 | 0.7600 | 0.1810 | 0.0648 | -0.0800 |
| `sim10` | baseline | 0.8400 | 0.8000 | 0.8000 | 0.1333 | 0.0665 | -0.0400 |
| `sim10` | disable encoder | 0.8800 | 0.8400 | 0.8400 | 0.1143 | 0.0741 | -0.0400 |
| `sim10` | gaussian iid | 0.8400 | 0.8000 | 0.8000 | 0.1810 | 0.0472 | -0.0400 |
| `sim11` | baseline | 0.6370 | 0.5926 | 0.5926 | 0.0750 | 0.0487 | -0.0444 |
| `sim11` | disable encoder | 0.6370 | 0.5481 | 0.5481 | 0.0333 | 0.0325 | -0.0889 |
| `sim11` | gaussian iid | 0.6370 | 0.5926 | 0.5926 | 0.0800 | 0.0442 | -0.0444 |
| `sim12` | baseline | 0.7818 | 0.7273 | 0.7273 | 0.2056 | 0.0340 | -0.0545 |
| `sim12` | disable encoder | 0.7818 | 0.6909 | 0.6909 | 0.1282 | 0.0262 | -0.0909 |
| `sim12` | gaussian iid | 0.8000 | 0.7455 | 0.7455 | 0.1749 | 0.0400 | -0.0545 |

读数：

- 禁用编码器在其它数据集上不是稳定中性：
  - 负向：`sim2/sim3/sim4/sim11/sim12`
  - 基本持平或略正：`sim8/sim10`
- 纯 `N(0,I)` 噪声不是全局负向：
  - 明显负向：`sim8`
  - 略负向：`sim2/sim4`
  - 基本持平：`sim10/sim11`
  - 正向：`sim3/sim12`
- `sim3` 是最重要的反例：
  - final primary F1 从 `0.9111` 升到 `0.9222`
  - final eps=0.1 从 `0.4100` 升到 `0.5331`
  - best 也从 `0.9222` 升到 `0.9556`
- `sim4` 的纯 `N(0,I)` 分支比较复杂：
  - best 从 `0.8820` 升到 `0.9311`
  - final 从 `0.8492` 降到 `0.8295`
  - final-best gap 明显扩大到 `-0.1016`
  - 这更像训练中出现更高峰值但 final 保持变差，不应按 final 推荐。
- `sim12` 的纯 `N(0,I)` 分支小幅提升 final：
  - final 从 `0.7273` 到 `0.7455`
  - margin 和 eps=0.1 的方向不完全一致，仍需谨慎。

结论：

- 第 18 节 fMRI 的“邻居/Patel 引导噪声更关键”不能外推为全局结论。
- 更准确的跨数据集结论是：
  - 时序编码器对多数 synthetic family 有帮助或至少不能随意移除。
  - 噪声引导机制存在明显数据集依赖；它帮助 fMRI/sim8，可能不帮助甚至压制 sim3/sim12。
  - `gaussian_iid` 值得作为 `sim3` 和可能的 `sim12` 后续正式候选，但不应全局替换默认 `guided`。
- 如果继续推进，应优先做：
  - `sim3 gaussian_iid` 的 selector/final 机制复核
  - `sim12 gaussian_iid` 的 5-seed 结果复核和方向指标复算
  - `sim4 gaussian_iid` 的 best-vs-final 退化分析
