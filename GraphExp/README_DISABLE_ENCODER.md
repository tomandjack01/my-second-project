# 禁用时序编码器功能说明

## 概述

现在你可以选择是否使用时序编码器 (`NodeSpecificTemporalEncoder`)。禁用编码器后，模型将直接对原始 fMRI 时序数据进行扩散过程，而不是先编码到低维特征空间。

## 两种模式对比

### 模式 1: 启用编码器 (默认)
```
原始数据 [N, 200]
  ↓ temporal_encoder
编码特征 [N, 64]
  ↓ 加噪 + 去噪
输出 [N, 64]
```

### 模式 2: 禁用编码器 (新增)
```
原始数据 [N, 200]
  ↓ 直接加噪 + 去噪
输出 [N, 200]
```

## 使用方法

### 1. 禁用编码器运行

```bash
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --epochs 100 \
    --disable_temporal_encoder
```

**关键参数:**
- `--disable_temporal_encoder`: 禁用时序编码器，直接对原始数据加噪

### 2. 启用编码器运行 (默认行为)

```bash
# 方式 A: 使用预训练编码器 (推荐)
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --epochs 100 \
    --pretrain_epochs 50

# 方式 B: 跳过预训练 (端到端训练，可能导致编码器崩溃)
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --epochs 100 \
    --skip_pretrain

# 方式 C: 加载已有的预训练权重
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --epochs 100 \
    --pretrain_checkpoint ./results/run_xxx/pretrained_encoder.pt
```

## 完整参数列表

```bash
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --time_points 200 \
    --epochs 100 \
    --lr 1e-3 \
    --lambda_l1 0.1 \
    --num_hidden 64 \
    --num_layers 2 \
    --batch_size 4 \
    --device cuda \
    --seed 42 \
    --log_interval 10 \
    --top_k_edges 50 \
    --disable_temporal_encoder \  # 新增参数
    --debug_checks  # 可选：启用调试检查
```

## 预训练相关参数

当**不使用** `--disable_temporal_encoder` 时，以下参数生效：

```bash
--pretrain_epochs 50              # 编码器预训练轮数
--pretrain_lr 1e-3                # 预训练学习率
--pretrain_split_ratio 0.75       # 输入/预测分割比例 (150/50)
--skip_pretrain                   # 跳过预训练 (不推荐)
--pretrain_checkpoint PATH        # 加载已有预训练权重
```

## 何时使用哪种模式？

### 使用编码器 (默认)
**优点:**
- 降维到 64 维，计算更快
- 编码器可以学习时序模式
- 适合长时序数据 (T > 100)

**缺点:**
- 需要预训练编码器 (额外时间)
- 可能丢失部分时序信息
- 编码器可能崩溃 (需要监控 collapse diagnostics)

### 禁用编码器
**优点:**
- 保留完整的时序信息 (200 维)
- 无需预训练，直接开始训练
- 避免编码器崩溃问题

**缺点:**
- 计算量更大 (200 维 vs 64 维)
- 训练可能更慢
- 需要更多内存

## 测试验证

运行测试脚本验证功能：

```bash
cd GraphExp
python test_disable_encoder.py
```

预期输出：
```
============================================================
测试 1: 启用时序编码器 (默认行为)
============================================================
Temporal Encoder: 200 time points → 64 features
前向传播成功! Loss: 1.2130

============================================================
测试 2: 禁用时序编码器 (直接对原始数据加噪)
============================================================
Temporal Encoder: DISABLED - Using raw time series directly
前向传播成功! Loss: 1.0481

============================================================
测试 3: 维度一致性检查
============================================================
启用编码器 - Loss: 0.8291
禁用编码器 - Loss: 1.0253

✓ 两种模式都能正常工作!
============================================================
所有测试通过! ✓
============================================================
```

## 输出文件

训练完成后，结果保存在 `./results/run_TIMESTAMP/` 目录：

```
results/run_20260303_123456/
├── learned_adjacency.npy         # 学习到的邻接矩阵
├── learned_adjacency.csv
├── loss_curve.png                # 损失曲线
├── loss_history.csv
├── pearson_matrix.csv            # Pearson 相关矩阵
├── config.npy                    # 配置参数
├── pretrained_encoder.pt         # 预训练编码器权重 (仅启用编码器时)
├── collapse_diagnostics.png      # 编码器崩溃诊断 (仅启用编码器时)
└── collapse_diagnostics.csv      # 编码器崩溃指标 (仅启用编码器时)
```

## 代码修改位置

如果你需要进一步定制，以下是修改的核心文件：

1. **`models/DDM.py`**
   - 添加了 `use_temporal_encoder` 参数
   - 修改了 `forward()`, `embed()` 方法

2. **`main_structure_learning.py`**
   - 添加了 `--disable_temporal_encoder` 参数
   - 修改了预训练和冻结逻辑
   - 修改了 collapse diagnostics 逻辑

3. **`test_disable_encoder.py`** (新增)
   - 测试脚本，验证两种模式

## 常见问题

### Q1: 禁用编码器后训练变慢了？
A: 正常现象。去噪网络需要处理 200 维而不是 64 维数据。可以尝试：
- 减少 `--num_layers`
- 减少 `--num_hidden`
- 使用更小的 `--batch_size`

### Q2: 禁用编码器后 Loss 更高？
A: 这是正常的。200 维空间的重构难度比 64 维更大。关注 Loss 的收敛趋势，而不是绝对值。

### Q3: 如何判断哪种模式效果更好？
A: 比较最终学习到的邻接矩阵与 ground truth 的相似度 (如果有)，或者使用下游任务 (如分类) 的性能。

### Q4: 可以中途切换模式吗？
A: 不建议。两种模式的模型结构不同，无法直接迁移权重。

## 技术细节

### 维度变化

**启用编码器:**
```python
x: [N, 200]                    # 原始时序
  → temporal_encoder
x_encoded: [N, 64]             # 编码特征
  → layer_norm
  → sample_q (加噪)
x_t: [N, 64]                   # 加噪后
  → denoising_unet
out: [N, 64]                   # 去噪输出
```

**禁用编码器:**
```python
x: [N, 200]                    # 原始时序
  → layer_norm
  → sample_q (加噪)
x_t: [N, 200]                  # 加噪后
  → denoising_unet
out: [N, 200]                  # 去噪输出
```

### 去噪网络输入维度

- 启用编码器: `Denoising_Unet(in_dim=64, out_dim=64, ...)`
- 禁用编码器: `Denoising_Unet(in_dim=200, out_dim=200, ...)`

## 示例实验

### 实验 1: 对比两种模式

```bash
# 运行 1: 启用编码器
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --epochs 100 \
    --pretrain_epochs 50 \
    --seed 42

# 运行 2: 禁用编码器
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/sim4.csv \
    --epochs 100 \
    --disable_temporal_encoder \
    --seed 42
```

然后比较两个 `results/` 目录中的 `learned_adjacency.csv`。

### 实验 2: 消融研究

```bash
# Baseline: 启用编码器 + 预训练
python main_structure_learning.py --epochs 100 --pretrain_epochs 50

# Ablation 1: 启用编码器 + 跳过预训练
python main_structure_learning.py --epochs 100 --skip_pretrain

# Ablation 2: 禁用编码器
python main_structure_learning.py --epochs 100 --disable_temporal_encoder
```

## 参考

- 原始论文: Directional Diffusion Models (NeurIPS 2023)
- 编码器预训练: `pretrain_temporal_encoder.py`
- 崩溃诊断: `diagnose_encoder_collapse()`
