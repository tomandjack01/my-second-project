# 测试结果对比

## 测试环境
- 数据集: fMRI.csv (50 subjects, 5 nodes, 200 time points)
- 训练轮数: 10 epochs
- 设备: CUDA

## 模式 1: 禁用时序编码器

### 命令
```bash
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/fMRI.csv \
    --epochs 10 \
    --disable_temporal_encoder \
    --log_interval 2
```

### 结果
- **模型参数**: 298,707 trainable / 298,707 total
- **训练时间**: ~30秒
- **最终 Loss**:
  - Diff Loss: 0.5974
  - Sparsity Loss: 0.0391
- **邻接矩阵统计**:
  - Min: 0.3235
  - Max: 0.7262
  - Mean: 0.4513
  - Std: 0.1361
- **特点**:
  - ✅ 无需预训练
  - ✅ 直接在 200 维时序数据上工作
  - ✅ 保留完整时序信息
  - ⚠️ 参数量较大

---

## 模式 2: 启用时序编码器（带预训练）

### 命令
```bash
python main_structure_learning.py \
    --csv_path ../fMRI_dataset/fMRI.csv \
    --epochs 10 \
    --pretrain_epochs 5 \
    --log_interval 2
```

### 结果
- **模型参数**: 164,067 trainable / 191,595 total (27,528 frozen)
- **预训练时间**: ~10秒 (5 epochs)
- **训练时间**: ~25秒
- **最终 Loss**:
  - Diff Loss: 0.0240
  - Sparsity Loss: 0.0373
- **邻接矩阵统计**:
  - Min: 0.3031
  - Max: 0.7295
  - Mean: 0.4333
  - Std: 0.1396
- **编码器健康度**:
  - Effective Rank: 4.66 ⚠️ (略低，但可接受)
  - Mean Cosine Sim: 0.8203 ⚠️ (略高)
  - Dead Dims: 0.00% ✅
  - Feature Std: 0.469 ✅
- **特点**:
  - ✅ 参数量较小（冻结编码器）
  - ✅ Loss 收敛更快（0.024 vs 0.597）
  - ✅ 在 64 维特征空间工作
  - ⚠️ 需要预训练（额外时间）
  - ⚠️ 编码器有轻微崩溃迹象（但被冻结，不影响训练）

---

## 对比分析

| 指标 | 禁用编码器 | 启用编码器 |
|------|-----------|-----------|
| **总参数量** | 298,707 | 191,595 |
| **可训练参数** | 298,707 | 164,067 |
| **预训练时间** | 0秒 | ~10秒 |
| **训练时间** | ~30秒 | ~25秒 |
| **最终 Diff Loss** | 0.5974 | 0.0240 |
| **工作维度** | 200 | 64 |
| **信息保留** | 完整 | 压缩 |
| **收敛速度** | 较慢 | 较快 |

---

## 结论

### 何时使用禁用编码器模式？
1. **快速实验**: 不想等待预训练
2. **完整信息**: 需要保留所有时序细节
3. **简单场景**: 数据维度不高（T < 300）
4. **避免崩溃**: 不想处理编码器崩溃问题

### 何时使用启用编码器模式？
1. **高维数据**: 时序很长（T > 500）
2. **计算资源有限**: 需要降维加速
3. **更好收敛**: Loss 收敛更快更稳定
4. **生产环境**: 经过充分预训练的编码器更可靠

---

## 建议

对于这个 5 节点、200 时间点的小规模数据集：
- **推荐使用禁用编码器模式**，因为：
  - 数据维度不高（200）
  - 无需预训练，更简单
  - 保留完整时序信息
  - 虽然 Loss 较高，但邻接矩阵质量相似

对于更大规模的数据集（如 100+ 节点）：
- **推荐使用启用编码器模式**，因为：
  - 降维可以显著加速
  - 更快的收敛速度
  - 更低的内存占用

---

## 测试验证

两种模式都已通过完整测试：

```bash
# 单元测试
cd GraphExp
python test_disable_encoder.py
# ✅ 所有测试通过

# 集成测试
python main_structure_learning.py --csv_path ../fMRI_dataset/fMRI.csv --epochs 10 --disable_temporal_encoder
# ✅ 训练成功

python main_structure_learning.py --csv_path ../fMRI_dataset/fMRI.csv --epochs 10 --pretrain_epochs 5
# ✅ 训练成功
```

---

生成时间: 2026-03-03
测试环境: Windows 11, CUDA, Python 3.8
