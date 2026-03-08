# Brain Connectivity Structure Learning with Directional Diffusion Models

基于 [Directional Diffusion Models (DDM)](https://arxiv.org/abs/2306.13210) (NeurIPS 2023) 的 fMRI 脑连接结构学习框架。通过方向性扩散过程从 fMRI 时间序列数据中学习稀疏的脑区连接矩阵。

## 方法概述

本项目将 DDM 的方向性噪声机制应用于脑功能连接分析：

1. **时序编码器 (NodeSpecificTemporalEncoder)**：因果膨胀卷积网络（dilation 1-2-4），将每个脑区的 fMRI 时间序列 `[N, T]` 编码为因果特征 `[N, T]`（同维度，无压缩）
2. **编码器自回归预训练 (Anti-Collapse)**：用 t 时刻预测 t+1 时刻的自回归目标独立预训练编码器后冻结，防止与扩散过程端到端训练时发生表征坍塌
3. **方向性扩散**：在原始时间维度空间中执行前向扩散（数据依赖的各向异性噪声）+ U-Net 去噪
4. **结构学习**：通过可学习邻接矩阵 + L1 稀疏正则化，从扩散过程中提取脑区间的连接模式

### 编码器预训练流程

`NodeSpecificTemporalEncoder` 在端到端训练时会坍塌（cosine sim → 1.0, Diff Loss → 0.0）。解决方案：自回归预训练后冻结。

**自回归预训练损失：**
- 使用 `pretrain_forward()`：通过 MSE 在未归一化特征上预测 t+1 时刻
- 绕过 LayerNorm 以保留信号尺度
- `pred_head`（Linear）将编码器输出映射到预测空间

### 坍塌诊断指标

训练过程中自动监控以下健康指标：

| 指标 | 健康范围 | 坍塌信号 |
|:---|:---|:---|
| `effective_rank` | > 5（理想 > 10） | < 5 |
| `mean_cosine_sim` | < 0.5（理想 < 0.3） | > 0.8 |
| `dead_dims_ratio` | 0% | > 30% |
| `feature_std_mean` | > 0.1 | → 0.0 |

## 环境配置

```shell
conda create -n ddm python=3.8
conda activate ddm
pip install -r requirements.txt
```

PyTorch、torchvision 和 DGL (CUDA 11.3) 需单独安装，详见 `requirements.txt` 中的注释行。

## 使用方法

所有命令在 `GraphExp/` 目录下执行：

```shell
cd GraphExp
```

### 完整流程（预训练 + 冻结 + 扩散训练）

```shell
python main_structure_learning.py --epochs 100 --pretrain_epochs 50
```

### 跳过预训练

```shell
python main_structure_learning.py --epochs 100 --skip_pretrain
```

### 加载已有预训练权重

```shell
python main_structure_learning.py --pretrain_checkpoint ./results/run_xxx/pretrained_encoder.pt
```

### 禁用时序编码器（直接在原始时序上扩散）

```shell
python main_structure_learning.py --epochs 100 --disable_temporal_encoder
```

## 主要参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--csv_path` | `../fMRI_dataset/sim4.csv` | fMRI 数据路径（无表头 CSV） |
| `--time_points` | 200 | 每个被试的时间点数 |
| `--epochs` | 100 | 扩散训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--lambda_l1` | 0.1 | L1 稀疏正则系数（按 N² 归一化） |
| `--num_hidden` | 64 | 隐藏层维度 |
| `--num_layers` | 2 | GNN 层数 |
| `--batch_size` | 4 | 被试批大小 |
| `--pretrain_epochs` | 50 | 自回归预训练轮数 |
| `--pretrain_lr` | 1e-3 | 预训练学习率 |
| `--skip_pretrain` | False | 跳过预训练（等价于 `--pretrain_epochs 0`） |
| `--pretrain_checkpoint` | None | 已有预训练权重路径 |
| `--disable_temporal_encoder` | False | 禁用时序编码器，直接在原始时序上扩散 |
| `--debug_checks` | False | 启用首步调试检查 |

## 训练流程

```
fMRI CSV [Total_Rows, N]
    │
    ├─ reshape → data_3d [Num_Subjects, N, T]
    ├─ Pearson 相关矩阵 → init_features (邻接矩阵初始化)
    └─ Patel 连接矩阵 → noise_guide_adj (邻居噪声引导)
         │
         ▼
┌─────────────────────────────────┐
│  1. 编码器自回归预训练 (可选)     │
│     predict t+1 from t (MSE)   │
│     → 冻结 temporal_encoder     │
├─────────────────────────────────┤
│  2. 扩散训练                     │
│     temporal_encoder(x) → z     │
│     [N, T] → [N, T] (同维度)    │
│     sample_q(t, z) → z_t        │
│     Denoising_Unet(z_t) → ẑ    │
│     Loss: cosine_sim + L1_adj   │
└─────────────────────────────────┘
         │
         ▼
   learned_adjacency [N, N]
```

## 输出文件

训练结果保存在 `./results/run_<timestamp>/` 下：

- `learned_adjacency.csv` — 学习到的脑区连接矩阵
- `loss_curve.png` — 训练收敛曲线
- `collapse_diagnostics.png` — 编码器坍塌诊断图
- `collapse_diagnostics.csv` — 坍塌指标原始数据
- `pearson_matrix.csv` — Pearson 相关矩阵（参考基线）
- `loss_history.csv` — 逐 epoch 损失记录
- `pretrained_encoder.pt` — 预训练编码器权重
- `config.npy` — 运行配置

## 核心依赖

- PyTorch
- DGL (Deep Graph Library)
- NumPy / Pandas
- scikit-learn
- Matplotlib

## 引用

```bibtex
@inproceedings{yang2023directional,
  title={Directional Diffusion Models for Graph Representation Learning},
  author={Yang, Run and Yang, Yuling and Zhou, Fan and Sun, Qiang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
