# QWEN.md — DDM (Directional Diffusion Models) for Brain Connectivity Structure Learning

## Project Overview

本项目基于 **Directional Diffusion Models (DDM)** (NeurIPS 2023) 实现了从 fMRI 时间序列数据中学习**有向因果脑连接矩阵**的框架。原始 DDM 是用于图表示学习（图/节点分类）的方向性扩散模型，本项目将其改造用于**脑连接因果结构学习**。

**核心任务：**

1. **骨架发现** — 哪些脑区对之间存在连接
2. **方向辨识** — 因果方向 A→B 还是 B→A
3. **检查点选择** — 训练中哪个 epoch 的邻接矩阵最优

**主要技术栈：** PyTorch、DGL (Deep Graph Library)、NumPy、Pandas、scikit-learn、Matplotlib

## Directory Structure

| 目录                 | 说明                                                                |
| :------------------- | :------------------------------------------------------------------ |
| `GraphExp/`          | 图分类实验 + **fMRI 结构学习主程序** (`main_structure_learning.py`) |
| `NodeExp/`           | 节点分类实验                                                        |
| `nni_search/`        | NNI 超参数搜索配置                                                  |
| `json_config/`       | 数据集 JSON 配置（MUTAG、Cora、IMDB-B 等）                          |
| `fMRI_dataset/`      | 仿真 fMRI 数据集（sim2/sim3/sim4, 含 GT 文件 h1-h4.txt）            |
| `patel/`             | Patel tau/kappa 参考 MATLAB 实现                                    |
| `GraphExp/results/`  | 训练输出（不提交 git）                                              |
| `GraphExp/models/`   | 模型定义（DDM.py 等）                                               |
| `GraphExp/utils/`    | 工具函数                                                            |
| `GraphExp/datasets/` | 数据集加载器                                                        |
| `GraphExp/yamls/`    | 实验 YAML 配置                                                      |

## Building and Running

### Environment Setup

```shell
conda create -n ddm python=3.8
conda activate ddm
pip install -r requirements.txt
```

> **注意：** PyTorch、torchvision 和 DGL (CUDA 11.3) 需单独安装（见 `requirements.txt` 中注释行）。

### Running Experiments

**Graph Classification** (GraphExp 目录):

```shell
cd GraphExp
python main_graph.py --yaml_dir ./yamls/MUTAG.yaml
```

**Node Classification** (NodeExp 目录):

```shell
cd NodeExp
python main_node.py --yaml_dir ./yamls/photo.yaml
```

**Brain Connectivity Structure Learning** (核心功能):

```shell
cd GraphExp

# 完整流程（预训练 + 扩散训练）
python main_structure_learning.py --epochs 100 --pretrain_epochs 50

# 跳过预训练
python main_structure_learning.py --epochs 100 --skip_pretrain

# 加载已有预训练权重
python main_structure_learning.py --pretrain_checkpoint ./results/run_xxx/pretrained_encoder.pt

# 禁用时序编码器（直接在原始时序上扩散）
python main_structure_learning.py --epochs 100 --disable_temporal_encoder
```

### 关键 CLI 参数

| 参数                         | 默认值                     | 说明               |
| :--------------------------- | :------------------------- | :----------------- |
| `--csv_path`                 | `../fMRI_dataset/sim4.csv` | fMRI 数据路径      |
| `--time_points`              | 200                        | 每个被试的时间点数 |
| `--epochs`                   | 100                        | 扩散训练轮数       |
| `--lr`                       | 1e-3                       | 学习率             |
| `--lambda_l1`                | 0.1                        | L1 稀疏正则系数    |
| `--pretrain_epochs`          | 50                         | 自回归预训练轮数   |
| `--skip_pretrain`            | False                      | 跳过预训练         |
| `--disable_temporal_encoder` | False                      | 禁用时序编码器     |

## Architecture

### Training Pipeline

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

### Core Model Components

**DDM Model** (`GraphExp/models/DDM.py`):

- `DDM` class: 主扩散模型，支持 `coupled` 和 `support_direction` 两种结构参数化
- `build_noise()`: 构建邻居引导的各向异性噪声
- `sample_q()`: 前向扩散过程
- `Denoising_Unet`: GraphConv/GCN-based U-Net 去噪网络
- `NodeSpecificTemporalEncoder`: 因果膨胀时序编码器（dilation 1-2-4）

**结构参数化模式：**

1. **`coupled`**: 单一 `sender @ receiver.T + bias` → sigmoid → 邻接矩阵
2. **`support_direction`** (当前主力):
   - 支持分支（对称）+ 方向分支（非对称）
   - `adj[i,j] = sigmoid(support_logits) × sigmoid(D[i,j] - D[j,i])`

### Datasets

| 文件     | 节点数 | GT 边数 | GT 文件 | 真实密度 |
| :------- | :----- | :------ | :------ | :------- |
| fMRI.csv | 5      | 5       | h1.txt  | 50%      |
| sim2.csv | 10     | 11      | h2.txt  | ~24%     |
| sim3.csv | 15     | 18      | h3.txt  | ~17%     |
| sim4.csv | 50     | 61      | h4.txt  | ~5%      |

所有数据集：50 subjects × 200 time points。

### Key Code Paths

| 功能                    | 文件位置                                      |
| :---------------------- | :-------------------------------------------- |
| 结构参数化初始化        | `GraphExp/models/DDM.py:300-330`              |
| 支持 logits（对称强制） | `GraphExp/models/DDM.py:366-372`              |
| 方向 logits（非对称）   | `GraphExp/models/DDM.py:374-382`              |
| 最终邻接矩阵组合        | `GraphExp/models/DDM.py:384-415`              |
| 噪声构建                | `GraphExp/models/DDM.py:581-651`              |
| 梯度路由                | `GraphExp/main_structure_learning.py:720-786` |
| Causal-lag 主损失       | `GraphExp/main_structure_learning.py:1754`    |
| 方向 margin loss        | `GraphExp/main_structure_learning.py:1075`    |
| Best-epoch 评分         | `GraphExp/main_structure_learning.py:1947`    |

### Output Files

训练结果保存在 `GraphExp/results/run_<timestamp>/`:

- `learned_adjacency.csv` — 学习到的脑区连接矩阵
- `loss_curve.png` — 训练收敛曲线
- `collapse_diagnostics.png` — 编码器坍塌诊断图
- `pearson_matrix.csv` — Pearson 相关矩阵（参考基线）
- `loss_history.csv` — 逐 epoch 损失记录
- `pretrained_encoder.pt` — 预训练编码器权重
- `config.npy` — 运行配置

## Development Conventions

### Coding Style

- Python: 4-space indentation, PEP 8–style naming
- `snake_case` for functions/vars, `CamelCase` for classes
- Type hints for new/changed public functions
- Keep diffs small and targeted; avoid sweeping reformatting

### Testing

- No unified test runner; use targeted script checks:
  - `python GraphExp/test_temporal_encoder.py`
  - `python GraphExp/test_eval.py --gt ../fMRI_dataset/h4.txt --top_k 61`

### Static Analysis

- Run `pyright` for type checking (config in `pyrightconfig.json`)
- Extra paths configured for `./GraphExp`

## Experimental Constraints (from constrict.md)

**冻结主线策略：** 当前共享训练循环已冻结，不向主训练循环添加新辅助损失。

**两条独立实验线：**

1. **机制线 (Mechanism)** — 针对导出空间方向保留问题，候选方案：exported-gate floor、exported-margin hinge、support-preservation constraint
2. **选择线 (Selection)** — 针对检查点选择问题，修改 `compute_epoch_quality()` 和守卫规则

**实验规则：**

- 每个实验必须声明：分支名称、数据集、测试的明确声明、停止条件
- 不要同时修改机制和选择器
- 与冻结主线对比，不与移动目标对比

## Key Experimental Conclusions

1. **扩散主损失对因果方向完全无感** — 方向学习完全依赖外部辅助损失
2. **`support_direction` 参数化是有效的** — 解耦支持（对称）和方向（非对称）学习
3. **`maxgap_kappa` 固定支持掩码不可去除** — 去掉后支持选择性崩溃
4. **Causal-lag main loss 是最强方向信号** — 滞后重建差异提供方向敏感梯度
5. **导出空间方向保留是主瓶颈** — 内部方向解正确但导出邻接矩阵方向近似对称

## Documentation Index

| 文件                      | 内容                                             |
| :------------------------ | :----------------------------------------------- |
| `CLAUDE.md`               | 详细的项目架构、实验结论和代码路径               |
| `constrict.md`            | 主线冻结策略 + 机制线/选择线实验日志             |
| `experiment_synthesis.md` | 全实验历程综合整理                               |
| `plan.md`                 | Causal-lag 扩散计划（含 Phase 0/0B/0C 诊断结果） |
| `ablation.md`             | Support learning 消融实验记录                    |
| `AGENTS.md`               | 仓库指南（构建、测试、编码规范）                 |
| `README.md`               | 项目概述和基本使用方法                           |
