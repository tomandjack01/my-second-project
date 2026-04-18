# DDM 因果结构学习算法：无代码完整讲解

> 本文档面向无法阅读代码的 AI 或人类读者，用自然语言 + 数学公式完整描述算法的每个关键模块。

---

## 1. 问题定义

**输入**：$N$ 个脑区的 fMRI 时间序列，维度 $[N, T]$（$N$ = 脑区数，$T$ = 时间点数，通常 200）。数据集包含多个被试（subject），每个被试都有独立的 $[N, T]$ 矩阵。

**输出**：一个 $N \times N$ 的有向因果邻接矩阵 $A_{\text{causal}}$，其中 $A_{\text{causal}}[i, j] > 0$ 表示脑区 $i$ 因果性地影响脑区 $j$（$i \to j$）。

**三个子任务**：

1. **骨架发现**（skeleton）：哪些脑区对之间存在连接？
2. **方向辨识**（orientation）：连接方向是 $A \to B$ 还是 $B \to A$？
3. **检查点选择**（checkpoint selection）：训练过程中哪个 epoch 的邻接矩阵最优？

---

## 2. 整体框架：扩散模型用于结构学习

本算法基于 **去噪扩散模型（Denoising Diffusion Model, DDM）**，原本用于图/节点分类的表示学习。核心改造是：**将图的邻接矩阵从固定已知变为可学习参数**。

### 2.1 扩散模型回顾

扩散模型分两个过程：

**前向扩散**（加噪）：给干净数据 $x_0$ 逐步加噪得到 $x_t$：

$$x_t = \sqrt{\bar\alpha_t} \cdot x_0 + \sqrt{1 - \bar\alpha_t} \cdot \epsilon'$$

其中 $\bar\alpha_t = \prod_{s=1}^{t} (1 - \beta_s)$ 是噪声调度系数，$\epsilon'$ 是结构化噪声（见第 4 节）。

**反向去噪**（重建）：训练一个神经网络（GraphConv/GCN-based U-Net），从带噪数据 $x_t$ 和时间步 $t$ 预测原始干净数据 $x_0$。损失函数是预测值与真实 $x_0$ 之间的 smooth-L1 + cosine 混合损失。

### 2.2 关键改造

原始 DDM 在**已知固定图结构**上做表示学习。我们的改造是：

- 邻接矩阵变成可学习参数
- 去噪网络（GNN）使用学习到的邻接矩阵传递消息
- 邻接矩阵的质量直接影响去噪性能，形成间接的自监督信号

---

## 3. 邻接矩阵参数化（support_direction 模式）

这是算法的核心创新。邻接矩阵被分解为两个独立分支：

### 3.1 支持分支（Support Branch）— 对称，回答"是否连接"

每个节点 $i$ 有一对可学习嵌入向量 $s_i^{\text{sender}}, s_i^{\text{receiver}} \in \mathbb{R}^d$。

**原始 logits**：

$$L_{ij} = s_i^{\text{sender}} \cdot (s_j^{\text{receiver}})^\top + b_{ij}$$

**对称强制**：

$$S_{ij} = \frac{1}{2}(L_{ij} + L_{ji})$$

对称化确保 $S_{ij} = S_{ji}$，即支持分支不编码方向信息。数值裁剪到 $[-6, 6]$ 防止梯度爆炸。

**支持权重**：

$$W_{ij}^{\text{support}} = \sigma(S_{ij})$$

其中 $\sigma$ 是 sigmoid 函数，输出在 $(0, 1)$。

### 3.2 方向分支（Direction Branch）— 非对称，回答"谁影响谁"

另一组独立嵌入 $d_i^{\text{sender}}, d_i^{\text{receiver}} \in \mathbb{R}^d$。

**方向 logits**：

$$D_{ij} = d_i^{\text{sender}} \cdot (d_j^{\text{receiver}})^\top$$

注意：$D_{ij} \neq D_{ji}$（不做对称化），这一矩阵编码方向偏好。

**方向门控**：

$$G_{ij} = \sigma(D_{ij} - D_{ji})$$

这个设计有一个优雅的数学性质：$G_{ij} + G_{ji} = 1$。即对于任何节点对 $(i, j)$，如果 $i \to j$ 的门控值为 0.7，则 $j \to i$ 的门控值必为 0.3。方向是零和博弈。

### 3.3 最终邻接矩阵

$$A_{ij} = W_{ij}^{\text{support}} \times G_{ij} \times M_{ij}^{\text{diag}} \times M_{ij}^{\text{fixed}}$$

其中：

- $W_{ij}^{\text{support}}$：对称支持权重（连接强度）
- $G_{ij}$：方向门控（方向偏好）
- $M_{ij}^{\text{diag}} = 1 - \delta_{ij}$：对角线掩码（禁止自环）
- $M_{ij}^{\text{fixed}}$：固定硬支持掩码（见第 3.4 节）

### 3.4 固定硬支持掩码（maxgap_kappa）

这是一个**不可学习的二值矩阵**，在训练前从统计先验计算得到，训练中保持固定。

**来源**：Patel kappa 统计量。对每对脑区 $(i, j)$，计算 Patel kappa 值（衡量两个脑区共激活的一致性，类似 Cohen's kappa）。

**掩码生成方法**（maxgap 算法）：

1. 将所有节点对的 kappa 值从大到小排序
2. 计算相邻 kappa 值之间的差距（gap）
3. 找到最大 gap 的位置作为分割点
4. 高于分割点的节点对标记为 1（允许连接），低于的标记为 0（禁止连接）

**作用**：将搜索空间从 $N^2$ 限制到一个小子集。这个掩码被证明是**不可移除的关键组件**——去掉后 F1 从 0.87 崩溃到 0.08。

**实验证据**：虽然掩码依赖 Patel 先验，但用 Pearson 相关（另一种简单统计量）生成掩码在 maxgap 算子下产生完全相同的骨架，暗示掩码对先验来源不敏感。

---

## 4. 噪声构建（Guided Noise Construction）

标准扩散模型使用各向同性高斯噪声。本算法使用**邻居引导的各向异性噪声**，使噪声与图结构关联。

### 4.1 噪声引导邻接矩阵

有一个**固定的对称先验邻接矩阵** $A_{\text{guide}}$（来自 Patel kappa 或 Pearson 相关），注册为不可训练的 buffer。

### 4.2 噪声统计量

对每个节点 $i$，利用其邻居的信号分布计算噪声参数：

**邻居均值**：

$$\mu_i = \sum_j A_{\text{guide}}[i,j] \cdot x[j]$$

**邻居方差**：

$$\sigma_i^2 = \sum_j A_{\text{guide}}[i,j] \cdot x[j]^2 - \mu_i^2$$

**邻居标准差**：

$$\sigma_i = \sqrt{\max(\sigma_i^2, \; 10^{-6})}$$

### 4.3 噪声生成

采样标准高斯 $\epsilon \sim \mathcal{N}(0, I)$，然后：

$$\epsilon'_i = \epsilon_i \cdot \sigma_i$$

注意：使用"零均值模式"（默认开启），丢弃 $\mu_i$ 偏置。这是因为加入邻居均值会导致噪声与信号产生相关性，违反扩散模型的前提假设。

噪声生成后可选 LayerNorm 归一化（即在特征维度做标准化）。

### 4.4 关键限制

$A_{\text{guide}}$ 是**固定 buffer，不参与梯度计算**。这意味着去噪主损失对噪声构建无梯度反馈，不能通过噪声质量间接学习结构。这是"扩散主损失对方向无感"这一核心发现的部分原因。

---

## 5. 时序编码器（Temporal Encoder）

### 5.1 架构

因果膨胀卷积网络，包含三层一维卷积：

- 第1层：膨胀率=1，核大小=3
- 第2层：膨胀率=2，核大小=3（感受野6个时间点）
- 第3层：膨胀率=4，核大小=3（感受野14个时间点）

**因果性保证**：使用左填充（left-padding），然后截断右侧多余的输出，确保时刻 $t$ 的输出只依赖 $t$ 及之前的输入，不会"偷看未来"。

输入 $[N, T]$ → 输出 $[N, T]$（维度不变）。

### 5.2 预训练

编码器可以独立预训练：用时刻 $t$ 的特征预测时刻 $t+1$ 的原始信号值（自回归任务）。预训练后冻结编码器参数，结构学习阶段不再更新。

### 5.3 与扩散过程的关系

编码器将原始时间序列转化为"因果表示"（每个时间点的输出是该时间点及过去信息的非线性摘要），扩散过程在这个表示空间上操作，而非在原始信号上。

---

## 6. 去噪网络（Denoising U-Net）

### 6.1 架构

基于 GraphConv（GCN 风格）的 U-Net 结构：

- **编码器**：多层 GraphConv，逐层提取图上的节点表示
- **解码器**：反向多层 GraphConv，带跳跃连接
- **时间嵌入**：扩散时间步 $t$ 通过 sinusoidal 编码注入到每层

### 6.2 消息传递

去噪网络使用**学习到的邻接矩阵**作为边权重进行消息传递。这是邻接矩阵获得梯度的关键路径：

- 好的邻接矩阵 → 消息传递更有效 → 去噪更准确 → 损失更低
- 差的邻接矩阵 → 错误的消息传递 → 去噪不准 → 损失更高

### 6.3 去噪损失

$$\mathcal{L}_{\text{denoise}} = \text{smooth\_L1}(\hat{x}_0, x_0) + \lambda_{\cos} \cdot (1 - \cos(\hat{x}_0, x_0))$$

其中 $\hat{x}_0$ 是去噪网络的输出，$x_0$ 是真实干净数据，$\cos$ 是余弦相似度。smooth-L1 提供稳健的逐元素匹配，cosine 提供方向对齐。$\lambda_{\cos} = 0.1$。

---

## 7. 训练损失体系

训练包含一个主损失和多个可选辅助损失。

### 7.1 主损失：去噪重建

$$\mathcal{L}_{\text{main}} = \mathcal{L}_{\text{denoise}} + \lambda_{L1} \cdot \|A\|_1$$

$\lambda_{L1}$ 是 L1 稀疏正则系数（通常 0.02），鼓励邻接矩阵稀疏。

**关键发现**：去噪主损失对**因果方向完全无感**。翻转 $A[i,j]$ 与 $A[j,i]$ 不改变去噪质量。原因：

- 噪声引导矩阵是固定的，不受学习邻接矩阵影响
- 去噪是标量重建任务，对称交换两个方向不影响重建误差
- 原始 DDM 假设图结构已知，方向噪声仅用于保留信噪比

因此，**所有方向学习必须依赖辅助损失**。

### 7.2 辅助损失 1：Patel 方向 margin loss

**先验来源**：Patel tau 统计量。对每对脑区 $(i, j)$，$\tau_{ij}$ 衡量 $i$ 激活后 $j$ 也激活的条件概率不对称性。$\tau_{ij} > \tau_{ji}$ 暗示 $i \to j$ 的因果方向。

**损失定义**：

$$\mathcal{L}_{\text{dir}} = \sum_{(i,j)} \max(0, \; m - \text{sign}(\tau_{ij} - \tau_{ji}) \cdot (D_{ij} - D_{ji}))$$

这是一个 margin loss（类似 SVM 的合页损失）：要求学习到的方向 logits 差异 $D_{ij} - D_{ji}$ 与 Patel tau 先验的方向一致，且至少有 $m$ 的 margin。

**Kappa 门控**：只对 Patel kappa 值高于某分位数（如 50%）的节点对施加方向监督。低 kappa 意味着两个脑区共激活的信号弱，其 tau 方向估计不可靠，强行施加可能引入错误監督。

### 7.3 辅助损失 2：Causal-lag main loss（滞后因果重建损失）

**这是 2026 年 3 月引入的最强方向学习信号。**

**核心思想**：如果 $i$ 因果性地影响 $j$，那么 $i$ 的过去应该能预测 $j$ 的未来（Granger 因果性的操作化）。

**具体过程**：

1. 取干净时间序列 $x \in \mathbb{R}^{N \times T}$
2. 构建因果邻接矩阵 $A_{\text{causal}}[i,j]$ 表示 $i \to j$ 的权重
3. 对时间滞后 $\tau$（如 $\tau = 1, 2$），预测每个节点 $j$ 在时刻 $t$ 的值：

$$\hat{x}_j(t) = \sum_i A_{\text{causal}}[i,j] \cdot x_i(t - \tau)$$

即：用所有候选父节点 $i$ 的滞后 $\tau$ 步的信号，按邻接矩阵加权求和，预测目标节点 $j$ 的当前值。

4. 损失 = 预测值与真实值的 smooth-L1 距离（z-score 归一化后）

**为什么这提供方向信号**？训练中同时计算"正向损失"（用 $A_{\text{causal}}$）和"反向损失"（用 $A_{\text{causal}}^\top$）。如果方向正确，正向预测误差应小于反向。梯度会推动邻接矩阵向正确方向调整。

**多滞后聚合**：支持多个 $\tau$ 值（如 1 和 2），各滞后的损失按权重聚合（通常取均值）。

### 7.4 辅助损失 3：Parent entropy（父节点熵）

对每个节点 $j$，计算其入度分布的熵：

$$H_j = -\sum_i p_{ij} \log p_{ij}, \quad p_{ij} = \frac{A[i,j]}{\sum_k A[k,j]}$$

最小化 $\sum_j H_j$ 鼓励每个节点的父节点分布尖锐化（少数主导父节点），而非均匀分布。这是精度导向的剪枝。

### 7.5 辅助损失 4：Parent cap（父节点数上限）

Hinge loss 限制每个节点的有效父节点数：

$$\mathcal{L}_{\text{cap}} = \sum_j \max(0, \; \text{effective\_parents}_j - K_{\max})$$

其中有效父节点数 = 入边权重之和。超过上限 $K_{\max}$ 的部分被惩罚。实验显示被删除的 90%+ 是幻觉边。

### 7.6 辅助损失 5：Ungated symmetry regularization

抑制方向门控之外的残余假阳性：

$$\mathcal{L}_{\text{sym}} = \sum_{(i,j) \notin \text{mask}} |A[i,j] - A[j,i]|$$

对不在硬掩码内的节点对，惩罚其邻接矩阵的不对称性（它们理论上不应该有边，更不应该有方向）。

---

## 8. 梯度路由系统（Gradient Routing）

因为支持分支和方向分支有不同的学习目标，需要精细控制哪个损失函数的梯度更新哪个分支的参数。

### 8.1 三种模式

| 模式                       | 主损失/正则更新                | Causal-lag 损失更新      |
| -------------------------- | ------------------------------ | ------------------------ |
| **legacy**                 | 支持 + 方向（detach epoch 前） | 支持 + 方向              |
| **orthogonal**             | 仅支持                         | 仅方向                   |
| **warmup_then_orthogonal** | 早期：联合；晚期：仅支持       | 早期：联合；晚期：仅方向 |

### 8.2 当前推荐：warmup_then_orthogonal

**前期（epoch < 23）**：联合学习。所有损失的梯度同时流向两个分支。此阶段让模型初步建立合理的结构。

**后期（epoch ≥ 23）**：正交分离。

- 去噪主损失只更新支持分支（决定连接强度）
- Causal-lag 损失只更新方向分支（决定因果方向）

**实现方式**：通过 PyTorch 的 `detach()` 操作，在计算某个损失时将不需要更新的分支从计算图断开。例如：计算去噪损失时，方向门控被 detach，其梯度不回传到方向嵌入。

### 8.3 方向冻结

**epoch 30 后**：彻底冻结方向分支的所有参数（`requires_grad=False`）。方向不再更新，仅支持分支继续微调。

---

## 9. 训练循环概览

### 9.1 可选：编码器预训练

1. 训练时序编码器做自回归预测（$t \to t+1$）
2. 预训练完成后冻结编码器

### 9.2 主训练循环（每个 epoch）

对每个被试的数据 $x \in \mathbb{R}^{N \times T}$：

1. **前向扩散**：用时序编码器得到干净表示 $x_0$，随机采样时间步 $t$，加结构化噪声得到 $x_t$

2. **去噪**：用学习的邻接矩阵构建图，GraphConv/GCN-based U-Net 从 $x_t$ 预测 $x_0$

3. **计算损失栈**：
   - 主损失 = 去噪重建 + L1 正则
   - 辅助损失 = 方向 margin + causal-lag + parent entropy + parent cap + ungated symmetry
   - 各损失按各自权重和调度系数加权求和

4. **梯度路由**：根据当前 epoch，决定梯度流向

5. **参数更新**：支持 "subject" 模式（每个被试独立更新）或 "batch_mean" 模式（累积所有被试再更新一次）

### 9.3 被试间聚合

每个 epoch 遍历所有被试。结构参数（邻接矩阵的嵌入向量）在所有被试间共享——即我们学习一个**跨被试共享的因果结构**，这反映了 fMRI 数据中个体间共享的脑连接模式。

---

## 10. 检查点选择系统（Best-Epoch Selection）

训练过程中邻接矩阵不断变化，最终 epoch 不一定最优。需要一个评分系统选出最佳 epoch。

### 10.1 评分模式

每个 epoch 结束后，对当前邻接矩阵计算一个质量分数。4 种评分公式：

**1. Legacy（原始启发式）**：
$$\text{score} = w_1 \cdot \text{skeleton\_overlap} + w_2 \cdot \text{patel\_agreement} + w_3 \cdot \text{density\_penalty} + w_4 \cdot \text{margin\_bonus}$$

- skeleton overlap：当前骨架与 Patel 先验骨架的重叠度
- patel agreement：方向与 Patel tau 的一致率
- density penalty：边密度偏离先验密度的惩罚
- margin bonus：方向 margin 的强度

**2. Causal-lag composite（当前推荐）**：
$$\text{score} = w_{\text{lag}} \cdot \text{causal\_lag\_gap} + w_{\text{agree}} \cdot \text{soft\_agreement} - w_{\text{margin}} \cdot \text{margin\_penalty}$$

- causal lag gap：正向-反向重建损失差异的 sigmoid 映射
- soft agreement：方向 margin 与 Patel tau 一致的平均程度
- margin penalty：平均 margin 偏弱时的惩罚

**3. Causal-lag entropy composite**：在 composite 基础上加入跨被试稳定性指标。

**4. Causal-lag primary**：causal-lag 信号主导，弱 tiebreak。

### 10.2 守卫规则（Guardrails）

评分前先进行保守筛选，不满足以下条件的 epoch 被降级：

- **骨架保留率**：当前骨架与先验骨架重叠必须超过阈值
- **密度比**：邻接矩阵密度不能偏离先验密度太远

只有通过守卫规则的 epoch 才能成为"受保护候选"。

### 10.3 三层优先级

导出阶段按以下优先级选择 epoch：

1. **Guarded best**：通过守卫规则的最高分 epoch
2. **Score-only fallback**：如果没有 epoch 通过守卫，退回到纯分数最高的 epoch
3. **Final epoch fallback**：如果以上都失败，使用最后一个 epoch

### 10.4 选择器的核心问题

**选择器不影响训练**，它只在训练结束后决定导出哪个 epoch 的邻接矩阵。因此选择器是一个**后处理模块**，可以在训练完成后重新用不同公式重新评估（离线重打分）。

实验发现 legacy 选择器的 proxy 分数与 ground truth F1 相关性为 -0.09（反向相关！），而 causal-lag composite 选择器达到 +0.71，显著改善了导出质量。

---

## 11. 内部约定：Raw vs Causal

算法内部有两种矩阵约定，极易混淆：

| 约定               | 含义                                 | 用途             |
| ------------------ | ------------------------------------ | ---------------- |
| **Raw**（内部）    | $A[i,j]$ 表示 $j \to i$（效果←原因） | GNN 消息传递方向 |
| **Causal**（外部） | $A[i,j]$ 表示 $i \to j$（原因→效果） | 可解释的因果方向 |

转换关系：$A_{\text{causal}} = A_{\text{raw}}^\top$（转置）。

评估 ground truth 时一律使用 causal convention。

---

## 12. 核心实验结论

### 12.1 扩散主损失不提供方向信号

多个诊断实验证实：

- Random init 下所有方向 margin → 0
- 梯度探针显示去噪损失在各向同性噪声下无方向偏好
- 边消融显示去噪器不含可用方向信息

### 12.2 固定硬掩码不可移除

移除 maxgap_kappa 掩码后 F1 从 0.87 崩溃到 0.08。掩码将搜索空间限制在合理范围，是结构学习成功的必要条件。

### 12.3 Causal-lag 是目前最强的方向信号

引入 causal-lag main loss 后：

- Best GT strict-F1 从 0.820 提升到 0.869
- 35/40 个 epoch 显示正向损失 < 反向损失

### 12.4 导出空间方向保留是当前瓶颈

内部方向解正确（方向分支学到了正确方向），但通过 $\text{support} \times \text{direction\_gate}$ 导出后近似对称。即 $A[i,j] \approx A[j,i]$，方向信息被支持权重的对称性"稀释"。

### 12.5 sim3 训练不稳定

sim3（15 节点）数据集在 5 个不同种子上 F1 均值 0.69±0.15，种子 11（原报告最优）是 1.7σ 异常值。

### 12.6 没有单一选择器在所有数据集上最优

- sim3 偏好 entropy_composite 选择器
- sim4 偏好 legacy 选择器
- 两者都倾向选择较晚 epoch

---

## 13. 数据集概况

| 数据集 | 节点数 | Ground Truth 边数 | 真实密度 | 被试数 × 时间点 |
| ------ | ------ | ----------------- | -------- | --------------- |
| fMRI   | 5      | 5                 | 50%      | 50 × 200        |
| sim2   | 10     | 11                | ~24%     | 50 × 200        |
| sim3   | 15     | 18                | ~17%     | 50 × 200        |
| sim4   | 50     | 61                | ~5%      | 50 × 200        |

---

## 14. Patel 先验的 4 个接触点

算法在以下位置使用 Patel 统计量：

1. **固定硬支持掩码**（maxgap_kappa）：用 kappa 值生成二值掩码限制搜索空间 — **不可移除**
2. **噪声引导矩阵**（noise_guide_adj）：用 kappa 值作为噪声构建的邻居权重 — 固定 buffer
3. **方向 margin loss 的目标方向**：用 tau 值决定每条边的目标因果方向 — 可被 causal-lag 替代
4. **方向 margin loss 的置信门控**：用 kappa 值过滤低置信度节点对 — 可被阈值替代

减少先验依赖的路线图：接触点 3、4 可通过增强 causal-lag 损失替代；接触点 2 影响有限；接触点 1 目前不可替代。

---

## 15. 术语对照表

| 术语             | 含义                                               |
| ---------------- | -------------------------------------------------- |
| support          | 支持/骨架，指两节点间是否存在连接（无关方向）      |
| direction gate   | 方向门控，sigmoid(D_ij - D_ji)，编码因果方向       |
| margin           | 方向 logit 差 D_ij - D_ji 的绝对值，越大方向越确定 |
| skeleton         | 无向骨架，忽略方向只看是否有边                     |
| Patel tau        | 条件概率不对称性指标，估计因果方向                 |
| Patel kappa      | 共激活一致性指标，估计连接强度                     |
| causal-lag       | 基于时间滞后的因果预测                             |
| gradient routing | 控制损失梯度流向哪些参数的机制                     |
| guardrail        | 选择器的保守筛选规则                               |
| detach           | 从计算图断开，阻止梯度回传                         |
| epoch quality    | 每个 epoch 的邻接矩阵质量评分                      |
| strict F1        | 有向边的精确率和召回率的调和平均                   |
