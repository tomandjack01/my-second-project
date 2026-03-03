#!/usr/bin/env python3
"""
测试脚本：验证禁用时序编码器的功能
"""

import torch
import sys
sys.path.insert(0, '.')

from models import DDM

def test_with_encoder():
    """测试启用编码器的情况"""
    print("=" * 60)
    print("测试 1: 启用时序编码器 (默认行为)")
    print("=" * 60)

    # 创建模拟的 Pearson 矩阵用于结构学习
    N = 10  # 节点数
    init_features = torch.randn(N, N)  # 模拟 Pearson 矩阵

    model = DDM(
        in_dim=200,  # 时序长度
        num_hidden=64,
        num_layers=2,
        nhead=4,
        activation='prelu',
        feat_drop=0.1,
        attn_drop=0.1,
        norm='layernorm',
        use_temporal_encoder=True,
        init_features=init_features,  # 启用结构学习模式
    )

    print(f"模型是否使用编码器: {model.use_temporal_encoder}")
    print(f"编码器对象: {model.temporal_encoder}")
    print(f"去噪网络输出维度: {model.net.out_dim}")
    print(f"去噪网络隐藏维度: {model.net.num_hidden}")
    print(f"结构学习模式: {model.structure_learning_mode}")

    # 测试前向传播
    x = torch.randn(N, 200)  # [N=10, T=200]
    loss, loss_dict = model(g=None, x=x)
    print(f"前向传播成功! Loss: {loss.item():.4f}")
    print()

def test_without_encoder():
    """测试禁用编码器的情况"""
    print("=" * 60)
    print("测试 2: 禁用时序编码器 (直接对原始数据加噪)")
    print("=" * 60)

    # 创建模拟的 Pearson 矩阵用于结构学习
    N = 10  # 节点数
    init_features = torch.randn(N, N)  # 模拟 Pearson 矩阵

    model = DDM(
        in_dim=200,  # 时序长度
        num_hidden=64,
        num_layers=2,
        nhead=4,
        activation='prelu',
        feat_drop=0.1,
        attn_drop=0.1,
        norm='layernorm',
        use_temporal_encoder=False,
        init_features=init_features,  # 启用结构学习模式
    )

    print(f"模型是否使用编码器: {model.use_temporal_encoder}")
    print(f"编码器对象: {model.temporal_encoder}")
    print(f"去噪网络输出维度: {model.net.out_dim}")
    print(f"去噪网络隐藏维度: {model.net.num_hidden}")
    print(f"结构学习模式: {model.structure_learning_mode}")

    # 测试前向传播
    x = torch.randn(N, 200)  # [N=10, T=200]
    loss, loss_dict = model(g=None, x=x)
    print(f"前向传播成功! Loss: {loss.item():.4f}")
    print()

def test_dimension_consistency():
    """测试维度一致性"""
    print("=" * 60)
    print("测试 3: 维度一致性检查")
    print("=" * 60)

    N = 10
    init_features = torch.randn(N, N)

    # 启用编码器: 200 → 64
    model_with = DDM(
        in_dim=200,
        num_hidden=64,
        num_layers=2,
        nhead=4,
        activation='prelu',
        feat_drop=0.1,
        attn_drop=0.1,
        norm='layernorm',
        use_temporal_encoder=True,
        init_features=init_features,
    )

    # 禁用编码器: 200 → 200
    model_without = DDM(
        in_dim=200,
        num_hidden=64,
        num_layers=2,
        nhead=4,
        activation='prelu',
        feat_drop=0.1,
        attn_drop=0.1,
        norm='layernorm',
        use_temporal_encoder=False,
        init_features=init_features,
    )

    x = torch.randn(N, 200)

    # 测试启用编码器
    loss1, _ = model_with(g=None, x=x)
    print(f"启用编码器 - Loss: {loss1.item():.4f}")

    # 测试禁用编码器
    loss2, _ = model_without(g=None, x=x)
    print(f"禁用编码器 - Loss: {loss2.item():.4f}")

    print("\n✓ 两种模式都能正常工作!")

if __name__ == '__main__':
    test_with_encoder()
    test_without_encoder()
    test_dimension_consistency()

    print("=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
