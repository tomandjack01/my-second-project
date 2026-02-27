#!/usr/bin/env python3
"""Test script for DDM with Temporal Encoder."""

import torch
from models import DDM

def test_ddm():
    print("=" * 60)
    print("Testing DDM with Temporal Encoder")
    print("=" * 60)
    
    # Test initialization
    print("\n1. Creating model...")
    model = DDM(
        in_dim=200,
        num_hidden=64,
        num_layers=2,
        nhead=4,
        activation='prelu',
        feat_drop=0.1,
        attn_drop=0.1,
        norm='layernorm',
        init_features=torch.randn(15, 15)  # 15 brain regions
    )
    print("   Model created successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    temporal_params = sum(p.numel() for p in model.temporal_encoder.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Temporal encoder parameters: {temporal_params:,}")
    
    # Test temporal encoder alone
    print("\n2. Testing temporal encoder...")
    x_raw = torch.randn(15, 200)  # [Nodes, TimePoints] - unbatched
    x_encoded = model.temporal_encoder(x_raw)
    print(f"   Input shape:  {list(x_raw.shape)}")
    print(f"   Output shape: {list(x_encoded.shape)}")
    
    # Test forward pass with unbatched input (as used in training)
    print("\n3. Testing forward pass (unbatched)...")
    try:
        loss, loss_dict = model(None, x_raw)
        print(f"   Forward pass successful!")
        print(f"   Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test embed
    print("\n4. Testing embed...")
    try:
        hidden = model.embed(None, x_raw, T=500)
        print(f"   Embed output shape: {list(hidden.shape)}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_ddm()
