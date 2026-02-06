"""
Test suite for VLAPolicy implementation.

Tests the complete forward pass with proper batch dimensions and validates
compatibility with VLADataset outputs.
"""

import os
import sys
import torch
import torch.nn as nn

# Add the project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.learning.policy import VLAPolicy


def test_policy_initialization():
    """Test that the VLAPolicy initializes correctly with proper architecture."""
    print("=" * 60)
    print("Test 1: Policy Initialization")
    print("=" * 60)
    
    try:
        policy = VLAPolicy(freeze_backbone=True)
        
        # Check that all components exist
        assert hasattr(policy, 'visual_backbone'), "Missing visual_backbone"
        assert hasattr(policy, 'depth_encoder'), "Missing depth_encoder"
        assert hasattr(policy, 'projector'), "Missing projector"
        assert hasattr(policy, 'pos_head'), "Missing pos_head"
        assert hasattr(policy, 'quat_head'), "Missing quat_head"
        
        # Check depth encoder architecture
        depth_layers = [m for m in policy.depth_encoder if isinstance(m, nn.Conv2d)]
        assert len(depth_layers) == 3, f"Expected 3 Conv2d layers, got {len(depth_layers)}"
        
        # Check projector architecture (should have 3 Linear layers)
        projector_linear = [m for m in policy.projector if isinstance(m, nn.Linear)]
        assert len(projector_linear) == 3, f"Expected 3 Linear layers in projector, got {len(projector_linear)}"
        
        # Check LayerNorm in projector
        projector_ln = [m for m in policy.projector if isinstance(m, nn.LayerNorm)]
        assert len(projector_ln) == 2, f"Expected 2 LayerNorm layers in projector, got {len(projector_ln)}"
        
        # Check that visual backbone is frozen
        for param in policy.visual_backbone.parameters():
            assert not param.requires_grad, "Visual backbone should be frozen"
        
        # Check that other components are trainable
        for param in policy.depth_encoder.parameters():
            assert param.requires_grad, "Depth encoder should be trainable"
        
        for param in policy.projector.parameters():
            assert param.requires_grad, "Projector should be trainable"
        
        print("✓ Policy initialized correctly")
        print(f"✓ Visual backbone frozen: {not next(policy.visual_backbone.parameters()).requires_grad}")
        print(f"✓ Depth encoder has {len(depth_layers)} Conv2d layers")
        print(f"✓ Projector has {len(projector_linear)} Linear layers and {len(projector_ln)} LayerNorm layers")
        print(f"✓ Position head output: 3D")
        print(f"✓ Quaternion head output: 4D")
        
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with correct input shapes matching VLADataset."""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass with Batch Dimensions")
    print("=" * 60)
    
    try:
        policy = VLAPolicy(freeze_backbone=True)
        policy.eval()  # Set to eval mode
        
        batch_size = 2
        
        # Create dummy inputs matching expected shapes from VLADataset
        # Images: (B, 3, 224, 224) - normalized RGB
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # Depth: (B, 1, 224, 224) - single channel depth map
        depth_values = torch.randn(batch_size, 1, 224, 224)
        
        # Text features: (B, 512) - pre-computed CLIP embeddings
        text_features = torch.randn(batch_size, 512)
        
        print(f"Input shapes:")
        print(f"  pixel_values: {pixel_values.shape}")
        print(f"  depth_values: {depth_values.shape}")
        print(f"  text_features: {text_features.shape}")
        
        # Forward pass
        with torch.no_grad():
            actions = policy(pixel_values, depth_values, text_features)
        
        # Validate output shape
        assert actions.shape == (batch_size, 7), f"Expected shape ({batch_size}, 7), got {actions.shape}"
        
        # Validate position bounds (should be in [-1, 1] due to tanh)
        position = actions[:, :3]
        assert torch.all(position >= -1.0) and torch.all(position <= 1.0), \
            "Position values should be bounded in [-1, 1]"
        
        # Quaternion can be unbounded (will be normalized during inference)
        quaternion = actions[:, 3:]
        
        print(f"\n✓ Forward pass successful")
        print(f"✓ Output shape: {actions.shape}")
        print(f"✓ Position range: [{position.min().item():.3f}, {position.max().item():.3f}]")
        print(f"✓ Quaternion range: [{quaternion.min().item():.3f}, {quaternion.max().item():.3f}]")
        print(f"\nSample output:")
        print(f"  Action 0: pos={position[0].numpy()}, quat={quaternion[0].numpy()}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow properly through trainable components."""
    print("\n" + "=" * 60)
    print("Test 3: Gradient Flow")
    print("=" * 60)
    
    try:
        policy = VLAPolicy(freeze_backbone=True)
        policy.train()
        
        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        depth_values = torch.randn(batch_size, 1, 224, 224)
        text_features = torch.randn(batch_size, 512)
        
        # Forward pass
        actions = policy(pixel_values, depth_values, text_features)
        
        # Create dummy loss and backpropagate
        loss = actions.mean()
        loss.backward()
        
        # Check gradients in trainable components
        depth_encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in policy.depth_encoder.parameters()
        )
        projector_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in policy.projector.parameters()
        )
        pos_head_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in policy.pos_head.parameters()
        )
        quat_head_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in policy.quat_head.parameters()
        )
        
        # Check that frozen backbone has no gradients
        backbone_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in policy.visual_backbone.parameters()
        )
        
        assert depth_encoder_has_grad, "Depth encoder should have gradients"
        assert projector_has_grad, "Projector should have gradients"
        assert pos_head_has_grad, "Position head should have gradients"
        assert quat_head_has_grad, "Quaternion head should have gradients"
        assert not backbone_has_grad, "Visual backbone should NOT have gradients (frozen)"
        
        print("✓ Gradients flow correctly through trainable components")
        print("✓ Frozen backbone has no gradients")
        print("✓ Depth encoder has gradients")
        print("✓ Projector has gradients")
        print("✓ Action heads have gradients")
        
        return True
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_batch():
    """Test with batch size of 1 to ensure no dimension issues."""
    print("\n" + "=" * 60)
    print("Test 4: Single Batch (B=1)")
    print("=" * 60)
    
    try:
        policy = VLAPolicy(freeze_backbone=True)
        policy.eval()
        
        # Single sample
        pixel_values = torch.randn(1, 3, 224, 224)
        depth_values = torch.randn(1, 1, 224, 224)
        text_features = torch.randn(1, 512)
        
        with torch.no_grad():
            actions = policy(pixel_values, depth_values, text_features)
        
        assert actions.shape == (1, 7), f"Expected shape (1, 7), got {actions.shape}"
        
        print("✓ Single batch forward pass successful")
        print(f"✓ Output shape: {actions.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Single batch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_count():
    """Display parameter counts for each component."""
    print("\n" + "=" * 60)
    print("Test 5: Parameter Count Analysis")
    print("=" * 60)
    
    try:
        policy = VLAPolicy(freeze_backbone=True)
        
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        def count_trainable_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        total_params = count_params(policy)
        trainable_params = count_trainable_params(policy)
        
        backbone_params = count_params(policy.visual_backbone)
        depth_params = count_params(policy.depth_encoder)
        projector_params = count_params(policy.projector)
        pos_head_params = count_params(policy.pos_head)
        quat_head_params = count_params(policy.quat_head)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"\nComponent breakdown:")
        print(f"  Visual backbone (frozen): {backbone_params:,}")
        print(f"  Depth encoder: {depth_params:,}")
        print(f"  Projector: {projector_params:,}")
        print(f"  Position head: {pos_head_params:,}")
        print(f"  Quaternion head: {quat_head_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Parameter count test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("VLAPolicy Test Suite")
    print("=" * 60)
    
    tests = [
        ("Initialization", test_policy_initialization),
        ("Forward Pass", test_forward_pass),
        ("Gradient Flow", test_gradient_flow),
        ("Single Batch", test_single_batch),
        ("Parameter Count", test_parameter_count),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
