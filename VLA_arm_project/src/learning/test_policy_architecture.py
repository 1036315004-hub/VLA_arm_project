"""
Lightweight test suite for VLAPolicy architecture without downloading models.

Tests the policy architecture by mocking the visual backbone to avoid network dependencies.
"""

import os
import sys
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

# Add the project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


class MockDinov2Output:
    """Mock output from DINOv2 model."""
    def __init__(self, batch_size):
        # DINOv2 ViT-S returns 384-dim features
        # Output includes CLS token at position 0
        self.last_hidden_state = torch.randn(batch_size, 257, 384)  # 256 patches + 1 CLS


class MockDinov2Model(nn.Module):
    """Mock DINOv2 model for testing without network access."""
    def __init__(self):
        super().__init__()
        # Add a dummy parameter so parameters() works
        self.dummy_param = nn.Parameter(torch.randn(1))
        self.dummy_param.requires_grad = False
    
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        return MockDinov2Output(batch_size)


def create_mock_dinov2():
    """Create a mock DINOv2 model for testing."""
    return MockDinov2Model()


def test_architecture_components():
    """Test that all required components exist with correct architectures."""
    print("=" * 60)
    print("Test 1: Architecture Components")
    print("=" * 60)
    
    try:
        # Import and patch
        from src.learning import policy
        
        # Patch the Dinov2Model.from_pretrained method
        original_from_pretrained = policy.Dinov2Model.from_pretrained
        policy.Dinov2Model.from_pretrained = lambda *args, **kwargs: create_mock_dinov2()
        
        try:
            from src.learning.policy import VLAPolicy
            
            # Create policy with mocked backbone
            vla_policy = VLAPolicy(freeze_backbone=True)
            
            # Check that all components exist
            assert hasattr(vla_policy, 'visual_backbone'), "Missing visual_backbone"
            assert hasattr(vla_policy, 'depth_encoder'), "Missing depth_encoder"
            assert hasattr(vla_policy, 'projector'), "Missing projector"
            assert hasattr(vla_policy, 'pos_head'), "Missing pos_head"
            assert hasattr(vla_policy, 'quat_head'), "Missing quat_head"
            
            # Check depth encoder architecture
            depth_layers = [m for m in vla_policy.depth_encoder if isinstance(m, nn.Conv2d)]
            assert len(depth_layers) == 3, f"Expected 3 Conv2d layers, got {len(depth_layers)}"
            
            # Verify Conv2d layer configurations
            assert depth_layers[0].in_channels == 1, "First Conv2d should accept 1 channel"
            assert depth_layers[0].out_channels == 32, "First Conv2d should output 32 channels"
            assert depth_layers[0].stride == (2, 2), "First Conv2d should have stride 2"
            
            assert depth_layers[1].out_channels == 64, "Second Conv2d should output 64 channels"
            assert depth_layers[1].stride == (2, 2), "Second Conv2d should have stride 2"
            
            assert depth_layers[2].out_channels == 128, "Third Conv2d should output 128 channels"
            assert depth_layers[2].stride == (2, 2), "Third Conv2d should have stride 2"
            
            # Check for Global Average Pooling
            has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in vla_policy.depth_encoder)
            assert has_gap, "Depth encoder should have AdaptiveAvgPool2d (Global Average Pooling)"
            
            # Check projector architecture (should have 3 Linear layers)
            projector_linear = [m for m in vla_policy.projector if isinstance(m, nn.Linear)]
            assert len(projector_linear) == 3, f"Expected 3 Linear layers in projector, got {len(projector_linear)}"
            
            # Check input/output dimensions of projector
            assert projector_linear[0].in_features == 1024, "Projector input should be 1024 (384+128+512)"
            assert projector_linear[0].out_features == 512, "Projector first layer output should be 512"
            assert projector_linear[-1].out_features == 512, "Projector final output should be 512"
            
            # Check LayerNorm in projector
            projector_ln = [m for m in vla_policy.projector if isinstance(m, nn.LayerNorm)]
            assert len(projector_ln) == 2, f"Expected 2 LayerNorm layers in projector, got {len(projector_ln)}"
            
            # Check ReLU in projector
            projector_relu = [m for m in vla_policy.projector if isinstance(m, nn.ReLU)]
            assert len(projector_relu) >= 2, f"Expected at least 2 ReLU layers in projector, got {len(projector_relu)}"
            
            # Check action heads
            assert isinstance(vla_policy.pos_head, nn.Linear), "pos_head should be Linear layer"
            assert isinstance(vla_policy.quat_head, nn.Linear), "quat_head should be Linear layer"
            assert vla_policy.pos_head.in_features == 512, "pos_head input should be 512"
            assert vla_policy.pos_head.out_features == 3, "pos_head output should be 3"
            assert vla_policy.quat_head.in_features == 512, "quat_head input should be 512"
            assert vla_policy.quat_head.out_features == 4, "quat_head output should be 4"
            
            print("✓ All components exist with correct architecture")
            print(f"✓ Depth encoder: 3 Conv2d layers with stride 2, output 128 channels")
            print(f"✓ Depth encoder: Has Global Average Pooling")
            print(f"✓ Projector: 3 Linear layers (1024->512->512->512)")
            print(f"✓ Projector: 2 LayerNorm + 2 ReLU layers")
            print(f"✓ Position head: 512 -> 3")
            print(f"✓ Quaternion head: 512 -> 4")
            
            return True
        finally:
            # Restore original method
            policy.Dinov2Model.from_pretrained = original_from_pretrained
            
    except Exception as e:
        print(f"✗ Architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_shapes():
    """Test forward pass produces correct output shapes."""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass Shapes")
    print("=" * 60)
    
    try:
        from src.learning import policy
        
        # Patch the Dinov2Model.from_pretrained method
        original_from_pretrained = policy.Dinov2Model.from_pretrained
        policy.Dinov2Model.from_pretrained = lambda *args, **kwargs: create_mock_dinov2()
        
        try:
            from src.learning.policy import VLAPolicy
            
            vla_policy = VLAPolicy(freeze_backbone=True)
            vla_policy.eval()
            
            batch_size = 4
            
            # Create dummy inputs
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            depth_values = torch.randn(batch_size, 1, 224, 224)
            text_features = torch.randn(batch_size, 512)
            
            print(f"Input shapes:")
            print(f"  pixel_values: {pixel_values.shape}")
            print(f"  depth_values: {depth_values.shape}")
            print(f"  text_features: {text_features.shape}")
            
            # Forward pass
            with torch.no_grad():
                actions = vla_policy(pixel_values, depth_values, text_features)
            
            # Validate output shape
            assert actions.shape == (batch_size, 7), f"Expected shape ({batch_size}, 7), got {actions.shape}"
            
            # Validate position bounds (should be in [-1, 1] due to tanh)
            position = actions[:, :3]
            assert torch.all(position >= -1.0) and torch.all(position <= 1.0), \
                "Position values should be bounded in [-1, 1]"
            
            # Quaternion can be unbounded
            quaternion = actions[:, 3:]
            
            print(f"\n✓ Forward pass successful")
            print(f"✓ Output shape: {actions.shape}")
            print(f"✓ Position bounded: [{position.min().item():.3f}, {position.max().item():.3f}]")
            print(f"✓ Quaternion unbounded: [{quaternion.min().item():.3f}, {quaternion.max().item():.3f}]")
            
            return True
        finally:
            policy.Dinov2Model.from_pretrained = original_from_pretrained
            
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_dimensions():
    """Test with different batch sizes."""
    print("\n" + "=" * 60)
    print("Test 3: Different Batch Sizes")
    print("=" * 60)
    
    try:
        from src.learning import policy
        
        original_from_pretrained = policy.Dinov2Model.from_pretrained
        policy.Dinov2Model.from_pretrained = lambda *args, **kwargs: create_mock_dinov2()
        
        try:
            from src.learning.policy import VLAPolicy
            
            vla_policy = VLAPolicy(freeze_backbone=True)
            vla_policy.eval()
            
            test_batch_sizes = [1, 2, 8, 16]
            
            for batch_size in test_batch_sizes:
                pixel_values = torch.randn(batch_size, 3, 224, 224)
                depth_values = torch.randn(batch_size, 1, 224, 224)
                text_features = torch.randn(batch_size, 512)
                
                with torch.no_grad():
                    actions = vla_policy(pixel_values, depth_values, text_features)
                
                assert actions.shape == (batch_size, 7), \
                    f"Batch {batch_size}: Expected shape ({batch_size}, 7), got {actions.shape}"
                
                print(f"✓ Batch size {batch_size:2d}: Output shape {actions.shape}")
            
            return True
        finally:
            policy.Dinov2Model.from_pretrained = original_from_pretrained
            
    except Exception as e:
        print(f"✗ Batch dimension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainable_parameters():
    """Test that only the right components are trainable."""
    print("\n" + "=" * 60)
    print("Test 4: Trainable Parameters")
    print("=" * 60)
    
    try:
        from src.learning import policy
        
        original_from_pretrained = policy.Dinov2Model.from_pretrained
        policy.Dinov2Model.from_pretrained = lambda *args, **kwargs: create_mock_dinov2()
        
        try:
            from src.learning.policy import VLAPolicy
            
            vla_policy = VLAPolicy(freeze_backbone=True)
            
            # Check that depth encoder is trainable
            depth_params = list(vla_policy.depth_encoder.parameters())
            assert len(depth_params) > 0, "Depth encoder should have parameters"
            assert all(p.requires_grad for p in depth_params), "Depth encoder params should be trainable"
            
            # Check that projector is trainable
            proj_params = list(vla_policy.projector.parameters())
            assert len(proj_params) > 0, "Projector should have parameters"
            assert all(p.requires_grad for p in proj_params), "Projector params should be trainable"
            
            # Check that action heads are trainable
            pos_params = list(vla_policy.pos_head.parameters())
            quat_params = list(vla_policy.quat_head.parameters())
            assert all(p.requires_grad for p in pos_params), "Position head params should be trainable"
            assert all(p.requires_grad for p in quat_params), "Quaternion head params should be trainable"
            
            # Count parameters
            def count_params(module):
                return sum(p.numel() for p in module.parameters())
            
            depth_count = count_params(vla_policy.depth_encoder)
            proj_count = count_params(vla_policy.projector)
            pos_count = count_params(vla_policy.pos_head)
            quat_count = count_params(vla_policy.quat_head)
            
            print(f"✓ Depth encoder: {depth_count:,} trainable parameters")
            print(f"✓ Projector: {proj_count:,} trainable parameters")
            print(f"✓ Position head: {pos_count:,} trainable parameters")
            print(f"✓ Quaternion head: {quat_count:,} trainable parameters")
            print(f"✓ Total trainable (excluding backbone): {depth_count + proj_count + pos_count + quat_count:,}")
            
            return True
        finally:
            policy.Dinov2Model.from_pretrained = original_from_pretrained
            
    except Exception as e:
        print(f"✗ Trainable parameters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_computation():
    """Test that gradients can be computed."""
    print("\n" + "=" * 60)
    print("Test 5: Gradient Computation")
    print("=" * 60)
    
    try:
        from src.learning import policy
        
        original_from_pretrained = policy.Dinov2Model.from_pretrained
        policy.Dinov2Model.from_pretrained = lambda *args, **kwargs: create_mock_dinov2()
        
        try:
            from src.learning.policy import VLAPolicy
            
            vla_policy = VLAPolicy(freeze_backbone=True)
            vla_policy.train()
            
            # Create dummy inputs
            batch_size = 2
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            depth_values = torch.randn(batch_size, 1, 224, 224)
            text_features = torch.randn(batch_size, 512)
            
            # Forward pass
            actions = vla_policy(pixel_values, depth_values, text_features)
            
            # Compute dummy loss and backpropagate
            loss = actions.mean()
            loss.backward()
            
            # Check that gradients exist for trainable components
            depth_has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 
                for p in vla_policy.depth_encoder.parameters() if p.requires_grad
            )
            proj_has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 
                for p in vla_policy.projector.parameters() if p.requires_grad
            )
            pos_has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 
                for p in vla_policy.pos_head.parameters() if p.requires_grad
            )
            quat_has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0 
                for p in vla_policy.quat_head.parameters() if p.requires_grad
            )
            
            assert depth_has_grad, "Depth encoder should have gradients"
            assert proj_has_grad, "Projector should have gradients"
            assert pos_has_grad, "Position head should have gradients"
            assert quat_has_grad, "Quaternion head should have gradients"
            
            print("✓ Gradients computed successfully")
            print("✓ Depth encoder has gradients")
            print("✓ Projector has gradients")
            print("✓ Position head has gradients")
            print("✓ Quaternion head has gradients")
            
            return True
        finally:
            policy.Dinov2Model.from_pretrained = original_from_pretrained
            
    except Exception as e:
        print(f"✗ Gradient computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("VLAPolicy Architecture Test Suite")
    print("(Testing without downloading models)")
    print("=" * 60)
    
    tests = [
        ("Architecture Components", test_architecture_components),
        ("Forward Pass Shapes", test_forward_pass_shapes),
        ("Batch Dimensions", test_batch_dimensions),
        ("Trainable Parameters", test_trainable_parameters),
        ("Gradient Computation", test_gradient_computation),
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
    
    if passed == total:
        print("\n🎉 All tests passed! VLAPolicy implementation is correct.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
