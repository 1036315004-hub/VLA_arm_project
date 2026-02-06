"""
Test compatibility between VLAPolicy and VLADataset.

Validates that the policy can process data in the format provided by VLADataset.
"""

import os
import sys
import torch
import torch.nn as nn

# Add the project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


class MockDinov2Output:
    """Mock output from DINOv2 model."""
    def __init__(self, batch_size):
        self.last_hidden_state = torch.randn(batch_size, 257, 384)


class MockDinov2Model(nn.Module):
    """Mock DINOv2 model for testing."""
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.randn(1))
        self.dummy_param.requires_grad = False
    
    def forward(self, pixel_values):
        return MockDinov2Output(pixel_values.shape[0])


def test_dataset_compatibility():
    """Test that VLAPolicy works with VLADataset output format."""
    print("=" * 60)
    print("VLAPolicy-VLADataset Compatibility Test")
    print("=" * 60)
    
    try:
        from src.learning import policy
        
        # Mock the backbone
        original_from_pretrained = policy.Dinov2Model.from_pretrained
        policy.Dinov2Model.from_pretrained = lambda *args, **kwargs: MockDinov2Model()
        
        try:
            from src.learning.policy import VLAPolicy
            
            # Create policy
            vla_policy = VLAPolicy(freeze_backbone=True)
            vla_policy.eval()
            
            # Simulate VLADataset batch output
            # Based on dataset.py, the dataset returns:
            # - images: (T, H, W, C) normalized with ImageNet stats
            # - depth: (T, H, W, 1) 
            # - actions: (T, 7)
            # - lang_embed: (T, 512)
            # where T is trajectory length
            
            # For policy, we need to reshape:
            # - images: (B, C, H, W) for DINOv2
            # - depth: (B, 1, H, W) for depth encoder
            # - lang_embed: (B, 512) for text features
            
            print("\nSimulating VLADataset batch...")
            batch_size = 4
            
            # Simulate dataset output (needs to be transposed/reshaped for policy)
            # In practice, a collate_fn or data loader would handle this
            images_from_dataset = torch.randn(batch_size, 224, 224, 3)  # (B, H, W, C)
            depth_from_dataset = torch.randn(batch_size, 224, 224, 1)   # (B, H, W, 1)
            lang_embed = torch.randn(batch_size, 512)                   # (B, 512)
            
            print(f"Dataset output shapes:")
            print(f"  images: {images_from_dataset.shape} (B, H, W, C)")
            print(f"  depth: {depth_from_dataset.shape} (B, H, W, 1)")
            print(f"  lang_embed: {lang_embed.shape} (B, 512)")
            
            # Transform to policy input format
            # Images: (B, H, W, C) -> (B, C, H, W)
            pixel_values = images_from_dataset.permute(0, 3, 1, 2)
            
            # Depth: (B, H, W, 1) -> (B, 1, H, W)
            depth_values = depth_from_dataset.permute(0, 3, 1, 2)
            
            print(f"\nPolicy input shapes:")
            print(f"  pixel_values: {pixel_values.shape} (B, C, H, W)")
            print(f"  depth_values: {depth_values.shape} (B, 1, H, W)")
            print(f"  text_features: {lang_embed.shape} (B, 512)")
            
            # Forward pass
            with torch.no_grad():
                actions = vla_policy(pixel_values, depth_values, lang_embed)
            
            print(f"\nPolicy output:")
            print(f"  actions: {actions.shape} (B, 7)")
            print(f"  position: {actions[:, :3].shape} (B, 3)")
            print(f"  quaternion: {actions[:, 3:].shape} (B, 4)")
            
            # Verify output
            assert actions.shape == (batch_size, 7), f"Expected (4, 7), got {actions.shape}"
            
            # Verify position is bounded
            position = actions[:, :3]
            assert torch.all(position >= -1.0) and torch.all(position <= 1.0), \
                "Position should be bounded to [-1, 1]"
            
            print(f"\n✓ Compatibility test passed!")
            print(f"✓ VLAPolicy correctly processes VLADataset output")
            print(f"✓ Position bounded: [{position.min().item():.3f}, {position.max().item():.3f}]")
            
            return True
            
        finally:
            policy.Dinov2Model.from_pretrained = original_from_pretrained
            
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_example():
    """Show a complete example of using VLAPolicy with dataset-like data."""
    print("\n" + "=" * 60)
    print("Integration Example")
    print("=" * 60)
    
    try:
        from src.learning import policy
        
        original_from_pretrained = policy.Dinov2Model.from_pretrained
        policy.Dinov2Model.from_pretrained = lambda *args, **kwargs: MockDinov2Model()
        
        try:
            from src.learning.policy import VLAPolicy
            
            print("\n# Example: Using VLAPolicy with batched data")
            print("# " + "-" * 56)
            
            # Initialize policy
            vla_policy = VLAPolicy(freeze_backbone=True)
            vla_policy.eval()
            
            print("""
# Step 1: Load data from VLADataset
# (In practice, use DataLoader with proper collate_fn)
batch = {
    'images': torch.randn(4, 224, 224, 3),   # RGB images
    'depth': torch.randn(4, 224, 224, 1),    # Depth maps
    'lang_embed': torch.randn(4, 512),       # CLIP embeddings
}

# Step 2: Prepare inputs for policy
pixel_values = batch['images'].permute(0, 3, 1, 2)  # (B, 3, 224, 224)
depth_values = batch['depth'].permute(0, 3, 1, 2)   # (B, 1, 224, 224)
text_features = batch['lang_embed']                  # (B, 512)

# Step 3: Forward pass
with torch.no_grad():
    actions = vla_policy(pixel_values, depth_values, text_features)

# Step 4: Extract position and quaternion
position = actions[:, :3]      # (B, 3) - bounded to [-1, 1]
quaternion = actions[:, 3:]    # (B, 4) - needs normalization

# Step 5: (Optional) Normalize quaternion for deployment
quaternion_norm = quaternion / quaternion.norm(dim=1, keepdim=True)
""")
            
            # Actually run the example
            batch = {
                'images': torch.randn(4, 224, 224, 3),
                'depth': torch.randn(4, 224, 224, 1),
                'lang_embed': torch.randn(4, 512),
            }
            
            pixel_values = batch['images'].permute(0, 3, 1, 2)
            depth_values = batch['depth'].permute(0, 3, 1, 2)
            text_features = batch['lang_embed']
            
            with torch.no_grad():
                actions = vla_policy(pixel_values, depth_values, text_features)
            
            position = actions[:, :3]
            quaternion = actions[:, 3:]
            quaternion_norm = quaternion / quaternion.norm(dim=1, keepdim=True)
            
            print(f"\nExample output:")
            print(f"  Actions shape: {actions.shape}")
            print(f"  Position range: [{position.min():.3f}, {position.max():.3f}]")
            print(f"  Quaternion norm (before): {quaternion.norm(dim=1).mean():.3f}")
            print(f"  Quaternion norm (after):  {quaternion_norm.norm(dim=1).mean():.3f}")
            
            print(f"\n✓ Integration example successful!")
            
            return True
            
        finally:
            policy.Dinov2Model.from_pretrained = original_from_pretrained
            
    except Exception as e:
        print(f"✗ Integration example failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VLAPolicy-VLADataset Integration Tests")
    print("=" * 60)
    
    test1 = test_dataset_compatibility()
    test2 = test_integration_example()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'✓ PASS' if test1 else '✗ FAIL'}: Dataset Compatibility")
    print(f"{'✓ PASS' if test2 else '✗ FAIL'}: Integration Example")
    
    if test1 and test2:
        print("\n🎉 All integration tests passed!")
    
    sys.exit(0 if (test1 and test2) else 1)
