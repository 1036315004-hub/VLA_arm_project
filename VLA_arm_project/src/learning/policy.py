import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Dinov2Model

# Get the project root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

class VLAPolicy(nn.Module):
    """
    Production-ready Vision-Language-Action (VLA) policy with multimodal fusion.

    Architecture:
    - Visual backbone: Frozen DINOv2 ViT-S/14 (with registers) for RGB features (384-dim)
    - Depth encoder: Lightweight CNN for depth map processing (128-dim output)
    - Text features: Pre-computed CLIP embeddings (512-dim)
    - Fusion: Concatenates RGB + Depth + Text features (1024-dim total)
    - Policy backbone: 3-layer MLP projector with LayerNorm and ReLU (512-dim output)
    - Action heads:
        - Position head: Outputs 3D position with tanh activation (bounded to [-1, 1])
        - Quaternion head: Outputs 4D quaternion (unbounded, normalized during inference)
    
    The tanh activation on position output ensures bounded actions suitable for
    robot workspace constraints. Quaternion outputs are kept unbounded to allow
    proper gradient flow during training and are normalized post-hoc.
    """

    def __init__(
        self,
        model_id="facebook/dinov2-small-with-registers",
        cache_subdir="huggingface",
        visual_dim=384,
        depth_dim=128,
        text_dim=512,
        proj_dim=512,
        freeze_backbone=True
    ):
        """
        Initialize the VLA policy with multimodal encoders and action heads.

        Args:
            model_id: HuggingFace model ID for DINOv2 backbone
            cache_subdir: Subdirectory for caching pretrained models
            visual_dim: Dimension of RGB visual features from DINOv2 (384 for ViT-S)
            depth_dim: Dimension of depth features from depth encoder (128)
            text_dim: Dimension of text features from CLIP (512)
            proj_dim: Dimension of policy backbone output (512)
            freeze_backbone: Whether to freeze DINOv2 backbone weights
        """
        super().__init__()

        checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "pretrained", cache_subdir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # DINOv2 backbone (pretrained) - Frozen for efficiency and stability
        self.visual_backbone = Dinov2Model.from_pretrained(
            model_id,
            cache_dir=checkpoint_dir,
            local_files_only=False,
            from_pt=True
        )

        if freeze_backbone:
            for param in self.visual_backbone.parameters():
                param.requires_grad = False

        # Depth encoder: Lightweight CNN for processing 224x224x1 depth maps
        # Architecture: 3 Conv2d layers with stride 2 + Global Average Pooling
        # Input: (B, 1, 224, 224) -> Output: (B, 128)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),  # (B, 32, 112, 112)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # (B, 64, 56, 56)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, 28, 28)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # (B, 128, 1, 1) -> will be flattened to (B, 128)
        )

        # Projector: 3-layer MLP to fuse multimodal features
        # Concatenated input: RGB (384) + Depth (128) + Text (512) = 1024
        # Output: 512-dimensional policy features
        fused_dim = visual_dim + depth_dim + text_dim  # 1024
        self.projector = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # Action heads: Separate heads for position and quaternion
        # Position: 3D coordinates (x, y, z) with tanh for bounded output
        self.pos_head = nn.Linear(proj_dim, 3)
        
        # Quaternion: 4D orientation (qw, qx, qy, qz) - normalized post-hoc
        self.quat_head = nn.Linear(proj_dim, 4)

    def forward(self, pixel_values, depth_values, text_features):
        """
        Forward pass through the VLA policy.

        Args:
            pixel_values: RGB images, shape (B, 3, 224, 224)
            depth_values: Depth maps, shape (B, 1, 224, 224) 
            text_features: Pre-computed CLIP text embeddings, shape (B, 512)

        Returns:
            actions: 7D action vector (position + quaternion), shape (B, 7)
                - First 3 dimensions: position (x, y, z) bounded to [-1, 1]
                - Last 4 dimensions: quaternion (qw, qx, qy, qz) unbounded
        """
        batch_size = pixel_values.shape[0]

        # === RGB Path ===
        # Pass through frozen DINOv2 backbone
        # Output: last_hidden_state with shape (B, num_patches + 1, 384)
        # The first token (index 0) is the CLS token
        visual_outputs = self.visual_backbone(pixel_values)
        rgb_features = visual_outputs.last_hidden_state[:, 0, :]  # (B, 384) - CLS token

        # === Depth Path ===
        # Pass through depth encoder CNN
        # Input: (B, 1, 224, 224) -> Output: (B, 128, 1, 1)
        depth_features = self.depth_encoder(depth_values)  # (B, 128, 1, 1)
        depth_features = depth_features.view(batch_size, -1)  # (B, 128)

        # === Fusion ===
        # Concatenate RGB, depth, and text features
        # Total: 384 + 128 + 512 = 1024 dimensions
        fused_features = torch.cat([rgb_features, depth_features, text_features], dim=1)  # (B, 1024)

        # === Policy Backbone ===
        # Pass through projector MLP
        policy_features = self.projector(fused_features)  # (B, 512)

        # === Action Output ===
        # Position: 3D coordinates with tanh activation for bounded output [-1, 1]
        # This ensures the predicted positions stay within a normalized workspace
        position = torch.tanh(self.pos_head(policy_features))  # (B, 3)

        # Quaternion: 4D orientation representation (unbounded during training)
        # Will be normalized to unit quaternion during inference/deployment
        quaternion = self.quat_head(policy_features)  # (B, 4)

        # Concatenate position and quaternion to form 7D action vector
        actions = torch.cat([position, quaternion], dim=1)  # (B, 7)

        return actions
