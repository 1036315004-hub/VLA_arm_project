import os
import torch
from torch import nn
from transformers import Dinov2Model

# Get the project root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

class VLAPolicy(nn.Module):
    """
    VLA policy skeleton with a frozen DINOv2 backbone, a projector, and an action head.

    - Visual backbone: DINOv2 ViT-S/14 (with registers)
    - Projector: merges visual + text (e.g., CLIP) features
    - Action head: outputs 7D action vector
    """

    def __init__(
        self,
        model_id="facebook/dinov2-small-with-registers",
        cache_subdir="huggingface",
        visual_dim=384,
        text_dim=512,
        proj_dim=512,
        action_dim=7,
        freeze_backbone=True
    ):
        super().__init__()

        checkpoint_dir = os.path.join(PROJECT_ROOT, "models", "pretrained", cache_subdir)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # DINOv2 backbone (pretrained)
        self.visual_backbone = Dinov2Model.from_pretrained(
            model_id,
            cache_dir=checkpoint_dir,
            local_files_only=False,
            from_pt=True
        )

        if freeze_backbone:
            for param in self.visual_backbone.parameters():
                param.requires_grad = False

        # Projector to fuse visual + text features
        self.projector = nn.Sequential(
            nn.Linear(visual_dim + text_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # Action head (7D action output)
        self.action_head = nn.Linear(proj_dim, action_dim)

    def forward(self, pixel_values, text_features):
        raise NotImplementedError("Forward pass not implemented yet.")
