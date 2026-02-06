#!/usr/bin/env python3
"""
Training script for VLAPolicy model.

Trains the Vision-Language-Action policy using HDF5 data from VLADataset.
Supports checkpoint resumption, weighted loss with keyframe reinforcement,
and automatic best-model saving based on validation loss.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

# Add project root to path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.learning.dataset import VLADataset
from src.learning.policy import VLAPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLAPolicy model")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "processed"),
        help="Path to processed HDF5 data directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "trained"),
        help="Directory to save trained model weights",
    )
    return parser.parse_args()


def compute_loss(pred_actions, gt_actions, is_keyframe, pos_weight=5.0, quat_weight=1.0):
    """
    Compute weighted MSE loss with keyframe reinforcement.

    Args:
        pred_actions: Predicted 7D actions (B, 7) — first 3 pos, last 4 quat.
        gt_actions: Ground-truth 7D actions (B, 7).
        is_keyframe: Boolean tensor (B,) indicating keyframe samples.
        pos_weight: Weight multiplier for position loss.
        quat_weight: Weight multiplier for quaternion loss.

    Returns:
        total_loss, pos_loss, quat_loss (all scalar tensors).
    """
    mse = nn.MSELoss(reduction="none")

    # Per-sample position loss: mean over 3 dims → (B,)
    pos_loss_per_sample = mse(pred_actions[:, :3], gt_actions[:, :3]).mean(dim=1)

    # Per-sample quaternion loss: mean over 4 dims → (B,)
    quat_loss_per_sample = mse(pred_actions[:, 3:], gt_actions[:, 3:]).mean(dim=1)

    # Keyframe reinforcement: multiply loss by 2.0 for keyframe samples
    keyframe_weight = torch.ones_like(pos_loss_per_sample)
    keyframe_weight[is_keyframe] = 2.0

    pos_loss = (pos_loss_per_sample * keyframe_weight).mean()
    quat_loss = (quat_loss_per_sample * keyframe_weight).mean()

    total_loss = pos_weight * pos_loss + quat_weight * quat_loss
    return total_loss, pos_loss, quat_loss


def _flatten_batch(batch, device):
    """
    Flatten episode-level batch tensors to timestep-level.

    VLADataset returns per-episode data with shape (T, ...).
    DataLoader collates these into (B, T, ...).
    This function flattens to (B*T, ...) for the model.
    If the data is already 2D/3D (no trajectory dim), it is returned as-is.
    """
    images = batch["images"].to(device)
    depth = batch["depth"].to(device)
    lang_embed = batch["lang_embed"].to(device)
    actions = batch["actions"].to(device)
    is_keyframe = batch["is_keyframe"].to(device)

    # Detect episode-level batching: images will be (B, T, H, W, C) instead of (B, H, W, C)
    if images.dim() == 5:
        B, T = images.shape[:2]
        images = images.reshape(B * T, *images.shape[2:])
        depth = depth.reshape(B * T, *depth.shape[2:])
        lang_embed = lang_embed.reshape(B * T, *lang_embed.shape[2:])
        actions = actions.reshape(B * T, *actions.shape[2:])
        is_keyframe = is_keyframe.reshape(B * T)

    # Convert (N, H, W, C) → (N, C, H, W)
    pixel_values = images.permute(0, 3, 1, 2)
    depth_values = depth.permute(0, 3, 1, 2)

    return pixel_values, depth_values, lang_embed, actions, is_keyframe


def train_one_epoch(model, dataloader, optimizer, device):
    """Run one training epoch and return average losses."""
    model.train()
    total_loss_sum = 0.0
    pos_loss_sum = 0.0
    quat_loss_sum = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        pixel_values, depth_values, lang_embed, actions, is_keyframe = (
            _flatten_batch(batch, device)
        )

        pred_actions = model(pixel_values, depth_values, lang_embed)
        total_loss, pos_loss, quat_loss = compute_loss(
            pred_actions, actions, is_keyframe
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        pos_loss_sum += pos_loss.item()
        quat_loss_sum += quat_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(
                f"  [Batch {batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {total_loss.item():.4f} "
                f"(pos: {pos_loss.item():.4f}, quat: {quat_loss.item():.4f})"
            )

    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return (
        total_loss_sum / num_batches,
        pos_loss_sum / num_batches,
        quat_loss_sum / num_batches,
    )


@torch.no_grad()
def validate(model, dataloader, device):
    """Run validation and return average losses."""
    model.eval()
    total_loss_sum = 0.0
    pos_loss_sum = 0.0
    quat_loss_sum = 0.0
    num_batches = 0

    for batch in dataloader:
        pixel_values, depth_values, lang_embed, actions, is_keyframe = (
            _flatten_batch(batch, device)
        )

        pred_actions = model(pixel_values, depth_values, lang_embed)
        total_loss, pos_loss, quat_loss = compute_loss(
            pred_actions, actions, is_keyframe
        )

        total_loss_sum += total_loss.item()
        pos_loss_sum += pos_loss.item()
        quat_loss_sum += quat_loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return (
        total_loss_sum / num_batches,
        pos_loss_sum / num_batches,
        quat_loss_sum / num_batches,
    )


def main():
    args = parse_args()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset & DataLoader ---
    print(f"Loading data from: {args.data_dir}")
    full_dataset = VLADataset(data_dir=args.data_dir)
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        print("ERROR: No HDF5 episodes found in data directory.")
        sys.exit(1)

    val_size = max(1, dataset_size // 10)
    train_size = dataset_size - val_size
    print(f"Dataset: {dataset_size} episodes (train: {train_size}, val: {val_size})")

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Model ---
    print("Initializing VLAPolicy...")
    model = VLAPolicy(freeze_backbone=True)
    model.to(device)

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float("inf")

    # --- Resume from checkpoint ---
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # --- Save directory ---
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "vla_model_best.pth")

    # --- Training loop ---
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        train_loss, train_pos, train_quat = train_one_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_pos, val_quat = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1} Summary:\n"
            f"  Train — Loss: {train_loss:.4f}, Pos: {train_pos:.4f}, Quat: {train_quat:.4f}\n"
            f"  Val   — Loss: {val_loss:.4f}, Pos: {val_pos:.4f}, Quat: {val_quat:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ Best model saved to {best_model_path}")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
