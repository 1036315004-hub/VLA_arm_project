#!/usr/bin/env python3
"""
Preprocess VLA episodes into HDF5 format for training.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

IMAGE_SIZE = (224, 224)
PIL_RESAMPLING = getattr(Image, "Resampling", Image)
RGB_RESAMPLE = PIL_RESAMPLING.BILINEAR
DEPTH_RESAMPLE = PIL_RESAMPLING.NEAREST
DEPTH_MIN = 0.1
DEPTH_MAX = 1.5
IDENTITY_QUATERNION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
POSITION_BOUNDS = np.array(
    [[0.3, 1.1], [-0.7, 0.7], [0.2, 1.0]], dtype=np.float32
)


logger = logging.getLogger("preprocess")


def setup_logging(output_dir: Path) -> None:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = output_dir / "preprocess.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def normalize_position(position: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32)
    clipped = np.clip(position, POSITION_BOUNDS[:, 0], POSITION_BOUNDS[:, 1])
    scaled = (clipped - POSITION_BOUNDS[:, 0]) / (
        POSITION_BOUNDS[:, 1] - POSITION_BOUNDS[:, 0]
    )
    return scaled * 2.0 - 1.0


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat = quat / norm
    else:
        quat = IDENTITY_QUATERNION.copy()
    if quat[3] < 0:
        quat = -quat
    return quat


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        rgb_img = rgb_img.resize(IMAGE_SIZE, RGB_RESAMPLE)
        return np.asarray(rgb_img, dtype=np.uint8)


def load_depth(depth_path: Path) -> np.ndarray:
    depth = np.load(depth_path).astype(np.float32)
    depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX)
    depth = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    depth_img = Image.fromarray(depth, mode="F").resize(IMAGE_SIZE, DEPTH_RESAMPLE)
    depth = np.asarray(depth_img, dtype=np.float32)
    return depth[..., None]


def embed_instruction(
    instruction: str,
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    device: torch.device,
    cache: dict,
) -> np.ndarray:
    if instruction in cache:
        return cache[instruction]

    tokens = tokenizer(
        instruction,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
        embedding = outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)

    cache[instruction] = embedding
    return embedding


def episode_output_path(metadata_path: Path, metadata: dict, output_dir: Path) -> Path:
    episode_id = metadata.get("episode_id")
    if episode_id is None:
        episode_id = metadata_path.stem.replace("episode_", "")
    return output_dir / f"episode_{episode_id}.h5"


def process_episode(
    metadata_path: Path,
    input_dir: Path,
    output_dir: Path,
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    device: torch.device,
    cache: dict,
) -> None:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    steps = metadata.get("steps", [])
    if not steps:
        raise ValueError("metadata missing steps")

    keyframe_steps = {
        keyframe.get("step_id")
        for keyframe in metadata.get("keyframes", [])
        if keyframe.get("step_id") is not None
    }

    images = []
    depths = []
    actions = []
    is_keyframe = []

    for step in steps:
        image_rel = step.get("image_path")
        depth_rel = step.get("depth_path")
        if not image_rel or not depth_rel:
            raise FileNotFoundError("step missing image or depth path")

        image_path = input_dir / image_rel
        depth_path = input_dir / depth_rel
        if not image_path.exists():
            raise FileNotFoundError(f"missing image: {image_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"missing depth: {depth_path}")

        images.append(load_image(image_path))
        depths.append(load_depth(depth_path))

        position = step.get("ee_position_base")
        orientation = step.get("ee_orientation_base")
        if position is None or orientation is None:
            raise ValueError("step missing ee pose in base coordinates")

        action = np.concatenate(
            [normalize_position(position), normalize_quaternion(orientation)]
        )
        actions.append(action)

        is_keyframe.append(step.get("step_id") in keyframe_steps)

    images = np.stack(images, axis=0)
    depths = np.stack(depths, axis=0)
    actions = np.stack(actions, axis=0).astype(np.float32)
    is_keyframe = np.asarray(is_keyframe, dtype=bool)

    instruction = metadata.get("instruction", "")
    lang_embed = embed_instruction(instruction, tokenizer, model, device, cache)

    output_path = episode_output_path(metadata_path, metadata, output_dir)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("images", data=images, dtype=np.uint8)
        h5_file.create_dataset("depth", data=depths, dtype=np.float32)
        h5_file.create_dataset("actions", data=actions, dtype=np.float32)
        h5_file.create_dataset("lang_embed", data=lang_embed, dtype=np.float32)
        h5_file.create_dataset("is_keyframe", data=is_keyframe, dtype=bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess VLA raw data to HDF5")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "raw"),
        help="Input raw data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "processed"),
        help="Output directory for processed data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    metadata_dir = input_dir / "metadata"
    if not metadata_dir.exists():
        logger.error("Metadata directory not found: %s", metadata_dir)
        return

    episode_paths = sorted(metadata_dir.glob("episode_*.json"))
    if not episode_paths:
        logger.warning("No episodes found in %s", metadata_dir)
        return

    logger.info("Loading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    cache = {}

    for metadata_path in tqdm(episode_paths, desc="Processing episodes"):
        try:
            process_episode(
                metadata_path,
                input_dir,
                output_dir,
                tokenizer,
                model,
                device,
                cache,
            )
        except Exception as exc:
            logger.exception("Skipping %s: %s", metadata_path.name, exc)
            continue

    logger.info("Preprocessing complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
