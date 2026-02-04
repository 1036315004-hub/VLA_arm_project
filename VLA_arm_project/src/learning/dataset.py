import os
from typing import List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VLADataset(Dataset):
    def __init__(self, data_dir: str, episodes: Optional[List[str]] = None):
        self.data_dir = os.path.abspath(data_dir)
        if episodes is None:
            self.episodes = sorted(
                file for file in os.listdir(self.data_dir) if file.endswith(".h5")
            )
        else:
            self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        episode_name = self.episodes[idx]
        episode_path = os.path.join(self.data_dir, episode_name)
        with h5py.File(episode_path, "r") as h5_file:
            images = h5_file["images"][:].astype(np.float32) / 255.0
            images = (images - IMAGE_NET_MEAN) / IMAGE_NET_STD
            depth = h5_file["depth"][:].astype(np.float32)
            actions = h5_file["actions"][:].astype(np.float32)
            lang_embed = h5_file["lang_embed"][:].astype(np.float32)
            is_keyframe = h5_file["is_keyframe"][:].astype(bool)

        return {
            "images": torch.from_numpy(images),
            "depth": torch.from_numpy(depth),
            "actions": torch.from_numpy(actions),
            "lang_embed": torch.from_numpy(lang_embed),
            "is_keyframe": torch.from_numpy(is_keyframe),
        }
