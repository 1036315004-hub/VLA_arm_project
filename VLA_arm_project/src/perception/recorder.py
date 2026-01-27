import os
import json
import time
import cv2
import numpy as np
import pybullet as p


class DataRecorder:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.current_episode = {}
        self.metadata = {}
        self.episode_idx = self._get_next_episode_idx()

        # 确保目录存在
        self.img_dir = os.path.join(save_dir, "images")
        self.meta_dir = os.path.join(save_dir, "metadata")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

    def _get_next_episode_idx(self):
        """扫描现有文件以确定下一个 episode ID"""
        if not os.path.exists(os.path.join(self.save_dir, "metadata")):
            return 0
        existing_files = [f for f in os.listdir(os.path.join(self.save_dir, "metadata")) if f.endswith(".json")]
        if not existing_files:
            return 0

        indices = []
        for f in existing_files:
            # 格式: episode_0.json
            parts = f.split('_')
            if len(parts) > 1 and parts[1].split('.')[0].isdigit():
                indices.append(int(parts[1].split('.')[0]))

        if not indices:
            return 0
        return max(indices) + 1

    def start_new_episode(self, instruction, initial_ee_pos=None):
        """开始一个新的录制回合"""
        self.episode_idx = self._get_next_episode_idx()
        self.current_episode = {
            "instruction": instruction,
            "steps": [],
            "success": False
        }
        # metadata 用于存储全局信息，可以在外部被 collect_data 修改
        self.metadata = {
            "episode_id": self.episode_idx,
            "instruction": instruction,
            "timestamp": time.time()
        }
        print(f"[Recorder] Started Episode {self.episode_idx} - '{instruction}'")

    def record_step(self, robot_id, joint_indices, ee_index, sensor_data):
        """记录单个时间步的数据"""
        rgb, depth = sensor_data

        # 1. 采集机械臂状态
        joint_states = p.getJointStates(robot_id, joint_indices)
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]

        ee_state = p.getLinkState(robot_id, ee_index)
        ee_pos = ee_state[0]
        ee_orn = ee_state[1]  # Quaternion

        step_idx = len(self.current_episode["steps"])

        # 2. 保存图像
        img_filename = f"ep_{self.episode_idx}_step_{step_idx}.jpg"
        img_path_full = os.path.join(self.img_dir, img_filename)

        if rgb is not None:
            # PyBullet RGB 是 RGB 顺序，OpenCV 需要 BGR
            bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path_full, bgr_img)

        # 3. 组装数据包
        step_data = {
            "step_id": step_idx,
            "image_path": os.path.join("images", img_filename),  # 存相对路径
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_position": list(ee_pos),
            "ee_orientation": list(ee_orn)
        }

        self.current_episode["steps"].append(step_data)

    def save_episode(self, success=True):
        """结束回合并将数据写入 JSON"""
        self.metadata["success"] = success
        self.metadata["num_steps"] = len(self.current_episode["steps"])
        self.metadata["steps"] = self.current_episode["steps"]

        filename = os.path.join(self.meta_dir, f"episode_{self.episode_idx}.json")

        # 辅助函数：处理 numpy 数据类型无法被 json 序列化的问题
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            return str(obj)

        with open(filename, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=default_serializer)

        print(f"[Recorder] Episode {self.episode_idx} saved. Success: {success}")
