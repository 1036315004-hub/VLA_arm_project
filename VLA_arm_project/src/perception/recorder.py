import os
import json
import time
import cv2
import numpy as np
import pybullet as p


class DataRecorder:
    """
    Enhanced DataRecorder supporting RGB-D capture, keyframe recording,
    and quality gate metadata for training data collection.
    """
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.current_episode = {}
        self.metadata = {}
        self.episode_idx = self._get_next_episode_idx()
        self.keyframes = []  # List of keyframe data
        self.robot_base_world_pose = None  # (pos, orn) of robot base in world

        # 确保目录存在
        self.img_dir = os.path.join(save_dir, "images")
        self.depth_dir = os.path.join(save_dir, "depth")
        self.meta_dir = os.path.join(save_dir, "metadata")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
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

    def set_robot_base_pose(self, pos, orn):
        """
        Set the robot base pose in world coordinates.
        This is required for world-to-base coordinate transformation.
        
        Args:
            pos: [x, y, z] position in world frame
            orn: [qx, qy, qz, qw] quaternion orientation in world frame
        """
        self.robot_base_world_pose = (list(pos), list(orn))
    
    def world_to_base(self, pos_world, orn_world):
        """
        Transform a pose from world coordinates to robot base coordinates.
        
        Args:
            pos_world: [x, y, z] position in world frame
            orn_world: [qx, qy, qz, qw] quaternion in world frame
            
        Returns:
            tuple: (pos_base, orn_base) in robot base frame
        """
        if self.robot_base_world_pose is None:
            # If no base pose set, return identity transform
            return list(pos_world), list(orn_world)
        
        base_pos, base_orn = self.robot_base_world_pose
        
        # Compute inverse of base transform
        inv_base_pos, inv_base_orn = p.invertTransform(base_pos, base_orn)
        
        # Transform point from world to base
        pos_base, orn_base = p.multiplyTransforms(
            inv_base_pos, inv_base_orn,
            pos_world, orn_world
        )
        
        return list(pos_base), list(orn_base)

    def start_new_episode(self, instruction, initial_ee_pos=None, camera_config=None):
        """
        开始一个新的录制回合
        
        Args:
            instruction: Text instruction for this episode
            initial_ee_pos: Initial end-effector position (optional)
            camera_config: Camera configuration dict (optional)
        """
        self.episode_idx = self._get_next_episode_idx()
        self.current_episode = {
            "instruction": instruction,
            "steps": [],
            "success": False
        }
        self.keyframes = []  # Reset keyframes
        
        # metadata 用于存储全局信息，可以在外部被 collect_data 修改
        self.metadata = {
            "episode_id": self.episode_idx,
            "instruction": instruction,
            "timestamp": time.time(),
            "camera": camera_config if camera_config else {},
            "robot": {
                "base_world_pose": {
                    "position": self.robot_base_world_pose[0] if self.robot_base_world_pose else None,
                    "orientation": self.robot_base_world_pose[1] if self.robot_base_world_pose else None
                },
                "ee_pose_definition": "ee_pose_base = T_base_world^(-1) * ee_pose_world"
            },
            "target": {},
            "quality": {
                "accepted": False,
                "reasons": [],
                "metrics": {}
            }
        }
        print(f"[Recorder] Started Episode {self.episode_idx} - '{instruction}'")

    def record_step(self, robot_id, joint_indices, ee_index, sensor_data, phase=None):
        """
        记录单个时间步的数据
        
        Args:
            robot_id: PyBullet robot body ID
            joint_indices: List of joint indices
            ee_index: End-effector link index
            sensor_data: Tuple of (rgb, depth) images
            phase: Optional phase name (e.g., 'hover', 'pre_contact', 'contact', 'lift')
        """
        rgb, depth = sensor_data

        # 1. 采集机械臂状态
        joint_states = p.getJointStates(robot_id, joint_indices)
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]

        ee_state = p.getLinkState(robot_id, ee_index)
        ee_pos_world = ee_state[0]
        ee_orn_world = ee_state[1]  # Quaternion
        
        # Transform to base frame
        ee_pos_base, ee_orn_base = self.world_to_base(ee_pos_world, ee_orn_world)

        step_idx = len(self.current_episode["steps"])

        # 2. 保存图像 (RGB as PNG for quality)
        img_filename = f"ep_{self.episode_idx}_step_{step_idx}.png"
        img_path_full = os.path.join(self.img_dir, img_filename)

        if rgb is not None:
            # PyBullet RGB 是 RGB 顺序，OpenCV 需要 BGR
            bgr_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path_full, bgr_img)
        
        # 3. 保存深度 (as .npy for precision)
        depth_filename = f"ep_{self.episode_idx}_step_{step_idx}_depth.npy"
        depth_path_full = os.path.join(self.depth_dir, depth_filename)
        
        if depth is not None:
            np.save(depth_path_full, depth.astype(np.float32))

        # 4. 组装数据包
        step_data = {
            "step_id": step_idx,
            "image_path": os.path.join("images", img_filename),  # 存相对路径
            "depth_path": os.path.join("depth", depth_filename) if depth is not None else None,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_position_world": list(ee_pos_world),
            "ee_orientation_world": list(ee_orn_world),
            "ee_position_base": ee_pos_base,
            "ee_orientation_base": ee_orn_base,
            "phase": phase
        }

        self.current_episode["steps"].append(step_data)
        return step_idx
    
    def record_keyframe(self, name, robot_id, ee_index, target_pos_world, step_id=None):
        """
        Record a keyframe at the current moment.
        
        Args:
            name: Keyframe name ('hover', 'pre_contact', 'contact', 'lift')
            robot_id: PyBullet robot body ID
            ee_index: End-effector link index
            target_pos_world: Target position in world coordinates (for convergence calc)
            step_id: Optional step ID, uses current step count if None
        """
        ee_state = p.getLinkState(robot_id, ee_index)
        ee_pos_world = ee_state[0]
        ee_orn_world = ee_state[1]
        
        # Transform to base frame
        ee_pos_base, ee_orn_base = self.world_to_base(ee_pos_world, ee_orn_world)
        
        # Calculate convergence distance
        convergence_dist = np.linalg.norm(np.array(ee_pos_world) - np.array(target_pos_world))
        
        keyframe_data = {
            "name": name,
            "step_id": step_id if step_id is not None else len(self.current_episode["steps"]) - 1,
            "ee_position_world": list(ee_pos_world),
            "ee_orientation_world": list(ee_orn_world),
            "ee_position_base": ee_pos_base,
            "ee_orientation_base": ee_orn_base,
            "target_position_world": list(target_pos_world),
            "convergence_distance": float(convergence_dist)
        }
        
        self.keyframes.append(keyframe_data)
        print(f"[Recorder] Keyframe '{name}' recorded at step {keyframe_data['step_id']}, "
              f"convergence: {convergence_dist:.4f}m")
        
        return keyframe_data
    
    def set_target_info(self, obj_id, name, gras_pos_world):
        """
        Set target object information in metadata.
        
        Args:
            obj_id: PyBullet object ID
            name: Object name
            gras_pos_world: Grasp position in world coordinates
        """
        self.metadata["target"] = {
            "object_id": obj_id,
            "name": name,
            "gras_pos_world": list(gras_pos_world)
        }

    def save_episode(self, success=True, accepted=True, quality_reasons=None, quality_metrics=None):
        """
        结束回合并将数据写入 JSON
        
        Args:
            success: Whether the task was successful
            accepted: Whether episode passed quality gate
            quality_reasons: List of rejection reasons (if not accepted)
            quality_metrics: Dict of quality metrics
        """
        self.metadata["success"] = success
        self.metadata["accepted"] = accepted
        self.metadata["num_steps"] = len(self.current_episode["steps"])
        self.metadata["steps"] = self.current_episode["steps"]
        self.metadata["keyframes"] = self.keyframes
        
        # Update quality gate info
        self.metadata["quality"] = {
            "accepted": accepted,
            "reasons": quality_reasons if quality_reasons else [],
            "metrics": quality_metrics if quality_metrics else {}
        }

        filename = os.path.join(self.meta_dir, f"episode_{self.episode_idx}.json")

        # 辅助函数：处理 numpy 数据类型无法被 json 序列化的问题
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return str(obj)

        with open(filename, 'w') as f:
            json.dump(self.metadata, f, indent=4, default=default_serializer)

        status = "accepted" if accepted else "rejected"
        print(f"[Recorder] Episode {self.episode_idx} saved. Success: {success}, Status: {status}")
        
        return filename
