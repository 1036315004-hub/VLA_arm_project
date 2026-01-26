"""
Multi-Stage Pose Planner for Elite EC63 Robot

This module implements multi-stage pose planning to solve occlusion problems
during grasping and recognition tasks. The EC63 robot arm's link structure
often blocks the visual target, making YOLO+CLIP recognition difficult.

The planner provides:
1. Pre-observation pose - Arm positioned to avoid camera view occlusion
2. Observation pose - Optimal position for Eye-to-hand camera viewing
3. Pre-grasp pose - Approach position before grasping
4. Grasp pose - Final grasping position
5. Retreat pose - Safe retreat after grasping
"""

import math
import time
import numpy as np
from typing import List, Tuple, Optional, Dict
import pybullet as p


class MultiStagePlanner:
    """
    Multi-stage pose planner for solving occlusion issues during grasping tasks.
    
    The planner computes waypoints that ensure the robot arm doesn't block
    the camera's view of the target object during different phases of operation.
    """
    
    # Planning stages
    STAGE_IDLE = 0
    STAGE_PRE_OBSERVE = 1
    STAGE_OBSERVE = 2
    STAGE_PRE_GRASP = 3
    STAGE_GRASP = 4
    STAGE_RETREAT = 5
    
    # Default safe heights (in meters)
    DEFAULT_PRE_OBSERVE_HEIGHT = 0.60
    DEFAULT_OBSERVE_HEIGHT = 0.55
    DEFAULT_PRE_GRASP_HEIGHT = 0.25
    DEFAULT_GRASP_HEIGHT = 0.05
    DEFAULT_RETREAT_HEIGHT = 0.45
    
    # Occlusion avoidance parameters
    OCCLUSION_OFFSET_DISTANCE = 0.15  # Lateral offset to avoid camera view
    
    def __init__(
        self,
        robot_id: int,
        end_effector_index: int,
        joint_indices: List[int],
        joint_force: float = 500.0,
        table_surface_height: float = 0.4
    ):
        """
        Initialize the multi-stage pose planner.
        
        Args:
            robot_id: PyBullet body ID of the robot
            end_effector_index: Index of the end effector link
            joint_indices: List of joint indices to control
            joint_force: Maximum force for joint motors
            table_surface_height: Height of the table surface in meters
        """
        self.robot_id = robot_id
        self.end_effector_index = end_effector_index
        self.joint_indices = joint_indices
        self.joint_force = joint_force
        self.table_surface_height = table_surface_height
        
        self.current_stage = self.STAGE_IDLE
        self._waypoints = []
        
    def compute_occlusion_free_pose(
        self,
        target_pos: np.ndarray,
        camera_pos: np.ndarray,
        offset_direction: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute a pose that avoids occluding the camera's view of the target.
        
        Args:
            target_pos: Target object position [x, y, z]
            camera_pos: Eye-to-hand camera position [x, y, z]
            offset_direction: Direction to offset the arm (default: perpendicular to camera-target line)
            
        Returns:
            Occlusion-free position for the end effector
        """
        target = np.array(target_pos)
        camera = np.array(camera_pos)
        
        # Compute vector from camera to target
        cam_to_target = target - camera
        cam_to_target[2] = 0  # Project to XY plane
        
        if np.linalg.norm(cam_to_target) < 0.01:
            cam_to_target = np.array([1.0, 0.0, 0.0])
        else:
            cam_to_target = cam_to_target / np.linalg.norm(cam_to_target)
        
        # Compute perpendicular direction in XY plane
        if offset_direction is None:
            offset_direction = np.array([-cam_to_target[1], cam_to_target[0], 0.0])
        
        # Offset the position to avoid blocking camera view
        occlusion_free_pos = target.copy()
        occlusion_free_pos[:2] += offset_direction[:2] * self.OCCLUSION_OFFSET_DISTANCE
        
        return occlusion_free_pos
    
    def plan_observation_waypoints(
        self,
        camera_pos: np.ndarray,
        workspace_center: np.ndarray,
        num_viewpoints: int = 5
    ) -> List[np.ndarray]:
        """
        Plan waypoints for systematic scanning to minimize occlusion.
        
        Args:
            camera_pos: Eye-to-hand camera position
            workspace_center: Center of the workspace to observe
            num_viewpoints: Number of observation viewpoints
            
        Returns:
            List of arm positions that won't occlude camera view
        """
        waypoints = []
        
        # Compute safe zone boundary (where arm won't block camera)
        cam_to_workspace = workspace_center[:2] - camera_pos[:2]
        
        if np.linalg.norm(cam_to_workspace) < 0.01:
            cam_to_workspace = np.array([1.0, 0.0])
        else:
            cam_to_workspace = cam_to_workspace / np.linalg.norm(cam_to_workspace)
        
        # Place arm waypoints on the opposite side of the workspace from camera
        for i in range(num_viewpoints):
            angle = math.pi * (0.3 + 0.4 * i / (num_viewpoints - 1)) if num_viewpoints > 1 else math.pi * 0.5
            
            # Position behind/beside the workspace relative to camera
            offset_x = math.cos(angle) * self.OCCLUSION_OFFSET_DISTANCE * 2
            offset_y = math.sin(angle) * self.OCCLUSION_OFFSET_DISTANCE * 2
            
            waypoint = np.array([
                workspace_center[0] + cam_to_workspace[0] * 0.3 + offset_x,
                workspace_center[1] + cam_to_workspace[1] * 0.3 + offset_y,
                self.table_surface_height + self.DEFAULT_PRE_OBSERVE_HEIGHT
            ])
            waypoints.append(waypoint)
        
        return waypoints
    
    def plan_grasp_sequence(
        self,
        target_pos: np.ndarray,
        camera_pos: np.ndarray,
        approach_direction: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Plan a complete grasp sequence that avoids camera occlusion.
        
        Args:
            target_pos: Target object position
            camera_pos: Eye-to-hand camera position
            approach_direction: Preferred approach direction for grasping
            
        Returns:
            Dictionary of poses for each stage
        """
        target = np.array(target_pos)
        
        # Compute occlusion-free approach direction
        occlusion_free_base = self.compute_occlusion_free_pose(target, camera_pos)
        
        poses = {
            'pre_observe': np.array([
                occlusion_free_base[0],
                occlusion_free_base[1],
                self.table_surface_height + self.DEFAULT_PRE_OBSERVE_HEIGHT
            ]),
            'observe': np.array([
                occlusion_free_base[0],
                occlusion_free_base[1],
                self.table_surface_height + self.DEFAULT_OBSERVE_HEIGHT
            ]),
            'pre_grasp': np.array([
                target[0],
                target[1],
                target[2] + self.DEFAULT_PRE_GRASP_HEIGHT
            ]),
            'grasp': np.array([
                target[0],
                target[1],
                target[2] + self.DEFAULT_GRASP_HEIGHT
            ]),
            'retreat': np.array([
                target[0],
                target[1],
                self.table_surface_height + self.DEFAULT_RETREAT_HEIGHT
            ])
        }
        
        self._waypoints = [poses['pre_observe'], poses['observe'], 
                          poses['pre_grasp'], poses['grasp'], poses['retreat']]
        
        return poses
    
    def get_arm_retreat_pose(
        self,
        camera_pos: np.ndarray,
        workspace_bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Get a safe retreat pose where the arm won't block the camera view.
        
        Args:
            camera_pos: Eye-to-hand camera position
            workspace_bounds: (x_min, x_max, y_min, y_max) of workspace
            
        Returns:
            Safe retreat position for the end effector
        """
        x_min, x_max, y_min, y_max = workspace_bounds
        workspace_center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            self.table_surface_height
        ])
        
        # Move arm to the side opposite to camera
        cam_to_center = workspace_center[:2] - camera_pos[:2]
        if np.linalg.norm(cam_to_center) > 0.01:
            cam_to_center = cam_to_center / np.linalg.norm(cam_to_center)
        else:
            cam_to_center = np.array([1.0, 0.0])
        
        # Perpendicular direction
        retreat_dir = np.array([-cam_to_center[1], cam_to_center[0]])
        
        # Choose which side based on current arm position
        current_ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
        current_pos = np.array(current_ee_state[0])
        
        # Go to the side that's closer to current position
        retreat_pos1 = workspace_center[:2] + retreat_dir * 0.4
        retreat_pos2 = workspace_center[:2] - retreat_dir * 0.4
        
        if np.linalg.norm(current_pos[:2] - retreat_pos1) < np.linalg.norm(current_pos[:2] - retreat_pos2):
            retreat_xy = retreat_pos1
        else:
            retreat_xy = retreat_pos2
        
        return np.array([
            retreat_xy[0],
            retreat_xy[1],
            self.table_surface_height + self.DEFAULT_RETREAT_HEIGHT
        ])
    
    def compute_home_pose_joint_angles(self) -> List[float]:
        """
        Compute joint angles for a home pose that keeps the arm out of camera view.
        
        Returns:
            List of joint angles in radians
        """
        # Home pose: arm folded back and to the side
        # These are typical safe angles for EC63 that keep the arm compact
        home_angles = [
            0.0,           # Joint 1: Base rotation - neutral
            -math.pi / 4,  # Joint 2: Shoulder - slightly back
            math.pi / 2,   # Joint 3: Elbow - folded
            0.0,           # Joint 4: Wrist 1 - neutral
            math.pi / 4,   # Joint 5: Wrist 2 - angled
            0.0            # Joint 6: Wrist 3 - neutral
        ]
        return home_angles[:len(self.joint_indices)]
    
    def move_to_pose(
        self,
        target_pos: np.ndarray,
        target_orn: Optional[np.ndarray] = None,
        steps: int = 100,
        sleep_time: float = 1.0 / 240.0,
        tolerance: float = 0.01,
        use_gui: bool = True
    ) -> bool:
        """
        Move the end effector to a target pose using IK.
        
        Args:
            target_pos: Target position [x, y, z]
            target_orn: Target orientation quaternion (default: pointing down)
            steps: Maximum number of simulation steps
            sleep_time: Sleep time between steps (for GUI visualization)
            tolerance: Position tolerance for convergence
            use_gui: Whether to use GUI timing
            
        Returns:
            True if target reached within tolerance
        """
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        
        max_steps = steps * 2
        max_vel = 2.5
        
        for i in range(max_steps):
            joint_positions = p.calculateInverseKinematics(
                self.robot_id, 
                self.end_effector_index, 
                target_pos, 
                target_orn,
                maxNumIterations=100, 
                residualThreshold=1e-5
            )
            
            for idx_j, joint_idx in enumerate(self.joint_indices):
                if idx_j < len(joint_positions):
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=joint_positions[idx_j],
                        force=self.joint_force,
                        maxVelocity=max_vel
                    )
            
            p.stepSimulation()
            if use_gui and sleep_time > 0:
                time.sleep(sleep_time)
            
            if i % 10 == 0:
                current_state = p.getLinkState(self.robot_id, self.end_effector_index)
                current_pos = current_state[0]
                dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                if dist < tolerance:
                    return True
        
        return False
    
    def set_joint_positions(self, joint_angles: List[float]) -> None:
        """
        Directly set joint positions (instant, for initialization).
        
        Args:
            joint_angles: List of joint angles in radians
        """
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(joint_angles):
                p.resetJointState(self.robot_id, joint_idx, joint_angles[i])
