"""
Data Recorder Module for VLA Arm Project

This module provides a data recorder that supports Eye-to-hand perspective
and automatic randomization of:
- Camera position
- Object poses
- Robot arm initial state

The recorder is designed to work with the Elite EC63 robot and solves
occlusion issues caused by the robot arm blocking the camera view.

Note: This version does not record data - it sets up the simulation
environment with randomization for future data collection.
"""

import os
import sys
import math
import random
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

import pybullet as p
import pybullet_data

# Import local modules
from .objects.pybullet_object_generator import PyBulletObjectGenerator
from .robot.multi_stage_planner import MultiStagePlanner


class EyeToHandCamera:
    """
    Eye-to-hand camera configuration and image capture.
    
    The camera is mounted externally (not on the robot arm) providing
    a fixed or configurable viewpoint of the workspace.
    """
    
    def __init__(
        self,
        position: List[float],
        target: List[float],
        up_vector: List[float] = None,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 3.0
    ):
        """
        Initialize the Eye-to-hand camera.
        
        Args:
            position: Camera position [x, y, z]
            target: Point the camera is looking at [x, y, z]
            up_vector: Camera up direction (default: [0, 0, 1])
            width: Image width in pixels
            height: Image height in pixels
            fov: Field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
        """
        self.position = np.array(position)
        self.target = np.array(target)
        self.up_vector = np.array(up_vector) if up_vector else np.array([0, 0, 1])
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        
        self._update_matrices()
    
    def _update_matrices(self):
        """Update view and projection matrices."""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.position.tolist(),
            cameraTargetPosition=self.target.tolist(),
            cameraUpVector=self.up_vector.tolist()
        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )
    
    def set_position(self, position: List[float], target: Optional[List[float]] = None):
        """
        Set camera position and optionally update target.
        
        Args:
            position: New camera position
            target: New target point (optional)
        """
        self.position = np.array(position)
        if target is not None:
            self.target = np.array(target)
        self._update_matrices()
    
    def capture_image(self, use_gui: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture RGB, depth, and segmentation images.
        
        Args:
            use_gui: Whether to use hardware OpenGL renderer
            
        Returns:
            Tuple of (rgb_image, depth_buffer, segmentation_mask)
        """
        renderer = p.ER_BULLET_HARDWARE_OPENGL if use_gui else p.ER_TINY_RENDERER
        
        _, _, rgba, depth, segmentation = p.getCameraImage(
            self.width,
            self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=renderer
        )
        
        rgb = np.reshape(rgba, (self.height, self.width, 4))[:, :, :3].astype(np.uint8)
        depth_buffer = np.reshape(depth, (self.height, self.width))
        seg_mask = np.reshape(segmentation, (self.height, self.width))
        
        return rgb, depth_buffer, seg_mask
    
    def get_intrinsic_matrix(self) -> np.ndarray:
        """
        Compute camera intrinsic matrix from projection parameters.
        
        Returns:
            3x3 intrinsic matrix
        """
        fov_rad = self.fov * math.pi / 180.0
        fx = self.width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx  # Assuming square pixels
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    def pixel_to_world(self, u: int, v: int, depth: float) -> np.ndarray:
        """
        Convert pixel coordinates to world coordinates.
        
        Args:
            u: Pixel x coordinate
            v: Pixel y coordinate
            depth: Depth value from depth buffer
            
        Returns:
            World coordinates [x, y, z]
        """
        view_mat = np.array(self.view_matrix).reshape((4, 4), order='F')
        proj_mat = np.array(self.projection_matrix).reshape((4, 4), order='F')
        inv_proj_view = np.linalg.inv(proj_mat @ view_mat)
        
        x_ndc = (2.0 * u / (self.width - 1)) - 1.0
        y_ndc = 1.0 - (2.0 * v / (self.height - 1))
        z_ndc = (2.0 * depth) - 1.0
        
        clip = np.array([x_ndc, y_ndc, z_ndc, 1.0])
        world = inv_proj_view @ clip
        world /= world[3]
        
        return world[:3]


class DataRecorderConfig:
    """Configuration for the data recorder."""
    
    def __init__(self):
        # Workspace bounds
        self.workspace_x_min = 0.45
        self.workspace_x_max = 0.90
        self.workspace_y_min = -0.40
        self.workspace_y_max = 0.40
        
        # Table settings
        self.table_position = [0.8, 0.0, 0.0]
        self.table_surface_height = 0.4
        
        # Robot settings
        self.robot_base_position = [0.35, 0.0, 0.40]
        self.robot_base_orientation = [0, 0, 0, 1]
        
        # Camera randomization ranges (Eye-to-hand)
        self.camera_x_range = (0.0, 0.3)
        self.camera_y_range = (-0.5, 0.5)
        self.camera_z_range = (0.9, 1.3)
        
        # Number of objects
        self.min_objects = 2
        self.max_objects = 5
        
        # Image settings
        self.image_width = 640
        self.image_height = 480
        self.camera_fov = 60.0


class DataRecorder:
    """
    Data recorder with Eye-to-hand camera support and automatic randomization.
    
    This recorder sets up a simulation environment with:
    - Eye-to-hand camera with randomizable position
    - Random object generation and placement
    - Random robot arm initial state
    - Multi-stage pose planning to avoid occlusion
    """
    
    def __init__(
        self,
        config: Optional[DataRecorderConfig] = None,
        urdf_path: Optional[str] = None,
        table_urdf_path: Optional[str] = None,
        use_gui: bool = True
    ):
        """
        Initialize the data recorder.
        
        Args:
            config: Configuration object (uses defaults if None)
            urdf_path: Path to robot URDF file
            table_urdf_path: Path to table URDF file
            use_gui: Whether to use PyBullet GUI
        """
        self.config = config if config else DataRecorderConfig()
        self.urdf_path = urdf_path
        self.table_urdf_path = table_urdf_path
        self.use_gui = use_gui
        
        # State
        self.connected = False
        self.robot_id = None
        self.table_id = None
        self.plane_id = None
        self.joint_indices = []
        self.end_effector_index = 6
        
        # Components
        self.camera: Optional[EyeToHandCamera] = None
        self.object_generator: Optional[PyBulletObjectGenerator] = None
        self.pose_planner: Optional[MultiStagePlanner] = None
        
    def connect(self) -> bool:
        """
        Connect to PyBullet and initialize simulation.
        
        Returns:
            True if connection successful
        """
        if self.connected:
            return True
        
        try:
            if self.use_gui:
                connection_id = p.connect(p.GUI)
                if connection_id < 0:
                    p.connect(p.DIRECT)
                    self.use_gui = False
            else:
                p.connect(p.DIRECT)
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            
            self.connected = True
            return True
        except Exception as e:
            print(f"[DataRecorder] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from PyBullet."""
        if self.connected:
            p.disconnect()
            self.connected = False
    
    def setup_environment(self) -> bool:
        """
        Set up the simulation environment.
        
        Returns:
            True if setup successful
        """
        if not self.connected:
            if not self.connect():
                return False
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table
        if self.table_urdf_path and os.path.exists(self.table_urdf_path):
            self.table_id = p.loadURDF(
                self.table_urdf_path,
                basePosition=self.config.table_position,
                useFixedBase=True
            )
        
        # Load robot
        if self.urdf_path and os.path.exists(self.urdf_path):
            self.robot_id = p.loadURDF(
                self.urdf_path,
                basePosition=self.config.robot_base_position,
                baseOrientation=self.config.robot_base_orientation,
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION
            )
            
            # Get joint indices
            num_joints = p.getNumJoints(self.robot_id)
            self.joint_indices = list(range(min(num_joints, 6)))
            self.end_effector_index = min(6, num_joints - 1)
        else:
            print(f"[DataRecorder] Robot URDF not found: {self.urdf_path}")
            return False
        
        # Initialize components
        self.object_generator = PyBulletObjectGenerator()
        
        if self.robot_id is not None:
            self.pose_planner = MultiStagePlanner(
                robot_id=self.robot_id,
                end_effector_index=self.end_effector_index,
                joint_indices=self.joint_indices,
                table_surface_height=self.config.table_surface_height
            )
        
        # Initialize camera at default position
        self._setup_camera()
        
        return True
    
    def _setup_camera(self):
        """Initialize the Eye-to-hand camera."""
        # Default camera position looking at workspace center
        workspace_center = [
            (self.config.workspace_x_min + self.config.workspace_x_max) / 2,
            (self.config.workspace_y_min + self.config.workspace_y_max) / 2,
            self.config.table_surface_height
        ]
        
        default_camera_pos = [0.1, 0.0, 1.1]
        
        self.camera = EyeToHandCamera(
            position=default_camera_pos,
            target=workspace_center,
            width=self.config.image_width,
            height=self.config.image_height,
            fov=self.config.camera_fov
        )
    
    def randomize_camera_position(self) -> np.ndarray:
        """
        Randomize the Eye-to-hand camera position.
        
        Returns:
            New camera position
        """
        x = random.uniform(*self.config.camera_x_range)
        y = random.uniform(*self.config.camera_y_range)
        z = random.uniform(*self.config.camera_z_range)
        
        new_position = [x, y, z]
        
        # Camera looks at workspace center
        workspace_center = [
            (self.config.workspace_x_min + self.config.workspace_x_max) / 2,
            (self.config.workspace_y_min + self.config.workspace_y_max) / 2,
            self.config.table_surface_height
        ]
        
        self.camera.set_position(new_position, workspace_center)
        
        return np.array(new_position)
    
    def randomize_objects(self, num_objects: Optional[int] = None) -> List[int]:
        """
        Generate random objects in the workspace.
        
        Args:
            num_objects: Number of objects (random if None)
            
        Returns:
            List of created object IDs
        """
        if self.object_generator is None:
            return []
        
        # Clear existing objects
        self.object_generator.clear_all_objects()
        
        if num_objects is None:
            num_objects = random.randint(
                self.config.min_objects,
                self.config.max_objects
            )
        
        workspace_bounds = (
            self.config.workspace_x_min,
            self.config.workspace_x_max,
            self.config.workspace_y_min,
            self.config.workspace_y_max
        )
        
        object_ids = self.object_generator.generate_random_objects(
            num_objects=num_objects,
            workspace_bounds=workspace_bounds,
            table_height=self.config.table_surface_height,
            min_distance=0.1
        )
        
        # Let objects settle
        self.object_generator.settle_objects(100)
        
        return object_ids
    
    def randomize_robot_pose(self) -> List[float]:
        """
        Randomize the robot arm's initial joint positions.
        
        Returns:
            List of joint angles
        """
        if self.robot_id is None or self.pose_planner is None:
            return []
        
        # Generate random but valid joint positions
        # These ranges are typical safe ranges for a 6-axis arm
        joint_ranges = [
            (-math.pi * 0.5, math.pi * 0.5),      # Joint 1: Base rotation
            (-math.pi * 0.25, -math.pi * 0.75),   # Joint 2: Shoulder
            (math.pi * 0.25, math.pi * 0.75),     # Joint 3: Elbow
            (-math.pi * 0.5, math.pi * 0.5),      # Joint 4: Wrist 1
            (0, math.pi * 0.5),                    # Joint 5: Wrist 2
            (-math.pi * 0.5, math.pi * 0.5),      # Joint 6: Wrist 3
        ]
        
        joint_angles = []
        for i, (low, high) in enumerate(joint_ranges):
            if i < len(self.joint_indices):
                angle = random.uniform(low, high)
                joint_angles.append(angle)
        
        # Apply joint positions
        self.pose_planner.set_joint_positions(joint_angles)
        
        # Step simulation to apply
        for _ in range(10):
            p.stepSimulation()
        
        return joint_angles
    
    def randomize_all(self) -> Dict[str, Any]:
        """
        Randomize camera, objects, and robot pose.
        
        Returns:
            Dictionary with randomization results
        """
        results = {
            'camera_position': None,
            'object_ids': [],
            'robot_joint_angles': [],
            'object_positions': {}
        }
        
        # Randomize camera
        results['camera_position'] = self.randomize_camera_position()
        
        # Randomize objects
        results['object_ids'] = self.randomize_objects()
        
        # Randomize robot
        results['robot_joint_angles'] = self.randomize_robot_pose()
        
        # Get object positions
        if self.object_generator:
            results['object_positions'] = self.object_generator.get_object_positions()
        
        return results
    
    def move_arm_to_observation_pose(self) -> bool:
        """
        Move the robot arm to a pose that doesn't occlude the camera view.
        
        Returns:
            True if successful
        """
        if self.pose_planner is None or self.camera is None:
            return False
        
        workspace_bounds = (
            self.config.workspace_x_min,
            self.config.workspace_x_max,
            self.config.workspace_y_min,
            self.config.workspace_y_max
        )
        
        retreat_pose = self.pose_planner.get_arm_retreat_pose(
            self.camera.position,
            workspace_bounds
        )
        
        return self.pose_planner.move_to_pose(
            retreat_pose,
            use_gui=self.use_gui
        )
    
    def capture_observation(self) -> Dict[str, Any]:
        """
        Capture images from the Eye-to-hand camera.
        
        Returns:
            Dictionary with captured data
        """
        if self.camera is None:
            return {}
        
        rgb, depth, segmentation = self.camera.capture_image(self.use_gui)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'segmentation': segmentation,
            'camera_position': self.camera.position.copy(),
            'camera_target': self.camera.target.copy(),
            'intrinsic_matrix': self.camera.get_intrinsic_matrix(),
            'view_matrix': np.array(self.camera.view_matrix),
            'projection_matrix': np.array(self.camera.projection_matrix)
        }
    
    def plan_grasp_for_object(self, object_id: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Plan a grasp sequence for a specific object that avoids camera occlusion.
        
        Args:
            object_id: PyBullet body ID of the target object
            
        Returns:
            Dictionary of grasp poses or None if planning failed
        """
        if self.pose_planner is None or self.camera is None:
            return None
        
        try:
            pos, _ = p.getBasePositionAndOrientation(object_id)
            target_pos = np.array(pos)
        except Exception:
            return None
        
        return self.pose_planner.plan_grasp_sequence(
            target_pos,
            self.camera.position
        )
    
    def run_episode(self, steps: int = 500) -> Dict[str, Any]:
        """
        Run a single episode with full randomization.
        
        Args:
            steps: Number of simulation steps to run
            
        Returns:
            Episode data dictionary
        """
        episode_data = {
            'randomization': {},
            'observations': [],
            'success': False
        }
        
        # Randomize everything
        episode_data['randomization'] = self.randomize_all()
        
        # Move arm to safe observation pose
        self.move_arm_to_observation_pose()
        
        # Let simulation settle
        for _ in range(50):
            p.stepSimulation()
            if self.use_gui:
                time.sleep(1.0 / 240.0)
        
        # Capture initial observation
        observation = self.capture_observation()
        episode_data['observations'].append(observation)
        
        # Run simulation steps
        for step in range(steps):
            p.stepSimulation()
            if self.use_gui:
                time.sleep(1.0 / 240.0)
        
        episode_data['success'] = True
        return episode_data
    
    def demo_randomization(self, num_iterations: int = 5, delay: float = 2.0):
        """
        Demonstrate the randomization capabilities.
        
        Args:
            num_iterations: Number of randomization iterations
            delay: Delay between iterations in seconds
        """
        print("\n" + "=" * 60)
        print("  Data Recorder - Randomization Demo")
        print("=" * 60)
        
        for i in range(num_iterations):
            print(f"\n--- Iteration {i + 1}/{num_iterations} ---")
            
            # Randomize everything
            results = self.randomize_all()
            
            print(f"Camera position: {results['camera_position']}")
            print(f"Objects created: {len(results['object_ids'])}")
            print(f"Robot joints: {[f'{a:.2f}' for a in results['robot_joint_angles']]}")
            
            # Move arm to observation pose
            self.move_arm_to_observation_pose()
            
            # Let settle
            for _ in range(100):
                p.stepSimulation()
                if self.use_gui:
                    time.sleep(1.0 / 240.0)
            
            # Capture and display info
            obs = self.capture_observation()
            print(f"Captured image size: {obs['rgb'].shape}")
            
            if self.use_gui:
                time.sleep(delay)
        
        print("\n" + "=" * 60)
        print("  Demo Complete")
        print("=" * 60)


def create_data_recorder(
    project_root: str,
    use_gui: bool = True
) -> DataRecorder:
    """
    Factory function to create a data recorder with default paths.
    
    Args:
        project_root: Root directory of the VLA_arm_project
        use_gui: Whether to use PyBullet GUI
        
    Returns:
        Configured DataRecorder instance
    """
    urdf_path = os.path.join(project_root, "assets", "ec63", "urdf", "ec63_description.urdf")
    table_path = os.path.join(project_root, "assets", "objects", "table.urdf")
    
    config = DataRecorderConfig()
    
    recorder = DataRecorder(
        config=config,
        urdf_path=urdf_path,
        table_urdf_path=table_path,
        use_gui=use_gui
    )
    
    return recorder
