#!/usr/bin/env python3
"""
Automated Data Collection Script for VLA Arm Project

This script implements fully automated data collection with Oracle-based
supervision labels and quality gates. Only episodes passing all quality
checks are saved as accepted training data.

Usage:
    python auto_collect.py --num_episodes 100 --save_dir data/keyframe_demos --gui

Features:
- Fixed Camera1 configuration (256x256 RGB-D)
- Oracle-based target selection and grasp position (AABB top center)
- 4-phase keyframe execution: hover -> pre_contact -> contact -> lift
- Quality gate validation for each episode
- Statistics reporting and episode inspection utilities
"""

import os
import sys
import argparse
import math
import time
import random
import json
from collections import defaultdict

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Adjust path to Project Root
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pybullet as p
import pybullet_data

from src.perception.camera_utils import get_camera_image, get_camera1_matrices, CAMERA1_CONFIG
from src.perception.scene_manager import SceneManager
from src.perception.recorder import DataRecorder
from src.perception.oracle import Oracle

# ============================================================================
# Constants
# ============================================================================

JOINT_FORCE = 500
PYBULLET_DATA_PATH = pybullet_data.getDataPath()

# Fixed keyframe parameters (Section 3.1)
HOVER_HEIGHT = 0.65          # h1: hover Z height
PRE_CONTACT_OFFSET = 0.04    # h2: pre_contact = contact_z + 0.04
LIFT_DELTA = 0.10            # lift_z = contact_z + 0.10

# Quality gate thresholds (Section 6.1)
CONVERGENCE_THRESHOLDS = {
    "hover": 0.015,
    "pre_contact": 0.010,
    "contact": 0.008,
    "lift": 0.015
}
MIN_STEPS = 20
MIN_MASK_AREA = 150  # pixels (optional gate F)

# Fixed robot base position
ROBOT_BASE_POS = [0.40, 0.00, 0.40]
ROBOT_BASE_YAW = None  # Computed to face table center


def log(message):
    print(f"[AutoCollect] {message}")


# ============================================================================
# Quality Gate Implementation (Section 6)
# ============================================================================

class QualityGate:
    """
    Implements quality validation for collected episodes.
    """
    
    def __init__(self, 
                 convergence_thresholds=None,
                 min_steps=20,
                 enable_mask_gate=False,
                 min_mask_area=150):
        self.convergence_thresholds = convergence_thresholds or CONVERGENCE_THRESHOLDS
        self.min_steps = min_steps
        self.enable_mask_gate = enable_mask_gate
        self.min_mask_area = min_mask_area
        
    def validate(self, keyframes, steps, contact_detected, depth_dir, episode_idx):
        """
        Validate an episode against all quality gates.
        
        Args:
            keyframes: List of keyframe dicts
            steps: List of step dicts
            contact_detected: Whether contact was detected at contact phase
            depth_dir: Directory containing depth files
            episode_idx: Episode index for depth file lookup
            
        Returns:
            tuple: (accepted: bool, reasons: list, metrics: dict)
        """
        reasons = []
        metrics = {}
        
        # Gate A: All keyframes exist
        keyframe_names = {kf["name"] for kf in keyframes}
        required = {"hover", "pre_contact", "contact", "lift"}
        missing = required - keyframe_names
        metrics["keyframes_present"] = list(keyframe_names)
        metrics["keyframes_missing"] = list(missing)
        
        if missing:
            reasons.append(f"Gate A: Missing keyframes: {missing}")
        
        # Gate B: Convergence distances
        convergence_metrics = {}
        for kf in keyframes:
            name = kf["name"]
            dist = kf.get("convergence_distance", float('inf'))
            convergence_metrics[name] = dist
            
            if name in self.convergence_thresholds:
                threshold = self.convergence_thresholds[name]
                if dist > threshold:
                    reasons.append(f"Gate B: {name} convergence {dist:.4f}m > {threshold}m")
        
        metrics["convergence_distances"] = convergence_metrics
        
        # Gate C: Contact detection
        metrics["contact_detected"] = contact_detected
        if not contact_detected:
            reasons.append("Gate C: No contact detected at contact phase")
        
        # Gate D: Valid depth at keyframes
        depth_valid = self._check_depth_validity(keyframes, steps, depth_dir, episode_idx)
        metrics["depth_valid"] = depth_valid
        if not all(depth_valid.values()):
            invalid = [k for k, v in depth_valid.items() if not v]
            reasons.append(f"Gate D: Invalid depth at keyframes: {invalid}")
        
        # Gate E: Minimum sequence length
        num_steps = len(steps)
        metrics["num_steps"] = num_steps
        if num_steps < self.min_steps:
            reasons.append(f"Gate E: Only {num_steps} steps < {self.min_steps} minimum")
        
        accepted = len(reasons) == 0
        return accepted, reasons, metrics
    
    def _check_depth_validity(self, keyframes, steps, depth_dir, episode_idx):
        """Check if depth files at keyframes are valid."""
        validity = {}
        
        for kf in keyframes:
            name = kf["name"]
            step_id = kf.get("step_id", -1)
            
            if step_id < 0 or step_id >= len(steps):
                validity[name] = False
                continue
            
            step = steps[step_id]
            depth_path = step.get("depth_path")
            
            if depth_path is None:
                validity[name] = False
                continue
            
            full_path = os.path.join(os.path.dirname(depth_dir), depth_path)
            if not os.path.exists(full_path):
                validity[name] = False
                continue
            
            try:
                depth = np.load(full_path)
                
                # Check for NaN/Inf
                if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
                    validity[name] = False
                    continue
                
                # Check center 3x3 region has valid values
                h, w = depth.shape
                cy, cx = h // 2, w // 2
                center_region = depth[cy-1:cy+2, cx-1:cx+2]
                
                # Valid if at least 5 out of 9 pixels are non-zero
                valid_count = np.sum(center_region > 0.01)
                validity[name] = valid_count >= 5
                
            except Exception:
                validity[name] = False
        
        return validity


# ============================================================================
# Robot Control Functions
# ============================================================================

def _has_contact(robot_id, ee_link_index, obj_id, max_contact_dist=0.001):
    """Return True if end-effector link is in contact with the object."""
    pts = p.getContactPoints(bodyA=robot_id, bodyB=obj_id, linkIndexA=ee_link_index)
    for pt in pts:
        if pt[8] <= max_contact_dist:
            return True
    return False


def move_arm_with_recording(robot_id, joint_indices, end_effector_index, 
                            target_pos, target_orn, 
                            max_steps, use_gui, tolerance=0.01,
                            recorder=None, view_mat=None, proj_mat=None,
                            record_stride=5, phase=None):
    """
    Move the end effector using IK, stepping simulation until convergence or max steps.
    Records frames during motion at specified stride.
    
    Returns:
        tuple: (converged: bool, final_distance: float, recorded_step_ids: list)
    """
    max_vel = 2.0
    recorded_steps = []
    
    for i in range(max_steps):
        # Continuous IK calculation
        joint_positions = p.calculateInverseKinematics(
            robot_id, end_effector_index, target_pos, target_orn,
            maxNumIterations=100, residualThreshold=1e-5
        )
        
        # Set controls
        for idx_j, joint_idx in enumerate(joint_indices):
            if idx_j < len(joint_positions):
                p.setJointMotorControl2(
                    bodyIndex=robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[idx_j],
                    force=JOINT_FORCE,
                    maxVelocity=max_vel
                )
        
        p.stepSimulation()
        if use_gui:
            time.sleep(1.0 / 240.0)
        
        # Recording at stride interval
        if recorder and view_mat and proj_mat and (i % record_stride == 0):
            rgb, depth = get_camera_image(
                view_mat, proj_mat, 
                width=CAMERA1_CONFIG["width"], 
                height=CAMERA1_CONFIG["height"],
                renderer=p.ER_TINY_RENDERER
            )
            step_id = recorder.record_step(
                robot_id, joint_indices, end_effector_index, 
                (rgb, depth), phase=phase
            )
            recorded_steps.append(step_id)
        
        # Check convergence
        current_state = p.getLinkState(robot_id, end_effector_index)
        current_pos = current_state[0]
        dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        if dist < tolerance:
            return True, dist, recorded_steps
    
    # Final distance
    current_state = p.getLinkState(robot_id, end_effector_index)
    current_pos = current_state[0]
    final_dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
    
    return False, final_dist, recorded_steps


def wait_for_contact(robot_id, ee_index, obj_id, max_steps=60, use_gui=True):
    """
    Wait for contact detection between end-effector and object.
    
    Returns:
        bool: Whether contact was detected
    """
    for _ in range(max_steps):
        if _has_contact(robot_id, ee_index, obj_id):
            return True
        p.stepSimulation()
        if use_gui:
            time.sleep(1.0 / 240.0)
    return False


# ============================================================================
# Episode Collection
# ============================================================================

def run_episode(scene_manager, recorder, oracle, robot_id, joint_indices, 
                end_effector_index, use_gui, view_mat, proj_mat, cam_config,
                record_stride, max_steps_per_phase):
    """
    Execute a single episode of data collection.
    
    Returns:
        tuple: (success: bool, accepted: bool, reasons: list, metrics: dict)
    """
    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    
    # 1. Select random target from scene
    target_info = oracle.get_random_target()
    if target_info is None:
        return False, False, ["No target objects in scene"], {}
    
    obj_id = target_info["id"]
    obj_name = target_info["name"]
    gras_pos = target_info["gras_pos"]  # AABB top center (world coords)
    
    # 2. Generate instruction
    instruction = Oracle.generate_instruction(obj_name)
    log(f"Target: {obj_name}, Instruction: '{instruction}'")
    
    # 3. Set up recorder
    recorder.set_target_info(obj_id, obj_name, gras_pos)
    recorder.start_new_episode(instruction, camera_config=cam_config)
    
    # Update camera config in metadata with matrices
    recorder.metadata["camera"]["view_matrix"] = list(view_mat)
    recorder.metadata["camera"]["proj_matrix"] = list(proj_mat)
    
    # 4. Capture initial frame
    rgb, depth = get_camera_image(
        view_mat, proj_mat,
        width=CAMERA1_CONFIG["width"],
        height=CAMERA1_CONFIG["height"],
        renderer=p.ER_TINY_RENDERER
    )
    recorder.record_step(robot_id, joint_indices, end_effector_index, (rgb, depth), phase="initial")
    
    # 5. Calculate keyframe target positions (Section 3.2)
    contact_x, contact_y, contact_z = gras_pos
    
    keyframe_targets = {
        "hover": [contact_x, contact_y, HOVER_HEIGHT],
        "pre_contact": [contact_x, contact_y, contact_z + PRE_CONTACT_OFFSET],
        "contact": [contact_x, contact_y, contact_z],
        "lift": [contact_x, contact_y, contact_z + LIFT_DELTA]
    }
    
    # 6. Execute 4-phase motion and record keyframes
    keyframe_order = ["hover", "pre_contact", "contact", "lift"]
    contact_detected = False
    initial_obj_z = target_info["pos"][2]
    
    for phase_name in keyframe_order:
        target_pos = keyframe_targets[phase_name]
        log(f"Phase: {phase_name} -> {[f'{v:.3f}' for v in target_pos]}")
        
        # Move to target
        converged, final_dist, recorded_steps = move_arm_with_recording(
            robot_id, joint_indices, end_effector_index,
            target_pos, target_orn,
            max_steps=max_steps_per_phase,
            use_gui=use_gui,
            tolerance=CONVERGENCE_THRESHOLDS[phase_name],
            recorder=recorder,
            view_mat=view_mat,
            proj_mat=proj_mat,
            record_stride=record_stride,
            phase=phase_name
        )
        
        # Record keyframe at convergence
        current_step_id = len(recorder.current_episode["steps"]) - 1
        if current_step_id < 0:
            # Ensure at least one frame is recorded
            rgb, depth = get_camera_image(
                view_mat, proj_mat,
                width=CAMERA1_CONFIG["width"],
                height=CAMERA1_CONFIG["height"],
                renderer=p.ER_TINY_RENDERER
            )
            current_step_id = recorder.record_step(
                robot_id, joint_indices, end_effector_index,
                (rgb, depth), phase=phase_name
            )
        
        recorder.record_keyframe(phase_name, robot_id, end_effector_index, 
                                 target_pos, step_id=current_step_id)
        
        # Special handling for contact phase
        if phase_name == "contact":
            # Wait and check for contact
            contact_detected = wait_for_contact(robot_id, end_effector_index, obj_id, 
                                                max_steps=60, use_gui=use_gui)
            if contact_detected:
                log("Contact detected!")
                # Hold briefly
                for _ in range(30):
                    p.stepSimulation()
                    if use_gui:
                        time.sleep(1.0 / 240.0)
            else:
                log("WARNING: No contact detected")
    
    # 7. Verify success (object lifted)
    obj_pos_final, _ = p.getBasePositionAndOrientation(obj_id)
    height_change = obj_pos_final[2] - initial_obj_z
    success = height_change > 0.05
    
    if success:
        log(f"SUCCESS: Object lifted by {height_change:.3f}m")
    else:
        log(f"FAILURE: Height change {height_change:.3f}m < 0.05m")
    
    # 8. Run quality gate
    quality_gate = QualityGate()
    accepted, reasons, metrics = quality_gate.validate(
        keyframes=recorder.keyframes,
        steps=recorder.current_episode["steps"],
        contact_detected=contact_detected,
        depth_dir=recorder.depth_dir,
        episode_idx=recorder.episode_idx
    )
    
    # 9. Save episode
    recorder.save_episode(
        success=success,
        accepted=accepted,
        quality_reasons=reasons,
        quality_metrics=metrics
    )
    
    return success, accepted, reasons, metrics


# ============================================================================
# Main Collection Loop
# ============================================================================

def setup_simulation(use_gui):
    """Initialize PyBullet simulation."""
    if use_gui:
        try:
            connection_id = p.connect(p.GUI)
            if connection_id < 0:
                raise RuntimeError("GUI connection failed")
            log("Connected to PyBullet with GUI.")
        except Exception:
            p.connect(p.DIRECT)
            log("GUI unavailable, using DIRECT mode.")
            use_gui = False
    else:
        p.connect(p.DIRECT)
        log("Connected to PyBullet in DIRECT mode.")
    
    p.setAdditionalSearchPath(PYBULLET_DATA_PATH)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    return use_gui


def setup_robot(scene_manager):
    """Load and configure robot."""
    robot_path = os.path.join(PROJECT_ROOT, "assets", "ec63", "urdf", "ec63_description.urdf")
    
    # Calculate yaw to face table center
    table_center_xy = np.array([0.8, 0.0])
    rx, ry, rz = ROBOT_BASE_POS
    dx = table_center_xy[0] - rx
    dy = table_center_xy[1] - ry
    yaw = math.atan2(dy, dx)
    spawn_orn = p.getQuaternionFromEuler([0, 0, yaw])
    
    robot_id = p.loadURDF(
        robot_path, 
        basePosition=ROBOT_BASE_POS,
        baseOrientation=spawn_orn,
        useFixedBase=True, 
        flags=p.URDF_USE_SELF_COLLISION
    )
    
    log(f"Robot loaded at {ROBOT_BASE_POS}, yaw={math.degrees(yaw):.1f}°")
    
    # Configure dynamics
    num_joints = p.getNumJoints(robot_id)
    for link_idx in range(-1, num_joints):
        lat_fric = 1.1 if link_idx == (num_joints - 1) else 0.8
        p.changeDynamics(
            bodyUniqueId=robot_id,
            linkIndex=link_idx,
            lateralFriction=lat_fric,
            rollingFriction=0.001,
            spinningFriction=0.02,
            restitution=0.0,
            contactStiffness=20000,
            contactDamping=800
        )
    
    joint_indices = list(range(num_joints))
    end_effector_index = 6 if num_joints > 6 else (num_joints - 1)
    
    # Set initial pose (upright/candle)
    candle_pose = [0, -1.57, 0, -1.57, 0, 0]
    for i, val in enumerate(candle_pose[:len(joint_indices)]):
        p.resetJointState(robot_id, joint_indices[i], val)
    
    return robot_id, joint_indices, end_effector_index, list(spawn_orn)


def run_collection(args):
    """Main collection loop."""
    # Initialize
    use_gui = setup_simulation(args.gui)
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        log(f"Random seed set to {args.seed}")
    
    # Setup scene manager
    scene_manager = SceneManager(os.path.join(PROJECT_ROOT, "assets"))
    
    # Load table
    table_id = scene_manager.load_table()
    if table_id is None:
        log("WARNING: Custom table not found, using default")
    
    # Setup robot
    robot_id, joint_indices, end_effector_index, robot_orn = setup_robot(scene_manager)
    
    # Setup recorder
    save_dir = os.path.join(PROJECT_ROOT, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    recorder = DataRecorder(save_dir)
    recorder.set_robot_base_pose(ROBOT_BASE_POS, robot_orn)
    
    # Setup camera
    view_mat, proj_mat, cam_config = get_camera1_matrices()
    
    # Setup oracle
    oracle = Oracle(scene_manager)
    
    # Statistics tracking
    stats = {
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "success": 0,
        "rejection_reasons": defaultdict(int)
    }
    
    log(f"Starting collection: {args.num_episodes} episodes")
    log(f"Save directory: {save_dir}")
    
    # Collection loop
    for episode_num in range(args.num_episodes):
        log(f"\n{'='*50}")
        log(f"Episode {episode_num + 1}/{args.num_episodes}")
        log(f"{'='*50}")
        
        # Reset scene
        scene_manager.clear_scene()
        
        # Reload table (it gets cleared)
        table_id = scene_manager.load_table()
        
        # Reset robot pose
        candle_pose = [0, -1.57, 0, -1.57, 0, 0]
        for i, val in enumerate(candle_pose[:len(joint_indices)]):
            p.resetJointState(robot_id, joint_indices[i], val)
        
        # Spawn obstacles and random objects
        obstacle_names = scene_manager.spawn_obstacles(
            robot_id=robot_id, 
            robot_base_pos=ROBOT_BASE_POS
        )
        
        max_total = 6
        remaining = max(0, max_total - len(obstacle_names))
        num_objects = random.randint(1, max(1, remaining))
        random_names = scene_manager.spawn_random_objects(
            num_objects, 
            robot_id=robot_id, 
            robot_base_pos=ROBOT_BASE_POS
        )
        
        # Settle objects
        scene_manager.settle_objects()
        
        all_objects = obstacle_names + random_names
        log(f"Scene objects: {all_objects}")
        
        if not all_objects:
            log("WARNING: No objects spawned, skipping episode")
            continue
        
        # Simulate a few steps to stabilize
        for _ in range(50):
            p.stepSimulation()
        
        # Run episode
        try:
            success, accepted, reasons, metrics = run_episode(
                scene_manager=scene_manager,
                recorder=recorder,
                oracle=oracle,
                robot_id=robot_id,
                joint_indices=joint_indices,
                end_effector_index=end_effector_index,
                use_gui=use_gui,
                view_mat=view_mat,
                proj_mat=proj_mat,
                cam_config=cam_config,
                record_stride=args.record_stride,
                max_steps_per_phase=args.max_steps_per_phase
            )
            
            # Update statistics
            stats["total"] += 1
            if success:
                stats["success"] += 1
            if accepted:
                stats["accepted"] += 1
            else:
                stats["rejected"] += 1
                for reason in reasons:
                    # Extract gate type from reason
                    gate = reason.split(":")[0] if ":" in reason else "Unknown"
                    stats["rejection_reasons"][gate] += 1
            
            log(f"Result: success={success}, accepted={accepted}")
            if reasons:
                log(f"Rejection reasons: {reasons}")
                
        except Exception as e:
            log(f"ERROR in episode: {e}")
            import traceback
            traceback.print_exc()
            stats["total"] += 1
            stats["rejected"] += 1
            stats["rejection_reasons"]["Error"] += 1
    
    # Print final statistics
    print_statistics(stats, save_dir)
    
    # Save statistics to file
    stats_file = os.path.join(save_dir, "collection_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(dict(stats), f, indent=2)
    log(f"Statistics saved to {stats_file}")
    
    # Cleanup
    p.disconnect()
    log("Collection completed.")


def print_statistics(stats, save_dir):
    """Print collection statistics summary."""
    print("\n" + "="*60)
    print("COLLECTION STATISTICS")
    print("="*60)
    print(f"Total episodes:     {stats['total']}")
    print(f"Successful:         {stats['success']} ({100*stats['success']/max(1,stats['total']):.1f}%)")
    print(f"Accepted:           {stats['accepted']} ({100*stats['accepted']/max(1,stats['total']):.1f}%)")
    print(f"Rejected:           {stats['rejected']} ({100*stats['rejected']/max(1,stats['total']):.1f}%)")
    print(f"Save directory:     {save_dir}")
    
    if stats["rejection_reasons"]:
        print("\nRejection reasons breakdown:")
        for reason, count in sorted(stats["rejection_reasons"].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count}")
    print("="*60 + "\n")


# ============================================================================
# Episode Inspection Utility
# ============================================================================

def inspect_episode(save_dir, episode_id):
    """
    Inspect a specific episode for debugging/verification.
    
    Args:
        save_dir: Data save directory
        episode_id: Episode ID to inspect
    """
    meta_file = os.path.join(save_dir, "metadata", f"episode_{episode_id}.json")
    
    if not os.path.exists(meta_file):
        print(f"Episode {episode_id} not found at {meta_file}")
        return
    
    with open(meta_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"EPISODE {episode_id} INSPECTION")
    print(f"{'='*60}")
    print(f"Instruction: {data.get('instruction', 'N/A')}")
    print(f"Success: {data.get('success', 'N/A')}")
    print(f"Accepted: {data.get('accepted', 'N/A')}")
    print(f"Num steps: {data.get('num_steps', 0)}")
    
    # Target info
    target = data.get('target', {})
    print(f"\nTarget:")
    print(f"  Name: {target.get('name', 'N/A')}")
    print(f"  Grasp pos: {target.get('gras_pos_world', 'N/A')}")
    
    # Keyframes
    print(f"\nKeyframes:")
    for kf in data.get('keyframes', []):
        print(f"  {kf['name']}:")
        print(f"    Step ID: {kf.get('step_id', 'N/A')}")
        print(f"    EE pos (base): {[f'{v:.4f}' for v in kf.get('ee_position_base', [])]}")
        print(f"    Convergence: {kf.get('convergence_distance', 'N/A'):.4f}m")
    
    # Quality
    quality = data.get('quality', {})
    print(f"\nQuality Gate:")
    print(f"  Accepted: {quality.get('accepted', 'N/A')}")
    if quality.get('reasons'):
        print(f"  Reasons: {quality['reasons']}")
    
    # Show keyframe image paths
    print(f"\nKeyframe images:")
    steps = data.get('steps', [])
    for kf in data.get('keyframes', []):
        step_id = kf.get('step_id', -1)
        if 0 <= step_id < len(steps):
            step = steps[step_id]
            img_path = os.path.join(save_dir, step.get('image_path', ''))
            depth_path = os.path.join(save_dir, step.get('depth_path', ''))
            print(f"  {kf['name']}: {img_path}")
            print(f"            depth: {depth_path}")
    
    print(f"{'='*60}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated VLA Data Collection with Quality Gates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--num_episodes", type=int, default=10,
        help="Number of episodes to collect"
    )
    parser.add_argument(
        "--save_dir", type=str, default="data/keyframe_demos",
        help="Directory to save collected data (relative to project root)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Enable PyBullet GUI visualization"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--record_stride", type=int, default=5,
        help="Record frame every N simulation steps"
    )
    parser.add_argument(
        "--max_steps_per_phase", type=int, default=150,
        help="Maximum simulation steps per motion phase"
    )
    parser.add_argument(
        "--inspect", type=int, default=None,
        help="Inspect a specific episode ID instead of collecting"
    )
    
    args = parser.parse_args()
    
    if args.inspect is not None:
        save_dir = os.path.join(PROJECT_ROOT, args.save_dir)
        inspect_episode(save_dir, args.inspect)
    else:
        run_collection(args)


if __name__ == "__main__":
    main()
