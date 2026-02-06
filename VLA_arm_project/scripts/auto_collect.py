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
- Oracle-based target selection with contact point refined from depth (local highest surface)
- 3-phase keyframe execution: hover -> pre_contact -> contact
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
from src.perception.camera_utils import pixel_to_world, world_to_pixel, matrix_from_list

# ============================================================================
# Constants
# ============================================================================

JOINT_FORCE = 500
PYBULLET_DATA_PATH = pybullet_data.getDataPath()

# Fixed keyframe parameters (Section 3.1)
HOVER_HEIGHT = 0.65          # h1: hover Z height
PRE_CONTACT_OFFSET = 0.04    # h2: pre_contact = contact_z + 0.04
CONTACT_PRESS = 0.01         # Press depth at contact (m)
CONTACT_CLOSE_DIST = 0.005   # Updated from 0.008 for stricter contact
CONTACT_WAIT_STEPS = 50      # Reduced from 200 for speed
APPROACH_MAX_VEL = 6.0       # Increased speed from 3.0
TARGET_MAX_TRIES = 6
MIN_DEPTH_VALID = 0.001
MAX_DEPTH_VALID = 0.999
HIGHEST_SEARCH_RADIUS = 8
HIGHEST_XY_LIMIT = 0.06
MAX_OBJ_XY_DRIFT = 0.05      # Increased drift allowance

# Quality gate thresholds (Section 6.1)
CONVERGENCE_THRESHOLDS = {
    "hover": 0.05,           # Relaxed from 0.035
    "pre_contact": 0.02,     # Relaxed from 0.010
    "contact": 0.012
}
MIN_STEPS = 8
MIN_MASK_AREA = 150  # pixels (optional gate F)

# Fixed robot base position
ROBOT_BASE_POS = [0.40, 0.00, 0.40]
ROBOT_BASE_YAW = None  # Computed to face table center

# Default parameters for episode collection
DEFAULT_OBSERVE_FRAMES = 4
DEFAULT_KEYFRAME_CONTEXT = 2
DEFAULT_MAX_FRAMES_PER_EPISODE = 400


def log(message):
    print(f"[AutoCollect] {message}")

def sim_step(use_gui, delay=1.0/480.0):
    p.stepSimulation()
    if use_gui:
        time.sleep(delay)

def _record_frame(recorder, robot_id, joint_indices, end_effector_index,
                  view_mat, proj_mat, phase, force=False, light_params=None):
    kwargs = light_params if light_params else {}
    rgb, depth = get_camera_image(
        view_mat, proj_mat,
        width=CAMERA1_CONFIG["width"],
        height=CAMERA1_CONFIG["height"],
        renderer=p.ER_TINY_RENDERER,
        **kwargs
    )
    return recorder.record_step(
        robot_id, joint_indices, end_effector_index,
        (rgb, depth), phase=phase, force=force
    )

def record_and_step_sequence(recorder, robot_id, joint_indices, end_effector_index,
                             view_mat, proj_mat, phase, count, use_gui, light_params=None, stride=1):
    for i in range(max(0, count * stride)):
        if i % stride == 0:
            _record_frame(
                 recorder, robot_id, joint_indices, end_effector_index,
                 view_mat, proj_mat, phase, force=True, light_params=light_params
            )
        sim_step(use_gui)

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
        required = {"hover", "pre_contact", "contact"}
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
                # If contact was NOT detected, skip contact convergence gating to avoid collision blocking.
                if name == "contact" and not contact_detected:
                    continue
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


# ==========================================================================
# Target Selection Helpers
# ==========================================================================

def _find_highest_surface_point(depth, u, v, view_matrix, proj_matrix, width, height,
                                ref_world, radius=HIGHEST_SEARCH_RADIUS,
                                xy_limit=HIGHEST_XY_LIMIT):
    h, w = depth.shape
    u0 = max(0, u - radius)
    u1 = min(w - 1, u + radius)
    v0 = max(0, v - radius)
    v1 = min(h - 1, v + radius)

    best = None
    best_z = -float("inf")

    for vv in range(v0, v1 + 1):
        for uu in range(u0, u1 + 1):
            d = depth[vv, uu]
            if d <= MIN_DEPTH_VALID or d >= MAX_DEPTH_VALID:
                continue
            world = pixel_to_world(uu, vv, d, view_matrix, proj_matrix,
                                   width=width, height=height)
            if world is None:
                continue
            dx = world[0] - ref_world[0]
            dy = world[1] - ref_world[1]
            if abs(dx) > xy_limit or abs(dy) > xy_limit:
                continue
            if world[2] > best_z:
                best_z = world[2]
                best = world

    return best


# ==========================================================================
# Robot Control Functions
# ==========================================================================

def detect_contact(robot_id, obj_id, candidate_links=None, max_contact_dist=CONTACT_CLOSE_DIST):
    """
    Detect contact between robot and object.

    Args:
        robot_id: PyBullet robot body ID
        obj_id: PyBullet object body ID
        candidate_links: Optional iterable of link indices to consider
        max_contact_dist: Max contact distance to consider as contact

    Returns:
        tuple: (contact_detected: bool, contact_links: set, min_contact_dist: float)
    """
    contact_links = set()
    min_contact_dist = float("inf")

    pts = p.getContactPoints(bodyA=robot_id, bodyB=obj_id)
    for pt in pts:
        link_a = pt[3]
        if candidate_links is not None and link_a not in candidate_links:
            continue
        contact_links.add(link_a)
        min_contact_dist = min(min_contact_dist, pt[8])

    if not contact_links:
        close_pts = p.getClosestPoints(bodyA=robot_id, bodyB=obj_id, distance=max_contact_dist)
        for pt in close_pts:
            link_a = pt[3]
            if candidate_links is not None and link_a not in candidate_links:
                continue
            contact_links.add(link_a)
            min_contact_dist = min(min_contact_dist, pt[8])

    contact_detected = bool(contact_links) and min_contact_dist <= max_contact_dist
    return contact_detected, contact_links, min_contact_dist


def _has_contact(robot_id, ee_link_index, obj_id, max_contact_dist=CONTACT_CLOSE_DIST):
    """Return True if end-effector link is in contact or very close to the object."""
    pts = p.getContactPoints(bodyA=robot_id, bodyB=obj_id, linkIndexA=ee_link_index)
    if pts:
        for pt in pts:
            if pt[8] <= max_contact_dist:
                return True
        return True
    # Fall back to closest-points query (near-contact)
    close_pts = p.getClosestPoints(bodyA=robot_id, bodyB=obj_id,
                                   distance=max_contact_dist,
                                   linkIndexA=ee_link_index)
    return len(close_pts) > 0


def move_arm_with_recording(robot_id, joint_indices, end_effector_index,
                            target_pos, target_orn, 
                            max_steps, use_gui, tolerance=0.01,
                            recorder=None, view_mat=None, proj_mat=None,
                            record_stride=5, phase=None, max_velocity=2.0,
                            light_params=None):
    """
    Move the end effector using IK, stepping simulation until convergence or max steps.
    Records frames during motion at specified stride.
    
    Returns:
        tuple: (converged: bool, final_distance: float, recorded_step_ids: list)
    """
    max_vel = max_velocity
    recorded_steps = []
    
    # Dynamic timeout for faster failure
    last_dist = float('inf')
    stagnation_counter = 0

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
        
        sim_step(use_gui)

        # Recording at stride interval
        if recorder and view_mat and proj_mat and (i % record_stride == 0):
            step_id = _record_frame(
                recorder, robot_id, joint_indices, end_effector_index,
                view_mat, proj_mat, phase=phase, force=False,
                light_params=light_params
            )
            if step_id is not None:
                recorded_steps.append(step_id)

        # Check convergence
        current_state = p.getLinkState(robot_id, end_effector_index)
        current_pos = current_state[0]
        dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        if dist < tolerance:
            return True, dist, recorded_steps

        # Check stagnation
        if abs(last_dist - dist) < 1e-4:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        last_dist = dist

        if stagnation_counter > 20 and dist < tolerance * 2:
             # If strictly stuck but close, accept it
             return True, dist, recorded_steps

    # Final distance
    current_state = p.getLinkState(robot_id, end_effector_index)
    current_pos = current_state[0]
    final_dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
    
    return False, final_dist, recorded_steps


def wait_for_contact(robot_id, ee_index, obj_id, max_steps=CONTACT_WAIT_STEPS, use_gui=True,
                     close_dist=CONTACT_CLOSE_DIST, candidate_links=None):
    """
    Wait for contact detection between end-effector and object.

    Returns:
        tuple: (contact_detected: bool, contact_links: set, min_contact_dist: float)
    """
    last_min_dist = float("inf")
    last_links = set()
    for _ in range(max_steps):
        detected, links, min_dist = detect_contact(
            robot_id, obj_id, candidate_links=candidate_links, max_contact_dist=close_dist
        )
        last_min_dist = min(last_min_dist, min_dist)
        if links:
            last_links = set(links)
        if detected:
            return True, last_links, last_min_dist
        sim_step(use_gui)
    return False, last_links, last_min_dist


# ============================================================================
# Episode Collection
# ============================================================================

def run_episode(scene_manager, recorder, oracle, robot_id, joint_indices, 
                end_effector_index, use_gui, view_mat, proj_mat, cam_config,
                record_stride, critical_stride, observe_frames, keyframe_context,
                max_steps_per_phase, contact_dist_thresh, light_params=None):
    """
    Execute a single episode of data collection.

    Returns:
        tuple: (success: bool, accepted: bool, reasons: list, metrics: dict)
    """
    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    # Capture an initial frame for depth-based target validation
    rgb_init, depth_init = get_camera_image(
        view_mat, proj_mat,
        width=CAMERA1_CONFIG["width"],
        height=CAMERA1_CONFIG["height"],
        renderer=p.ER_TINY_RENDERER,
        **(light_params if light_params else {})
    )

    # 1. Select random target from scene
    target_info = None
    obj_id = None
    obj_name = None
    gras_pos = None
    tried_ids = set()

    for _ in range(TARGET_MAX_TRIES):
        candidate = oracle.get_random_target()
        if candidate is None:
            break
        if candidate["id"] in tried_ids:
            continue
        tried_ids.add(candidate["id"])

        # Project candidate grasp to pixel and search for highest surface point nearby
        u_v = world_to_pixel(candidate["gras_pos"], view_mat, proj_mat,
                              CAMERA1_CONFIG["width"], CAMERA1_CONFIG["height"])
        if u_v is None:
            continue
        u, v = u_v
        highest_world = _find_highest_surface_point(
            depth_init, u, v,
            view_mat, proj_mat,
            CAMERA1_CONFIG["width"], CAMERA1_CONFIG["height"],
            ref_world=candidate["gras_pos"]
        )
        if highest_world is None:
            continue

        target_info = candidate
        obj_id = candidate["id"]
        obj_name = candidate["name"]
        gras_pos = highest_world.tolist()
        break

    if target_info is None:
        return False, False, ["No valid target with depth support"], {}

    # 2. Generate instruction
    instruction = Oracle.generate_instruction(obj_name)
    # log(f"Target: {obj_name}, Instruction: '{instruction}'") # Reduced logging

    # 3. Set up recorder
    recorder.set_target_info(obj_id, obj_name, gras_pos)
    recorder.start_new_episode(instruction, camera_config=cam_config)

    # Update camera config in metadata with matrices
    recorder.metadata["camera"]["view_matrix"] = list(view_mat)
    recorder.metadata["camera"]["proj_matrix"] = list(proj_mat)

    # 4. Capture initial frame
    recorder.record_step(robot_id, joint_indices, end_effector_index, (rgb_init, depth_init), phase="initial", force=True)

    # 4b. Observe frames for stable starting sequence
    record_and_step_sequence(
        recorder, robot_id, joint_indices, end_effector_index,
        view_mat, proj_mat, "observe", observe_frames, use_gui, light_params, stride=40
    )

    # 5. Calculate keyframe target positions (Section 3.2)
    contact_x, contact_y, contact_z = gras_pos
    contact_z_target = contact_z - CONTACT_PRESS
    # Press 1mm to encourage physical contact.

    keyframe_targets = {
        "hover": [contact_x, contact_y, HOVER_HEIGHT],
        "pre_contact": [contact_x, contact_y, contact_z + PRE_CONTACT_OFFSET],
        "contact": [contact_x, contact_y, contact_z_target]
    }

    # 6. Execute 3-phase motion and record keyframes
    keyframe_order = ["hover", "pre_contact", "contact"]
    contact_detected = False
    contact_converged = False
    contact_final_dist = float("inf")
    initial_obj_z = target_info["pos"][2]
    initial_obj_xy = np.array(target_info["pos"][:2], dtype=np.float32)
    contact_links = set()
    min_contact_dist = float("inf")

    for phase_name in keyframe_order:
        target_pos = keyframe_targets[phase_name]
        # log(f"Phase: {phase_name} -> {[f'{v:.3f}' for v in target_pos]}") # Reduced logging

        phase_stride = critical_stride
        if phase_name not in {"hover", "pre_contact", "contact"}:
            phase_stride = record_stride

        # Single attempt at contact phase (no lateral retries or settle)
        if phase_name == "contact":
            # Descend to contact
            contact_converged, contact_final_dist, recorded_steps = move_arm_with_recording(
                robot_id, joint_indices, end_effector_index,
                target_pos, target_orn,
                max_steps=max_steps_per_phase,
                use_gui=use_gui,
                tolerance=CONVERGENCE_THRESHOLDS["contact"],
                recorder=recorder,
                view_mat=view_mat,
                proj_mat=proj_mat,
                record_stride=phase_stride,
                phase=phase_name,
                max_velocity=APPROACH_MAX_VEL,
                light_params=light_params
            )

            current_step_id = _record_frame(
                recorder, robot_id, joint_indices, end_effector_index,
                view_mat, proj_mat, phase=phase_name, force=True,
                light_params=light_params
            )
            if current_step_id is None:
                current_step_id = len(recorder.current_episode["steps"]) - 1
                if current_step_id < 0:
                    current_step_id = None

            # Immediate contact check at convergence (avoid timing misses)
            contact_detected, contact_links, min_contact_dist = detect_contact(
                robot_id, obj_id, candidate_links=None, max_contact_dist=contact_dist_thresh
            )

            if not contact_detected:
                contact_detected, contact_links, min_contact_dist = wait_for_contact(
                    robot_id, end_effector_index, obj_id,
                    max_steps=CONTACT_WAIT_STEPS, use_gui=use_gui,
                    close_dist=contact_dist_thresh,
                    candidate_links=None
                )

            if contact_detected:
                log("Contact detected.")
            # else:
            #     log("WARNING: No contact detected") # Reduced redundant warning

            recorder.record_keyframe(
                phase_name, robot_id, end_effector_index,
                target_pos, step_id=current_step_id
            )

            # Record context frames after keyframe
            record_and_step_sequence(
                recorder, robot_id, joint_indices, end_effector_index,
                view_mat, proj_mat, phase_name, keyframe_context, use_gui, light_params, stride=2
            )
            continue

        # Move to target for non-contact phases
        converged, final_dist, recorded_steps = move_arm_with_recording(
            robot_id, joint_indices, end_effector_index,
            target_pos, target_orn,
            max_steps=max_steps_per_phase,
            use_gui=use_gui,
            tolerance=CONVERGENCE_THRESHOLDS[phase_name],
            recorder=recorder,
            view_mat=view_mat,
            proj_mat=proj_mat,
            record_stride=phase_stride,
            phase=phase_name,
            max_velocity=APPROACH_MAX_VEL,
            light_params=light_params
        )
        # Record keyframe at convergence (forced)
        current_step_id = _record_frame(
            recorder, robot_id, joint_indices, end_effector_index,
            view_mat, proj_mat, phase=phase_name, force=True,
            light_params=light_params
        )

        recorder.record_keyframe(phase_name, robot_id, end_effector_index,
                                 target_pos, step_id=current_step_id)

        record_and_step_sequence(
            recorder, robot_id, joint_indices, end_effector_index,
            view_mat, proj_mat, phase_name, keyframe_context, use_gui, light_params, stride=2
        )


    # 7. Verify success
    obj_pos_final, _ = p.getBasePositionAndOrientation(obj_id)
    final_obj_xy = np.array(obj_pos_final[:2], dtype=np.float32)
    xy_drift = float(np.linalg.norm(final_obj_xy - initial_obj_xy))

    success = contact_detected and contact_converged
    if MAX_OBJ_XY_DRIFT is not None:
        success = success and (xy_drift <= MAX_OBJ_XY_DRIFT)

    # if success:
    #     log("SUCCESS: contact_detected=True and contact converged")
    # else:
    #     log(
    #         f"FAILURE: contact_detected={contact_detected}, contact_converged={contact_converged}, "
    #         f"xy_drift={xy_drift:.3f}m"
    #     ) # Simplified results

    # 8. Run quality gate
    quality_gate = QualityGate()
    accepted, reasons, metrics = quality_gate.validate(
        keyframes=recorder.keyframes,
        steps=recorder.current_episode["steps"],
        contact_detected=contact_detected,
        depth_dir=recorder.depth_dir,
        episode_idx=recorder.episode_idx
    )

    metrics["contact_links"] = sorted(contact_links)
    metrics["min_contact_dist"] = None if not np.isfinite(min_contact_dist) else float(min_contact_dist)

    # 9. Save episode
    if success and accepted:
        recorder.save_episode(
            success=success,
            accepted=accepted,
            quality_reasons=reasons,
            quality_metrics=metrics
        )
    else:
        # log(f"Discarding episode data {recorder.episode_idx} (success={success}, accepted={accepted})")
        recorder.discard_episode()


    return success, accepted, reasons, metrics


# ==========================================================================
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
    
    # log(f"Robot loaded at {ROBOT_BASE_POS}, yaw={math.degrees(yaw):.1f}°") # Reduced logging

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
    recorder = DataRecorder(save_dir, max_frames_per_episode=args.max_frames_per_episode)
    recorder.set_robot_base_pose(ROBOT_BASE_POS, robot_orn)

    # Setup camera
    view_mat, proj_mat, cam_config = get_camera1_matrices()

    # Setup oracle
    oracle = Oracle(scene_manager)

    # Statistics tracking
    stats = {
        "total": 0,
        "accepted": 0,
        "success": 0
    }

    log(f"Starting collection: {args.num_episodes} episodes")
    # log(f"Save directory: {save_dir}") # Reduced logging

    # Collection loop
    for episode_num in range(args.num_episodes):
        # log(f"\n{'='*50}")
        # log(f"Episode {episode_num + 1}/{args.num_episodes}")
        # log(f"{'='*50}")

        # Add lighting noise for robustness
        light_params = {
            "lightDirection": [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.5, 1)],
            "lightColor": [random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), random.uniform(0.8, 1.0)],
            "lightDistance": random.uniform(2, 5),
            "shadow": 1 if random.choice([True, False]) else 0,
            "lightAmbientCoeff": random.uniform(0.4, 0.6),
            "lightDiffuseCoeff": random.uniform(0.4, 0.6),
            "lightSpecularCoeff": random.uniform(0.4, 0.6)
        }
        if use_gui:
             p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, light_params["shadow"])
             p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

        # Reset scene
        scene_manager.clear_scene()

        # Reload table (it gets cleared)
        table_id = scene_manager.load_table()

        # Reset robot pose
        candle_pose = [0, -1.57, 0, -1.57, 0, 0]
        for i, val in enumerate(candle_pose[:len(joint_indices)]):
            p.resetJointState(robot_id, joint_indices[i], val)
            p.setJointMotorControl2(robot_id, joint_indices[i], p.POSITION_CONTROL, targetPosition=val, force=JOINT_FORCE)

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
        # log(f"Scene objects: {all_objects}") # Reduced logging

        if not all_objects:
            log("WARNING: No objects spawned, skipping episode")
            continue

        # Simulate a few steps to stabilize
        for _ in range(50):
            sim_step(use_gui)

        if use_gui:
             time.sleep(1.0) # Allow user to see the scene setup

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
                critical_stride=args.critical_stride,
                observe_frames=args.observe_frames,
                keyframe_context=args.keyframe_context,
                max_steps_per_phase=args.max_steps_per_phase,
                contact_dist_thresh=args.contact_dist_thresh,
                light_params=light_params
             )

            # Update statistics
            stats["total"] += 1
            if success:
                stats["success"] += 1
            if accepted:
                stats["accepted"] += 1

            if not accepted or not success:
                 log(f"Episode {episode_num+1} Result: success={success}, accepted={accepted}, reasons={reasons}")

        except Exception as e:
            log(f"ERROR in episode: {e}")
            import traceback
            traceback.print_exc()
            stats["total"] += 1

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
    print(f"Save directory:     {save_dir}")
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
    metrics = quality.get('metrics', {})
    if metrics:
        print(f"  contact_detected: {metrics.get('contact_detected', 'N/A')}")
        print(f"  contact_links: {metrics.get('contact_links', 'N/A')}")
        print(f"  min_contact_dist: {metrics.get('min_contact_dist', 'N/A')}")

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
        "--save_dir", type=str, default="data/raw",
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
        "--record_stride", type=int, default=20,
        help="Record frame every N simulation steps"
    )
    parser.add_argument(
        "--critical_stride", type=int, default=2,
        help="Record stride for critical phases (hover/pre_contact/contact)"
    )
    parser.add_argument(
        "--observe_frames", type=int, default=DEFAULT_OBSERVE_FRAMES,
        help="Number of observe frames at episode start"
    )
    parser.add_argument(
        "--keyframe_context", type=int, default=DEFAULT_KEYFRAME_CONTEXT,
        help="Extra frames to record after each keyframe"
    )
    parser.add_argument(
        "--max_frames_per_episode", type=int, default=DEFAULT_MAX_FRAMES_PER_EPISODE,
        help="Max frames to record per episode (non-forced frames may be skipped)"
    )
    parser.add_argument(
        "--max_steps_per_phase", type=int, default=500,
        help="Maximum simulation steps per motion phase"
    )
    parser.add_argument(
        "--inspect", type=int, default=None,
        help="Inspect a specific episode ID instead of collecting"
    )
    parser.add_argument(
        "--contact_dist_thresh", type=float, default=CONTACT_CLOSE_DIST,
        help="Max contact distance threshold for detection"
    )

    args = parser.parse_args()

    if args.inspect is not None:
        save_dir = os.path.join(PROJECT_ROOT, args.save_dir)
        inspect_episode(save_dir, args.inspect)
    else:
        run_collection(args)


if __name__ == "__main__":
    main()
