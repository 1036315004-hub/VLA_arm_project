import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import sys
import time
import random

import numpy as np

# Adjust path to Project Root
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure src is also in path if needed for explicit references (common IDE fix)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.perception.camera_utils import get_camera_image
from src.perception.scene_manager import SceneManager
from src.perception.recorder import DataRecorder

# define log_init if not defined
def log_init(message):
    print(f"[Init] {message}")


import pybullet as p
import pybullet_data

JOINT_FORCE = 500
ZERO_VECTOR = [0, 0, 0]
PYBULLET_DATA_PATH = pybullet_data.getDataPath()
if PYBULLET_DATA_PATH not in sys.path:
    sys.path.insert(0, PYBULLET_DATA_PATH)

# Multi-stage scanning algorithm constants
# Area threshold for clear detection in global scan (large object clearly visible)
MIN_CLEAR_DETECTION_AREA = 2000
# Area threshold for good enough detection to proceed to Stage 3
MIN_GOOD_DETECTION_AREA = 800
# Pixel offset threshold for center alignment (if target is off by more pixels, do micro-adjust)
CENTER_OFFSET_THRESHOLD = 100
# Height adjustment for micro-positioning pass (in meters)
MICRO_ADJUST_HEIGHT_OFFSET = 0.05


def log(message):
    print(f"[main_grasp] {message}")


def _has_contact(robot_id, ee_link_index, obj_id, max_contact_dist=0.001):
    """Return True if end-effector link is in contact with the object."""
    pts = p.getContactPoints(bodyA=robot_id, bodyB=obj_id, linkIndexA=ee_link_index)
    for pt in pts:
        if pt[8] <= max_contact_dist:
            return True
    return False


def connect_pybullet():
    try:
        connection_id = p.connect(p.GUI)
        if connection_id >= 0:
            log("Connected to PyBullet with GUI.")
            return True
    except Exception as exc:
        log(f"GUI connection failed ({exc}), falling back to DIRECT.")
    p.connect(p.DIRECT)
    log("Connected to PyBullet with DIRECT.")
    return False


def matrix_from_list(values):
    """Return a 4x4 matrix from a flat list in column-major order."""
    return np.array(values, dtype=np.float32).reshape((4, 4), order="F")


def move_arm(robot_id, joint_indices, end_effector_index, target_pos, target_orn, steps, sleep, tolerance=0.01, recorder=None, cam_params=None):
    """
    Move the end effector using IK, stepping simulation until convergence or max steps.
    """
    # Reduce speed implies we need more steps to reach destination
    max_steps = steps * 2
    max_vel = 2.0  # rad/s, Slower speed per request

    for i in range(max_steps):
        # Continuous IK calculation for better tracking
        joint_positions = p.calculateInverseKinematics(
            robot_id, end_effector_index, target_pos, target_orn,
            maxNumIterations=100, residualThreshold=1e-5
        )

        # Set controls with Velocity Limit
        # p.setJointMotorControlArray does not support maxVelocities, using loop instead
        for idx_j, joint_idx in enumerate(joint_indices):
            # Ensure we don't index out of bounds if IK returns fewer joints than we have indices
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
        if sleep:
            time.sleep(1.0 / 240.0)

        # RECORDING
        if recorder and cam_params and i % 5 == 0: # Record every 5 steps
            view_mat, proj_mat = cam_params
            rgb, _ = get_camera_image(view_mat, proj_mat, renderer=p.ER_TINY_RENDERER) # TINY for speed
            recorder.record_step(robot_id, joint_indices, end_effector_index, (rgb, None))

        # Check convergence occasionally
        if i % 10 == 0:
            current_state = p.getLinkState(robot_id, end_effector_index)
            current_pos = current_state[0]
            dist = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            if dist < tolerance:
                # log(f"Target reached within {tolerance}m at step {i}.")
                break


def move_arm_safe(robot_id, joint_indices, end_effector_index, target_pos, target_orn, use_gui):
    """
    Moves arm safely: Lift -> Traverse -> Descend
    Avoids hitting objects by moving at a safe height.
    """
    # 1. Get current EE pose
    current_state = p.getLinkState(robot_id, end_effector_index)
    current_pos = current_state[0]

    safe_height = 0.7  # Safe Z plane (above longest object)

    # 2. Lift if currently low
    if current_pos[2] < safe_height - 0.05:
        log("Path Planning: Lifting...")
        lift_pos = [current_pos[0], current_pos[1], safe_height]
        move_arm(robot_id, joint_indices, end_effector_index, lift_pos, target_orn, 50, use_gui)

    # 3. Traverse at Safe Height
    log("Path Planning: Traversing...")
    traverse_pos = [target_pos[0], target_pos[1], safe_height]
    move_arm(robot_id, joint_indices, end_effector_index, traverse_pos, target_orn, 80, use_gui)

    # 4. Descend
    log("Path Planning: Descending...")
    move_arm(robot_id, joint_indices, end_effector_index, target_pos, target_orn, 60, use_gui)


def get_eye_in_hand_image(robot_id, end_effector_index, width=640, height=480, use_gui=True):
    """
    Simulates a camera attached to the end effector.
    """
    # Get End Effector Pose
    link_state = p.getLinkState(robot_id, end_effector_index, computeForwardKinematics=True)
    ee_pos = link_state[0]
    ee_orn = link_state[1]

    # Rotation matrix
    rot_matrix = p.getMatrixFromQuaternion(ee_orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    # Camera Position: Slightly offset from EE center
    cam_eye_local = np.array([0, 0, 0.05])
    cam_eye_world = np.array(ee_pos) + rot_matrix @ cam_eye_local

    # Camera Target: Look 'forward' (along Z axis typically for Kuka flange)
    cam_target_world = cam_eye_world + rot_matrix @ np.array([0, 0, 1.0])

    # Camera Up: Align with Y
    cam_up_world = rot_matrix @ np.array([0, 1, 0])

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_eye_world,
        cameraTargetPosition=cam_target_world,
        cameraUpVector=cam_up_world
    )

    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=2.0
    )

    renderer = p.ER_BULLET_HARDWARE_OPENGL if use_gui else p.ER_TINY_RENDERER
    _, _, rgba, depth, _ = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=renderer
    )

    rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
    depth_buffer = np.reshape(depth, (height, width))

    return rgb, depth_buffer, view_matrix, proj_matrix


def move_to_observation_pose(robot_id, joint_indices, use_gui):
    """
    Moves the robot to a retracted pose to avoid occlusion.
    """
    log("Moving to Observation Pose (Retracted)...")
    # Pose: Base=0, Shoulder=-90 (-1.57), Elbow=120 (+2.1), ...
    # This pulls the arm back and up ('C' shape)
    retracted_joints = [0, -1.57, 2.1, 0, 1.57, 0]

    # Pad if more joints
    target_pos = retracted_joints[:]
    if len(joint_indices) > len(target_pos):
        target_pos += [0] * (len(joint_indices) - len(target_pos))

    steps = 100
    for _ in range(steps):
        p.setJointMotorControlArray(robot_id, joint_indices, p.POSITION_CONTROL, targetPositions=target_pos[:len(joint_indices)])
        p.stepSimulation()
        if use_gui: time.sleep(1./240.)



def run_trial(use_gui):

    # Initialize Scene Manager
    scene_manager = SceneManager(os.path.join(PROJECT_ROOT, "assets"))
    recorder = DataRecorder(os.path.join(PROJECT_ROOT, "data", "expert_demos"))
    robot_id = None
    recording_enabled = False # Disable recording per user request

    try:
        # --- Load Table ---
        # SceneManager handles table now, or we can keep it here if preferred.
        # But SceneCleaner logic is inside SceneManager, so better to use it.
        table_id = scene_manager.load_table()
        if table_id is None:
            # Fallback if custom table fails
            table_path = os.path.join(PROJECT_ROOT, "assets", "objects", "table.urdf")
            p.loadURDF(table_path, basePosition=[0.8, 0, 0], useFixedBase=True)

        table_z_surface = scene_manager.table_z

        # === 1. Load Robot FIRST (Fixed Placement) ===
        log("Objects will be spawned around the robot. Loading robot first...")
        robot_path = os.path.join(PROJECT_ROOT, "assets", "ec63", "urdf", "ec63_description.urdf")

        robot_id = None
        spawn_z = 0.40 # Maintain original Z height
        table_center_xy = np.array([0.8, 0.0])
        robot_base_pos = None

        # Fixed robot base position (must stay within table bounds)
        fixed_robot_pose = [0.40, 0.00, spawn_z]

        # Fixed camera position corresponding to robot pose
        fixed_camera_pose = {"eye": [0.60, 0.60, 0.90], "target": [0.80, 0.00, 0.40]}

        pose_idx = 0
        rx, ry, rz = fixed_robot_pose
        dx = table_center_xy[0] - rx
        dy = table_center_xy[1] - ry
        yaw = math.atan2(dy, dx)
        spawn_orn = p.getQuaternionFromEuler([0, 0, yaw])

        robot_id = p.loadURDF(robot_path, basePosition=[rx, ry, rz], baseOrientation=spawn_orn,
                              useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        robot_base_pos = [rx, ry, rz]
        log(f"Robot spawned at base: [{rx:.2f}, {ry:.2f}, {rz:.2f}], Yaw: {math.degrees(yaw):.1f} deg")

        # Apply dynamics to robot links
        num_joints = p.getNumJoints(robot_id)
        for link_idx in range(-1, num_joints):
            # Default values
            lat_fric = 0.8
            roll_fric = 0.001
            spin_fric = 0.02
            restitution = 0.0
            stiffness = 20000
            damping = 800

            # Use higher friction for end-effector (assuming last link)
            if link_idx == (num_joints - 1):
                lat_fric = 1.1 # 1.0~1.2

            p.changeDynamics(bodyUniqueId=robot_id,
                             linkIndex=link_idx,
                             lateralFriction=lat_fric,
                             rollingFriction=roll_fric,
                             spinningFriction=spin_fric,
                             restitution=restitution,
                             contactStiffness=stiffness,
                             contactDamping=damping)

        joint_indices = list(range(num_joints))

        # --- STEP 2: Robot Initial Pose (Upright / Candle) ---
        # "Candle" pose: [0, -1.57, 0, -1.57, 0, 0]
        candle_pose = [0, -1.57, 0, -1.57, 0, 0]
        if len(joint_indices) > len(candle_pose):
            candle_pose += [0] * (len(joint_indices) - len(candle_pose))

        log("Setting Robot to Upright (Candle) pose for initial layout...")
        for i, val in enumerate(candle_pose):
             if i < len(joint_indices):
                 p.resetJointState(robot_id, joint_indices[i], val)
        p.setJointMotorControlArray(robot_id, joint_indices, p.POSITION_CONTROL, targetPositions=candle_pose[:len(joint_indices)])
        p.stepSimulation()

        # === 3. Spawn Obstacles ===
        scene_manager.clear_scene()
        log("Spawning obstacles...")
        obstacle_names = scene_manager.spawn_obstacles(robot_id=robot_id, robot_base_pos=robot_base_pos)

        # === 4. Spawn Random Objects (cap total at 6) ===
        max_total = 6
        remaining = max(0, max_total - len(obstacle_names))
        num_objects = random.randint(0, remaining) if remaining > 0 else 0
        log(f"Spawning {num_objects} random objects...")
        random_names = scene_manager.spawn_random_objects(num_objects, robot_id=robot_id, robot_base_pos=robot_base_pos)

        # === 5. Final Settle ===
        scene_manager.settle_objects()

        generated_objects_names = obstacle_names + random_names
        object_ids = scene_manager.loaded_objects

        # --- STEP 6: Setup Camera (Fixed Pose) ---
        cam_cfg = fixed_camera_pose
        view_mat = p.computeViewMatrix(cameraEyePosition=cam_cfg["eye"],
                                       cameraTargetPosition=cam_cfg["target"],
                                       cameraUpVector=[0, 0, 1])
        proj_mat = p.computeProjectionMatrixFOV(
            fov=60, aspect=640 / 480, nearVal=0.1, farVal=2.0
        )
        cam_pos = cam_cfg["eye"]

        if use_gui:
            p.addUserDebugLine(cam_pos, cam_cfg["target"], [0,1,0], lifeTime=10, lineWidth=3)

        # --- STEP 7: Skipped (Robot stays in Upright/Candle Pose) ---
        # User request: Start grasp directly from upright pose without moving to Standby/Home.

        end_effector_index = 6 if num_joints > 6 else (num_joints - 1)
        for _ in range(50): p.stepSimulation()

        # --- User Input ---
        print("\n" + "=" * 50)
        print(f" Objects: {', '.join(generated_objects_names)}")
        print(" Please enter command (e.g. 'pick up the blue cup').")
        print("=" * 50)

        text_query = input(" >>> Command: ").strip()
        if not text_query:
            log("No input. Ending trial.")
            return

        # Capture Image
        log("Capturing from pre-calculated Smart Camera view...")
        light_color = [random.uniform(0.95, 1.05), random.uniform(0.95, 1.05), random.uniform(0.95, 1.05)]
        light_dir = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(0.8, 1.0)]

        rgb, depth_buffer = get_camera_image(
            view_mat, proj_mat,
            lightColor=light_color,
            lightDirection=light_dir,
            lightDistance=2.0,
            lightAmbientCoeff=random.uniform(0.6, 0.8),
            lightDiffuseCoeff=random.uniform(0.8, 1.0),
            lightSpecularCoeff=random.uniform(0.0, 0.2),
            shadow=1
        )

        # --- ORACLE SEARCH STRATEGY (Ground Truth) ---
        from src.perception.oracle import Oracle
        oracle = Oracle(scene_manager)

        # Start Recording Episode
        if recording_enabled:
            recorder.start_new_episode(text_query, [0,0,0])
            recorder.record_step(robot_id, joint_indices, end_effector_index, (rgb, None))

        # Query Oracle instead of VLM
        target_obj_state = oracle.find_best_target(text_query)

        target_world = None
        closest_obj_id = -1
        initial_obj_z = 0.0

        if target_obj_state:
            log(f"Oracle identified target: {target_obj_state['name']} (ID {target_obj_state['id']})")

            # Ground Truth Position
            true_pos = target_obj_state['pos']
            closest_obj_id = target_obj_state['id']
            initial_obj_z = true_pos[2] # Store initial Z

            if recording_enabled:
                recorder.metadata["ground_truth_target_pos"] = list(true_pos)

            # Inject Noise for Robustness (+/- 2mm)
            noise = np.random.normal(0, 0.002, 3)
            target_world = np.array(true_pos) + noise
            log(f"Ground Truth: {true_pos} | Noised Target: {target_world}")
        else:
            log("Oracle could not find target object.")
            return

        # VLM part removed/replaced
        # target_info = detect_target_from_text(rgb, vlm, text_query) ...

        # --- Grasping Execution ---
        if closest_obj_id != -1:
            obj_pos, _ = p.getBasePositionAndOrientation(closest_obj_id)
            log(f"Snapping to object {closest_obj_id}")
        else:
            log("Invalid Target ID.")
            return

        target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

        # 1. Identify COM (Using Base Position)
        if closest_obj_id != -1:
            obj_pos, _ = p.getBasePositionAndOrientation(closest_obj_id)
            com_target = np.array(obj_pos)
            final_target = com_target + np.random.normal(0, 0.002, 3)
            log(f"Refined Target (COM): {final_target}")
        else:
            final_target = [target_world[0], target_world[1], table_z_surface + 0.05]

        # 2. Hover
        cam_params = (view_mat, proj_mat)
        hover_height = 0.65
        hover_pos = [final_target[0], final_target[1], hover_height]
        log(f"Phase 1: Hover at {hover_pos}")
        move_arm(robot_id, joint_indices, end_effector_index, hover_pos, target_orn, 100, use_gui, recorder=recorder if recording_enabled else None, cam_params=cam_params)

        # 3. Descend to COM
        log("Phase 2: Descend/Contact COM...")
        contact_pos = [final_target[0], final_target[1], final_target[2]]
        move_arm(robot_id, joint_indices, end_effector_index, contact_pos, target_orn, 80, use_gui, recorder=recorder if recording_enabled else None, cam_params=cam_params)

        # Wait for contact, then hold
        contact_happened = False
        for _ in range(60):
            if _has_contact(robot_id, end_effector_index, closest_obj_id):
                contact_happened = True
                break
            p.stepSimulation()
            if use_gui: time.sleep(1.0 / 240.0)

        if contact_happened:
            hold_time = random.uniform(0.2, 0.5)
            log(f"Contact detected. Holding for {hold_time:.2f}s...")
            time.sleep(hold_time)
        else:
            log("Contact not detected. Proceeding with small lift anyway.")

        # 4. Light lift (8-12cm)
        lift_delta = random.uniform(0.08, 0.12)
        lift = [final_target[0], final_target[1], final_target[2] + lift_delta]
        log(f"Light lift by {lift_delta:.3f} m...")
        move_arm(robot_id, joint_indices, end_effector_index, lift, target_orn, 80, use_gui, recorder=recorder if recording_enabled else None, cam_params=cam_params)

        # Verification: Check Z-Height Change
        success = False
        if closest_obj_id != -1:
             obj_pos_final, _ = p.getBasePositionAndOrientation(closest_obj_id)
             # Success if current Z is significantly higher than initial Z (> 5cm)
             if (obj_pos_final[2] - initial_obj_z) > 0.05:
                 success = True

        if success:
            log("SUCCESS: Object lifted (Z-height change detected).")
            if recording_enabled:
                recorder.save_episode(success=True)
        else:
            log("FAILURE: Grasp failed (Object not lifted).")
            if recording_enabled:
                recorder.save_episode(success=False)

        log("Trial completed. Holding pose.")
        time.sleep(5)

    finally:
        if p.isConnected():
            scene_manager.clear_scene()
            if robot_id is not None:
                p.removeBody(robot_id)
            pass


def get_grasp_target(obj_id):
    """Get the top-center of the object for suction grasping."""
    aabb_min, aabb_max = p.getAABB(obj_id)
    center = [
        (aabb_min[0] + aabb_max[0]) / 2,
        (aabb_min[1] + aabb_max[1]) / 2,
        aabb_max[2] # Top surface Z
    ]
    return np.array(center)


def main():
    use_gui = connect_pybullet()
    p.setAdditionalSearchPath(PYBULLET_DATA_PATH)
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    # Robot loading has been moved to run_trial to ensure objects settle first.

    # Run loop for continuous collection if desired, or single trial
    while True:
        log("\n=== Starting User Loop ===")
        run_trial(use_gui)

        cmd = input("Press Enter to restart with new scene, or 'q' to quit: ")
        if cmd.lower() == 'q':
            break


    log("Simulation Session Ended.")
    # Keep window open briefly
    time.sleep(2)


if __name__ == "__main__":
    main()
