import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import sys
import time
import random
import cv2

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

from src.perception.camera_utils import get_random_eye_to_hand_pose, get_camera_image
from src.perception.camera_utils import pixel_to_world as pixel_to_world_utils
from src.perception.scene_manager import SceneManager

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


def check_color_in_bbox(rgb, bbox, color_name):
    """
    Verify if the dominant color in the bbox matches the requested color name using HSV.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Clip to image
    x1 = max(0, min(x1, rgb.shape[1]-1))
    x2 = max(0, min(x2, rgb.shape[1]-1))
    y1 = max(0, min(y1, rgb.shape[0]-1))
    y2 = max(0, min(y2, rgb.shape[0]-1))

    if x2 <= x1 or y2 <= y1: return False

    roi = rgb[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Define HSV range for the color
    mask = None
    if color_name == "red":
        mask1 = cv2.inRange(hsv_roi, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv_roi, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = mask1 | mask2
    elif color_name == "green":
        mask = cv2.inRange(hsv_roi, np.array([35, 70, 50]), np.array([85, 255, 255]))
    elif color_name == "blue":
        mask = cv2.inRange(hsv_roi, np.array([95, 70, 50]), np.array([145, 255, 255]))
    elif color_name == "yellow":
        mask = cv2.inRange(hsv_roi, np.array([15, 70, 50]), np.array([45, 255, 255]))
    elif color_name == "black":
        mask = cv2.inRange(hsv_roi, np.array([0, 0, 0]), np.array([180, 255, 50]))
    elif color_name == "white":
        mask = cv2.inRange(hsv_roi, np.array([0, 0, 200]), np.array([180, 50, 255]))

    if mask is not None:
        # Check ratio of pixels matching color
        ratio = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
        # If > 10% of the bbox is the target color, acceptable for simple shapes
        return ratio > 0.10

    return True # If color not defined, assume pass


def build_vlm():
    try:
        from src.perception.expert_teacher import VLM_YOLO_CLIP

        yolo_model = os.path.join(PROJECT_ROOT, "checkpoints", "yolov8n.pt")
        if not os.path.exists(yolo_model):
            log(f"Warning: YOLO model not found at {yolo_model}")
            yolo_model = "yolov8n.pt" # Fallback or error

        vlm = VLM_YOLO_CLIP(yolo_model=yolo_model)
        log("Initialized VLM_YOLO_CLIP.")
        return vlm
    except Exception as exc:
        log(f"VLM import failed. ({exc})")
        return None


def detect_target_from_text(rgb, vlm, text_query):
    """
    Detect the target object in the RGB image using VLM or HSV color segmentation.

    Returns:
        tuple (cx, cy, area) if target found, None otherwise.
        - cx, cy: center pixel coordinates
        - area: bounding box or contour area (used for determining detection quality)
    """
    # Extract intended color and shape from text for verification
    text_lower = text_query.lower()
    target_color = None
    for c in ["red", "green", "blue", "yellow", "black", "white"]:
        if c in text_lower:
            target_color = c
            break

    # VLM Branch
    if vlm is not None:
        try:
            # Get top 3 candidates to filter
            detections = vlm.query_image(rgb, text_query, topk=3)
        except Exception as exc:
            log(f"VLM query failed. ({exc})")
            detections = []

        best_det = None
        for det in detections:
            # Verify color if specified
            if target_color:
                if check_color_in_bbox(rgb, det["bbox"], target_color):
                    best_det = det
                    break
                else:
                    log(f"VLM candidate '{det['label']}' rejected: Color mismatch for {target_color}.")
            else:
                best_det = det
                break

        if best_det:
            log(f"VLM selected '{best_det['label']}' with score {best_det.get('clip_score', 0):.2f}")
            x1, y1, x2, y2 = map(int, best_det["bbox"])
            x1 = max(0, min(x1, rgb.shape[1] - 1))
            x2 = max(0, min(x2, rgb.shape[1] - 1))
            y1 = max(0, min(y1, rgb.shape[0] - 1))
            y2 = max(0, min(y2, rgb.shape[0] - 1))
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            return (center_x, center_y, area)

    # Enhanced fallback for colors using HSV
    log(f"VLM failed or target filtered, falling back to HSV color segmentation for '{text_lower}'.")

    # Convert to HSV for robust color detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = None

    # HSV Ranges (OpenCV H: 0-179, S: 0-255, V: 0-255)
    if "red" in text_lower:
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
    elif "green" in text_lower:
        mask = cv2.inRange(hsv, np.array([35, 70, 50]), np.array([85, 255, 255]))
    elif "blue" in text_lower:
        mask = cv2.inRange(hsv, np.array([95, 70, 50]), np.array([145, 255, 255]))
    elif "yellow" in text_lower:
        mask = cv2.inRange(hsv, np.array([15, 70, 50]), np.array([45, 255, 255]))
    elif "black" in text_lower:
         mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    elif "white" in text_lower:
         mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))

    if mask is not None:
        # Morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_cands = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (<20) and large objects like robot parts (>3000)
            if 20 < area < 3000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    valid_cands.append((area, cX, cY))

        if valid_cands:
            # Pick largest valid blob
            valid_cands.sort(key=lambda x: x[0], reverse=True)
            best_area, best_cX, best_cY = valid_cands[0]
            log(f"Found {len(valid_cands)} candidate contours. Best area: {best_area}")
            return (best_cX, best_cY, best_area)

    log("Target detection failed.")
    return None


def move_arm(robot_id, joint_indices, end_effector_index, target_pos, target_orn, steps, sleep, tolerance=0.01):
    """
    Move the end effector using IK, stepping simulation until convergence or max steps.
    """
    # Reduce speed implies we need more steps to reach destination
    max_steps = steps * 2
    max_vel = 2.5  # rad/s, Fast speed

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
    robot_id = None

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

        # --- Generate Random Objects (Real Messy Scene) ---
        num_objects = random.randint(3, 5)
        log(f"--- Setup --- Generating {num_objects} realistic objects...")

        generated_objects_names = scene_manager.generate_messy_scene(num_objects)
        object_ids = scene_manager.loaded_objects

        # === Load Robot AFTER Objects Settle ===
        log("Objects settled. Loading robot...")
        robot_path = os.path.join(PROJECT_ROOT, "assets", "ec63", "urdf", "ec63_description.urdf")
        # Load EC63 with self-collision enabled
        robot_base_pos = [0.35, 0.0, 0.40]
        robot_base_orn = [0, 0, 0, 1]
        robot_id = p.loadURDF(robot_path, basePosition=robot_base_pos, baseOrientation=robot_base_orn,
                              useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        num_joints = p.getNumJoints(robot_id)
        joint_indices = list(range(num_joints))
        end_effector_index = 6 if num_joints > 6 else (num_joints - 1)

        # Reset Arm to Home - Randomized Start Pose
        # Increase randomization range for visibility
        base_home = [0, -0.5, 1.0, 0, 0.5, 0]
        # Add noise +/- 0.3 rad (approx 17 degrees)
        home_pos = [val + random.uniform(-0.3, 0.3) for val in base_home]

        # Pad with zeros
        if len(joint_indices) > len(home_pos):
            home_pos += [0] * (len(joint_indices) - len(home_pos))
        else:
            home_pos = home_pos[:len(joint_indices)]

        p.setJointMotorControlArray(robot_id, joint_indices, p.POSITION_CONTROL, targetPositions=home_pos)
        for _ in range(50): p.stepSimulation()

        # --- User Input (Blocking) ---
        print("\n" + "=" * 50)
        print(f" Objects available: {', '.join(generated_objects_names)}")
        print(" Please enter command (e.g. 'pick up the blue cup').")
        print("=" * 50)

        # This will block until user types in the console
        text_query = input(" >>> Command: ").strip()

        if not text_query:
            log("No input provided. Ending trial.")
            return

        # --- EYE-TO-HAND SEARCH STRATEGY ---
        vlm = build_vlm()
        found_target = False
        target_world = None

        # STAGE 1: Solve Occlusion
        # Move robot out of the way before taking the picture
        move_to_observation_pose(robot_id, joint_indices, use_gui)

        # STAGE 2: External Camera Capture
        # Generate a random camera position looking at the table center
        log("Generating synthetic Eye-to-Hand camera view...")
        table_center = [0.8, 0.0, table_z_surface]
        view_mat, proj_mat, cam_pos = get_random_eye_to_hand_pose(table_center)

        # Visual Debug of Camera Line
        if use_gui:
            p.addUserDebugLine(cam_pos, table_center, [1,1,0], lifeTime=5)

        rgb, depth_buffer = get_camera_image(view_mat, proj_mat)

        # STAGE 3: Detection
        target_info = detect_target_from_text(rgb, vlm, text_query)

        if target_info:
            cx, cy, area = target_info
            log(f"Target detected at pixels ({cx}, {cy}). Computing world coordinates...")

            # Use utility function for pixel_to_world which calculates inverse matrix internally
            # Note: pixel_to_world_utils expects (u, v, depth, view_mat, proj_mat, w, h)
            target_world = pixel_to_world_utils(cx, cy, depth_buffer[cy, cx], view_mat, proj_mat, 640, 480)

            log(f"Computed World Position: {target_world}")
            found_target = True
        else:
            log("Target not detected in Eye-to-Hand view.")
            return

        # --- Grasping Execution ---
        # No need for secondary scans if Eye-to-Hand gives good global coordinate
        # The external camera with Depth is usually quite accurate in PyBullet

        closest_obj_id = -1
        # Slightly larger search radius since camera angle might introduce minor parallax errors
        # if not perfectly calibrated, but eye-in-hand is usually accurate.
        min_dist = 0.15

        for obj_id in object_ids:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            # 2D distance check
            dist = np.linalg.norm(np.array(pos[:2]) - target_world[:2])
            if dist < min_dist:
                min_dist = dist
                closest_obj_id = obj_id

        if closest_obj_id != -1:
            obj_pos, _ = p.getBasePositionAndOrientation(closest_obj_id)
            log(f"Snapping to object {closest_obj_id} (correction: {min_dist:.3f}m)")
            target_world = np.array(obj_pos)
        else:
            log("Warning: No object near detected point.")

        target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

        # --- REFINED GRASP LOGIC ---
        if closest_obj_id != -1:
            # Use Top Center for more accurate grasping
            final_target = get_grasp_target(closest_obj_id)
            log(f"Refining target using Object Top-Center: {final_target}")
        else:
            final_target = target_world

        # Grasp Strategy (adjusted for table height)
        # Go exactly to top surface (suction cup needs contact)
        grasp = [final_target[0], final_target[1], final_target[2]]
        # Lift: 40cm above table surface
        lift = [final_target[0], final_target[1], table_z_surface + 0.40]

        log("Executing Safe Approach...")
        move_arm_safe(robot_id, joint_indices, end_effector_index, grasp, target_orn, use_gui)

        # Constraint / Suction
        if closest_obj_id != -1:
            ee_pos = p.getLinkState(robot_id, end_effector_index)[0]
            dist_to_obj = np.linalg.norm(np.array(ee_pos) - np.array(final_target))
            # Increased tolerance slightly for large objects or awkward shapes
            if dist_to_obj > 0.10: # Stricter tolerance now that we target surface
                log(f"Missed grasp! Distance: {dist_to_obj:.2f}m")
            else:
                log("Activating suction.")
                # Kill velocity to stop it sliding away
                p.resetBaseVelocity(closest_obj_id, [0] * 3, [0] * 3)

                p.createConstraint(
                    robot_id, end_effector_index, closest_obj_id, -1,
                    p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0]
                )

        log("Lifting...")
        # Direct lift is usually safe if we go straight up, but using move_arm for simplicity
        move_arm(robot_id, joint_indices, end_effector_index, lift, target_orn, 100, use_gui)

        log("Trial completed. Holding pose for verification.")
        time.sleep(5)

    finally:
        # Cleanup
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
