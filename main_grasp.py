import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import sys
import time
import random
import cv2

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pybullet as p
import pybullet_data

JOINT_FORCE = 500
ZERO_VECTOR = [0, 0, 0]
PYBULLET_DATA_PATH = pybullet_data.getDataPath()
if PYBULLET_DATA_PATH not in sys.path:
    sys.path.insert(0, PYBULLET_DATA_PATH)


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


def pixel_to_world(u, v, depth, inv_proj_view, width, height):
    """Convert a pixel and depth to world coordinates using the inverse PV matrix."""
    if width <= 1 or height <= 1:
        raise ValueError("Width and height must be greater than 1 for projection math.")
    x_ndc = (2.0 * u / (width - 1)) - 1.0
    y_ndc = 1.0 - (2.0 * v / (height - 1))
    z_ndc = (2.0 * depth) - 1.0
    clip = np.array([x_ndc, y_ndc, z_ndc, 1.0], dtype=np.float32)
    world = inv_proj_view @ clip
    world /= world[3]
    return world[:3]


def build_vlm():
    try:
        from src.perception.vlm_yolo_clip import VLM_YOLO_CLIP

        yolo_model = os.path.join(ROOT, "yolov8n.pt")
        if not os.path.exists(yolo_model):
            yolo_model = "yolov8n.pt"
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
    if vlm is not None:
        try:
            detections = vlm.query_image(rgb, text_query, topk=1)
        except Exception as exc:
            log(f"VLM query failed. ({exc})")
            detections = []

        if detections:
            # Check if detection score is reasonable if needed, but for now take top 1
            det = detections[0]
            log(f"VLM detected '{det['label']}' with score {det.get('clip_score', 0):.2f}")
            x1, y1, x2, y2 = map(int, det["bbox"])
            x1 = max(0, min(x1, rgb.shape[1] - 1))
            x2 = max(0, min(x2, rgb.shape[1] - 1))
            y1 = max(0, min(y1, rgb.shape[0] - 1))
            y2 = max(0, min(y2, rgb.shape[0] - 1))
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            log(f"Target center at pixel ({center_x}, {center_y}), area: {area}.")
            return (center_x, center_y, area)

    # Enhanced fallback for colors using HSV
    text_lower = text_query.lower()
    log(f"VLM failed or not found, falling back to HSV color segmentation for '{text_lower}'.")

    # Convert to HSV for robust color detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = None

    # HSV Ranges (OpenCV H: 0-179, S: 0-255, V: 0-255)
    # Refined ranges for typical simulated colors
    if "red" in text_lower:
        # Red wraps around 0/180
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
    elif "green" in text_lower:
        # Green ~60
        mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
    elif "blue" in text_lower:
        # Blue ~120
        mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([140, 255, 255]))
    elif "yellow" in text_lower:
        # Yellow ~30
        mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))

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
    max_vel = 2.5 # rad/s, Fast speed

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


def run_trial(robot_id, plane_id, joint_indices, end_effector_index, use_gui):
    # --- Load Table ---
    # Table visual box is 0.4m tall, origin at z=0.2 inside the file, so placing at z=0 puts top at 0.4
    table_pos = [0.6, 0.0, 0.0]
    p.loadURDF("table.urdf", basePosition=table_pos, useFixedBase=True)
    table_z_surface = 0.4
    log(f"Table loaded at {table_pos}, surface height ~{table_z_surface}m")

    # --- Generate Random Objects ---
    available_colors = {
        "red cube": [1, 0, 0, 1],
        "green cube": [0, 1, 0, 1],
        "blue cube": [0, 0, 1, 1],
        "yellow cube": [1, 1, 0, 1]
    }

    object_ids = []

    try:
        # Randomly select 2 different colors
        selected_names = random.sample(list(available_colors.keys()), 2)

        # Shapes
        shapes = [p.GEOM_BOX, p.GEOM_SPHERE]
        random.shuffle(shapes)

        log(f"--- Setup --- Objects on table: {selected_names}")

        for i, name in enumerate(selected_names):
            rgba = available_colors[name]
            shape_type = shapes[i]

            # Random position ON TABLE - Closer to robot (0.45 to 0.65)
            pos_x = random.uniform(0.45, 0.65)
            pos_y = random.uniform(-0.15, 0.15)
            pos_z = table_z_surface + 0.03

            visual_shape = p.createVisualShape(shape_type, halfExtents=[0.03, 0.03, 0.03], radius=0.03, rgbaColor=rgba)
            collision_shape = p.createCollisionShape(shape_type, halfExtents=[0.03, 0.03, 0.03], radius=0.03)

            obj_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[pos_x, pos_y, pos_z]
            )
            # High friction to stop sliding on table
            p.changeDynamics(obj_id, -1, lateralFriction=1.0, rollingFriction=0.001, spinningFriction=0.001)
            object_ids.append(obj_id)

        # Let objects settle
        for _ in range(50):
            p.stepSimulation()

        # Reset Arm to Home
        home_pos = [0] * len(joint_indices)
        p.setJointMotorControlArray(robot_id, joint_indices, p.POSITION_CONTROL, targetPositions=home_pos)
        for _ in range(50): p.stepSimulation()

        # --- User Input (Blocking) ---
        print("\n" + "="*50)
        print(f" Objects available: {', '.join(selected_names)}")
        print(" Please enter command (e.g. 'pick up the red cube').")
        print("="*50)

        # This will block until user types in the console
        text_query = input(" >>> Command: ").strip()

        if not text_query:
            log("No input provided. Ending trial.")
            return

        # --- EYE-IN-HAND: Multi-Stage Scanning Algorithm ---
        vlm = build_vlm()
        found_target = False
        target_world = None
        width, height = 640, 480
        scan_orn = p.getQuaternionFromEuler([math.pi, 0, 0])  # Look down

        # Helper function to stabilize arm and capture image
        def stabilize_and_capture():
            if use_gui:
                for _ in range(50): p.stepSimulation(); time.sleep(0.01)
            else:
                for _ in range(50): p.stepSimulation()
            return get_eye_in_hand_image(robot_id, end_effector_index, width, height, use_gui)

        # Helper function to convert pixel to world coordinates
        def compute_world_position(cx, cy, depth_buffer, view_matrix, proj_matrix):
            view_mat = matrix_from_list(view_matrix)
            proj_mat = matrix_from_list(proj_matrix)
            inv_proj_view = np.linalg.inv(proj_mat @ view_mat)
            u = int(np.clip(cx, 0, width - 1))
            v = int(np.clip(cy, 0, height - 1))
            depth_value = float(depth_buffer[v, u])
            return pixel_to_world(u, v, depth_value, inv_proj_view, width, height)

        # ========================================
        # STAGE 1: Global Scan (High Altitude Overview)
        # ========================================
        log("=== STAGE 1: Global Scan ===")
        global_scan_pos = [0.55, 0.0, 0.80]  # High position for wide field of view
        log(f"Performing global scan from {global_scan_pos}...")
        move_arm(robot_id, joint_indices, end_effector_index, global_scan_pos, scan_orn, 80, use_gui)

        rgb, depth, view_matrix, proj_matrix = stabilize_and_capture()
        target_info = detect_target_from_text(rgb, vlm, text_query)

        preliminary_world_pos = None
        if target_info:
            cx, cy, area = target_info
            log(f"Global scan detected target! Area: {area}, Position: ({cx}, {cy})")
            preliminary_world_pos = compute_world_position(cx, cy, depth, view_matrix, proj_matrix)
            
            # If very clear detection in global scan (large area), consider it found
            if area > 2000:
                log("Large target clearly visible in global scan.")
                found_target = True
                target_world = preliminary_world_pos

        # ========================================
        # STAGE 2: Systematic Left-to-Right Scanning
        # ========================================
        if not found_target:
            log("=== STAGE 2: Systematic Left-to-Right Scanning ===")
            # Scan the table area systematically from left to right
            # Table area: x=[0.45, 0.65], y=[-0.20, 0.20]
            # Scan at lower altitude for better detail detection
            
            # Y-axis scan positions from left (positive Y) to right (negative Y)
            y_positions = [0.25, 0.15, 0.05, -0.05, -0.15, -0.25]
            # X-axis positions for near and far rows
            x_positions = [0.50, 0.60]
            scan_height = 0.60  # Lower than global scan for better resolution
            
            best_detection = None  # Track best detection (highest area)
            best_world_pos = None
            
            for x_scan in x_positions:
                if found_target:
                    break
                for y_scan in y_positions:
                    scan_pos = [x_scan, y_scan, scan_height]
                    log(f"Stage 2: Scanning at x={x_scan:.2f}, y={y_scan:.2f}...")
                    move_arm(robot_id, joint_indices, end_effector_index, scan_pos, scan_orn, 60, use_gui)
                    
                    rgb, depth, view_matrix, proj_matrix = stabilize_and_capture()
                    target_info = detect_target_from_text(rgb, vlm, text_query)
                    
                    if target_info:
                        cx, cy, area = target_info
                        log(f"Target detected at ({cx}, {cy}), area: {area}")
                        world_pos = compute_world_position(cx, cy, depth, view_matrix, proj_matrix)
                        
                        # Track the best detection (largest area = clearest view)
                        if best_detection is None or area > best_detection[2]:
                            best_detection = (cx, cy, area)
                            best_world_pos = world_pos
                        
                        # If we have a good enough detection, proceed
                        if area > 800:
                            log(f"Good detection found (area={area}). Moving to Stage 3.")
                            preliminary_world_pos = world_pos
                            found_target = True
                            break
            
            # Use best detection if no single scan reached threshold but we found something
            if not found_target and best_detection is not None:
                log(f"Using best detection from Stage 2 scan (area={best_detection[2]})")
                preliminary_world_pos = best_world_pos
                found_target = True

        # ========================================
        # STAGE 3: Secondary Scan and Precise Positioning
        # ========================================
        if found_target and preliminary_world_pos is not None:
            log("=== STAGE 3: Secondary Scan and Precise Positioning ===")
            
            # Move directly above the detected position at closer range
            refine_height = 0.50  # Lower height for precision
            refine_pos = [preliminary_world_pos[0], preliminary_world_pos[1], refine_height]
            log(f"Moving to refinement position: {refine_pos}")
            move_arm(robot_id, joint_indices, end_effector_index, refine_pos, scan_orn, 80, use_gui)
            
            rgb_r, depth_r, view_matrix_r, proj_matrix_r = stabilize_and_capture()
            target_info_r = detect_target_from_text(rgb_r, vlm, text_query)
            
            if target_info_r:
                cx_r, cy_r, area_r = target_info_r
                log(f"Refinement scan confirmed target. Area: {area_r}, Position: ({cx_r}, {cy_r})")
                target_world = compute_world_position(cx_r, cy_r, depth_r, view_matrix_r, proj_matrix_r)
                
                # If target is not centered (off by more than 100 pixels), do micro-adjustment
                center_offset_x = abs(cx_r - width // 2)
                center_offset_y = abs(cy_r - height // 2)
                
                if center_offset_x > 100 or center_offset_y > 100:
                    log("Target not centered. Performing micro-adjustment...")
                    # Move towards the detected target position for final alignment
                    micro_pos = [target_world[0], target_world[1], refine_height - 0.05]
                    move_arm(robot_id, joint_indices, end_effector_index, micro_pos, scan_orn, 60, use_gui)
                    
                    rgb_m, depth_m, view_matrix_m, proj_matrix_m = stabilize_and_capture()
                    target_info_m = detect_target_from_text(rgb_m, vlm, text_query)
                    
                    if target_info_m:
                        cx_m, cy_m, area_m = target_info_m
                        log(f"Micro-adjustment confirmed. Final area: {area_m}")
                        target_world = compute_world_position(cx_m, cy_m, depth_m, view_matrix_m, proj_matrix_m)
                    else:
                        log("Target lost during micro-adjustment. Using previous position.")
            else:
                log("Target lost during refinement. Using preliminary position.")
                target_world = preliminary_world_pos
        elif found_target and preliminary_world_pos is None:
            # Edge case: marked as found but no position (shouldn't happen)
            log("Warning: Target marked as found but no position available.")
            found_target = False

        if not found_target:
            log("Target not found via Vision after multi-stage scanning.")
            return

        # --- Snapping & Refinement ---
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

        # Grasp Strategy (adjusted for table height)
        # Pre-grasp: 20cm above object
        pre_grasp = [target_world[0], target_world[1], target_world[2] + 0.20]
        # Grasp: 5cm above center
        grasp = [target_world[0], target_world[1], target_world[2] + 0.05]
        # Lift: 40cm above table surface
        lift = [target_world[0], target_world[1], table_z_surface + 0.40]

        log("Moving to pre-grasp...")
        move_arm(robot_id, joint_indices, end_effector_index, pre_grasp, target_orn, 100, use_gui)

        log("Descending...")
        move_arm(robot_id, joint_indices, end_effector_index, grasp, target_orn, 100, use_gui)

        # Constraint / Suction
        if closest_obj_id != -1:
            ee_pos = p.getLinkState(robot_id, end_effector_index)[0]
            dist_to_obj = np.linalg.norm(np.array(ee_pos) - np.array(target_world))
            if dist_to_obj > 0.15:
                # Allow a bit more specific error tolerance
                log(f"Missed grasp! Distance: {dist_to_obj:.2f}m")
            else:
                log("Activating suction.")
                p.resetBaseVelocity(closest_obj_id, [0]*3, [0]*3)
                p.createConstraint(
                    robot_id, end_effector_index, closest_obj_id, -1,
                    p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0]
                )

        log("Lifting...")
        move_arm(robot_id, joint_indices, end_effector_index, lift, target_orn, 100, use_gui)

        log("Trial completed. Holding pose for verification.")
        time.sleep(5)

    finally:
        # Cleanup
        for obj_id in object_ids:
            p.removeBody(obj_id)


def main():
    use_gui = connect_pybullet()
    p.setAdditionalSearchPath(PYBULLET_DATA_PATH)
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    plane_id = 0 # usually 0

    num_joints = p.getNumJoints(robot_id)
    joint_indices = list(range(num_joints))
    default_ee_index = 6
    end_effector_index = default_ee_index if num_joints > default_ee_index else joint_indices[-1]

    # Run single trial with user interaction
    log("\n=== Starting User Loop ===")
    run_trial(robot_id, plane_id, joint_indices, end_effector_index, use_gui)

    log("Simulation Session Ended.")
    # Keep window open briefly
    time.sleep(2)



if __name__ == "__main__":
    main()
