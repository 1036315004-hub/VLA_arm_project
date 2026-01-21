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
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            log(f"Target center at pixel {center}.")
            return center

    # Enhanced fallback for colors using Contour Filtering
    text_lower = text_query.lower()
    log(f"VLM failed or not found, falling back to color segmentation for '{text_lower}'.")
    mask = None

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Kuka orange is roughly (255, 127, 0). We need to exclude it from 'red'.
    # Red object ~ (255, 0, 0).
    if "red" in text_lower:
        mask = (r > 160) & (g < 80) & (b < 80)
    elif "green" in text_lower:
        mask = (g > 160) & (r < 100) & (b < 100)
    elif "blue" in text_lower:
        mask = (b > 160) & (r < 100) & (g < 100)
    elif "yellow" in text_lower:
        mask = (r > 160) & (g > 160) & (b < 100)

    if mask is not None:
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_cands = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise (<20) and large objects like robot parts (>3000)
            if 20 < area < 3000:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    valid_cands.append((area, (cX, cY)))

        if valid_cands:
            # Pick largest valid blob
            valid_cands.sort(key=lambda x: x[0], reverse=True)
            log(f"Found {len(valid_cands)} candidate contours. Best area: {valid_cands[0][0]}")
            return valid_cands[0][1]

    log("Target detection failed.")
    return None


def move_arm(robot_id, joint_indices, end_effector_index, target_pos, target_orn, steps, sleep):
    """Move the end effector using IK, stepping simulation for a fixed number of steps."""
    joint_positions = p.calculateInverseKinematics(
        robot_id, end_effector_index, target_pos, target_orn
    )
    joint_positions = list(joint_positions)
    if len(joint_positions) < len(joint_indices):
        # Fill remaining joints with current positions to keep non-IK joints stable.
        remaining_indices = joint_indices[len(joint_positions) :]
        joint_positions.extend(p.getJointState(robot_id, idx)[0] for idx in remaining_indices)
    p.setJointMotorControlArray(
        robot_id,
        joint_indices,
        p.POSITION_CONTROL,
        targetPositions=joint_positions[: len(joint_indices)],
        forces=[JOINT_FORCE] * len(joint_indices),
    )
    for _ in range(steps):
        p.stepSimulation()
        if sleep:
            time.sleep(1.0 / 240.0)


def main():
    use_gui = connect_pybullet()
    p.setAdditionalSearchPath(PYBULLET_DATA_PATH)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    log("Loading plane and robot.")
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True)

    # --- Generate Random Objects ---
    available_colors = {
        "red object": [1, 0, 0, 1],
        "green object": [0, 1, 0, 1],
        "blue object": [0, 0, 1, 1],
        "yellow object": [1, 1, 0, 1]
    }

    # Randomly select 2 different colors
    selected_names = random.sample(list(available_colors.keys()), 2)
    object_ids = []
    spawned_positions = []

    # Define shapes: One Box, One Sphere
    shapes = [p.GEOM_BOX, p.GEOM_SPHERE]
    random.shuffle(shapes)

    log(f"Spawning objects: {selected_names} (Shapes: {['Box' if s == p.GEOM_BOX else 'Sphere' for s in shapes]})")

    for i, name in enumerate(selected_names):
        rgba = available_colors[name]
        shape_type = shapes[i]

        # Random position in front of robot
        # x: 0.5 to 0.7, y: -0.2 to 0.2
        pos_x = random.uniform(0.5, 0.7)
        pos_y = random.uniform(-0.2, 0.2)
        pos_z = 0.03


        spawned_positions.append([pos_x, pos_y, pos_z])

        # Create shape (using both halfExtents and radius to cover both Box and Sphere args)
        visual_shape = p.createVisualShape(shape_type, halfExtents=[0.03, 0.03, 0.03], radius=0.03, rgbaColor=rgba)
        collision_shape = p.createCollisionShape(shape_type, halfExtents=[0.03, 0.03, 0.03], radius=0.03)

        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[pos_x, pos_y, pos_z]
        )

        # Increase friction to prevent unrealistic rolling/sliding
        # lateralFriction: friction with the floor
        # rollingFriction: resistance to rolling (critical for spheres)
        p.changeDynamics(obj_id, -1, lateralFriction=1.0, rollingFriction=0.05, spinningFriction=0.05)

        object_ids.append(obj_id)

    # Let objects settle
    for _ in range(100):
        p.stepSimulation()

    num_joints = p.getNumJoints(robot_id)
    joint_indices = list(range(num_joints))
    default_ee_index = 6
    end_effector_index = default_ee_index if num_joints > default_ee_index else joint_indices[-1]
    if end_effector_index != default_ee_index:
        log("Using last joint as end effector fallback.")

    width, height = 640, 480
    camera_target = [0.5, 0.0, 0.05]
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        camera_target, distance=1.1, yaw=45, pitch=-30, roll=0, upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=2.0
    )
    renderer = p.ER_BULLET_HARDWARE_OPENGL if use_gui else p.ER_TINY_RENDERER
    log("Capturing camera image.")
    _, _, rgba, depth, _ = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=renderer
    )
    rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
    depth = np.reshape(depth, (height, width))

    vlm = build_vlm()

    # --- User Interaction ---
    print("\n" + "="*40)
    print(" Simulation Ready. Objects available: " + ", ".join(selected_names))
    print(" Please enter a command (e.g. 'pick up the red object').")
    text_query = input(" >>> Command: ").strip()
    if not text_query:
        text_query = "pick up the " + selected_names[0]
        log(f"No input provided. Defaulting to: '{text_query}'")
    print("="*40 + "\n")

    target_pixel = detect_target_from_text(rgb, vlm, text_query)

    view_mat = matrix_from_list(view_matrix)
    proj_mat = matrix_from_list(proj_matrix)
    inv_proj_view = np.linalg.inv(proj_mat @ view_mat)

    if target_pixel is None:
        log("Could not find target. Simulation ending.")
        return

    u = int(np.clip(target_pixel[0], 0, width - 1))
    v = int(np.clip(target_pixel[1], 0, height - 1))
    depth_value = float(depth[v, u])
    target_world = pixel_to_world(u, v, depth_value, inv_proj_view, width, height)
    log(f"Target world position from depth: {target_world}.")

    # --- Snapping Logic ---
    # Find nearest object to the detected point and snap to it if close
    closest_obj_id = -1
    min_dist = 0.25  # SNAP TOLERANCE: 25cm radius

    for obj_id in object_ids:
        pos, _ = p.getBasePositionAndOrientation(obj_id)
        dist = np.linalg.norm(np.array(pos) - target_world)
        if dist < min_dist:
            min_dist = dist
            closest_obj_id = obj_id

    if closest_obj_id != -1:
        obj_pos, _ = p.getBasePositionAndOrientation(closest_obj_id)
        log(f"Snapping target from {target_world} to object {closest_obj_id} at {obj_pos}")
        target_world = np.array(obj_pos)
    else:
        log("No object found near target point. Continuing with raw coordinates.")

    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    # Adjust grasp heights based on reliable target center
    pre_grasp = [target_world[0], target_world[1], target_world[2] + 0.2]
    # Go to top surface of object (approx 3cm radius) to avoid penetration collision
    grasp = [target_world[0], target_world[1], target_world[2] + 0.032]
    lift = [target_world[0], target_world[1], target_world[2] + 0.3]

    log("Moving to pre-grasp position.")
    move_arm(robot_id, joint_indices, end_effector_index, pre_grasp, target_orn, 240, use_gui)
    log("Descending to grasp position.")
    move_arm(robot_id, joint_indices, end_effector_index, grasp, target_orn, 240, use_gui)

    log("Creating fixed constraint to grasp cube.")
    link_state = p.getLinkState(robot_id, end_effector_index)
    ee_pos, ee_orn = link_state[0], link_state[1]

    constraint_axis = ZERO_VECTOR  # fixed joint has no axis
    parent_frame_pos = ZERO_VECTOR  # end effector frame origin
    parent_frame_orn = [0, 0, 0, 1]

    # We constrain to the closest object we identified earlier
    if closest_obj_id != -1:
        log(f"Attaching to object ID {closest_obj_id}")

        # Stop object motion before attaching to prevent "floating away" or rolling
        # Keep collision active for realism, but rely on surface contact grasp
        p.resetBaseVelocity(closest_obj_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        # Disable collision between gripper and object to allow perfect centering without explosion
        p.setCollisionFilterPair(robot_id, closest_obj_id, end_effector_index, -1, enableCollision=0)

        # Snap object to the center of the end effector (Magnetic Grasp)
        # We define the pivot point (0,0,0) in both frames to force alignment
        p.createConstraint(
            robot_id,
            end_effector_index,
            closest_obj_id,
            -1,
            p.JOINT_FIXED,
            constraint_axis,
            [0, 0, 0],   # parentFramePosition
            [0, 0, 0],   # childFramePosition
            [0, 0, 0, 1],# parentFrameOrientation
            [0, 0, 0, 1] # childFrameOrientation
        )
    else:
        log("No confirmed object to attach to.")

    log("Lifting cube.")
    move_arm(robot_id, joint_indices, end_effector_index, lift, target_orn, 240, use_gui)

    log("Grasp sequence complete.")
    for _ in range(240):
        p.stepSimulation()
        if use_gui:
            time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
