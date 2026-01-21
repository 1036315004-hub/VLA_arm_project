import math
import os
import sys
import time

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
        from perception.vlm_yolo_clip import VLM_YOLO_CLIP

        yolo_model = os.path.join(ROOT, "yolov8n.pt")
        if not os.path.exists(yolo_model):
            yolo_model = "yolov8n.pt"
        vlm = VLM_YOLO_CLIP(yolo_model=yolo_model)
        log("Initialized VLM_YOLO_CLIP.")
        return vlm
    except Exception as exc:
        log(f"VLM import failed, falling back to color segmentation. ({exc})")
        return None


def detect_red_center(rgb, vlm):
    if vlm is not None:
        try:
            detections = vlm.query_image(rgb, "red cube", topk=1)
        except Exception as exc:
            log(f"VLM query failed, using fallback. ({exc})")
            detections = []
        if detections:
            x1, y1, x2, y2 = map(int, detections[0]["bbox"])
            x1 = max(0, min(x1, rgb.shape[1] - 1))
            x2 = max(0, min(x2, rgb.shape[1] - 1))
            y1 = max(0, min(y1, rgb.shape[0] - 1))
            y2 = max(0, min(y2, rgb.shape[0] - 1))
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            log(f"VLM detected red cube at pixel {center}.")
            return center

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r > 150) & (g < 120) & (b < 120)
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        log("Red segmentation failed; returning no target.")
        return None
    y_center = int(coords[:, 0].mean())
    x_center = int(coords[:, 1].mean())
    log(f"Red segmentation center at pixel {(x_center, y_center)}.")
    return (x_center, y_center)


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

    log("Loading plane, robot, and cube.")
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    cube_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[1, 0, 0, 1]
    )
    cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
    cube_start_pos = [0.6, 0.0, 0.03]
    cube_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=cube_collision,
        baseVisualShapeIndex=cube_visual,
        basePosition=cube_start_pos,
    )

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
    target_pixel = detect_red_center(rgb, vlm)

    view_mat = matrix_from_list(view_matrix)
    proj_mat = matrix_from_list(proj_matrix)
    inv_proj_view = np.linalg.inv(proj_mat @ view_mat)
    if target_pixel is None:
        target_world = np.array(cube_start_pos, dtype=np.float32)
        log(f"Using default cube position as target: {target_world}.")
    else:
        u = int(np.clip(target_pixel[0], 0, width - 1))
        v = int(np.clip(target_pixel[1], 0, height - 1))
        depth_value = float(depth[v, u])
        target_world = pixel_to_world(u, v, depth_value, inv_proj_view, width, height)
        log(f"Target world position from depth: {target_world}.")

    target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    pre_grasp = [target_world[0], target_world[1], target_world[2] + 0.2]
    grasp = [target_world[0], target_world[1], target_world[2] + 0.02]
    lift = [target_world[0], target_world[1], target_world[2] + 0.3]

    log("Moving to pre-grasp position.")
    move_arm(robot_id, joint_indices, end_effector_index, pre_grasp, target_orn, 240, use_gui)
    log("Descending to grasp position.")
    move_arm(robot_id, joint_indices, end_effector_index, grasp, target_orn, 240, use_gui)

    log("Creating fixed constraint to grasp cube.")
    link_state = p.getLinkState(robot_id, end_effector_index)
    ee_pos, ee_orn = link_state[0], link_state[1]
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
    cube_inv_pos, cube_inv_orn = p.invertTransform(cube_pos, cube_orn)
    parent_in_cube_pos, parent_in_cube_orn = p.multiplyTransforms(
        cube_inv_pos, cube_inv_orn, ee_pos, ee_orn
    )
    constraint_axis = ZERO_VECTOR  # fixed joint has no axis
    parent_frame_pos = ZERO_VECTOR  # end effector frame origin
    child_frame_pos = list(parent_in_cube_pos)  # same world point in cube frame
    parent_frame_orn = [0, 0, 0, 1]
    child_frame_orn = list(parent_in_cube_orn)
    p.createConstraint(
        robot_id,
        end_effector_index,
        cube_id,
        -1,
        p.JOINT_FIXED,
        constraint_axis,
        parent_frame_pos,
        child_frame_pos,
        parent_frame_orn,
        child_frame_orn,
    )
    log("Lifting cube.")
    move_arm(robot_id, joint_indices, end_effector_index, lift, target_orn, 240, use_gui)

    log("Grasp sequence complete.")
    for _ in range(240):
        p.stepSimulation()
        if use_gui:
            time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
