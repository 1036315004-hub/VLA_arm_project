import pybullet as p
import numpy as np
import random
import math

def matrix_from_list(values):
    """Return a 4x4 matrix from a flat list in column-major order."""
    if isinstance(values, np.ndarray) and values.shape == (4, 4):
        return values
    return np.array(values, dtype=np.float32).reshape((4, 4), order="F")


def world_to_pixel(world_pos, view_matrix, proj_matrix, width, height):
    """Convert world coordinates to pixel coordinates."""
    view_mat = matrix_from_list(view_matrix)
    proj_mat = matrix_from_list(proj_matrix)
    world = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float32)
    clip = proj_mat @ (view_mat @ world)
    if clip[3] == 0:
        return None
    ndc = clip[:3] / clip[3]
    if ndc[2] < -1.0 or ndc[2] > 1.0:
        return None
    u = int(round((ndc[0] + 1.0) * 0.5 * (width - 1)))
    v = int(round((1.0 - ndc[1]) * 0.5 * (height - 1)))
    if u < 0 or u >= width or v < 0 or v >= height:
        return None
    return u, v


# Fixed Camera1 configuration for data collection
CAMERA1_CONFIG = {
    "eye": [1.2, 0.0, 0.9],     # Positioned in front, height increased from 0.7 to 0.9
    "target": [0.4, 0.0, 0.45], # Looking towards the robot base/arm area
    "up": [0, 0, 1],
    "fov": 60,
    "near": 0.1,
    "far": 2.0,
    "width": 256,
    "height": 256
}


def get_camera1_matrices(config=None):
    """
    Returns the view and projection matrices for the fixed Camera1 configuration.
    
    Args:
        config: Optional camera config dict. Uses CAMERA1_CONFIG if None.
    
    Returns:
        tuple: (view_matrix, proj_matrix, config_dict)
    """
    if config is None:
        config = CAMERA1_CONFIG
    
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=config["eye"],
        cameraTargetPosition=config["target"],
        cameraUpVector=config["up"]
    )
    
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=config["fov"],
        aspect=config["width"] / config["height"],
        nearVal=config["near"],
        farVal=config["far"]
    )
    
    return view_matrix, proj_matrix, config

def get_random_eye_to_hand_pose(target_pos, dist_min=0.4, dist_max=0.7, azim_range=(-180, 180), elev_range=(40, 75)):
    """
    Generates a random camera pose looking at `target_pos`.

    Args:
        target_pos (list): [x, y, z] target to look at.
        dist_min (float): Min distance from target.
        dist_max (float): Max distance from target.
        azim_range (tuple): (min, max) azimuth in degrees. Use (-180, 180) for full circle.
        elev_range (tuple): (min, max) elevation in degrees (0 is horizontal, 90 is vertical).

    Returns:
        tuple: (view_matrix, proj_matrix, cam_position, cam_up)
    """
    dist = random.uniform(dist_min, dist_max)
    azim = math.radians(random.uniform(*azim_range))
    elev = math.radians(random.uniform(*elev_range))

    # Spherical to Cartesian relative to target
    # Standard math: z is up.
    # x = d * cos(elev) * cos(azim)
    # y = d * cos(elev) * sin(azim)
    # z = d * sin(elev)

    # Check PyBullet coord system: Z is up.
    # Let's assume we want the camera generally "front-ish".
    # We can center the azimuth logic around the robot's front workspace.

    dx = dist * math.cos(elev) * math.cos(azim)
    dy = dist * math.cos(elev) * math.sin(azim)
    dz = dist * math.sin(elev)

    cam_pos = [target_pos[0] + dx, target_pos[1] + dy, target_pos[2] + dz]

    # Compute View Matrix
    # We want UP vector to be roughly World Z, but adjusted? PyBullet usually handles [0,0,1] okay unless looking straight down.
    # If looking straight down, up should be Y.
    cam_up = [0, 0, 1]

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=cam_up
    )

    # Projection Matrix (standard intrinsics)
    fov = 60
    aspect = 640 / 480
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect, nearVal=0.1, farVal=2.0
    )

    return view_matrix, proj_matrix, cam_pos

def get_camera_image(view_matrix, proj_matrix, width=640, height=480, renderer=p.ER_BULLET_HARDWARE_OPENGL, **kwargs):
    _, _, rgba, depth, _ = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=renderer, **kwargs
    )
    rgb = np.reshape(rgba, (height, width, 4))[:, :, :3].astype(np.uint8)
    depth_buffer = np.reshape(depth, (height, width))
    return rgb, depth_buffer

def pixel_to_world(u, v, depth, view_matrix, proj_matrix, width=640, height=480):
    """
    Convert a pixel and depth to world coordinates using the view and projection matrices.
    """
    view_mat = matrix_from_list(view_matrix)
    proj_mat = matrix_from_list(proj_matrix)
    inv_proj_view = np.linalg.inv(proj_mat @ view_mat)

    x_ndc = (2.0 * u / (width - 1)) - 1.0
    y_ndc = 1.0 - (2.0 * v / (height - 1))
    z_ndc = (2.0 * depth) - 1.0

    clip = np.array([x_ndc, y_ndc, z_ndc, 1.0], dtype=np.float32)
    world = inv_proj_view @ clip
    world /= world[3]
    return world[:3]
