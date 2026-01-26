import pybullet as p
import numpy as np
import random
import math

def matrix_from_list(values):
    """Return a 4x4 matrix from a flat list in column-major order."""
    return np.array(values, dtype=np.float32).reshape((4, 4), order="F")

def get_random_eye_to_hand_pose(target_pos, dist_min=0.5, dist_max=0.8, azim_range=(-60, 60), elev_range=(30, 70)):
    """
    Generates a random camera pose looking at `target_pos`.

    Args:
        target_pos (list): [x, y, z] target to look at.
        dist_min (float): Min distance from target.
        dist_max (float): Max distance from target.
        azim_range (tuple): (min, max) azimuth in degrees relative to target (0 is +X direction typically, or depends on setup).
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

def get_camera_image(view_matrix, proj_matrix, width=640, height=480, renderer=p.ER_BULLET_HARDWARE_OPENGL):
    _, _, rgba, depth, _ = p.getCameraImage(
        width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=renderer
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
