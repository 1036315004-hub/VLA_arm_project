import os
import sys
import time
import random
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import math

# Adjust path to Project Root
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.perception.camera_utils import get_random_eye_to_hand_pose, get_camera_image, pixel_to_world

# Constants
JOINT_FORCE = 500
PYBULLET_DATA_PATH = pybullet_data.getDataPath()

class DataRecorder:
    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        self.connect_pybullet()
        self.robot_id = None
        self.table_id = None
        self.object_ids = []

        # Large VLM/YOLO models removed; keep lightweight pipeline only

    def connect_pybullet(self):
        try:
            p.connect(p.GUI if self.use_gui else p.DIRECT)
            p.setAdditionalSearchPath(PYBULLET_DATA_PATH)
            p.setGravity(0, 0, -9.81)
            print("[DataRecorder] Connected to PyBullet.")
        except Exception as e:
            print(f"[DataRecorder] Connection failed: {e}")

    def load_scene(self):
        p.loadURDF("plane.urdf")

        # Load Table
        table_path = os.path.join(PROJECT_ROOT, "assets", "objects", "table.urdf")
        # Ensure table path exists, else fallback or use standard cube
        if os.path.exists(table_path):
            self.table_id = p.loadURDF(table_path, basePosition=[0.8, 0, 0], useFixedBase=True)
            self.table_surface_z = 0.4
        else:
            self.table_id = p.loadURDF("table/table.urdf", basePosition=[0.8, 0, 0], useFixedBase=True)
            self.table_surface_z = 0.625 # Default pybullet table height

        # Load Robot
        robot_path = os.path.join(PROJECT_ROOT, "assets", "ec63", "urdf", "ec63_description.urdf")
        # Default start, will be randomized
        self.robot_id = p.loadURDF(robot_path, basePosition=[0.35, 0, 0.4], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(self.num_joints))
        self.end_effector_index = 6 if self.num_joints > 6 else (self.num_joints - 1)

    def randomize_episode(self):
        """
        Randomizes:
        1. Robot Initial Joint State
        2. Object Positions and Colors
        3. Camera Position (Eye-to-Hand) - actually generated on demand during perception
        """
        # 1. Randomize Robot Start State (Safe random)
        # Randomize around the "standing" pose to avoid self-collision at start
        base_pose = [0, -math.pi/4, math.pi/2, 0, math.pi/4, 0]
        noise = np.random.uniform(-0.1, 0.1, size=6)
        start_pose = []
        for i, val in enumerate(base_pose):
            start_pose.append(val + noise[i])

        for i, j_idx in enumerate(self.joint_indices[:6]):
             p.resetJointState(self.robot_id, j_idx, start_pose[i])

        # 2. Randomize Objects
        # Clear old objects
        for obj in self.object_ids:
            p.removeBody(obj)
        self.object_ids = []

        colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1]] # R, G, B
        shapes = [p.GEOM_BOX, p.GEOM_SPHERE]

        # Spawn 1-2 objects
        for _ in range(random.randint(1, 2)):
            col = random.choice(colors)
            shp = random.choice(shapes)

            # Area on table
            pos_x = random.uniform(0.6, 0.9)
            pos_y = random.uniform(-0.3, 0.3)
            pos_z = self.table_surface_z + 0.03

            vis = p.createVisualShape(shp, halfExtents=[0.03]*3, radius=0.03, rgbaColor=col)
            col_shp = p.createCollisionShape(shp, halfExtents=[0.03]*3, radius=0.03)

            uid = p.createMultiBody(0.1, col_shp, vis, basePosition=[pos_x, pos_y, pos_z])
            p.changeDynamics(uid, -1, lateralFriction=1.0)
            self.object_ids.append(uid)

        # Step to settle
        for _ in range(20): p.stepSimulation()

    def move_arm(self, target_pos, target_orn=None, steps=100):
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

        for i in range(steps):
            joint_poses = p.calculateInverseKinematics(
                self.robot_id, self.end_effector_index, target_pos, target_orn,
                maxNumIterations=100, residualThreshold=1e-5
            )
            for idx, j_idx in enumerate(self.joint_indices[:len(joint_poses)]):
                p.setJointMotorControl2(self.robot_id, j_idx, p.POSITION_CONTROL,
                                        targetPosition=joint_poses[idx], force=JOINT_FORCE, maxVelocity=2.0)
            p.stepSimulation()
            if self.use_gui: time.sleep(1./240.)

    def move_to_observation_pose(self):
        """
        Moves the robot out of the way for Eye-to-Hand camera.
        Retracted pose: high up and pulled back.
        """
        # Joint space control is often safer for "Home" than IK
        # Pose: Base=0, Shoulder=-90, Elbow=120...
        retracted_joints = [0, -1.5, 2.0, 0, 0, 0] # Approximate "C" shape pulled back

        print("[DataRecorder] Moving to Observation Pose (Retracted)...")
        for _ in range(100):
            for i, val in enumerate(retracted_joints):
                 p.setJointMotorControl2(self.robot_id, self.joint_indices[i], p.POSITION_CONTROL,
                                         targetPosition=val, force=JOINT_FORCE)
            p.stepSimulation()
            if self.use_gui: time.sleep(1./240.)

    def run(self):
        self.load_scene()

        while True:
            cmd = input("Press Enter to Run Episode (or 'q' to quit): ")
            if cmd == 'q': break

            print("\n--- New Episode ---")
            self.randomize_episode()

            # STAGE 1: Move Robot Out of View (Solve Occlusion)
            self.move_to_observation_pose()

            # STAGE 2: Perception (Eye-to-Hand)
            # Generate random camera pose looking at table center
            table_center = [0.8, 0.0, self.table_surface_z]
            view_mat, proj_mat, cam_pos = get_random_eye_to_hand_pose(table_center)

            # Capture
            if self.use_gui:
                # Debug draw camera line
                p.addUserDebugLine(cam_pos, table_center, [1,1,0], lifeTime=3)

            rgb, depth = get_camera_image(view_mat, proj_mat)
            print(f"[DataRecorder] Captured Eye-to-Hand Image from {np.round(cam_pos, 2)}")

            # Detect (Simple Red Object logic)
            target_pixel = None

            # Fallback Color Detect (Red)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            # Red
            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
            mask = mask1 | mask2
            pts = cv2.findNonZero(mask)
            if pts is not None:
                x, y, w, h = cv2.boundingRect(pts)
                target_pixel = (x + w//2, y + h//2)
                print(f"[DataRecorder] Detected Red Object at {target_pixel}")

            if target_pixel:
                # STAGE 3: Compute World Position
                world_pos = pixel_to_world(target_pixel[0], target_pixel[1], depth[target_pixel[1], target_pixel[0]],
                                           view_mat, proj_mat)
                print(f"[DataRecorder] Target World Position: {world_pos}")

                # STAGE 4: Grasp
                # Only grasp if reachable
                if 0.3 < world_pos[0] < 1.0:
                    print("[DataRecorder] Executing Grasp...")
                    pre_grasp = [world_pos[0], world_pos[1], world_pos[2] + 0.2]
                    grasp_pos = [world_pos[0], world_pos[1], world_pos[2] + 0.05]

                    self.move_arm(pre_grasp)
                    self.move_arm(grasp_pos)
                else:
                    print("[DataRecorder] Target out of range.")
            else:
                print("[DataRecorder] No target detected.")

if __name__ == "__main__":
    recorder = DataRecorder()
    recorder.run()
