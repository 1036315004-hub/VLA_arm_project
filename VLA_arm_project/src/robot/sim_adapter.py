import pybullet as p
import pybullet_data
import time
import math
import threading
from typing import Any, Dict, List, Optional
import os

class SimulationAdapter:
    def __init__(self):
        self.connected = False
        self.robot_id = None
        self.joint_indices = []
        self._thread = None
        self._running = False
        self.target_joint_positions = []

    def connect(self):
        if self.connected:
            return

        try:
            # Connect to PyBullet GUI
            p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.8)
            p.loadURDF("plane.urdf")

            # Load robot model
            # Priority:
            # 1. Custom EC63 URDF in local directory
            # 2. Fallback to built-in Kuka IIWA

            startPos = [0, 0, 0]
            startOrientation = p.getQuaternionFromEuler([0, 0, 0])

            urdf_names = ["ec63_description.urdf", "ec63.urdf", "ec66.urdf"]
            found_urdf = None
            for name in urdf_names:
                if os.path.exists(name):
                    found_urdf = name
                    break

            if found_urdf:
                print(f"Loading custom URDF: {found_urdf}")
                self.robot_id = p.loadURDF(found_urdf, startPos, startOrientation)
            else:
                print("Custom URDF not found, loading fallback (Kuka IIWA)...")
                self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation)

            # Find revolute joints
            num_joints = p.getNumJoints(self.robot_id)
            self.joint_indices = []
            for i in range(num_joints):
                info = p.getJointInfo(self.robot_id, i)
                # qIndex > -1 ensures it's not a fixed joint. info[2] is jointType
                if info[2] == p.JOINT_REVOLUTE:
                    self.joint_indices.append(i)

            # Limit to 6 if we want to mimic EC63/66 (Elite robots are 6 axis)
            if len(self.joint_indices) > 6:
                self.joint_indices = self.joint_indices[:6]

            self.target_joint_positions = [0.0] * len(self.joint_indices)

            self.connected = True

            # Start simulation thread
            self._running = True
            self._thread = threading.Thread(target=self._sim_loop, daemon=True)
            self._thread.start()

        except Exception as e:
            print(f"PyBullet Initialization Error: {e}")
            self.connected = False

    def disconnect(self):
        self._running = False
        if self.connected:
            p.disconnect()
            self.connected = False

    def _sim_loop(self):
        while self._running:
            p.stepSimulation()
            time.sleep(1./240.)

    def process_command(self, cmd: str, params: Optional[dict] = None) -> Any:
        if not self.connected:
            # If not connected formally, maybe auto-connect?
            pass

        # === Robot State ===
        if cmd == "getRobotState":
            # Return PLAY status (3) so scripts don't block
            # RobotState.PLAY = 3
            return 3

        if cmd == "getSoftVersion":
            return "v3.0.0 (Simulated)"

        if cmd == "getRobotMode":
             # RobotMode.PLAY = 1
             return 1

        # === Kinematics / Motion ===
        if cmd == "getJointPos" or cmd == "get_joint_pos":
            # Return current joint angles in degrees
            angles = []
            for idx in self.joint_indices:
                state = p.getJointState(self.robot_id, idx)
                # PyBullet is radians, Elite is degrees
                angles.append(math.degrees(state[0]))
            # If fewer joints than expected, pad with 0? Elite has 6.
            while len(angles) < 6:
                angles.append(0.0)
            return angles

        if cmd == "moveByJoint" or cmd == "moveByLine":
            # Basic move implementation
            target_pos = None
            if params and "targetPos" in params:
                 target_pos = params["targetPos"]

            if target_pos:
                # Elite SDK uses degrees. Convert to radians for PyBullet
                target_rads = [math.radians(angle) for angle in target_pos]

                # Length check
                if len(target_rads) > len(self.joint_indices):
                    target_rads = target_rads[:len(self.joint_indices)]

                # Move
                p.setJointMotorControlArray(
                    self.robot_id,
                    self.joint_indices[:len(target_rads)],
                    p.POSITION_CONTROL,
                    targetPositions=target_rads
                )

                return True

        if cmd == "inverseKinematic":
            # Simple IK wrapper
            # In a real implementation we would use calculateInverseKinematics
            # For now return dummy to avoid errors
            return [0.0] * 6

        if cmd == "get_tcp_pose" or cmd == "getRobotPose":
             # Return dummy pose [x, y, z, rx, ry, rz]
             return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Default return for unimplemented commands to prevent crashes
        # The real SDK often returns a result dict or boolean.
        # Returning True is a safe default for Setter commands.
        # For Getter, it might be problematic, but this is a mvp adapter.
        return True
