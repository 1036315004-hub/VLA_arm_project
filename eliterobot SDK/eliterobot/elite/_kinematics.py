'''
Author: Elite_zhangjunjie
CreateDate: 
LastEditors: Elite_zhangjunjie
LastEditTime: 2022-05-20 11:47:19
Description: 
'''

from typing import List,Optional
from ._baseec import BaseEC

class ECKinematics(BaseEC):
    """EC运动学类,提供了机器人本身的运动学相关接口
    """
# 运动学服务
    def get_inverse_kinematic(self, pose: List[float] ,ref_joint: Optional[List[float]] = None, unit_type: Optional[int] = None) -> List[float]:
        """运动学逆解

        Args
        ----
            pose (List[float]): 需要进行逆解的位姿
            ref_joint (List[float]): 参考关节角
            unit_type (int, optional): 输入和返回位姿的单位类型,0:角度, 1:弧度, 不填默认弧度. Defaults to None.

        Returns
        -------
            list: 逆解后的关节角
        """
        if ref_joint == None:
            if unit_type is not None:
                return self.send_CMD("inverseKinematic",{"targetPose":pose, "unit_type":unit_type})
            else:
                return self.send_CMD("inverseKinematic",{"targetPose":pose})
        else:
            if unit_type is not None:
                return self.send_CMD("inverseKinematic",{"targetPose":pose,"referencePos":ref_joint, "unit_type":unit_type})
            else:
                return self.send_CMD("inverseKinematic",{"targetPose":pose,"referencePos":ref_joint})


    def get_forward_kinematic(self, joint: List[float], unit_type: Optional[int] = None) -> List[float]:
        """运动学正解

        Args
        ----
            joint (List[float]): 需要进行正解的关节角
            unit_type (int, optional): 输入和返回位姿的单位类型,0:角度, 1:弧度, 不填默认弧度. Defaults to None.

        Returns
        -------
            list: 正解后的位姿
        """
        if unit_type is not None:
            return self.send_CMD("forwardKinematic",{"targetPos":joint, "unit_type":unit_type})
        else:
            return self.send_CMD("forwardKinematic",{"targetPos":joint})


    def pose_mul(self, pose1: List[float], pose2: List[float], unit_type: Optional[int] = None) -> List[float]:
        """位姿相乘

        Args
        ----
            pose1 (List[float]): 位姿信息
            pose2 (List[float]): 位姿信息
            unit_type (int, optional): 输入和返回位姿的单位类型,0:角度, 1:弧度, 不填默认弧度. Defaults to None.

        Returns
        -------
            List[float]: 位姿相乘后的结果
        """
        if unit_type is not None:
            return self.send_CMD("poseMul",{"pose1":pose1, "pose2":pose2, "unit_type":unit_type})
        else:
            return self.send_CMD("poseMul",{"pose1":pose1, "pose2":pose2})
        

    def pose_inv(self, pose: List[float], unit_type: Optional[int] = None) -> List[float]:
        """位姿求逆

        Args
        ----
            pose (List[float]): 要求逆的位姿
            unit_type (int, optional): 输入和返回位姿的单位类型,0:角度, 1:弧度, 不填默认弧度. Defaults to None.

        Returns
        -------
            List[float]: 求逆后的结果
        """
        if unit_type is not None:
            return self.send_CMD("poseInv",{"pose":pose, "unit_type":unit_type})
        else:
            return self.send_CMD("poseInv",{"pose":pose})
        
        
    def convert_base_pose_to_user_pose(self, base_pose: List[float], user_num: int, unit_type: Optional[int] = None) -> List[float]:
        """基坐标系位姿转化为用户坐标系位姿

        Args
        ----
            cart_pose (List[float]): 基坐标系下的位姿数据
            user_num (int): 用户坐标系号
            unit_type (int, optional): 输入和返回位姿的单位类型,0:角度, 1:弧度, 不填默认弧度. Defaults to None.

        Returns
        -------
            List[float]: 用户坐标系下的位姿信息
        """
        if unit_type is not None:
            return self.send_CMD("convertPoseFromCartToUser",{"TargetPose":base_pose, "userNo":user_num, "unit_type":unit_type})
        else:
            return self.send_CMD("convertPoseFromCartToUser",{"TargetPose":base_pose, "userNo":user_num})
            
        
    def convert_user_pose_to_base_pose(self, user_pose: List[float], user_num: int, unit_type: Optional[int] = None) -> List[float]:
        """用户坐标系转化为基坐标系

        Args
        ----
            user_pose (List[float]): 用户坐标系下的数据
            user_num (int): 用户坐标系号
            unit_type (int, optional): 输入和返回位姿的单位类型,0:角度, 1:弧度, 不填默认弧度. Defaults to None.

        Returns
        -------
            List[float]: 基坐标系下的位姿信息
            
        Versions:
        

        """
        if unit_type is not None:
            return self.send_CMD("convertPoseFromUserToCart",{"TargetPose":user_pose, "userNo":user_num, "unit_type":unit_type})
        else:
            return self.send_CMD("convertPoseFromUserToCart",{"TargetPose":user_pose, "userNo":user_num})

    def load_hand_eye_calibration(self, csv_path: str):
        """Load hand-eye calibration matrix from CSV"""
        import csv
        import numpy as np

        matrix = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                matrix.append([float(x) for x in row])

        self.hand_eye_matrix = np.array(matrix)[:4, :4]
        # Check units roughly? No, just store it.
        return self.hand_eye_matrix.tolist()

    def pixel_to_base(self, u: float, v: float, z_depth: float, intrinsic_matrix: List[float], eye_in_hand: bool = False) -> Optional[List[float]]:
        """
        Convert pixel to robot base coordinates.
        Result unit depends on calibration matrix and depth unit.
        If calibration matrix is in meters and z_depth in meters, result is in meters.
        If eye_in_hand=True, robot pose (mm) is converted to meters automatically for calculation.
        """
        import numpy as np
        import math

        if not hasattr(self, 'hand_eye_matrix'):
            print("Error: Hand-eye calibration matrix not loaded. load_hand_eye_calibration() first.")
            return None

        # Intrinsic matrix 3x3 passed as list or array
        K = np.array(intrinsic_matrix).reshape(3, 3)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        x_c = (u - cx) * z_depth / fx
        y_c = (v - cy) * z_depth / fy
        z_c = z_depth

        P_cam = np.array([x_c, y_c, z_c, 1.0])

        if eye_in_hand:
            # Need current robot pose
            # Assuming self.current_pose returns [x, y, z, rx, ry, rz] in (mm, rad)
            if not hasattr(self, 'current_pose'):
                print("Error: current_pose method not available.")
                return None

            current_pose = self.current_pose
            if not current_pose:
                 print("Error: Could not get current pose")
                 return None

            x, y, z, rx, ry, rz = current_pose

            # Convert mm to meters for robot pose
            x /= 1000.0
            y /= 1000.0
            z /= 1000.0

            # Helper to convert euler to rotation matrix
            Rx = np.array([
                [1, 0, 0],
                [0, math.cos(rx), -math.sin(rx)],
                [0, math.sin(rx), math.cos(rx)]
            ])

            Ry = np.array([
                [math.cos(ry), 0, math.sin(ry)],
                [0, 1, 0],
                [-math.sin(ry), 0, math.cos(ry)]
            ])

            Rz = np.array([
                [math.cos(rz), -math.sin(rz), 0],
                [math.sin(rz), math.cos(rz), 0],
                [0, 0, 1]
            ])

            R_base2gripper = Rz @ Ry @ Rx
            T_base2gripper = np.eye(4)
            T_base2gripper[:3, :3] = R_base2gripper
            T_base2gripper[:3, 3] = [x, y, z]

            P_gripper = self.hand_eye_matrix @ P_cam
            P_base = T_base2gripper @ P_gripper

            return P_base[:3].tolist()
        else:
            # P_base = T_cam2base * P_cam
            P_base = self.hand_eye_matrix @ P_cam
            return P_base[:3].tolist()
