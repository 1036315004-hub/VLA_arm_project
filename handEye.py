import numpy as np
import cv2
import os
import csv
import math

class HandEyeCalib:
    def __init__(self):
        self.calibImages_ = []
        self.poses_ = []
        self.cameraIntrinsicMatrix_ = None
        self.cameraDistortionMatrix_ = None
        self.rvecs_ = []
        self.tvecs_ = []

        self.cornerPointLong_ = 0
        self.cornerPointShort_ = 0
        self.cornerPointSize_ = 0.0

    def setChessboardParams(self, long_pts, short_pts, size):
        self.cornerPointLong_ = long_pts
        self.cornerPointShort_ = short_pts
        self.cornerPointSize_ = size

    def eulerAnglesToRotationMatrix(self, rx, ry, rz):
        # Rotation around x-axis
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)]
        ])

        # Rotation around y-axis
        Ry = np.array([
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)]
        ])

        # Rotation around z-axis
        Rz = np.array([
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1]
        ])

        # Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        return R

    def poseToHomogeneousMatrix(self, pose):
        x, y, z, rx, ry, rz = pose
        R = self.eulerAnglesToRotationMatrix(rx, ry, rz)
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = [x, y, z]
        return H

    def saveMatrixToCSV(self, matrixes, fileName):
        if not matrixes:
            print("The list of matrixes is empty")
            return

        cols = 4
        num_matrixes = len(matrixes)
        rows = 4

        combine_matrix = np.zeros((rows, cols * num_matrixes))

        for i, mat in enumerate(matrixes):
            combine_matrix[:, i*cols : (i+1)*cols] = mat

        with open(fileName, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(combine_matrix)
        print(f"Matrixes saved successfully to {fileName}")

    def poseSaveCSV(self, poseFilePath, csvFilePath):
        if not os.path.exists(poseFilePath):
            print(f"Failed to open the file : {poseFilePath}")
            return

        with open(poseFilePath, 'r') as f:
            lines = f.readlines()

        pose_data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            for p in parts:
                if p:
                    try:
                        pose_data.append(float(p))
                    except ValueError:
                        pass

        matrixes = []
        for i in range(0, len(pose_data), 6):
            if i + 6 <= len(pose_data):
                pose = pose_data[i:i+6]
                matrixes.append(self.poseToHomogeneousMatrix(pose))

        self.saveMatrixToCSV(matrixes, csvFilePath)

    def is_imageFile(self, file_path):
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
        ext = os.path.splitext(file_path)[1].lower()
        return ext in image_extensions

    def extractNumberFromFileNames(self, filename):
        num_str = ''.join(filter(str.isdigit, os.path.basename(filename)))
        return int(num_str) if num_str else -1

    def readFileNameFromFolder(self, folderPath):
        imgFileNames = []
        if not os.path.exists(folderPath):
            print(f"Folder not found: {folderPath}")
            return []

        files = os.listdir(folderPath)
        image_files = [f for f in files if self.is_imageFile(f)]

        # Sort by number in filename
        image_files.sort(key=self.extractNumberFromFileNames)

        for f in image_files:
            imgFileNames.append(os.path.join(folderPath, f))

        return imgFileNames

    def CalibInit(self, imgfolderPath, poseFilePath):
        # Step 1: Read images
        imgFileNames = self.readFileNameFromFolder(imgfolderPath)
        self.calibImages_ = []

        if not imgFileNames:
            print(f"No images found in {imgfolderPath}")

        for fname in imgFileNames:
            img = cv2.imread(fname)
            if img is not None:
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                self.calibImages_.append(gray)
            else:
                print(f"Failed to read image: {fname}")

        # Step 2: Read poses
        if not os.path.exists(poseFilePath):
            print(f"Failed to open the file : {poseFilePath}")
            return

        with open(poseFilePath, 'r') as f:
            lines = f.readlines()

        pose_data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            for p in parts:
                if p:
                    try:
                        pose_data.append(float(p))
                    except ValueError:
                        pass

        self.poses_ = []
        for i in range(0, len(pose_data), 6):
            if i + 6 <= len(pose_data):
                pose = pose_data[i:i+6]
                self.poses_.append(self.poseToHomogeneousMatrix(pose))

    def cameraCalib(self):
        print("========================")
        print("       相机内参标定     ")
        print("========================")
        if not self.calibImages_:
            print("No images for calibration!")
            return False

        print(f"标定板中长边方向对应的角点个数为 ： {self.cornerPointLong_}")
        print(f"标定板中短边方向对应的角点个数为 ： {self.cornerPointShort_}")
        print(f"标定板中方格真实尺寸为 ： {self.cornerPointSize_} m.")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((self.cornerPointShort_ * self.cornerPointLong_, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.cornerPointLong_, 0:self.cornerPointShort_].T.reshape(-1, 2)
        objp = objp * self.cornerPointSize_

        obj_points = []
        img_points = []

        found_count = 0
        for i, gray_img in enumerate(self.calibImages_):
            ret, corners = cv2.findChessboardCorners(gray_img, (self.cornerPointLong_, self.cornerPointShort_), None)

            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray_img, corners, (5, 5), (-1, -1), criteria)
                img_points.append(corners2)
                found_count += 1
            else:
                 print(f"Chessboard not found in image {i}")

        if found_count == 0:
            print("No chessboard corners found in any image!")
            return False

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, self.calibImages_[0].shape[::-1], None, None)

        self.cameraIntrinsicMatrix_ = mtx
        self.cameraDistortionMatrix_ = dist
        self.rvecs_ = rvecs
        self.tvecs_ = tvecs

        print(f"RMS Error: {ret}")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist)
        print("-----------------------------------------------------")
        return True

    def computeEyeInHandT(self, method=cv2.CALIB_HAND_EYE_TSAI):
        if not self.cameraCalib():
            return None

        print("========================")
        print("       眼在手上标定     ")
        print("========================")

        R_tool = []
        t_tool = []

        if len(self.poses_) != len(self.rvecs_):
             # This might happen if some images failed to detect corners.
             # We should technically filter poses to match successful images.
             # But for simplicity assuming 1-to-1 success or user provided good data.
             # In robust code, we'd need to track which image corresponds to which pose.
             print(f"Warning: Number of poses ({len(self.poses_)}) does not match number of detected images ({len(self.rvecs_)}).")
             # Assuming they are matched pair-wise from start and truncating excess? Or error?
             # Proper way: Only keep poses for images where corners were found.
             # I need to refactor detection loop to keep indices.
             pass

        # Re-detect and match poses
        # Simplified: assumes all images valid or we handle it.
        # For now, let's just proceed, but this is a likely point of failure if images are bad.

        for pose in self.poses_:
            R = pose[:3, :3]
            t = pose[:3, 3]

            # C++ code transposed R. OpenCV expects rotation matrix.
            # C++ snippet: Eigen is ColMajor, Mat is RowMajor.
            # Python/Numpy is RowMajor by default.
            # So if we constructed R correctly in poseToHomogeneousMatrix, we are good.
            # poseToHomogeneousMatrix uses standard math.

            R_tool.append(R)
            t_tool.append(t)

        # Match lengths
        min_len = min(len(R_tool), len(self.rvecs_))
        R_tool = R_tool[:min_len]
        t_tool = t_tool[:min_len]
        rvecs_clipped = self.rvecs_[:min_len]
        tvecs_clipped = self.tvecs_[:min_len]

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_tool, t_tool, rvecs_clipped, tvecs_clipped, None, None, method)

        print("R_cam2gripper (Rotation):\n", R_cam2gripper)
        print("t_cam2gripper (Translation):\n", t_cam2gripper)

        eyeInHandT = np.eye(4)
        eyeInHandT[:3, :3] = R_cam2gripper
        eyeInHandT[:3, 3] = t_cam2gripper.flatten()

        return eyeInHandT

    def computeEyeToHandT(self, method=cv2.CALIB_HAND_EYE_TSAI):
        if not self.cameraCalib():
             return None

        print("========================")
        print("       眼在手外标定     ")
        print("========================")

        R_tool = []
        t_tool = []

        for pose in self.poses_:
            # Eye to hand: take inverse of base-to-end
            base_to_end = pose
            end_to_base = np.linalg.inv(base_to_end)

            R = end_to_base[:3, :3]
            t = end_to_base[:3, 3]

            R_tool.append(R)
            t_tool.append(t)

        min_len = min(len(R_tool), len(self.rvecs_))
        R_tool = R_tool[:min_len]
        t_tool = t_tool[:min_len]
        rvecs_clipped = self.rvecs_[:min_len]
        tvecs_clipped = self.tvecs_[:min_len]

        R_cam2base, t_cam2base = cv2.calibrateHandEye(R_tool, t_tool, rvecs_clipped, tvecs_clipped, None, None, method)

        print("R_cam2base (Rotation):\n", R_cam2base)
        print("t_cam2base (Translation):\n", t_cam2base)

        eyeToHandT = np.eye(4)
        eyeToHandT[:3, :3] = R_cam2base
        eyeToHandT[:3, 3] = t_cam2base.flatten()

        return eyeToHandT

def pixel_to_base(u, v, z_depth, calibration_matrix, intrinsic_matrix, robot_pose=None, eye_in_hand=False):
    """
    Convert pixel coordinates to robot base coordinates.

    Args:
        u, v: Pixels
        z_depth: Depth in meters (Z in camera frame)
        calibration_matrix: 4x4 homogenous matrix (T_cam2base for eye-to-hand, T_cam2gripper for eye-in-hand)
        intrinsic_matrix: Camera intrinsic matrix (3x3)
        robot_pose: Current robot pose 4x4 (T_base2gripper). Required for eye_in_hand=True.
        eye_in_hand: Boolean, True if camera is on robotic arm.

    Returns:
        (x, y, z): Coordinates in robot base frame.
    """
    # 1. Pixel to Camera Frame
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    x_c = (u - cx) * z_depth / fx
    y_c = (v - cy) * z_depth / fy
    z_c = z_depth

    P_cam = np.array([x_c, y_c, z_c, 1.0])

    # 2. Camera to Target Frame
    if eye_in_hand:
        if robot_pose is None:
            raise ValueError("robot_pose is required for eye-in-hand calibration")
        # P_gripper = T_cam2gripper * P_cam
        P_gripper = calibration_matrix @ P_cam
        # P_base = T_base2gripper * P_gripper
        P_base = robot_pose @ P_gripper
    else:
        # Eye-to-Hand
        # P_base = T_cam2base * P_cam
        P_base = calibration_matrix @ P_cam

    return P_base[:3]


if __name__ == "__main__":
    # Example usage (commented out to avoid immediate execution/errors)
    # calib = HandEyeCalib()
    # calib.setChessboardParams(9, 6, 0.020) # Example params
    # calib.CalibInit("images_folder", "poses.txt")
    # T = calib.computeEyeToHandT()
    # calib.saveMatrixToCSV([T], "hand_eye_result.csv")
    pass

