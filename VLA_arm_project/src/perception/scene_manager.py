import os
import random
import pybullet as p
import math
import numpy as np

class SceneManager:
    def __init__(self, assets_root):
        """
        assets_root: Path to 'assets' folder
        """
        self.objects_path = os.path.join(assets_root, "objects")
        self.table_z = 0.4  # Default, updated when table loads
        self.loaded_objects = []
        self.scan_available_objects()

    def scan_available_objects(self):
        """
        Scans the objects directory for folders containing model.urdf
        """
        self.available_models = []
        if not os.path.exists(self.objects_path):
            print(f"[SceneManager] Warning: Objects path not found: {self.objects_path}")
            return

        for item in os.listdir(self.objects_path):
            full_path = os.path.join(self.objects_path, item)
            if os.path.isdir(full_path):
                urdf_path = os.path.join(full_path, "model.urdf")
                if os.path.exists(urdf_path):
                    self.available_models.append({
                        "name": item,
                        "path": urdf_path
                    })

        print(f"[SceneManager] Found {len(self.available_models)} object models.")

    def load_table(self):
        """
        Loads the table URDF.
        """
        # Previous logic put table at [0.8, 0, 0]
        table_urdf = os.path.join(self.objects_path, "table.urdf")
        if not os.path.exists(table_urdf):
             # Fallback to pybullet_data plain table if custom not valid
             print("[SceneManager] Custom table.urdf not found, checking pybullet_data...")
             return None

        table_id = p.loadURDF(table_urdf, basePosition=[0.8, 0.0, 0.0], useFixedBase=True)
        self.table_z = 0.4 # Adjust based on your specific table model
        print(f"[SceneManager] Table loaded. Surface Z set to {self.table_z}")
        return table_id

    def clear_scene(self):
        for obj_id in self.loaded_objects:
            p.removeBody(obj_id)
        self.loaded_objects = []

    def _check_overlap(self, x, y, min_dist=0.15):
        """Check if new position is too close to existing objects."""
        for obj_id in self.loaded_objects:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            dist = math.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            if dist < min_dist:
                return True
        return False

    def _apply_dynamics_fix(self, obj_id, model_name):
        """Apply specific physics properties based on object type."""
        model_name = model_name.lower()

        # Defaults
        mass = 0.5
        lateral_friction = 1.0 # High friction to stop sliding
        rolling_friction = 0.02
        spinning_friction = 0.02
        linear_damping = 0.04
        angular_damping = 0.04

        if "book" in model_name:
            mass = 0.8
            rolling_friction = 0.5 # Books don't roll well
            lateral_friction = 1.5
        elif "pen" in model_name or "marker" in model_name:
            mass = 0.05
            rolling_friction = 0.1
        elif "cup" in model_name or "mug" in model_name:
            mass = 0.3
            # Cups might wobble if center of mass is high, add damping
            angular_damping = 0.1

        p.changeDynamics(obj_id, -1,
                         mass=mass,
                         lateralFriction=lateral_friction,
                         rollingFriction=rolling_friction,
                         spinningFriction=spinning_friction,
                         linearDamping=linear_damping,
                         angularDamping=angular_damping)

    def generate_messy_scene(self, num_objects=5):
        """
        Randomly places objects on the table.
        """
        self.clear_scene()
        if not self.available_models:
            print("[SceneManager] No models available to generate scene.")
            return []

        generated_info = []

        # Table Setup Area
        bounds_x = (0.50, 0.80)  # Reduced range to keep objects on table
        bounds_y = (-0.30, 0.30)

        for i in range(num_objects):
            # Try to find a non-overlapping position
            valid_pos = False
            for _ in range(20):
                pos_x = random.uniform(*bounds_x)
                pos_y = random.uniform(*bounds_y)
                if not self._check_overlap(pos_x, pos_y):
                    valid_pos = True
                    break

            if not valid_pos:
                continue

            model_data = random.choice(self.available_models)
            model_name = model_data["name"]

            # Drop from different heights to avoid collision during spawn
            pos_z = self.table_z + 0.15 + (i * 0.05)

            # Random Orientation (Yaw only to respect gravity alignment)
            orn = p.getQuaternionFromEuler([
                0.0,
                0.0,
                random.uniform(-math.pi, math.pi)
            ])

            # Scale - random small variation
            scale = random.uniform(0.9, 1.1)

            try:
                # Load URDF with Texture Support
                flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL

                obj_id = p.loadURDF(
                    model_data["path"],
                    basePosition=[pos_x, pos_y, pos_z],
                    baseOrientation=orn,
                    globalScaling=scale,
                    flags=flags
                )

                # Reset visual color to white to allow texture to show through
                p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 1])

                # Apply Dynamics Fixes
                self._apply_dynamics_fix(obj_id, model_name)

                self.loaded_objects.append(obj_id)
                generated_info.append(model_name)

                # Short settle for individual object to avoid immediate collision with next spawn
                for _ in range(50): p.stepSimulation()

            except Exception as e:
                print(f"[SceneManager] Failed to load {model_name}: {e}")

        # Final Robust Settle
        print("[SceneManager] Waiting for objects to settle...")
        for _ in range(240):
            p.stepSimulation()

        return generated_info
