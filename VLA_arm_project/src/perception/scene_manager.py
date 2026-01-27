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

    def _check_overlap(self, x, y, min_dist=0.15, extra_bodies=None):
        """Check if new position is too close to existing objects or extra bodies."""
        bodies = self.loaded_objects + (extra_bodies if extra_bodies else [])
        for obj_id in bodies:
            if obj_id is None: continue
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            dist = math.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            if dist < min_dist:
                return True
        return False

    def settle_objects(self, max_steps=1000, velocity_threshold=0.005):
        """Waits for objects to stop moving."""
        print("[SceneManager] Waiting for objects to settle...")
        for step in range(max_steps):
            p.stepSimulation()
            if step % 20 == 0:
                is_static = True
                for obj_id in self.loaded_objects:
                    v_lin, v_ang = p.getBaseVelocity(obj_id)
                    if np.linalg.norm(v_lin) > velocity_threshold or np.linalg.norm(v_ang) > velocity_threshold:
                        is_static = False
                        break
                if is_static and step > 50:
                    print(f"[SceneManager] Objects settled at step {step}.")
                    return
        print("[SceneManager] Settle timeout reached.")

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

    def spawn_obstacles(self, robot_id=None, robot_base_pos=None):
        """Spawns fixed obstacles (Book Holder, Pen Container, Books) relative to robot."""
        # Table bounds
        table_x_min, table_x_max = 0.4, 1.2
        table_y_min, table_y_max = -0.6, 0.6

        # Identify obstacle types
        target_keywords = ["book_holder", "pen_container", "book"]
        obstacle_models = []
        for m in self.available_models:
            name_lower = m["name"].lower()
            if any(k in name_lower for k in target_keywords):
                obstacle_models.append(m)

        if not obstacle_models: return []

        # Allow all obstacles (remove random subset limit)
        selected_obstacles = obstacle_models

        spawned = []
        extra_bodies = [robot_id] if robot_id is not None else []

        # Track naming
        object_counts = {}

        for model_data in selected_obstacles:
            model_name = model_data["name"]

            # Generate Unique Name
            if model_name not in object_counts:
                object_counts[model_name] = 1
            else:
                object_counts[model_name] += 1

            unique_name = f"{model_name}_{object_counts[model_name]}"

            valid_pos = False
            pos_x, pos_y = 0.8, 0.0 # Default fallback

            # Try to place relative to robot if provided
            for _ in range(50):
                if robot_base_pos is not None:
                    # Spawn in front/around robot
                    # Distance: 0.3 to 0.7m
                    # Angle: -60 to +60 degrees relative to robot facing table center?
                    # Robot faces table center. Vector Robot->Center.
                    # We want objects roughly in that direction.

                    dist = random.uniform(0.35, 0.75)
                    angle = random.uniform(-math.pi/2, math.pi/2)

                    # Robot yaw logic was: pointing to table center.
                    # We can infer robot yaw or just use table center direction?
                    # Let's just use global table area but bias towards robot?
                    # Simpler: Generate in table bounds, check distance to robot.

                    pos_x = random.uniform(table_x_min, table_x_max)
                    pos_y = random.uniform(table_y_min, table_y_max)

                    # Check reachability from robot
                    dist_to_robot = math.sqrt((pos_x - robot_base_pos[0])**2 + (pos_y - robot_base_pos[1])**2)
                    if dist_to_robot < 0.3 or dist_to_robot > 0.85: # Too close or too far
                        continue
                else:
                    pos_x = random.uniform(0.5, 0.9)
                    pos_y = random.uniform(-0.3, 0.3)

                if not self._check_overlap(pos_x, pos_y, min_dist=0.20, extra_bodies=extra_bodies):
                    valid_pos = True
                    break

            if not valid_pos: continue

            pos_z = self.table_z + 0.05

            # Orientation logic
            orn = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)])

            # Special handling for books to be flat
            is_book_item = "book" in model_name.lower() and "holder" not in model_name.lower()

            try:
                flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
                obj_id = p.loadURDF(model_data["path"], [pos_x, pos_y, pos_z], orn, globalScaling=1.0, flags=flags)

                if is_book_item:
                    # Check dimensions to ensure flat
                    aabb_min, aabb_max = p.getAABB(obj_id)
                    dims = np.array(aabb_max) - np.array(aabb_min)
                    # If Z is not the smallest dimension, rotate 90 deg around X or Y
                    sorted_dims_idx = np.argsort(dims) # 0 is smallest

                    # We want smallest dimension to be Z (index 2 in world, but might be different local)
                    # If loaded with 0,0,0, and Z is big, we need to rotate.
                    # Hardcoded heuristics for common book meshes:
                    # Often X is thickness or Y is thickness.
                    # Try rotating 90 deg around X
                    if dims[2] > dims[0] or dims[2] > dims[1]:
                        # Re-spawn or reset orientation
                        # Try laying flat: Pitch=90
                        orn_flat = p.getQuaternionFromEuler([1.57, 0, random.uniform(-math.pi, math.pi)])
                        p.resetBasePositionAndOrientation(obj_id, [pos_x, pos_y, pos_z], orn_flat)

                p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 1])
                self._apply_dynamics_fix(obj_id, model_name)
                p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])
                self.loaded_objects.append(obj_id)
                spawned.append(unique_name)
                for _ in range(10): p.stepSimulation()
            except Exception as e:
                print(f"[SceneManager] Failed to load {model_name}: {e}")

        return spawned

    def spawn_random_objects(self, num_objects, robot_id=None, robot_base_pos=None):
        """Spawns random drop objects relative to robot."""
        # Table spread area
        table_x_min, table_x_max = 0.5, 0.9
        table_y_min, table_y_max = -0.4, 0.4

        target_keywords = ["book_holder", "pen_container", "book"]
        random_models = []
        for m in self.available_models:
            name_lower = m["name"].lower()
            if not any(k in name_lower for k in target_keywords):
                random_models.append(m)

        if not random_models: return []

        spawned = []
        extra_bodies = [robot_id] if robot_id is not None else []
        object_counts = {}

        for i in range(num_objects):
            valid_pos = False
            pos_x, pos_y = 0.8, 0.0 # Default fallback
            for _ in range(50):
                if robot_base_pos is not None:
                    pos_x = random.uniform(table_x_min, table_x_max)
                    pos_y = random.uniform(table_y_min, table_y_max)

                    # Check reachability from robot
                    dist_to_robot = math.sqrt((pos_x - robot_base_pos[0])**2 + (pos_y - robot_base_pos[1])**2)
                    if dist_to_robot < 0.35 or dist_to_robot > 0.70:
                        continue
                else:
                    pos_x = random.uniform(table_x_min, table_x_max)
                    pos_y = random.uniform(table_y_min, table_y_max)

                if not self._check_overlap(pos_x, pos_y, min_dist=0.15, extra_bodies=extra_bodies):
                    valid_pos = True
                    break

            if not valid_pos: continue

            model_data = random.choice(random_models)
            model_name = model_data["name"]

            # Generate Unique Name
            if model_name not in object_counts:
                object_counts[model_name] = 1
            else:
                object_counts[model_name] += 1
            unique_name = f"{model_name}_{object_counts[model_name]}"

            # Spawn close to table (no drop)
            pos_z = self.table_z + 0.05
            # Only Yaw rotation
            orn = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)])
            scale = random.uniform(0.9, 1.1)

            try:
                flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
                obj_id = p.loadURDF(model_data["path"], [pos_x, pos_y, pos_z], orn, globalScaling=scale, flags=flags)
                p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 1])
                self._apply_dynamics_fix(obj_id, model_name)
                p.resetBaseVelocity(obj_id, [0,0,0], [0,0,0])

                self.loaded_objects.append(obj_id)
                spawned.append(unique_name)
                for _ in range(20): p.stepSimulation()
            except Exception as e:
                print(f"[SceneManager] Failed to load {model_name}: {e}")

        return spawned

    def generate_messy_scene(self, num_objects=5):
        """Legacy wrapper if needed, but prefer calling spawn methods directly."""
        self.clear_scene()
        names = []
        names += self.spawn_obstacles()
        self.settle_objects()
        names += self.spawn_random_objects(num_objects)
        self.settle_objects()
        return names





