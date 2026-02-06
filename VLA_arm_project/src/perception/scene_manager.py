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
        self.object_registry = {} # Maps unique_name -> obj_id
        self.scan_available_objects()

    def scan_available_objects(self):
        """
        Scans the objects directory for folders containing model.urdf
        """
        self.available_models = []
        if not os.path.exists(self.objects_path):
            # print(f"[SceneManager] Warning: Objects path not found: {self.objects_path}")
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

        # print(f"[SceneManager] Found {len(self.available_models)} object models.")

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
        # print(f"[SceneManager] Table loaded. Surface Z set to {self.table_z}")
        return table_id

    def clear_scene(self):
        for obj_id in self.loaded_objects:
            p.removeBody(obj_id)
        self.loaded_objects = []
        self.object_registry = {}

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

    def _aabb_xy_overlap(self, aabb_a, aabb_b, margin=0.002):
        """Return True if two AABBs overlap in XY within a small margin."""
        (a_min, a_max) = aabb_a
        (b_min, b_max) = aabb_b
        x_overlap = (a_min[0] - margin) <= b_max[0] and (a_max[0] + margin) >= b_min[0]
        y_overlap = (a_min[1] - margin) <= b_max[1] and (a_max[1] + margin) >= b_min[1]
        return x_overlap and y_overlap

    def settle_objects(self, max_steps=1000, velocity_threshold=0.005):
        """Waits for objects to stop moving."""
        # print("[SceneManager] Waiting for objects to settle...")
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
                    # print(f"[SceneManager] Objects settled at step {step}.")
                    return
        # print("[SceneManager] Settle timeout reached.")

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

    def _is_pose_valid(self, new_obj_id, extra_bodies=None, min_clearance=0.02):
        """
        Check if the newly spawned object collides with any existing object or extra bodies.
        Uses physics engine AABB/ClosestPoints query instead of simple distance.
        """
        check_targets = self.loaded_objects + (extra_bodies if extra_bodies else [])

        p.performCollisionDetection()
        new_aabb = p.getAABB(new_obj_id)
        for target_id in check_targets:
            if target_id is None or target_id == new_obj_id: continue

            # Prevent stacking: reject XY overlap with existing AABB
            try:
                target_aabb = p.getAABB(target_id)
                if self._aabb_xy_overlap(new_aabb, target_aabb):
                    return False
            except Exception:
                pass

            # Get closest points
            # If distance is negative (penetration) or < min_clearance, it's a 'hit'
            pts = p.getClosestPoints(bodyA=new_obj_id, bodyB=target_id, distance=min_clearance)
            if len(pts) > 0:
                # Double check - sometimes pts are returned for nearby but not colliding if distance > 0
                for pt in pts:
                    if pt[8] < min_clearance: # pt[8] is contact distance
                        return False
        return True

    def _drop_and_settle(self, obj_id, steps=150):
        """
        Let the object fall under gravity to find a stable pose.
        Returns False if object fell off the table.
        """
        for _ in range(steps):
            p.stepSimulation()

        pos, _ = p.getBasePositionAndOrientation(obj_id)
        # Check if object fell below table level (margin 5cm)
        if pos[2] < (self.table_z - 0.05):
            return False
        return True

    def _spawn_single_object(self, model_data, unique_name, bounds, robot_base_pos, extra_bodies):
        """
        Attempts to spawn a single object with physics validation.
        """
        x_min, x_max, y_min, y_max = bounds

        for attempt in range(20): # Try 20 times to find a valid spot
            # 1. Sample Position
            if robot_base_pos:
                # Spawn in arc around robot or focused area
                pos_x = random.uniform(x_min, x_max)
                pos_y = random.uniform(y_min, y_max)
                # Ensure reachability constraint
                dist_to_robot = math.sqrt((pos_x - robot_base_pos[0])**2 + (pos_y - robot_base_pos[1])**2)
                if dist_to_robot < 0.35 or dist_to_robot > 0.70:
                    continue
            else:
                pos_x = random.uniform(x_min, x_max)
                pos_y = random.uniform(y_min, y_max)

            # 2. Setup Initial Orientation & Height
            # Spawn slightly higher to ensure no initial overlap
            pos_z = self.table_z + 0.02
            orn = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)])

            # Book special handling (Orientation)
            is_book = "book" in model_data["name"].lower() and "holder" not in model_data["name"].lower()
            if is_book:
                # Start flat (Pitch=0)
                orn = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)])

            # 3. Load & Validate
            flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            scale = random.uniform(0.9, 1.1) if not is_book else 1.0

            obj_id = p.loadURDF(model_data["path"], [pos_x, pos_y, pos_z], orn, globalScaling=scale, flags=flags)

            # Place object directly on the table surface to avoid drop stacking
            aabb_min, aabb_max = p.getAABB(obj_id)
            delta_z = (self.table_z + 0.001) - aabb_min[2]
            if abs(delta_z) > 1e-6:
                p.resetBasePositionAndOrientation(obj_id, [pos_x, pos_y, pos_z + delta_z], orn)

            # Check for Book Dimensions & Rotate if needed (to lay flat)
            if is_book:
                aabb_min, aabb_max = p.getAABB(obj_id)
                dims = np.array(aabb_max) - np.array(aabb_min)
                # If Z is large (standing up), rotate it flat
                if dims[2] > dims[0] or dims[2] > dims[1]:
                    p.resetBasePositionAndOrientation(obj_id, [pos_x, pos_y, pos_z + delta_z],
                                                      p.getQuaternionFromEuler([1.57, 0, random.uniform(-math.pi, math.pi)]))

            # 4. Check Overlap using Physics (Immediate Check)
            if not self._is_pose_valid(obj_id, extra_bodies, min_clearance=0.02):
                p.removeBody(obj_id)
                continue # Retry new position

            # 5. Apply Dynamics
            self._apply_dynamics_fix(obj_id, model_data["name"])

            # 6. Minimal settle to relax any tiny penetrations
            if not self._drop_and_settle(obj_id, steps=30):
                # Fell off table
                p.removeBody(obj_id)
                continue

            # 7. Final Check after settling (did it roll into something?)
            if not self._is_pose_valid(obj_id, extra_bodies, min_clearance=0.005):
                # If it settled INTO another object, removing is safer for clean data
                p.removeBody(obj_id)
                continue

            # Success
            p.changeVisualShape(obj_id, -1, rgbaColor=[1, 1, 1, 1])
            self.loaded_objects.append(obj_id)
            self.object_registry[unique_name] = obj_id
            return unique_name

        # print(f"[SceneManager] Failed to place {unique_name} after max attempts.")
        return None

    def spawn_obstacles(self, robot_id=None, robot_base_pos=None):
        """Spawns specific obstacles (Books/Containers)."""
        table_bounds = (0.4, 1.0, -0.6, 0.6) # X_min, X_max, Y_min, Y_max

        target_keywords = ["book_holder", "pen_container", "book"]
        obstacle_models = [m for m in self.available_models if any(k in m["name"].lower() for k in target_keywords)]
        if not obstacle_models: return []

        spawned = []
        extra_bodies = [robot_id] if robot_id is not None else []
        object_counts = {}

        # Prioritize list order or random subset? Prioritize usually.
        # User previously wanted random subset.
        # Let's take a random subset if list is large? Or all.
        # Original logic: "Allow all obstacles"

        for model_data in obstacle_models:
            name = model_data["name"]
            object_counts[name] = object_counts.get(name, 0) + 1
            unique_name = f"{name}_{object_counts[name]}"

            res = self._spawn_single_object(model_data, unique_name, table_bounds, robot_base_pos, extra_bodies)
            if res: spawned.append(res)

        return spawned

    def spawn_random_objects(self, num_objects, robot_id=None, robot_base_pos=None):
        """Spawns random drop objects relative to robot."""
        # Table spread area
        # Tighter bounds for small objects central area
        table_bounds = (0.45, 0.95, -0.35, 0.35)

        target_keywords = ["book_holder", "pen_container", "book"]
        random_models = [m for m in self.available_models if not any(k in m["name"].lower() for k in target_keywords)]
        if not random_models: return []

        spawned = []
        extra_bodies = [robot_id] if robot_id is not None else []
        # Count needs to be globally unique if possible, but here local + existing check

        for i in range(num_objects):
            model_data = random.choice(random_models)
            name = model_data["name"]

            # Simple global check simulation
            # (In a real system we might query registry, but here we can just append index based on loaded)
            existing_of_type = sum(1 for k in self.object_registry if k.startswith(name))
            unique_name = f"{name}_{existing_of_type + 1}"

            res = self._spawn_single_object(model_data, unique_name, table_bounds, robot_base_pos, extra_bodies)
            if res: spawned.append(res)

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

