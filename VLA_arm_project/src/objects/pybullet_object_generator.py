"""
PyBullet Object Generator

This module uses PyBullet's built-in Python resources to auto-generate
classic grasping objects for simulation. Objects can be loaded and
randomly positioned for data collection and training.

Supported object types:
- Basic shapes: cubes, spheres, cylinders, capsules
- PyBullet built-in objects: duck, teddy, mug, etc.
- Random colored objects for visual diversity
"""

import os
import random
import math
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pybullet as p
import pybullet_data


class PyBulletObjectGenerator:
    """
    Generator for creating and managing grasping objects in PyBullet simulation.
    
    Uses PyBullet's built-in resources and generates custom shapes to provide
    a diverse set of objects for robotic grasping tasks.
    """
    
    # Available colors (RGBA)
    COLORS = {
        'red': [1.0, 0.0, 0.0, 1.0],
        'green': [0.0, 1.0, 0.0, 1.0],
        'blue': [0.0, 0.0, 1.0, 1.0],
        'yellow': [1.0, 1.0, 0.0, 1.0],
        'cyan': [0.0, 1.0, 1.0, 1.0],
        'magenta': [1.0, 0.0, 1.0, 1.0],
        'orange': [1.0, 0.5, 0.0, 1.0],
        'purple': [0.5, 0.0, 1.0, 1.0],
        'pink': [1.0, 0.5, 0.75, 1.0],
        'brown': [0.6, 0.3, 0.0, 1.0],
    }
    
    # Shape types
    SHAPE_BOX = 'box'
    SHAPE_SPHERE = 'sphere'
    SHAPE_CYLINDER = 'cylinder'
    SHAPE_CAPSULE = 'capsule'
    
    # Built-in PyBullet objects
    BUILTIN_OBJECTS = [
        'duck_vhacd.urdf',
        'teddy_vhacd.urdf',
        'sphere2.urdf',
        'cube.urdf',
        'lego/lego.urdf',
        'random_urdfs/000/000.urdf',
    ]
    
    def __init__(self, pybullet_data_path: Optional[str] = None):
        """
        Initialize the object generator.
        
        Args:
            pybullet_data_path: Path to PyBullet data directory
        """
        if pybullet_data_path is None:
            self.pybullet_data_path = pybullet_data.getDataPath()
        else:
            self.pybullet_data_path = pybullet_data_path
            
        self.generated_objects: List[Dict[str, Any]] = []
        self.object_ids: List[int] = []
        
    def create_basic_shape(
        self,
        shape_type: str,
        position: List[float],
        color: Optional[str] = None,
        size: Optional[float] = None,
        mass: float = 0.1,
        orientation: Optional[List[float]] = None
    ) -> int:
        """
        Create a basic shape object.
        
        Args:
            shape_type: Type of shape ('box', 'sphere', 'cylinder', 'capsule')
            position: [x, y, z] position
            color: Color name from COLORS dict (random if None)
            size: Size scale factor (random if None)
            mass: Object mass in kg
            orientation: Quaternion orientation (identity if None)
            
        Returns:
            PyBullet body ID of created object
        """
        # Random color if not specified
        if color is None:
            color = random.choice(list(self.COLORS.keys()))
        rgba = self.COLORS.get(color, self.COLORS['red'])
        
        # Random size if not specified
        if size is None:
            size = random.uniform(0.02, 0.05)
        
        # Default orientation
        if orientation is None:
            orientation = [0, 0, 0, 1]
        
        # Create shape based on type
        if shape_type == self.SHAPE_BOX:
            half_extents = [size, size, size]
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents
            )
        elif shape_type == self.SHAPE_SPHERE:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=size,
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=size
            )
        elif shape_type == self.SHAPE_CYLINDER:
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=size,
                length=size * 2,
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=size,
                height=size * 2
            )
        elif shape_type == self.SHAPE_CAPSULE:
            visual_shape = p.createVisualShape(
                p.GEOM_CAPSULE,
                radius=size,
                length=size * 2,
                rgbaColor=rgba
            )
            collision_shape = p.createCollisionShape(
                p.GEOM_CAPSULE,
                radius=size,
                height=size * 2
            )
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        # Create multi-body
        object_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        # Set friction for stable grasping
        p.changeDynamics(
            object_id, -1,
            lateralFriction=1.0,
            rollingFriction=0.001,
            spinningFriction=0.001
        )
        
        # Store object info
        obj_info = {
            'id': object_id,
            'type': 'basic',
            'shape': shape_type,
            'color': color,
            'size': size,
            'position': position,
            'mass': mass
        }
        self.generated_objects.append(obj_info)
        self.object_ids.append(object_id)
        
        return object_id
    
    def load_builtin_object(
        self,
        urdf_name: str,
        position: List[float],
        scale: float = 1.0,
        orientation: Optional[List[float]] = None
    ) -> int:
        """
        Load a built-in PyBullet object.
        
        Args:
            urdf_name: Name of the URDF file
            position: [x, y, z] position
            scale: Global scale factor
            orientation: Quaternion orientation
            
        Returns:
            PyBullet body ID of loaded object
        """
        if orientation is None:
            orientation = [0, 0, 0, 1]
        
        urdf_path = os.path.join(self.pybullet_data_path, urdf_name)
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        
        object_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=orientation,
            globalScaling=scale
        )
        
        # Set friction
        p.changeDynamics(
            object_id, -1,
            lateralFriction=1.0,
            rollingFriction=0.001,
            spinningFriction=0.001
        )
        
        obj_info = {
            'id': object_id,
            'type': 'builtin',
            'urdf': urdf_name,
            'scale': scale,
            'position': position
        }
        self.generated_objects.append(obj_info)
        self.object_ids.append(object_id)
        
        return object_id
    
    def generate_random_objects(
        self,
        num_objects: int,
        workspace_bounds: Tuple[float, float, float, float],
        table_height: float,
        min_distance: float = 0.1,
        use_builtin: bool = False
    ) -> List[int]:
        """
        Generate random objects within the workspace.
        
        Args:
            num_objects: Number of objects to generate
            workspace_bounds: (x_min, x_max, y_min, y_max) of workspace
            table_height: Height of the table surface
            min_distance: Minimum distance between objects
            use_builtin: Whether to include PyBullet built-in objects
            
        Returns:
            List of created object IDs
        """
        x_min, x_max, y_min, y_max = workspace_bounds
        created_ids = []
        positions = []
        
        shape_types = [self.SHAPE_BOX, self.SHAPE_SPHERE, self.SHAPE_CYLINDER]
        colors = list(self.COLORS.keys())
        
        for _ in range(num_objects):
            # Try to find a valid position
            for attempt in range(50):
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                
                # Check minimum distance from other objects
                valid = True
                for px, py in positions:
                    if math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_distance:
                        valid = False
                        break
                
                if valid:
                    break
            else:
                continue  # Skip if can't find valid position
            
            # Random size
            size = random.uniform(0.025, 0.04)
            z = table_height + size + 0.005
            
            # Decide whether to use basic shape or builtin
            if use_builtin and random.random() < 0.3:
                # Try to load a builtin object
                builtin_urdfs = ['duck_vhacd.urdf', 'sphere2.urdf', 'cube.urdf']
                urdf = random.choice(builtin_urdfs)
                try:
                    obj_id = self.load_builtin_object(
                        urdf,
                        [x, y, z],
                        scale=random.uniform(0.5, 1.5)
                    )
                    created_ids.append(obj_id)
                    positions.append((x, y))
                except FileNotFoundError:
                    # Fallback to basic shape
                    shape = random.choice(shape_types)
                    color = random.choice(colors)
                    obj_id = self.create_basic_shape(
                        shape, [x, y, z], color, size
                    )
                    created_ids.append(obj_id)
                    positions.append((x, y))
            else:
                # Create basic shape
                shape = random.choice(shape_types)
                color = random.choice(colors)
                
                # Random orientation for boxes
                if shape == self.SHAPE_BOX:
                    yaw = random.uniform(0, 2 * math.pi)
                    orientation = p.getQuaternionFromEuler([0, 0, yaw])
                else:
                    orientation = None
                
                obj_id = self.create_basic_shape(
                    shape, [x, y, z], color, size, orientation=orientation
                )
                created_ids.append(obj_id)
                positions.append((x, y))
        
        return created_ids
    
    def generate_color_set(
        self,
        workspace_bounds: Tuple[float, float, float, float],
        table_height: float,
        colors_to_use: Optional[List[str]] = None
    ) -> List[int]:
        """
        Generate a set of objects with specific colors for color-based tasks.
        
        Args:
            workspace_bounds: (x_min, x_max, y_min, y_max) of workspace
            table_height: Height of table surface
            colors_to_use: List of color names to use (random selection if None)
            
        Returns:
            List of created object IDs
        """
        if colors_to_use is None:
            available = list(self.COLORS.keys())
            num_colors = random.randint(2, min(4, len(available)))
            colors_to_use = random.sample(available, num_colors)
        
        x_min, x_max, y_min, y_max = workspace_bounds
        created_ids = []
        
        shapes = [self.SHAPE_BOX, self.SHAPE_SPHERE, self.SHAPE_CYLINDER]
        
        for i, color in enumerate(colors_to_use):
            shape = shapes[i % len(shapes)]
            size = random.uniform(0.025, 0.04)
            
            # Distribute objects across workspace
            num_colors = len(colors_to_use)
            section_width = (x_max - x_min) / num_colors
            
            x = x_min + section_width * (i + 0.5) + random.uniform(-0.03, 0.03)
            y = random.uniform(y_min + 0.05, y_max - 0.05)
            z = table_height + size + 0.005
            
            obj_id = self.create_basic_shape(shape, [x, y, z], color, size)
            created_ids.append(obj_id)
        
        return created_ids
    
    def clear_all_objects(self) -> None:
        """Remove all generated objects from simulation."""
        for obj_id in self.object_ids:
            try:
                p.removeBody(obj_id)
            except Exception:
                pass
        
        self.object_ids.clear()
        self.generated_objects.clear()
    
    def get_object_positions(self) -> Dict[int, np.ndarray]:
        """
        Get current positions of all generated objects.
        
        Returns:
            Dictionary mapping object ID to position array
        """
        positions = {}
        for obj_id in self.object_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obj_id)
                positions[obj_id] = np.array(pos)
            except Exception:
                pass
        return positions
    
    def get_object_info(self, object_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific object.
        
        Args:
            object_id: PyBullet body ID
            
        Returns:
            Object info dictionary or None if not found
        """
        for obj_info in self.generated_objects:
            if obj_info['id'] == object_id:
                return obj_info
        return None
    
    def randomize_object_poses(
        self,
        workspace_bounds: Tuple[float, float, float, float],
        table_height: float
    ) -> None:
        """
        Randomize positions of all existing objects.
        
        Args:
            workspace_bounds: (x_min, x_max, y_min, y_max)
            table_height: Height of table surface
        """
        x_min, x_max, y_min, y_max = workspace_bounds
        positions = []
        
        for obj_id in self.object_ids:
            # Find valid position
            for _ in range(50):
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                
                valid = True
                for px, py in positions:
                    if math.sqrt((x - px) ** 2 + (y - py) ** 2) < 0.1:
                        valid = False
                        break
                
                if valid:
                    break
            
            # Get object info for height calculation
            obj_info = self.get_object_info(obj_id)
            size = obj_info.get('size', 0.03) if obj_info else 0.03
            z = table_height + size + 0.005
            
            # Random orientation
            yaw = random.uniform(0, 2 * math.pi)
            orientation = p.getQuaternionFromEuler([0, 0, yaw])
            
            p.resetBasePositionAndOrientation(obj_id, [x, y, z], orientation)
            p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])
            positions.append((x, y))
    
    def settle_objects(self, num_steps: int = 100) -> None:
        """
        Run simulation steps to let objects settle.
        
        Args:
            num_steps: Number of simulation steps
        """
        for _ in range(num_steps):
            p.stepSimulation()
