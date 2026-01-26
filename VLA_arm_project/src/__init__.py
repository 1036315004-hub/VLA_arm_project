# VLA Arm Project - Source Package
"""
VLA Arm Project source modules.

Modules:
- data_recorder: Data recorder with Eye-to-hand camera and randomization
- objects: PyBullet object generation
- robot: Robot control and pose planning
- perception: Vision and detection modules
- utils: Utility functions
"""

from .data_recorder import DataRecorder, DataRecorderConfig, EyeToHandCamera, create_data_recorder

__all__ = [
    'DataRecorder',
    'DataRecorderConfig', 
    'EyeToHandCamera',
    'create_data_recorder'
]
