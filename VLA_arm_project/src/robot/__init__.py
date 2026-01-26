# src/robot/__init__.py
"""
Robot control and pose planning modules.

Modules:
- multi_stage_planner: Multi-stage pose planning for occlusion avoidance
- sim_adapter: PyBullet simulation adapter
- coords_bridge: Coordinate transformation utilities
"""

from .multi_stage_planner import MultiStagePlanner
from .sim_adapter import SimulationAdapter

__all__ = [
    'MultiStagePlanner',
    'SimulationAdapter'
]
