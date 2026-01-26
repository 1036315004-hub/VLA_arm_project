from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    center: Tuple[int, int]
    area: float
    world_position: Optional[Vector3] = None


@dataclass(frozen=True)
class ScanPose:
    position: Vector3
    description: str
    stage: int


class OcclusionAwareAgent:
    """Plan multi-stage scan poses to reduce arm occlusion for eye-to-hand tasks."""

    def __init__(
        self,
        global_scan_height: float = 0.85,
        stage2_height: float = 0.65,
        refine_height: float = 0.50,
        micro_adjust_height_offset: float = 0.05,
        min_clear_detection_area: float = 2000.0,
        min_good_detection_area: float = 800.0,
        center_offset_threshold: int = 100,
        stage2_x_positions: Optional[Sequence[float]] = None,
        stage2_y_positions: Optional[Sequence[float]] = None,
    ) -> None:
        self.global_scan_height = global_scan_height
        self.stage2_height = stage2_height
        self.refine_height = refine_height
        self.micro_adjust_height_offset = micro_adjust_height_offset
        self.min_clear_detection_area = min_clear_detection_area
        self.min_good_detection_area = min_good_detection_area
        self.center_offset_threshold = center_offset_threshold
        self.stage2_x_positions = list(stage2_x_positions) if stage2_x_positions else [0.55, 0.75]
        self.stage2_y_positions = list(stage2_y_positions) if stage2_y_positions else [0.4, 0.2, 0.0, -0.2, -0.4]

    def _global_scan_pose(self, base_position: Vector3) -> ScanPose:
        return ScanPose(
            position=(base_position[0], base_position[1], self.global_scan_height),
            description="global scan",
            stage=1,
        )

    def _stage2_scan_poses(self) -> Iterable[ScanPose]:
        for x_pos in self.stage2_x_positions:
            for y_pos in self.stage2_y_positions:
                yield ScanPose(
                    position=(x_pos, y_pos, self.stage2_height),
                    description=f"stage2 scan x={x_pos:.2f}, y={y_pos:.2f}",
                    stage=2,
                )

    def _refine_pose(self, world_position: Vector3) -> ScanPose:
        return ScanPose(
            position=(world_position[0], world_position[1], self.refine_height),
            description="refine scan",
            stage=3,
        )

    def _micro_adjust_pose(self, world_position: Vector3) -> ScanPose:
        return ScanPose(
            position=(
                world_position[0],
                world_position[1],
                self.refine_height - self.micro_adjust_height_offset,
            ),
            description="micro adjustment",
            stage=3,
        )

    def _is_center_offset_large(
        self,
        center: Tuple[int, int],
        image_size: Tuple[int, int],
    ) -> bool:
        width, height = image_size
        offset_x = abs(center[0] - width // 2)
        offset_y = abs(center[1] - height // 2)
        return offset_x > self.center_offset_threshold or offset_y > self.center_offset_threshold

    def plan_scan_poses(
        self,
        base_position: Vector3,
        image_size: Tuple[int, int],
        detect: Callable[[ScanPose], Optional[DetectionResult]],
    ) -> Tuple[List[ScanPose], Optional[DetectionResult]]:
        """Run multi-stage scan planning with a detection callback.

        Args:
            base_position: Reference position (x, y, z) for global scan.
            image_size: (width, height) used for center offset checks.
            detect: Callable that executes a scan at the given pose and returns DetectionResult.

        Returns:
            (poses_executed, best_detection)
        """
        poses_executed: List[ScanPose] = []
        best_detection: Optional[DetectionResult] = None

        global_pose = self._global_scan_pose(base_position)
        poses_executed.append(global_pose)
        global_detection = detect(global_pose)

        if global_detection:
            best_detection = global_detection
            if global_detection.area >= self.min_clear_detection_area:
                return poses_executed, best_detection

        for scan_pose in self._stage2_scan_poses():
            poses_executed.append(scan_pose)
            detection = detect(scan_pose)
            if detection is None:
                continue
            if best_detection is None or detection.area > best_detection.area:
                best_detection = detection
            if detection.area >= self.min_good_detection_area:
                break

        if best_detection is None:
            return poses_executed, None

        if best_detection.world_position is None:
            return poses_executed, best_detection

        refine_pose = self._refine_pose(best_detection.world_position)
        poses_executed.append(refine_pose)
        refine_detection = detect(refine_pose)

        if refine_detection:
            best_detection = refine_detection
            if refine_detection.world_position and self._is_center_offset_large(refine_detection.center, image_size):
                micro_pose = self._micro_adjust_pose(refine_detection.world_position)
                poses_executed.append(micro_pose)
                micro_detection = detect(micro_pose)
                if micro_detection:
                    best_detection = micro_detection

        return poses_executed, best_detection


__all__ = ["DetectionResult", "ScanPose", "OcclusionAwareAgent"]
