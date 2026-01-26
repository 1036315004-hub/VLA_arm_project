import os
import sys

# Adjust path to Project Root
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent.occlusion_agent import DetectionResult, OcclusionAwareAgent


def demo_agent():
    agent = OcclusionAwareAgent()
    image_size = (640, 480)
    base_position = (0.55, 0.0, 0.8)

    def detect(pose):
        # Placeholder callback for demonstration/testing.
        # Replace with real perception pipeline in integration.
        return None

    poses, detection = agent.plan_scan_poses(base_position, image_size, detect)
    print("Planned scan poses:")
    for pose in poses:
        print(f"Stage {pose.stage}: {pose.description} -> {pose.position}")
    print("Detection:", detection)


if __name__ == "__main__":
    demo_agent()
