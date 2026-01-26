#!/usr/bin/env python3
"""
Data Recorder Demo Script

This script demonstrates the data recorder module capabilities:
- Eye-to-hand camera with randomizable position
- Automatic object generation using PyBullet built-in resources
- Random object pose placement
- Random robot arm initial state
- Multi-stage pose planning to avoid camera occlusion

Usage:
    python scripts/run_data_recorder.py [--no-gui] [--iterations N]
"""

import os
import sys
import argparse
import time

# Adjust path to project root
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set environment variable before importing other modules
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def log(message: str):
    """Print log message with prefix."""
    print(f"[DataRecorderDemo] {message}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Recorder Demo for VLA Arm Project"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI (headless mode)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of randomization iterations (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between iterations in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=None,
        help="Number of objects to generate (random if not specified)"
    )
    return parser.parse_args()


def main():
    """Main function to run the data recorder demo."""
    args = parse_args()
    use_gui = not args.no_gui
    
    log("=" * 60)
    log("  VLA Arm Project - Data Recorder Demo")
    log("=" * 60)
    log(f"Project Root: {PROJECT_ROOT}")
    log(f"GUI Mode: {'Enabled' if use_gui else 'Disabled'}")
    log(f"Iterations: {args.iterations}")
    log("")
    
    # Import after path setup
    try:
        from src.data_recorder import create_data_recorder, DataRecorderConfig
        log("Successfully imported data_recorder module")
    except ImportError as e:
        log(f"Failed to import data_recorder: {e}")
        log("Make sure all dependencies are installed.")
        sys.exit(1)
    
    # Create recorder with custom config if needed
    log("Creating data recorder...")
    recorder = create_data_recorder(PROJECT_ROOT, use_gui=use_gui)
    
    # Optionally customize config
    if args.num_objects is not None:
        recorder.config.min_objects = args.num_objects
        recorder.config.max_objects = args.num_objects
    
    # Connect and setup
    log("Connecting to PyBullet...")
    if not recorder.connect():
        log("Failed to connect to PyBullet!")
        sys.exit(1)
    
    log("Setting up environment...")
    if not recorder.setup_environment():
        log("Failed to setup environment!")
        recorder.disconnect()
        sys.exit(1)
    
    log("Environment setup complete!")
    log("")
    
    # Run demo
    try:
        log("Starting randomization demo...")
        log("=" * 60)
        
        for iteration in range(args.iterations):
            log(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")
            
            # Randomize camera position
            camera_pos = recorder.randomize_camera_position()
            log(f"Camera position: [{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]")
            
            # Generate random objects
            object_ids = recorder.randomize_objects()
            log(f"Generated {len(object_ids)} objects")
            
            # Get object info
            if recorder.object_generator:
                for obj_id in object_ids:
                    obj_info = recorder.object_generator.get_object_info(obj_id)
                    if obj_info:
                        pos = obj_info.get('position', [0, 0, 0])
                        color = obj_info.get('color', 'unknown')
                        shape = obj_info.get('shape', 'unknown')
                        log(f"  Object {obj_id}: {color} {shape} at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            # Randomize robot pose
            joint_angles = recorder.randomize_robot_pose()
            log(f"Robot joint angles: {[f'{a:.2f}' for a in joint_angles]}")
            
            # Move arm to safe observation pose (doesn't block camera)
            log("Moving arm to observation pose (avoiding camera occlusion)...")
            recorder.move_arm_to_observation_pose()
            
            # Let simulation settle
            for _ in range(100):
                import pybullet as p
                p.stepSimulation()
                if use_gui:
                    time.sleep(1.0 / 240.0)
            
            # Capture observation
            obs = recorder.capture_observation()
            log(f"Captured RGB image: {obs['rgb'].shape}")
            log(f"Captured depth buffer: {obs['depth'].shape}")
            
            # Demonstrate grasp planning for first object
            if object_ids:
                target_id = object_ids[0]
                grasp_poses = recorder.plan_grasp_for_object(target_id)
                if grasp_poses:
                    log(f"Planned grasp sequence for object {target_id}:")
                    for stage, pose in grasp_poses.items():
                        log(f"  {stage}: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
            
            # Wait before next iteration
            if use_gui and iteration < args.iterations - 1:
                log(f"Waiting {args.delay}s before next iteration...")
                time.sleep(args.delay)
        
        log("\n" + "=" * 60)
        log("Demo complete!")
        log("=" * 60)
        
        # Keep window open if using GUI
        if use_gui:
            log("\nPress Ctrl+C to exit...")
            try:
                while True:
                    import pybullet as p
                    p.stepSimulation()
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                log("Exiting...")
    
    except KeyboardInterrupt:
        log("\nInterrupted by user")
    except Exception as e:
        log(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log("Disconnecting from PyBullet...")
        recorder.disconnect()
        log("Done.")


if __name__ == "__main__":
    main()
