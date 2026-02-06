#!/usr/bin/env python3
import subprocess
import argparse
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, ".."))

def run_command(command):
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["CURL_CA_BUNDLE"] = ""
    env["REQUESTS_CA_BUNDLE"] = ""
    env["PYTHONHTTPSVERIFY"] = "0"

    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="VLA Data Collection and Preprocessing Pipeline")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect")
    parser.add_argument("--gui", action="store_true", help="Enable GUI for collection")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory for raw data")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory for processed training data")

    args = parser.parse_args()

    # 1. Start automated collection
    print("=== Phase 1: Data Collection ===")
    collect_cmd = [
        sys.executable,
        os.path.join("scripts", "auto_collect.py"),
        "--num_episodes", str(args.num_episodes),
        "--save_dir", args.raw_dir
    ]
    if args.gui:
        collect_cmd.append("--gui")

    run_command(collect_cmd)

    # 2. Start preprocessing
    print("\n=== Phase 2: Preprocessing ===")
    preprocess_cmd = [
        sys.executable,
        os.path.join("scripts", "preprocess_data.py"),
        "--input_dir", args.raw_dir,
        "--output_dir", args.train_dir
    ]

    run_command(preprocess_cmd)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
