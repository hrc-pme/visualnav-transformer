#!/usr/bin/env python3
"""
ROS2 Dataset Recording Script

This script records ROS2 bag files for training the Visual Navigation Transformer.
Configure settings in data.yaml before running.
"""

import rclpy
from rclpy.node import Node
import subprocess
import os
from datetime import datetime
import yaml

# ============================================================================
# Load Configuration from YAML
# ============================================================================

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "data.yaml")

# Load configuration
try:
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: Configuration file not found at {CONFIG_PATH}")
    print("Please create data.yaml with required settings.")
    exit(1)

# Extract configuration variables
DATASET_NAME = config['dataset']['name']
IMAGE_TOPIC = config['topics']['image']
POSE_TOPIC = config['topics']['pose']
ADDITIONAL_TOPICS = config['topics'].get('additional', [])

COMPRESSION_MODE = config['recording']['compression_mode']
COMPRESSION_FORMAT = config['recording']['compression_format']
STORAGE_TYPE = config['recording']['storage_type']

# ROS2 Bag output directory
BAG_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "../datasets",
    DATASET_NAME,
    "rosbags"
)

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


class DatasetRecorder(Node):
    """Node for recording ROS2 bag files for dataset collection."""
    
    def __init__(self):
        super().__init__('dataset_recorder')
        
        # Create output directory if it doesn't exist
        os.makedirs(BAG_OUTPUT_DIR, exist_ok=True)
        
        # Generate bag file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bag_name = f"{DATASET_NAME}_{timestamp}"
        self.bag_path = os.path.join(BAG_OUTPUT_DIR, self.bag_name)
        
        self.get_logger().info(f"Dataset Recorder initialized")
        self.get_logger().info(f"Bag output path: {self.bag_path}")
        
    def start_recording(self):
        """Start recording ROS2 bag."""
        # Build topic list
        topics = [IMAGE_TOPIC, POSE_TOPIC] + ADDITIONAL_TOPICS
        topics_str = " ".join(topics)
        
        # Build ros2 bag record command
        cmd = [
            "ros2", "bag", "record",
            "-o", self.bag_path,
            "--storage", STORAGE_TYPE,
        ]
        
        if COMPRESSION_MODE != "none":
            cmd.extend(["--compression-mode", COMPRESSION_MODE])
            cmd.extend(["--compression-format", COMPRESSION_FORMAT])
        
        cmd.extend(topics)
        
        self.get_logger().info(f"Starting recording with topics: {topics}")
        self.get_logger().info(f"Command: {' '.join(cmd)}")
        self.get_logger().info("Press Ctrl+C to stop recording")
        
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            self.get_logger().info("Recording stopped by user")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Recording failed: {e}")


def main(args=None):
    """Main function to run the dataset recorder."""
    rclpy.init(args=args)
    
    recorder = DatasetRecorder()
    
    print("\n" + "="*70)
    print("ROS2 Dataset Recorder for Visual Navigation Transformer")
    print("="*70)
    print(f"\nConfiguration (from data.yaml):")
    print(f"  Image Topic:    {IMAGE_TOPIC}")
    print(f"  Pose Topic:     {POSE_TOPIC}")
    print(f"  Additional:     {ADDITIONAL_TOPICS if ADDITIONAL_TOPICS else 'None'}")
    print(f"  Output Dir:     {BAG_OUTPUT_DIR}")
    print(f"  Dataset Name:   {DATASET_NAME}")
    print(f"  Compression:    {COMPRESSION_MODE} ({COMPRESSION_FORMAT})")
    print(f"  Storage:        {STORAGE_TYPE}")
    print("="*70 + "\n")
    
    try:
        recorder.start_recording()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
