#!/usr/bin/env python3
"""
ROS2 Dataset Recording Script

This script records ROS2 bag files for training the Visual Navigation Transformer.
Configure the topics and bag file settings below before running.
"""

import rclpy
from rclpy.node import Node
import subprocess
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION SECTION - Modify these variables as needed
# ============================================================================

# ROS2 Topics to record
IMAGE_TOPIC = "/camera/image_raw"           # Image topic (sensor_msgs/Image or sensor_msgs/CompressedImage)
ODOM_TOPIC = "/odom"                        # Odometry topic (nav_msgs/Odometry)

# Additional topics (optional)
ADDITIONAL_TOPICS = [
    # "/imu/data",
    # "/cmd_vel",
]

# ROS2 Bag output directory
# Bags will be saved in workspace/train/rosbags/
BAG_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../rosbags"
)

# Dataset name (used for bag file naming)
DATASET_NAME = "my_dataset"

# Recording settings
COMPRESSION_MODE = "zstd"  # Options: "none", "zstd", "lz4"
STORAGE_TYPE = "sqlite3"   # Options: "sqlite3", "mcap"

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
        topics = [IMAGE_TOPIC, ODOM_TOPIC] + ADDITIONAL_TOPICS
        topics_str = " ".join(topics)
        
        # Build ros2 bag record command
        cmd = [
            "ros2", "bag", "record",
            "-o", self.bag_path,
            "--storage", STORAGE_TYPE,
        ]
        
        if COMPRESSION_MODE != "none":
            cmd.extend(["--compression-mode", COMPRESSION_MODE])
        
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
    print(f"\nConfiguration:")
    print(f"  Image Topic:    {IMAGE_TOPIC}")
    print(f"  Odometry Topic: {ODOM_TOPIC}")
    print(f"  Additional:     {ADDITIONAL_TOPICS if ADDITIONAL_TOPICS else 'None'}")
    print(f"  Output Dir:     {BAG_OUTPUT_DIR}")
    print(f"  Dataset Name:   {DATASET_NAME}")
    print(f"  Compression:    {COMPRESSION_MODE}")
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
