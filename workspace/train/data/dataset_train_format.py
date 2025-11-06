#!/usr/bin/env python3
"""
ROS2 Bag to Training Format Converter

This script converts ROS2 bag files into the training format required by ViNT.
It processes bags from the train/rosbags/ directory and outputs formatted data.
"""

import os
import sys
import pickle
from PIL import Image
import argparse
import yaml
from tqdm import tqdm
import random
import shutil

# Add train path to import vint_train modules
TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, TRAIN_DIR)

from vint_train.process_data.process_data_utils import (
    get_images_and_odom,
    filter_backwards,
)

# ============================================================================
# CONFIGURATION SECTION - Modify these variables as needed
# ============================================================================

# Dataset Configuration
DATASET_NAME = "my_dataset"          # Name of your dataset

# ROS2 Bag Settings
BAG_INPUT_DIR = os.path.join(TRAIN_DIR, "rosbags")  # Input directory containing ROS2 bags
PROCESSED_OUTPUT_DIR = os.path.join(TRAIN_DIR, "processed_data", DATASET_NAME)  # Output for processed trajectories
DATA_SPLITS_DIR = os.path.join(TRAIN_DIR, "vint_train/data/data_splits", DATASET_NAME)  # Train/test split output

# Topic Configuration (must match your ROS2 bag topics)
IMAGE_TOPIC = "/camera/image_raw"    # ROS2 image topic
ODOM_TOPIC = "/odom"                  # ROS2 odometry topic

# Processing Settings
SAMPLE_RATE = 4.0                     # Sampling rate in Hz
ANGULAR_OFFSET = 0.0                  # Angular offset to add to odometry (in radians)
TRAIN_TEST_SPLIT = 0.8                # Train/test split ratio (0.8 = 80% train, 20% test)

# Image Processing
IMAGE_SIZE = (160, 120)               # Target image size (width, height)
IMAGE_PROCESS_FUNC = "process_ros2_image"     # Image processing function name
ODOM_PROCESS_FUNC = "nav_to_xy_yaw"           # Odometry processing function name

# Filter Settings
FILTER_BACKWARDS = True               # Remove backward motion segments
START_SLACK = 0                       # Points to ignore at trajectory start
END_SLACK = 3                         # Points to ignore at trajectory end (collision buffer)

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


def process_ros2_image(msg):
    """
    Process ROS2 image message to PIL Image.
    Modify this function based on your image message type and encoding.
    """
    import numpy as np
    import cv2
    
    # For sensor_msgs/Image (raw image)
    if hasattr(msg, 'encoding'):
        if 'bgr8' in msg.encoding or 'rgb8' in msg.encoding:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            if 'bgr8' in msg.encoding:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    # For sensor_msgs/CompressedImage
    elif hasattr(msg, 'format'):
        import io
        img = Image.open(io.BytesIO(msg.data))
        pil_image = img
    else:
        raise ValueError("Unknown image message type")
    
    # Resize to target size
    pil_image = pil_image.resize(IMAGE_SIZE)
    return pil_image


def nav_to_xy_yaw(odom_msg, ang_offset=0.0):
    """
    Process ROS2 odometry message to position and yaw.
    """
    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    
    # Convert quaternion to yaw
    import numpy as np
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4) + ang_offset
    
    return [position.x, position.y], yaw


def process_ros2_bag(bag_path, output_dir, sample_rate):
    """
    Process a single ROS2 bag file.
    
    Args:
        bag_path: Path to the ROS2 bag directory
        output_dir: Output directory for processed data
        sample_rate: Sampling rate in Hz
    """
    try:
        # Import ROS2 bag reader
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        print("Error: rosbags library not found. Install with: pip install rosbags")
        sys.exit(1)
    
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    bag_name = os.path.basename(bag_path)
    print(f"Processing bag: {bag_name}")
    
    synced_imdata = []
    synced_odomdata = []
    
    with Reader(bag_path) as reader:
        # Get bag start time
        start_time = reader.start_time / 1e9  # Convert from nanoseconds
        currtime = start_time
        
        curr_imdata = None
        curr_odomdata = None
        
        # Create connections for topics
        connections = [x for x in reader.connections if x.topic in [IMAGE_TOPIC, ODOM_TOPIC]]
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            t = timestamp / 1e9  # Convert to seconds
            
            if connection.topic == IMAGE_TOPIC:
                curr_imdata = msg
            elif connection.topic == ODOM_TOPIC:
                curr_odomdata = msg
            
            # Sample at specified rate
            if (t - currtime) >= 1.0 / sample_rate:
                if curr_imdata is not None and curr_odomdata is not None:
                    synced_imdata.append(curr_imdata)
                    synced_odomdata.append(curr_odomdata)
                    currtime = t
    
    if len(synced_imdata) == 0:
        print(f"Warning: No synchronized data found in {bag_name}")
        return []
    
    # Process images
    print(f"Processing {len(synced_imdata)} images...")
    img_data = []
    for img_msg in tqdm(synced_imdata, desc="Images"):
        img = process_ros2_image(img_msg)
        img_data.append(img)
    
    # Process odometry
    print("Processing odometry data...")
    import numpy as np
    xys = []
    yaws = []
    for odom_msg in synced_odomdata:
        xy, yaw = nav_to_xy_yaw(odom_msg, ANGULAR_OFFSET)
        xys.append(xy)
        yaws.append(yaw)
    traj_data = {"position": np.array(xys), "yaw": np.array(yaws)}
    
    # Filter backwards motion
    if FILTER_BACKWARDS:
        print("Filtering backward motion...")
        cut_trajs = filter_backwards(img_data, traj_data, START_SLACK, END_SLACK)
    else:
        cut_trajs = [(img_data, traj_data)]
    
    # Save trajectories
    saved_trajs = []
    for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
        traj_name = f"{bag_name}_{i}"
        traj_folder = os.path.join(output_dir, traj_name)
        os.makedirs(traj_folder, exist_ok=True)
        
        # Save trajectory data
        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data_i, f)
        
        # Save images
        for j, img in enumerate(img_data_i):
            img.save(os.path.join(traj_folder, f"{j}.jpg"))
        
        saved_trajs.append(traj_name)
        print(f"Saved trajectory: {traj_name} ({len(img_data_i)} frames)")
    
    return saved_trajs


def create_train_test_split(processed_dir, splits_dir, split_ratio):
    """
    Create train/test split from processed trajectories.
    
    Args:
        processed_dir: Directory containing processed trajectories
        splits_dir: Output directory for split files
        split_ratio: Ratio for train/test split (e.g., 0.8 for 80% train)
    """
    # Get all trajectory folders
    traj_names = [
        f for f in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, f))
        and "traj_data.pkl" in os.listdir(os.path.join(processed_dir, f))
    ]
    
    if len(traj_names) == 0:
        print("Error: No trajectories found!")
        return
    
    # Shuffle and split
    random.shuffle(traj_names)
    split_idx = int(split_ratio * len(traj_names))
    train_names = traj_names[:split_idx]
    test_names = traj_names[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Total trajectories: {len(traj_names)}")
    print(f"  Train: {len(train_names)} ({len(train_names)/len(traj_names)*100:.1f}%)")
    print(f"  Test:  {len(test_names)} ({len(test_names)/len(traj_names)*100:.1f}%)")
    
    # Create split directories
    train_dir = os.path.join(splits_dir, "train")
    test_dir = os.path.join(splits_dir, "test")
    
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    # Write split files
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for name in train_names:
            f.write(name + "\n")
    
    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for name in test_names:
            f.write(name + "\n")
    
    print(f"\nSplit files created in: {splits_dir}")


def main():
    """Main processing function."""
    print("\n" + "="*70)
    print("ROS2 Bag to Training Format Converter")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dataset Name:    {DATASET_NAME}")
    print(f"  Input Dir:       {BAG_INPUT_DIR}")
    print(f"  Output Dir:      {PROCESSED_OUTPUT_DIR}")
    print(f"  Splits Dir:      {DATA_SPLITS_DIR}")
    print(f"  Image Topic:     {IMAGE_TOPIC}")
    print(f"  Odom Topic:      {ODOM_TOPIC}")
    print(f"  Sample Rate:     {SAMPLE_RATE} Hz")
    print(f"  Train/Test:      {TRAIN_TEST_SPLIT*100:.0f}% / {(1-TRAIN_TEST_SPLIT)*100:.0f}%")
    print("="*70 + "\n")
    
    # Check if input directory exists
    if not os.path.exists(BAG_INPUT_DIR):
        print(f"Error: Input directory not found: {BAG_INPUT_DIR}")
        print("Please record some bags first using dataset_record.py")
        sys.exit(1)
    
    # Find all ROS2 bag directories
    bag_dirs = []
    for item in os.listdir(BAG_INPUT_DIR):
        item_path = os.path.join(BAG_INPUT_DIR, item)
        if os.path.isdir(item_path):
            # Check if it's a ROS2 bag (has metadata.yaml)
            if os.path.exists(os.path.join(item_path, "metadata.yaml")):
                bag_dirs.append(item_path)
    
    if len(bag_dirs) == 0:
        print(f"Error: No ROS2 bags found in {BAG_INPUT_DIR}")
        sys.exit(1)
    
    print(f"Found {len(bag_dirs)} ROS2 bag(s) to process\n")
    
    # Create output directory
    os.makedirs(PROCESSED_OUTPUT_DIR, exist_ok=True)
    
    # Process each bag
    all_trajs = []
    for bag_path in bag_dirs:
        trajs = process_ros2_bag(bag_path, PROCESSED_OUTPUT_DIR, SAMPLE_RATE)
        all_trajs.extend(trajs)
        print()
    
    print(f"Total trajectories created: {len(all_trajs)}\n")
    
    # Create train/test split
    if len(all_trajs) > 0:
        print("Creating train/test split...")
        create_train_test_split(PROCESSED_OUTPUT_DIR, DATA_SPLITS_DIR, TRAIN_TEST_SPLIT)
        
        print("\n" + "="*70)
        print("Processing complete!")
        print("="*70)
        print(f"\nProcessed data location:  {PROCESSED_OUTPUT_DIR}")
        print(f"Split files location:     {DATA_SPLITS_DIR}")
        print(f"\nNext step: Update config/vint.yaml with the following paths:")
        print(f"  datasets:")
        print(f"    {DATASET_NAME}:")
        print(f"      data_folder: {PROCESSED_OUTPUT_DIR}")
        print(f"      train: {os.path.join(DATA_SPLITS_DIR, 'train/')}")
        print(f"      test: {os.path.join(DATA_SPLITS_DIR, 'test/')}")
        print(f"\nThen run: python train.py --config config/vint.yaml")
        print("="*70 + "\n")
    else:
        print("Error: No trajectories were created. Check your bag files and topic configuration.")


if __name__ == "__main__":
    main()
