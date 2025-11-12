#!/usr/bin/env python3
"""
ROS2 Bag to Training Format Converter

This script converts ROS2 bag files into the training format required by ViNT.
It processes bags from the train/rosbags/ directory and outputs formatted data.
Configure settings in data.yaml before running.
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

# Import numpy for data processing
import numpy as np

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
    sys.exit(1)

# Extract configuration variables
DATASET_NAME = config['dataset']['name']
IMAGE_TOPIC = config['topics']['image']
POSE_TOPIC = config['topics']['pose']
IMAGE_SIZE = tuple(config['image']['size'])
IMAGE_COMPRESSED = config['image']['compressed']

SAMPLE_RATE = config['processing']['sample_rate']
ANGULAR_OFFSET = config['processing']['angular_offset']
TRAIN_TEST_SPLIT = config['processing']['train_test_split']
FILTER_BACKWARDS = config['processing']['filter_backwards']
START_SLACK = config['processing']['start_slack']
END_SLACK = config['processing']['end_slack']

# Directory paths
DATASETS_ROOT_DIR = os.path.join(TRAIN_DIR, "datasets")
# Note: Individual dataset paths will be constructed dynamically when processing all datasets

# ============================================================================
# END CONFIGURATION SECTION
# ============================================================================


def pose_to_xy_yaw(pose_msg, ang_offset=0.0):
    """
    Process ROS2 pose message (PoseStamped or Odometry) to position and yaw.
    
    Args:
        pose_msg: ROS2 PoseStamped or Odometry message
        ang_offset: Angular offset to add (radians)
    
    Returns:
        tuple: ([x, y], yaw)
    """
    import numpy as np
    
    # Handle different message types
    if hasattr(pose_msg, 'pose'):
        # Could be PoseStamped or Odometry
        if hasattr(pose_msg.pose, 'pose'):
            # Odometry message (has pose.pose)
            position = pose_msg.pose.pose.position
            orientation = pose_msg.pose.pose.orientation
        else:
            # PoseStamped message (has pose directly)
            position = pose_msg.pose.position
            orientation = pose_msg.pose.orientation
    else:
        raise ValueError(f"Unknown pose message type: {type(pose_msg)}")
    
    # Convert quaternion to yaw
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4) + ang_offset
    
    return [position.x, position.y], yaw


def nav_to_xy_yaw(odom_msg, ang_offset=0.0):
    """
    Legacy function for backward compatibility.
    Redirects to pose_to_xy_yaw.
    """
    return pose_to_xy_yaw(odom_msg, ang_offset)


def filter_backwards(img_data, traj_data, start_slack=0, end_slack=0):
    """
    Remove backward motion segments from trajectory.
    
    Args:
        img_data: List of PIL images
        traj_data: Dict with 'position' and 'yaw' arrays
        start_slack: Number of points to ignore at start
        end_slack: Number of points to ignore at end
    
    Returns:
        List of (img_data, traj_data) tuples for each forward segment
    """
    import numpy as np
    
    positions = traj_data["position"]
    yaws = traj_data["yaw"]
    
    # Calculate forward motion
    forward_segments = []
    current_segment = []
    
    for i in range(len(positions) - 1):
        # Calculate direction of motion
        dx = positions[i+1][0] - positions[i][0]
        dy = positions[i+1][1] - positions[i][1]
        motion_yaw = np.arctan2(dy, dx)
        
        # Calculate angle difference
        yaw_diff = np.abs(np.arctan2(np.sin(motion_yaw - yaws[i]), 
                                      np.cos(motion_yaw - yaws[i])))
        
        # Check if moving forward (yaw_diff < 90 degrees)
        if yaw_diff < np.pi / 2:
            current_segment.append(i)
        else:
            if len(current_segment) > start_slack + end_slack + 1:
                forward_segments.append(current_segment)
            current_segment = []
    
    # Add last point to current segment
    if len(current_segment) > 0:
        current_segment.append(len(positions) - 1)
        if len(current_segment) > start_slack + end_slack + 1:
            forward_segments.append(current_segment)
    
    # If no valid segments or only one segment, return all data
    if len(forward_segments) == 0:
        return [(img_data, traj_data)]
    
    # Extract segments
    result = []
    for segment in forward_segments:
        if len(segment) <= start_slack + end_slack:
            continue
        
        # Apply slack
        seg_start = segment[start_slack]
        seg_end = segment[-1 - end_slack]
        
        if seg_end <= seg_start:
            continue
        
        seg_imgs = img_data[seg_start:seg_end+1]
        seg_traj = {
            "position": positions[seg_start:seg_end+1],
            "yaw": yaws[seg_start:seg_end+1]
        }
        result.append((seg_imgs, seg_traj))
    
    return result if len(result) > 0 else [(img_data, traj_data)]


def process_ros2_image(msg):
    """
    Process ROS2 image message to PIL Image.
    Handles both sensor_msgs/Image (raw) and sensor_msgs/CompressedImage.
    """
    import numpy as np
    import cv2
    import io
    
    try:
        # For sensor_msgs/CompressedImage
        if hasattr(msg, 'format') or IMAGE_COMPRESSED:
            # Compressed image - decode from JPEG/PNG bytes
            if hasattr(msg, 'data'):
                img = Image.open(io.BytesIO(bytes(msg.data)))
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pil_image = img
            else:
                raise ValueError("CompressedImage message has no data field")
        
        # For sensor_msgs/Image (raw image)
        elif hasattr(msg, 'encoding'):
            if 'bgr8' in msg.encoding or 'rgb8' in msg.encoding:
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, -1)
                if 'bgr8' in msg.encoding:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img)
            else:
                raise ValueError(f"Unsupported image encoding: {msg.encoding}")
        else:
            raise ValueError("Unknown image message type")
        
        # Resize to target size
        pil_image = pil_image.resize(IMAGE_SIZE)
        return pil_image
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise


def nav_to_xy_yaw_duplicate_removal(pose_msg, ang_offset=0.0):
    """Alias for backward compatibility."""
    return pose_to_xy_yaw(pose_msg, ang_offset)


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
        curr_posedata = None
        
        # Create connections for topics
        connections = [x for x in reader.connections if x.topic in [IMAGE_TOPIC, POSE_TOPIC]]
        
        if len(connections) == 0:
            print(f"Warning: No matching topics found in bag. Available topics:")
            for conn in reader.connections:
                print(f"  - {conn.topic}")
            return []
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            t = timestamp / 1e9  # Convert to seconds
            
            if connection.topic == IMAGE_TOPIC:
                curr_imdata = msg
            elif connection.topic == POSE_TOPIC:
                curr_posedata = msg
            
            # Sample at specified rate
            if (t - currtime) >= 1.0 / sample_rate:
                if curr_imdata is not None and curr_posedata is not None:
                    synced_imdata.append(curr_imdata)
                    synced_odomdata.append(curr_posedata)
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
    print("Processing pose data...")
    import numpy as np
    xys = []
    yaws = []
    for pose_msg in synced_odomdata:
        xy, yaw = pose_to_xy_yaw(pose_msg, ANGULAR_OFFSET)
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


def process_single_dataset(dataset_name, dataset_path):
    """
    Process a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset directory
    
    Returns:
        bool: True if processing was successful
    """
    print("\n" + "="*70)
    print(f"Processing Dataset: {dataset_name}")
    print("="*70)
    
    # Define paths for this dataset
    bag_input_dir = os.path.join(dataset_path, "rosbags")
    processed_output_dir = os.path.join(dataset_path, "processed_data", dataset_name)
    data_splits_dir = os.path.join(TRAIN_DIR, "vint_train/data/data_splits", dataset_name)
    
    print(f"  Input (rosbags):  {bag_input_dir}")
    print(f"  Output (processed): {processed_output_dir}")
    print(f"  Splits:           {data_splits_dir}")
    
    # Check if rosbags directory exists
    if not os.path.exists(bag_input_dir):
        print(f"  ⚠ Skipping: No rosbags directory found")
        return False
    
    # Find all ROS2 bag directories
    bag_dirs = []
    for item in os.listdir(bag_input_dir):
        item_path = os.path.join(bag_input_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a ROS2 bag (has metadata.yaml)
            if os.path.exists(os.path.join(item_path, "metadata.yaml")):
                bag_dirs.append(item_path)
    
    if len(bag_dirs) == 0:
        print(f"  ⚠ Skipping: No ROS2 bags found in rosbags directory")
        return False
    
    print(f"  Found {len(bag_dirs)} ROS2 bag(s) to process")
    
    # Create output directory
    os.makedirs(processed_output_dir, exist_ok=True)
    
    # Process each bag
    all_trajs = []
    for bag_path in bag_dirs:
        trajs = process_ros2_bag(bag_path, processed_output_dir, SAMPLE_RATE)
        all_trajs.extend(trajs)
        print()
    
    print(f"  Total trajectories created: {len(all_trajs)}")
    
    # Create train/test split
    if len(all_trajs) > 0:
        print(f"  Creating train/test split...")
        create_train_test_split(processed_output_dir, data_splits_dir, TRAIN_TEST_SPLIT)
        print(f"  ✓ Dataset processed successfully!")
        return True
    else:
        print(f"  ✗ Error: No trajectories were created")
        return False


def main():
    """Main processing function - processes all datasets."""
    parser = argparse.ArgumentParser(description="Convert ROS2 bags to training format")
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="Process only a specific dataset (optional, default: all datasets)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available datasets and exit"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ROS2 Bag to Training Format Converter")
    print("="*70)
    print(f"\nConfiguration (from data.yaml):")
    print(f"  Datasets Root:   {DATASETS_ROOT_DIR}")
    print(f"  Image Topic:     {IMAGE_TOPIC} {'(compressed)' if IMAGE_COMPRESSED else '(raw)'}")
    print(f"  Pose Topic:      {POSE_TOPIC}")
    print(f"  Sample Rate:     {SAMPLE_RATE} Hz")
    print(f"  Image Size:      {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"  Train/Test:      {TRAIN_TEST_SPLIT*100:.0f}% / {(1-TRAIN_TEST_SPLIT)*100:.0f}%")
    print(f"  Filter Backwards: {FILTER_BACKWARDS}")
    print("="*70 + "\n")
    
    # Check if datasets root directory exists
    if not os.path.exists(DATASETS_ROOT_DIR):
        print(f"Error: Datasets root directory not found: {DATASETS_ROOT_DIR}")
        sys.exit(1)
    
    # Find all dataset directories
    all_datasets = []
    for item in os.listdir(DATASETS_ROOT_DIR):
        item_path = os.path.join(DATASETS_ROOT_DIR, item)
        if os.path.isdir(item_path):
            # Check if it has a rosbags subdirectory
            rosbags_path = os.path.join(item_path, "rosbags")
            if os.path.exists(rosbags_path):
                all_datasets.append(item)
    
    all_datasets.sort()  # Sort alphabetically
    
    if len(all_datasets) == 0:
        print(f"Error: No datasets found in {DATASETS_ROOT_DIR}")
        print("Expected structure: datasets/<dataset_name>/rosbags/")
        sys.exit(1)
    
    # Handle --list option
    if args.list:
        print(f"Found {len(all_datasets)} dataset(s):\n")
        for i, dataset_name in enumerate(all_datasets, 1):
            dataset_path = os.path.join(DATASETS_ROOT_DIR, dataset_name)
            rosbags_path = os.path.join(dataset_path, "rosbags")
            
            # Count bags
            bag_count = 0
            if os.path.exists(rosbags_path):
                for item in os.listdir(rosbags_path):
                    item_path = os.path.join(rosbags_path, item)
                    if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "metadata.yaml")):
                        bag_count += 1
            
            # Check if already processed
            processed_path = os.path.join(dataset_path, "processed_data", dataset_name)
            status = "✓ processed" if os.path.exists(processed_path) else "○ not processed"
            
            print(f"  {i}. {dataset_name:30s} [{bag_count} bags] {status}")
        
        print(f"\nTo process all datasets: python dataset_train_format.py")
        print(f"To process specific dataset: python dataset_train_format.py --dataset <name>")
        sys.exit(0)
    
    # Determine which datasets to process
    if args.dataset:
        # Process specific dataset
        if args.dataset not in all_datasets:
            print(f"Error: Dataset '{args.dataset}' not found")
            print(f"\nAvailable datasets: {', '.join(all_datasets)}")
            sys.exit(1)
        datasets_to_process = [args.dataset]
        print(f"Processing specific dataset: {args.dataset}\n")
    else:
        # Process all datasets
        datasets_to_process = all_datasets
        print(f"Found {len(all_datasets)} dataset(s) to process:")
        for dataset_name in all_datasets:
            print(f"  - {dataset_name}")
        print()
    
    # Process datasets
    successful = []
    failed = []
    skipped = []
    
    for dataset_name in datasets_to_process:
        dataset_path = os.path.join(DATASETS_ROOT_DIR, dataset_name)
        
        try:
            result = process_single_dataset(dataset_name, dataset_path)
            if result:
                successful.append(dataset_name)
            else:
                skipped.append(dataset_name)
        except Exception as e:
            print(f"  ✗ Error processing dataset: {e}")
            failed.append(dataset_name)
    
    # Print summary
    print("\n" + "="*70)
    print("Processing Summary")
    print("="*70)
    print(f"  Total datasets: {len(datasets_to_process)}")
    print(f"  ✓ Successful:   {len(successful)}")
    print(f"  ⚠ Skipped:      {len(skipped)}")
    print(f"  ✗ Failed:       {len(failed)}")
    
    if successful:
        print(f"\nSuccessfully processed datasets:")
        for name in successful:
            print(f"  ✓ {name}")
    
    if skipped:
        print(f"\nSkipped datasets (no bags or already processed):")
        for name in skipped:
            print(f"  ⚠ {name}")
    
    if failed:
        print(f"\nFailed datasets:")
        for name in failed:
            print(f"  ✗ {name}")
    
    if successful:
        print("\n" + "="*70)
        print("Next Steps")
        print("="*70)
        print(f"\nUpdate config/vint.yaml with the following dataset paths:")
        print(f"\ndatasets:")
        for name in successful:
            processed_dir = os.path.join(DATASETS_ROOT_DIR, name, "processed_data", name)
            splits_dir = os.path.join(TRAIN_DIR, "vint_train/data/data_splits", name)
            print(f"  {name}:")
            print(f"    data_folder: {processed_dir}")
            print(f"    train: {os.path.join(splits_dir, 'train/')}")
            print(f"    test: {os.path.join(splits_dir, 'test/')}")
        
        print(f"\nThen run: python train.py --config config/vint.yaml")
        print("="*70 + "\n")
    else:
        print("\nNo datasets were successfully processed.")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
