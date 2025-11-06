# Dataset Recording and Processing Tools

This directory contains tools for recording and processing ROS2 bag files for Visual Navigation Transformer (ViNT) training.

## Files

### Configuration
- **`data.yaml`** - Central configuration file for both recording and processing
  - Dataset settings (name, description)
  - ROS2 topic names
  - Image settings (size, compression)
  - Recording settings (compression, storage)
  - Processing settings (sample rate, filtering)

### Scripts
- **`dataset_record.py`** - Records ROS2 bag files
- **`dataset_train_format.py`** - Converts ROS2 bags to training format

## Quick Start

### 1. Configure Settings

Edit `data.yaml` to match your robot's setup:

```yaml
dataset:
  name: "my_dataset"

topics:
  image: "/camera/camera/color/image_raw/compressed"  # Your image topic
  pose: "/relative_pose_stamped"                       # Your pose topic
  
image:
  size: [160, 120]
  compressed: true  # Set to true if using CompressedImage
```

### 2. Record Dataset

```bash
cd /workspace/train/data
python3 dataset_record.py
```

Press Ctrl+C to stop recording. Bags will be saved to `train/datasets/{dataset_name}/rosbags/`

### 3. Process Dataset

```bash
python3 dataset_train_format.py
```

This will:
- Extract images and poses from bag files
- Filter out backward motion (optional)
- Create train/test splits
- Save to `train/datasets/{dataset_name}/processed_data/`

## Supported Message Types

### Images
- **Raw images**: `sensor_msgs/Image` (bgr8, rgb8)
- **Compressed images**: `sensor_msgs/CompressedImage` (JPEG, PNG)

### Pose
- **Odometry**: `nav_msgs/Odometry`
- **Pose**: `geometry_msgs/PoseStamped`

## Directory Structure

```
train/
  datasets/
    {dataset_name}/
      rosbags/              # Recorded ROS2 bags
      processed_data/       # Processed trajectories
        {dataset_name}/
          trajectory_0/
            traj_data.pkl
            0.jpg
            1.jpg
            ...
  vint_train/
    data/
      data_splits/
        {dataset_name}/
          train/
            traj_names.txt
          test/
            traj_names.txt
```

## Configuration Options

### Recording Settings
- `compression_mode`: "none", "file", "message"
- `compression_format`: "zstd", "fake_comp"
- `storage_type`: "sqlite3", "mcap"

### Processing Settings
- `sample_rate`: Sampling frequency in Hz
- `filter_backwards`: Remove backward motion segments
- `start_slack`: Ignore points at trajectory start
- `end_slack`: Ignore points at trajectory end (collision buffer)
- `train_test_split`: Ratio for train/test split (0.8 = 80% train)

## Troubleshooting

### Topic not found
Run `ros2 topic list` to see available topics and update `data.yaml`

### Compressed image error
Make sure `image.compressed` is set to `true` in `data.yaml` if using CompressedImage topics

### No trajectories created
- Check that topics are publishing data during recording
- Verify topic names match exactly (including leading `/`)
- Check bag file with `ros2 bag info <bag_path>`
