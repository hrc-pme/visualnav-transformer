# ViNT Training Pipeline - Complete Guide

## 🎯 Overview

This workspace now has a **fully automated multi-dataset training pipeline** that:
- ✅ Automatically discovers and uses ALL datasets in `train/datasets/`
- ✅ Unified configuration via `data.yaml` for recording and processing
- ✅ Supports compressed images (`sensor_msgs/CompressedImage`)
- ✅ Handles both `PoseStamped` and `Odometry` messages
- ✅ Auto-registers new datasets in training configuration

---

## 📁 Directory Structure

```
train/
├── data/
│   ├── data.yaml                    # 📝 Main configuration file
│   ├── dataset_record.py            # 🎥 Record ROS2 bags
│   ├── dataset_train_format.py      # 🔄 Convert bags to training format
│   └── copy_config_to_dataset.py    # 🔧 Copy config to datasets
│
├── datasets/                        # 📦 All your datasets
│   ├── dataset_name_1/
│   │   ├── data.yaml               # Dataset-specific config (optional)
│   │   ├── rosbags/                # Recorded ROS2 bags
│   │   └── processed_data/         # Processed trajectories
│   └── dataset_name_2/
│       └── ...
│
├── config/
│   └── vint.yaml                    # 🎓 Training configuration
│
└── train.py                         # 🚀 Training script
```

---

## 🚀 Quick Start Workflow

### Step 1: Configure Your Robot Setup

Edit `/workspace/train/data/data.yaml`:

```yaml
dataset:
  name: "my_new_dataset"      # Change this for each new dataset

topics:
  image: "/camera/camera/color/image_raw/compressed"  # Your camera topic
  pose: "/relative_pose_stamped"                       # Your pose topic

image:
  size: [160, 120]            # Target size
  compressed: true            # true for CompressedImage

training:
  metric_waypoint_spacing: 0.25  # Adjust based on robot speed (meters)
```

### Step 2: Record Dataset

```bash
cd /workspace/train/data
python3 dataset_record.py
```

- Press **Ctrl+C** to stop recording
- Bags saved to: `datasets/{dataset_name}/rosbags/`

### Step 3: Process Dataset

```bash
python3 dataset_train_format.py
```

This will:
- Extract images and poses
- Filter backward motion
- Create train/test splits
- Save to: `datasets/{dataset_name}/processed_data/`

### Step 4: Train Model

```bash
cd /workspace/train
python3 train.py
```

**The training script will automatically:**
- 🔍 Scan all datasets in `train/datasets/`
- ✅ Validate each dataset
- 🔧 Auto-register them in `data_config.yaml`
- 🎯 Train on ALL datasets simultaneously

---

## ⚙️ Configuration Details

### ROS2 Topics Supported

**Images:**
- `sensor_msgs/Image` (raw: bgr8, rgb8)
- `sensor_msgs/CompressedImage` (JPEG, PNG) ✅

**Pose:**
- `nav_msgs/Odometry` ✅
- `geometry_msgs/PoseStamped` ✅

### Current Configuration

Based on your `data.yaml`:
```yaml
Image Topic:  /camera/camera/color/image_raw/compressed
Pose Topic:   /relative_pose_stamped
Compression:  file (zstd)
Image Size:   160x120 → resized to 85x64 for training
Sample Rate:  4.0 Hz
```

---

## 📊 Dataset Requirements

For successful training, each dataset should have:

### Minimum Data
- **At least 3-5 trajectories** (more is better)
- **50+ waypoints per trajectory**
- **Varied environments** and lighting conditions

### Data Quality
- ✅ Smooth motion (avoid jerky movements)
- ✅ Forward-facing navigation
- ✅ Consistent camera mounting
- ✅ Good lighting conditions

### Tips
- **Record multiple runs** of the same path from different angles
- **Include turns** and straight segments
- **Avoid obstacles** in training data (unless intentional)
- **End slack = 3** removes last 3 waypoints (collision buffer)

---

## 🎓 Training Configuration

Edit `/workspace/train/config/vint.yaml`:

### Key Settings

```yaml
training_name: "vint_baseline"    # Output model name
use_wandb: false                  # Enable W&B logging
batch_size: 16                    # Reduce if GPU memory limited
epochs: 30                        # Training epochs
context_size: 2                   # Number of context frames

# Dataset parameters (applied to all auto-discovered datasets)
default_dataset_params:
  end_slack: 0                    # Waypoints to ignore at end
  goals_per_obs: 3                # Goals per observation (data aug)
  negative_mining: true           # ViNG negative mining
```

---

## 🔧 Advanced Usage

### Per-Dataset Configuration

Each dataset can have its own `data.yaml`:

```bash
cd /workspace/train/data
python3 copy_config_to_dataset.py
```

Then edit `datasets/{name}/data.yaml` to customize:
- `metric_waypoint_spacing` - Based on robot speed
- Image processing parameters
- Filtering settings

### Manual Dataset Registration

If auto-registration fails, manually add to `vint_train/data/data_config.yaml`:

```yaml
my_dataset:
  metric_waypoint_spacing: 0.25
```

### WandB Integration

To enable Weights & Biases logging:

```yaml
# In config/vint.yaml
use_wandb: true
wandb_entity: "your-username"  # or team name
```

---

## 🐛 Troubleshooting

### "No valid datasets found"
- Check directory structure: `datasets/{name}/processed_data/{name}/`
- Verify splits exist: `vint_train/data/data_splits/{name}/train/traj_names.txt`
- Run `dataset_train_format.py` to process bags

### "Dataset not found in data_config.yaml"
- Should auto-register on first training run
- Check `vint_train/data/data_config.yaml` was updated
- Manually add if needed

### "num_samples=0" Error
- Dataset too small (< 10 waypoints)
- Record more data or increase trajectory length
- Check train/test split isn't filtering all data

### "CompressedImage" errors
- Set `image.compressed: true` in `data.yaml`
- Verify topic type: `ros2 topic info /your/topic`

### "Permission denied" (WandB)
- Set `use_wandb: false` in `vint.yaml`
- Or update `wandb_entity` to your account

---

## 📈 Training Progress

Models are saved to: `/workspace/model/{training_name}_{timestamp}/`

```
model/
└── vint_baseline_2025_11_07_001234/
    ├── latest.pth              # Latest checkpoint
    ├── best.pth                # Best validation loss
    └── config.yaml             # Training configuration
```

---

## 🎯 Next Steps After Training

1. **Evaluate Model**: Check validation metrics
2. **Test on Robot**: Deploy to `/workspace/deployment/`
3. **Collect More Data**: Improve model with diverse scenarios
4. **Fine-tune**: Use pre-trained model as starting point

---

## 📚 Key Files Summary

| File | Purpose | When to Edit |
|------|---------|--------------|
| `data/data.yaml` | Recording/processing config | Before recording each dataset |
| `config/vint.yaml` | Training config | Before training |
| `train.py` | Training script | Usually don't need to edit |
| `vint_train/data/data_config.yaml` | Dataset registry | Auto-updated |

---

## ✅ Your Current Setup

Based on terminal output:

- ✅ 2 datasets found: `testing`, `testing_22`
- ⚠️  Small datasets (1 trajectory each, ~13 frames)
- 💡 **Recommendation**: Record longer trajectories (50+ waypoints)

### To improve:

```bash
# 1. Update data.yaml with longer recording
cd /workspace/train/data
nano data.yaml  # Change dataset name

# 2. Record longer trajectory
python3 dataset_record.py
# Drive robot for at least 1-2 minutes

# 3. Process
python3 dataset_train_format.py

# 4. Repeat 2-3 times for variety

# 5. Train
cd /workspace/train
python3 train.py
```

---

**Happy Training! 🚀**
