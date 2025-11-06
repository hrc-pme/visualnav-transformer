# ROS2 Training Pipeline Architecture

## 系統概覽

本文檔描述新的 ROS2 訓練流程架構，用於 Visual Navigation Transformer (ViNT) 模型訓練。

---

## 架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROS2 System (Other Container)                │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Camera Node  │──►   │  Nav Stack   │──►   │ Robot Control│  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                      │                                │
│    /camera/image_raw        /odom                               │
└─────────┼────────────────────┼─────────────────────────────────┘
          │                    │
          └────────┬───────────┘
                   │ ROS2 Topics
                   ▼
    ┌──────────────────────────────────┐
    │   Step 1: Data Recording         │
    │   dataset_record.py              │
    │   ➜ Records ROS2 bags            │
    └──────────────┬───────────────────┘
                   │
                   ▼
         workspace/train/rosbags/
         └── my_dataset_YYYYMMDD_HHMMSS/
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   Step 2: Format Conversion      │
    │   dataset_train_format.py        │
    │   ➜ Converts to training format  │
    └──────────────┬───────────────────┘
                   │
                   ├─► workspace/train/processed_data/my_dataset/
                   │   └── trajectory_*/
                   │       ├── traj_data.pkl
                   │       └── *.jpg
                   │
                   └─► workspace/train/vint_train/data/data_splits/
                       └── my_dataset/
                           ├── train/traj_names.txt
                           └── test/traj_names.txt
                   │
                   ▼
    ┌──────────────────────────────────┐
    │   Step 3: Model Training         │
    │   train.py                       │
    │   ➜ Trains ViNT model            │
    └──────────────┬───────────────────┘
                   │
                   ▼
         workspace/train/logs/
         └── <project>/<run>/latest.pth
```

---

## 核心組件

### 1. Dataset Recording (`dataset_record.py`)
- **位置**: `workspace/deployment/src/data/dataset_record.py`
- **功能**: 從 ROS2 topics 錄製資料
- **輸出**: `workspace/train/rosbags/`

### 2. Format Conversion (`dataset_train_format.py`)
- **位置**: `workspace/deployment/src/data/dataset_train_format.py`
- **功能**: 轉換 ROS2 bags 為訓練格式
- **輸出**: 
  - `workspace/train/processed_data/`
  - `workspace/train/vint_train/data/data_splits/`

### 3. Model Training (`train.py`)
- **位置**: `workspace/train/train.py`
- **功能**: 訓練 ViNT 模型
- **輸出**: `workspace/train/logs/`

---

## 資料格式

### ROS2 Bag → 處理後格式
```
trajectory_0/
├── traj_data.pkl          # {'position': [[x,y],...], 'yaw': [θ,...]}
├── 0.jpg
├── 1.jpg
└── ...
```

---

## 工作流程

```bash
# 1. 錄製
cd workspace/deployment/src/data
./dataset_record.py

# 2. 轉換
./dataset_train_format.py

# 3. 訓練
cd ../../train
python train.py --config config/vint.yaml
```

詳細說明請參考 README.md
