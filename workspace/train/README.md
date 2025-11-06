# ViNT Training Guide (ROS2)

本指南說明如何在 ROS2 系統中進行資料收集、處理和模型訓練。

---

## 系統架構

### 架構概述
- **ROS2 系統**: 使用 ROS2 進行資料收集和機器人控制
- **資料錄製**: 主程式來自其他 container，透過 ROS2 topics 進行錄製
- **資料處理**: 將 ROS2 bag 檔案轉換為訓練格式
- **模型訓練**: 使用處理後的資料訓練 Visual Navigation Transformer

### 資料夾結構
```
workspace/
├── deploy/                            # 部署相關檔案
│   ├── config/
│   │   ├── models.yaml               # 模型配置
│   │   ├── robot.yaml                # 機器人配置
│   │   └── waypoint_visualization.yaml
│   ├── src/
│   │   ├── navigate.ros2.py          # 導航主程式
│   │   ├── waypoint_to_goal_pose.py  # 航點轉換
│   │   ├── explore.py                # 探索模式
│   │   └── utils.py                  # 工具函數
│   └── ...
├── train/                             # 訓練相關檔案
│   ├── data/
│   │   ├── dataset_record.py         # ROS2 bag 錄製腳本
│   │   └── dataset_train_format.py   # 資料格式轉換腳本
│   ├── rosbags/                      # ROS2 bag 儲存位置
│   ├── processed_data/               # 處理後的訓練資料
│   ├── vint_train/                   # 訓練套件
│   │   └── data/data_splits/         # 訓練/測試集分割
│   ├── config/                       # 訓練配置檔案
│   ├── legacy/                       # 舊版腳本 (ROS1)
│   │   ├── process_bags.py
│   │   ├── data_split.py
│   │   └── ...
│   ├── train.py                      # 訓練腳本
│   └── README.md                     # 本文件
└── model/                             # 模型權重
    ├── vint.pth                      # ViNT 模型
    ├── gnm.pth                       # GNM 模型
    ├── nomad.pth                     # NoMaD 模型
    └── README.md
```

---

## 完整訓練流程

### Step 1: 資料錄製

#### 1.1 設定錄製參數
編輯 `workspace/train/data/dataset_record.py`，設定以下變數：

```python
# ROS2 Topics to record
IMAGE_TOPIC = "/camera/image_raw"      # 影像 topic
ODOM_TOPIC = "/odom"                   # 里程計 topic

# Dataset name
DATASET_NAME = "my_dataset"            # 資料集名稱

# Additional topics (可選)
ADDITIONAL_TOPICS = [
    # "/imu/data",
    # "/cmd_vel",
]
```

**重要設定：**
- `IMAGE_TOPIC`: 機器人相機的影像 topic（必須是 sensor_msgs/Image 或 CompressedImage）
- `ODOM_TOPIC`: 機器人里程計 topic（必須是 nav_msgs/Odometry）
- `DATASET_NAME`: 為您的資料集命名，建議使用有意義的名稱
- `BAG_OUTPUT_DIR`: 自動設定為 `workspace/train/rosbags/`

#### 1.2 開始錄製
確保主程式 container 正在運行並發布所需的 topics，然後執行：

```bash
cd workspace/train/data
python3 dataset_record.py
```

錄製過程中：
- 腳本會顯示當前設定和錄製資訊
- 按 `Ctrl+C` 停止錄製
- ROS2 bag 會自動儲存到 `workspace/train/rosbags/` 資料夾
- 檔名格式：`<DATASET_NAME>_<timestamp>`

#### 1.3 驗證錄製
檢查錄製的 bag 檔案：

```bash
cd workspace/train/rosbags
ros2 bag info <your_bag_name>
```

---

### Step 2: 資料格式轉換

#### 2.1 設定轉換參數
編輯 `workspace/train/data/dataset_train_format.py`，設定以下變數：

```python
# Dataset Configuration
DATASET_NAME = "my_dataset"           # 必須與錄製時相同

# Topic Configuration（必須與 bag 中的 topics 一致）
IMAGE_TOPIC = "/camera/image_raw"     # 影像 topic
ODOM_TOPIC = "/odom"                  # 里程計 topic

# Processing Settings
SAMPLE_RATE = 4.0                     # 取樣率（Hz）
TRAIN_TEST_SPLIT = 0.8                # 訓練/測試分割比例（80/20）
ANGULAR_OFFSET = 0.0                  # 角度偏移（radians）

# Filter Settings
FILTER_BACKWARDS = True               # 是否過濾倒退動作
END_SLACK = 3                         # 軌跡結尾緩衝（避免碰撞數據）
```

**重要設定：**
- `SAMPLE_RATE`: 控制資料密度，4.0 Hz 代表每秒取 4 個資料點
- `TRAIN_TEST_SPLIT`: 0.8 表示 80% 訓練，20% 測試
- `FILTER_BACKWARDS`: 建議設為 True，移除機器人倒退的片段
- `END_SLACK`: 移除軌跡結尾的 N 個點（通常結尾可能有碰撞）

#### 2.2 執行格式轉換
```bash
cd workspace/train/data
python3 dataset_train_format.py
```

轉換過程：
1. 讀取 `workspace/train/rosbags/` 中的所有 ROS2 bags
2. 同步影像和里程計資料
3. 按指定頻率取樣
4. 過濾倒退動作（如果啟用）
5. 儲存為訓練格式：
   - 影像：JPG 格式
   - 軌跡資料：pickle 格式（包含位置和方向）
6. 自動分割訓練/測試集

#### 2.3 驗證轉換結果
轉換完成後，檢查生成的檔案：

```bash
# 檢查處理後的資料
ls workspace/train/processed_data/my_dataset/

# 檢查分割檔案
cat workspace/train/vint_train/data/data_splits/my_dataset/train/traj_names.txt
cat workspace/train/vint_train/data/data_splits/my_dataset/test/traj_names.txt
```

預期結構：
```
processed_data/my_dataset/
├── my_dataset_20241105_143022_0/
│   ├── traj_data.pkl
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── my_dataset_20241105_143022_1/
    └── ...
```

---

### Step 3: 配置訓練參數

#### 3.1 更新訓練配置
編輯 `workspace/train/config/vint.yaml`，加入您的資料集：

```yaml
datasets:
  my_dataset:
    data_folder: /home/pomelo925/Desktop/visualnav-transformer/workspace/train/processed_data/my_dataset
    train: /home/pomelo925/Desktop/visualnav-transformer/workspace/train/vint_train/data/data_splits/my_dataset/train/
    test: /home/pomelo925/Desktop/visualnav-transformer/workspace/train/vint_train/data/data_splits/my_dataset/test/
    end_slack: 3                  # 與格式轉換設定一致
    goals_per_obs: 1              # 每個觀察的目標數量
    negative_mining: True         # 負樣本挖掘
```

**注意**: 路徑必須是絕對路徑！使用 `dataset_train_format.py` 完成後顯示的路徑。

#### 3.2 調整訓練超參數（可選）
```yaml
# Training setup
batch_size: 256                   # 批次大小（根據 GPU 記憶體調整）
epochs: 100                       # 訓練輪數
lr: 5e-4                          # 學習率
optimizer: adamw                  # 優化器

# Model params
model_type: vint                  # 模型類型：gnm, vint, nomad
context_size: 5                   # 歷史觀察數量
len_traj_pred: 5                  # 預測軌跡長度
learn_angle: True                 # 是否學習角度

# WandB logging
use_wandb: True                   # 是否使用 WandB 記錄
```

---

### Step 4: 開始訓練

#### 4.1 執行訓練
```bash
cd workspace/train
python train.py --config config/vint.yaml
```

#### 4.2 訓練監控
訓練過程中可以透過以下方式監控：

1. **終端輸出**: 顯示即時損失和指標
2. **本地日誌**: 儲存在 `logs/<project_name>/<run_name>/`
3. **WandB 儀表板**（如果啟用）:
   - 即時訓練曲線
   - 驗證指標
   - 樣本預測視覺化

#### 4.3 訓練輸出
訓練完成後，模型權重儲存在：
```
workspace/train/logs/<project_name>/<run_name>/latest.pth
```

---

## 快速參考

### 完整工作流程（一條龍）
```bash
# 1. 錄製資料
cd workspace/deployment/src/data
python3 dataset_record.py
# (操作機器人收集資料，完成後 Ctrl+C)

# 2. 轉換格式
python3 dataset_train_format.py

# 3. 更新配置（編輯 vint.yaml）
# 4. 開始訓練
cd ../../train
python train.py --config config/vint.yaml
```

### 重要路徑
- **錄製腳本**: `workspace/train/data/dataset_record.py`
- **轉換腳本**: `workspace/train/data/dataset_train_format.py`
- **Bag 儲存**: `workspace/train/rosbags/`
- **處理後資料**: `workspace/train/processed_data/`
- **訓練配置**: `workspace/train/config/vint.yaml`
- **訓練腳本**: `workspace/train/train.py`
- **模型輸出**: `workspace/train/logs/`
- **部署模型**: `workspace/model/`

### 常見問題

**Q: 錄製時找不到 topics？**
- 確認主程式 container 正在運行
- 使用 `ros2 topic list` 檢查可用的 topics
- 更新 `dataset_record.py` 中的 topic 名稱

**Q: 格式轉換失敗？**
- 檢查 topic 名稱是否與 bag 中的一致
- 確認影像訊息類型（Image vs CompressedImage）
- 修改 `process_ros2_image()` 函數以符合您的影像格式

**Q: 訓練時記憶體不足？**
- 減少 `batch_size`
- 減少 `num_workers`
- 使用較小的模型（gnm 而非 vint）

**Q: 如何使用訓練好的模型？**
- 模型權重在 `logs/<project_name>/<run_name>/latest.pth`
- 參考 `workspace/deployment/` 中的部署腳本進行推論

---

## 進階配置

### 自定義影像處理
如果您的影像格式特殊，可以修改 `dataset_train_format.py` 中的 `process_ros2_image()` 函數。

### 多資料集訓練
在 `vint.yaml` 中加入多個資料集：
```yaml
datasets:
  dataset1:
    data_folder: /path/to/dataset1
    train: /path/to/dataset1/train/
    test: /path/to/dataset1/test/
  dataset2:
    data_folder: /path/to/dataset2
    train: /path/to/dataset2/train/
    test: /path/to/dataset2/test/
```

### 繼續訓練
在 `vint.yaml` 中加入：
```yaml
load_run: <previous_run_name>  # 從之前的訓練繼續
```

---

**更新日期**: 2024-11-05  
**ROS 版本**: ROS2  
**支援的模型**: GNM, ViNT, NoMaD

