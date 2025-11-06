# Quick Start Guide - ROS2 Training Pipeline

本快速指南提供最精簡的訓練流程步驟。

## 三步驟完成訓練

### 📹 Step 1: 錄製資料

```bash
cd workspace/train/data

# 1. 編輯 dataset_record.py，設定：
#    - IMAGE_TOPIC（影像 topic）
#    - ODOM_TOPIC（里程計 topic）
#    - DATASET_NAME（資料集名稱）

# 2. 開始錄製
python3 dataset_record.py

# 3. 操作機器人收集資料，完成後按 Ctrl+C
```

**輸出**: ROS2 bags 儲存在 `workspace/train/rosbags/`

---

### 🔄 Step 2: 轉換格式

```bash
cd workspace/train/data

# 1. 編輯 dataset_train_format.py，確認：
#    - DATASET_NAME（與錄製時相同）
#    - IMAGE_TOPIC 和 ODOM_TOPIC（與 bag 中的 topics 一致）
#    - SAMPLE_RATE（建議 4.0 Hz）
#    - TRAIN_TEST_SPLIT（建議 0.8）

# 2. 執行轉換
python3 dataset_train_format.py
```

**輸出**: 
- 處理後資料: `workspace/train/processed_data/my_dataset/`
- 訓練/測試分割: `workspace/train/vint_train/data/data_splits/my_dataset/`

腳本會顯示需要加入 `config/vint.yaml` 的路徑配置。

---

### 🚀 Step 3: 訓練模型

```bash
cd workspace/train

# 1. 編輯 config/vint.yaml，加入資料集（使用 Step 2 顯示的路徑）：
# datasets:
#   my_dataset:
#     data_folder: /absolute/path/to/processed_data/my_dataset
#     train: /absolute/path/to/data_splits/my_dataset/train/
#     test: /absolute/path/to/data_splits/my_dataset/test/
#     end_slack: 3
#     goals_per_obs: 1
#     negative_mining: True

# 2. 開始訓練
python train.py --config config/vint.yaml
```

**輸出**: 訓練好的模型 `workspace/train/logs/<project_name>/<run_name>/latest.pth`

**複製到部署位置**:
```bash
cp workspace/train/logs/<project>/<run>/latest.pth workspace/model/vint.pth
```

---

## 檢查清單

### 錄製前
- [ ] ROS2 主程式 container 正在運行
- [ ] 確認 topics 存在：`ros2 topic list`
- [ ] 已設定正確的 IMAGE_TOPIC 和 ODOM_TOPIC

### 轉換前
- [ ] ROS2 bags 已成功錄製在 `workspace/train/rosbags/`
- [ ] DATASET_NAME 與錄製時相同
- [ ] Topics 設定與 bag 中一致

### 訓練前
- [ ] 已在 `config/vint.yaml` 中加入資料集配置
- [ ] 路徑使用絕對路徑
- [ ] 根據 GPU 記憶體調整 batch_size

---

## 常用配置

### 資料錄製 (dataset_record.py)
```python
IMAGE_TOPIC = "/camera/image_raw"
ODOM_TOPIC = "/odom"
DATASET_NAME = "my_dataset"
```

### 格式轉換 (dataset_train_format.py)
```python
DATASET_NAME = "my_dataset"
IMAGE_TOPIC = "/camera/image_raw"
ODOM_TOPIC = "/odom"
SAMPLE_RATE = 4.0
TRAIN_TEST_SPLIT = 0.8
```

### 訓練配置 (config/vint.yaml)
```yaml
batch_size: 256
epochs: 100
lr: 5e-4
model_type: vint
context_size: 5
len_traj_pred: 5
```

---

## 故障排除

| 問題 | 解決方案 |
|------|---------|
| 找不到 topics | 確認主程式運行，使用 `ros2 topic list` 檢查 |
| 轉換失敗 | 檢查 topic 名稱，修改 `process_ros2_image()` |
| 記憶體不足 | 降低 batch_size 和 num_workers |
| 訓練不收斂 | 檢查資料品質，增加 epochs，調整學習率 |

---

詳細說明請參考: `README.md`
