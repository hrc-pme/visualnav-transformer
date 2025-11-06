# Visual Navigation Transformer (ViNT) 訓練工作區

本工作區提供完整的視覺導航 Transformer 模型訓練流程。

## 快速開始訓練流程

### 1. 資料收集

編輯 `train/data/dataset_record.py`，設定資料集名稱：
```python
DATASET_NAME = "my_dataset"  # 改成你的資料集名稱
```

執行錄製 ROS2 bag 檔案：
```bash
cd /workspace/train/data
python dataset_record.py
```

### 2. 資料處理

編輯 `train/data/dataset_train_format.py`，設定同樣的資料集名稱：
```python
DATASET_NAME = "my_dataset"  # 必須與上面相同
```

將 ROS2 bags 轉換為訓練格式：
```bash
cd /workspace/train/data
python dataset_train_format.py
```

### 3. 資料分割

處理腳本會自動建立訓練/測試分割檔案：
- `vint_train/data/data_splits/{dataset_name}/train/traj_names.txt`
- `vint_train/data/data_splits/{dataset_name}/test/traj_names.txt`

### 4. 註冊資料集

將資料集加入 `vint_train/data/data_config.yaml`：
```yaml
my_dataset:
  metric_waypoint_spacing: 0.25  # 視覺化用的航點間距（公尺）
```

### 5. 配置訓練

編輯模型設定檔（例如 `config/vint.yaml`）：
```yaml
# 訓練識別名稱
training_name: "vint_baseline"      # 這次訓練的名稱

# 資料集配置
dataset_name: "my_dataset"          # 你的資料集資料夾名稱

# 訓練超參數
batch_size: 16
epochs: 30
learning_rate: 5e-4
```

### 6. 開始訓練

```bash
cd /workspace/train
python train.py --config config/vint.yaml
```

訓練好的模型會儲存到：`/workspace/model/{training_name}_{timestamp}/`

## 可用模型

- **ViNT** (`config/vint.yaml`): Visual Navigation Transformer 基礎模型
- **GNM** (`config/gnm.yaml`): General Navigation Model
- **NoMaD** (`config/nomad.yaml`): 目標遮罩擴散策略
- **Late Fusion** (`config/late_fusion.yaml`): 多上下文後期融合變體

## 目錄結構

```
/workspace/
├── train/
│   ├── datasets/           # 所有資料集
│   │   └── {dataset_name}/
│   │       ├── rosbags/           # 原始 ROS2 bags
│   │       ├── processed_data/    # 處理後的軌跡
│   │       └── trained_weights/   # 資料集專屬模型
│   ├── config/             # 訓練設定檔
│   ├── data/              # 資料收集與處理腳本
│   └── vint_train/        # 訓練函式庫
└── model/                 # 訓練輸出的模型
```

## 注意事項

- **Wandb**: 所有設定檔預設 `use_wandb: true`，entity 為 `gnmv2`
- **多資料集**: `train.py` 會自動發現並使用 `datasets/` 中的所有資料集
- **資料需求**: 建議每個資料集至少 10-20 條軌跡才能可靠訓練

