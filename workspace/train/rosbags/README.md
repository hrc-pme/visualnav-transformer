# ROS2 Bags Directory

此資料夾用於儲存使用 `dataset_record.py` 錄製的 ROS2 bag 檔案。

## 使用方式

1. 執行 `workspace/deployment/src/data/dataset_record.py` 錄製資料
2. ROS2 bags 會自動儲存在此資料夾
3. 使用 `workspace/deployment/src/data/dataset_train_format.py` 轉換格式

## 檔名格式

```
<DATASET_NAME>_<YYYYMMDD_HHMMSS>/
```

例如：`my_dataset_20241105_143022/`
