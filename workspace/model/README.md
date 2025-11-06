# Model Weights Directory

此資料夾用於存放訓練完成的模型權重，以及部署時使用的模型檔案。

## 目錄結構

```
model/
├── gnm.pth          # GNM 模型權重
├── vint.pth         # ViNT 模型權重
├── nomad.pth        # NoMaD 模型權重
└── README.md        # 本文件
```

## 用途

### 訓練輸出
訓練完成後，模型權重會從 `train/logs/<project>/<run>/latest.pth` 複製到此處：

```bash
# 訓練完成後複製模型
cp train/logs/vint-release/vint_YYYYMMDD_HHMMSS/latest.pth model/vint.pth
```

### 部署使用
部署腳本會從此資料夾載入模型：

- `deploy/src/navigate.ros2.py` - 載入模型進行導航
- `deploy/gpu_container/inference_server.py` - GPU 容器中的推論服務

## 模型下載

如果沒有訓練好的模型，可以下載預訓練權重：

- **GNM**: https://drive.google.com/file/d/1bzCPd_OsXjS2aGPTQladbI8ImxLZwrQh/view
- **ViNT**: https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5/view
- **NoMaD**: https://drive.google.com/file/d/1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb/view

## 配置

模型配置檔案位於 `deploy/config/models.yaml`，指定每個模型使用的權重路徑。

```yaml
vint: 
  config_path: "../../train/config/vint.yaml"
  ckpt_path: "../model/vint.pth"  # 指向此資料夾
```

---

**更新日期**: 2024-11-06
