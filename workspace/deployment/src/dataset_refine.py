import os
import re
import shutil
from PIL import Image
from tqdm import tqdm

# 設定輸入與輸出資料夾路徑
INPUT_DIR = "../topomaps/6e-dr"  # 要處理的資料集
OUTPUT_DIR = "../topomaps/6e-dr"  # 處理後的資料集
DIVISION = 1  # 每 DIVISION 張圖片保留一張（例如 2 代表保留 1/2）

# 原始資料保存設定
SAVE = False  # 是否保存原始資料
SAVE_DIR = "../topomaps/6e6-raw"  # 原始資料保存路徑

def get_numeric_sort_key(filename):
    # 擷取檔名中的數字 (例如 "12.png" -> 12)
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else float('inf')

def filter_and_rename_images(input_dir, output_dir, division, save=False, save_dir=None):
    assert division >= 1, "DIVISION must be at least 1"

    # 如果需要保存原始資料，先備份
    if save and save_dir:
        if os.path.exists(save_dir):
            print(f"⚠️ Save directory '{save_dir}' already exists. Skipping backup.")
        else:
            print(f"📁 Backing up original data to '{save_dir}'...")
            shutil.copytree(input_dir, save_dir)
            print(f"✅ Original data backed up to '{save_dir}'")

    # 取得所有圖片檔案，並依數字排序
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg"))
    ], key=get_numeric_sort_key)

    if not image_files:
        print("⚠️ No image files found in the input directory.")
        return

    print(f"Total images found: {len(image_files)}")
    print(f"Keeping 1 out of every {division} images...")

    # 每 division 張取一張
    selected_images = image_files[division - 1::division]

    # 如果 input_dir 和 output_dir 相同，需要先處理到臨時資料夾
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        temp_dir = output_dir + "_temp"
        os.makedirs(temp_dir, exist_ok=True)
        actual_output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
        actual_output_dir = output_dir

    for idx, filename in enumerate(tqdm(selected_images, desc="Processing images")):
        src_path = os.path.join(input_dir, filename)
        dst_filename = f"{idx}.png"  # 修改為從 0 開始
        dst_path = os.path.join(actual_output_dir, dst_filename)

        try:
            img = Image.open(src_path).convert("RGB")
            img.save(dst_path)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    # 如果使用了臨時資料夾，替換原始資料夾
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        # 清除原始資料夾的所有圖片檔案
        for f in os.listdir(input_dir):
            if f.lower().endswith((".png", ".jpg")):
                os.remove(os.path.join(input_dir, f))
        
        # 將處理後的圖片移到原始資料夾
        for f in os.listdir(temp_dir):
            shutil.move(os.path.join(temp_dir, f), os.path.join(output_dir, f))
        
        # 刪除臨時資料夾
        os.rmdir(temp_dir)

    print(f"✅ Done. {len(selected_images)} images saved to '{output_dir}'.")

if __name__ == "__main__":
    filter_and_rename_images(INPUT_DIR, OUTPUT_DIR, DIVISION, SAVE, SAVE_DIR)
