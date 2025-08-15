import os
import re
from PIL import Image
from tqdm import tqdm

# 設定輸入與輸出資料夾路徑
INPUT_DIR = "../topomap"
OUTPUT_DIR = "../topomap"
DIVISION = 1  # 每 DIVISION 張圖片保留一張（例如 2 代表保留 1/2）

def get_numeric_sort_key(filename):
    # 擷取檔名中的數字 (例如 "12.png" -> 12)
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else float('inf')

def filter_and_rename_images(input_dir, output_dir, division):
    assert division >= 1, "DIVISION must be at least 1"

    os.makedirs(output_dir, exist_ok=True)

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

    for idx, filename in enumerate(tqdm(selected_images, desc="Copying images")):
        src_path = os.path.join(input_dir, filename)
        dst_filename = f"{idx + 1}.png"
        dst_path = os.path.join(output_dir, dst_filename)

        try:
            img = Image.open(src_path).convert("RGB")
            img.save(dst_path)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    print(f"✅ Done. {len(selected_images)} images saved to '{output_dir}'.")

if __name__ == "__main__":
    filter_and_rename_images(INPUT_DIR, OUTPUT_DIR, DIVISION)
