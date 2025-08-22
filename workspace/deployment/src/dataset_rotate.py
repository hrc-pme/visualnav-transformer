import os
from PIL import Image
from tqdm import tqdm

# 設定輸入資料夾路徑
INPUT_DIR = "../topomap"
ROTATE = 270  # 支援 0, 90, 180, 270

def rotate_images_in_place(directory, angle):
    assert angle in [0, 90, 180, 270], "Only 0, 90, 180, 270 degrees are supported."

    image_files = [f for f in os.listdir(directory) if (f.lower().endswith(".jpg") or f.lower().endswith(".png"))]

    if not image_files:
        print("⚠️ No JPG images found in the directory.")
        return

    print(f"Rotating {len(image_files)} images by {angle} degrees...")

    for filename in tqdm(image_files, desc="Rotating images"):
        path = os.path.join(directory, filename)
        try:
            img = Image.open(path).convert("RGB")
            img_rotated = img.rotate(angle, expand=True)
            img_rotated.save(path)  # 直接覆蓋原圖
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

    print("✅ All images rotated and saved.")

if __name__ == "__main__":
    rotate_images_in_place(INPUT_DIR, ROTATE)
