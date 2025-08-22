import os
import pickle
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ======== CONSTANT SETTINGS ========
INPUT_DIR = "../topomaps/topomap"    # 原始影像與 traj_data.pkl 所在資料夾
OVERWRITE = True                     # 是否覆蓋 INPUT_DIR 的 processed 資料夾; False 會寫入 OUTPUT_DIR
GENERATE_VIDEO = True               # 是否生成影片，False 就只做標註不生成影片
FPS = 150                           # 影片每秒幀數 (只在 GENERATE_VIDEO=True 時有效)
ROTATE = 90                        # 旋轉角度 (0, 90, 180, 270)
PROCESS_PKL = True                 # 是否處理 traj_data.pkl，False 就跳過
# 若 OVERWRITE == False，則設定 OUTPUT_DIR
OUTPUT_DIR = "../topomaps/topomap_annotated"
# ===================================



def draw_pose_info(image_path, pose, output_path):
    image = Image.open(image_path).convert("RGB")

    if ROTATE in [90, 180, 270]:
        image = image.rotate(ROTATE, expand=True)

    draw = ImageDraw.Draw(image)
    text = f"x: {pose['x']:.2f}, y: {pose['y']:.2f}, yaw: {pose['yaw']:.2f}"

    try:
        font = ImageFont.truetype("arial.ttf", size=36)
    except:
        font = ImageFont.load_default()

    draw.rectangle([5, 5, 5 + 480, 5 + 40], fill="black")
    draw.text((10, 10), text, font=font, fill="white")

    filename = os.path.basename(image_path)
    frame_num = os.path.splitext(filename)[0]
    frame_text = f"Frame: {frame_num}"

    try:
        font_small = ImageFont.truetype("arial.ttf", size=28)
    except:
        font_small = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), frame_text, font=font_small)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x_pos = image.width - text_width - 10
    y_pos = image.height - text_height - 10

    draw.rectangle([x_pos - 5, y_pos - 5, x_pos + text_width + 5, y_pos + text_height + 5], fill="black")
    draw.text((x_pos, y_pos), frame_text, font=font_small, fill="white")

    image.save(output_path)


def make_video(image_dir, output_path, fps):
    def numerical_sort(value):
        return int(os.path.splitext(value)[0])

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    image_files.sort(key=numerical_sort)

    if not image_files:
        print("No images found for video generation.")
        return

    first_img_path = os.path.join(image_dir, image_files[0])
    first_img = cv2.imread(first_img_path)
    height, width = first_img.shape[:2]
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(output_path, fourcc, fps, size)

    for img_file in tqdm(image_files, desc="Creating video"):
        img_path = os.path.join(image_dir, img_file)
        frame = cv2.imread(img_path)
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, size)
        video.write(frame)

    video.release()
    print(f"✅ Video saved to {output_path}")

    # 刪除 processed 裡所有 jpg
    for f in image_files:
        try:
            os.remove(os.path.join(image_dir, f))
        except Exception as e:
            print(f"Failed to delete {f}: {e}")


def main():
    if OVERWRITE:
        processed_dir = os.path.join(INPUT_DIR, "processed")
    else:
        processed_dir = os.path.join(OUTPUT_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    if PROCESS_PKL:
        traj_pkl_path = os.path.join(INPUT_DIR, "traj_data.pkl")
        with open(traj_pkl_path, "rb") as f:
            traj_data = pickle.load(f)

        print("Annotating images...")
        for idx, pose in enumerate(traj_data):
            img_filename = f"{idx}.jpg"
            img_input_path = os.path.join(INPUT_DIR, img_filename)
            img_output_path = os.path.join(processed_dir, img_filename)
            if os.path.exists(img_input_path):
                draw_pose_info(img_input_path, pose, img_output_path)
    else:
        print("Skipping processing of traj_data.pkl")

    if GENERATE_VIDEO:
        print("Generating video...")
        video_output_path = os.path.join(processed_dir, "topomap_video.avi")
        make_video(processed_dir, video_output_path, FPS)
    else:
        print("Skipping video generation.")


if __name__ == "__main__":
    main()
