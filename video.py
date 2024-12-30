import cv2
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# 設定參數
image_folder = r"C:\Users\USER\Desktop\school\runs\detect\predict"  # 存放圖片的資料夾
output_folder = r"C:\Users\USER\Desktop\school\output_videos"  # 儲存影片的資料夾
frame_rate = 30  # 每秒幀數（FPS）

# 建立輸出資料夾
Path(output_folder).mkdir(parents=True, exist_ok=True)

# 解析圖片名稱並分組
grouped_images = defaultdict(list)
for image_path in Path(image_folder).glob("*.jpg"):
    filename = image_path.stem  # 取得檔名（無副檔名）
    parts = filename.split("_")
    if len(parts) == 3:  # 確保名稱結構正確
        camera, period, sequence = parts
        key = f"{camera}_{period}"  # 分組鍵值
        grouped_images[key].append((int(sequence), image_path))  # 按序號存入分組

# 為每個組別生成影片
for group, images in grouped_images.items():
    # 按時間序號排序
    sorted_images = sorted(images, key=lambda x: x[0])
    image_paths = [img[1] for img in sorted_images]

    # 讀取第一張圖片以確定影片尺寸
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        print(f"Error: Cannot read first image for group {group}. Skipping.")
        continue
    height, width, layers = first_image.shape
    video_size = (width, height)

    # 初始化影片編碼器
    video_path = Path(output_folder) / f"{group}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 編碼
    video_writer = cv2.VideoWriter(str(video_path), fourcc, frame_rate, video_size)

    # 將圖片寫入影片
    pbar = tqdm(image_paths)
    for image_path in pbar:
        pbar.set_description(f'{group}')
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read image {image_path}, skipping.")
            continue
        video_writer.write(img)

    # 釋放資源
    video_writer.release()
    print(f"Video saved as {video_path}")

print("All videos have been generated!")