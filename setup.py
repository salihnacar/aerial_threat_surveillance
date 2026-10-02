import urllib.request
import os

os.makedirs(os.path.join("data", "input"), exist_ok=True)
video_path = os.path.join("data", "input", "input.mp4")

print("[INFO] Downloading surveillance test video...")
url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
urllib.request.urlretrieve(url, video_path)
print(f"[SUCCESS] Video downloaded to '{video_path}'")