from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent

NORMAL_DIR = ROOT / "dataset" / "Unsorted_Normal"
ABNORMAL_DIR = ROOT / "dataset" / "UCF_Crime" / "Videos"

OUTPUT_DIR = ROOT / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "video_labels.csv"

EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}

print("Normal folder:", NORMAL_DIR)
print("Normal folder exists:", NORMAL_DIR.exists())

print("Abnormal folder:", ABNORMAL_DIR)
print("Abnormal folder exists:", ABNORMAL_DIR.exists())

# Find normal videos
normal_videos = [
    p for p in NORMAL_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
]

# Find abnormal videos
abnormal_videos = [
    p for p in ABNORMAL_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
]

print("\nNormal videos detected:", len(normal_videos))
print("Abnormal videos detected:", len(abnormal_videos))

# Write CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video_name", "video_path", "label"])

    for video in sorted(normal_videos):
        writer.writerow([video.name, str(video.resolve()), 0])

    for video in sorted(abnormal_videos):
        writer.writerow([video.name, str(video.resolve()), 1])

print("\nLABEL PREPARATION COMPLETED")
print("Total videos:", len(normal_videos) + len(abnormal_videos))
print("Normal videos:", len(normal_videos))
print("Abnormal videos:", len(abnormal_videos))
print("Saved:", OUTPUT_FILE)