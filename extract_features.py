import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent.parent

LABEL_FILE = ROOT / "features" / "video_labels.csv"
NORMAL_DIR = ROOT / "dataset" / "Unsorted_Normal"
OUTPUT_DIR = ROOT / "features" / "extracted"

DEVICE = torch.device("cpu")
FRAMES_PER_VIDEO = 32

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD MODEL ----------------

print("Loading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Identity()

model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- HELPER FUNCTIONS ----------------

def file_hash(path):
    """Return SHA256 hash to identify exact duplicate videos."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def extract_video_features(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(
        0, total_frames - 1, FRAMES_PER_VIDEO
    ).astype(int)

    frames = []

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frames.append(transform(image))

    cap.release()

    if not frames:
        return None

    batch = torch.stack(frames).to(DEVICE)

    with torch.inference_mode():
        features = model(batch)

    return features.cpu().numpy().astype(np.float32)


# ---------------- COLLECT VIDEO PATHS ----------------

video_items = []

# Existing videos listed in the label CSV
if LABEL_FILE.exists():
    with open(LABEL_FILE, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for item in reader:
            path = Path(item["video_path"])

            if not path.is_absolute():
                path = ROOT / path

            video_items.append(path)

else:
    print("Warning: video_labels.csv was not found.")
    print("Only the unsorted normal videos will be processed.")

# Newly downloaded normal videos
if NORMAL_DIR.exists():
    normal_videos = sorted(
        path for path in NORMAL_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in
        [".mp4", ".avi", ".mov", ".mkv"]
    )

    # Remove exact duplicate downloads using file hashes
    seen_hashes = set()
    unique_normal_videos = []

    for path in normal_videos:
        try:
            digest = file_hash(path)

            if digest in seen_hashes:
                print("Duplicate skipped:", path.name)
                continue

            seen_hashes.add(digest)
            unique_normal_videos.append(path)

        except OSError as error:
            print("Could not check:", path.name, error)

    print("Normal files found:", len(normal_videos))
    print("Unique normal files:", len(unique_normal_videos))

    video_items.extend(unique_normal_videos)

else:
    print("Normal folder not found:", NORMAL_DIR)

# Remove repeated paths
unique_paths = []
seen_paths = set()

for path in video_items:
    key = str(path.resolve())

    if key not in seen_paths:
        seen_paths.add(key)
        unique_paths.append(path)

print("\nTotal candidate videos:", len(unique_paths))

# ---------------- EXTRACT FEATURES ----------------

successful = 0
skipped = 0
failed = 0

for video_path in tqdm(unique_paths, desc="Extracting features"):

    if not video_path.exists():
        print("\nMissing video:", video_path)
        failed += 1
        continue

    output_path = OUTPUT_DIR / (video_path.stem + ".npy")

    # Preserve existing feature files
    if output_path.exists():
        skipped += 1
        continue

    try:
        features = extract_video_features(video_path)

        if features is None:
            print("\nCould not read:", video_path.name)
            failed += 1
            continue

        np.save(output_path, features)
        successful += 1

    except Exception as error:
        print("\nError:", video_path.name, error)
        failed += 1

# ---------------- FINAL REPORT ----------------

print("\nFEATURE EXTRACTION COMPLETED")
print("New feature files created:", successful)
print("Existing feature files skipped:", skipped)
print("Failed videos:", failed)
print("Feature output folder:", OUTPUT_DIR)
print("Feature dimension: 512")
print("Frames per video:", FRAMES_PER_VIDEO)