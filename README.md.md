# UCF-Crime Video Anomaly Detection (CPU)

## Project summary
This project implements binary video-level anomaly classification on an available subset of UCF-Crime. It extracts frame features using pretrained ResNet18 and classifies each video using an attention-based Multiple Instance Learning (MIL) model.

**Important:** This is a ResNet18 + attention-MIL implementation. It is not a full TimeSformer-MIL implementation and is not an exact reproduction of a research paper.

## Dataset and labels
- Normal = `0`
- Abnormal = `1`
- Available unique videos in the reported experiment: 105
  - Normal: 36
  - Abnormal: 69
- Initial CSV entries: 117 (41 normal, 76 abnormal); 12 duplicate entries were removed by content hashing.
- The abnormal category folders/names in the available subset include Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Shooting, Shoplifting and Stealing.

The full UCF-Crime dataset is not included in this repository. Obtain it through the dataset's official distribution process and follow its terms. Do not publicly upload dataset videos unless you have permission.

## Method
1. Read normal and abnormal videos and assign binary labels.
2. Identify exact duplicate files using SHA-256 hashes.
3. Sample 32 frames per video.
4. Resize frames to 224 × 224 and apply the pretrained ResNet18 image preprocessing.
5. Extract one 512-dimensional feature vector per frame: `(32, 512)` per video.
6. Apply attention over frame features and weighted pooling to create a video representation.
7. Train a binary classifier and evaluate with accuracy, precision, recall, F1-score and confusion matrix.

## Reported experiment
The selected custom test split contained 21 videos (7 normal, 14 abnormal).

| Metric | Result |
|---|---:|
| Accuracy | 85.71% |
| Precision | 82.35% |
| Recall | 100.00% |
| F1-score | 90.30% |
| Balanced accuracy | 78.57% |

Confusion matrix (actual rows; predicted columns; order Normal, Abnormal):

```text
[[4, 3],
 [0, 14]]
```

Three of seven normal test videos were flagged as abnormal. The false-alarm rate on normal test videos was 42.86%.

**Evaluation limitation:** These are results on a small custom split of a 105-video subset. The official UCF-Crime train/test split was not verified. Do not describe these as official benchmark results or compare them as if the protocol were identical to a paper.

## Suggested repository structure
```text
project/
├── dataset/                 # Keep local; normally do not commit videos
├── features/
│   ├── clean_extracted/     # Generated .npy features; optional, may be large
│   └── clean_labels.csv
├── models/                  # Model checkpoints; check file size before commit
├── results/                 # Metrics, confusion matrix and plots
├── src/
│   ├── prepare_labels.py
│   ├── extract_features.py
│   ├── train.py
│   └── evaluate.py
├── UCF_Crime_System_Architecture.png
├── README.md
└── requirements.txt
```

## Run locally
Activate the virtual environment in PowerShell, then install dependencies:
```powershell
python -m pip install -r requirements.txt
python src/prepare_labels.py
python src/extract_features.py
python src/train.py
```
Check the actual script filenames in `src/` and update commands if yours differ.

## GitHub upload
Do not commit `.venv/`, raw UCF-Crime videos, private information, or huge generated files. Add a `.gitignore` such as:
```gitignore
.venv/
__pycache__/
*.pyc
dataset/
*.mp4
*.avi
*.mkv
*.mov
```
If you have permission to share generated features or model weights, check their sizes and include them only if suitable. Otherwise document how to regenerate them. Add your own genuine VS Code, training and result screenshots under `docs/screenshots/`.

## Citation
Add the exact UCF-Crime dataset paper citation and the references for ResNet and attention-based MIL used in your implementation. Verify all bibliographic details before submission.
