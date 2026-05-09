"""
YOLOv8n fine-tune on the local person dataset (CPU-only, fast settings).
Output: runs/train/queueflow/weights/best.pt
Copy that file to Model/yolov8n.pt when done.
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(__file__).parent / "data" / "merged" / "data.yaml"
OUTPUT_DIR = Path(__file__).parent / "runs" / "train"

model = YOLO("yolov8n.pt")   # downloads ~6 MB if not cached

results = model.train(
    data=str(DATA_YAML),
    epochs=10,
    imgsz=320,
    batch=8,
    device="cpu",
    workers=2,
    project=str(OUTPUT_DIR),
    name="queueflow",
    exist_ok=True,
    verbose=True,
    patience=5,        # early-stop if no improvement for 5 epochs
    cache=False,       # don't cache images — saves RAM on CPU machines
    plots=False,       # skip matplotlib plots — faster
)

best = Path(results.save_dir) / "weights" / "best.pt"
dest = Path(__file__).parent.parent / "Model" / "yolov8n.pt"
dest.parent.mkdir(parents=True, exist_ok=True)

import shutil
shutil.copy2(best, dest)
print(f"\nDone. Weights saved to {dest}")
