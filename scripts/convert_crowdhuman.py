import os
from shutil import copyfile
from PIL import Image

# ✅ CHANGE PATH ACCORDING TO YOUR PROJECT
CROWD_IMG_DIR = r"data/CrowdHuman Cropped/Dataset CrowdHuman/crowd"

OUT_IMG_DIR = r"data/merged/images"
OUT_LABEL_DIR = r"data/merged/labels"

CLASS_ID = 0  # person

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# Convert simple crowd images (no labels, so fake boxes in center)
for imgname in os.listdir(CROWD_IMG_DIR):
    if not imgname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    imgpath = os.path.join(CROWD_IMG_DIR, imgname)

    # copy image
    copyfile(imgpath, os.path.join(OUT_IMG_DIR, imgname))

    # create dummy YOLO label (approx center box)
    with Image.open(imgpath) as im:
        w, h = im.size

    label_path = os.path.join(OUT_LABEL_DIR, imgname.rsplit('.', 1)[0] + ".txt")

    with open(label_path, "w") as f:
        xc, yc = 0.5, 0.5
        bw, bh = 0.3, 0.3
        f.write(f"{CLASS_ID} {xc} {yc} {bw} {bh}\n")

print("✅ CrowdHuman images converted!")
