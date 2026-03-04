import os
import glob
import shutil
import random

SRC_IMG = r'data/merged/images'
SRC_LABEL = r'data/merged/labels'

DST_IMG_TRAIN = r'data/merged/images/train'
DST_IMG_VAL = r'data/merged/images/val'
DST_LABEL_TRAIN = r'data/merged/labels/train'
DST_LABEL_VAL = r'data/merged/labels/val'

# Create output directories
for d in [DST_IMG_TRAIN, DST_IMG_VAL, DST_LABEL_TRAIN, DST_LABEL_VAL]:
    os.makedirs(d, exist_ok=True)

# Get all images EXCEPT train/val folders
imgs = glob.glob(os.path.join(SRC_IMG, '*.jpg')) + glob.glob(os.path.join(SRC_IMG, '*.png'))

print(f"Total images found: {len(imgs)}")

random.shuffle(imgs)

val_ratio = 0.1
val_count = int(len(imgs) * val_ratio)

for i, img_path in enumerate(imgs):
    name = os.path.basename(img_path)
    label = os.path.join(SRC_LABEL, os.path.splitext(name)[0] + '.txt')

    if i < val_count:
        # VAL SET
        shutil.move(img_path, os.path.join(DST_IMG_VAL, name))
        if os.path.exists(label):
            shutil.move(label, os.path.join(DST_LABEL_VAL, os.path.basename(label)))
    else:
        # TRAIN SET
        shutil.move(img_path, os.path.join(DST_IMG_TRAIN, name))
        if os.path.exists(label):
            shutil.move(label, os.path.join(DST_LABEL_TRAIN, os.path.basename(label)))

print("✅ Train/Val split completed successfully!")
