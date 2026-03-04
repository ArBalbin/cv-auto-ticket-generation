import os
import scipy.io as sio
from PIL import Image

SHANGHAI_ROOT = r"data/ShanghaiTech"
OUT_IMG_DIR = r"data/merged/images"
OUT_LABEL_DIR = r"data/merged/labels"

CLASS_ID = 0
BOX_W = 30
BOX_H = 30

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

parts = ['part_A', 'part_B']

for part in parts:
    for split in ['train_data', 'test_data']:
        ann_dir = os.path.join(SHANGHAI_ROOT, part, split, 'ground-truth')
        img_dir = os.path.join(SHANGHAI_ROOT, part, split, 'images')

        if not os.path.exists(ann_dir):
            continue

        for matfile in os.listdir(ann_dir):
            if not matfile.endswith('.mat'):
                continue

            matpath = os.path.join(ann_dir, matfile)
            data = sio.loadmat(matpath)

            points = data['image_info'][0][0][0][0][0]

            img_name = matfile.replace('.mat', '.jpg')
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            im = Image.open(img_path)
            w, h = im.size

            # copy image
            im.save(os.path.join(OUT_IMG_DIR, img_name))

            label_path = os.path.join(OUT_LABEL_DIR, img_name.replace('.jpg', '.txt'))

            with open(label_path, "w") as f:
                for p in points:
                    x, y = float(p[0]), float(p[1])

                    xc = x / w
                    yc = y / h
                    bw = BOX_W / w
                    bh = BOX_H / h

                    f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

print("✅ ShanghaiTech converted!")
