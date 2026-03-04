# inference/heatmap.py
import numpy as np
import cv2

def make_heatmap(frame, boxes, grid=(3,3)):
    h, w = frame.shape[:2]
    grid_h = grid[0]
    grid_w = grid[1]

    counts = np.zeros((grid_h, grid_w), dtype=np.float32)
    for (x1,y1,x2,y2) in boxes:
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        col = min(grid_w-1, int(cx / (w/grid_w)))
        row = min(grid_h-1, int(cy / (h/grid_h)))
        counts[row, col] += 1

    # normalize counts to 0-255
    if counts.max() > 0:
        norm = (counts / counts.max() * 255).astype(np.uint8)
    else:
        norm = counts.astype(np.uint8)

    heat = cv2.resize(norm, (w, h), interpolation=cv2.INTER_NEAREST)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heat_color
