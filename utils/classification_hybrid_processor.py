import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 64

LABEL_COLOR = {
    0: (0, 0, 255),     # UG
    1: (0, 255, 0),     # G
    2: (255, 165, 0),   # A
}

# Hybrid thresholds (from training statistics)
UG_THR = 30
G_THR  = 650

# ============================================================
# GRID HELPERS
# ============================================================
def split_grid(img, rows, cols):
    H, W = img.shape[:2]
    ch, cw = H // rows, W // cols
    cells = []
    for r in range(rows):
        for c in range(cols):
            crop = img[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            cells.append((r, c, crop))
    return cells, ch, cw


def extract_inner_cells(cells, rows, cols):
    return [
        (r, c, crop)
        for (r, c, crop) in cells
        if 0 < r < rows-1 and 0 < c < cols-1
    ]


# ============================================================
# CNN INPUT PREP
# ============================================================
def prepare_binary_cnn_input(inner_cells):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    X = []

    for _, _, cell in inner_cells:
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            (30, 35, 30),
            (90, 255, 255)
        )

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        X.append(mask)

    X = np.array(X, dtype=np.float32) / 255.0
    return X[..., None]


def stitch_cnn_grid(X, rows, cols):
    X = (X[..., 0] * 255).astype(np.uint8)
    idx = 0
    rows_img = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(X[idx])
            idx += 1
        rows_img.append(np.hstack(row))
    return np.vstack(rows_img)


# ============================================================
# HYBRID CLASSIFIER
# ============================================================
def count_white_pixels(img):
    return int((img > 0.5).sum())


def hybrid_predict_batch(X, cnn_model):
    """
    Hybrid prediction:
      - Fast rule-based for UG/G
      - CNN fallback for ambiguous cases
    """
    labels = []

    for x in X:
        wp = count_white_pixels(x)

        if wp < UG_THR:
            labels.append(0)
        elif wp > G_THR:
            labels.append(1)
        else:
            pred = cnn_model.predict(
                x.reshape(1, IMG_SIZE, IMG_SIZE, 1),
                verbose=0
            )
            labels.append(int(np.argmax(pred[0])))

    return np.array(labels, dtype=int)


# ============================================================
# VISUALIZATION
# ============================================================
def draw_classified_grid(img, inner_cells, labels, ch, cw):
    out = img.copy()

    for (r, c, _), label in zip(inner_cells, labels):
        color = LABEL_COLOR[int(label)]
        y1, y2 = r * ch, (r + 1) * ch
        x1, x2 = c * cw, (c + 1) * cw
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    return out


# ============================================================
# MAIN ENTRY POINT (DROP-IN)
# ============================================================
def classify_shifted_image(
    shifted,
    warped,
    cnn_model,
    rows,
    cols
):
    """
    Args:
        shifted   : output of shift_boxes_to_grid_centers
        warped    : perspective-corrected image
        cnn_model : loaded FINAL_seed_germination_model.h5
        rows,cols : grid size (e.g. 16, 9)

    Returns:
        vis_shifted
        vis_warped
        vis_stitched
        labels
    """

    # -------------------------------
    # Grid split
    # -------------------------------
    cells, ch, cw = split_grid(shifted, rows, cols)
    inner_cells = extract_inner_cells(cells, rows, cols)

    # -------------------------------
    # CNN input
    # -------------------------------
    X = prepare_binary_cnn_input(inner_cells)
    vis_stitched = stitch_cnn_grid(X, rows-2, cols-2)

    # -------------------------------
    # Hybrid classification
    # -------------------------------
    labels = hybrid_predict_batch(X, cnn_model)

    # -------------------------------
    # Visualization
    # -------------------------------
    vis_shifted = draw_classified_grid(
        shifted, inner_cells, labels, ch, cw
    )

    vis_warped = draw_classified_grid(
        warped, inner_cells, labels, ch, cw
    )

    return vis_shifted, vis_warped, vis_stitched, labels
