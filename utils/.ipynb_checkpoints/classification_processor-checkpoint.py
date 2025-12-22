import cv2
import numpy as np

# ============================================================
# CONFIG (kept local on purpose)
# ============================================================
LABEL_COLOR = {
    0: (0, 0, 255),     # UG - Red
    1: (0, 255, 0),     # G  - Green
    2: (255, 165, 0),   # A  - Orange
}

# ============================================================
# GRID HELPERS
# ============================================================

import matplotlib.pyplot as plt

def stitch_cnn_grid(X, rows, cols, scale_to_255=True):
    """
    Stitch CNN input cells back into a single image.

    Args:
        X    : (N, 64, 64, 1)
        rows : inner grid rows (e.g. 14)
        cols : inner grid cols (e.g. 7)

    Returns:
        stitched image (H, W) uint8
    """
    assert X.shape[0] == rows * cols, \
        f"Expected {rows*cols}, got {X.shape[0]}"

    X = X[..., 0]  # drop channel

    grid_rows = []
    idx = 0
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            img = X[idx]
            if scale_to_255:
                img = (img * 255).astype(np.uint8)
            row_imgs.append(img)
            idx += 1
        grid_rows.append(np.hstack(row_imgs))

    stitched = np.vstack(grid_rows)
    return stitched


def show_cnn_input_grid(X, rows=14, cols=7, title="CNN input (BW)"):
    """
    Visualize CNN input images in a grid.

    Args:
        X    : numpy array (N, 64, 64, 1)
        rows : number of grid rows
        cols : number of grid cols
    """
    assert X.shape[0] == rows * cols, \
        f"Expected {rows*cols} images, got {X.shape[0]}"

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    fig.suptitle(title, fontsize=14)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(X[idx, :, :, 0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            idx += 1

    plt.tight_layout()
    plt.show()


def split_grid(img, rows, cols):
    """
    Split image into grid cells.
    Returns: list of (r, c, crop)
    """
    H, W = img.shape[:2]
    ch, cw = H // rows, W // cols

    cells = []
    for r in range(rows):
        for c in range(cols):
            crop = img[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            cells.append((r, c, crop))
    return cells, ch, cw


def extract_inner_cells(cells, rows, cols):
    """
    Keep only inner grid cells (drop border).
    """
    return [
        (r, c, crop)
        for (r, c, crop) in cells
        if 0 < r < rows-1 and 0 < c < cols-1
    ]


# ============================================================
# PREPROCESS FOR CNN
# ============================================================
def prepare_binary_cnn_input(inner_cells):
    """
    Convert inner cells → binary-mask CNN input
    """
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

        mask = cv2.resize(mask, (64, 64))
        X.append(mask)

    X = np.array(X, dtype=np.float32) / 255.0
    return X[..., None]

# ============================================================
# VISUALIZATION
# ============================================================
def draw_classified_grid(
    img,
    inner_cells,
    labels,
    ch,
    cw
):
    """
    Draw colored box boundaries on shifted image.
    """
    out = img.copy()

    for (r, c, _), label in zip(inner_cells, labels):
        color = LABEL_COLOR[int(label)]
        y1, y2 = r * ch, (r + 1) * ch
        x1, x2 = c * cw, (c + 1) * cw

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    return out


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def classify_shifted_image(
    shifted,warped,
    cnn_model,
    rows,
    cols
):
    """
    Args:
        shifted   : output of shift_boxes_to_grid_centers
        cnn_model : trained CNN (or None)
        rows,cols : grid size used in shifting

    Returns:
        vis_img   : shifted image with colored inner-grid boxes
        labels    : predicted labels (inner cells only)
    """

    # --------------------------------------------------
    # Split grid
    # --------------------------------------------------
    cells, ch, cw = split_grid(shifted, rows, cols)
    inner_cells = extract_inner_cells(cells, rows, cols)

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------
    if cnn_model is None:
        labels = np.ones(len(inner_cells), dtype=int)
    else:
        X = prepare_binary_cnn_input(inner_cells)
        vis_stiched = stitch_cnn_grid(X, rows-2, cols-2, scale_to_255=True)
        
        # 🔍 DEBUG: visualize CNN input
        #show_cnn_input_grid(
        #X,
        #rows=rows-2,
        #cols=cols-2,
        #title="CNN input cells (14 × 7)"
        #)
        
        
        
        preds = cnn_model.predict(X, verbose=0)
        labels = np.argmax(preds, axis=1)
    
        print(np.mean(preds, axis=0))

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    vis_shifted = draw_classified_grid(
        shifted,
        inner_cells,
        labels,
        ch,
        cw
    )
    
    vis_warped = draw_classified_grid(
        warped,
        inner_cells,
        labels,
        ch,
        cw
    )

    return vis_shifted,vis_warped,vis_stiched, labels

