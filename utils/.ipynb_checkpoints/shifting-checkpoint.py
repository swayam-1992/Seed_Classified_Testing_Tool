import cv2
import numpy as np


# ============================================================
# Grid reconstruction (must match assignment logic)
# ============================================================
def erase_box_overlap(region, region_rect, boxes):
    """
    Remove pixels in `region` that overlap with any box.
    region_rect = (gx1, gy1, gx2, gy2)
    """
    gx1, gy1, gx2, gy2 = region_rect
    cleaned = region.copy()

    for bx1, by1, bx2, by2 in boxes:
        ix1 = max(gx1, bx1)
        iy1 = max(gy1, by1)
        ix2 = min(gx2, bx2)
        iy2 = min(gy2, by2)

        if ix1 < ix2 and iy1 < iy2:
            rx1, ry1 = ix1 - gx1, iy1 - gy1
            rx2, ry2 = ix2 - gx1, iy2 - gy1
            cleaned[ry1:ry2, rx1:rx2] = 0

    return cleaned


def build_extended_grid(img_shape, rows, cols):
    """
    Build (rows+2) x (cols+2) grid rectangles.
    """
    h, w = img_shape[:2]
    row_step = h / (rows + 1)
    col_step = w / (cols + 1)

    grid = {}
    for r in range(rows + 2):
        for c in range(cols + 2):
            x1 = int(round(c * col_step))
            y1 = int(round(r * row_step))
            x2 = int(round((c + 1) * col_step))
            y2 = int(round((r + 1) * row_step))
            grid[(r, c)] = (x1, y1, x2, y2)

    return grid


# ============================================================
# Core shifting logic
# ============================================================

def shift_boxes_to_grid_centers(
    warped_original,
    boxes,
    cell_to_box,
    rows,
    cols,
):
    h, w = warped_original.shape[:2]
    canvas = np.zeros_like(warped_original)

    grid = build_extended_grid(warped_original.shape, rows, cols)

    for (r, c), (gx1, gy1, gx2, gy2) in grid.items():
        grid_w = gx2 - gx1
        grid_h = gy2 - gy1

        # --------------------------------------------
        # Case 1: no box → copy grid WITHOUT box pixels
        # --------------------------------------------
        if (r, c) not in cell_to_box:
            region = warped_original[gy1:gy2, gx1:gx2]
            cleaned = erase_box_overlap(
                region,
                (gx1, gy1, gx2, gy2),
                boxes
            )
            canvas[gy1:gy2, gx1:gx2] = cleaned
            continue

        # --------------------------------------------
        # Case 2: box assigned → crop & center
        # --------------------------------------------
        bi = cell_to_box[(r, c)]
        bx1, by1, bx2, by2 = map(int, boxes[bi])

        crop = warped_original[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue

        ch, cw = crop.shape[:2]

        cx = gx1 + (grid_w - cw) // 2
        cy = gy1 + (grid_h - ch) // 2

        cx = max(gx1, cx)
        cy = max(gy1, cy)
        cx2 = min(cx + cw, gx2)
        cy2 = min(cy + ch, gy2)

        crop = crop[: cy2 - cy, : cx2 - cx]
        canvas[cy:cy2, cx:cx2] = crop

    return canvas
