import cv2
import numpy as np
from collections import defaultdict


# ============================================================
# Geometry helpers
# ============================================================

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(box, rect):
    """
    box, rect: (x1, y1, x2, y2)
    """
    x1 = max(box[0], rect[0])
    y1 = max(box[1], rect[1])
    x2 = min(box[2], rect[2])
    y2 = min(box[3], rect[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


# ============================================================
# Grid construction
# ============================================================

def build_extended_grid(img_shape, rows, cols):
    """
    Build (rows+2) x (cols+2) grid rectangles.
    Returns dict[(r, c)] = (x1, y1, x2, y2)
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


def is_outer_cell(r, c, rows, cols):
    return (
        r == 0 or c == 0 or
        r == rows + 1 or c == cols + 1
    )


def neighboring_cells(r, c):
    return [
        (r - 1, c), (r + 1, c),
        (r, c - 1), (r, c + 1),
        (r - 1, c - 1), (r - 1, c + 1),
        (r + 1, c - 1), (r + 1, c + 1),
    ]


# ============================================================
# Core assignment logic
# ============================================================

def assign_boxes_to_grid(
    warped_img,
    boxes,
    rows,
    cols,
    draw=True
):
    """
    Parameters
    ----------
    warped_img : np.ndarray
    boxes : list of [x1,y1,x2,y2]
    rows, cols : int
        Logical grid size (inner grid)
    draw : bool
        Whether to draw visualization

    Returns
    -------
    assignments : dict
        grid_cell -> box_index
    box_to_cell : dict
        box_index -> (r,c)
    vis_img : np.ndarray
        visualization
    """

    grid = build_extended_grid(warped_img.shape, rows, cols)

    # --------------------------------------------------------
    # Step 1: compute overlaps (inner cells only)
    # --------------------------------------------------------
    overlaps = defaultdict(list)

    for bi, box in enumerate(boxes):
        for (r, c), rect in grid.items():
            if is_outer_cell(r, c, rows, cols):
                continue

            ia = intersection_area(box, rect)
            if ia > 0:
                overlaps[bi].append(((r, c), ia))

    # --------------------------------------------------------
    # Step 2: primary assignment (max overlap per box)
    # --------------------------------------------------------
    cell_to_box = {}
    box_to_cell = {}

    sorted_boxes = sorted(
        overlaps.keys(),
        key=lambda b: max(v[1] for v in overlaps[b]),
        reverse=True
    )

    for bi in sorted_boxes:
        candidates = sorted(overlaps[bi], key=lambda x: x[1], reverse=True)

        for (r, c), _ in candidates:
            if (r, c) not in cell_to_box:
                cell_to_box[(r, c)] = bi
                box_to_cell[bi] = (r, c)
                break

    # --------------------------------------------------------
    # Step 3: secondary neighbor assignment
    # --------------------------------------------------------
    for bi, candidates in overlaps.items():
        if bi in box_to_cell:
            continue

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        (r0, c0), _ = candidates[0]

        for rn, cn in neighboring_cells(r0, c0):
            if (
                1 <= rn <= rows and
                1 <= cn <= cols and
                (rn, cn) not in cell_to_box
            ):
                cell_to_box[(rn, cn)] = bi
                box_to_cell[bi] = (rn, cn)
                break

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    vis_img = warped_img.copy()

    if draw:
        # draw boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if i in box_to_cell:
                r, c = box_to_cell[i]
                label = f"({r},{c})"
                cv2.putText(
                    vis_img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        # draw grid
        h, w = warped_img.shape[:2]
        row_step = h / (rows + 1)
        col_step = w / (cols + 1)

        for r in range(rows + 2):
            y = int(round(r * row_step))
            cv2.line(vis_img, (0, y), (w, y), (255, 255, 255), 1)

        for c in range(cols + 2):
            x = int(round(c * col_step))
            cv2.line(vis_img, (x, 0), (x, h), (255, 255, 255), 1)

    return cell_to_box, box_to_cell, vis_img

