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
    x1 = max(box[0], rect[0])
    y1 = max(box[1], rect[1])
    x2 = min(box[2], rect[2])
    y2 = min(box[3], rect[3])
    return max(0, x2 - x1) * max(0, y2 - y1)

# ============================================================
# Grid construction - INNER CELLS ONLY
# ============================================================
def build_inner_grid(img_shape, rows, cols):
    """
    Build a grid where only inner cells are allowed for assignment.
    If rows=3, cols=3 → total cells = 9, but only the center 1x1 (row 1, col 1) is allowed? No.
    
    Standard understanding for "no outer grids":
    - For rows=3 → allow only row index 1 (middle row)
    - For cols=3 → allow only col index 1 (middle column)
    - General: allowed rows = 1 .. rows-2, allowed cols = 1 .. cols-2
    This excludes the outermost border cells on all four sides.
    """
    h, w = img_shape[:2]
    row_step = h / rows
    col_step = w / cols
    
    grid = {}
    allowed_row_range = range(1, rows - 1) if rows > 2 else range(rows)
    allowed_col_range = range(1, cols - 1) if cols > 2 else range(cols)
    
    for r in allowed_row_range:
        for c in allowed_col_range:
            x1 = int(round(c * col_step))
            y1 = int(round(r * row_step))
            x2 = int(round((c + 1) * col_step))
            y2 = int(round((r + 1) * row_step))
            grid[(r, c)] = (x1, y1, x2, y2)
    
    return grid, row_step, col_step, allowed_row_range, allowed_col_range

def neighboring_cells(r, c):
    return [
        (r - 1, c), (r + 1, c),
        (r, c - 1), (r, c + 1),
        (r - 1, c - 1), (r - 1, c + 1),
        (r + 1, c - 1), (r + 1, c + 1),
    ]

def is_valid_inner_cell(r, c, rows, cols):
    if rows <= 2:
        row_ok = 0 <= r < rows
    else:
        row_ok = 1 <= r <= rows - 2
    if cols <= 2:
        col_ok = 0 <= c < cols
    else:
        col_ok = 1 <= c <= cols - 2
    return row_ok and col_ok

# ============================================================
# Core assignment logic - NO OUTER CELLS
# ============================================================
def assign_boxes_to_grid(
    warped_img,
    boxes,
    rows,
    cols,
    draw=True,
    debug=False
):
    """
    STRICT RULE: Boxes are NEVER assigned to outermost border cells.
    - For rows >= 3: allowed rows = 1 to rows-2
    - For cols >= 3: allowed cols = 1 to cols-2
    - If rows <= 2 or cols <= 2, all cells are allowed (no outer to exclude)
    Greedy max-overlap → neighbor fallback within allowed inner cells only.
    Grid lines are still drawn as rows x cols (full division).
    """
    grid, row_step, col_step, allowed_rows, allowed_cols = build_inner_grid(warped_img.shape, rows, cols)

    # --------------------------------------------------------
    # Step 1: Compute overlaps only with allowed (inner) cells
    # --------------------------------------------------------
    overlaps = defaultdict(list)
    for bi, box in enumerate(boxes):
        for (r, c), rect in grid.items():
            ia = intersection_area(box, rect)
            if ia > 0:
                overlaps[bi].append(((r, c), ia))

    # --------------------------------------------------------
    # Step 2: Greedy assignment by max overlap (inner cells only)
    # --------------------------------------------------------
    cell_to_box = {}
    box_to_cell = {}

    sorted_boxes = sorted(
        overlaps.keys(),
        key=lambda b: max((v[1] for v in overlaps[b]), default=0),
        reverse=True
    )

    for bi in sorted_boxes:
        if bi in box_to_cell:
            continue
        for (r, c), _ in sorted(overlaps[bi], key=lambda x: x[1], reverse=True):
            if (r, c) not in cell_to_box:
                cell_to_box[(r, c)] = bi
                box_to_cell[bi] = (r, c)
                break

    # --------------------------------------------------------
    # Step 3: Neighbor fallback - only to allowed inner cells
    # --------------------------------------------------------
    for bi in list(overlaps.keys()):
        if bi in box_to_cell:
            continue
        if not overlaps[bi]:
            continue
        (r0, c0), _ = max(overlaps[bi], key=lambda x: x[1])
        for rn, cn in neighboring_cells(r0, c0):
            if (
                is_valid_inner_cell(rn, cn, rows, cols) and
                (rn, cn) not in cell_to_box
            ):
                cell_to_box[(rn, cn)] = bi
                box_to_cell[bi] = (rn, cn)
                break

    # --------------------------------------------------------
    # Debug assertions
    # --------------------------------------------------------
    if debug:
        for (r, c) in cell_to_box:
            assert is_valid_inner_cell(r, c, rows, cols), \
                f"Assigned to forbidden outer cell: {(r, c)}"

    # --------------------------------------------------------
    # Visualization (full grid lines, but labels only on assigned inner cells)
    # --------------------------------------------------------
    vis_img = warped_img.copy()
    if draw:
        h, w = warped_img.shape[:2]
        # Draw full grid lines (rows x cols divisions)
        for r in range(rows + 1):
            y = int(round(r * row_step))
            cv2.line(vis_img, (0, y), (w, y), (255, 255, 255), 1)
        for c in range(cols + 1):
            x = int(round(c * col_step))
            cv2.line(vis_img, (x, 0), (x, h), (255, 255, 255), 1)

        # Optionally highlight allowed inner region (uncomment if helpful)
        # if rows > 2:
        #     cv2.rectangle(vis_img, (0, int(row_step)), (w, int((rows-1)*row_step)), (100, 100, 100), 2)
        # if cols > 2:
        #     cv2.rectangle(vis_img, (int(col_step), 0), (int((cols-1)*col_step), h), (100, 100, 100), 2)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if i in box_to_cell:
                r, c = box_to_cell[i]
                cv2.putText(
                    vis_img,
                    f"({r},{c})",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    vis_img,
                    "NO",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

    return cell_to_box, box_to_cell, vis_img