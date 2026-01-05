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


def assign_boxes_to_grid(
    warped_img,
    boxes,
    rows,
    cols,
    draw=True,
    debug=False
):
    grid, row_step, col_step, allowed_rows, allowed_cols = build_inner_grid(
        warped_img.shape, rows, cols
    )

    # --------------------------------------------------------
    # Step 1: compute overlaps (inner cells only)
    # --------------------------------------------------------
    overlaps = defaultdict(list)
    for bi, box in enumerate(boxes):
        for (r, c), rect in grid.items():
            ia = intersection_area(box, rect)
            if ia > 0:
                overlaps[bi].append(((r, c), ia))

    # --------------------------------------------------------
    # Step 2: FIRST PASS — assign each box to its best-overlap cell
    # (many boxes per cell allowed)
    # --------------------------------------------------------
    grid_to_boxes = defaultdict(list)
    box_best_cell = {}

    for bi, ols in overlaps.items():
        if not ols:
            continue
        (r, c), best_area = max(ols, key=lambda x: x[1])
        box_best_cell[bi] = (r, c)
        grid_to_boxes[(r, c)].append((bi, best_area))

    # --------------------------------------------------------
    # Step 3: resolve conflicts using outer-overlap priority
    # --------------------------------------------------------
    cell_to_box = {}
    box_to_cell = {}

    for (r, c), box_list in grid_to_boxes.items():

        # ----------------------------------------------------
        # Identify boxes touching OUTER grid cells
        # ----------------------------------------------------
        outer_touching = []

        for bi, _ in box_list:
            for rr in range(rows):
                for cc in range(cols):
                    if not is_valid_inner_cell(rr, cc, rows, cols):
                        rect = (
                            int(cc * col_step),
                            int(rr * row_step),
                            int((cc + 1) * col_step),
                            int((rr + 1) * row_step),
                        )
                        if intersection_area(boxes[bi], rect) > 0:
                            outer_touching.append(bi)
                            break
                else:
                    continue
                break

        # ----------------------------------------------------
        # Choose anchor box
        # ----------------------------------------------------
        if outer_touching:
            # If multiple touch outer, choose the one with max overlap to this grid
            anchor = max(
                outer_touching,
                key=lambda bi: max(
                    ia for (rc, ia) in overlaps[bi] if rc == (r, c)
                )
            )
        else:
            # fallback: highest overlap in this grid
            anchor = max(
                box_list,
                key=lambda x: x[1]
            )[0]

        # Assign anchor
        cell_to_box[(r, c)] = anchor
        box_to_cell[anchor] = (r, c)

        # ----------------------------------------------------
        # Disperse remaining boxes
        # ----------------------------------------------------
        for bi, _ in box_list:
            if bi == anchor:
                continue

            best_neighbor = None
            best_score = 0

            for rn, cn in neighboring_cells(r, c):
                if not is_valid_inner_cell(rn, cn, rows, cols):
                    continue
                if (rn, cn) in cell_to_box:
                    continue

                rect = grid.get((rn, cn))
                if rect is None:
                    continue

                ia = intersection_area(boxes[bi], rect)
                if ia > best_score:
                    best_score = ia
                    best_neighbor = (rn, cn)

            if best_neighbor is not None:
                cell_to_box[best_neighbor] = bi
                box_to_cell[bi] = best_neighbor
            # else: remains unassigned
    
        
        
        
   
    # --------------------------------------------------------
    # Debug safety
    # --------------------------------------------------------
    if debug:
        for (r, c) in cell_to_box:
            assert is_valid_inner_cell(r, c, rows, cols), \
                f"Assigned to forbidden outer cell: {(r, c)}"

    # --------------------------------------------------------
    # Visualization (UNCHANGED)
    # --------------------------------------------------------
    vis_img = warped_img.copy()
    if draw:
        h, w = warped_img.shape[:2]
        for r in range(rows + 1):
            y = int(round(r * row_step))
            cv2.line(vis_img, (0, y), (w, y), (255, 255, 255), 1)
        for c in range(cols + 1):
            x = int(round(c * col_step))
            cv2.line(vis_img, (x, 0), (x, h), (255, 255, 255), 1)

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