import cv2
import numpy as np

def perspective_correction_with_buffer(
    img,
    corners,
    rows,
    cols,
    output_size=None
):
    """
    Apply perspective correction using 4 INNER grid corner points,
    expanding exactly by one grid cell on all sides.

    This version uses exact grid geometry (no heuristic buffers).

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR)
    corners : list or np.ndarray
        4 INNER corner points [(x,y), ...]
        Order: TL, TR, BR, BL
    rows : int
        Number of INNER grid rows
    cols : int
        Number of INNER grid columns
    output_size : tuple (w, h), optional
        Override computed output size

    Returns
    -------
    warped : np.ndarray
        Perspective corrected image
    meta : dict
        Metadata (cell size, grid size, inner dst corners)
    """

    src_inner = np.array(corners, dtype=np.float32)

    # --------------------------------------------------
    # Robust cell size estimation (exact logic)
    # --------------------------------------------------
    width1  = np.linalg.norm(src_inner[1] - src_inner[0])
    width2  = np.linalg.norm(src_inner[2] - src_inner[3])
    height1 = np.linalg.norm(src_inner[3] - src_inner[0])
    height2 = np.linalg.norm(src_inner[2] - src_inner[1])

    cell_w = (width1 + width2) / 2.0 / cols
    cell_h = (height1 + height2) / 2.0 / rows

    # --------------------------------------------------
    # Add EXACT one-cell buffer around inner grid
    # --------------------------------------------------
    buffer_rows = 1
    buffer_cols = 1

    total_rows = rows + 2 * buffer_rows
    total_cols = cols + 2 * buffer_cols

    out_w = int(total_cols * cell_w)
    out_h = int(total_rows * cell_h)

    if output_size is not None:
        out_w, out_h = output_size

    # --------------------------------------------------
    # Destination points for INNER grid
    # --------------------------------------------------
    offset_x = buffer_cols * cell_w
    offset_y = buffer_rows * cell_h

    dst_inner = np.array([
        [offset_x,               offset_y],
        [out_w - 1 - offset_x,   offset_y],
        [out_w - 1 - offset_x,   out_h - 1 - offset_y],
        [offset_x,               out_h - 1 - offset_y]
    ], dtype=np.float32)

    # --------------------------------------------------
    # Perspective warp
    # --------------------------------------------------
    M = cv2.getPerspectiveTransform(src_inner, dst_inner)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))

    meta = {
        "cell_size": (cell_w, cell_h),
        "inner_grid": (rows, cols),
        "output_grid": (total_rows, total_cols),
        "output_size": (out_w, out_h),
        "inner_dst_corners": dst_inner.tolist()
    }

    return warped, meta
