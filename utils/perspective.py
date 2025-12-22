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
    Apply perspective correction using 4 inner corner points,
    expanding crop by buffer = (height/rows, width/cols).

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR)
    corners : list or np.ndarray
        4 points inside tray [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        Order: top-left, top-right, bottom-right, bottom-left
    rows : int
        Number of grid rows
    cols : int
        Number of grid columns
    output_size : tuple (w, h), optional
        Final warped size. If None, auto-computed.

    Returns
    -------
    warped : np.ndarray
        Perspective corrected image
    meta : dict
        Metadata (buffers, final corners)
    """

    h, w = img.shape[:2]
    pts = np.array(corners, dtype=np.float32)

    # ----------------------------------
    # Compute cropped size
    # ----------------------------------
    crop_width = np.linalg.norm(pts[1] - pts[0])
    crop_height = np.linalg.norm(pts[3] - pts[0])

    # ----------------------------------
    # Compute buffers
    # ----------------------------------
    buffer_x = crop_width / cols
    buffer_y = crop_height / rows

    # ----------------------------------
    # Expand corners with buffers
    # ----------------------------------
    expanded = np.array([
        [pts[0][0] - buffer_x, pts[0][1] - buffer_y],  # TL
        [pts[1][0] + buffer_x, pts[1][1] - buffer_y],  # TR
        [pts[2][0] + buffer_x, pts[2][1] + buffer_y],  # BR
        [pts[3][0] - buffer_x, pts[3][1] + buffer_y],  # BL
    ], dtype=np.float32)

    # Clip to image bounds
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)

    # ----------------------------------
    # Output size
    # ----------------------------------
    out_w = int(np.linalg.norm(expanded[1] - expanded[0]))
    out_h = int(np.linalg.norm(expanded[3] - expanded[0]))

    if output_size is not None:
        out_w, out_h = output_size

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    # ----------------------------------
    # Perspective transform
    # ----------------------------------
    M = cv2.getPerspectiveTransform(expanded, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h))

    meta = {
        "buffer_x": buffer_x,
        "buffer_y": buffer_y,
        "expanded_corners": expanded.tolist(),
        "output_size": (out_w, out_h)
    }

    return warped, meta

