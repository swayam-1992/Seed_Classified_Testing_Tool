import cv2
import numpy as np


def draw_extended_grid(
    warped_img: np.ndarray,
    rows: int,
    cols: int,
    color=(255, 255, 255),
    thickness=1
) -> np.ndarray:
    """
    Draw a grid on a warped image with (rows + 2) horizontal
    and (cols + 2) vertical lines.

    Parameters
    ----------
    warped_img : np.ndarray
        Input image (H, W, 3)
    rows : int
        Number of logical grid rows
    cols : int
        Number of logical grid columns
    color : tuple
        Grid line color (BGR)
    thickness : int
        Line thickness

    Returns
    -------
    np.ndarray
        Image with grid overlay
    """

    img = warped_img.copy()
    h, w = img.shape[:2]

    # Grid spacing (extended)
    row_step = h / (rows + 1)
    col_step = w / (cols + 1)

    # Horizontal lines
    for i in range(rows + 2):
        y = int(round(i * row_step))
        cv2.line(img, (0, y), (w, y), color, thickness)

    # Vertical lines
    for j in range(cols + 2):
        x = int(round(j * col_step))
        cv2.line(img, (x, 0), (x, h), color, thickness)

    return img

