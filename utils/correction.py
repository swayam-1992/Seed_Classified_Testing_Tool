import cv2
import numpy as np

LABEL_COLOR = {
    0: (0, 0, 255),     # UG
    1: (0, 255, 0),     # G
    2: (255, 165, 0),   # A
}

def correct_grid_labels(
    warped,
    labels,
    rows,
    cols
):
    """
    Interactive correction of inner-grid labels.

    Args:
        warped : original warped image (clean, no drawings)
        labels : numpy array (len = (rows-2)*(cols-2))
        rows, cols : full grid size (including border)

    Returns:
        corrected_warped : image with corrected grid colors
        corrected_labels : updated labels array
    """

    H, W = warped.shape[:2]
    ch, cw = H // rows, W // cols

    labels = labels.copy()
    vis = warped.copy()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def is_inner_cell(r, c):
        return 0 < r < rows-1 and 0 < c < cols-1

    def rc_to_index(r, c):
        """Map inner grid (r,c) → labels index"""
        return (r - 1) * (cols - 2) + (c - 1)

    def draw_all():
        vis[:] = warped
        idx = 0
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                label = labels[idx]
                color = LABEL_COLOR[int(label)]

                y1, y2 = r * ch, (r + 1) * ch
                x1, x2 = c * cw, (c + 1) * cw

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                idx += 1

    # --------------------------------------------------
    # Mouse callback
    # --------------------------------------------------
    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        c = x // cw
        r = y // ch

        if not is_inner_cell(r, c):
            return

        idx = rc_to_index(r, c)

        # cycle label: UG → G → A → UG
        labels[idx] = (labels[idx] + 1) % 3
        draw_all()

    # --------------------------------------------------
    # Start UI
    # --------------------------------------------------
    draw_all()
    cv2.namedWindow("Correct Grid", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Correct Grid", on_click)

    print("🖱 Click cell to cycle: UG → G → A")
    print("⌨ Press 'q' or ESC to finish")

    while True:
        cv2.imshow("Correct Grid", vis)
        key = cv2.waitKey(20) & 0xFF

        if key in [27, ord("q")]:
            break

    cv2.destroyAllWindows()
    return vis, labels

