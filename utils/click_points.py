import cv2
import numpy as np

def click_four_corners(img, window_name="Click 4 inner corners"):
    """
    Interactive utility to click 4 corner points on an image.

    Controls:
      - Left click: add point
      - r : reset points
      - Enter : confirm points
      - Esc : abort

    Returns
    -------
    corners : list of (x, y)
        Ordered as TL, TR, BR, BL
    """

    clone = img.copy()
    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(clone, (x, y), 6, (0, 255, 0), -1)

                if len(points) > 1:
                    cv2.line(clone, points[-2], points[-1], (255, 0, 0), 2)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise RuntimeError("Point selection aborted")

        if key == ord("r"):
            clone = img.copy()
            points = []

        if key == 13 and len(points) == 4:  # ENTER
            break

    cv2.destroyAllWindows()

    return _order_points(points)


def _order_points(pts):
    """
    Order points as TL, TR, BR, BL
    """
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]

