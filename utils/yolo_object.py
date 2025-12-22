import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION (defaults, can be overridden)
# ============================================================
CONF_THR = 0.30
GREEN_THR = 0.02
IOU_THR = 0.50


# ============================================================
# GEOMETRY HELPERS
# ============================================================
def intersection_area(b1, b2):
    return (
        max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) *
        max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    )


def iou(b1, b2):
    inter = intersection_area(b1, b2)
    if inter == 0:
        return 0.0

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


# ============================================================
# COLOR FILTER (YOUR LOGIC, UNCHANGED)
# ============================================================
def has_enough_green(img, box, green_thr=GREEN_THR):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

    green_ratio = np.sum(mask > 0) / (crop.size // 3 + 1e-6)
    return green_ratio >= green_thr


# ============================================================
# OVERLAP SUPPRESSION (NMS)
# ============================================================
def suppress_overlaps(boxes, scores, iou_thr=IOU_THR):
    """
    Simple IoU-based NMS.
    Keeps highest-score box when overlap exceeds threshold.
    """
    if len(boxes) == 0:
        return []

    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        idxs = [
            j for j in idxs[1:]
            if iou(boxes[i], boxes[j]) < iou_thr
        ]

    return [boxes[i] for i in keep]


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_yolo_objects(
    model_path,
    img,
    conf_thr=CONF_THR,
    green_thr=GREEN_THR,
    iou_thr=IOU_THR,
    show=True
):
    """
    Run YOLO, apply ALL retain/reject logic, return final boxes.

    Retain conditions:
      ✔ YOLO confidence >= conf_thr
      ✔ Enough green pixels
      ✔ Survives overlap suppression (IoU)

    Parameters
    ----------
    model_path : str
        Path to YOLO model (.pt)
    img : np.ndarray
        Input image (BGR)
    conf_thr : float
        YOLO confidence threshold
    green_thr : float
        Green pixel ratio threshold
    iou_thr : float
        IoU threshold for overlap suppression
    show : bool
        Show visualization

    Returns
    -------
    final_boxes : list[(x1,y1,x2,y2)]
    vis_img : np.ndarray
    """

    model = YOLO(model_path)
    res = model(img)[0]

    raw_boxes = []
    scores = []

    # -------------------------------
    # 1. Confidence + green filtering
    # -------------------------------
    for b in res.boxes:
        conf = float(b.conf)
        if conf < conf_thr:
            continue

        box = tuple(map(int, b.xyxy[0].cpu().numpy()))

        if not has_enough_green(img, box, green_thr):
            continue

        raw_boxes.append(box)
        scores.append(conf)

    # -------------------------------
    # 2. Overlap suppression
    # -------------------------------
    final_boxes = suppress_overlaps(raw_boxes, scores, iou_thr)

    # -------------------------------
    # 3. Visualization
    # -------------------------------
    vis_img = img.copy()

    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if show:
        cv2.imshow("YOLO Objects (Filtered)", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_boxes, vis_img

