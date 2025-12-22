import os
import json
import cv2
import numpy as np


# ============================================================
# Image saving
# ============================================================

def save_image(out_dir, filename, img):
    """
    Save an image to disk, ensuring directory exists.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, img)


# ============================================================
# JSON-safe conversion
# ============================================================

def _to_json_safe(obj):
    """
    Recursively convert numpy types to Python-native types
    so they can be serialized by json.dump().
    """
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    return obj


# ============================================================
# JSON saving
# ============================================================

def save_json(out_dir, filename, data):
    """
    Save JSON data safely (handles numpy objects).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    try:
        safe_data = _to_json_safe(data)
        with open(path, "w") as f:
            json.dump(safe_data, f, indent=2)

    except Exception as e:
        print(f"❌ Failed to save JSON: {path}")
        raise e
