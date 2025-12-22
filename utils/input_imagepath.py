import os
import cv2

def ask_image_path(prompt="Enter image path: "):
    """
    Ask user to enter an image path from terminal.
    Keeps asking until a valid file path is provided.
    """
    while True:
        image_path = input(prompt).strip()

        if not image_path:
            print("❌ Path cannot be empty. Try again.")
            continue

        if not os.path.isfile(image_path):
            print(f"❌ File not found: {image_path}")
            continue

        return image_path

def load_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"❌ Failed to load image: {image_path}")

    return img


