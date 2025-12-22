import cv2

def rotate_image_interactive(img):
    """
    Ask user for rotation angle (0/90/180/270),
    rotate the image accordingly, and return:
      - rotated image
      - angle used
    """

    valid_angles = {0, 90, 180, 270}

    while True:
        try:
            angle = int(input("Rotate image by (0 / 90 / 180 / 270): ").strip())
        except ValueError:
            print("❌ Please enter a number (0, 90, 180, 270).")
            continue

        if angle not in valid_angles:
            print("❌ Invalid angle. Choose from 0, 90, 180, 270.")
            continue

        break

    if angle == 0:
        rotated = img
    elif angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return rotated, angle

