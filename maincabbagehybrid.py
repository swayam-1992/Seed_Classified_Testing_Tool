from utils.input_imagepath import ask_image_path, load_image
from utils.rotate import rotate_image_interactive
from utils.click_points import click_four_corners
from utils.perspective_improved import perspective_correction_with_buffer

from utils.run_manager import init_temp_run, finalize_run
from utils.artifact_writer import save_image, save_json

import cv2
import os


# --------------------------------------------------
# Main workflow
# --------------------------------------------------
def main():

    # -----------------------------
    # Init temp run (SAFE)
    # -----------------------------
    temp_dir = init_temp_run()

    try:
        # -----------------------------
        # Step 1: Load image
        # -----------------------------
        image_path = ask_image_path()
        print(f"✅ Image selected: {image_path}")

        img = load_image(image_path)
        save_image(temp_dir, "I1_original.jpg", img)

        cv2.imshow("Original Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # -----------------------------
        # Step 2: Rotate image
        # -----------------------------
        rotated_img, angle = rotate_image_interactive(img)
        print(f"✅ Image rotated by {angle} degrees")

        cv2.imshow(f"Rotated Image ({angle}°)", rotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # -----------------------------
        # Step 3: Grid parameters
        # -----------------------------
        rows = 14
        cols = 7

        # -----------------------------
        # Step 4: Click inner corners
        # -----------------------------
        print("\n👉 Click 4 INNER tray corners (TL → TR → BR → BL)")
        corners = click_four_corners(rotated_img)

        # -----------------------------
        # Step 5: Perspective correction
        # -----------------------------
        warped, _ = perspective_correction_with_buffer(img,corners,
        rows,cols)

        save_image(temp_dir, "I2_warped.jpg", warped)

        cv2.imshow("Warped + Buffered", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # -----------------------------
        # YOLO detection
        # -----------------------------
        from utils.yolo_object import run_yolo_objects
        boxes, vis = run_yolo_objects(
            model_path="cabbage_models/best.pt",
            img=warped,
            conf_thr=0.30,
            green_thr=0.02,
            iou_thr=0.5,
            show=True
        )
        
        # -----------------------------
        # Grid visualization
        # -----------------------------
        from utils.grid_visualization import draw_extended_grid
        grid_img = draw_extended_grid(
            warped,
            rows=rows + 1,
            cols=cols + 1,
            color=(255, 255, 255),
            thickness=1
        )

        cv2.imshow("With Grid", grid_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # -----------------------------
        # Assign boxes to grid
        # -----------------------------
        from utils.box_grid_assignment_cab import assign_boxes_to_grid
        cell_to_box, box_to_cell, vis = assign_boxes_to_grid(
            warped_img=warped,
            boxes=boxes,
            rows=rows + 2,
            cols=cols + 2
        )

        save_image(temp_dir, "I3_box_grid.jpg", vis)

        cv2.imshow("With box + grid", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # -----------------------------
        # Shift boxes to grid centers
        # -----------------------------
        from utils.shifting import shift_boxes_to_grid_centers
        shifted = shift_boxes_to_grid_centers(
            warped_original=warped,
            boxes=boxes,
            cell_to_box=cell_to_box,
            rows=rows + 1,
            cols=cols + 1
        )

        save_image(temp_dir, "I4_shifted.jpg", shifted)

        cv2.imshow("Shifted Grid", shifted)
        cv2.waitKey(0)

        # -----------------------------
        # Classification
        # -----------------------------
        from utils.classification_hybrid_processor import classify_shifted_image
        from tensorflow.keras.models import load_model

        cnn_model = load_model("cabbage_models/cabbage_3class_hybrid_germination_cnn.h5", compile=False)

        vis_shifted, vis_warped, vis_stiched, labels = classify_shifted_image(
            shifted=shifted,
            warped=warped,
            cnn_model=cnn_model,
            rows=rows + 2,
            cols=cols + 2
        )

        save_image(temp_dir, "I5_shifted_binary.jpg", vis_stiched)

        save_image(temp_dir, "I6_shifted_AIclassified.jpg", vis_shifted)
        save_image(temp_dir, "I7_warped_AIclassified.jpg", vis_warped)

        
        save_json(temp_dir, "labels_auto.json", labels)

        cv2.imshow("3-class classification (shifted)", vis_shifted)
        cv2.waitKey(0)

        # -----------------------------
        # Manual correction
        # -----------------------------
        from utils.correction import correct_grid_labels

        corrected_warped, corrected_labels = correct_grid_labels(
            warped=warped,
            labels=labels,
            rows=rows + 2,
            cols=cols + 2
        )

        save_json(temp_dir, "labels_corrected.json", corrected_labels)

        save_image(temp_dir, "I8_warped_Manualclassified.jpg", corrected_warped)

        cv2.imshow("Corrected", corrected_warped)
        cv2.waitKey(0)

        # -----------------------------
        # Save metadata (ONCE)
        # -----------------------------
        meta = {
            "image_path": image_path,
            "rotation_angle": angle,
            "clicked_corners": corners,
            "grid": {
                "rows": rows,
                "cols": cols,
                "extended_rows": rows + 2,
                "extended_cols": cols + 2
            },
            "models": {
                "yolo": "best.pt",
                "cnn": "HP_3class_germination_cnn.h5"
            }
        }

        save_json(temp_dir, "meta.json", meta)

        # -----------------------------
        # Finalize run (MOVE temp → runs/)
        # -----------------------------
        finalize_run()

    except Exception as e:
        print("\n❌ ERROR occurred — temp_run preserved for debugging")
        print(e)
        raise


# --------------------------------------------------
if __name__ == "__main__":
    main()

