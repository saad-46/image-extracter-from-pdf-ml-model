"""
Synthetic verification test for Red Box diagnostic, recovery, and YOLO dataset preparation.
Simulates:
- Clean original PDF with diagram and surrounding question text.
- Marked PDF with red bounding box (both vector drawing and rasterized).
- Runs diagnostic, crops from original PDF without red line, verifies YOLO labels.
"""

import cv2
from pathlib import Path
import fitz
from PIL import Image, ImageDraw
import numpy as np
from red_box_extractor import inspect_red_boxes_nature, extract_labeled_crops, export_yolo_dataset

def generate_synthetic_test_pair(test_dir: Path):
    orig_path = test_dir / "Copy of original (4).pdf"
    mark_path = test_dir / "Copy of original (4) red boxes marked.pdf"

    # 1. Create Original Clean PDF
    doc_orig = fitz.open()
    page1 = doc_orig.new_page(width=595, height=842)
    page1.insert_text((50, 50), "JEE Advanced Exam - Test Paper", fontsize=14)
    page1.insert_text((50, 80), "Q. Study the following ray optics lens system:", fontsize=11)

    # Draw synthetic optics diagram (vector shapes)
    shape = page1.new_shape()
    shape.draw_line(fitz.Point(100, 200), fitz.Point(400, 200)) # Principal axis
    shape.draw_oval(fitz.Rect(230, 140, 270, 260))             # Lens
    shape.finish(width=2, color=(0, 0, 0))
    shape.commit()
    page1.insert_text((100, 280), "Find the focal length of the convex lens.", fontsize=11)
    doc_orig.save(str(orig_path))
    doc_orig.close()

    # 2. Create Marked PDF with red box around the diagram
    doc_mark = fitz.open(str(orig_path))
    page_m = doc_mark[0]
    shape_m = page_m.new_shape()
    # Draw red rectangle around the diagram region (x0=90, y0=130, x1=410, y1=270)
    shape_m.draw_rect(fitz.Rect(90, 130, 410, 270))
    shape_m.finish(width=2, color=(1.0, 0.0, 0.0))  # Pure Red
    shape_m.commit()
    doc_mark.save(str(mark_path))
    doc_mark.close()

    return orig_path, mark_path

def main():
    import shutil
    test_dir = Path("synthetic_redbox_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    orig_pdf, mark_pdf = generate_synthetic_test_pair(test_dir)
    print(f"Generated test PDFs:\n  Original: {orig_pdf}\n  Marked: {mark_pdf}")

    # Stage 1: Diagnostic
    print("\n--- Running Stage 1: Diagnostic ---")
    diag = inspect_red_boxes_nature(str(mark_pdf))
    print("Diagnostic Result:", diag)
    assert diag["detected_type"] in ["vector_drawings", "pdf_annotations", "flattened_pixels"]

    # Stage 2 & 3: Extraction of labeled crops from original PDF
    print("\n--- Running Stage 2 & 3: Crop Extraction & Border Exclusion ---")
    crops_output = test_dir / "crops_output"
    records = extract_labeled_crops(
        original_pdf_path=str(orig_pdf),
        marked_pdf_path=str(mark_pdf),
        output_dir=str(crops_output),
        render_dpi=200,
        border_inward_trim=4
    )
    print(f"Recovered {len(records)} crop records.")
    assert len(records) >= 1, "Failed to recover crops!"
    crop_file = Path(records[0]["crop_path"])
    assert crop_file.exists(), f"Crop file missing: {crop_file}"

    # Verify that the extracted crop does NOT contain the red border
    crop_img = cv2.imread(str(crop_file))
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
    red_pixels = cv2.countNonZero(cv2.bitwise_or(mask1, mask2))
    print(f"Red pixels detected in crop: {red_pixels} (expected near 0)")
    assert red_pixels < 25, "Red border was not excluded from the crop!"

    # Stage 5: Export YOLO dataset
    print("\n--- Running Stage 5: YOLO Dataset Export ---")
    yolo_dir = test_dir / "yolo_dataset"
    export_yolo_dataset(
        original_pdf_path=str(orig_pdf),
        extracted_records=records,
        dataset_dir=str(yolo_dir),
        render_dpi=200
    )

    label_files = list((yolo_dir / "labels").glob("*.txt"))
    image_files = list((yolo_dir / "images").glob("*.png"))
    data_yaml = yolo_dir / "data.yaml"

    assert len(label_files) == 1, "Expected 1 label file"
    assert len(image_files) == 1, "Expected 1 image file"
    assert data_yaml.exists(), "data.yaml missing"

    with open(label_files[0], "r") as f:
        lbl_content = f.read().strip()
    print(f"Sample YOLO label line: {lbl_content}")
    parts = [float(x) for x in lbl_content.split()]
    assert len(parts) == 5, "YOLO format must have 5 values: class, x_center, y_center, w, h"
    assert parts[0] == 0.0, "Class ID should be 0"
    for v in parts[1:]:
        assert 0.0 <= v <= 1.0, f"Value out of bounds: {v}"

    print("\nAll synthetic tests PASSED successfully!")

if __name__ == "__main__":
    main()
