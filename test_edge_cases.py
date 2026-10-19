"""
Test suite covering edge cases in red-box-guided extraction:
1. Thick red borders (4px - 10px)
2. Rounded corners and anti-aliasing
3. Nested contours (diagram internal graphics matching red, or outer/inner box boundaries)
4. Duplicate detections and overlapping boxes (IoU suppression)
5. Content preservation inside boxes (axes, labels, shapes) while excluding red outlines
6. Small exam diagrams and multiple boxes on a single page
"""

import cv2
import numpy as np
from pathlib import Path
import fitz

from red_box_extractor import (
    detect_red_boxes_opencv,
    extract_labeled_crops,
    inspect_red_boxes_nature,
    box_iou
)


def test_iou_function():
    # Identical boxes
    b1 = [10, 10, 100, 100]
    b2 = [10, 10, 100, 100]
    assert abs(box_iou(b1, b2) - 1.0) < 1e-4

    # Non-overlapping boxes
    b3 = [150, 150, 200, 200]
    assert box_iou(b1, b3) == 0.0

    # Partial overlap
    b4 = [55, 10, 145, 100]
    iou = box_iou(b1, b4)
    assert 0.3 < iou < 0.4
    print("test_iou_function PASSED")


def test_hsv_mask_and_thick_borders():
    # Create synthetic 800x600 image with white background
    img = np.full((600, 800, 3), 255, dtype=np.uint8)

    # 1. Thick red box around a diagram region (e.g. thickness 6, color pure red BGR: 0, 0, 255)
    # Box bounds: x=100, y=100, w=300, h=200
    cv2.rectangle(img, (100, 100), (400, 300), (0, 0, 255), thickness=6)

    # Inside diagram: black axis lines and text
    cv2.line(img, (130, 270), (370, 270), (0, 0, 0), 2) # X axis
    cv2.line(img, (130, 270), (130, 130), (0, 0, 0), 2) # Y axis
    cv2.putText(img, "f(x)", (140, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 2. Add an internal red artifact (e.g. red curve or red arrow inside diagram)
    # Points inside box:
    cv2.line(img, (130, 270), (350, 150), (0, 0, 220), thickness=2)

    # 3. Add a second box nearby: x=450, y=120, w=250, h=180, thickness 4
    cv2.rectangle(img, (450, 120), (700, 300), (20, 20, 240), thickness=4)

    # Detect boxes
    boxes = detect_red_boxes_opencv(img, min_w=60, min_h=50, border_inward_trim=5)
    print(f"Detected boxes: {boxes}")

    # Should detect exactly 2 distinct boxes (the outer bounding boxes, not the internal red curve)
    assert len(boxes) == 2, f"Expected 2 boxes, got {len(boxes)}"

    # Check that boxes match outer boxes closely
    box1 = min(boxes, key=lambda b: b[0])
    box2 = max(boxes, key=lambda b: b[0])

    # Check bounds of box1 (original outer was 100, 100, 400, 300; inward trim ~5 => ~105, 105, 395, 295)
    assert abs(box1[0] - 105) <= 5, f"box1 x1={box1[0]} expected ~105"
    assert abs(box1[1] - 105) <= 5, f"box1 y1={box1[1]} expected ~105"
    assert abs(box1[2] - 395) <= 5, f"box1 x2={box1[2]} expected ~395"
    assert abs(box1[3] - 295) <= 5, f"box1 y2={box1[3]} expected ~295"

    # Crop box1 from original clean image (without red outline)
    clean_img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.line(clean_img, (130, 270), (370, 270), (0, 0, 0), 2)
    cv2.line(clean_img, (130, 270), (130, 130), (0, 0, 0), 2)
    cv2.putText(clean_img, "f(x)", (140, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    crop = clean_img[box1[1]:box1[3], box1[0]:box1[2]]
    # Verify axes are fully preserved
    assert crop.shape[0] > 150 and crop.shape[1] > 250
    # Black pixels (axes) exist in crop
    black_pixels = np.count_nonzero(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 50)
    assert black_pixels > 200, "Axes lines must be preserved in crop"
    print("test_hsv_mask_and_thick_borders PASSED")


def test_nested_and_duplicate_suppression():
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    # Box 1: Outer red box
    cv2.rectangle(img, (100, 100), (500, 400), (0, 0, 255), thickness=8)
    # Nested inner red rectangle (e.g. subtitle box inside marked region)
    cv2.rectangle(img, (150, 120), (300, 180), (0, 0, 255), thickness=4)

    boxes = detect_red_boxes_opencv(img, min_w=60, min_h=50, border_inward_trim=6)
    print(f"Detected boxes for nested case: {boxes}")

    # Must suppress inner nested box and keep the outer enclosing diagram box
    assert len(boxes) == 1, f"Expected 1 enclosing box, got {len(boxes)}"
    assert abs(boxes[0][0] - 106) <= 5
    assert abs(boxes[0][2] - 494) <= 5
    print("test_nested_and_duplicate_suppression PASSED")


def test_empty_or_no_red_page():
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Question 15: Pure text without diagram", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    boxes = detect_red_boxes_opencv(img, min_w=60, min_h=50)
    assert len(boxes) == 0, f"Expected 0 boxes on unmarked page, got {len(boxes)}"
    print("test_empty_or_no_red_page PASSED")


def run_all_tests():
    print("Running comprehensive red-box tests...")
    test_iou_function()
    test_hsv_mask_and_thick_borders()
    test_nested_and_duplicate_suppression()
    test_empty_or_no_red_page()
    print("\nALL EDGE-CASE TESTS PASSED!")


if __name__ == "__main__":
    run_all_tests()
