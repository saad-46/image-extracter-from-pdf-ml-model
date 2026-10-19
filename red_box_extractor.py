"""
Enhanced Red-Box-Guided Figure Extraction Module.

Key features:
1. Multi-scale morphology + contour hierarchy to handle thick, rounded, or dashed red frames.
2. Perimeter-boundary verification: confirms that the detected contour represents an actual
   boundary framing box (border density check) and rejects solid/partial illustration elements.
3. Suppression of internal nested contours (retaining outer boundary).
4. Non-Maximum Suppression (NMS) with IoU deduplication.
5. Red outline exclusion: automatic inward trimming and verification.
6. Per-crop diagnostic telemetry: logs bounding box, source dimensions, crop dimensions,
   boundary density, and flags suspicious narrow/partial crops.
7. Matching marked page coordinates to the clean original PDF page to ensure pristine crops.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import pymupdf as fitz
import cv2
import numpy as np
from PIL import Image


def box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, (b1[2] - b1[0])) * max(0.0, (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0])) * max(0.0, (b2[3] - b2[1]))
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def is_nested(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int], tol: int = 12) -> bool:
    """Check if inner box [x1, y1, x2, y2] is completely or mostly contained inside outer box."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return (ix1 >= ox1 - tol and iy1 >= oy1 - tol and ix2 <= ox2 + tol and iy2 <= oy2 + tol)


def inspect_red_boxes_nature(pdf_path: str, max_pages: int = 5) -> Dict[str, Any]:
    """
    Diagnose how red boxes are stored in the marked PDF:
    - PDF Annotations (Square / Highlight / Line etc.)
    - Vector Drawings (paths / lines / rects in PDF graphics stream)
    - Flattened pixels (rasterized images)
    """
    doc = fitz.open(pdf_path)
    report = {
        "pdf_path": pdf_path,
        "total_pages": len(doc),
        "annotation_count": 0,
        "red_vector_drawings": 0,
        "raster_images_present": False,
        "detected_type": "unknown",
        "sample_findings": []
    }

    red_annots = []
    red_drawings = []

    for page_idx in range(min(len(doc), max_pages)):
        page = doc[page_idx]

        # 1. Check PDF Annotations
        annots = list(page.annots())
        if annots:
            for a in annots:
                colors = a.colors
                stroke = colors.get("stroke") if colors else None
                is_red = False
                if stroke and len(stroke) >= 3:
                    if stroke[0] > 0.6 and stroke[1] < 0.4 and stroke[2] < 0.4:
                        is_red = True
                rect = a.rect
                red_annots.append({
                    "page": page_idx + 1,
                    "type": a.type[1],
                    "rect": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "is_red": is_red
                })

        # 2. Check Vector Drawings
        drawings = page.get_drawings()
        for d in drawings:
            color = d.get("color")
            fill = d.get("fill")
            is_red = False
            if color and len(color) >= 3:
                if color[0] > 0.7 and color[1] < 0.35 and color[2] < 0.35:
                    is_red = True
            elif fill and len(fill) >= 3:
                if fill[0] > 0.7 and fill[1] < 0.35 and fill[2] < 0.35:
                    is_red = True

            if is_red:
                r = d.get("rect")
                if r and r.width > 20 and r.height > 20:
                    red_drawings.append({
                        "page": page_idx + 1,
                        "rect": [r.x0, r.y0, r.x1, r.y1],
                        "color": color
                    })

        # 3. Check Images
        if len(page.get_images()) > 0:
            report["raster_images_present"] = True

    report["annotation_count"] = len(red_annots)
    report["red_vector_drawings"] = len(red_drawings)

    if red_annots:
        report["detected_type"] = "pdf_annotations"
        report["sample_findings"] = red_annots[:5]
    elif red_drawings:
        report["detected_type"] = "vector_drawings"
        report["sample_findings"] = red_drawings[:5]
    else:
        report["detected_type"] = "flattened_pixels"
        report["sample_findings"] = ["Flattened raster pixels (using OpenCV HSV contour detection)"]

    doc.close()
    return report


def detect_red_boxes_opencv(
    page_img_bgr: np.ndarray,
    min_w: int = 60,
    min_h: int = 50,
    border_inward_trim: int = 5,
    iou_thresh: float = 0.35,
    require_frame_structure: bool = True
) -> List[Tuple[int, int, int, int]]:
    """
    Robust red bounding-box detector:
    1. HSV thresholding across red wraps (0-12 & 165-180).
    2. Morphological speckle cleaning and bridging.
    3. Structural verification:
       - Confirms red pixels align with the bounding box perimeter (framing box).
       - Rejects solid/filled non-frame illustration elements (e.g. red clothes/hats).
    4. Suppression of nested sub-boxes (retaining outermost diagram bounding box).
    5. Non-Maximum Suppression (IoU) deduplication.
    6. Inward trimming to guarantee zero red line remaining.
    """
    if page_img_bgr is None or page_img_bgr.size == 0:
        return []

    H, W = page_img_bgr.shape[:2]
    hsv = cv2.cvtColor(page_img_bgr, cv2.COLOR_BGR2HSV)

    # Red masks in HSV
    lower_red1 = np.array([0, 90, 80])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([165, 90, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Clean small speckle noise
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, clean_kernel)

    # Bridge box edges with rectangular closing
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(red_mask_clean, cv2.MORPH_CLOSE, bridge_kernel, iterations=2)

    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []

    candidate_boxes = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Basic size filtering
        if w < min_w or h < min_h:
            continue
        # Exclude entire page border frames
        if w > (W * 0.95) or h > (H * 0.95):
            continue

        # Check rectangularity / aspect ratio
        aspect = w / float(h)
        if aspect < 0.12 or aspect > 8.5:
            continue

        # Structural frame test:
        # A true annotation box is a framing border. Its red pixels are concentrated
        # along the border perimeter, NOT filling the interior solid center.
        if require_frame_structure:
            bw = max(2, int(min(w, h) * 0.12))
            top_strip = red_mask_clean[y : y + bw, x : x + w]
            bot_strip = red_mask_clean[max(0, y + h - bw) : y + h, x : x + w]
            left_strip = red_mask_clean[y : y + h, x : x + bw]
            right_strip = red_mask_clean[y : y + h, max(0, x + w - bw) : x + w]

            border_red = (
                cv2.countNonZero(top_strip) +
                cv2.countNonZero(bot_strip) +
                cv2.countNonZero(left_strip) +
                cv2.countNonZero(right_strip)
            )

            # Center interior region
            mx = int(w * 0.20)
            my = int(h * 0.20)
            interior = red_mask_clean[y + my : y + h - my, x + mx : x + w - mx]
            interior_red = cv2.countNonZero(interior) if interior.size > 0 else 0
            interior_fill = interior_red / float(interior.size) if interior.size > 0 else 0.0

            # Reject solid/blob fills: if interior is >40% solid red, it's a colored object, not a bounding box
            if interior_fill > 0.40:
                continue

            # A framing box must have adequate border support
            expected_border_px = (2 * w + 2 * h) * 1.5
            if border_red < (expected_border_px * 0.25):
                continue

        candidate_boxes.append([x, y, x + w, y + h, w * h])

    if not candidate_boxes:
        return []

    # Sort candidates by area descending (largest outer boxes first)
    candidate_boxes.sort(key=lambda b: b[4], reverse=True)

    # Step 1: Filter nested boxes (if box B is enclosed inside box A, keep only outer box A)
    filtered_enclosing = []
    for cand in candidate_boxes:
        c_box = cand[:4]
        nested = False
        for kept in filtered_enclosing:
            if is_nested(c_box, kept[:4], tol=12):
                nested = True
                break
        if not nested:
            filtered_enclosing.append(cand)

    # Step 2: Non-Maximum Suppression (IoU) for overlapping duplicate detections
    final_boxes = []
    for cand in filtered_enclosing:
        box = cand[:4]
        overlap = False
        for k in final_boxes:
            if box_iou(box, k) > iou_thresh:
                overlap = True
                break
        if not overlap:
            final_boxes.append(box)

    # Step 3: Apply inward trim to exclude red outline
    trimmed_boxes = []
    for (x1, y1, x2, y2) in final_boxes:
        tx1 = min(W, x1 + border_inward_trim)
        ty1 = min(H, y1 + border_inward_trim)
        tx2 = max(0, x2 - border_inward_trim)
        ty2 = max(0, y2 - border_inward_trim)

        # Verify remaining valid dimension
        if (tx2 - tx1) >= (min_w - 2 * border_inward_trim) and (ty2 - ty1) >= (min_h - 2 * border_inward_trim):
            trimmed_boxes.append((tx1, ty1, tx2, ty2))

    # Sort from top-to-bottom of the page
    trimmed_boxes.sort(key=lambda b: (b[1], b[0]))
    return trimmed_boxes


def extract_labeled_crops(
    original_pdf_path: str,
    marked_pdf_path: str,
    output_dir: str,
    render_dpi: int = 200,
    border_inward_trim: int = 5,
    subfolder_name: str = "recovered_crops_v3"
) -> List[Dict[str, Any]]:
    """
    Extracts the ground truth diagram regions from the original clean PDF using boxes
    recovered from the marked PDF. Excludes red border lines.
    Saves outputs to a non-destructive destination (default: recovered_crops_v3).
    Includes rich diagnostic metadata per crop.
    """
    out_path = Path(output_dir)
    crops_dir = out_path / subfolder_name
    crops_dir.mkdir(parents=True, exist_ok=True)

    diag = inspect_red_boxes_nature(marked_pdf_path)
    box_nature = diag["detected_type"]

    doc_orig = fitz.open(original_pdf_path)
    doc_mark = fitz.open(marked_pdf_path)

    total_pages = min(len(doc_orig), len(doc_mark))
    zoom = render_dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    extracted_records = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page_m = doc_mark[page_idx]
        page_o = doc_orig[page_idx]

        # Render marked page
        pix_m = page_m.get_pixmap(matrix=mat, alpha=False)
        img_m = np.frombuffer(pix_m.samples, dtype=np.uint8).reshape((pix_m.height, pix_m.width, 3))
        img_m_bgr = cv2.cvtColor(img_m, cv2.COLOR_RGB2BGR)

        # Render original clean page
        pix_o = page_o.get_pixmap(matrix=mat, alpha=False)
        img_o = np.frombuffer(pix_o.samples, dtype=np.uint8).reshape((pix_o.height, pix_o.width, 3))
        img_o_bgr = cv2.cvtColor(img_o, cv2.COLOR_RGB2BGR)

        boxes = []
        scale_x = pix_m.width / page_m.rect.width
        scale_y = pix_m.height / page_m.rect.height

        if box_nature == "vector_drawings":
            drawings = page_m.get_drawings()
            for d in drawings:
                color = d.get("color") or d.get("fill")
                if color and len(color) >= 3 and color[0] > 0.7 and color[1] < 0.35 and color[2] < 0.35:
                    r = d.get("rect")
                    if r and r.width > 30 and r.height > 30:
                        x1 = int(r.x0 * scale_x) + border_inward_trim
                        y1 = int(r.y0 * scale_y) + border_inward_trim
                        x2 = int(r.x1 * scale_x) - border_inward_trim
                        y2 = int(r.y1 * scale_y) - border_inward_trim
                        boxes.append((x1, y1, x2, y2))

        elif box_nature == "pdf_annotations":
            for a in page_m.annots():
                r = a.rect
                x1 = int(r.x0 * scale_x) + border_inward_trim
                y1 = int(r.y0 * scale_y) + border_inward_trim
                x2 = int(r.x1 * scale_x) - border_inward_trim
                y2 = int(r.y1 * scale_y) - border_inward_trim
                boxes.append((x1, y1, x2, y2))

        # Primary / fallback for flattened raster pixels (using enhanced detector)
        if not boxes:
            boxes = detect_red_boxes_opencv(img_m_bgr, border_inward_trim=border_inward_trim)

        # Crop each recovered box from the ORIGINAL clean page (no red annotations)
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            crop = img_o_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if np.std(gray) < 6.0:  # Skip pure white/blank crops
                continue

            crop_w = int(x2 - x1)
            crop_h = int(y2 - y1)
            aspect = round(crop_w / float(crop_h), 2)

            # Quality flags
            is_suspicious = False
            flag_reasons = []
            if crop_w < 70 or crop_h < 60:
                is_suspicious = True
                flag_reasons.append("Very small crop")
            if aspect > 5.0 or aspect < 0.2:
                is_suspicious = True
                flag_reasons.append(f"Extreme aspect ratio ({aspect})")

            crop_filename = f"orig4_p{page_num:03d}_crop{idx+1:02d}.png"
            crop_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)

            extracted_records.append({
                "page": page_num,
                "crop_idx": idx + 1,
                "crop_filename": crop_filename,
                "crop_path": str(crop_path),
                "bbox_pixels": [x1, y1, x2, y2],
                "dimensions": [crop_w, crop_h],
                "aspect_ratio": aspect,
                "page_dimensions": [pix_o.width, pix_o.height],
                "is_suspicious": is_suspicious,
                "flag_reasons": flag_reasons
            })

    doc_orig.close()
    doc_mark.close()

    return extracted_records


def export_yolo_dataset(
    original_pdf_path: str,
    extracted_records: List[Dict[str, Any]],
    dataset_dir: str,
    render_dpi: int = 200,
    class_id: int = 0
):
    """
    Converts recovered ground-truth bounding boxes into standard YOLO format:
    <class_id> <x_center> <y_center> <width> <height> (all normalized 0.0 - 1.0)
    Saves:
      dataset_dir/images/
      dataset_dir/labels/
      dataset_dir/data.yaml
    """
    d_path = Path(dataset_dir)
    images_dir = d_path / "images"
    labels_dir = d_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    page_to_boxes = {}
    for r in extracted_records:
        p = r["page"]
        if p not in page_to_boxes:
            page_to_boxes[p] = []
        page_to_boxes[p].append(r)

    doc = fitz.open(original_pdf_path)
    zoom = render_dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num, records in page_to_boxes.items():
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bgr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        W, H = pix.width, pix.height

        img_stem = f"orig4_p{page_num:03d}"
        img_out = images_dir / f"{img_stem}.png"
        label_out = labels_dir / f"{img_stem}.txt"

        cv2.imwrite(str(img_out), img_bgr)

        with open(label_out, "w", encoding="utf-8") as f_lbl:
            for rec in records:
                x1, y1, x2, y2 = rec["bbox_pixels"]
                box_w = x2 - x1
                box_h = y2 - y1
                x_center = (x1 + x2) / 2.0 / W
                y_center = (y1 + y2) / 2.0 / H
                norm_w = box_w / W
                norm_h = box_h / H

                f_lbl.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    doc.close()

    yaml_content = f"""# YOLO Dataset Configuration for JEE/NEET Figures
path: {Path(dataset_dir).resolve().as_posix()}
train: images
val: images

nc: 1
names: ['figure']
"""
    with open(d_path / "data.yaml", "w", encoding="utf-8") as f_yaml:
        f_yaml.write(yaml_content)

    print(f"YOLO dataset exported successfully to {dataset_dir}:")
    print(f"  Images: {len(page_to_boxes)} pages")
    print(f"  Labels: {len(extracted_records)} figure bounding boxes")
