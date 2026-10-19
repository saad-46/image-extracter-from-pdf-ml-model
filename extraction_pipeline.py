"""
JEE/NEET PDF Image & Diagram Extraction Pipeline
Combines:
1. Native embedded image extraction (PyMuPDF) with format preservation
2. Pretrained Layout Detection (DocLayNet YOLO) for rendered pages
3. Vector graphic & drawing cluster detection (PyMuPDF drawings)
4. Duplicate / overlap suppression and border cleaning
"""

import os
import sys
import glob
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

try:
    import torch
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
except ImportError:
    torch = None
    YOLO = None
    hf_hub_download = None


def compute_phash(img: Image.Image, hash_size: int = 8) -> int:
    """Compute 64-bit perceptual hash (pHash) for image deduplication."""
    # Resize to hash_size * 4 for DCT
    N = hash_size * 4
    gray = img.convert('L').resize((N, N), Image.Resampling.LANCZOS)
    pixels = np.array(gray, dtype=np.float32)
    # 2D DCT
    dct = cv2.dct(pixels)
    # Extract top-left low frequencies (excluding DC term at [0,0])
    sub_dct = dct[0:hash_size, 0:hash_size]
    med = np.median(sub_dct)
    diff = sub_dct > med
    # Flatten to integer bitmask
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')


def is_near_blank(img_np: np.ndarray, std_thresh: float = 8.0, white_ratio_thresh: float = 0.985) -> bool:
    """Check if an image/crop is virtually blank, uniform, or pure white."""
    if img_np is None or img_np.size == 0:
        return True
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np
    
    std = float(np.std(gray))
    if std < std_thresh:
        return True
    
    white_pixels = np.count_nonzero(gray > 248)
    if (white_pixels / gray.size) > white_ratio_thresh:
        return True
    return False


def trim_borders(img_np: np.ndarray, tol: int = 245, pad: int = 8) -> np.ndarray:
    """Auto-trim uniform white or near-white borders around diagrams."""
    if img_np is None or img_np.size == 0:
        return img_np
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np

    mask = gray < tol
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return img_np
    
    x, y, w, h = cv2.boundingRect(coords)
    H, W = img_np.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    if (x2 - x1) < 15 or (y2 - y1) < 15:
        return img_np
    return img_np[y1:y2, x1:x2]


def box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return inter / union


class LayoutDetector:
    """DocLayNet YOLO layout detector wrapper."""
    def __init__(self, model_repo: str = "hantian/yolo-doclaynet", model_filename: str = "yolov8m-doclaynet.pt", device: str = "auto"):
        self.device = device
        if device == "auto":
            if torch and torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        
        print(f"Loading DocLayNet YOLO model from {model_repo}/{model_filename} on {self.device}...")
        try:
            model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
        except Exception as e:
            print(f"Warning: Failed downloading {model_filename}, falling back to yolov8n-doclaynet.pt: {e}")
            model_path = hf_hub_download(repo_id=model_repo, filename="yolov8n-doclaynet.pt")

        self.model = YOLO(model_path)
        self.classes = self.model.names
        print(f"Model loaded successfully. Recognized classes: {self.classes}")

        # Target class indices: Picture (6) and optionally Formula (2) / Table (8)
        self.target_indices = []
        for idx, name in self.classes.items():
            if name.lower() in ["picture", "figure", "formula", "diagram"]:
                self.target_indices.append(idx)
        print(f"Targeting layout classes: {[self.classes[i] for i in self.target_indices]}")

    def detect_figures(self, img_pil: Image.Image, conf_thresh: float = 0.25, imgsz: int = 1024) -> List[Dict[str, Any]]:
        """Run inference on page image and return bounding boxes."""
        results = self.model.predict(
            source=img_pil,
            imgsz=imgsz,
            conf=conf_thresh,
            device=self.device,
            verbose=False
        )
        detections = []
        if not results:
            return detections
        
        boxes = results[0].boxes
        if boxes is None:
            return detections
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id in self.target_indices:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.classes[cls_id],
                    "conf": round(conf, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })
        return detections


class PDFImageExtractor:
    """
    Complete hybrid extractor:
    - Native embedded objects
    - DocLayNet YOLO visual figure detection
    - Vector drawings clustering
    - Deduplication & border cleaning
    """
    def __init__(
        self,
        output_dir: str,
        detector: Optional[LayoutDetector] = None,
        render_dpi: int = 200,
        min_width: int = 40,
        min_height: int = 40,
        enable_yolo: bool = True,
        enable_drawings: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.detector = detector
        self.render_dpi = render_dpi
        self.min_width = min_width
        self.min_height = min_height
        self.enable_yolo = enable_yolo
        self.enable_drawings = enable_drawings

        self.embedded_dir = self.output_dir / "embedded_images"
        self.figures_dir = self.output_dir / "cropped_figures"
        self.manifest_path = self.output_dir / "extraction_manifest.jsonl"

        self.embedded_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.processed_pdfs = self._load_processed_history()

    def _load_processed_history(self) -> set:
        """Find PDFs that were already completed to support resume."""
        completed = set()
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            completed.add(record.get("pdf_name"))
            except Exception:
                pass
        return completed

    def _append_manifest(self, record: Dict[str, Any]):
        with open(self.manifest_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")

    def extract_native_images(self, doc: fitz.Document, pdf_stem: str) -> List[Dict[str, Any]]:
        """Extract embedded image streams directly using PyMuPDF."""
        out_records = []
        pdf_out_dir = self.embedded_dir / pdf_stem
        seen_hashes = set()

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                if not base_image:
                    continue

                image_bytes = base_image.get("image")
                image_ext = base_image.get("ext", "png")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Filter out tiny icon / decoration artefacts
                if width < self.min_width or height < self.min_height:
                    continue

                # Filter full page background scans if requested
                page_rect = page.rect
                # If image covers virtually the entire page and there is only 1 image, it might be scanned page
                is_full_page = (width >= 1000 and height >= 1400)

                # Duplicate filter by content hash
                h = hashlib.md5(image_bytes).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                # Verify image validity with PIL
                try:
                    import io
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    # Quick blank check
                    np_img = np.array(pil_img)
                    if is_near_blank(np_img):
                        continue
                except Exception:
                    continue

                pdf_out_dir.mkdir(parents=True, exist_ok=True)
                out_filename = f"{pdf_stem}_p{page_num:03d}_xref{xref}.{image_ext}"
                out_path = pdf_out_dir / out_filename

                if not out_path.exists():
                    with open(out_path, "wb") as f_out:
                        f_out.write(image_bytes)

                rec = {
                    "pdf_name": pdf_stem,
                    "page": page_num,
                    "type": "native_embedded",
                    "xref": xref,
                    "file_path": str(out_path.relative_to(self.output_dir)),
                    "dimensions": [width, height],
                    "bbox": None,
                    "is_full_page_scan": is_full_page
                }
                out_records.append(rec)
                self._append_manifest(rec)

        return out_records

    def extract_vector_drawings(self, page: fitz.Page, page_pix: fitz.Pixmap, pdf_stem: str, page_num: int, pdf_out_dir: Path) -> List[Dict[str, Any]]:
        """Cluster PyMuPDF vector drawing paths (useful for circuit diagrams, graphs, geometry)."""
        drawings = page.get_drawings()
        if not drawings:
            return []

        page_w = page.rect.width
        page_h = page.rect.height
        pix_w = page_pix.width
        pix_h = page_pix.height
        scale_x = pix_w / page_w
        scale_y = pix_h / page_h

        # Group drawing rects that are close together
        boxes = []
        for d in drawings:
            r = d.get("rect")
            if not r:
                continue
            # Filter out full-page border outlines or tiny line artifacts
            if r.width >= (page_w * 0.95) and r.height < 5:
                continue
            if r.height >= (page_h * 0.95) and r.width < 5:
                continue
            if r.width < 5 and r.height < 5:
                continue
            boxes.append([r.x0, r.y0, r.x1, r.y1])

        if not boxes:
            return []

        # Merge overlapping / adjacent bounding boxes
        merged = []
        margin = 35  # point tolerance for merging diagram components
        for b in boxes:
            placed = False
            for m in merged:
                # Check overlap with margin
                if not (b[0] > m[2] + margin or b[2] < m[0] - margin or
                        b[1] > m[3] + margin or b[3] < m[1] - margin):
                    m[0] = min(m[0], b[0])
                    m[1] = min(m[1], b[1])
                    m[2] = max(m[2], b[2])
                    m[3] = max(m[3], b[3])
                    placed = True
                    break
            if not placed:
                merged.append(list(b))

        # Filter meaningful clusters
        clusters = []
        for idx, m in enumerate(merged):
            w = m[2] - m[0]
            h = m[3] - m[1]
            # Must be substantial enough to be a diagram, but not entire page
            if (w >= (self.min_width / scale_x) and h >= (self.min_height / scale_y) and
                w < (page_w * 0.90) and h < (page_h * 0.90)):
                # Convert to pixel space
                px1 = max(0, int(m[0] * scale_x))
                py1 = max(0, int(m[1] * scale_y))
                px2 = min(pix_w, int(m[2] * scale_x))
                py2 = min(pix_h, int(m[3] * scale_y))
                clusters.append([px1, py1, px2, py2])

        return clusters

    def process_pdf(self, pdf_path: str, skip_if_done: bool = True) -> int:
        """Process a single PDF completely."""
        p_path = Path(pdf_path)
        pdf_stem = p_path.stem

        if skip_if_done and pdf_stem in self.processed_pdfs:
            print(f"Skipping already processed PDF: {pdf_stem}")
            return 0

        print(f"\nProcessing PDF: {p_path.name}")
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}")
            return 0

        total_saved = 0

        # 1. Native image stream extraction
        try:
            native_items = self.extract_native_images(doc, pdf_stem)
            total_saved += len(native_items)
            print(f"  Extracted {len(native_items)} native image streams.")
        except Exception as e:
            print(f"  Warning during native extraction: {e}")

        # 2. Rendered Page Analysis (YOLO layout + Vector drawings)
        figures_pdf_dir = self.figures_dir / pdf_stem
        page_count = len(doc)

        for page_idx in range(page_count):
            page_num = page_idx + 1
            page = doc[page_idx]

            # Render page at desired DPI (default 200 DPI gives ~1654x2338 for A4)
            zoom = self.render_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
            except Exception as e:
                print(f"  Failed rendering p.{page_num}: {e}")
                continue

            # Convert to PIL and numpy BGR
            page_img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            page_img_bgr = cv2.cvtColor(page_img_np, cv2.COLOR_RGB2BGR)

            candidate_boxes = []

            # 2a. DocLayNet YOLO layout detection
            if self.enable_yolo and self.detector is not None:
                try:
                    yolo_dets = self.detector.detect_figures(page_img_pil, conf_thresh=0.25, imgsz=1024)
                    for det in yolo_dets:
                        candidate_boxes.append({
                            "bbox": det["bbox"],
                            "source": f"yolo_{det['class_name']}",
                            "conf": det["conf"]
                        })
                except Exception as e:
                    print(f"  Layout detection error on p.{page_num}: {e}")

            # 2b. Vector drawing clusters
            if self.enable_drawings:
                try:
                    drawing_boxes = self.extract_vector_drawings(page, pix, pdf_stem, page_num, figures_pdf_dir)
                    for dbox in drawing_boxes:
                        candidate_boxes.append({
                            "bbox": dbox,
                            "source": "vector_drawing_cluster",
                            "conf": 0.8
                        })
                except Exception as e:
                    print(f"  Vector drawing clustering error on p.{page_num}: {e}")

            # Deduplicate candidate boxes using NMS / IoU
            candidate_boxes.sort(key=lambda x: x["conf"], reverse=True)
            kept_boxes = []
            for cand in candidate_boxes:
                b = cand["bbox"]
                w = b[2] - b[0]
                h = b[3] - b[1]
                # Reject full page crops
                if w >= (pix.width * 0.92) and h >= (pix.height * 0.92):
                    continue
                if w < self.min_width or h < self.min_height:
                    continue

                # Check overlap against already kept
                overlap = False
                for k in kept_boxes:
                    if box_iou(b, k["bbox"]) > 0.45:
                        overlap = True
                        break
                if not overlap:
                    kept_boxes.append(cand)

            # Crop, auto-trim borders, check blank, and save
            for crop_idx, crop_info in enumerate(kept_boxes):
                bx1, by1, bx2, by2 = [int(v) for v in crop_info["bbox"]]
                # Safety margins (6 pixels)
                bx1 = max(0, bx1 - 6)
                by1 = max(0, by1 - 6)
                bx2 = min(pix.width, bx2 + 6)
                by2 = min(pix.height, by2 + 6)

                crop_np = page_img_bgr[by1:by2, bx1:bx2]
                if crop_np.size == 0:
                    continue

                # Auto-trim blank borders
                trimmed_np = trim_borders(crop_np)
                if is_near_blank(trimmed_np):
                    continue

                figures_pdf_dir.mkdir(parents=True, exist_ok=True)
                out_filename = f"{pdf_stem}_p{page_num:03d}_fig{crop_idx + 1:02d}.png"
                out_path = figures_pdf_dir / out_filename

                cv2.imwrite(str(out_path), trimmed_np)
                total_saved += 1

                rec = {
                    "pdf_name": pdf_stem,
                    "page": page_num,
                    "type": "rendered_crop",
                    "source": crop_info["source"],
                    "confidence": crop_info["conf"],
                    "file_path": str(out_path.relative_to(self.output_dir)),
                    "dimensions": [int(trimmed_np.shape[1]), int(trimmed_np.shape[0])],
                    "bbox": [bx1, by1, bx2, by2]
                }
                self._append_manifest(rec)

        doc.close()
        self.processed_pdfs.add(pdf_stem)
        print(f"Finished {pdf_stem}: extracted total {total_saved} items.")
        return total_saved


def run_pipeline(
    input_dir: str,
    output_dir: str,
    enable_yolo: bool = True,
    model_name: str = "yolov8m-doclaynet.pt",
    render_dpi: int = 200
):
    """
    Main entry point:
    Recursively scans input_dir for all .pdf files, ignores .xlsx / .xls,
    and runs the extraction pipeline.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Find all PDFs recursively
    pdf_files = sorted(list(input_path.glob("**/*.pdf")) + list(input_path.glob("**/*.PDF")))
    # Explicitly ignore non-PDFs / excel files
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    if not pdf_files:
        print("No PDF files found. Please ensure Google Drive is mounted and shortcut is configured.")
        return

    # Initialize YOLO detector if enabled
    detector = None
    if enable_yolo and YOLO is not None:
        try:
            detector = LayoutDetector(model_filename=model_name)
        except Exception as e:
            print(f"Warning: Could not initialize YOLO LayoutDetector: {e}")
            print("Pipeline will proceed with native embedded extraction + vector graphics clustering.")

    extractor = PDFImageExtractor(
        output_dir=output_dir,
        detector=detector,
        render_dpi=render_dpi,
        enable_yolo=(detector is not None),
        enable_drawings=True
    )

    total_extracted_all = 0
    for idx, pdf_file in enumerate(pdf_files):
        print(f"\n[{idx + 1}/{len(pdf_files)}] {pdf_file.name}")
        count = extractor.process_pdf(str(pdf_file))
        total_extracted_all += count

    print(f"\n=======================================================")
    print(f"Extraction Pipeline Complete!")
    print(f"Total images & figures extracted across all PDFs: {total_extracted_all}")
    print(f"Output saved in: {output_dir}")
    print(f"Manifest written to: {extractor.manifest_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JEE/NEET PDF Image & Figure Extractor")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing PDFs")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory to save images")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO layout detection")
    parser.add_argument("--model", type=str, default="yolov8m-doclaynet.pt", help="DocLayNet model checkpoint")
    parser.add_argument("--dpi", type=int, default=200, help="Rendering DPI for PDF pages")
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        enable_yolo=(not args.no_yolo),
        model_name=args.model,
        render_dpi=args.dpi
    )
