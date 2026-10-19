# Automatic JEE/NEET PDF Image & Diagram Extraction Pipeline

A complete, runnable pipeline built with **PyMuPDF**, **OpenCV**, **Pillow**, and pretrained **DocLayNet YOLO** layout models from Hugging Face for automated extraction of embedded images, rendered diagrams, physics circuits, and chemistry apparatus from JEE/NEET question papers.

---

## Deliverables

1. [jee_neet_pdf_image_extraction.ipynb](file:///d:/henceprove/img%20extraction%20model/jee_neet_pdf_image_extraction.ipynb): Complete, self-contained Google Colab notebook ready to run with GPU/CPU fallback and persistent Google Drive storage.
2. [extraction_pipeline.py](file:///d:/henceprove/img%20extraction%20model/extraction_pipeline.py): Standalone Python module for local or server-side batch extraction.
3. [smoke_test.py](file:///d:/henceprove/img%20extraction%20model/smoke_test.py): Self-contained end-to-end verification script testing native embedded extraction and vector drawing clustering.

---

## Google Drive Setup: Single Manual Step

To allow Google Colab to access the shared folder:
1. Open the shared Google Drive folder: [https://drive.google.com/drive/folders/1JsHHTtHVNjcxBeTZf5oQPO0VxtJEmLkJ](https://drive.google.com/drive/folders/1JsHHTtHVNjcxBeTZf5oQPO0VxtJEmLkJ)
2. Click **"Add shortcut to Drive"** (or click the folder header -> **Organize** -> **Add shortcut**).
3. Choose **"My Drive"** as the destination.
4. In the Colab notebook (Step 3), set `INPUT_DIR` to the shortcut name, for example:
   ```python
   INPUT_DIR = "/content/drive/MyDrive/<Your_Shortcut_Folder_Name>"
   OUTPUT_DIR = "/content/drive/MyDrive/JEE_NEET_Extracted_Images"
   ```

---

## Pipeline Architecture

- **Native Embedded Stream Extraction:** Directly extracts raw compressed image streams (`doc.extract_image`) using PyMuPDF to preserve 100% original image quality.
- **Pretrained Layout Detector (DocLayNet YOLO):** Uses Hugging Face weights (`hantian/yolo-doclaynet` - `yolov8m-doclaynet.pt` / `yolov8n-doclaynet.pt`) targeting layout classes like `Picture` and `Formula`.
- **Vector Path Clustering:** Examines PyMuPDF vector drawing paths (`page.get_drawings`) to capture circuits, optics diagrams, and geometric shapes rendered without embedded raster objects.
- **Auto-Cropping & Border Trimming:** Uses OpenCV to remove excessive white borders around diagrams while safeguarding diagram contents with safety margins.
- **Deduplication & Blank Filtering:** Filters out near-blank crops, tiny icon noise (<40x40px), and duplicate detections via IoU-based Non-Maximum Suppression (NMS).
- **Graceful Resumption:** Checks existing entries in `extraction_manifest.jsonl` to skip previously completed PDFs upon restart.

---

## Output Structure

The pipeline automatically writes to the configured output directory in Google Drive:

```text
JEE_NEET_Extracted_Images/
├── embedded_images/
│   └── <pdf_name>/
│       └── <pdf_name>_p001_xref12.png
├── cropped_figures/
│   └── <pdf_name>/
│       └── <pdf_name>_p002_fig01.png
└── extraction_manifest.jsonl
```

### Sample Manifest Entry
```json
{
  "pdf_name": "JEE_Main_2023_Shift1",
  "page": 4,
  "type": "rendered_crop",
  "source": "yolo_Picture",
  "confidence": 0.8752,
  "file_path": "cropped_figures/JEE_Main_2023_Shift1/JEE_Main_2023_Shift1_p004_fig02.png",
  "dimensions": [480, 320],
  "bbox": [124, 450, 604, 770]
}
```

---

## Running Locally

```bash
# 1. Install dependencies
pip install pymupdf opencv-python pillow ultralytics huggingface_hub

# 2. Run extraction
python extraction_pipeline.py --input path/to/pdfs --output path/to/output

# 3. Run verification smoke test
python smoke_test.py
```
