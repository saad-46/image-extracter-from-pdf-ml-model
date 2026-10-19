"""
Smoke test script to verify PyMuPDF, OpenCV, Pillow, and pipeline flow.
Generates a mock JEE physics/math exam PDF with embedded images and vector diagrams,
then runs the extraction pipeline on it.
"""

import os
import shutil
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import numpy as np

def create_sample_jee_pdf(output_pdf_path: str):
    doc = fitz.open()

    # --- Page 1: Native embedded image + text ---
    page1 = doc.new_page(width=595, height=842) # A4
    page1.insert_text((50, 50), "JEE Advanced Mock Exam - Physics", fontsize=16)
    page1.insert_text((50, 80), "Q1. Consider the circuit diagram shown below:", fontsize=12)

    # Create a small diagram image using PIL
    img = Image.new("RGB", (250, 180), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 240, 170], outline=(0, 0, 0), width=2)
    draw.ellipse([50, 40, 120, 110], outline=(200, 0, 0), width=3)
    draw.line([120, 75, 200, 75], fill=(0, 0, 200), width=2)
    draw.text((65, 70), "R1", fill=(0, 0, 0))
    draw.text((150, 60), "R2", fill=(0, 0, 0))

    img_temp_path = "temp_diag.png"
    img.save(img_temp_path)

    # Insert image as embedded stream
    img_rect = fitz.Rect(50, 100, 300, 280)
    page1.insert_image(img_rect, filename=img_temp_path)
    if os.path.exists(img_temp_path):
        os.remove(img_temp_path)

    page1.insert_text((50, 310), "Find the equivalent resistance between terminals A and B.", fontsize=11)

    # --- Page 2: Vector drawing diagram (no embedded image stream) ---
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 50), "JEE Advanced Mock Exam - Mechanics", fontsize=16)
    page2.insert_text((50, 80), "Q2. A block of mass m is placed on an inclined wedge:", fontsize=12)

    # Draw vector shapes using PyMuPDF Shape
    shape = page2.new_shape()
    # Draw an inclined triangle (wedge)
    shape.draw_polyline([fitz.Point(100, 300), fitz.Point(350, 300), fitz.Point(350, 180), fitz.Point(100, 300)])
    shape.finish(width=2, color=(0, 0, 0), fill=(0.92, 0.92, 0.92))

    # Draw a block on the incline
    shape.draw_rect(fitz.Rect(220, 200, 270, 240))
    shape.finish(width=2, color=(0.8, 0, 0), fill=(0.95, 0.8, 0.8))
    shape.commit()

    page2.insert_text((100, 340), "Calculate the minimum coefficient of static friction.", fontsize=11)

    doc.save(output_pdf_path)
    doc.close()
    print(f"Sample test PDF created at: {output_pdf_path}")


def main():
    test_dir = Path("smoke_test_run")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    input_dir = test_dir / "input_pdfs"
    output_dir = test_dir / "extracted_output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Also add a dummy excel file to confirm it is ignored
    with open(input_dir / "dummy_roster.xlsx", "wb") as f:
        f.write(b"PK\x03\x04DummyExcelFile")

    sample_pdf = input_dir / "sample_jee_physics.pdf"
    create_sample_jee_pdf(str(sample_pdf))

    print("\nRunning extraction pipeline smoke test...")
    from extraction_pipeline import run_pipeline

    run_pipeline(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        enable_yolo=False,  # Test native + vector graphics logic
        render_dpi=150
    )

    # Verify extracted results
    embedded_files = list((output_dir / "embedded_images").glob("**/*.*"))
    figures_files = list((output_dir / "cropped_figures").glob("**/*.*"))
    manifest_file = output_dir / "extraction_manifest.jsonl"

    print("\n--- Smoke Test Verification ---")
    print(f"Embedded images extracted: {len(embedded_files)}")
    for f in embedded_files:
        print(f"  - {f.name}")
    print(f"Cropped vector figures extracted: {len(figures_files)}")
    for f in figures_files:
        print(f"  - {f.name}")
    print(f"Manifest exists: {manifest_file.exists()} (size: {manifest_file.stat().st_size} bytes)")

    assert len(embedded_files) >= 1, "Failed: Should have extracted native embedded image."
    assert len(figures_files) >= 1, "Failed: Should have extracted vector drawing diagram."
    assert manifest_file.exists(), "Failed: Manifest file not created."

    print("\nSmoke test PASSED successfully!")


if __name__ == "__main__":
    main()
