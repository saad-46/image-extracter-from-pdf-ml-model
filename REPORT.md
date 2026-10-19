# JEE/NEET figure extraction: audit, approach, results

## What I could access
| Resource | Status |
|---|---|
| Local copies of both reference PDFs (Downloads) | read and analysed |
| Google Drive (connector, read/search) | listed JEE/NEET folders and the existing outputs; **did not read** the old notebook or scripts |
| Drive file contents / writes, Colab runtime, Hugging Face | **not used.** No Colab execution happened, so every number below was measured locally on 2 PDFs |
| `red_box_extractor.py`, `test_*.py` | not found locally or on Drive search, so the Antigravity code was **not audited**. The pipeline here is a new implementation |

## Dataset facts (Drive listing)
- About 80 PDFs: JEE folder (`original (N)` files plus the Reference_PDFs folder) and about 30 in NEET.
- Several byte-identical duplicates (e.g. `original (4)` and `original (3)` sizes, `(27)`/`(28)`/`(30)`, `(32)`/`(34)`). Detected by sha1 in the inventory step.
- **Only one human-marked PDF exists** (40 pages). A 30–40 PDF validation is therefore **not possible yet**.

## Reference PDF audit (measured)
- 40 pages in each file, but not the same geometry: original pages are 595.3×841.9 / 594.96×841.92, marked pages are 595.5×842.25. The marked file is a Canva re-export with downsampled images.
- Residual misalignment after scaling is below 0.3 px at 100 dpi (phase correlation), so a plain scale mapping is safe here. The pipeline still measures the shift on every page.
- The marked file has 0 annotations and the boxes are raster pixels. Only 7 pages contain boxes: **11 boxes** on pages 6, 7, 9(×2), 13, 14(×2), 16, 17(×3).
- The original has red pixels on page 13 (the cartoons) and page 3, which is why plain thresholding fails.

## Approach and why
1. **Box recovery by diffing the marked page against the original.** Only red pixels that are *new* count. Red inside figures cancels out.
2. **Frame verification as a second signal.** This works without an original: four-side support ≥ 0.55, interior fill ≤ 0.30, and a plausible stroke thickness. Without the original it still rejects the 4 page-13 cartoon blobs. In diff mode they never appear.
3. **Clean crops are rendered from the original** at 300 dpi, using the frame's inner edge, then whitespace is trimmed.
4. **Detector for unmarked PDFs, PDF-native.** It uses embedded images and vector drawings. It filters repeated header/footer graphics, drops ruled tables (stroke-network detection), inline equation images, glyph outlines, fraction bars and banners, then clusters elements and attaches short label text. A raster fallback exists for scanned documents.
5. **No ML model.** I did not benchmark YOLO or layout models. The evidence available (one marked PDF with 11 targets) is too small to train or fairly evaluate one, so training is **not justified yet**. Revisit once about 10 or more PDFs are marked, especially if the dataset includes scanned PDFs where the native detector is blind.

## Results (measured locally)
- Tests: `python -m pytest tests` → **34 passed** (synthetic: thick, thin, rounded and broken frames; maroon; blobs; L-strokes; multi-box; empty page; rotation and size mapping; corrupt PDF isolation; resume idempotency; GT→YOLO. Real: page-13 cartoon regression, pages 6/7/9 completeness, detector vs human boxes).
- Ground-truth recovery on the reference PDF: **11 boxes, 0 rejected, 0 flagged** by quality checks. Page 13's cartoons are not detected as boxes, and its X–Y axes figure is kept.
- Unmarked detector vs human boxes (IoU ≥ 0.5): **precision 1.00, recall 1.00 (11/11), 0 false positives on the 33 pages without targets**, mean IoU 0.93, completeness 0.97, contamination 0.04.
- **Caveat: these are development scores.** The detector rules were tuned on this same PDF after seeing its false positives (equations, tables, banner). They are not independent validation.

## Not done / blockers
- 30–40 PDF validation: needs more marked PDFs. Cheapest route: mark about 8–10 pages in each of 4–5 *different* source PDFs (different layouts and sources, including NEET), then hold those out.
- Never run on the other ~60 PDFs, so scanned pages, other layouts and colored-figure cases are unseen. The raster fallback is untested on real data.
- Colab run, Drive writes, and HF were not exercised. The notebook compiles (syntax-checked) but has **never been executed in Colab**.
- Page-level (not document-level) YOLO split because there is only one marked PDF.

## Files
- `figx/core.py`, `figx/detect.py`, `figx/pipeline.py`, `tests/test_figx.py`, `build_nb.py`, `jee_neet_figx_v4.ipynb` (self-contained, embeds the modules).

## To run
1. Upload `jee_neet_figx_v4.ipynb` to Colab (CPU runtime is enough).
2. Make sure `My Drive/Reference_PDFs/` holds both reference PDFs and that `JEE`/`NEET` are reachable in My Drive (add shortcuts, since they are owned by another account).
3. Run top to bottom. Start with `LIMIT = 5` in section 6, check `bulk_preview.png`, then set `LIMIT = None`.
