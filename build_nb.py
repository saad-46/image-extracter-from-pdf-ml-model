import json
from pathlib import Path
def md(t): return {"cell_type":"markdown","metadata":{},"source":t.strip().splitlines(True)}
def code(t): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip().splitlines(True)}
mods="".join(f'%%writefile /content/figx/{n}.py\n{Path(f"figx/{n}.py").read_text(encoding="utf8")}\n' for n in [])
cells=[md("""# JEE/NEET figure extraction (v4)
Run top to bottom. Sections: 0 setup · 1 config · 2 verify refs · 3 ground truth · 4 review · 5 held-out eval · 6 bulk (unmarked) · 7 export.
Everything is resumable; outputs go to a **new versioned folder** and existing `recovered_crops*` are never touched."""),
code("!pip -q install \"pymupdf>=1.24\" opencv-python-headless pytest 2>&1 | tail -1\nimport pymupdf, cv2, PIL, sys, shutil, os\nprint(pymupdf.__doc__.split(chr(10))[0], cv2.__version__, PIL.__version__, 'RAM/disk:', shutil.disk_usage('/content').free//2**30, 'GB free')\n!nvidia-smi -L 2>&1 | head -1  # GPU is NOT needed for this pipeline"),
]
cells.append(code("!mkdir -p /content/figx"))
for n in ["__init__","core","detect","pipeline"]:
    src=Path(f"figx/{n}.py").read_text(encoding="utf8")
    cells.append(code(f"%%writefile /content/figx/{n}.py\n{src}"))
cells+= [
code("""from pathlib import Path
from google.colab import drive
MOUNT = Path('/content/drive')
if not (MOUNT/'MyDrive').exists():
    drive.mount(str(MOUNT))          # mounts once; skipped if already mounted
assert (MOUNT/'MyDrive').exists(), 'Drive not mounted'
sys.path.insert(0, '/content')"""),
md("## 1. Configuration (edit only this cell)"),
code("""DRIVE = MOUNT/'MyDrive'
REF_DIR   = DRIVE/'Reference_PDFs'                      # holds the original + red-boxes-marked PDFs
ORIG_PDF  = REF_DIR/'Copy of original (4).pdf'
MARK_PDF  = REF_DIR/'Copy of original (4) red boxes marked.pdf'
PDF_ROOTS = [DRIVE/'JEE', DRIVE/'NEET']                  # adjust: shortcuts to the shared dataset folders must live in My Drive
OUT       = DRIVE/'JEE_NEET_Extracted_Images'/'v4_figx'  # NEW folder; old stage2_* outputs untouched
LIMIT     = None                                         # set e.g. 5 for a trial bulk run
for p in (ORIG_PDF, MARK_PDF): print(p.exists(), p)
for r in PDF_ROOTS: print(r.exists(), r, len(list(r.rglob('*.pdf'))) if r.exists() else 0, 'pdfs')"""),
md("## 2. Reference verification + inventory\nChecks page counts, geometry, duplicates (by sha1), marked/original pairing, scans, corrupt files."),
code("""from figx.pipeline import *
import pymupdf, json
assert ORIG_PDF.exists() and MARK_PDF.exists(), 'Put both reference PDFs in REF_DIR'
with pymupdf.open(ORIG_PDF) as O, pymupdf.open(MARK_PDF) as M:
    print('pages', len(O), len(M), '| page0', O[0].rect, M[0].rect)
rows = inventory([r for r in PDF_ROOTS if r.exists()] + [REF_DIR], OUT)
import collections
print(collections.Counter((r['exam'], r['status']) for r in rows), '| unique:', len({r.get('sha1') for r in rows}))"""),
md("## 3. Ground truth from the marked PDF\nDiff-based box recovery (marked vs original), clean crops rendered from the ORIGINAL at 300 dpi."""),
code("""gt = run_ground_truth(MARK_PDF, ORIG_PDF, OUT)
print(sum(r['status']=='accepted' for r in gt), 'accepted,', sum(r['status']=='rejected' for r in gt), 'rejected')"""),
md("## 4. Visual review (no reprocessing needed)"),
code("""from IPython.display import Image, display
display(Image(str(contact_sheet(gt, OUT/'reports'/'gt_preview.png', thumb_h=130, cols=4))))"""),
md("## 5. Evaluate the unmarked-PDF detector against the human boxes\n⚠ Only one marked PDF exists, and the detector rules were tuned on it, so this is a **development** score, not an independent one. Mark 3–5 more PDFs (different layouts) and add them to `ground_truth` for a real held-out test."""),
code("""det, st = extract_pdf(ORIG_PDF, OUT, subdir='eval_detections')
res = evaluate(gt, det)
print({k: res[k] for k in ('gt','det','tp','fp','fn','precision','recall','mean_iou','mean_completeness','mean_contamination','fp_on_pages_without_gt')}, st)
display(Image(str(contact_sheet(det, OUT/'reports'/'det_preview.png', thumb_h=130, cols=4))))"""),
md("## 6. Bulk processing of unmarked PDFs (resumable)\nRun with `LIMIT=5` first and inspect the preview. Re-running skips PDFs already in `manifests/completed_pdfs.jsonl`."),
code("""pdfs = sorted(p for r in PDF_ROOTS if r.exists() for p in r.rglob('*.pdf') if not MARK_RE.search(p.stem))
summary = run_bulk(pdfs, OUT, limit=LIMIT)
print(summary)
cm = read_jsonl(OUT/'manifests'/'crop_manifest.jsonl')
print('accepted', sum(r['status']=='accepted' for r in cm), '| needs_review', sum(r.get('review_status')=='needs_review' for r in cm))
sheet = contact_sheet(cm, OUT/'reports'/'bulk_preview.png', thumb_h=110, cols=6, max_items=120)
sheet and display(Image(str(sheet)))"""),
md("## 7. Export\nCSV manifest + YOLO labels (from accepted ground truth only)."),
code("""import csv
cm = [r for r in read_jsonl(OUT/'manifests'/'crop_manifest.jsonl') if r['status']=='accepted']
with open(OUT/'manifests'/'crop_manifest.csv','w',newline='',encoding='utf8') as f:
    w = csv.writer(f); w.writerow(['crop_id','source_pdf','page','box_tight_pt','crop_px','method','review_status','flags','crop_path'])
    for r in cm: w.writerow([r['crop_id'], r['source_pdf'], r['page'], r['box_tight_pt'], r['crop_px'], r['method'], r['review_status'], ';'.join(r['quality']['flags']), r['crop_path']])
print(export_yolo(read_jsonl(OUT/'manifests'/'ground_truth.jsonl'), OUT))"""),
]
nb={"cells":cells,"metadata":{"colab":{"name":"jee_neet_figx_v4.ipynb"},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
Path("jee_neet_figx_v4.ipynb").write_text(json.dumps(nb,indent=1),encoding="utf8")
