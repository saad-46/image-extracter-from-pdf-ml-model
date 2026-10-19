"""Executes the real code cells of holdout_scan.ipynb against synthetic PDFs in a fake Drive.

Only the Colab-specific parts are substituted: package dir, Drive location, IPython display.
Synthetic results prove the plumbing/guards work; they say NOTHING about real-world accuracy.
"""
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
JEE = ["Copy of original (24).pdf", "Copy of original (33).pdf", "Copy of original (41).pdf"]
NEET = ["Copy of NEET 2016.pdf", "Copy of NEET 2024.pdf", "Copy of NEET 2005.pdf"]
BODY = "Q{n}. A particle moves with velocity varying as shown in the graph below. Find the displacement. "


def text_pdf(path, tag, pages=10, figures=False):
    d = pymupdf.open()
    for i in range(pages):
        p = d.new_page(width=595, height=842)
        p.insert_textbox(pymupdf.Rect(60, 80, 540, 200), (BODY.format(n=i) + tag + " ") * 4, fontsize=11)
        if figures:
            y = 300 + 20 * (i % 4)
            p.draw_line((200, y + 150), (400, y + 150), width=1)
            p.draw_line((200, y + 150), (200, y), width=1)
            p.draw_bezier((200, y + 150), (250, y), (350, y), (400, y + 120), width=1.2)
            p.insert_text((404, y + 154), "t(s)", fontsize=9)
    d.save(path)
    return path


def scan_pdf_file(path, seed, pages=6):
    rng = np.random.default_rng(seed)
    d = pymupdf.open()
    for _ in range(pages):
        p = d.new_page()
        arr = rng.integers(0, 256, (120, 90), dtype=np.uint8)
        p.insert_image(p.rect, pixmap=pymupdf.Pixmap(pymupdf.csGRAY, 90, 120, arr.tobytes(), False))
    d.save(path)
    return path


def build_drive(root: Path) -> Path:
    D = root / "MyDrive"
    for x in ("JEE", "NEET", "Reference_PDFs"):
        (D / x).mkdir(parents=True)
    text_pdf(D / "JEE" / JEE[0], "alpha jee24", figures=True)               # text-based, several figure candidates
    text_pdf(D / "JEE" / JEE[1], "bravo jee33", pages=6, figures=False)     # text-based, no figures
    scan_pdf_file(D / "JEE" / JEE[2], seed=11)                              # scanned, no extractable text
    text_pdf(D / "NEET" / NEET[0], "charlie neet16", figures=True)
    text_pdf(D / "NEET" / NEET[1], "delta neet24", figures=True)
    scan_pdf_file(D / "NEET" / NEET[2], seed=22)                            # a second, unrelated scan
    text_pdf(D / "JEE" / "Copy of original (14).pdf", "echo fallback14", pages=6)
    return D


def tree_hashes(root: Path):
    return {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def harness(tmp_path):
    saved_mods = {k: v for k, v in sys.modules.items() if k == "figx" or k.startswith("figx.")}
    saved_path = list(sys.path)
    ipy = types.ModuleType("IPython"); disp = types.ModuleType("IPython.display")
    disp.Markdown = lambda x: x; disp.display = lambda x: None
    sys.modules["IPython"], sys.modules["IPython.display"] = ipy, disp
    nb = json.loads((ROOT / "holdout_scan.ipynb").read_text(encoding="utf8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and not "".join(c["source"]).startswith("%%")]
    cells = [c for c in cells if "pip" not in c and "rmtree('/content/holdout_pkg'" not in c]   # install/writefile cells: Colab-only

    def run(drive_root: Path, pkg_root: Path = None, before_scan=None):
        pkg_root = pkg_root or tmp_path / "pkg"
        if not pkg_root.exists():
            pkg_root.mkdir(); shutil.copytree(ROOT / "figx", pkg_root / "figx", ignore=shutil.ignore_patterns("__pycache__"))
        g = {}
        for c in cells:
            c = c.replace("/content/holdout_pkg", pkg_root.as_posix())
            c = c.replace("[Path('/content/drive'), Path('/content/gdrive')]", f"[Path('{drive_root.as_posix()}')]")
            exec(c, g)
            if before_scan and "TRUSTED_V4" in c:
                before_scan(g)
        return g
    yield run
    for k in [k for k in sys.modules if k == "figx" or k.startswith("figx.")]:
        del sys.modules[k]
    sys.modules.update(saved_mods)
    sys.path[:] = saved_path
    for k in ("IPython", "IPython.display"):
        sys.modules.pop(k, None)


def out_dir(D):
    return D / "JEE_NEET_Extracted_Images" / "v4_figx" / "holdout_prep"


def test_happy_path_text_scanned_nofigure_and_figure_pdfs(tmp_path, harness):
    root = tmp_path / "fake"; D = build_drive(root)
    before = tree_hashes(D)
    g = harness(root)
    sm = g["summary"]
    assert sm["complete"] and sm["fingerprint_verified"] is True
    assert (sm["expected"], sm["pdfs_located"], sm["pdfs_opened_and_scanned"], sm["pdfs_fully_scanned"], sm["pages_failed"]) == (6, 6, 6, 6, 0)
    assert sm["confirmed_duplicates"] == [] and sm["suspected_duplicates"] == [], "unrelated PDFs flagged as duplicates"
    by = {s["file"]: s for s in g["scans"]}
    assert by[JEE[2]]["scanned_frac"] == 1.0 and by[JEE[2]]["text_sig"] is None            # scanned, empty text
    assert by[JEE[0]]["text_sig"] is not None
    assert sum(p["n_figs"] > 0 for p in by[JEE[0]]["pages"]) >= 4, "synthetic figure pages not detected at all"
    assert sum(p["n_figs"] for p in by[JEE[1]]["pages"]) == 0                              # no-figure PDF
    sel = g["selections"][JEE[1]]
    assert any("only 0 page(s) have detector figure candidates" in x for x in sel["shortages"])
    o = out_dir(D)
    assert sorted(p.name for p in o.iterdir()) == ["holdout_manifest.json", "marking_checklist.md", "page_scan.json"]
    md = (o / "marking_checklist.md").read_text(encoding="utf8")
    assert "COMPLETE" in md and "VERIFIED against trusted v4 baseline" in md and "not been validated" in md
    assert "MACHINE-GENERATED SUGGESTIONS" in md
    after = tree_hashes(D)
    assert {k: v for k, v in after.items() if not k.startswith("JEE_NEET_Extracted_Images")} == before   # sources untouched
    man = json.loads((o / "holdout_manifest.json").read_text())
    assert [d["file"] for d in man["documents"]] == JEE + NEET and all(d["sha1"] for d in man["documents"])


def test_duplicates_are_confirmed_and_replacement_only_suggested(tmp_path, harness):
    root = tmp_path / "fake"; D = build_drive(root)
    shutil.copy(D / "JEE" / JEE[1], D / "JEE" / JEE[2])                  # (41) becomes a byte copy of (33)
    before = tree_hashes(D)
    g = harness(root)
    sm = g["summary"]
    assert any("CONFIRMED byte-identical" in d for d in sm["confirmed_duplicates"])
    assert g["replacements"] and "Copy of original (14).pdf" in g["replacements"][0] and JEE[2] in g["replacements"][0]
    md = (out_dir(D) / "marking_checklist.md").read_text(encoding="utf8")
    assert "needs your approval; never applied automatically" in md
    assert {k: v for k, v in tree_hashes(D).items() if not k.startswith("JEE_NEET_Extracted_Images")} == before


def test_ambiguous_same_name_files_stop_the_run(tmp_path, harness):
    root = tmp_path / "fake"; D = build_drive(root)
    text_pdf(D / "Reference_PDFs" / JEE[0], "a different file with the same name", figures=True)
    from figx.scan import MissingFilesError
    with pytest.raises(Exception) as e:
        harness(root)
    assert type(e.value).__name__ == "MissingFilesError" and "Ambiguous" in str(e.value) and JEE[0] in str(e.value)
    assert not out_dir(D).exists()


def test_missing_input_file_stops_before_any_output(tmp_path, harness):
    root = tmp_path / "fake"; D = build_drive(root)
    (D / "NEET" / NEET[2]).unlink()
    with pytest.raises(Exception) as e:
        harness(root)
    assert type(e.value).__name__ == "MissingFilesError" and f"Missing: {NEET[2]}" in str(e.value)
    assert "Drive roots searched" in str(e.value)
    assert not out_dir(D).exists(), "a checklist/output dir must not appear when inputs are missing"


def test_simulated_page_error_makes_summary_incomplete(tmp_path, harness):
    root = tmp_path / "fake"; D = build_drive(root)

    def inject(g):
        real = g["scan"].detect_page
        def flaky(page, boiler, cfg, sf=0.0):
            if page.parent.name.endswith("neet16.pdf") or "NEET 2016" in page.parent.name:
                if page.number == 2:
                    raise RuntimeError("simulated page-processing error")
            return real(page, boiler, cfg, sf)
        g["scan"].detect_page = flaky
    g = harness(root, before_scan=inject)
    sm = g["summary"]
    assert not sm["complete"] and sm["pdfs_with_page_errors"] == [NEET[0]]
    assert sm["pages_failed"] == 1 and sm["pdfs_fully_scanned"] == 5 and sm["pdfs_opened_and_scanned"] == 6
    md = (out_dir(D) / "marking_checklist.md").read_text(encoding="utf8")
    assert "INCOMPLETE" in md and "Partially scanned" in md and "with page errors 1" in md


def test_tampered_detector_is_refused_before_scanning(tmp_path, harness):
    root = tmp_path / "fake"; D = build_drive(root)
    pkg = tmp_path / "pkg_t"; pkg.mkdir()
    shutil.copytree(ROOT / "figx", pkg / "figx", ignore=shutil.ignore_patterns("__pycache__"))
    f = pkg / "figx" / "detect.py"
    f.write_text(f.read_text(encoding="utf8").replace("max_aspect: float = 9.0", "max_aspect: float = 8.0"), encoding="utf8")
    with pytest.raises(Exception) as e:
        harness(root, pkg_root=pkg)
    assert type(e.value).__name__ == "FingerprintMismatch" and "detect.py source differs" in str(e.value)
    assert not out_dir(D).exists()
