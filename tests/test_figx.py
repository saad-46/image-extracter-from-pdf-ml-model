"""Unit + integration + real-data regression tests.

Real-data tests need the two reference PDFs; point FIGX_REF_DIR at the folder
holding them (Colab: /content/drive/MyDrive/Reference_PDFs). They are skipped,
not faked, when the files are absent.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from figx.core import (RedConfig, crop_region, iou, map_marked_to_original, quality_flags,
                       recover_annotations, red_mask, suppress_duplicates, tighten, verify_frame)
from figx.detect import DetectConfig, boilerplate_keys, detect_page
from figx.pipeline import evaluate, export_yolo, inventory, read_jsonl, run_bulk, run_ground_truth

RED = (1, 0, 0)            # pymupdf RGB 0..1
MAROON = (0.5, 0.05, 0.1)
REF = Path(os.environ.get("FIGX_REF_DIR", Path.home() / "Downloads"))
ORIG = REF / "Copy of original (4).pdf"
MARK = REF / "Copy of original (4) red boxes marked.pdf"
real = pytest.mark.skipif(not (ORIG.exists() and MARK.exists()), reason="reference PDFs not available")


# ----------------------------------------------------------- helpers
def make_pdf(path: Path, pages: list, size=(595, 842), rotate=0) -> Path:
    """pages: list of callables(page) drawing content."""
    doc = pymupdf.open()
    for draw in pages:
        p = doc.new_page(width=size[0], height=size[1])
        if rotate:
            p.set_rotation(rotate)
        draw(p)
    doc.save(path)
    return path


def figure(p, r=(200, 300, 400, 450)):
    """A small vector 'graph': axes + curve + labels."""
    x0, y0, x1, y1 = r
    p.draw_line((x0, y1), (x1, y1), width=1)
    p.draw_line((x0, y1), (x0, y0), width=1)
    p.draw_bezier((x0, y1), (x0 + 50, y0), (x1 - 50, y0), (x1, y1 - 30), width=1.2)
    p.insert_text((x1 + 2, y1 + 4), "t(s)", fontsize=9)
    p.insert_text((x0 - 10, y0 - 4), "v", fontsize=9)


def frame(p, r, color=RED, width=4, radius=None):
    p.draw_rect(pymupdf.Rect(r), color=color, width=width, radius=radius)


def blob(p, r, color=RED):
    p.draw_rect(pymupdf.Rect(r), color=color, fill=color)


def text(p):
    p.insert_textbox(pymupdf.Rect(60, 90, 540, 200), "Q1. A particle moves such that its velocity varies "
                     "with time as shown in the graph. Find the displacement in the first 20 s. " * 3, fontsize=11)


def mask_of(draw, dpi=100):
    doc = pymupdf.open(); p = doc.new_page(); draw(p)
    pm = p.get_pixmap(dpi=dpi)
    img = cv2.cvtColor(np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3), cv2.COLOR_RGB2BGR)
    return red_mask(img), p


# ------------------------------------------------------------- geometry
def test_iou_and_dedup_keeps_distinct_overlapping_targets():
    a, b = (0, 0, 100, 100), (2, 2, 101, 99)       # same region, inner/outer contour
    c = (80, 0, 180, 100)                          # different target overlapping a
    assert iou(a, b) > 0.9
    keep = suppress_duplicates([a, b, c])
    assert len(keep) == 2 and 2 in keep


def test_nested_small_box_inside_big_box_is_kept():
    big, small = (0, 0, 400, 400), (50, 50, 120, 120)   # box-in-box annotations are both valid
    assert len(suppress_duplicates([big, small])) == 2


# ------------------------------------------------------ red/maroon masks
@pytest.mark.parametrize("color", [RED, MAROON, (0.75, 0.1, 0.15)])
def test_red_and_maroon_frames_detected(color):
    m, _ = mask_of(lambda p: frame(p, (100, 100, 300, 250), color))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    x, y, w, h, _ = max(st[1:], key=lambda s: s[4])
    assert verify_frame(m, x, y, w, h).ok


@pytest.mark.parametrize("color", [(1, 0.55, 0), (0.6, 0.4, 0.2), (0, 0, 0)])
def test_orange_brown_black_not_red(color):
    m, _ = mask_of(lambda p: frame(p, (100, 100, 300, 250), color))
    assert m.sum() == 0


def test_solid_red_blob_rejected_as_frame():
    m, _ = mask_of(lambda p: blob(p, (100, 100, 160, 180)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    x, y, w, h, _ = st[1]
    chk = verify_frame(m, x, y, w, h)
    assert not chk.ok and ("solid" in chk.reason or "side" in chk.reason)


def test_l_shaped_red_stroke_rejected():
    def d(p):
        p.draw_line((100, 100), (300, 100), color=RED, width=4)
        p.draw_line((100, 100), (100, 250), color=RED, width=4)
    m, _ = mask_of(d)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    x, y, w, h, _ = st[1]
    assert not verify_frame(m, x, y, w, h).ok


@pytest.mark.parametrize("width,radius", [(2, None), (8, None), (5, 0.08)])
def test_thick_thin_and_rounded_frames(width, radius):
    doc = pymupdf.open(); p = doc.new_page()
    frame(p, (100, 100, 350, 300), width=width, radius=radius)
    anns, _ = recover_annotations(p, None)
    acc = [a for a in anns if a.accepted]
    assert len(acc) == 1
    assert iou(acc[0].box_pt, (100 - width / 2, 100 - width / 2, 350 + width / 2, 300 + width / 2)) > 0.9
    # interior excludes the border
    assert acc[0].inner_pt[0] > 100 and acc[0].inner_pt[2] < 350


def test_broken_noisy_frame_still_detected():
    doc = pymupdf.open(); p = doc.new_page()
    for (a, b) in [((100, 100), (180, 100)), ((184, 100), (300, 100)), ((300, 100), (300, 250)),
                   ((300, 250), (100, 250)), ((100, 250), (100, 175)), ((100, 171), (100, 100))]:
        p.draw_line(a, b, color=RED, width=4)       # 4pt gaps in two sides
    anns, _ = recover_annotations(p, None)
    assert sum(a.accepted for a in anns) == 1


def test_multiple_targets_on_one_page_and_empty_page():
    doc = pymupdf.open(); p = doc.new_page()
    frame(p, (60, 100, 250, 250)); frame(p, (300, 100, 520, 260)); frame(p, (100, 500, 450, 700), MAROON)
    assert sum(a.accepted for a in recover_annotations(p, None)[0]) == 3
    assert recover_annotations(doc.new_page(), None)[0] == []


def test_extremely_small_mark_ignored():
    doc = pymupdf.open(); p = doc.new_page()
    frame(p, (100, 100, 110, 108), width=1)
    assert not any(a.accepted for a in recover_annotations(p, None)[0])


def test_diff_mode_ignores_red_content_that_exists_in_original(tmp_path):
    """Figure contains a red-outlined rectangle (looks exactly like a frame). Only the
    human box added in the marked copy must be recovered."""
    def base(p):
        figure(p)
        p.draw_rect(pymupdf.Rect(230, 330, 300, 400), color=RED, width=3)  # red content IN the figure
        blob(p, (320, 340, 350, 380))
    orig = make_pdf(tmp_path / "o.pdf", [base])
    mark = make_pdf(tmp_path / "m.pdf", [lambda p: (base(p), frame(p, (180, 280, 430, 470), MAROON))])
    with pymupdf.open(orig) as O, pymupdf.open(mark) as M:
        diff = [a for a in recover_annotations(M[0], O[0])[0] if a.accepted]
        single = [a for a in recover_annotations(M[0], None)[0] if a.accepted]
    assert len(diff) == 1 and iou(diff[0].box_pt, (178, 278, 432, 472)) > 0.9
    assert len(single) >= 1          # single mode cannot tell; that is why diff mode exists


def test_red_figure_without_annotation_is_not_a_target(tmp_path):
    def d(p):
        figure(p); blob(p, (250, 350, 290, 420))      # cartoon-like red region
    with pymupdf.open(make_pdf(tmp_path / "a.pdf", [d])) as D:
        assert not any(a.accepted for a in recover_annotations(D[0], None)[0])


# ---------------------------------------------------- mapping and crops
@pytest.mark.parametrize("rotate", [0, 90])
def test_mapping_between_different_page_sizes(tmp_path, rotate):
    orig = make_pdf(tmp_path / "o.pdf", [figure], size=(595, 842), rotate=rotate)
    mark = make_pdf(tmp_path / "m.pdf", [figure], size=(612, 866), rotate=rotate)  # re-exported, scaled
    with pymupdf.open(orig) as O, pymupdf.open(mark) as M:
        b = (100, 200, 300, 400)
        m = map_marked_to_original(b, M[0], O[0])
        sx = O[0].rect.width / M[0].rect.width
        assert abs(m[0] - 100 * sx) < 1e-6 and abs(m[2] - 300 * sx) < 1e-6
        assert O[0].rect.width == (842 if rotate else 595)


def test_crop_bounds_clipped_and_whitespace_trimmed(tmp_path):
    with pymupdf.open(make_pdf(tmp_path / "f.pdf", [figure])) as D:
        img, tight = crop_region(D[0], (-50, 250, 700, 500), dpi=150)
        assert tight[0] >= 0 and tight[2] <= D[0].rect.width
        assert 180 < tight[0] < 205 and 400 < tight[2] < 440      # trimmed to the figure
        q = quality_flags(img, D[0], tight)
        assert "empty" not in q["flags"] and "red_border_remains" not in q["flags"]


def test_tighten_blank_image_is_safe():
    img = np.full((50, 80, 3), 255, np.uint8)
    out, box = tighten(img)
    assert out.shape == img.shape and box == (0, 0, 80, 50)


def test_quality_flags_red_border_and_empty(tmp_path):
    def d(p):
        frame(p, (100, 100, 300, 300))
    with pymupdf.open(make_pdf(tmp_path / "r.pdf", [d])) as D:
        img, t = crop_region(D[0], (90, 90, 310, 310), dpi=100, tight=False)
        assert "red_border_remains" in quality_flags(img, D[0], t)["flags"]
        img, t = crop_region(D[0], (400, 500, 500, 600), dpi=100, tight=False)
        assert "empty" in quality_flags(img, D[0], t)["flags"]


# -------------------------------------------------------- detector
def test_detector_finds_figure_skips_text_header_and_table(tmp_path):
    def page(p):
        p.draw_line((40, 60), (555, 60), width=1)             # header rule on every page
        p.insert_text((40, 50), "ALLEN  Test Series", fontsize=10)
        text(p); figure(p)
        for k in range(5):                                       # ruled table
            p.draw_line((80, 550 + 25 * k), (400, 550 + 25 * k), width=0.8)
        for x in (80, 200, 400):
            p.draw_line((x, 550), (x, 650), width=0.8)
        for k in range(4):
            p.insert_text((90, 568 + 25 * k), f"(P{k}) entry text", fontsize=9)
    with pymupdf.open(make_pdf(tmp_path / "d.pdf", [page] * 4)) as D:
        B = boilerplate_keys(D, DetectConfig())
        acc = [d for d in detect_page(D[1], B) if d["accepted"]]
    assert len(acc) == 1
    assert iou(acc[0]["box_pt"], (188, 290, 420, 458)) > 0.75


def test_detector_empty_page(tmp_path):
    with pymupdf.open(make_pdf(tmp_path / "e.pdf", [lambda p: None])) as D:
        assert detect_page(D[0], set()) == []


# ------------------------------------------------- integration / IO
def test_corrupt_and_missing_inputs_isolated(tmp_path):
    (tmp_path / "in" / "JEE").mkdir(parents=True)
    make_pdf(tmp_path / "in" / "JEE" / "good.pdf", [lambda p: (text(p), figure(p))])
    (tmp_path / "in" / "JEE" / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    out = tmp_path / "out"
    rows = inventory([tmp_path / "in"], out)
    st = {Path(r["path"]).name: r["status"] for r in rows}
    assert st == {"good.pdf": "ok", "bad.pdf": "error"}
    s = run_bulk(sorted((tmp_path / "in").rglob("*.pdf")), out)
    assert s["processed"] == 1 and s["failed"] == 1
    assert read_jsonl(out / "manifests" / "failed_files.jsonl")[0]["path"].endswith("bad.pdf")
    with pytest.raises(Exception):
        run_ground_truth(tmp_path / "nope.pdf", None, out)


def test_resume_and_idempotent_ids(tmp_path):
    pdfs = [make_pdf(tmp_path / f"f{k}.pdf", [lambda p: (text(p), figure(p))] * 2) for k in range(2)]
    pdfs.append(make_pdf(tmp_path / "f_dup.pdf", [lambda p: (text(p), figure(p))] * 2))  # identical bytes? not guaranteed
    out = tmp_path / "out"
    s1 = run_bulk(pdfs[:1], out)
    s2 = run_bulk(pdfs[:2], out)          # first file must be skipped
    s3 = run_bulk(pdfs[:2], out)
    assert s1["processed"] == 1 and s2["processed"] == 1 and s3["processed"] == 0
    ids = [r["crop_id"] for r in read_jsonl(out / "manifests" / "crop_manifest.jsonl") if r["status"] == "accepted"]
    assert len(ids) == len(set(ids))


def test_ground_truth_end_to_end_and_yolo(tmp_path):
    def base(p):
        text(p); figure(p); blob(p, (330, 360, 360, 400))
    orig = make_pdf(tmp_path / "doc.pdf", [base, lambda p: text(p)])
    mark = make_pdf(tmp_path / "doc red boxes marked.pdf",
                    [lambda p: (base(p), frame(p, (170, 270, 440, 475))), lambda p: text(p)])
    out = tmp_path / "out"
    rows = run_ground_truth(mark, orig, out)
    acc = [r for r in rows if r["status"] == "accepted"]
    assert len(acc) == 1 and acc[0]["page"] == 1
    img = cv2.imread(acc[0]["crop_path"])
    assert img is not None and "red_border_remains" not in acc[0]["quality"]["flags"]   # frame removed
    assert red_mask(img).sum() > 0             # but the figure's own red content is preserved
    assert run_ground_truth(mark, orig, out) and len(read_jsonl(out / "manifests" / "ground_truth.jsonl")) == len(rows)
    det_rows = [{**r, "crop_id": "d" + r["crop_id"]} for r in acc]
    assert evaluate(acc, det_rows)["recall"] == 1.0
    root = export_yolo(acc, out, negatives_per_pdf=1)
    lab = (root / "labels" / "train" / "doc_p001.txt").read_text().split()
    assert lab[0] == "0" and all(0 < float(v) <= 1 for v in lab[1:])


def test_inventory_pairs_marked_with_original(tmp_path):
    make_pdf(tmp_path / "x.pdf", [figure]); make_pdf(tmp_path / "x red boxes marked.pdf", [figure])
    rows = inventory([tmp_path], tmp_path / "out")
    m = next(r for r in rows if r["marked"])
    assert m["original_path"].endswith("x.pdf")


# ---------------------------------------------- real-data regression
@pytest.fixture(scope="module")
def ref_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("ref")
    rows = run_ground_truth(MARK, ORIG, out)
    return out, rows


@real
def test_reference_page_counts_and_alignment():
    with pymupdf.open(ORIG) as O, pymupdf.open(MARK) as M:
        assert len(O) == len(M) == 40
        _, diag = recover_annotations(M[12], O[12])
        assert max(abs(v) for v in diag["shift_px"]) < 1.0


@real
def test_reference_ground_truth_boxes(ref_run):
    _, rows = ref_run
    acc = [r for r in rows if r["status"] == "accepted"]
    per_page = {}
    for r in acc:
        per_page[r["page"]] = per_page.get(r["page"], 0) + 1
    assert per_page == {6: 1, 7: 1, 9: 2, 13: 1, 14: 2, 16: 1, 17: 3}
    assert all(not r["quality"]["flags"] for r in acc)


@real
def test_page13_cartoons_rejected_xy_axes_kept():
    with pymupdf.open(ORIG) as O, pymupdf.open(MARK) as M:
        single = recover_annotations(M[12], None)[0]
        acc = [a for a in single if a.accepted]
        rej = [a for a in single if not a.accepted]
        diff = [a for a in recover_annotations(M[12], O[12])[0] if a.accepted]
    # X-Y axes frame at ~ (271,117)-(349,189) survives in both modes
    for anns in (acc, diff):
        assert len(anns) == 1 and iou(anns[0].box_pt, (271, 117, 349, 189)) > 0.85
    assert len(rej) >= 3           # cartoon clothing blobs found but rejected


@real
def test_regression_pages_6_7_9_crops_complete(ref_run):
    _, rows = ref_run
    acc = {(r["page"], r["crop_id"][-4:]): r for r in rows if r["status"] == "accepted"}
    for (pg, _), r in acc.items():
        if pg in (6, 7, 9):
            assert r["crop_px"][0] > 300 and r["crop_px"][1] > 300
            assert r["quality"]["edge_ink"] < 0.35 and r["quality"]["red_frac"] < 0.02


@real
def test_unmarked_detector_matches_human_boxes(ref_run):
    from figx.pipeline import extract_pdf
    out, rows = ref_run
    det, _ = extract_pdf(ORIG, out, subdir="det")
    res = evaluate(rows, det)
    (out / "eval.json").write_text(json.dumps(res))
    assert res["recall"] >= 0.9 and res["precision"] >= 0.8
