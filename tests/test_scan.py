"""Synthetic tests for the held-out preparation tooling. No held-out PDF is ever used.
These are unit/synthetic-integration tests; they say nothing about real-world accuracy."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest

import figx.scan as scan
from figx.scan import (FingerprintMismatch, MissingFilesError, assert_inside, build_summary, check_output_dir,
                       classify_pair, compare_fingerprints, detector_fingerprint, duplicate_report, duplicate_wording,
                       fingerprint_from_sources, holdout_manifest, locate, normalize_source, plan_drive_mount,
                       render_checklist, require_all, require_fingerprint, safe_write_text, scan_pdf, select_pages,
                       sha1_file, summary_lines, text_signature)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "figx"
FP = {"spec": "figx-fp-v2", "combined": "a" * 64}


# ------------------------------------------------------------ fixtures
def rec(page, n=0, kind="none", warm=0.0, words=100, status="ok"):
    return {"page": page, "status": status, "error": None, "n_figs": n, "kind": kind, "warm_share": warm,
            "words": words, "methods": [], "boxes": []}


def doc(pages, scanned_frac=0.0, name="d.pdf", status="ok"):
    bad = sum(p["status"] == "error" for p in pages)
    return {"file": name, "path": name, "status": status, "error": None, "n_pages": len(pages), "sha1": name,
            "bytes": 1, "scanned_frac": scanned_frac, "text_sig": "s", "thumbs": [], "pages": pages,
            "pages_ok": len(pages) - bad, "pages_failed": bad}


def roles(sel):
    return {e["page"]: e["roles"] for e in sel["selections"]}


def loc_for(*names, missing=(), ambiguous=None):
    return {"found": {n: Path(n) for n in names}, "missing": list(missing), "ambiguous": ambiguous or {}, "roots": []}


def checklist(d, s, fp_ok=True):
    sm = build_summary([d["file"]], loc_for(d["file"]), [d], [], fp_ok)
    return render_checklist(sm, [d], {d["file"]: s}, FP)


def mk(path, texts, meta=None):
    d = pymupdf.open()
    for t in texts:
        d.new_page().insert_text((72, 100), t, fontsize=12)
    if meta:
        d.set_metadata(meta)
    d.save(path)
    return path


def noise_pdf(path, seed, pages=4, meta=None):
    """Scanned-style PDF: full-page noise image per page, no text layer."""
    rng = np.random.default_rng(seed)
    d = pymupdf.open()
    for _ in range(pages):
        pg = d.new_page()
        arr = rng.integers(0, 256, (120, 90), dtype=np.uint8)
        pg.insert_image(pg.rect, pixmap=pymupdf.Pixmap(pymupdf.csGRAY, 90, 120, arr.tobytes(), False))
    if meta:
        d.set_metadata(meta)
    d.save(path)
    return path


# ------------------------------------------------- selection (unchanged behaviour)
def test_selection_spreads_eight_figure_pages_and_controls():
    pages = [rec(i, n=1 if i % 2 == 0 else 0, kind="raster" if i % 2 == 0 else "none") for i in range(1, 61)]
    pages[9]["warm_share"] = 0.02
    s = select_pages(doc(pages))
    r = roles(s)
    figs = [p for p, ro in r.items() if "figure_candidate" in ro]
    assert len(figs) == 8 and min(figs) <= 10 and max(figs) >= 50
    assert sum("negative_control" in ro for ro in r.values()) == 2
    assert "colour_control" in r[10] and not s["shortages"]


def test_colour_control_allowed_on_detected_figure_page_with_safe_wording():
    pages = [rec(1, n=2, kind="raster", warm=0.03)] + [rec(i) for i in range(2, 10)]
    d = doc(pages)
    s = select_pages(d, n_fig=1)
    assert set(next(e for e in s["selections"] if e["page"] == 1)["roles"]) == {"figure_candidate", "colour_control"}
    txt = checklist(d, s)
    assert "Do NOT box red/orange content that is not a target figure" in txt
    assert "MACHINE-GENERATED SUGGESTIONS" in txt and "not* proven figure-free" in txt
    assert "not evidence of extraction accuracy" in txt


def test_shortages_reported_not_invented():
    s = select_pages(doc([rec(1, n=1, kind="raster")] + [rec(i) for i in range(2, 8)]))
    assert any("only 1 page(s) have detector figure candidates" in x for x in s["shortages"])
    s = select_pages(doc([rec(i, n=1, kind="raster") for i in range(1, 12)]))
    assert any("plausible negative-control" in x for x in s["shortages"])
    assert any("no colour-control page can be suggested" in x for x in s["shortages"])
    assert not any("negative_control" in r or "colour_control" in r for r in roles(s).values())


def test_empty_all_error_and_zero_detection_documents():
    assert select_pages(doc([]))["selections"] == []
    s = select_pages(doc([rec(1, status="error"), rec(2, status="error")]))
    assert s["selections"] == [] and any("failed to scan" in x for x in s["shortages"])
    s = select_pages(doc([rec(i, n=0, words=200) for i in range(1, 12)]))
    reasons = " ".join(" ".join(e["reasons"]) for e in s["selections"])
    assert "PLAUSIBLE negative, not confirmed" in reasons and "has no figure" not in reasons


def test_scanned_pages_represented_and_flagged_unvalidated():
    d = doc([rec(i, n=1, kind="scanned_page", words=0) for i in range(1, 21)], scanned_frac=1.0, name="NEET 2005.pdf")
    s = select_pages(d)
    assert s["selections"] and all(e["rec"]["kind"] == "scanned_page" for e in s["selections"])
    txt = checklist(d, s)
    assert "not been validated" in txt and "scanned_page" in txt
    mixed = doc([rec(i, n=1, kind="raster") for i in range(1, 9)] + [rec(i, kind="scanned_page", words=0) for i in range(9, 15)], 0.4)
    assert any("scanned_sample" in r for r in roles(select_pages(mixed)).values())


# ------------------------------------------------------- A. empty-text duplicates
def test_text_signature_is_none_without_meaningful_text(tmp_path):
    assert text_signature(pymupdf.open(noise_pdf(tmp_path / "s.pdf", 1))) is None
    blank = pymupdf.open(); blank.new_page()
    assert text_signature(blank) is None
    short = mk(tmp_path / "short.pdf", ["Q1"])
    assert text_signature(pymupdf.open(short)) is None            # below the alphanumeric minimum
    assert text_signature(pymupdf.open(mk(tmp_path / "t.pdf", ["alpha beta gamma delta epsilon zeta eta theta iota kappa"]))) is not None


def test_two_unrelated_scanned_pdfs_with_empty_text_are_not_duplicates(tmp_path):
    a = scan_pdf(noise_pdf(tmp_path / "a.pdf", seed=1))
    b = scan_pdf(noise_pdf(tmp_path / "b.pdf", seed=2))
    assert a["text_sig"] is None and b["text_sig"] is None
    assert a["n_pages"] == b["n_pages"] == 4                      # same page count must not be enough
    c = classify_pair(a, b)
    assert c["level"] == "distinct"
    assert "text signature unavailable (no extractable text)" in c["evidence"]
    assert duplicate_report([a, b]) == []


def test_missing_signatures_never_match_each_other():
    a = doc([rec(1)], name="a.pdf"); b = doc([rec(1)], name="b.pdf")
    a.update(sha1="x", text_sig=None, n_pages=4); b.update(sha1="y", text_sig=None, n_pages=4)
    assert classify_pair(a, b)["level"] == "distinct"


def test_same_scan_resaved_is_suspected_visual_only_not_confirmed(tmp_path):
    a = scan_pdf(noise_pdf(tmp_path / "a.pdf", seed=7))
    b = scan_pdf(noise_pdf(tmp_path / "b.pdf", seed=7, meta={"title": "re-export"}))
    assert a["sha1"] != b["sha1"]
    c = classify_pair(a, b)
    assert c["level"] == "suspected_visual_only"
    assert "NOT proof" in duplicate_wording(c) and "CONFIRMED" not in duplicate_wording(c)


def test_blank_scans_are_not_visually_comparable(tmp_path):
    def blank_scan(p):
        d = pymupdf.open()
        for _ in range(4):
            pg = d.new_page()
            pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 90, 120), 0); pix.clear_with(240)
            pg.insert_image(pg.rect, pixmap=pix)
        d.save(p); return scan_pdf(p)
    a, b = blank_scan(tmp_path / "a.pdf"), blank_scan(tmp_path / "b.pdf")
    c = classify_pair({**a, "sha1": "1"}, {**b, "sha1": "2"})
    assert c["level"] == "distinct" and "too few non-blank sampled pages to compare visually" in c["evidence"]


def test_byte_identical_confirmed_and_text_matches_only_suspected(tmp_path):
    a = mk(tmp_path / "a.pdf", ["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 4)
    shutil.copy(a, tmp_path / "b.pdf")
    mk(tmp_path / "c.pdf", ["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 4, {"title": "re-export"})
    mk(tmp_path / "d.pdf", ["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 6)
    mk(tmp_path / "e.pdf", ["entirely different words about optics and thermodynamics laws"] * 4)
    S = {n: scan_pdf(tmp_path / f"{n}.pdf") for n in "abcde"}
    assert classify_pair(S["a"], S["b"])["level"] == "confirmed_identical_bytes"
    assert classify_pair(S["a"], S["c"])["level"] == "suspected_strong"
    assert classify_pair(S["a"], S["d"])["level"] == "suspected_text_only"
    assert classify_pair(S["a"], S["e"])["level"] == "distinct"
    words = [duplicate_wording(x) for x in duplicate_report(list(S.values()))]
    assert any(w.startswith("CONFIRMED byte-identical: a.pdf == b.pdf") for w in words)
    assert all("NOT proof" in w for w in words if not w.startswith("CONFIRMED"))


# --------------------------------------- B/C. fingerprint: spec, normalisation, baseline
def fp_of(core="x = 1\n", detect="y = 2\n", cfg="{}"):
    return fingerprint_from_sources({"core.py": core, "detect.py": detect}, cfg)


def test_normalize_source_rules():
    assert normalize_source("a  \r\nb\t\r\n\r\n\r\n") == "a\nb\n"
    assert normalize_source("﻿a\rb") == "a\nb\n"
    assert normalize_source("  a\n    b") == "  a\n    b\n"          # indentation kept
    assert normalize_source("a\n\nb\n") == "a\n\nb\n"                # interior blank lines kept


def test_trailing_whitespace_on_every_line_is_ignored():
    base = "def f(x):\n    return x\n\n# note\nz = 3\n"
    noisy = "def f(x):   \n    return x\t\n   \n# note  \nz = 3   \n\n\n"       # ordinary lines, not just the last
    assert fp_of(core=base) == fp_of(core=noisy)
    assert fp_of(core=base) == fp_of(core=noisy.replace("\n", "\r\n"))


def test_meaningful_changes_change_the_fingerprint():
    base = "def f(x):\n    return x\n"
    for changed in ("def f(x):\n    return x + 1\n", "def f(x):\n  return x\n", "# c\ndef f(x):\n    return x\n",
                    "def f(x):\n\n    return x\n"):
        assert fp_of(core=changed)["files"]["core.py"] != fp_of(core=base)["files"]["core.py"]
    assert fp_of(cfg='{"a":1}')["config"] != fp_of(cfg='{"a":2}')["config"]
    assert fp_of(detect="y = 3\n")["combined"] != fp_of()["combined"]


def test_compare_and_require_fingerprint_report_what_differs():
    a, b = fp_of(), fp_of(detect="y = 9\n", cfg='{"k":1}')
    diff = compare_fingerprints(b, a)
    assert "detect.py source differs" in diff and "default DetectConfig/RedConfig values differ" in diff
    with pytest.raises(FingerprintMismatch, match="trusted v4 baseline"):
        require_fingerprint(b, a, "test")
    require_fingerprint(a, a, "test")


def test_fingerprint_detects_source_and_runtime_config_change(tmp_path, monkeypatch):
    pkg = tmp_path / "figx"
    shutil.copytree(PKG, pkg, ignore=shutil.ignore_patterns("__pycache__"))
    assert detector_fingerprint(pkg) == detector_fingerprint(PKG)
    (pkg / "detect.py").write_text((pkg / "detect.py").read_text() + "\n# tweak\n")
    assert detector_fingerprint(pkg)["combined"] != detector_fingerprint(PKG)["combined"]
    import dataclasses, figx.detect as det
    monkeypatch.setattr(det.DetectConfig, "min_w_pt", 31.0)      # runtime patch of the imported default
    assert scan.config_json() != json.dumps({}) and detector_fingerprint(PKG)["config"] != \
        fingerprint_from_sources({"core.py": "", "detect.py": ""}, "{}")["config"]


def test_local_and_embedded_detectors_match_pinned_trusted_v4_baseline():
    """Independent comparison: the pin was computed from the v4 notebook by tools/make_baseline.py."""
    pinned = json.loads((PKG / "baseline_v4.json").read_text(encoding="utf8"))
    assert pinned["source"]["notebook"] == "jee_neet_figx_v4.ipynb"
    local = detector_fingerprint(PKG)
    assert compare_fingerprints(local, pinned) == [], "local detector drifted from the trusted v4 baseline"
    for nb, prefix in (("jee_neet_figx_v4.ipynb", "/content/figx/"), ("holdout_scan.ipynb", "/content/holdout_pkg/figx/")):
        src = scan.notebook_sources(ROOT / nb, prefix)
        for n in scan.FP_FILES:
            assert fingerprint_from_sources({**src, **{m: "" for m in scan.FP_FILES if m not in src}}, "{}")["files"][n] \
                == pinned["files"][n], f"{nb} embeds a different {n}"


def test_tampered_notebook_embedding_would_be_caught(tmp_path):
    nb = json.loads((ROOT / "holdout_scan.ipynb").read_text(encoding="utf8"))
    for c in nb["cells"]:
        s = "".join(c["source"])
        if s.startswith("%%writefile /content/holdout_pkg/figx/detect.py"):
            c["source"] = [s.replace("max_aspect: float = 9.0", "max_aspect: float = 99.0")]
    p = tmp_path / "tampered.ipynb"; p.write_text(json.dumps(nb), encoding="utf8")
    src = scan.notebook_sources(p, "/content/holdout_pkg/figx/")
    pinned = json.loads((PKG / "baseline_v4.json").read_text(encoding="utf8"))
    assert fingerprint_from_sources(src, "{}")["files"]["detect.py"] != pinned["files"]["detect.py"]


# ------------------------------------------------------ D. install handling
def test_notebook_install_cell_propagates_errors_and_verifies_imports():
    nb = json.loads((ROOT / "holdout_scan.ipynb").read_text(encoding="utf8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    install = next(c for c in cells if "pip" in c and "install" in c)
    assert "tail" not in install and "| tail" not in install
    assert "returncode != 0" in install and "raise RuntimeError" in install
    assert "import_module" in install and "critical import failed" in install
    assert not any("| tail" in c for c in cells)


# ----------------------------------------- E. scan completeness / page failures
def test_partial_page_failure_is_reported_not_hidden(tmp_path, monkeypatch):
    p = mk(tmp_path / "three.pdf", ["body text for page one is long enough"] * 3)
    real = scan.detect_page
    def flaky(page, boiler, cfg, sf=0.0):
        if page.number == 1:
            raise RuntimeError("simulated page failure")
        return real(page, boiler, cfg, sf)
    monkeypatch.setattr(scan, "detect_page", flaky)
    s = scan_pdf(p)
    assert s["status"] == "partial" and s["pages_ok"] == 2 and s["pages_failed"] == 1
    assert s["pages"][1]["status"] == "error" and "simulated page failure" in s["pages"][1]["error"]
    sm = build_summary(["three.pdf"], loc_for("three.pdf"), [s], [], True)
    assert not sm["complete"]
    assert sm["pdfs_opened_and_scanned"] == 1 and sm["pdfs_fully_scanned"] == 0
    assert sm["pdfs_with_page_errors"] == ["three.pdf"] and sm["pages_failed"] == 1 and sm["pages_processed_ok"] == 2
    text = "\n".join(summary_lines(sm))
    assert "INCOMPLETE" in text and "with page errors 1" in text and "pages failed 1" in text
    sel = select_pages(s)
    assert 2 not in [e["page"] for e in sel["selections"]]
    assert any("failed to scan" in x for x in sel["shortages"])
    assert "Partially scanned" in render_checklist(sm, [s], {"three.pdf": sel}, FP)


def test_summary_separates_located_opened_failed_missing_ambiguous_and_fingerprint():
    ok = doc([rec(1)], name="ok.pdf"); bad = doc([rec(1)], name="bad.pdf", status="error"); bad["error"] = "boom"
    bad["pages"], bad["pages_ok"] = [], 0
    loc = loc_for("ok.pdf", "bad.pdf", missing=["gone.pdf"], ambiguous={"dup.pdf": ["a", "b"]})
    T = ["ok.pdf", "bad.pdf", "gone.pdf", "dup.pdf", "never.pdf"]
    sm = build_summary(T, loc, [ok, bad], [], True)
    assert (sm["expected"], sm["pdfs_located"], sm["pdfs_opened_and_scanned"]) == (5, 2, 1)
    assert sm["pdfs_failed_to_open"] == {"bad.pdf": "boom"}
    assert sm["pdfs_missing"] == ["gone.pdf"] and sm["pdfs_ambiguous"] == ["dup.pdf"]
    assert set(sm["omitted"]) == {"bad.pdf", "gone.pdf", "dup.pdf", "never.pdf"} and not sm["complete"]
    full = build_summary(["ok.pdf"], loc_for("ok.pdf"), [ok], [], True)
    assert full["complete"]
    assert not build_summary(["ok.pdf"], loc_for("ok.pdf"), [ok], [], None)["complete"]      # unverified detector
    assert not build_summary(["ok.pdf"], loc_for("ok.pdf"), [ok], [], False)["complete"]
    assert "NOT CHECKED" in "\n".join(summary_lines(build_summary(["ok.pdf"], loc_for("ok.pdf"), [ok], [], None)))


def test_summary_ignores_optional_fallbacks():
    a, b = doc([rec(1)], name="a.pdf"), doc([rec(1)], name="b.pdf")
    loc = {"found": {"a.pdf": 1, "b.pdf": 2, "fallback.pdf": 3}, "missing": ["other.pdf"], "ambiguous": {}, "roots": []}
    sm = build_summary(["a.pdf", "b.pdf"], loc, [a, b], [], True)
    assert sm["pdfs_located"] == 2 and sm["complete"] and sm["omitted"] == []


# --------------------------------------------- files, ambiguity, mount, paths
def test_missing_and_ambiguous_files_stop_with_reasons(tmp_path):
    (tmp_path / "JEE").mkdir(); (tmp_path / "NEET").mkdir()
    (tmp_path / "JEE" / "a.pdf").write_bytes(b"x")
    (tmp_path / "JEE" / "same.pdf").write_bytes(b"1")
    (tmp_path / "NEET" / "same.pdf").write_bytes(b"2")
    loc = locate(["a.pdf", "b.pdf", "same.pdf"], [tmp_path / "JEE", tmp_path / "NEET", tmp_path / "ref"])
    assert loc["missing"] == ["b.pdf"] and "same.pdf" in loc["ambiguous"] and "same.pdf" not in loc["found"]
    with pytest.raises(MissingFilesError) as e:
        require_all(loc, ["a.pdf", "b.pdf", "same.pdf"])
    msg = str(e.value)
    assert "Missing: b.pdf" in msg and "Ambiguous (same name, different content)" in msg
    assert "NOT accessible" in msg and "accessible, 2 PDFs" in msg
    require_all(loc, ["a.pdf"])                                             # fine when only usable files required


def test_same_name_identical_content_is_not_ambiguous(tmp_path):
    for d in ("A", "B"):
        (tmp_path / d).mkdir(); (tmp_path / d / "same.pdf").write_bytes(b"same")
    loc = locate(["same.pdf"], [tmp_path / "A", tmp_path / "B"])
    assert "same.pdf" in loc["found"] and not loc["ambiguous"]


def test_plan_drive_mount_never_mounts_over_non_empty_dir(tmp_path):
    d1, d2 = tmp_path / "drive", tmp_path / "gdrive"
    d1.mkdir(); (d1 / "sample_data").mkdir()                                # non-empty, not Drive
    assert plan_drive_mount([d1, d2]) == ("mount", d2)                      # falls through to the empty/absent one
    d2.mkdir(); (d2 / "x").write_text("x")
    with pytest.raises(RuntimeError, match="no safe Drive mount point"):
        plan_drive_mount([d1, d2])
    (d2 / "MyDrive").mkdir()
    assert plan_drive_mount([d1, d2]) == ("use", d2)                        # detects an existing mount anywhere
    empty = tmp_path / "e"; empty.mkdir()
    assert plan_drive_mount([empty])[0] == "mount"


def test_output_paths_resolved_and_traversal_rejected(tmp_path):
    out = tmp_path / "holdout_prep"
    assert safe_write_text(out / "ok.md", "hi", out).read_text() == "hi"
    assert not list(out.glob("*.tmp"))
    for bad in (tmp_path / "evil.md", out / ".." / "evil.md", out / "sub" / ".." / ".." / "evil.md"):
        with pytest.raises(PermissionError):
            safe_write_text(bad, "x", out)
    assert not (tmp_path / "evil.md").exists()
    with pytest.raises(PermissionError):
        assert_inside(Path(out) / ".." / "holdout_prep_sibling" / "x", out)


def test_symlink_escape_rejected(tmp_path):
    out, outside = tmp_path / "out", tmp_path / "outside"
    out.mkdir(); outside.mkdir()
    try:
        os.symlink(outside, out / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(PermissionError):
        safe_write_text(out / "link" / "x.md", "x", out)
    assert not (outside / "x.md").exists()


def test_output_dir_may_not_overlap_source_roots(tmp_path):
    src = tmp_path / "JEE"; src.mkdir()
    for bad in (src, src / "holdout_prep", tmp_path):
        with pytest.raises(PermissionError, match="overlaps source root"):
            check_output_dir(bad, [src])
    assert check_output_dir(tmp_path / "elsewhere" / "holdout_prep", [src]).name == "holdout_prep"


# ------------------------------------------------ F. read-only scanning
def test_scan_never_modifies_source_pdf(tmp_path):
    p = mk(tmp_path / "ok.pdf", ["hello world, this is a long enough body of text"] * 3, {"title": "T", "author": "A"})
    before, mt, size = sha1_file(p), os.stat(p).st_mtime_ns, os.stat(p).st_size
    meta = pymupdf.open(p).metadata
    s = scan_pdf(p)
    assert s["status"] == "ok" and s["n_pages"] == 3 and len(s["pages"]) == 3
    assert sha1_file(p) == before and os.stat(p).st_mtime_ns == mt and os.stat(p).st_size == size
    assert pymupdf.open(p).metadata == meta
    assert sorted(x.name for x in tmp_path.iterdir()) == ["ok.pdf"]            # no sidecar files


def test_corrupt_pdf_is_an_error_not_a_crash(tmp_path):
    bad = tmp_path / "bad.pdf"; bad.write_bytes(b"%PDF-1.4 junk")
    s = scan_pdf(bad)
    assert s["status"] == "error" and s["error"] and s["pages"] == []
    assert scan_pdf(tmp_path / "missing.pdf")["status"] == "error"


def test_scanned_page_identified_in_real_scan(tmp_path):
    s = scan_pdf(noise_pdf(tmp_path / "scan.pdf", seed=3, pages=3))
    assert s["scanned_frac"] == 1.0 and all(p["kind"] == "scanned_page" for p in s["pages"])


def test_blank_document_yields_shortages_not_selections(tmp_path):
    d = pymupdf.open()
    for _ in range(4):
        d.new_page()
    d.save(tmp_path / "blank.pdf")
    s = select_pages(scan_pdf(tmp_path / "blank.pdf"))
    assert not any("figure_candidate" in r for r in roles(s).values()) and len(s["shortages"]) >= 2


def test_holdout_manifest_is_deterministic_and_content_addressed():
    a = doc([rec(1)], name="a.pdf"); a["sha1"] = "abc"
    m1, m2 = holdout_manifest(["a.pdf", "z.pdf"], [a], FP), holdout_manifest(["a.pdf", "z.pdf"], [a], FP)
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert m1["documents"][0]["sha1"] == "abc" and m1["documents"][1]["sha1"] is None
    assert m1["split"] == "heldout_test"
