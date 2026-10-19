"""Read-only page scan used to prepare the held-out marking checklist.

This is a dataset-preparation/audit tool. It is NOT a model and its output is
NOT evidence of extraction accuracy.

Guarantees
- Source PDFs are only opened for reading; nothing here saves or edits them.
- Every output goes through `safe_write_text`, which rejects paths that resolve
  outside the configured output directory (including `..` and symlink escapes).
- The detector is fingerprinted with `detector_fingerprint` and compared with a
  pinned trusted baseline (see FINGERPRINT SPEC below), not with itself.
- Page selections are machine-generated suggestions that need visual confirmation.

FINGERPRINT SPEC (spec id "figx-fp-v2")
- Files: figx/core.py and figx/detect.py (the production detector). scan.py,
  pipeline.py and __init__.py are NOT part of it.
- Source normalisation, applied identically everywhere: decode UTF-8, drop a
  leading BOM, convert CRLF and lone CR to LF, strip trailing spaces/tabs on
  EVERY line, drop trailing empty lines, end with exactly one LF. Indentation,
  comments, blank lines inside the file and all other characters are kept, so
  any meaningful source change (including comment edits) changes the hash.
- Config: default DetectConfig and RedConfig values, serialised as sorted JSON.
  This is taken from the *imported* classes, so it also catches a module that
  was patched at runtime.
- Hash: SHA-256 per file, of the config JSON, and a combined SHA-256 over
  "spec|core.py:<h>|detect.py:<h>|config:<h>".
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from .core import RedConfig, render
from .detect import DetectConfig, boilerplate_keys, detect_page, is_scanned_page, scanned_fraction

FP_SPEC = "figx-fp-v2"
FP_FILES = ("core.py", "detect.py")
WARM_MIN = 0.003           # share of red/orange pixels that makes a page a colour-risk page
MIN_SIG_CHARS = 40         # fewer alphanumeric chars than this = no usable text signature
THUMB_SIDE = 16            # 16x16 average hash = 256 bits
THUMB_HAMMING_MAX = 20     # of 256 bits: below this two sampled pages "look the same"
THUMB_MIN_STD = 4.0        # near-uniform pages (blank/white scans) give meaningless hashes


class MissingFilesError(RuntimeError):
    pass


class FingerprintMismatch(RuntimeError):
    pass


# ------------------------------------------------------------ provenance
def normalize_source(text: str) -> str:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip(" \t") for ln in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def config_json() -> str:
    return json.dumps({"DetectConfig": dataclasses.asdict(DetectConfig()),
                       "RedConfig": dataclasses.asdict(RedConfig())}, sort_keys=True, separators=(",", ":"))


def fingerprint_from_sources(sources: dict[str, str], cfg_json: str) -> dict:
    files = {n: _sha256(normalize_source(sources[n])) for n in FP_FILES}
    cfg = _sha256(cfg_json)
    combined = _sha256(f"{FP_SPEC}|" + "|".join(f"{n}:{files[n]}" for n in FP_FILES) + f"|config:{cfg}")
    return {"spec": FP_SPEC, "files": files, "config": cfg, "combined": combined}


def detector_fingerprint(pkg_dir: Path) -> dict:
    """Fingerprint of the detector source in `pkg_dir` plus the config of the imported classes."""
    pkg_dir = Path(pkg_dir)
    return fingerprint_from_sources({n: (pkg_dir / n).read_text(encoding="utf8") for n in FP_FILES}, config_json())


def compare_fingerprints(actual: dict, expected: dict) -> list[str]:
    """Human-readable list of mismatches (empty list = identical)."""
    out = []
    if actual.get("spec") != expected.get("spec"):
        out.append(f"spec {actual.get('spec')} != {expected.get('spec')}")
    for n in FP_FILES:
        if actual["files"].get(n) != expected["files"].get(n):
            out.append(f"{n} source differs")
    if actual.get("config") != expected.get("config"):
        out.append("default DetectConfig/RedConfig values differ")
    return out


def require_fingerprint(actual: dict, expected: dict, label: str) -> None:
    diff = compare_fingerprints(actual, expected)
    if diff:
        raise FingerprintMismatch(f"{label}: detector does not match the trusted v4 baseline: {'; '.join(diff)}")


def notebook_sources(nb_path: Path, prefix: str) -> dict[str, str]:
    """Extract `%%writefile <prefix><name>` cell bodies from a notebook."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf8"))
    out = {}
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        head, _, body = src.partition("\n")
        if head.startswith("%%writefile " + prefix):
            out[head.split(prefix, 1)[1].strip()] = body
    missing = [n for n in FP_FILES if n not in out]
    if missing:
        raise FingerprintMismatch(f"{nb_path} has no embedded {missing}")
    return out


# ------------------------------------------------------------ file access
def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def locate(targets: list[str], roots: list[Path]) -> dict:
    """Find target filenames under the roots. Never raises; the caller decides."""
    index: dict[str, list[Path]] = {}
    root_report = []
    for r in roots:
        r = Path(r)
        if r.exists():
            pdfs = sorted(r.rglob("*.pdf"))
            root_report.append({"root": str(r), "accessible": True, "pdfs": len(pdfs)})
            for p in pdfs:
                index.setdefault(p.name, []).append(p)
        else:
            root_report.append({"root": str(r), "accessible": False, "pdfs": 0})
    found, ambiguous = {}, {}
    for t in targets:
        if t not in index:
            continue
        paths = index[t]
        if len({sha1_file(p) for p in paths}) > 1:     # same name, different content: refuse to guess
            ambiguous[t] = [str(p) for p in paths]
        else:
            found[t] = paths[0]
    missing = [t for t in targets if t not in index]
    return {"found": found, "missing": missing, "ambiguous": ambiguous, "roots": root_report}


def require_all(loc: dict, required: list[str]) -> None:
    missing = [t for t in required if t in loc["missing"]]
    ambiguous = [t for t in required if t in loc["ambiguous"]]
    if missing or ambiguous:
        roots = "; ".join(f"{r['root']} ({'accessible, %d PDFs' % r['pdfs'] if r['accessible'] else 'NOT accessible'})"
                          for r in loc["roots"])
        parts = []
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        if ambiguous:
            parts.append("Ambiguous (same name, different content): " +
                         "; ".join(f"{t} -> {loc['ambiguous'][t]}" for t in ambiguous))
        raise MissingFilesError(
            f"{len(missing) + len(ambiguous)} required PDF(s) unusable, no checklist produced.\n"
            + "\n".join(parts) + f"\nDrive roots searched: {roots}\n"
            "Add shortcuts to the shared JEE/NEET folders under My Drive, fix the filenames, "
            "or remove the conflicting copy.")


def assert_inside(path: Path, out_dir: Path) -> Path:
    p, o = Path(path).resolve(), Path(out_dir).resolve()
    if o != p and o not in p.parents:
        raise PermissionError(f"refusing to write outside {o}: {p}")
    return p


def check_output_dir(out_dir: Path, source_roots: list[Path]) -> Path:
    """The output dir must not be inside a source root, nor contain one."""
    o = Path(out_dir).resolve()
    for r in source_roots:
        r = Path(r).resolve()
        if o == r or r in o.parents or o in r.parents:
            raise PermissionError(f"output dir {o} overlaps source root {r}")
    return o


def plan_drive_mount(candidates: list[Path]) -> tuple[str, Path]:
    """Decide how to expose Drive without mounting over a non-empty directory.

    Returns ("use", p) if Drive is already mounted at p (contains MyDrive), or
    ("mount", p) for the first candidate that is missing or empty. Raises when
    every candidate is a non-empty directory without MyDrive.
    """
    for c in map(Path, candidates):
        if (c / "MyDrive").exists():
            return "use", c
    for c in map(Path, candidates):
        if not c.exists() or not any(c.iterdir()):
            return "mount", c
    raise RuntimeError("no safe Drive mount point: " + ", ".join(f"{c} is non-empty" for c in candidates))


def safe_write_text(path: Path, text: str, out_dir: Path) -> Path:
    p = assert_inside(path, out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf8")
    tmp.replace(p)
    return p


# ------------------------------------------------------------ page signals
def warm_share(page, dpi=40) -> float:
    """Share of pixels that are red/orange (hue<=25 or >=165, saturated): false-positive risk."""
    img = render(page, dpi=dpi)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float((((h <= 25) | (h >= 165)) & (s >= 90) & (v >= 60)).mean())


def text_signature(doc, n_pages=4) -> str | None:
    """SHA-1 of the first pages' words, or None when there is no meaningful text
    (scanned PDFs). None never counts as a match."""
    words = []
    for p in list(doc)[:n_pages]:
        words += [w[4] for w in p.get_text("words")]
    text = " ".join(words)
    if sum(ch.isalnum() for ch in text) < MIN_SIG_CHARS:
        return None
    return hashlib.sha1(text.encode("utf8", "ignore")).hexdigest()[:12]


def page_ahash(page) -> str | None:
    """16x16 average hash of a low-res render; None for near-uniform pages."""
    g = cv2.cvtColor(render(page, dpi=30), cv2.COLOR_BGR2GRAY)
    if float(g.std()) < THUMB_MIN_STD:
        return None
    g = cv2.resize(g, (THUMB_SIDE, THUMB_SIDE), interpolation=cv2.INTER_AREA)
    bits = "".join("1" if b else "0" for b in (g > g.mean()).flatten())
    return format(int(bits, 2), f"0{THUMB_SIDE * THUMB_SIDE // 4}x")


def sample_pages(n: int) -> list[int]:
    return sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}) if n else []


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def scan_pdf(path: Path, cfg: DetectConfig = DetectConfig()) -> dict:
    """Never raises. status: 'ok' | 'partial' (some pages failed) | 'error' (could not open/scan)."""
    path = Path(path)
    out = {"file": path.name, "path": str(path), "status": "ok", "error": None, "n_pages": 0, "sha1": None,
           "bytes": None, "scanned_frac": 0.0, "text_sig": None, "thumbs": [], "pages": [],
           "pages_ok": 0, "pages_failed": 0}
    try:
        out["sha1"] = sha1_file(path)
        out["bytes"] = path.stat().st_size
        with pymupdf.open(path) as d:           # read-only; never saved
            if d.needs_pass:
                raise ValueError("encrypted")
            out["n_pages"] = len(d)
            boiler = boilerplate_keys(d, cfg)
            sf = scanned_fraction(d, cfg)
            out["scanned_frac"] = round(sf, 2)
            out["text_sig"] = text_signature(d)
            out["thumbs"] = [page_ahash(d[i]) for i in sample_pages(len(d))]
            for i, p in enumerate(d):
                rec = {"page": i + 1, "status": "ok", "error": None}
                try:
                    dets = [x for x in detect_page(p, boiler, cfg, sf) if x["accepted"]]
                    drawings = len(p.get_drawings())
                    has_img = bool(p.get_image_info())
                    kind = ("scanned_page" if is_scanned_page(p, cfg) else
                            "none" if not dets else
                            "raster+vector" if has_img and drawings > 50 else
                            "raster" if has_img else "vector")
                    rec.update(n_figs=len(dets), kind=kind, warm_share=round(warm_share(p), 4),
                               words=len(p.get_text("words")), methods=sorted({x["method"] for x in dets}),
                               boxes=[[round(v) for v in x["box_pt"]] for x in dets])
                except Exception as e:           # one bad page must not hide the rest
                    rec.update(status="error", error=f"{type(e).__name__}: {e}", n_figs=0, kind="unknown",
                               warm_share=0.0, words=0, methods=[], boxes=[])
                out["pages"].append(rec)
        out["pages_failed"] = sum(p["status"] == "error" for p in out["pages"])
        out["pages_ok"] = len(out["pages"]) - out["pages_failed"]
        if out["pages_failed"]:
            out["status"] = "partial"
            out["error"] = f"{out['pages_failed']} of {out['n_pages']} page(s) failed"
    except Exception as e:
        out["status"], out["error"] = "error", f"{type(e).__name__}: {e}"
    return out


# ------------------------------------------------------------ selection
def _spread(items: list, k: int) -> list:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    idx = sorted(set(np.linspace(0, len(items) - 1, k).round().astype(int)))
    return [items[i] for i in idx]


def select_pages(scan: dict, n_fig=8, n_neg=2, n_warm=1, n_scanned=3) -> dict:
    """Machine-generated suggestions. Returns {'selections': [...], 'shortages': [...]}.

    A detector result of 0 never proves a page has no figure; negatives are
    only 'plausible' and the user must confirm visually.
    """
    ok = [p for p in scan["pages"] if p["status"] == "ok"]
    sel: dict[int, dict] = {}
    short: list[str] = []

    def add(p, role, reason):
        e = sel.setdefault(p["page"], {"page": p["page"], "roles": [], "reasons": [], "rec": p})
        e["roles"].append(role)
        e["reasons"].append(reason)

    figs = [p for p in ok if p["n_figs"] > 0]
    for p in _spread(figs, n_fig):
        add(p, "figure_candidate", f"detector found {p['n_figs']} candidate figure(s) ({p['kind']}); spread across the document")
    if len(figs) < n_fig:
        short.append(f"only {len(figs)} page(s) have detector figure candidates (wanted ~{n_fig}); "
                     "the detector may be missing figures on this document, so inspect it by eye")

    scanned = [p for p in ok if p["kind"] == "scanned_page"]
    if scanned:
        for p in _spread([q for q in scanned if q["page"] not in sel], n_scanned):
            add(p, "scanned_sample", "scanned page (no text layer): detector uses the raster fallback, which is "
                                     "unvalidated; mark figures by eye")
    elif scan.get("scanned_frac", 0) > 0:
        short.append("scanned share > 0 but no individual scanned page identified")

    digital_neg = [p for p in ok if p["n_figs"] == 0 and p["words"] > 40 and p["kind"] != "scanned_page"
                   and p["page"] not in sel]
    pool, why = digital_neg, "text page, detector found 0 candidates: PLAUSIBLE negative, not confirmed"
    if len(pool) < n_neg:
        extra = [p for p in ok if p["n_figs"] == 0 and p["page"] not in sel and p not in pool]
        pool = pool + extra
        if extra:
            why = "detector found 0 candidates (low-text/scanned pages included): PLAUSIBLE negative, not confirmed"
    for p in _spread(pool, n_neg):
        add(p, "negative_control", why)
    if len(pool) < n_neg:
        short.append(f"only {len(pool)} plausible negative-control page(s) available (wanted {n_neg})")

    warm = sorted([p for p in ok if p["warm_share"] >= WARM_MIN], key=lambda p: -p["warm_share"])[:n_warm]
    for p in warm:
        add(p, "colour_control", f"red/orange content on {p['warm_share'] * 100:.1f}% of the page "
                                 f"({'detector also found a figure' if p['n_figs'] else 'no detector figure'})")
    if len(warm) < n_warm:
        short.append(f"no page reaches the red/orange threshold ({WARM_MIN * 100:.1f}% of pixels); "
                     "no colour-control page can be suggested")

    bad = [p["page"] for p in scan["pages"] if p["status"] == "error"]
    if bad:
        short.append(f"{len(bad)} page(s) failed to scan and could not be considered: {bad[:20]}")
    return {"selections": [sel[k] for k in sorted(sel)], "shortages": short}


# ------------------------------------------------------------ duplicates
def classify_pair(a: dict, b: dict) -> dict:
    """Evidence-based duplicate classification. Only identical bytes are 'confirmed'.

    Missing evidence (no text signature, uniform pages) is never treated as a match.
    """
    if a["sha1"] and a["sha1"] == b["sha1"]:
        return {"a": a["file"], "b": b["file"], "level": "confirmed_identical_bytes", "evidence": ["identical SHA-1"]}
    ev = []
    sig = a["text_sig"] is not None and a["text_sig"] == b["text_sig"]
    same_pages = a["n_pages"] == b["n_pages"] and a["n_pages"] > 0
    pairs = [(x, y) for x, y in zip(a["thumbs"], b["thumbs"]) if x is not None and y is not None]
    comparable = len(a["thumbs"]) == len(b["thumbs"]) and len(pairs) >= max(1, len(a["thumbs"]) // 2)
    visual = comparable and all(hamming(x, y) <= THUMB_HAMMING_MAX for x, y in pairs)
    if sig:
        ev.append("same first-pages text signature")
    if a["text_sig"] is None or b["text_sig"] is None:
        ev.append("text signature unavailable (no extractable text)")
    if same_pages:
        ev.append(f"same page count ({a['n_pages']})")
    if visual:
        ev.append(f"{len(pairs)} sampled page(s) look visually similar")
    elif not comparable:
        ev.append("too few non-blank sampled pages to compare visually")
    both_text = a["text_sig"] is not None and b["text_sig"] is not None
    if both_text and not sig:
        ev.append("text signatures differ (different content)")      # visual hashes of near-white pages are unreliable
        level = "distinct"
    elif sig and same_pages and visual:
        level = "suspected_strong"
    elif sig:
        level = "suspected_text_only"
    elif same_pages and visual and not both_text:
        level = "suspected_visual_only"                              # only when text evidence is unavailable
    else:
        level = "distinct"
    return {"a": a["file"], "b": b["file"], "level": level, "evidence": ev}


def duplicate_report(scans: list[dict]) -> list[dict]:
    ok = [s for s in scans if s["status"] in ("ok", "partial")]
    out = []
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            c = classify_pair(a, b)
            if c["level"] != "distinct":
                out.append(c)
    return out


def duplicate_wording(d: dict) -> str:
    if d["level"] == "confirmed_identical_bytes":
        return f"CONFIRMED byte-identical: {d['a']} == {d['b']}"
    tag = {"suspected_strong": "STRONG suspected duplicate", "suspected_text_only": "WEAK suspected duplicate (text only)",
           "suspected_visual_only": "WEAK suspected duplicate (visual only)"}[d["level"]]
    return f"{tag} (NOT proof): {d['a']} ~ {d['b']} [{'; '.join(d['evidence'])}]"


# ------------------------------------------------------------ summary + checklist
def build_summary(targets: list[str], loc: dict, scans: list[dict], dups: list[dict],
                  fingerprint_ok: bool | None, extra_warnings=()) -> dict:
    by = {s["file"]: s for s in scans}
    located = [t for t in targets if t in loc["found"]]
    opened = [t for t in located if t in by and by[t]["status"] in ("ok", "partial")]
    partial = [t for t in opened if by[t]["status"] == "partial"]
    failed_open = {t: by[t]["error"] for t in located if t in by and by[t]["status"] == "error"}
    omitted = [t for t in targets if t not in opened]
    s = {
        "expected": len(targets),
        "pdfs_located": len(located),
        "pdfs_opened_and_scanned": len(opened),
        "pdfs_fully_scanned": len(opened) - len(partial),
        "pdfs_with_page_errors": partial,
        "pdfs_failed_to_open": failed_open,
        "pdfs_not_scanned": [t for t in located if t not in by],
        "pdfs_missing": [t for t in targets if t in loc["missing"]],
        "pdfs_ambiguous": [t for t in targets if t in loc.get("ambiguous", {})],
        "omitted": omitted,
        "pages_processed_ok": sum(by[t]["pages_ok"] for t in opened),
        "pages_failed": sum(by[t]["pages_failed"] for t in opened),
        "confirmed_duplicates": [duplicate_wording(d) for d in dups if d["level"] == "confirmed_identical_bytes"],
        "suspected_duplicates": [duplicate_wording(d) for d in dups if d["level"] != "confirmed_identical_bytes"],
        "fingerprint_verified": fingerprint_ok,
        "warnings": list(extra_warnings),
    }
    s["complete"] = not omitted and not partial and s["pages_failed"] == 0 and fingerprint_ok is True
    return s


def summary_lines(s: dict) -> list[str]:
    fp = {True: "VERIFIED against trusted v4 baseline", False: "MISMATCH", None: "NOT CHECKED"}[s["fingerprint_verified"]]
    L = [f"- Status: **{'COMPLETE' if s['complete'] else 'INCOMPLETE: do not rely on this checklist as-is'}**",
         f"- Detector fingerprint: **{fp}**",
         f"- PDFs expected {s['expected']} | located {s['pdfs_located']} | opened & scanned {s['pdfs_opened_and_scanned']}"
         f" | fully scanned {s['pdfs_fully_scanned']} | with page errors {len(s['pdfs_with_page_errors'])}"
         f" | omitted {len(s['omitted'])}",
         f"- Pages processed OK {s['pages_processed_ok']} | pages failed {s['pages_failed']}"]
    if s["pdfs_with_page_errors"]:
        L.append(f"- PDFs with page-level errors: {', '.join(s['pdfs_with_page_errors'])}")
    for f, e in s["pdfs_failed_to_open"].items():
        L.append(f"- Failed to open/scan: {f}: {e}")
    if s["pdfs_missing"]:
        L.append(f"- Missing: {', '.join(s['pdfs_missing'])}")
    if s["pdfs_ambiguous"]:
        L.append(f"- Ambiguous (same name, different content): {', '.join(s['pdfs_ambiguous'])}")
    if s["omitted"]:
        L.append(f"- Omitted: {', '.join(s['omitted'])}")
    L += [f"- {d}" for d in s["confirmed_duplicates"] + s["suspected_duplicates"]] or \
         ["- No duplicate evidence among the scanned PDFs."]
    L += [f"- Warning: {w}" for w in s["warnings"]]
    return L


def render_checklist(summary: dict, scans: list[dict], selections: dict, fingerprint: dict, replacements=()) -> str:
    L = ["# Marking checklist (held-out set)", "",
         "> **MACHINE-GENERATED SUGGESTIONS. Verify every page visually.** Detector results are hints: a page "
         "with 0 detections is *not* proven figure-free, and a detected figure may be wrong. If you see a "
         "figure the detector missed, mark it anyway. Those misses are what the evaluation should reveal. "
         "This scan is a preparation tool, not evidence of extraction accuracy.", "",
         f"Detector fingerprint `{fingerprint['combined'][:16]}` (spec {fingerprint['spec']}).", "",
         "## Run summary", ""] + summary_lines(summary)
    for r in replacements:
        L.append(f"- Replacement SUGGESTION (needs your approval; never applied automatically): {r}")
    L += ["", "How to mark: draw one red/maroon rectangle around EACH target figure on the pages marked "
          "**MARK**. Save as `<name> red boxes marked.pdf` beside an untouched copy of the original in "
          "`Reference_PDFs/`. Do not crop, rotate or reorder pages.", ""]
    for s in scans:
        L += [f"## {s['file']}"]
        if s["status"] == "error":
            L += [f"**Not scanned:** {s['error']}", ""]
            continue
        sel = selections[s["file"]]
        L += [f"{s['n_pages']} pages | scanned share {s['scanned_frac']} | SHA-1 `{s['sha1'][:12]}` | "
              f"text signature {'available' if s['text_sig'] else 'UNAVAILABLE (no extractable text)'}", ""]
        if s["status"] == "partial":
            L += [f"⚠ **Partially scanned:** {s['error']}. Suggestions below ignore the failed pages.", ""]
        if s["scanned_frac"] > 0:
            L += ["⚠ Contains scanned pages. The raster fallback ran but has **not been validated** on real data.", ""]
        for sh in sel["shortages"]:
            L.append(f"- ⚠ Shortage: {sh}")
        if sel["shortages"]:
            L.append("")
        L += ["| Page | Action | Role | Candidates | Visual type | Scan status | Red/orange risk | Why (suggestion) |",
              "|---|---|---|---|---|---|---|---|"]
        for e in sel["selections"]:
            r = e["rec"]
            roles = "+".join(e["roles"])
            if "negative_control" in e["roles"]:
                act = "Leave unmarked ONLY if you confirm no target figure"
            elif "colour_control" in e["roles"]:
                act = "Do NOT box red/orange content that is not a target figure; mark a figure only if it truly is one"
            else:
                act = "MARK every target figure"
            risk = f"{r['warm_share'] * 100:.1f}%" + (" ⚠" if r["warm_share"] >= WARM_MIN else "")
            L.append(f"| {e['page']} | {act} | {roles} | {r['n_figs']} | {r['kind']} | {r['status']} | {risk} | "
                     f"{' / '.join(e['reasons'])} |")
        L.append("")
    return "\n".join(L)


def holdout_manifest(targets: list[str], scans: list[dict], fingerprint: dict) -> dict:
    """Frozen identity of the held-out set (by content hash) for leakage checks in later phases."""
    by = {s["file"]: s for s in scans}
    return {"split": "heldout_test", "purpose": "final evaluation only; never train/tune/select on these",
            "detector_fingerprint": fingerprint["combined"],
            "documents": [{"file": t, "sha1": by[t]["sha1"] if t in by else None,
                           "bytes": by[t]["bytes"] if t in by else None,
                           "n_pages": by[t]["n_pages"] if t in by else None} for t in targets]}
