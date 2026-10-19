"""Figure detection on UNMARKED pages (no human boxes).

PDF-native strategy: figures in these papers are embedded raster images and/or
vector drawings. We collect graphic elements, drop document boilerplate
(logos, barcodes, header rules repeated across pages), drop full-page scans
and table grids, cluster what remains, and attach short label text (axis
names, "10 m/s") that sits inside or right next to a cluster.

Scanned pages (one full-page image) fall back to a raster detector that
removes text-sized connected components and keeps large ink blobs; those
crops are always flagged for review.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np
import pymupdf

from .core import Box, area, containment, expand, iou, render, union


@dataclass
class DetectConfig:
    min_w_pt: float = 30.0
    min_h_pt: float = 22.0
    min_area_pt2: float = 1500.0
    max_page_frac: float = 0.60          # bigger than this = page scan / background
    merge_gap_pt: float = 9.0            # elements closer than this join one figure
    boilerplate_page_frac: float = 0.3   # element repeated on >=30% of pages = boilerplate
    margin_band_frac: float = 0.07       # header/footer band height as page fraction
    label_reach_pt: float = 14.0         # how far outside a cluster a label may sit
    label_max_chars: int = 18
    skip_tables: bool = True
    raster_fallback: bool = True
    max_aspect: float = 9.0              # wider/taller than this = banner/rule strip
    thin_pt: float = 2.5                 # stroke thickness of a "line" element
    scanned_doc_frac: float = 0.5        # raster fallback only if most pages are scans
    inline_max_h_pt: float = 36.0        # short images with text beside them are inline math


def _key(b: Box) -> tuple:
    return tuple(int(round(v / 3)) for v in b)   # 3pt tolerance


def boilerplate_keys(doc: pymupdf.Document, cfg: DetectConfig) -> set:
    """Graphic boxes that recur at the same position on many pages (logos, rules, barcodes)."""
    cnt: Counter = Counter()
    for p in doc:
        H = p.rect.height
        zone = lambda b: b[3] <= 0.15 * H or b[1] >= 0.85 * H   # header/footer zones only
        seen = set()
        for info in p.get_image_info():
            if zone(info["bbox"]):
                seen.add(_key(tuple(info["bbox"])))
        for d in p.get_drawings():
            if zone(d["rect"]):
                seen.add(_key(tuple(d["rect"])))
        cnt.update(seen)
    thr = max(3, int(cfg.boilerplate_page_frac * len(doc)))
    return {k for k, c in cnt.items() if c >= thr}


def _kind(b: Box, cfg: DetectConfig) -> str:
    w, h = b[2] - b[0], b[3] - b[1]
    if h <= cfg.thin_pt and w > cfg.thin_pt:
        return "hline"
    if w <= cfg.thin_pt and h > cfg.thin_pt:
        return "vline"
    return "shape"


def table_regions(lines: list[Box], page_w: float, page_h: float, cfg: DetectConfig) -> list[Box]:
    """Ruled tables from the network of thin horizontal/vertical strokes.

    Tables here are often drawn cell by cell, so strokes are rasterised at
    1px/pt and connected; a network with >=3 distinct horizontal rows and >=3
    distinct vertical columns is a table.
    """
    W, H = int(page_w) + 2, int(page_h) + 2
    m = np.zeros((H, W), np.uint8)
    for b in lines:
        cv2.rectangle(m, (int(b[0]), int(b[1])), (int(np.ceil(b[2])), int(np.ceil(b[3]))), 1, -1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((3, 3), np.uint8)), 8)
    out = []
    for k in range(1, n):
        x, y, w, h, _ = st[k]
        comp = [b for b in lines if lab[min(int(b[1]) + 1, H - 1), min(int(b[0]) + 1, W - 1)] == k]
        rows = {round(b[1] / 3) for b in comp if _kind(b, cfg) == "hline"}
        cols = {round(b[0] / 3) for b in comp if _kind(b, cfg) == "vline"}
        if len(rows) >= 3 and len(cols) >= 3 and w >= 40 and h >= 30:
            out.append((x - 1.5, y - 1.5, x + w + 1.5, y + h + 1.5))
    return out


def _graphic_elements(page: pymupdf.Page, boiler: set, cfg: DetectConfig) -> tuple[list[Box], bool]:
    W, H = page.rect.width, page.rect.height
    pa = W * H
    band = cfg.margin_band_frac * H
    els: list[Box] = []
    scanned = False
    words = page.get_text("words")
    for info in page.get_image_info():
        b = tuple(info["bbox"])
        if area(b) >= cfg.max_page_frac * pa:
            scanned = True
            continue
        if _key(b) in boiler or b[3] <= band or b[1] >= H - band:
            continue
        if _is_inline(b, words, cfg):
            continue
        els.append(b)
    for d in page.get_drawings():
        b = tuple(d["rect"])
        w, h = b[2] - b[0], b[3] - b[1]
        if _key(b) in boiler or b[3] <= band or b[1] >= H - band:
            continue
        if area(b) >= cfg.max_page_frac * pa:
            continue
        if w > 0.75 * W and h < 4:          # full-width separator rule
            continue
        if d.get("fill") is not None and w > 0.5 * W and h > 0.5 * H:   # page background
            continue
        if h <= 12 and w <= 80 and len(d["items"]) > 20:   # text drawn as vector outlines
            continue
        els.append(b)
    return els, scanned


def _is_inline(b: Box, words: list, cfg: DetectConfig) -> bool:
    """Short image sitting on a text line with words immediately left/right of it."""
    if b[3] - b[1] > cfg.inline_max_h_pt:
        return False
    for w in words:
        cy = (w[1] + w[3]) / 2
        if b[1] <= cy <= b[3] and (0 <= b[0] - w[2] <= 15 or 0 <= w[0] - b[2] <= 15):
            return True
    return False


def _cluster(boxes: list[Box], gap: float) -> list[Box]:
    clusters = [b for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        for b in clusters:
            for i, c in enumerate(out):
                if area((max(b[0], c[0] - gap), max(b[1], c[1] - gap),
                         min(b[2], c[2] + gap), min(b[3], c[3] + gap))) > 0:
                    out[i] = union(b, c)
                    changed = True
                    break
            else:
                out.append(b)
        clusters = out
    return clusters


def _table_boxes(page: pymupdf.Page) -> list[Box]:
    try:
        return [tuple(t.bbox) for t in page.find_tables().tables]
    except Exception:          # find_tables is best-effort
        return []


def _attach_labels(page: pymupdf.Page, box: Box, cfg: DetectConfig) -> Box:
    """Grow the box to include short text lines (labels) inside / adjacent to it.

    Long lines are question text and are never absorbed.
    """
    reach = expand(box, cfg.label_reach_pt)
    grown = box
    d = page.get_text("dict", clip=pymupdf.Rect(reach))
    for blk in d.get("blocks", []):
        for ln in blk.get("lines", []):
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            lb = tuple(ln["bbox"])
            if not txt or len(txt) > cfg.label_max_chars:
                continue
            if containment(lb, reach) < 0.9:
                continue
            grown = union(grown, lb)
    return grown


def _is_text_only(page: pymupdf.Page, box: Box) -> bool:
    words = page.get_text("words", clip=pymupdf.Rect(box))
    wa = sum(area(tuple(w[:4])) for w in words)
    return wa > 0.45 * area(box)


def detect_page_native(page: pymupdf.Page, boiler: set, cfg: DetectConfig = DetectConfig()) -> tuple[list[dict], bool]:
    els, scanned = _graphic_elements(page, boiler, cfg)
    tables: list[Box] = []
    if cfg.skip_tables:
        lines = [e for e in els if _kind(e, cfg) in ("hline", "vline")]
        tables = table_regions(lines, page.rect.width, page.rect.height, cfg) + _table_boxes(page)
        els = [e for e in els if not any(containment(e, t) > 0.9 for t in tables)]
    # cluster, remembering which element kinds each cluster holds
    clusters = _cluster(els, cfg.merge_gap_pt)
    dets = []
    for c in clusters:
        members = [e for e in els if containment(e, c) > 0.99]
        kinds = {_kind(e, cfg) for e in members}
        has_image = any(containment(tuple(i["bbox"]), c) > 0.99 for i in page.get_image_info())
        big_side = max((max(e[2] - e[0], e[3] - e[1]) for e in members if _kind(e, cfg) != "hline"), default=0)
        c = _attach_labels(page, c, cfg)
        w, h = c[2] - c[0], c[3] - c[1]
        reason = ""
        if w < cfg.min_w_pt or h < cfg.min_h_pt or area(c) < cfg.min_area_pt2:
            reason = "too_small"
        elif kinds <= {"hline"} or (not has_image and big_side < 20):
            reason = "equation_or_rule"      # fraction bars, vector accents, symbols
        elif max(w, h) / max(min(w, h), 1e-6) > cfg.max_aspect:
            reason = "banner_or_strip"
        elif area(c) > cfg.max_page_frac * page.rect.width * page.rect.height:
            reason = "too_large"
        elif any(containment(c, t) > 0.8 for t in tables):
            reason = "inside_table"
        elif any(iou(c, t) > 0.7 for t in tables):
            reason = "is_table"
        elif _is_text_only(page, c):
            reason = "text_dominated"
        dets.append({"box_pt": c, "accepted": not reason, "reason": reason, "method": "native"})
    # merge clusters that grew into each other after label attachment
    acc = [d for d in dets if d["accepted"]]
    merged = _cluster([d["box_pt"] for d in acc], 0)
    rej = [d for d in dets if not d["accepted"]]
    return [{"box_pt": b, "accepted": True, "reason": "", "method": "native"} for b in merged] + rej, scanned


def detect_page_raster(page: pymupdf.Page, cfg: DetectConfig = DetectConfig(), dpi: int = 100) -> list[dict]:
    """Fallback for scanned pages: big non-text ink blobs."""
    img = render(page, dpi=dpi)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (g < 170).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros_like(ink)
    text_h = 0.22 * dpi   # ~16pt glyph height
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if h > text_h * 1.6 or w > text_h * 6:     # bigger than a glyph/word
            keep[lab == k] = 1
    keep = cv2.dilate(keep, np.ones((15, 15), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(keep, 8)
    s = 72.0 / dpi
    H, W = g.shape
    out = []
    for k in range(1, n):
        x, y, w, h, _ = st[k]
        b = (x * s, y * s, (x + w) * s, (y + h) * s)
        reason = ""
        if w * s < cfg.min_w_pt or h * s < cfg.min_h_pt:
            reason = "too_small"
        elif w > 0.9 * W and h < 0.02 * H:
            reason = "rule"
        elif w * h > cfg.max_page_frac * W * H:
            reason = "too_large"
        out.append({"box_pt": b, "accepted": not reason, "reason": reason, "method": "raster"})
    return out


def is_scanned_page(page: pymupdf.Page, cfg: DetectConfig = DetectConfig()) -> bool:
    pa = page.rect.width * page.rect.height
    big = any(area(tuple(i["bbox"])) >= cfg.max_page_frac * pa for i in page.get_image_info())
    return big and len(page.get_text("words")) < 20


def scanned_fraction(doc: pymupdf.Document, cfg: DetectConfig = DetectConfig()) -> float:
    return sum(is_scanned_page(p, cfg) for p in doc) / max(len(doc), 1)


def detect_page(page: pymupdf.Page, boiler: set, cfg: DetectConfig = DetectConfig(),
                doc_scanned_frac: float = 0.0) -> list[dict]:
    """doc_scanned_frac: share of scanned pages in this document. Isolated scanned
    pages in a digital document are covers/instructions and are skipped."""
    dets, _ = detect_page_native(page, boiler, cfg)
    if (cfg.raster_fallback and doc_scanned_frac >= cfg.scanned_doc_frac
            and is_scanned_page(page, cfg) and not any(d["accepted"] for d in dets)):
        return detect_page_raster(page, cfg)
    return dets
