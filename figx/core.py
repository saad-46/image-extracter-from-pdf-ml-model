"""Core geometry, rendering, annotation-box recovery and crop QA.

Coordinates: all boxes crossing module boundaries are in PDF points of the
page's visible rect (page.rect, rotation already applied), as (x0, y0, x1, y1).
Pixel boxes are only used inside a single rendered image.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np
import pymupdf

Box = tuple[float, float, float, float]


# ----------------------------------------------------------------- geometry
def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a: Box, b: Box) -> float:
    i = intersection(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of `inner` that lies inside `outer`."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def expand(b: Box, d: float) -> Box:
    return (b[0] - d, b[1] - d, b[2] + d, b[3] + d)


def clip(b: Box, w: float, h: float) -> Box:
    return (max(0.0, b[0]), max(0.0, b[1]), min(w, b[2]), min(h, b[3]))


def suppress_duplicates(boxes: list[Box], scores: list[float] | None = None,
                        iou_thr: float = 0.85, contain_thr: float = 0.95) -> list[int]:
    """Keep-indices after removing near-identical boxes.

    Only boxes that are essentially the same region (IoU >= iou_thr, or one
    almost entirely inside the other AND similar in size) are merged, so two
    distinct neighbouring targets that merely overlap both survive.
    """
    scores = scores or [area(b) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep: list[int] = []
    for i in order:
        dup = False
        for k in keep:
            a, b = boxes[i], boxes[k]
            size_ratio = min(area(a), area(b)) / max(area(a), area(b), 1e-9)
            if iou(a, b) >= iou_thr or (
                max(containment(a, b), containment(b, a)) >= contain_thr and size_ratio >= 0.6
            ):
                dup = True
                break
        if not dup:
            keep.append(i)
    return sorted(keep)


# ---------------------------------------------------------------- rendering
def render(page: pymupdf.Page, dpi: float | None = None, size: tuple[int, int] | None = None,
           clip_box: Box | None = None) -> np.ndarray:
    """Render to BGR uint8. Either dpi, or an exact (w, h) pixel size for the full page."""
    r = page.rect
    if size is not None:
        mat = pymupdf.Matrix(size[0] / r.width, size[1] / r.height)
    else:
        z = (dpi or 150) / 72.0
        mat = pymupdf.Matrix(z, z)
    pm = page.get_pixmap(matrix=mat, alpha=False, colorspace=pymupdf.csRGB,
                         clip=pymupdf.Rect(clip_box) if clip_box else None)
    img = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------ red detection
@dataclass
class RedConfig:
    hue_lo: int = 10        # hue <= hue_lo  (OpenCV hue 0..179)
    hue_hi: int = 165       # or hue >= hue_hi  (covers red and maroon/crimson)
    sat_min: int = 80
    val_min: int = 50       # maroon is dark; keep low
    analysis_dpi: int = 100
    min_side_pt: float = 18.0
    min_side_support: float = 0.55   # each side must be >=55% red along its length
    max_interior_fill: float = 0.30  # solid blobs (cartoon clothing) fail this
    max_thickness_frac: float = 0.25


def red_mask(bgr: np.ndarray, cfg: RedConfig = RedConfig()) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h <= cfg.hue_lo) | (h >= cfg.hue_hi)) & (s >= cfg.sat_min) & (v >= cfg.val_min)
    return m.astype(np.uint8)


@dataclass
class FrameCheck:
    ok: bool
    side_support: tuple[float, float, float, float]   # top, bottom, left, right
    interior_fill: float
    thickness_px: int
    reason: str = ""


def verify_frame(mask: np.ndarray, x: int, y: int, w: int, h: int,
                 cfg: RedConfig = RedConfig()) -> FrameCheck:
    """Is the red component inside bbox (x,y,w,h) a rectangular *frame*?

    Signals: four-side support (each border band is mostly red), low interior
    fill (frames are hollow), and a plausible stroke thickness. Rounded corners
    are tolerated because support is measured on the middle 70% of each side.
    """
    sub = mask[y:y + h, x:x + w]
    # thickness: median red run length entering from each side along middle rows/cols
    def run(line: np.ndarray) -> int:
        nz = np.flatnonzero(line == 0)
        return int(nz[0]) if nz.size else len(line)
    mids_r = range(int(h * 0.3), max(int(h * 0.7), int(h * 0.3) + 1))
    mids_c = range(int(w * 0.3), max(int(w * 0.7), int(w * 0.3) + 1))
    runs = [run(sub[r, :]) for r in mids_r] + [run(sub[r, ::-1]) for r in mids_r] + \
           [run(sub[:, c]) for c in mids_c] + [run(sub[::-1, c]) for c in mids_c]
    t = max(1, int(np.median(runs))) if runs else 1
    band = max(t + 2, 3)
    cx0, cx1 = int(w * 0.15), max(int(w * 0.85), int(w * 0.15) + 1)
    cy0, cy1 = int(h * 0.15), max(int(h * 0.85), int(h * 0.15) + 1)
    sup = (
        float(sub[:band, cx0:cx1].max(0).mean()),
        float(sub[h - band:, cx0:cx1].max(0).mean()),
        float(sub[cy0:cy1, :band].max(1).mean()),
        float(sub[cy0:cy1, w - band:].max(1).mean()),
    )
    inner = sub[band + 1:h - band - 1, band + 1:w - band - 1]
    fill = float(inner.mean()) if inner.size else 1.0
    reason = ""
    if min(sup) < cfg.min_side_support:
        reason = f"weak_side_support({min(sup):.2f})"
    elif fill > cfg.max_interior_fill:
        reason = f"solid_interior({fill:.2f})"
    elif t > cfg.max_thickness_frac * min(w, h):
        reason = f"too_thick({t}px)"
    return FrameCheck(not reason, sup, fill, t, reason)


@dataclass
class Annotation:
    page: int                 # 0-based
    box_pt: Box               # outer frame, marked-page points
    inner_pt: Box             # frame interior (border removed), marked-page points
    thickness_pt: float
    method: str               # "diff" | "single"
    check: dict = field(default_factory=dict)
    accepted: bool = True
    reason: str = ""


def _components_to_annotations(mask: np.ndarray, page_idx: int, page_rect: pymupdf.Rect,
                               cfg: RedConfig, method: str) -> list[Annotation]:
    H, W = mask.shape
    sx, sy = page_rect.width / W, page_rect.height / H
    # close small gaps in hand-drawn/compressed borders so a frame is one component
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    out: list[Annotation] = []
    min_px = cfg.min_side_pt / sx
    for k in range(1, n):
        x, y, w, h, _ = (int(v) for v in stats[k])
        if w < min_px or h < min_px:
            continue
        chk = verify_frame(closed, x, y, w, h, cfg)
        t = chk.thickness_px + 1
        box = (x * sx, y * sy, (x + w) * sx, (y + h) * sy)
        inner = ((x + t) * sx, (y + t) * sy, (x + w - t) * sx, (y + h - t) * sy)
        out.append(Annotation(page_idx, box, inner, t * sx, method, asdict(chk), chk.ok, chk.reason))
    acc = [a for a in out if a.accepted]
    keep = set(suppress_duplicates([a.box_pt for a in acc]))
    for i, a in enumerate(acc):
        if i not in keep:
            a.accepted, a.reason = False, "duplicate"
    return out


def recover_annotations(marked: pymupdf.Page, original: pymupdf.Page | None,
                        cfg: RedConfig = RedConfig()) -> tuple[list[Annotation], dict]:
    """Find human-drawn red/maroon frames on a marked page.

    With the unmarked original available ("diff" mode) only red pixels that are
    NEW in the marked page are considered, so red content that already exists in
    the figure (cartoon clothing, red graph lines) cannot become a box. Without
    an original ("single" mode) frame verification alone must reject blobs.
    Returns (annotations, diagnostics) where diagnostics includes the measured
    alignment shift between the two renders.
    """
    W = int(round(marked.rect.width * cfg.analysis_dpi / 72))
    H = int(round(marked.rect.height * cfg.analysis_dpi / 72))
    m_img = render(marked, size=(W, H))
    rm = red_mask(m_img, cfg)
    diag: dict = {"red_px_marked": int(rm.sum())}
    if original is not None:
        o_img = render(original, size=(W, H))
        g1 = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(o_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(g2, g1)
        diag.update(shift_px=(round(dx, 2), round(dy, 2)), shift_response=round(resp, 3))
        if abs(dx) > 1 or abs(dy) > 1:   # re-register original onto marked before diffing
            o_img = cv2.warpAffine(o_img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H),
                                   borderValue=(255, 255, 255))
        ro = cv2.dilate(red_mask(o_img, cfg), np.ones((5, 5), np.uint8))
        rm = rm & (1 - ro)
        diag["red_px_added"] = int(rm.sum())
        method = "diff"
    else:
        method = "single"
    return _components_to_annotations(rm, marked.number, marked.rect, cfg, method), diag


def map_marked_to_original(box: Box, marked: pymupdf.Page, original: pymupdf.Page,
                           shift_px: tuple[float, float] = (0, 0), dpi: float = 100) -> Box:
    """Map a marked-page point box to the original page (scale + measured shift)."""
    sx = original.rect.width / marked.rect.width
    sy = original.rect.height / marked.rect.height
    dx, dy = -shift_px[0] * 72 / dpi, -shift_px[1] * 72 / dpi
    return (box[0] * sx + dx, box[1] * sy + dy, box[2] * sx + dx, box[3] * sy + dy)


# --------------------------------------------------------------- crop + QA
def tighten(img: np.ndarray, pad: int = 6, white: int = 235) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Trim surrounding whitespace, keep `pad` px margin. Returns crop and pixel bbox."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(g < white)
    if xs.size == 0:
        return img, (0, 0, img.shape[1], img.shape[0])
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    return img[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))


def quality_flags(img: np.ndarray, page: pymupdf.Page, box_pt: Box, cfg: RedConfig = RedConfig()) -> dict:
    """Heuristic review flags. They guide human review; they never delete crops."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = float((g < 200).mean())
    edge = 3
    edge_ink = max(float((g[:edge] < 160).mean()), float((g[-edge:] < 160).mean()),
                   float((g[:, :edge] < 160).mean()), float((g[:, -edge:] < 160).mean()))
    rm = red_mask(img, cfg)
    bw = max(2, int(0.06 * min(h, w)))       # annotation remnants live at the crop border
    red_left = max(float(rm[:bw].mean()), float(rm[-bw:].mean()),
                   float(rm[:, :bw].mean()), float(rm[:, -bw:].mean()))
    # text contamination: fraction of crop area covered by text words that are part of long lines
    rect = pymupdf.Rect(box_pt)
    words = page.get_text("words", clip=rect)
    word_area = sum(area((x0, y0, x1, y1)) for x0, y0, x1, y1, *_ in words)
    text_frac = word_area / max(area(box_pt), 1e-9)
    flags = []
    if ink < 0.004:
        flags.append("empty")
    if min(h, w) < 40 or max(h, w) / max(min(h, w), 1) > 12:
        flags.append("tiny_or_narrow")
    if edge_ink > 0.35:
        flags.append("content_touches_edge")
    if red_left > 0.15:
        flags.append("red_border_remains")
    if text_frac > 0.35:
        flags.append("text_heavy")
    if ink < 0.02 and min(h, w) > 200:
        flags.append("mostly_whitespace")
    return {"ink": round(ink, 4), "edge_ink": round(edge_ink, 3), "red_frac": round(red_left, 4),
            "text_frac": round(text_frac, 3), "words": len(words), "flags": flags}


def crop_region(page: pymupdf.Page, box_pt: Box, dpi: int = 300, tight: bool = True) -> tuple[np.ndarray, Box]:
    """Render a page region; optionally trim whitespace. Returns image and final point box."""
    box_pt = clip(box_pt, page.rect.width, page.rect.height)
    img = render(page, dpi=dpi, clip_box=box_pt)
    if not tight:
        return img, box_pt
    img2, (x0, y0, x1, y1) = tighten(img)
    s = 72.0 / dpi
    return img2, (box_pt[0] + x0 * s, box_pt[1] + y0 * s, box_pt[0] + x1 * s, box_pt[1] + y1 * s)


def save_png(img: np.ndarray, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"encode failed: {path}")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(path)
