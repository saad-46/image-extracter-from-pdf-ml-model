"""End-to-end pipeline: inventory -> ground truth -> evaluation -> bulk extraction.

All state is in JSONL manifests under <out>/manifests so a disconnected Colab
session resumes where it stopped. Crop ids are deterministic
(<pdf sha1[:10]>_p<page>_<gt|fig><n>), so reruns overwrite rather than duplicate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pymupdf

from .core import (RedConfig, area, containment, crop_region, iou, map_marked_to_original,
                   quality_flags, recover_annotations, render, save_png)
from .detect import DetectConfig, boilerplate_keys, detect_page, scanned_fraction

log = logging.getLogger("figx")
MARK_RE = re.compile(r"\s*red\s*box(es)?\s*marked", re.I)
CROP_DPI = 300


# ------------------------------------------------------------------ utils
def sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines() if l.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _r(b) -> list[float]:
    return [round(float(v), 2) for v in b]


def setup_logging(out: Path) -> None:
    (out / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    if not log.handlers:
        for name, lvl in (("extraction.log", logging.INFO), ("errors.log", logging.ERROR)):
            h = logging.FileHandler(out / "logs" / name, encoding="utf8")
            h.setLevel(lvl)
            h.setFormatter(fmt)
            log.addHandler(h)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)


# -------------------------------------------------------------- inventory
def inventory(roots: list[Path], out: Path) -> list[dict]:
    """Discover PDFs, hash them, flag duplicates, marked/original pairs, scans."""
    rows, seen = [], {}
    pdfs = sorted({p.resolve() for r in roots for p in Path(r).rglob("*") if p.suffix.lower() == ".pdf"})
    for p in pdfs:
        row = {"path": str(p), "name": p.name, "exam": next((x for x in ("JEE", "NEET") if x in p.parts), "OTHER"),
               "bytes": p.stat().st_size, "marked": bool(MARK_RE.search(p.stem))}
        try:
            row["sha1"] = sha1(p)
            with pymupdf.open(p) as d:
                if d.needs_pass:
                    raise ValueError("encrypted")
                row.update(pages=len(d), scanned_frac=round(scanned_fraction(d), 2),
                           page_size_pt=_r(d[0].rect[2:]) if len(d) else None)
            row["duplicate_of"] = seen.get(row["sha1"])
            seen.setdefault(row["sha1"], str(p))
            row["status"] = "ok"
        except Exception as e:  # corrupt / encrypted files are reported, not fatal
            row.update(status="error", error=f"{type(e).__name__}: {e}")
        rows.append(row)
    by_stem = {Path(r["path"]).stem.lower(): r for r in rows}
    for r in rows:
        if r["marked"]:
            orig = by_stem.get(MARK_RE.sub("", Path(r["path"]).stem).lower())
            r["original_path"] = orig["path"] if orig else None
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "manifests" / "input_pdfs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    return rows


# ----------------------------------------------------------- ground truth
def run_ground_truth(marked_path: Path, original_path: Path | None, out: Path,
                     cfg: RedConfig = RedConfig(), dpi: int = CROP_DPI) -> list[dict]:
    """Recover human boxes from a marked PDF; crop clean figures from the original."""
    marked_path = Path(marked_path)
    original_path = Path(original_path) if original_path else None
    src_sha = sha1(original_path or marked_path)
    stem = (original_path or marked_path).stem
    rows: list[dict] = []
    with pymupdf.open(marked_path) as M, (pymupdf.open(original_path) if original_path else pymupdf.open()) as O:
        if original_path and len(M) != len(O):
            raise ValueError(f"page count mismatch {len(M)} vs {len(O)}")
        for i in range(len(M)):
            orig_page = O[i] if original_path else None
            anns, diag = recover_annotations(M[i], orig_page, cfg)
            n = 0
            for a in anns:
                base = {"source_pdf": str(original_path or marked_path), "marked_pdf": str(marked_path),
                        "source_sha1": src_sha, "page": i + 1, "method": f"gt_{a.method}",
                        "marked_box_pt": _r(a.box_pt), "frame_thickness_pt": round(a.thickness_pt, 2),
                        "frame_check": a.check, "align_shift_px": diag.get("shift_px")}
                if not a.accepted:
                    rows.append({**base, "status": "rejected", "reason": a.reason})
                    continue
                n += 1
                crop_id = f"{src_sha[:10]}_p{i + 1:03d}_gt{n:02d}"
                page = orig_page or M[i]
                inner = map_marked_to_original(a.inner_pt, M[i], page, diag.get("shift_px") or (0, 0),
                                               cfg.analysis_dpi) if orig_page else a.inner_pt
                img, tight = crop_region(page, inner, dpi=dpi, tight=True)
                q = quality_flags(img, page, tight, cfg)
                path = out / "ground_truth" / "crops" / stem / f"{crop_id}.png"
                save_png(img, path, dpi)
                rows.append({**base, "status": "accepted", "crop_id": crop_id, "crop_path": str(path),
                             "box_inner_pt": _r(inner), "box_tight_pt": _r(tight),
                             "page_size_pt": _r(page.rect[2:]), "crop_px": [img.shape[1], img.shape[0]],
                             "dpi": dpi, "quality": q, "review_status": "needs_review" if q["flags"] else "auto_ok"})
    mf = out / "manifests" / "ground_truth.jsonl"
    keep = [r for r in read_jsonl(mf) if r.get("marked_pdf") != str(marked_path)]   # idempotent rerun
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep + rows), encoding="utf8")
    return rows


# --------------------------------------------------------------- detector
def extract_pdf(pdf: Path, out: Path, dcfg: DetectConfig = DetectConfig(), dpi: int = CROP_DPI,
                pages: list[int] | None = None, subdir: str = "figures") -> tuple[list[dict], dict]:
    pdf = Path(pdf)
    s = sha1(pdf)
    rows, stats = [], {"pages": 0, "candidates": 0, "accepted": 0, "rejected": 0}
    t0 = time.time()
    with pymupdf.open(pdf) as doc:
        boiler = boilerplate_keys(doc, dcfg)
        sf = scanned_fraction(doc, dcfg)
        for i in pages or range(len(doc)):
            page = doc[i]
            stats["pages"] += 1
            try:
                dets = detect_page(page, boiler, dcfg, sf)
            except Exception as e:
                log.error("page failed %s p%d: %s", pdf.name, i + 1, e)
                rows.append({"source_pdf": str(pdf), "source_sha1": s, "page": i + 1, "status": "error",
                             "reason": f"{type(e).__name__}: {e}"})
                continue
            n = 0
            for d in dets:
                stats["candidates"] += 1
                base = {"source_pdf": str(pdf), "source_sha1": s, "page": i + 1, "method": d["method"],
                        "candidate_box_pt": _r(d["box_pt"])}
                if not d["accepted"]:
                    stats["rejected"] += 1
                    rows.append({**base, "status": "rejected", "reason": d["reason"]})
                    continue
                n += 1
                crop_id = f"{s[:10]}_p{i + 1:03d}_fig{n:02d}"
                img, tight = crop_region(page, d["box_pt"], dpi=dpi, tight=True)
                q = quality_flags(img, page, tight)
                if d["method"] == "raster":
                    q["flags"].append("raster_fallback")
                path = out / subdir / pdf.stem / f"{crop_id}.png"
                save_png(img, path, dpi)
                stats["accepted"] += 1
                rows.append({**base, "status": "accepted", "crop_id": crop_id, "crop_path": str(path),
                             "box_tight_pt": _r(tight), "page_size_pt": _r(page.rect[2:]),
                             "crop_px": [img.shape[1], img.shape[0]], "dpi": dpi, "quality": q,
                             "review_status": "needs_review" if q["flags"] else "auto_ok"})
    stats["seconds"] = round(time.time() - t0, 2)
    return rows, stats


def run_bulk(pdfs: list[Path], out: Path, dcfg: DetectConfig = DetectConfig(), dpi: int = CROP_DPI,
             limit: int | None = None) -> dict:
    """Resumable bulk extraction. Completed PDFs (by sha1) are skipped on rerun."""
    setup_logging(out)
    mdir = out / "manifests"
    done = {r["sha1"] for r in read_jsonl(mdir / "completed_pdfs.jsonl")}
    todo = []
    for p in pdfs:
        h = sha1(Path(p))
        if h in done or h in {x[1] for x in todo}:
            continue
        todo.append((Path(p), h))
    todo = todo[:limit] if limit else todo
    log.info("bulk: %d to process, %d already done", len(todo), len(done))
    summary = {"processed": 0, "failed": 0, "accepted": 0}
    for k, (p, h) in enumerate(todo, 1):
        try:
            rows, st = extract_pdf(p, out, dcfg, dpi)
            append_jsonl(mdir / "crop_manifest.jsonl", rows)
            append_jsonl(mdir / "completed_pdfs.jsonl", [{"sha1": h, "path": str(p), **st}])
            summary["processed"] += 1
            summary["accepted"] += st["accepted"]
            log.info("[%d/%d] %s: %d pages, %d crops, %.1fs", k, len(todo), p.name, st["pages"], st["accepted"], st["seconds"])
        except Exception as e:     # one bad PDF never stops the run
            summary["failed"] += 1
            log.error("FAILED %s: %s", p, e)
            append_jsonl(mdir / "failed_files.jsonl", [{"path": str(p), "sha1": h, "error": f"{type(e).__name__}: {e}",
                                                       "trace": traceback.format_exc()[-1500:]}])
    return summary


# ------------------------------------------------------------- evaluation
def evaluate(gt_rows: list[dict], det_rows: list[dict], iou_thr: float = 0.5) -> dict:
    """Match detector crops to ground-truth crops (both as tight content boxes).

    Reports precision/recall at IoU>=iou_thr plus completeness (share of the GT
    content box covered by the matched detection) and contamination (share of
    the detection lying outside the GT box).
    """
    gts = [r for r in gt_rows if r["status"] == "accepted"]
    dets = [r for r in det_rows if r["status"] == "accepted"]
    pages = sorted({(r["page"]) for r in gts} | {r["page"] for r in dets})
    tp, matches, fps, fns = 0, [], [], []
    for pg in pages:
        G = [g for g in gts if g["page"] == pg]
        D = [d for d in dets if d["page"] == pg]
        pairs = sorted(((iou(tuple(d["box_tight_pt"]), tuple(g["box_tight_pt"])), di, gi)
                        for di, d in enumerate(D) for gi, g in enumerate(G)), reverse=True)
        ud, ug = set(), set()
        for v, di, gi in pairs:
            if v < iou_thr or di in ud or gi in ug:
                continue
            ud.add(di); ug.add(gi); tp += 1
            d, g = tuple(D[di]["box_tight_pt"]), tuple(G[gi]["box_tight_pt"])
            matches.append({"page": pg, "iou": round(v, 3), "completeness": round(containment(g, d), 3),
                            "contamination": round(1 - containment(d, g), 3),
                            "gt": G[gi]["crop_id"], "det": D[di]["crop_id"]})
        fps += [{"page": pg, "det": D[i]["crop_id"], "box": D[i]["box_tight_pt"]} for i in range(len(D)) if i not in ud]
        fns += [{"page": pg, "gt": G[i]["crop_id"], "box": G[i]["box_tight_pt"]} for i in range(len(G)) if i not in ug]
    gt_pages = {g["page"] for g in gts}
    neg_fp = [f for f in fps if f["page"] not in gt_pages]
    m = lambda k: round(float(np.mean([x[k] for x in matches])), 3) if matches else None
    return {"gt": len(gts), "det": len(dets), "tp": tp, "fp": len(fps), "fn": len(fns),
            "precision": round(tp / max(len(dets), 1), 3), "recall": round(tp / max(len(gts), 1), 3),
            "mean_iou": m("iou"), "mean_completeness": m("completeness"), "mean_contamination": m("contamination"),
            "fp_on_pages_without_gt": len(neg_fp), "matches": matches, "false_positives": fps, "misses": fns}


# ---------------------------------------------------------------- preview
def contact_sheet(rows: list[dict], path: Path, thumb_h: int = 160, cols: int = 5, max_items: int = 60) -> Path | None:
    items = [r for r in rows if r.get("status") == "accepted"][:max_items]
    if not items:
        return None
    tiles = []
    for r in items:
        im = cv2.imread(r["crop_path"])
        if im is None:
            continue
        s = thumb_h / im.shape[0]
        im = cv2.resize(im, (max(1, min(int(im.shape[1] * s), thumb_h * 3)), thumb_h), interpolation=cv2.INTER_AREA)
        flags = ",".join(r["quality"]["flags"]) if r.get("quality") else ""
        lab = f"p{r['page']} {r['crop_id'][-5:]} {flags}"[:48]
        im = cv2.copyMakeBorder(im, 22, 6, 6, 6, cv2.BORDER_CONSTANT,
                                value=(170, 170, 255) if flags else (200, 230, 200))
        cv2.putText(im, lab, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(im)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    save_png(sheet, Path(path), 72)
    return Path(path)


# ------------------------------------------------------------ YOLO export
def export_yolo(gt_rows: list[dict], out: Path, dpi: int = 150, val_pages: set | None = None,
                negatives_per_pdf: int = 5, seed: int = 0) -> Path:
    """Page images + normalized labels from REVIEWED/accepted ground truth only.

    Also exports a few pages without targets as negatives. Split is by page here
    because only one marked PDF exists; with more marked PDFs split by document.
    """
    rng = np.random.default_rng(seed)
    acc = [r for r in gt_rows if r["status"] == "accepted" and r.get("review_status") != "rejected"]
    root = out / "dataset_yolo"
    by_pdf: dict = {}
    for r in acc:
        by_pdf.setdefault(r["source_pdf"], []).append(r)
    for pdf, rs in by_pdf.items():
        with pymupdf.open(pdf) as doc:
            pos = sorted({r["page"] for r in rs})
            negs = [p for p in range(1, len(doc) + 1) if p not in pos]
            negs = list(rng.choice(negs, size=min(negatives_per_pdf, len(negs)), replace=False)) if negs else []
            for pg in pos + [int(n) for n in negs]:
                split = "val" if val_pages and pg in val_pages else "train"
                page = doc[pg - 1]
                img = render(page, dpi=dpi)
                stem = f"{Path(pdf).stem.replace(' ', '_')}_p{pg:03d}"
                save_png(img, root / "images" / split / f"{stem}.png", dpi)
                W, H = page.rect.width, page.rect.height
                lines = []
                for r in (x for x in rs if x["page"] == pg):
                    x0, y0, x1, y1 = r["box_tight_pt"]
                    cx, cy, bw, bh = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H
                    assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1, r["crop_id"]
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                lp = root / "labels" / split / f"{stem}.txt"
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_text("\n".join(lines) + ("\n" if lines else ""))
    (root / "data.yaml").write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames: ['figure']\n")
    return root


# -------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="JEE/NEET figure extraction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inventory"); a.add_argument("roots", nargs="+"); a.add_argument("--out", required=True)
    g = sub.add_parser("groundtruth"); g.add_argument("marked"); g.add_argument("--original"); g.add_argument("--out", required=True)
    b = sub.add_parser("bulk"); b.add_argument("roots", nargs="+"); b.add_argument("--out", required=True); b.add_argument("--limit", type=int)
    e = sub.add_parser("evaluate"); e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    setup_logging(out)
    if args.cmd == "inventory":
        rows = inventory([Path(r) for r in args.roots], out)
        print(json.dumps({"pdfs": len(rows), "unique": len({r.get('sha1') for r in rows}),
                          "marked": sum(r["marked"] for r in rows)}, indent=1))
    elif args.cmd == "groundtruth":
        rows = run_ground_truth(Path(args.marked), Path(args.original) if args.original else None, out)
        print(sum(r["status"] == "accepted" for r in rows), "accepted;", sum(r["status"] == "rejected" for r in rows), "rejected")
    elif args.cmd == "bulk":
        pdfs = [p for r in args.roots for p in Path(r).rglob("*.pdf") if not MARK_RE.search(p.stem)]
        print(run_bulk(sorted(pdfs), out, limit=args.limit))
    elif args.cmd == "evaluate":
        gt = read_jsonl(out / "manifests" / "ground_truth.jsonl")
        res = {}
        for src in sorted({r["source_pdf"] for r in gt}):
            det, _ = extract_pdf(Path(src), out, subdir="eval_detections")
            res[src] = evaluate([r for r in gt if r["source_pdf"] == src], det)
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "evaluation.json").write_text(json.dumps(res, indent=1))
        for k, v in res.items():
            print(Path(k).name, {x: v[x] for x in ("gt", "det", "tp", "fp", "fn", "precision", "recall", "mean_iou",
                                                    "mean_completeness", "mean_contamination", "fp_on_pages_without_gt")})


if __name__ == "__main__":
    main()
