"""Generate holdout_scan.ipynb. The pinned trusted baseline is embedded as EXPECTED."""
import json
from pathlib import Path

BASELINE = json.loads(Path("figx/baseline_v4.json").read_text(encoding="utf8"))
EXPECTED = {k: BASELINE[k] for k in ("spec", "files", "config", "combined")}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip().splitlines(True)}


cells = [
    md("""# Held-out marking checklist (read-only scan)
Dataset-preparation / audit tool. **Not a model, and not evidence of extraction accuracy.** Output is machine-generated suggestions to verify visually.

Enforced in code:
- Source PDFs are opened read-only; nothing is saved back to them.
- Everything is written only under `My Drive/JEE_NEET_Extracted_Images/v4_figx/holdout_prep/` (paths are resolved first; anything outside raises `PermissionError`, and the output folder may not overlap a source folder).
- The embedded detector (`core.py` + `detect.py`) is fingerprinted and compared with the **trusted v4 baseline pinned inside this notebook**; any difference stops the run. The detector is never tuned on these files.
- All six required PDFs must exist and be unambiguous, otherwise the run stops with the reasons.
- No training, no final evaluation, no uploads.

Run Runtime > Run all (CPU is enough)."""),
    code(r"""import subprocess, sys
# Errors propagate: check=True raises, and pip's output is shown only on failure.
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pymupdf>=1.24', 'opencv-python-headless', 'numpy'],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:], r.stderr[-3000:])
    raise RuntimeError(f'dependency installation failed (pip exit code {r.returncode})')
import importlib
for mod in ('pymupdf', 'cv2', 'numpy'):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        raise RuntimeError(f'critical import failed after install: {mod}: {e}') from e
    print('ok', mod, getattr(m, '__version__', getattr(m, 'VersionBind', '?')))"""),
    code(r"""import shutil, os
shutil.rmtree('/content/holdout_pkg', ignore_errors=True)    # fresh package dir: never reuse a stale figx
os.makedirs('/content/holdout_pkg/figx')"""),
]
for n in ["__init__", "core", "detect", "scan"]:
    cells.append(code("%%writefile /content/holdout_pkg/figx/" + n + ".py" + chr(10) + Path("figx/" + n + ".py").read_text(encoding="utf8")))

cells += [
    code(f"""import sys
for m in [m for m in sys.modules if m == 'figx' or m.startswith('figx.')]:
    del sys.modules[m]                                   # never reuse a previously imported figx
sys.path.insert(0, '/content/holdout_pkg')
from pathlib import Path
import figx, figx.scan as scan
assert Path(figx.__file__).resolve().parent == Path('/content/holdout_pkg/figx'), figx.__file__

# Trusted v4 baseline, pinned at build time from jee_neet_figx_v4.ipynb (NOT computed from these files).
TRUSTED_V4 = {json.dumps(EXPECTED)}
FP = scan.detector_fingerprint(Path('/content/holdout_pkg/figx'))
scan.require_fingerprint(FP, TRUSTED_V4, 'embedded detector')      # raises FingerprintMismatch on any difference
FINGERPRINT_OK = True
print('Detector fingerprint VERIFIED against trusted v4 baseline:', FP['combined'][:16])"""),
    code(r"""action, M = scan.plan_drive_mount([Path('/content/drive'), Path('/content/gdrive')])
print('Drive:', action, M)
if action == 'mount':
    from google.colab import drive
    drive.mount(str(M))                                  # only ever mounts at an absent/empty directory
DRIVE = M / 'MyDrive'
assert DRIVE.exists(), f'{DRIVE} not found after mount'"""),
    code(r"""ROOTS = [DRIVE / 'JEE', DRIVE / 'NEET', DRIVE / 'Reference_PDFs']
TARGETS = ['Copy of original (24).pdf', 'Copy of original (33).pdf', 'Copy of original (41).pdf',
           'Copy of NEET 2016.pdf', 'Copy of NEET 2024.pdf', 'Copy of NEET 2005.pdf']
JEE_FALLBACKS = ['Copy of original (14).pdf', 'Copy of original (46).pdf', 'Copy of original (11).pdf',
                 'Copy of original (12).pdf', 'Copy of original (5).pdf', 'Copy of original (6).pdf']
OUT = scan.check_output_dir(DRIVE / 'JEE_NEET_Extracted_Images' / 'v4_figx' / 'holdout_prep', ROOTS)

loc = scan.locate(TARGETS + JEE_FALLBACKS, ROOTS)
for r in loc['roots']:
    print(('OK      ' if r['accessible'] else 'MISSING ') + r['root'], '-', r['pdfs'], 'PDFs')
scan.require_all(loc, TARGETS)       # raises MissingFilesError listing missing / ambiguous files
print('All', len(TARGETS), 'required PDFs located and unambiguous. Output ->', OUT)"""),
    code(r"""import time
scans = []
for t in TARGETS:
    t0 = time.time()
    s = scan.scan_pdf(loc['found'][t])
    scans.append(s)
    print(f"{t}: {s['status']} | {s['n_pages']} pages ({s['pages_ok']} ok, {s['pages_failed']} failed) | scanned share {s['scanned_frac']} | "
          f"{sum(p.get('n_figs', 0) > 0 for p in s['pages'])} pages with candidates | {time.time() - t0:.1f}s", s['error'] or '')"""),
    code(r"""dups = scan.duplicate_report(scans)
for d in dups:
    print(scan.duplicate_wording(d))
if not dups:
    print('No duplicate evidence among the six.')

# Replacement SUGGESTIONS only (never applied). Triggered by confirmed OR suspected duplicates.
repl_scans, replacements, extra_warnings = [], [], []
for d in dups:
    victim = max(d['a'], d['b'], key=TARGETS.index)
    chosen = None
    for fb in [f for f in JEE_FALLBACKS if f in loc['found'] and f not in [s['file'] for s in repl_scans]]:
        cand = scan.scan_pdf(loc['found'][fb])
        others = [s for s in scans if s['file'] != victim] + repl_scans
        clash = [x for x in scan.duplicate_report(others + [cand]) if fb in (x['a'], x['b'])]
        if cand['status'] in ('ok', 'partial') and not clash:
            chosen = cand
            break
    if chosen:
        repl_scans.append(chosen)
        replacements.append(f"replace {victim} with {chosen['file']} (no duplicate evidence against the other scanned files). "
                            f"Reason: {scan.duplicate_wording(d)}")
    else:
        extra_warnings.append(f"duplicate evidence for {victim} but no usable distinct JEE fallback found among {JEE_FALLBACKS}")
print(replacements or 'No replacement suggested.', extra_warnings)"""),
    code(r"""import json
all_scans = scans + repl_scans
selections = {s['file']: scan.select_pages(s) if s['status'] != 'error' else {'selections': [], 'shortages': [s['error']]}
              for s in all_scans}
summary = scan.build_summary(TARGETS, loc, scans, dups, FINGERPRINT_OK, extra_warnings)
md_text = scan.render_checklist(summary, all_scans, selections, FP, replacements)
scan.safe_write_text(OUT / 'marking_checklist.md', md_text, OUT)
payload = {'detector_fingerprint': FP, 'trusted_baseline': TRUSTED_V4['combined'], 'summary': summary, 'duplicates': dups,
           'replacement_suggestions': replacements,
           'selections': {f: {'shortages': v['shortages'],
                              'pages': [{'page': e['page'], 'roles': e['roles'], 'reasons': e['reasons']} for e in v['selections']]}
                          for f, v in selections.items()},
           'scans': all_scans}
scan.safe_write_text(OUT / 'page_scan.json', json.dumps(payload, indent=1), OUT)
scan.safe_write_text(OUT / 'holdout_manifest.json', json.dumps(scan.holdout_manifest(TARGETS, scans, FP), indent=1), OUT)

print('=' * 64)
print('COMPLETION SUMMARY')
print('\n'.join(scan.summary_lines(summary)))
for f, v in selections.items():
    for sh in v['shortages']: print(f'  shortage [{f}]:', sh)
print('Written to:', OUT)
print('=' * 64)
from IPython.display import Markdown, display
display(Markdown(md_text))"""),
    md("Suggestions only: a page with 0 detections is not proven figure-free, and missed or wrong detections are what the later evaluation should reveal. Mark copies, never the source PDFs. Replacement suggestions need your explicit approval."),
]
nb = {"cells": cells, "metadata": {"colab": {"name": "holdout_scan.ipynb"}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                  "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path("holdout_scan.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf8")
for c in nb["cells"]:
    if c["cell_type"] == "code":
        s = "".join(c["source"])
        if s.startswith("%%"):
            continue
        compile(s.replace("from google.colab import drive", "drive=None"), "cell", "exec")
print("built", len(cells), "cells; trusted baseline", EXPECTED["combined"][:16])
