"""Pin the trusted detector baseline from the PRODUCTION NOTEBOOK, not from the local files.

    python tools/make_baseline.py jee_neet_figx_v4.ipynb   ->  figx/baseline_v4.json

The baseline is computed from the core.py/detect.py cell bodies embedded in
jee_neet_figx_v4.ipynb (the notebook whose detector produced the reference
evaluation). Their default DetectConfig/RedConfig values are read by importing
those extracted sources in a clean subprocess, so the config part does not come
from the local working copy either.

`verify` compares three independent copies against that pinned JSON and fails
loudly on any difference:  local figx/, the v4 notebook, and holdout_scan.ipynb.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figx import scan  # noqa: E402

BASELINE = ROOT / "figx" / "baseline_v4.json"
V4_PREFIX = "/content/figx/"
HOLDOUT_PREFIX = "/content/holdout_pkg/figx/"

CFG_SNIPPET = (
    "import sys, json, dataclasses\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from figx.detect import DetectConfig\n"
    "from figx.core import RedConfig\n"
    "print(json.dumps({'DetectConfig': dataclasses.asdict(DetectConfig()), 'RedConfig': dataclasses.asdict(RedConfig())},"
    " sort_keys=True, separators=(',', ':')))\n"
)


def config_json_of_sources(sources: dict[str, str]) -> str:
    """Default-config JSON of the detector in `sources`, evaluated in a clean interpreter."""
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "figx"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        for n, body in sources.items():
            (pkg / n).write_text(body, encoding="utf8")
        r = subprocess.run([sys.executable, "-I", "-c", CFG_SNIPPET, td], capture_output=True, text=True, check=True)
        return r.stdout.strip()


def fingerprint_of_notebook(nb: Path, prefix: str) -> dict:
    src = scan.notebook_sources(nb, prefix)
    return scan.fingerprint_from_sources(src, config_json_of_sources({n: src[n] for n in scan.FP_FILES}))


def make(nb_v4: Path) -> dict:
    fp = fingerprint_of_notebook(nb_v4, V4_PREFIX)
    fp["source"] = {"notebook": nb_v4.name, "notebook_sha256": hashlib.sha256(nb_v4.read_bytes()).hexdigest()}
    BASELINE.write_text(json.dumps(fp, indent=1), encoding="utf8")
    return fp


def verify(nb_v4: Path, nb_holdout: Path) -> dict:
    pinned = json.loads(BASELINE.read_text(encoding="utf8"))
    copies = {"local figx/": scan.detector_fingerprint(ROOT / "figx"),
              "v4 notebook (embedded)": fingerprint_of_notebook(nb_v4, V4_PREFIX),
              "holdout_scan notebook (embedded)": fingerprint_of_notebook(nb_holdout, HOLDOUT_PREFIX)}
    problems = []
    for name, fp in copies.items():
        diff = scan.compare_fingerprints(fp, pinned)
        print(f"{name:34s} {fp['combined'][:16]}  {'OK' if not diff else 'MISMATCH: ' + '; '.join(diff)}")
        problems += [f"{name}: {d}" for d in diff]
    print(f"{'pinned baseline (v4)':34s} {pinned['combined'][:16]}")
    if problems:
        raise scan.FingerprintMismatch("; ".join(problems))
    return pinned


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "verify":
        verify(ROOT / "jee_neet_figx_v4.ipynb", ROOT / "holdout_scan.ipynb")
    else:
        print(json.dumps(make(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "jee_neet_figx_v4.ipynb")), indent=1))
