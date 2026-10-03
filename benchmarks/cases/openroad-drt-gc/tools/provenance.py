#!/usr/bin/env python3
"""Refresh reference/provenance.json of each case: hashes of record.md and of every asset.

    tools/provenance.py          (run after editing a record or after tools/validate.sh)
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ORIGIN = {
    "upstream": "https://github.com/The-OpenROAD-Project/OpenROAD at ec10d069775173bc856608663c11d33c09f3c9b6",
    "study": "drperf profiling of OpenROAD's detailed router on aes and leon3 "
             "(icdslab2.epfl.ch:/home/ubuntu/avadrt-data/drperf_drt, patch scripts and reports); "
             "historical formulas are quoted from those reports",
    "generated": "assets/drperf by tools/validate.sh on this repository's drperf build, "
                 "g++ 13.3, Boost 1.83, glibc 2.39",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for case in sorted(HERE.glob("drt-gc-*")):
    ref = case / "reference"
    record = ref / "record.md"
    assets = sorted((p for p in (ref / "assets").rglob("*") if p.is_file()),
                    key=lambda p: str(p.relative_to(ref / "assets")))
    doc = {
        "record": {"archive_path": "record.md", "start_line": 1,
                   "end_line": record.read_text().count("\n"), "sha256": sha(record)},
        "origin": ORIGIN,
        "assets": [{"archive_path": str(p.relative_to(ref / "assets")), "sha256": sha(p)} for p in assets],
        "warning": "Assets include the reference annotation, the fix and measured formulas. "
                   "Evaluator evidence only; keep this directory out of agent exports.",
    }
    (ref / "provenance.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(case.name, len(assets), "assets")
