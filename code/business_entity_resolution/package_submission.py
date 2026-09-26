"""Build <team>_submission.zip in the exact layout the organisers require.

  python package_submission.py --team MyTeam --output-dir output \
      --doc ../Documentation_template.md

<team>_submission.zip
├── output/matching_results.tsv, output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, tests/, README.md, requirements*.txt, package_submission.py}
└── Documentation_template.md
"""
from __future__ import annotations

import argparse
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))          # code/business_entity_resolution
PROJ = os.path.abspath(os.path.join(HERE, "..", ".."))     # project root
# work_dev is the dev-scale cache; never ship caches or the venv
SKIP_DIRS = {"__pycache__", "work", "work_dev", "output", ".venv", ".git",
             ".ipynb_checkpoints", ".pytest_cache"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--output-dir", default=os.path.join(PROJ, "output"))
    ap.add_argument("--doc", default=os.path.join(PROJ, "student_resource",
                                                 "Documentation_template.md"))
    ap.add_argument("--dest", default=PROJ)
    a = ap.parse_args()

    need = [os.path.join(a.output_dir, "matching_results.tsv"),
            os.path.join(a.output_dir, "candidate_pairs.tsv"), a.doc]
    for p in need:
        if not os.path.isfile(p):
            raise SystemExit(f"missing required file: {p}")
    team = "".join(c if c.isalnum() or c in "-_" else "_" for c in a.team)
    zpath = os.path.abspath(os.path.join(a.dest, f"{team}_submission.zip"))
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(need[0], "output/matching_results.tsv")
        z.write(need[1], "output/candidate_pairs.tsv")
        z.write(a.doc, "Documentation_template.md")
        for root, dirs, files in os.walk(HERE):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fn in files:
                if fn.endswith((".pyc", ".zip", ".log")):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, HERE).replace(os.sep, "/")
                z.write(full, f"code/business_entity_resolution/{rel}")
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    for req in ("output/matching_results.tsv", "output/candidate_pairs.tsv",
                "Documentation_template.md", "code/business_entity_resolution/README.md",
                "code/business_entity_resolution/requirements.txt",
                "code/business_entity_resolution/src/run.py"):
        assert req in names, f"zip is missing {req}"
    print(f"wrote {zpath} ({os.path.getsize(zpath) / 2**20:.1f} MB, {len(names)} files)")


if __name__ == "__main__":
    main()
