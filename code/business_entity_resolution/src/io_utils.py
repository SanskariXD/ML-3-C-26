"""Robust TSV I/O and submission writing.

Why a hand-rolled reader instead of pandas.read_csv?
  * QUOTE handling: names contain ' and " characters; CSV quoting rules would silently
    merge lines. We split on TAB only.
  * A row with a stray TAB inside the address must NOT be dropped (a dropped S1 row
    is a rejected submission). We repair it: extra fields are glued back into the
    address column.
  * CRLF / BOM / trailing whitespace are stripped from every field.
The cost is ~5-10 s per 500 MB file, paid once (results are cached as parquet).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd

from utils import LOG

COLS = ("entity_id", "business_name", "business_address", "country")
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def read_source(path: str) -> pd.DataFrame:
    ids, names, addrs, ctrs = [], [], [], []
    repaired = padded = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        header = [h.strip().lower() for h in f.readline().rstrip("\r\n").split("\t")]
        if "entity_id" not in header:
            raise ValueError(f"{path}: header {header} has no entity_id column")
        pos = {c: (header.index(c) if c in header else -1) for c in COLS}
        ncol = len(header)
        standard = [pos[c] for c in COLS] == [0, 1, 2, 3] and ncol == 4
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            p = line.split("\t")
            if len(p) > ncol and standard:          # stray TAB(s) inside the address
                p = [p[0], p[1], " ".join(p[2:-1]), p[-1]]
                repaired += 1
            elif len(p) < ncol:
                p = p + [""] * (ncol - len(p))
                padded += 1
            ids.append(p[pos["entity_id"]].strip())
            names.append(p[pos["business_name"]].strip() if pos["business_name"] >= 0 else "")
            addrs.append(p[pos["business_address"]].strip() if pos["business_address"] >= 0 else "")
            ctrs.append(p[pos["country"]].strip() if pos["country"] >= 0 else "")
    df = pd.DataFrame({"entity_id": ids, "business_name": names,
                       "business_address": addrs, "country": ctrs})
    n0 = len(df)
    df = df[df["entity_id"] != ""]
    df = df.drop_duplicates("entity_id", keep="first").reset_index(drop=True)
    if repaired or padded or len(df) != n0:
        LOG.warning("%s: repaired=%d padded=%d dropped(empty/dup id)=%d",
                    os.path.basename(path), repaired, padded, n0 - len(df))
    LOG.info("read %-24s rows=%d", os.path.basename(path), len(df))
    return df


def read_ground_truth(path: str) -> dict:
    gt = {}
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            s1, _, rest = line.partition("\t")
            ids = [x.strip() for x in rest.split(",") if x.strip()]
            gt[s1.strip()] = ids
    LOG.info("ground truth: %d S1 rows, %d matched ids", len(gt), sum(map(len, gt.values())))
    return gt


def _clean_list(ids):
    """Dedupe (order-preserving) and keep only S2-/S3- ids."""
    seen = set()
    out = []
    for x in ids:
        x = x.strip()
        if x and x not in seen and (x.startswith("S2-") or x.startswith("S3-")):
            seen.add(x)
            out.append(x)
    return out


def write_id_lists(path: str, header, s1_ids, q_sorted: np.ndarray, id_strings: np.ndarray,
                   order_score: np.ndarray | None = None) -> int:
    """Stream one row per S1 id.

    q_sorted:  S1 row index (into s1_ids) of every selected pair, any order.
    id_strings: S2/S3 entity-id string of every selected pair (aligned with q_sorted).
    order_score: optional score used to order ids inside a row (desc).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = len(s1_ids)
    if len(q_sorted):
        key2 = -order_score if order_score is not None else np.zeros(len(q_sorted))
        order = np.lexsort((key2, q_sorted))
        q = q_sorted[order]
        ids = id_strings[order]
        starts = np.searchsorted(q, np.arange(n), "left")
        ends = np.searchsorted(q, np.arange(n), "right")
    else:
        ids = np.array([], dtype=object)
        starts = ends = np.zeros(n, dtype=np.int64)
    n_nonempty = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        buf = []
        for r in range(n):
            a, b = starts[r], ends[r]
            lst = _clean_list(ids[a:b].tolist()) if b > a else []
            n_nonempty += bool(lst)
            buf.append(f"{s1_ids[r]}\t{','.join(lst)}\n")
            if len(buf) >= 50_000:
                f.write("".join(buf))
                buf.clear()
        f.write("".join(buf))
    LOG.info("wrote %s: %d rows (%d non-empty)", path, n, n_nonempty)
    return n_nonempty


def self_check(path: str, header, required_ids) -> list:
    """Lightweight re-implementation of the organisers' gate (fast, no ID lookup)."""
    errs = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        h = [c.strip().lower() for c in f.readline().rstrip("\n").split("\t")]
        if tuple(h) != tuple(header):
            errs.append(f"bad header {h}")
        for ln, line in enumerate(f, 2):
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                errs.append(f"line {ln}: no tab")
                continue
            if s1 in seen:
                errs.append(f"duplicate row {s1}")
            seen.add(s1)
            ids = rest.split(",") if rest else []
            if len(ids) != len(set(ids)):
                errs.append(f"{s1}: duplicate ids")
            if any(not (x.startswith("S2-") or x.startswith("S3-")) for x in ids):
                errs.append(f"{s1}: bad prefix")
            if len(errs) > 20:
                break
    missing = set(required_ids) - seen
    extra = seen - set(required_ids)
    if missing:
        errs.append(f"{len(missing)} required S1 ids missing")
    if extra:
        errs.append(f"{len(extra)} unknown S1 ids")
    return errs


def run_official_validator(validator: str, matching: str, candidate: str, test_dir: str) -> int:
    if not validator or not os.path.isfile(validator):
        LOG.warning("official validator not found (%s) - skipped", validator)
        return 0
    cmd = [sys.executable, validator, "--matching", matching, "--candidate", candidate,
           "--test-dir", test_dir]
    LOG.info("running official validator: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (res.stdout + res.stderr).splitlines():
        LOG.info("  [validator] %s", line)
    return res.returncode
