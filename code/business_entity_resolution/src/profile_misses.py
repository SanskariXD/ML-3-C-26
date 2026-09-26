#!/usr/bin/env python3
"""Profile the blocking misses: WHY did these true pairs never become candidates?

Answers the question error_analysis.py raises -- 63% of errors are blocking misses,
and the printed examples all look like Devanagari. This measures it instead of
eyeballing it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import load_pairs                      # noqa: E402
from config import Config                            # noqa: E402
from prepare import load_gt, load_split              # noqa: E402
from utils import crc_fold                           # noqa: E402


def sa(s: str) -> str:
    n = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in n if not unicodedata.combining(c))


def toks(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", sa(s).lower()))


def is_nonlatin(s: str) -> bool:
    return not re.search(r"[a-z]", sa(s or "").lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work_dev")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.work_dir, "models", "meta.json")))
    cfg = Config.from_dict(meta["config"])
    cfg.work_dir = a.work_dir

    s1, idx, parts = load_split(cfg, "train", columns=[
        "entity_id", "business_name", "business_address", "country"])
    n_s1 = len(s1)
    fold_s1 = crc_fold(s1["entity_id"].tolist())
    gt_s1, gt_idx = load_gt(cfg)

    # every candidate pair actually generated, as a set of (s1_row, idx_row)
    cand = set()
    for ck, (s1_rows, idx_rows) in parts.items():
        P = load_pairs(cfg, "train", ck)
        qg = s1_rows[P["q"]].astype(np.int64)
        ig = idx_rows[P["i"]].astype(np.int64)
        cand.update(zip(qg.tolist(), ig.tolist()))

    hold = set(cfg.holdout_folds)
    n1 = s1["business_name"].to_numpy(dtype=object)
    a1 = s1["business_address"].to_numpy(dtype=object)
    c1 = s1["country"].to_numpy(dtype=object)
    n2 = idx["business_name"].to_numpy(dtype=object)
    a2 = idx["business_address"].to_numpy(dtype=object)

    prof = defaultdict(Counter)
    tot = Counter()
    for q_, i_ in zip(gt_s1.tolist(), gt_idx.tolist()):
        if fold_s1[q_] not in hold:
            continue
        found = (q_, i_) in cand
        bucket = "FOUND" if found else "MISSED"
        tot[bucket] += 1
        c = c1[q_]
        tot[f"{bucket}|{c}"] += 1

        nm1, nm2 = n1[q_], n2[i_]
        ad1, ad2 = a1[q_], a2[i_]
        tn = toks(nm1) & toks(nm2)
        ta = toks(ad1) & toks(ad2)
        p = prof[bucket]
        if is_nonlatin(nm2):
            p["name_non_latin"] += 1
        if not (ad2 or "").strip():
            p["addr_empty"] += 1
        if not tn:
            p["name_no_shared_token"] += 1
        if ad2 and not ta:
            p["addr_no_shared_token"] += 1
        if not tn and (not ad2 or not ta):
            p["NO_SHARED_TOKEN_AT_ALL"] += 1
        nums1 = set(re.findall(r"\d+", ad1 or ""))
        nums2 = set(re.findall(r"\d+", ad2 or ""))
        if nums1 and nums2 and not (nums1 & nums2):
            p["addr_numbers_disjoint"] += 1

    print(f"holdout true pairs: FOUND={tot['FOUND']:,}  MISSED={tot['MISSED']:,}  "
          f"recall={tot['FOUND']/(tot['FOUND']+tot['MISSED']):.4%}\n")
    for c in ("India", "US", "France"):
        f, m = tot[f"FOUND|{c}"], tot[f"MISSED|{c}"]
        if f + m:
            print(f"  {c:8s} recall={f/(f+m):.4%}  ({m:,} missed of {f+m:,})")

    keys = ["name_non_latin", "name_no_shared_token", "addr_empty",
            "addr_no_shared_token", "addr_numbers_disjoint", "NO_SHARED_TOKEN_AT_ALL"]
    print(f"\n{'characteristic':26s} {'MISSED':>10s} {'FOUND':>10s}   {'lift':>6s}")
    print("-" * 60)
    for k in keys:
        pm = prof["MISSED"][k] / max(1, tot["MISSED"])
        pf = prof["FOUND"][k] / max(1, tot["FOUND"])
        lift = pm / pf if pf else float("inf")
        print(f"{k:26s} {pm:9.2%} {pf:9.2%}   {lift:5.1f}x")
    print("\n'lift' = how over-represented a trait is among MISSED pairs. High lift = the "
          "trait blocking cannot handle.")


if __name__ == "__main__":
    main()
