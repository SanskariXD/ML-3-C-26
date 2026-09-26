#!/usr/bin/env python3
"""Diagnostics for the Business Entity Resolution challenge.

Answers the only four questions that matter when a submission plateaus:

  1. score    -- what is my exact macro F0.5, broken down BY COUNTRY, and how much
                 of the loss is false positives vs false negatives?
  2. blocking -- what is the recall CEILING my candidate set imposes, and how many
                 candidates per entity am I paying for it? (candidate-set size is
                 explicitly part of the final ranking)
  3. caps     -- am I emitting structurally impossible predictions? The generator
                 never produces more than 5 Source-2 or 6 Source-3 matches.
  4. holdout  -- carve a deterministic, entity-level validation split from train.

Stdlib + numpy only. Every number printed here is computed with the official
metric, reproduced exactly:

    G = |true|, P = |predicted|, TP = |true & predicted|
    G == 0 :  F = 1.0 if P == 0 else 0.0
    G  > 0 :  F = 0.0 if P == 0 else 1.25 * TP / (0.25 * G + P)

macro-averaged over ALL Source-1 entities (singletons included).

Usage
-----
    # exact score + per-country + FP/FN decomposition
    python3 diagnose.py score --pred output/matching_results.tsv \
        --gt dataset/train/train_ground_truth.tsv \
        --source1 dataset/train/train_source1.tsv

    # blocking recall ceiling and cost
    python3 diagnose.py blocking --cand output/candidate_pairs.tsv \
        --gt dataset/train/train_ground_truth.tsv \
        --source1 dataset/train/train_source1.tsv

    # structural sanity on a test submission (no GT needed)
    python3 diagnose.py caps --pred output/matching_results.tsv \
        --gt dataset/train/train_ground_truth.tsv

    # deterministic 10% entity holdout
    python3 diagnose.py holdout --gt dataset/train/train_ground_truth.tsv \
        --frac 0.1 --out work/holdout_ids.txt

Add --sample-mod N to any GT-based command to score a deterministic 1/N subsample
(cuts memory ~N x; the macro average is unbiased).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:  # keep the script usable without numpy
    np = None

BETA2 = 0.25
# Structural caps observed across all 2,206,821 training entities.
CAP_S2 = 5
CAP_S3 = 6


# --------------------------------------------------------------------------- io
def _bucket(key: str, mod: int) -> bool:
    """Deterministic subsample: keep the entity iff its hash falls in bucket 0."""
    if mod <= 1:
        return True
    h = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % mod == 0


def read_idmap(path: str, sample_mod: int = 1, restrict: set | None = None) -> dict:
    """Read a two-column '<s1_id>\\t<comma,separated,ids>' TSV into {s1: set(ids)}."""
    out: dict[str, set] = {}
    with open(path, encoding="utf-8") as fh:
        head = fh.readline()
        if head and not head.startswith("source1_entity_id"):
            fh.seek(0)  # no header -> rewind
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            tab = line.find("\t")
            if tab < 0:
                key, rest = line, ""
            else:
                key, rest = line[:tab], line[tab + 1 :]
            if restrict is not None and key not in restrict:
                continue
            if not _bucket(key, sample_mod):
                continue
            out[key] = {x for x in rest.split(",") if x} if rest else set()
    return out


def read_country(path: str, restrict: set | None = None) -> dict:
    """entity_id -> country from a source TSV (interned: only a few distinct values)."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            if restrict is not None and p[0] not in restrict:
                continue
            out[p[0]] = sys.intern(p[3])
    return out


# ----------------------------------------------------------------------- metric
def f05(G: int, P: int, TP: int) -> float:
    if G == 0:
        return 1.0 if P == 0 else 0.0
    if P == 0:
        return 0.0
    return (1.0 + BETA2) * TP / (BETA2 * G + P)


def _fmt(x: float) -> str:
    return f"{x:.6f}"


# ------------------------------------------------------------------------ score
def cmd_score(a) -> None:
    gt = read_idmap(a.gt, a.sample_mod)
    pred = read_idmap(a.pred, a.sample_mod, restrict=set(gt))
    country = read_country(a.source1, restrict=set(gt)) if a.source1 else {}

    missing = len(gt) - len(pred)
    agg: dict[str, dict] = defaultdict(lambda: dict(
        n=0, f=0.0, perfect=0, tp=0, fp=0, fn=0, g=0, p=0,
        sing_n=0, sing_ok=0, nonsing_n=0, empty_pred_on_nonsing=0,
        f_noFP=0.0, f_noFN=0.0))

    for s1, G_ids in gt.items():
        P_ids = pred.get(s1, set())
        G, P = len(G_ids), len(P_ids)
        TP = len(G_ids & P_ids)
        FP, FN = P - TP, G - TP
        f = f05(G, P, TP)
        # counterfactuals: how good would I be if one error type vanished?
        f_noFP = f05(G, TP, TP)            # drop every false positive
        f_noFN = f05(G, P + FN, G)         # add every missed true match

        keys = ["ALL"]
        c = country.get(s1)
        if c:
            keys.append(c)
        for k in keys:
            d = agg[k]
            d["n"] += 1
            d["f"] += f
            d["perfect"] += (f >= 1.0)
            d["tp"] += TP; d["fp"] += FP; d["fn"] += FN
            d["g"] += G; d["p"] += P
            d["f_noFP"] += f_noFP; d["f_noFN"] += f_noFN
            if G == 0:
                d["sing_n"] += 1
                d["sing_ok"] += (P == 0)
            else:
                d["nonsing_n"] += 1
                d["empty_pred_on_nonsing"] += (P == 0)

    if missing:
        print(f"!! {missing} ground-truth entities absent from --pred "
              f"(scored as empty predictions)\n")

    order = ["ALL"] + sorted(k for k in agg if k != "ALL")
    for k in order:
        d = agg[k]
        n = d["n"]
        if not n:
            continue
        score = d["f"] / n
        print(f"=== {k}  (n={n:,}) ===")
        print(f"  macro F0.5              : {_fmt(score)}")
        print(f"  entities scoring 1.000  : {d['perfect']/n:.3%}")
        print(f"  mean |true| / |pred|    : {d['g']/n:.3f} / {d['p']/n:.3f}")
        print(f"  micro TP/FP/FN          : {d['tp']:,} / {d['fp']:,} / {d['fn']:,}")
        if d["p"]:
            print(f"  micro precision / recall: {d['tp']/d['p']:.4f} / "
                  f"{d['tp']/max(1,d['g']):.4f}")
        print(f"  headroom if 0 FP        : {_fmt(d['f_noFP']/n)}  "
              f"(+{d['f_noFP']/n - score:.6f})")
        print(f"  headroom if 0 FN        : {_fmt(d['f_noFN']/n)}  "
              f"(+{d['f_noFN']/n - score:.6f})")
        if d["sing_n"]:
            print(f"  singletons              : {d['sing_n']:,}  "
                  f"correct(empty) {d['sing_ok']/d['sing_n']:.3%}  "
                  f"-> costs {(d['sing_n']-d['sing_ok'])/n:.6f}")
        if d["nonsing_n"]:
            print(f"  empty pred on non-singleton: {d['empty_pred_on_nonsing']:,} "
                  f"({d['empty_pred_on_nonsing']/d['nonsing_n']:.3%}) -> costs "
                  f"{d['empty_pred_on_nonsing']/n:.6f}")
        print()

    print("Read: 'headroom if 0 FP' vs 'headroom if 0 FN' tells you which side to "
          "attack. If the FP headroom is larger, tighten the decision rule; if FN, "
          "widen blocking / lower the threshold.")


# --------------------------------------------------------------------- blocking
def cmd_blocking(a) -> None:
    gt = read_idmap(a.gt, a.sample_mod)
    cand = read_idmap(a.cand, a.sample_mod, restrict=set(gt))
    country = read_country(a.source1, restrict=set(gt)) if a.source1 else {}

    agg: dict[str, dict] = defaultdict(lambda: dict(
        n=0, true=0, found=0, ncand=0, ceil=0.0, empty=0,
        true_s2=0, found_s2=0, true_s3=0, found_s3=0, lost_entities=0))

    for s1, G_ids in gt.items():
        C_ids = cand.get(s1, set())
        hit = G_ids & C_ids
        keys = ["ALL"]
        c = country.get(s1)
        if c:
            keys.append(c)
        # a perfect matcher restricted to this candidate set scores exactly this:
        ceil = f05(len(G_ids), len(hit), len(hit))
        s2t = sum(1 for x in G_ids if x.startswith("S2-"))
        s3t = len(G_ids) - s2t
        s2f = sum(1 for x in hit if x.startswith("S2-"))
        s3f = len(hit) - s2f
        for k in keys:
            d = agg[k]
            d["n"] += 1
            d["true"] += len(G_ids); d["found"] += len(hit)
            d["ncand"] += len(C_ids); d["ceil"] += ceil
            d["empty"] += (not C_ids)
            d["true_s2"] += s2t; d["found_s2"] += s2f
            d["true_s3"] += s3t; d["found_s3"] += s3f
            d["lost_entities"] += (len(hit) < len(G_ids))

    order = ["ALL"] + sorted(k for k in agg if k != "ALL")
    for k in order:
        d = agg[k]
        n = d["n"]
        if not n:
            continue
        print(f"=== {k}  (n={n:,}) ===")
        rec = d["found"] / max(1, d["true"])
        print(f"  blocking recall (pair)  : {rec:.5%}")
        if d["true_s2"]:
            print(f"    S2 recall             : {d['found_s2']/d['true_s2']:.5%}")
        if d["true_s3"]:
            print(f"    S3 recall             : {d['found_s3']/d['true_s3']:.5%}")
        print(f"  MACRO F0.5 CEILING      : {_fmt(d['ceil']/n)}   <-- perfect matcher on this candidate set")
        print(f"  candidates per entity   : {d['ncand']/n:.2f}")
        print(f"  entities missing >=1 tp : {d['lost_entities']/n:.3%}")
        print(f"  entities with 0 cands   : {d['empty']:,}")
        print()
    print("The CEILING is the hard upper bound on your leaderboard score. If it is "
          "below your target, no amount of model work will get you there -- fix "
          "blocking first. Then minimise 'candidates per entity' at fixed ceiling.")


# ------------------------------------------------------------------------- caps
def cmd_caps(a) -> None:
    pred = read_idmap(a.pred, 1)
    n = len(pred)
    cnt, s2c, s3c = Counter(), Counter(), Counter()
    viol = []
    for s1, ids in pred.items():
        s2 = sum(1 for x in ids if x.startswith("S2-"))
        s3 = sum(1 for x in ids if x.startswith("S3-"))
        cnt[len(ids)] += 1; s2c[s2] += 1; s3c[s3] += 1
        if s2 > CAP_S2 or s3 > CAP_S3:
            viol.append((s1, s2, s3))
    print(f"predicted rows: {n:,}")
    print(f"empty (singleton) predictions: {cnt[0]:,} ({cnt[0]/n:.4%})")
    print(f"mean ids per entity: {sum(k*v for k,v in cnt.items())/n:.3f}")
    print(f"\nstructural cap violations (S2>{CAP_S2} or S3>{CAP_S3}): {len(viol):,} "
          f"({len(viol)/n:.4%})")
    for s1, s2, s3 in viol[:10]:
        print(f"   {s1}  n_S2={s2} n_S3={s3}")
    if viol:
        print("   -> these are guaranteed false positives; truncating them is free precision.")

    if a.gt:
        gt = read_idmap(a.gt, a.sample_mod)
        g = len(gt)
        gc, g2, g3 = Counter(), Counter(), Counter()
        for ids in gt.values():
            s2 = sum(1 for x in ids if x.startswith("S2-"))
            gc[len(ids)] += 1; g2[s2] += 1; g3[len(ids) - s2] += 1
        print("\n  n | predicted |    truth   (distribution of match-count)")
        for kk in range(0, max(max(cnt), max(gc)) + 1):
            print(f" {kk:2d} | {cnt[kk]/n:9.4%} | {gc[kk]/g:9.4%}")
        print(f"\n  mean truth ids: {sum(k*v for k,v in gc.items())/g:.3f}")
        print("  A predicted mean below truth means you are leaving recall on the "
              "table; above means you are over-merging.")


# ---------------------------------------------------------------------- holdout
def cmd_holdout(a) -> None:
    mod = max(2, int(round(1.0 / a.frac)))
    ids = []
    with open(a.gt, encoding="utf-8") as fh:
        head = fh.readline()
        if head and not head.startswith("source1_entity_id"):
            fh.seek(0)
        for line in fh:
            tab = line.find("\t")
            key = line[:tab] if tab > 0 else line.rstrip("\n")
            if key and _bucket(key, mod):
                ids.append(key)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(ids) + "\n")
    print(f"wrote {len(ids):,} holdout entity ids (1/{mod}) -> {a.out}")
    print("This split is entity-level and hash-deterministic: the same ids come out "
          "on every machine, so train/validation never leak across runs.")


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="exact macro F0.5 + per-country + FP/FN headroom")
    s.add_argument("--pred", required=True)
    s.add_argument("--gt", required=True)
    s.add_argument("--source1", help="source1 TSV, enables the per-country breakdown")
    s.add_argument("--sample-mod", type=int, default=1)
    s.set_defaults(fn=cmd_score)

    b = sub.add_parser("blocking", help="recall ceiling + candidates per entity")
    b.add_argument("--cand", required=True)
    b.add_argument("--gt", required=True)
    b.add_argument("--source1")
    b.add_argument("--sample-mod", type=int, default=1)
    b.set_defaults(fn=cmd_blocking)

    c = sub.add_parser("caps", help="structural cap violations + count distribution")
    c.add_argument("--pred", required=True)
    c.add_argument("--gt", help="optional: compare against the true distribution")
    c.add_argument("--sample-mod", type=int, default=1)
    c.set_defaults(fn=cmd_caps)

    h = sub.add_parser("holdout", help="deterministic entity-level validation split")
    h.add_argument("--gt", required=True)
    h.add_argument("--frac", type=float, default=0.1)
    h.add_argument("--out", required=True)
    h.set_defaults(fn=cmd_holdout)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
