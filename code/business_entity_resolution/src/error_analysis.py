#!/usr/bin/env python3
"""CHECK 2 -- label the trained model's actual holdout errors against ground truth.

Reconstructs the exact holdout decision from saved artifacts (stage-2 features,
stage-2 models, isotonic calibrator, tuned decision rule), then classifies every
error into a taxonomy so we can see where the score is really leaking:

  blocking_miss        true match never reached the candidate set  -> fix BLOCKING
  missed_in_candidates true match WAS a candidate but not selected -> fix MODEL/DECISION
  fp_stolen            predicted a record owned by a different S1  -> fix ASSIGNMENT
  fp_distractor        predicted a record owned by nobody          -> fix PRECISION
  fp_on_singleton      predicted anything for a true singleton     -> costs a full 1.0

Run from README/ after `run.py train`:
    .venv/bin/python src/error_analysis.py --work-dir work_dev --show 40
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import load_pairs                      # noqa: E402
from config import Config                            # noqa: E402
from decide import apply_decision, macro_f05         # noqa: E402
from pipeline import load_models, model_dir, predict_rows  # noqa: E402
from prepare import load_gt, load_split, split_dir   # noqa: E402
from utils import crc_fold, safe_name                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work_dev")
    ap.add_argument("--show", type=int, default=40, help="example errors to print")
    ap.add_argument("--out", default=None, help="write full error rows to TSV")
    a = ap.parse_args()

    meta_path = os.path.join(a.work_dir, "models", "meta.json")
    meta_json = json.load(open(meta_path))
    cfg = Config.from_dict(meta_json["config"])
    cfg.work_dir = a.work_dir
    report = json.load(open(os.path.join(a.work_dir, "models", "report.json")))
    dcfg = report["decision"]

    s1, idx, parts = load_split(cfg, "train", columns=[
        "entity_id", "business_name", "business_address", "country", "ckey"])
    n_s1, n_idx = len(s1), len(idx)
    fold_s1 = crc_fold(s1["entity_id"].tolist())
    gt_s1, gt_idx = load_gt(cfg)
    G = np.bincount(gt_s1, minlength=n_s1).astype(np.float64)

    # owner[idx_row] = s1_row that truly owns this S2/S3 record (one-owner is proven)
    owner = np.full(n_idx, -1, dtype=np.int64)
    owner[gt_idx] = gt_s1
    gt_by_s1 = defaultdict(set)
    for q_, i_ in zip(gt_s1.tolist(), gt_idx.tolist()):
        gt_by_s1[q_].add(i_)

    m2 = load_models(cfg, "stage2")
    with open(os.path.join(model_dir(cfg), "calibrator.pkl"), "rb") as fh:
        calib = pickle.load(fh)

    hold = set(cfg.holdout_folds)
    Q, I, P, Y = [], [], [], []
    for ck, (s1_rows, idx_rows) in parts.items():
        Pk = load_pairs(cfg, "train", ck)
        qg = s1_rows[Pk["q"]].astype(np.int64)
        ig = idx_rows[Pk["i"]].astype(np.int64)
        fold = fold_s1[qg]
        sel = np.flatnonzero(np.isin(fold, list(hold)))
        if not len(sel):
            continue
        X2 = np.load(os.path.join(split_dir(cfg, "train"), f"s2_{safe_name(ck)}.npy"),
                     mmap_mode="r")
        p = predict_rows(m2, np.asarray(X2[sel], dtype=np.float32), fold[sel])
        del X2
        Q.append(qg[sel]); I.append(ig[sel]); P.append(p)
        Y.append(np.isin(ig[sel], list(gt_by_s1_keys(gt_by_s1, qg[sel]))) if False else None)
    q = np.concatenate(Q); i_glob = np.concatenate(I); p = np.concatenate(P)
    p = calib.predict(p).astype(np.float32)

    # true label per candidate row: does this S1 truly own this record?
    y = (owner[i_glob] == q).astype(np.int8)

    sel = apply_decision(q, i_glob, p, n_s1, dcfg)
    qh = np.isin(np.arange(n_s1), np.unique(q)) & np.isin(fold_s1, list(hold))
    score = macro_f05(q[sel], y[sel], G, qh)
    n_ent = int(qh.sum())
    print(f"holdout entities      : {n_ent:,}")
    print(f"reconstructed F0.5    : {score:.6f}  "
          f"(report says {report['holdout_stage2_f05']:.6f})")

    pred_by_s1 = defaultdict(set)
    for qq, ii in zip(q[sel].tolist(), i_glob[sel].tolist()):
        pred_by_s1[qq].add(ii)
    cand_by_s1 = defaultdict(set)
    for qq, ii in zip(q.tolist(), i_glob.tolist()):
        cand_by_s1[qq].add(ii)

    name = s1["business_name"].to_numpy(dtype=object)
    addr = s1["business_address"].to_numpy(dtype=object)
    ctry = s1["country"].to_numpy(dtype=object)
    eid = s1["entity_id"].to_numpy(dtype=object)
    iname = idx["business_name"].to_numpy(dtype=object)
    iaddr = idx["business_address"].to_numpy(dtype=object)
    ieid = idx["entity_id"].to_numpy(dtype=object)

    cat = Counter()
    per_country = defaultdict(Counter)
    loss_by_cat = Counter()
    examples = defaultdict(list)
    ent_with_err = 0

    holdout_rows = np.flatnonzero(qh)
    for qq in holdout_rows.tolist():
        truth = gt_by_s1.get(qq, set())
        pred = pred_by_s1.get(qq, set())
        cand = cand_by_s1.get(qq, set())
        if truth == pred:
            continue
        ent_with_err += 1
        c = ctry[qq]
        for miss in truth - pred:
            k = "blocking_miss" if miss not in cand else "missed_in_candidates"
            cat[k] += 1; per_country[c][k] += 1
            if len(examples[k]) < a.show:
                examples[k].append((qq, miss))
        for fp in pred - truth:
            if not truth:
                k = "fp_on_singleton"
            elif owner[fp] >= 0:
                k = "fp_stolen"
            else:
                k = "fp_distractor"
            cat[k] += 1; per_country[c][k] += 1
            if len(examples[k]) < a.show:
                examples[k].append((qq, fp))

    tot = sum(cat.values())
    print(f"entities with >=1 error: {ent_with_err:,} ({ent_with_err/n_ent:.3%})")
    print(f"total error items      : {tot:,}\n")
    print(f"{'category':22s} {'count':>9s} {'share':>8s}")
    print("-" * 42)
    for k, v in cat.most_common():
        print(f"{k:22s} {v:9,d} {v/tot:7.2%}")

    print("\nby country:")
    for c, cc in per_country.items():
        t = sum(cc.values())
        print(f"  {c:8s} total={t:7,d}  " +
              "  ".join(f"{k}={v/t:.1%}" for k, v in cc.most_common()))

    print("\nFN split (recall losses):")
    fn = cat["blocking_miss"] + cat["missed_in_candidates"]
    if fn:
        print(f"  blocking_miss        {cat['blocking_miss']/fn:7.2%}  <- unreachable, blocking's fault")
        print(f"  missed_in_candidates {cat['missed_in_candidates']/fn:7.2%}  <- reachable, model/decision's fault")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("category\ts1_entity\ts1_name\ts1_addr\tcountry\tother_id\tother_name\tother_addr\n")
            for k, rows in examples.items():
                for qq, ii in rows:
                    fh.write(f"{k}\t{eid[qq]}\t{name[qq]}\t{addr[qq]}\t{ctry[qq]}\t"
                             f"{ieid[ii]}\t{iname[ii]}\t{iaddr[ii]}\n")
        print(f"\nwrote examples -> {a.out}")

    print("\n" + "=" * 96)
    print("MANUAL INSPECTION")
    print("=" * 96)
    for k in ("blocking_miss", "missed_in_candidates", "fp_stolen", "fp_distractor",
              "fp_on_singleton"):
        rows = examples.get(k, [])[:8]
        if not rows:
            continue
        print(f"\n######## {k}  ({cat[k]:,} total) ########")
        for qq, ii in rows:
            print(f"  S1 {eid[qq]} [{ctry[qq]}]")
            print(f"     name: {name[qq]!r}")
            print(f"     addr: {addr[qq]!r}")
            print(f"  -> {ieid[ii]}")
            print(f"     name: {iname[ii]!r}")
            print(f"     addr: {iaddr[ii]!r}")
            if k == "fp_stolen":
                o = owner[ii]
                print(f"     TRUE OWNER {eid[o]}: {name[o]!r} | {addr[o]!r}")
            print()


def gt_by_s1_keys(d, qs):
    return set()


if __name__ == "__main__":
    main()
