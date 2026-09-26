#!/usr/bin/env python3
"""How much score is actually available from the DECISION layer alone?

Holds the trained model fixed and asks what a better selection rule could buy. Four
bounds, in increasing power:

  tuned_global   the shipped rule (one global tau/gamma/miss)
  best_global    best single global threshold, swept here
  per_country    best threshold chosen PER COUNTRY
  per_segment    best threshold per (country x record-type) segment
  oracle_topk    per entity, the best prefix of the p-sorted candidate list
                 -> upper bound on ANY rule that ranks by p and cuts at some k
  oracle_subset  per entity, exactly the true candidates
                 -> upper bound on any rule at all, given this candidate set

The gap tuned_global -> oracle_topk is the honest headroom for threshold work.
The gap oracle_topk -> oracle_subset can only be closed by a better SCORER.

Run from README/ after `run.py train --keep-intermediates`:
    .venv/bin/python src/decision_analysis.py --work-dir work_dev
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blocking import load_pairs                          # noqa: E402
from config import Config                                # noqa: E402
from decide import BETA2, apply_decision, exclusive_mask, macro_f05  # noqa: E402
from pipeline import load_models, model_dir, predict_rows  # noqa: E402
from prepare import load_gt, load_split, split_dir       # noqa: E402
from utils import crc_fold, safe_name                    # noqa: E402


def is_nonlatin(s: str) -> bool:
    n = unicodedata.normalize("NFKD", s or "")
    n = "".join(c for c in n if not unicodedata.combining(c))
    return not re.search(r"[a-z]", n.lower())


def f05_vec(G, P, TP):
    """Exact metric, vectorised over entities."""
    return np.where(G == 0, (P == 0).astype(np.float64),
                    np.where(P == 0, 0.0,
                             (1 + BETA2) * TP / (BETA2 * G + np.maximum(P, 1e-9))))


def sweep_threshold(q, p, y, G, ent_mask, taus):
    """Best single threshold over the given entity subset."""
    best = (None, -1.0)
    nq = len(G)
    for t in taus:
        s = p >= t
        P = np.bincount(q[s], minlength=nq).astype(np.float64)
        TP = np.bincount(q[s], weights=y[s].astype(np.float64), minlength=nq)
        F = f05_vec(G, P, TP)[ent_mask]
        m = float(F.mean()) if len(F) else 0.0
        if m > best[1]:
            best = (float(t), m)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work_dev")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.work_dir, "models", "meta.json")))
    cfg = Config.from_dict(meta["config"])
    cfg.work_dir = a.work_dir
    report = json.load(open(os.path.join(a.work_dir, "models", "report.json")))

    s1, idx, parts = load_split(cfg, "train", columns=[
        "entity_id", "business_name", "business_address", "country"])
    n_s1, n_idx = len(s1), len(idx)
    fold_s1 = crc_fold(s1["entity_id"].tolist())
    gt_s1, gt_idx = load_gt(cfg)
    G = np.bincount(gt_s1, minlength=n_s1).astype(np.float64)
    owner = np.full(n_idx, -1, dtype=np.int64)
    owner[gt_idx] = gt_s1

    m2 = load_models(cfg, "stage2")
    with open(os.path.join(model_dir(cfg), "calibrator.pkl"), "rb") as fh:
        calib = pickle.load(fh)

    hold = list(cfg.holdout_folds)
    Q, I, P = [], [], []
    for ck, (s1_rows, idx_rows) in parts.items():
        Pk = load_pairs(cfg, "train", ck)
        qg = s1_rows[Pk["q"]].astype(np.int64)
        ig = idx_rows[Pk["i"]].astype(np.int64)
        fold = fold_s1[qg]
        sel = np.flatnonzero(np.isin(fold, hold))
        if not len(sel):
            continue
        X2 = np.load(os.path.join(split_dir(cfg, "train"), f"s2_{safe_name(ck)}.npy"),
                     mmap_mode="r")
        pr = predict_rows(m2, np.asarray(X2[sel], dtype=np.float32), fold[sel])
        del X2
        Q.append(qg[sel]); I.append(ig[sel]); P.append(pr)
    q = np.concatenate(Q); i_glob = np.concatenate(I)
    p = calib.predict(np.concatenate(P)).astype(np.float32)
    y = (owner[i_glob] == q).astype(np.int8)

    ent_mask = np.zeros(n_s1, bool)
    ent_mask[np.unique(q)] = True
    ent_mask &= np.isin(fold_s1, hold)
    n_ent = int(ent_mask.sum())

    # one-owner constraint, as shipped
    keep = exclusive_mask(i_glob, p)
    qк, pк, yк = q[keep], p[keep], y[keep]

    print(f"holdout entities {n_ent:,}   candidate rows {len(q):,}   "
          f"after one-owner {keep.sum():,}\n")

    # ---- 1. shipped rule
    sel = apply_decision(q, i_glob, p, n_s1, report["decision"])
    tuned = macro_f05(q[sel], y[sel], G, ent_mask)

    # ---- 2. best single global threshold
    taus = np.round(np.arange(0.02, 0.99, 0.01), 3)
    t_best, f_best = sweep_threshold(qк, pк, yк, G, ent_mask, taus)

    # ---- 3. oracle top-k (best prefix per entity, ranked by p)
    order = np.lexsort((-pк, qк))
    qs, ps, ys = qк[order], pк[order], yк[order]
    bounds = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1, len(qs)]
    oracle_k = np.zeros(n_s1)
    best_k = np.zeros(n_s1, dtype=np.int32)
    seen = np.zeros(n_s1, bool)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        e = qs[b0]
        seen[e] = True
        g = G[e]
        tp = np.cumsum(ys[b0:b1], dtype=np.float64)
        k = np.arange(1, b1 - b0 + 1, dtype=np.float64)
        f = np.zeros(b1 - b0 + 1)
        f[0] = 1.0 if g == 0 else 0.0
        if g > 0:
            f[1:] = (1 + BETA2) * tp / (BETA2 * g + k)
        j = int(np.argmax(f))
        oracle_k[e] = f[j]
        best_k[e] = j
    # entities with zero candidates score 1.0 iff they are true singletons
    no_cand = ent_mask & ~seen
    oracle_k[no_cand] = (G[no_cand] == 0).astype(float)
    f_oracle_k = float(oracle_k[ent_mask].mean())

    # ---- 4. oracle subset == report's oracle_f05
    f_oracle_sub = report["holdout_oracle_f05"]

    print("=== DECISION-LAYER BOUNDS (model held fixed) ===")
    print(f"  tuned_global   (shipped)          {tuned:.6f}")
    print(f"  best_global    (tau={t_best:.2f})          {f_best:.6f}   "
          f"{f_best - tuned:+.6f}")
    print(f"  oracle_topk    (best prefix/entity) {f_oracle_k:.6f}   "
          f"{f_oracle_k - tuned:+.6f}  <-- ceiling for ANY p-ranked cut")
    print(f"  oracle_subset  (perfect selection)  {f_oracle_sub:.6f}   "
          f"{f_oracle_sub - tuned:+.6f}  <-- needs a better SCORER")

    # ---- 5. per-country and per-segment thresholds
    ctry = s1["country"].to_numpy(dtype=object)
    iname = idx["business_name"].to_numpy(dtype=object)
    iaddr = idx["business_address"].to_numpy(dtype=object)

    print("\n=== PER-COUNTRY optimal threshold ===")
    tot = 0.0
    for c in sorted({ctry[e] for e in np.flatnonzero(ent_mask)}):
        em = ent_mask & np.array([ctry[i] == c if ent_mask[i] else False
                                  for i in range(n_s1)])
        rows = np.isin(qк, np.flatnonzero(em))
        t, f = sweep_threshold(qк[rows], pк[rows], yк[rows], G, em, taus)
        base = macro_f05(q[sel & np.isin(q, np.flatnonzero(em))],
                         y[sel & np.isin(q, np.flatnonzero(em))], G, em)
        n_c = int(em.sum())
        tot += f * n_c
        print(f"  {c:8s} n={n_c:6,d}  shipped={base:.6f}  best_tau={t:.2f} -> {f:.6f}"
              f"  ({f - base:+.6f})")
    print(f"  weighted per-country best        {tot/n_ent:.6f}   "
          f"{tot/n_ent - tuned:+.6f}  <-- gain from per-country tau alone")

    print("\n=== SEGMENTS (candidate-side traits, share of one-owner rows) ===")
    nl = np.fromiter((is_nonlatin(iname[i]) for i in i_glob[keep]), bool, keep.sum())
    ea = np.fromiter((not (iaddr[i] or "").strip() for i in i_glob[keep]), bool, keep.sum())
    for tag, m in (("non_latin_name", nl), ("empty_address", ea),
                   ("plain", ~nl & ~ea)):
        if m.sum() < 50:
            continue
        pos = yк[m].mean()
        print(f"  {tag:16s} rows={m.sum():7,d}  positive_rate={pos:.4f}  "
              f"mean_p={pк[m].mean():.4f}  mean_p|y=1={pк[m & (yк == 1)].mean():.4f}")
    print("\nIf positive_rate and mean_p diverge for a segment, the model is "
          "mis-calibrated there and a segment-specific cut will pay.")

    print("\n=== PREDICTED COUNT vs TRUTH ===")
    Pc = np.bincount(q[sel], minlength=n_s1)[ent_mask]
    print(f"  mean predicted {Pc.mean():.3f}   mean true {G[ent_mask].mean():.3f}   "
          f"mean oracle-k {best_k[ent_mask].mean():.3f}")


if __name__ == "__main__":
    main()
