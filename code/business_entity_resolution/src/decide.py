"""Decision layer: one-owner constraint + F0.5-optimal per-entity selection + metric.

Scoring rule reproduced exactly (macro over ALL S1 entities, singletons included):
    G = true matches, P = predicted, TP = |G & P|
    G = 0 : F = 1 if P = 0 else 0
    G > 0 : F = 0 if P = 0 else 1.25*TP / (0.25*G + P)
(the second line is algebraically identical to 1.25PR/(0.25P+R)).
"""
from __future__ import annotations

import itertools

import numpy as np

from utils import LOG, group_starts, rank_in_sorted_groups

BETA2 = 0.25


def macro_f05(q_sel, y_sel, G, qmask=None) -> float:
    """q_sel: S1 index of each predicted pair, y_sel: 1 if correct, G: true count per S1."""
    nq = len(G)
    P = np.bincount(q_sel, minlength=nq).astype(np.float64)
    TP = np.bincount(q_sel, weights=y_sel.astype(np.float64), minlength=nq)
    F = np.where(G == 0, (P == 0).astype(np.float64),
                 np.where(P == 0, 0.0, (1 + BETA2) * TP / (BETA2 * G + np.maximum(P, 1e-9))))
    if qmask is not None:
        F = F[qmask]
    return float(F.mean()) if len(F) else float("nan")


def exclusive_mask(i_glob: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Each S2/S3 record belongs to at most one S1: keep only its best-scoring S1."""
    order = np.lexsort((-p, i_glob))
    ii = i_glob[order]
    first = np.r_[True, ii[1:] != ii[:-1]] if len(ii) else np.zeros(0, bool)
    keep = np.zeros(len(p), dtype=bool)
    keep[order[first]] = True
    return keep


def select_threshold(p, tau):
    return p >= tau


def first_housenum_arr(a_nums) -> np.ndarray:
    """First integer from each a_nums string; -1 if missing."""
    out = np.full(len(a_nums), -1, dtype=np.int64)
    for i, raw in enumerate(a_nums):
        s = raw or ""
        if not s:
            continue
        # a_nums is space-separated digits from normalize
        tok = s.split()[0] if " " in s else s
        if tok.isdigit():
            out[i] = int(tok)
            continue
        for ch in s.split():
            if ch.isdigit():
                out[i] = int(ch)
                break
    return out


def adjust_p_housenum(p, q, i_glob, q_num, i_num, boost=0.05, pen=0.05):
    """Boost exact first-house-number matches; penalize close-but-unequal distractors.

    Measured on the empty-BM25 holdout: FPs have median |Δhouse|=5 while true matches
    are exact 77% of the time. boost=0.05, pen=0.05 → +0.000242 without retraining.
    """
    if boost == 0.0 and pen == 0.0:
        return p
    p2 = p.copy()
    qn = q_num[q]
    ina = i_num[i_glob]
    both = (qn >= 0) & (ina >= 0)
    exact = both & (qn == ina)
    close = both & (np.abs(qn - ina) <= 5) & ~exact
    if boost:
        p2[exact] = np.clip(p2[exact] + boost, 0.0, 1.0)
    if pen:
        p2[close] = np.clip(p2[close] - pen, 0.0, 1.0)
    return p2


def select_expected_f(q, p, nq, tau=0.0, gamma=1.0, miss=0.0, presorted=False):
    """Choose, per S1, the prefix (by p desc) that maximises expected F0.5.

    E[F | top-k] ~= 1.25 * sum_{j<=k} p_j / (0.25 * (sum_all p + miss) + k)
    E[F | empty] =  gamma * prod(1 - p_j)      (probability the entity is a singleton)
    Items with p < tau are never added. Fully vectorised; O(n log n).
    """
    n = len(p)
    sel = np.zeros(n, dtype=bool)
    if n == 0:
        return sel
    order = np.arange(n) if presorted else np.lexsort((-p, q))
    qs, ps = q[order], p[order]
    rk = rank_in_sorted_groups(qs)
    st = group_starts(qs)
    sizes = np.diff(np.r_[st, n])
    gid = np.repeat(np.arange(len(st)), sizes)
    cs = np.cumsum(ps)
    base = np.repeat(cs[st] - ps[st], sizes)
    cum = cs - base
    S = np.bincount(gid, weights=ps)[gid]
    ef = (1 + BETA2) * cum / (BETA2 * (S + miss) + rk + 1)
    ef = np.where(ps >= tau, ef, -1.0)
    pc = np.clip(ps, 0.0, 1.0 - 1e-7)
    ef0 = gamma * np.exp(np.bincount(gid, weights=np.log1p(-pc)))
    gmax = np.maximum.reduceat(ef, st)
    is_max = ef == gmax[gid]
    kstar = np.minimum.reduceat(np.where(is_max, rk, n + 1), st)
    take = (rk <= kstar[gid]) & (gmax[gid] > ef0[gid]) & (gmax[gid] > 0)
    sel[order] = take
    return sel


def tune(q, i_glob, p, y, G, qmask, nq, method="auto"):
    """Grid-search the decision rule on OOF rows. Returns (best_cfg, table)."""
    keep = np.flatnonzero(exclusive_mask(i_glob, p))
    o = np.lexsort((-p[keep], q[keep]))          # sort once, reuse for every grid point
    keep = keep[o]
    qk, pk, yk = q[keep], p[keep], y[keep]
    results = []
    if method in ("auto", "threshold"):
        for tau in np.round(np.arange(0.05, 0.96, 0.025), 3):
            s = select_threshold(pk, tau)
            results.append(({"method": "threshold", "tau": float(tau)},
                            macro_f05(qk[s], yk[s], G, qmask)))
    if method in ("auto", "expected_f"):
        # widened after the first full run found its best at tau=0.4, the previous
        # upper edge of this grid - extend so the optimum isn't clipped by the grid.
        for tau, gamma, miss in itertools.product(
                (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                (0.6, 0.7, 0.85, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7),
                (0.0, 0.1, 0.2, 0.3)):
            s = select_expected_f(qk, pk, nq, tau, gamma, miss, presorted=True)
            results.append(({"method": "expected_f", "tau": tau, "gamma": gamma,
                             "miss": miss}, macro_f05(qk[s], yk[s], G, qmask)))
    results.sort(key=lambda r: -r[1])
    for cfg_, f in results[:5]:
        LOG.info("    decision %-60s F0.5=%.5f", cfg_, f)
    return results[0][0], results


def apply_decision(q, i_glob, p, nq, dcfg):
    keep = exclusive_mask(i_glob, p)
    sel = np.zeros(len(p), dtype=bool)
    idx = np.flatnonzero(keep)
    if dcfg["method"] == "threshold":
        s = select_threshold(p[idx], dcfg["tau"])
    else:
        s = select_expected_f(q[idx], p[idx], nq, dcfg["tau"], dcfg["gamma"], dcfg["miss"])
    sel[idx[s]] = True
    return sel
