"""Honest holdout evaluation of the cross-encoder reranker (src/rerank.py).

Recomputes p1/p2 for ONLY the untouched holdout folds (8, 9) from the already-cached
blocking (pairs_*.npz) and the saved stage-1/stage-2 models - the full feat_*.npy /
s2_*.npy caches from the main training run are deleted by design (see config.py:
keep_intermediates), so this rebuilds just the small holdout slice rather than
requiring a full retrain. Then it re-scores the pipeline's final SHORTLIST (the
pairs actually selected for submission on holdout) with the reranker and reports
macro F0.5 with vs. without it, on the identical holdout rows - directly comparable
to the report.json numbers from the main run.
"""
from __future__ import annotations

import os

import numpy as np

from blocking import load_pairs
from config import Config
from decide import apply_decision, exclusive_mask, macro_f05, tune
from features import ALL_FEATURES, PartitionFeaturizer, BASE_FEATURES
from pipeline import load_models, model_dir, predict_rows
from prepare import load_gt, load_split, split_dir
from stage2 import S2_FEATURES, stage2_matrix
from utils import LOG, crc_fold, load_json, timed

PRED_CHUNK = 2_000_000


def _features_for_subset(cfg, Q, I, qc, ic) -> np.ndarray:
    """Run the normal feature pipeline on an arbitrary (small) pair subset."""
    fz = PartitionFeaturizer(Q, I, cfg)
    n = len(qc)
    out = np.full((n, len(ALL_FEATURES)), np.nan, dtype=np.float32)
    step = 2_000_000
    j_nt, j_at = BASE_FEATURES.index("n_tset"), BASE_FEATURES.index("a_tset")
    keep_nt = np.empty(n, np.float32)
    keep_at = np.empty(n, np.float32)
    fake_P = {"bscore": np.zeros(n, np.float32), "brank": np.zeros(n, np.int16),
             "bits": np.zeros(n, np.uint16), "dcos": np.full(n, np.nan, np.float32)}
    for s in range(0, n, step):
        sl = slice(s, min(n, s + step))
        F = fz.chunk(qc[sl], ic[sl], fake_P, sl, None)
        out[sl, :len(BASE_FEATURES)] = F
        keep_nt[sl] = F[:, j_nt]
        keep_at[sl] = F[:, j_at]
    from features import context_features, CONTEXT_FEATURES
    ctx = context_features(qc.astype(np.int64), ic.astype(np.int64), keep_nt, keep_at,
                           fake_P["bscore"])
    for j, c in enumerate(CONTEXT_FEATURES):
        out[:, len(BASE_FEATURES) + j] = ctx[c]
    return out


def evaluate(cfg: Config, holdout_folds=(8, 9), reranker_dir: str | None = None,
            max_shortlist: int = 4_000_000):
    meta = load_json(os.path.join(model_dir(cfg), "meta.json"))
    m1 = load_models(cfg, "stage1")
    m2 = load_models(cfg, "stage2")
    import pickle
    with open(os.path.join(model_dir(cfg), "calibrator.pkl"), "rb") as f:
        iso = pickle.load(f)

    s1, idx, parts = load_split(cfg, "train", columns=["entity_id", "business_name",
                                                       "business_address", "country",
                                                       "n_core", "a_full"])
    fold_s1 = crc_fold(s1["entity_id"].to_numpy(dtype=object).tolist())
    n_s1 = len(s1)
    gt_s1, gt_idx = load_gt(cfg)
    G = np.bincount(gt_s1, minlength=n_s1).astype(np.float64)
    n_idx = len(idx)
    pos_keys = set((gt_s1.astype(np.int64) * n_idx + gt_idx.astype(np.int64)).tolist())

    Q_all, IG_all, Y_all, PC_all = [], [], [], []
    idx_ncore_all = idx["n_core"].to_numpy(dtype=object)
    idx_afull_all = idx["a_full"].to_numpy(dtype=object)
    for ck, (s1_rows, idx_rows) in parts.items():
        P = load_pairs(cfg, "train", ck)
        if len(P["q"]) == 0:
            continue
        fq = fold_s1[s1_rows[P["q"]]]
        m = np.isin(fq, holdout_folds)
        if not m.any():
            continue
        with timed(f"eval_reranker:{ck!r} holdout pairs={int(m.sum()):,}"):
            qloc = P["q"][m]
            iloc = P["i"][m]
            Q = s1.iloc[s1_rows].reset_index(drop=True)
            I = idx.iloc[idx_rows].reset_index(drop=True)
            feat = _features_for_subset(cfg, Q, I, qloc, iloc)
            p1 = np.empty(len(qloc), np.float32)
            for s in range(0, len(qloc), PRED_CHUNK):
                p1[s:s + PRED_CHUNK] = predict_rows(m1, feat[s:s + PRED_CHUNK])
            X2 = stage2_matrix(qloc.astype(np.int64), iloc.astype(np.int64), p1, feat,
                               idx_ncore_all[idx_rows], idx_afull_all[idx_rows])
            p2 = np.empty(len(qloc), np.float32)
            for s in range(0, len(qloc), PRED_CHUNK):
                p2[s:s + PRED_CHUNK] = predict_rows(m2, X2[s:s + PRED_CHUNK])
            pc = iso.predict(p2).astype(np.float32)
            gq = s1_rows[qloc].astype(np.int64)
            gi = idx_rows[iloc].astype(np.int64)
            key = gq * n_idx + gi
            y = np.fromiter((k in pos_keys for k in key.tolist()), dtype=np.int8, count=len(key))
            Q_all.append(gq)
            IG_all.append(gi)
            Y_all.append(y)
            PC_all.append(pc)

    q = np.concatenate(Q_all)
    ig = np.concatenate(IG_all)
    y = np.concatenate(Y_all)
    pc = np.concatenate(PC_all)

    with timed("eval_reranker:baseline decision (no reranker)"):
        sel = apply_decision(q, ig, pc, n_s1, meta["decision"])
        f05_base = macro_f05(q[sel], y[sel], G, None)
    LOG.info("HOLDOUT macro F0.5 WITHOUT reranker: %.5f (n_selected=%d)", f05_base, int(sel.sum()))

    if reranker_dir is None:
        reranker_dir = os.path.join(cfg.work_dir, "models", "reranker")
    if not os.path.isdir(reranker_dir):
        LOG.warning("no reranker at %s - skipping reranked comparison", reranker_dir)
        return f05_base, None

    from rerank import load_reranker, score_shortlist
    model = load_reranker(reranker_dir)

    ex = exclusive_mask(ig, pc)
    shortlist = np.flatnonzero(ex & (pc >= 0.02))  # only pairs plausible enough to matter
    if len(shortlist) > max_shortlist:
        order = np.argsort(-pc[shortlist])[:max_shortlist]
        shortlist = shortlist[order]
    LOG.info("reranking shortlist: %d pairs", len(shortlist))

    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    s1_names = s1["business_name"].to_numpy(dtype=object)
    s1_addrs = s1["business_address"].to_numpy(dtype=object)
    s1_ctry = s1["country"].to_numpy(dtype=object)
    idx_ids = idx["entity_id"].to_numpy(dtype=object)
    idx_names = idx["business_name"].to_numpy(dtype=object)
    idx_addrs = idx["business_address"].to_numpy(dtype=object)
    idx_ctry = idx["country"].to_numpy(dtype=object)
    # global id -> row lookup (train's s1/idx rows are 0..n-1 already = global here)
    with timed("eval_reranker:cross-encoder scoring"):
        p_ce_short = score_shortlist(model, s1_names[q[shortlist]], s1_addrs[q[shortlist]],
                                     s1_ctry[q[shortlist]], idx_names[ig[shortlist]],
                                     idx_addrs[ig[shortlist]], idx_ctry[ig[shortlist]])

    pc_re = pc.copy()
    for blend_w in (0.3, 0.5, 0.7, 1.0):
        pc_re[shortlist] = (1 - blend_w) * pc[shortlist] + blend_w * p_ce_short
        best, _ = tune(q, ig, pc_re, y, G, None, n_s1, "auto")
        sel_re = apply_decision(q, ig, pc_re, n_s1, best)
        f05_re = macro_f05(q[sel_re], y[sel_re], G, None)
        LOG.info("HOLDOUT macro F0.5 WITH reranker (blend=%.1f): %.5f  decision=%s",
                 blend_w, f05_re, best)
    return f05_base, None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--reranker-dir", default=None)
    a = ap.parse_args()
    from utils import setup_logging
    setup_logging()
    cfg = Config(data_dir=os.path.abspath(a.data_dir), work_dir=os.path.abspath(a.work_dir))
    meta = load_json(os.path.join(model_dir(cfg), "meta.json"))
    for k, v in meta["contract"].items():
        setattr(cfg, k, v)
    evaluate(cfg, reranker_dir=a.reranker_dir)
