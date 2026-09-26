"""Apply a validated reranker to an already-finished `predict` run's test output.

Cheap and safe: reuses work/test/final_scores.npz (q, i, p, sel - saved by every
`predict` run) instead of recomputing anything upstream. Re-scores only the shortlist
(pairs with p >= --min-p, one-owner-exclusive), blends it into the calibrated stage-2
probability, re-optimises the decision on TRAIN holdout (via eval_reranker's tuned
blend/decision if given, else re-tunes fresh here is not possible without labels - so
this script takes the blend weight and decision explicitly, both chosen from
`eval-reranker`'s holdout numbers) and rewrites matching_results.tsv /
candidate_pairs.tsv, then re-validates.

Run only AFTER `eval-reranker` has shown a real holdout improvement, with the same
blend_weight/decision it reported as best.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

from config import Config
from decide import apply_decision, exclusive_mask
from io_utils import CAND_HEADER, MATCH_HEADER, run_official_validator, self_check, write_id_lists
from pipeline import model_dir
from prepare import load_split, split_dir
from rerank import load_reranker, score_shortlist
from utils import LOG, load_json, setup_logging, timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--reranker-dir", default=None)
    ap.add_argument("--blend-weight", type=float, required=True,
                    help="from eval-reranker's best HOLDOUT result")
    ap.add_argument("--decision-method", choices=["threshold", "expected_f"], required=True)
    ap.add_argument("--decision-tau", type=float, required=True)
    ap.add_argument("--decision-gamma", type=float, default=1.0)
    ap.add_argument("--decision-miss", type=float, default=0.0)
    ap.add_argument("--min-p", type=float, default=0.02)
    a = ap.parse_args()
    setup_logging()

    cfg = Config(data_dir=os.path.abspath(a.data_dir), work_dir=os.path.abspath(a.work_dir),
                out_dir=os.path.abspath(a.out_dir))
    guess = os.path.join(os.path.dirname(cfg.data_dir), "utils", "validate_submission.py")
    cfg.validator = guess if os.path.isfile(guess) else ""
    meta = load_json(os.path.join(model_dir(cfg), "meta.json"))
    for k, v in meta["contract"].items():
        setattr(cfg, k, v)

    z = np.load(os.path.join(split_dir(cfg, "test"), "final_scores.npz"))
    q, ig, pc = z["q"], z["i"], z["p"].astype(np.float32)

    s1, idx, _ = load_split(cfg, "test", columns=["entity_id", "business_name",
                                                  "business_address", "country"])
    n_s1 = len(s1)

    reranker_dir = a.reranker_dir or os.path.join(cfg.work_dir, "models", "reranker")
    model = load_reranker(reranker_dir)

    ex = exclusive_mask(ig, pc)
    shortlist = np.flatnonzero(ex & (pc >= a.min_p))
    LOG.info("re-scoring shortlist: %d / %d pairs", len(shortlist), len(pc))

    s1_names = s1["business_name"].to_numpy(dtype=object)
    s1_addrs = s1["business_address"].to_numpy(dtype=object)
    s1_ctry = s1["country"].to_numpy(dtype=object)
    idx_names = idx["business_name"].to_numpy(dtype=object)
    idx_addrs = idx["business_address"].to_numpy(dtype=object)
    idx_ctry = idx["country"].to_numpy(dtype=object)

    with timed("cross-encoder scoring shortlist"):
        p_ce = score_shortlist(model, s1_names[q[shortlist]], s1_addrs[q[shortlist]],
                               s1_ctry[q[shortlist]], idx_names[ig[shortlist]],
                               idx_addrs[ig[shortlist]], idx_ctry[ig[shortlist]])

    pc_re = pc.copy()
    pc_re[shortlist] = (1 - a.blend_weight) * pc[shortlist] + a.blend_weight * p_ce

    if a.decision_method == "threshold":
        dcfg = {"method": "threshold", "tau": a.decision_tau}
    else:
        dcfg = {"method": "expected_f", "tau": a.decision_tau, "gamma": a.decision_gamma,
                "miss": a.decision_miss}
    sel = apply_decision(q, ig, pc_re, n_s1, dcfg)
    LOG.info("reranked selection: %d pairs (was %d)", int(sel.sum()), int(z["sel"].sum()))

    s1_ids = s1["entity_id"].to_numpy(dtype=object)
    idx_ids = idx["entity_id"].to_numpy(dtype=object)
    os.makedirs(cfg.out_dir, exist_ok=True)
    mpath = os.path.join(cfg.out_dir, "matching_results.tsv")
    cpath = os.path.join(cfg.out_dir, "candidate_pairs.tsv")
    write_id_lists(mpath, MATCH_HEADER, s1_ids, q[sel], idx_ids[ig[sel]], pc_re[sel])
    write_id_lists(cpath, CAND_HEADER, s1_ids, q, idx_ids[ig], pc_re)
    required = s1_ids.tolist()
    for path, header in ((mpath, MATCH_HEADER), (cpath, CAND_HEADER)):
        errs = self_check(path, header, required)
        if errs:
            raise RuntimeError(f"{path} failed self-check: {errs[:5]}")
        LOG.info("self-check PASS: %s", path)
    rc = run_official_validator(cfg.validator, mpath, cpath, os.path.join(cfg.data_dir, "test"))
    if rc != 0:
        raise RuntimeError("official validator FAILED - see log")
    LOG.info("done: %s, %s regenerated with the reranker applied", mpath, cpath)


if __name__ == "__main__":
    main()
