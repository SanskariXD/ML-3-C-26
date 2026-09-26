#!/usr/bin/env python3
"""Does schema-agnostic BM25 top-k blocking beat our key-based blocker?

Motivation (literature): Paulsen, Govind & Doan, "Sparkly: A Simple yet Surprisingly
Strong TF/IDF Blocker for Entity Matching", PVLDB 16(7):1507-1519, 2023
(https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf) report that top-k TF/IDF blocking
with Lucene outperforms 8 state-of-the-art blockers, and explicitly recommend top-k
blocking because it improves recall. Papadakis et al., "Blocking and Filtering
Techniques for Entity Resolution: A Survey" (ACM CSUR 2020, arXiv:1905.06167) likewise
find schema-agnostic blocking reaches higher recall than schema-based.

Why this should matter HERE, mechanically:
  our blocker matches exact keys and then DISCARDS any key whose document frequency
  exceeds max_df_pair=500 / max_df_name=200 (0.16% / 0.06% of a 310k-record index).
  A true pair whose only shared tokens are common ones therefore gets no candidate at
  all. BM25 top-k never discards: a common term simply earns little score, so the pair
  still surfaces when nothing better competes. FINDINGS.md measured that 99.92% of
  missed pairs DO share a token -- exactly the population this difference addresses.

This script only MEASURES. It changes no pipeline behaviour. Run it to decide whether
rebuilding blocking is justified (see the promotion gate in AGENTS.md).

    .venv/bin/python src/blocking_probe.py --work-dir work_dev

Compare its recall-vs-candidates curve against the incumbent numbers in
work_dev/models/report.json (holdout_blocking_recall, pairs_per_s1).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config            # noqa: E402
from prepare import load_gt          # noqa: E402
from utils import crc_fold           # noqa: E402

TOK = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(name: str, addr: str) -> list[str]:
    """Schema-agnostic bag of tokens: name and address pooled, plus name char 4-grams.

    Char n-grams are what let a transliterated or typo'd name still share signal; the
    survey above calls this the token-blocking / q-gram-blocking union.
    """
    n = (name or "").lower()
    a = (addr or "").lower()
    out = [f"n:{t}" for t in TOK.findall(n)] + [f"a:{t}" for t in TOK.findall(a)]
    squash = re.sub(r"[^0-9a-z\u0900-\u097f]", "", n)
    out += [f"g:{squash[i:i+4]}" for i in range(0, max(0, len(squash) - 3))]
    return out


def build_index(docs: list[list[str]], k1: float = 1.2, b: float = 0.75,
                max_df_ratio: float = 0.20):
    """BM25-weighted doc-term CSR matrix. Only absurdly common terms are dropped.

    max_df_ratio=0.20 is ~125x looser than the incumbent max_df_pair=500 on a 310k
    index; terms that common have near-zero IDF, so pruning them barely perturbs the
    top-k ordering while keeping the matmul tractable.
    """
    vocab: dict[str, int] = {}
    rows, cols, tfs = [], [], []
    for i, toks in enumerate(docs):
        c = defaultdict(int)
        for t in toks:
            c[t] += 1
        for t, f in c.items():
            j = vocab.setdefault(t, len(vocab))
            rows.append(i); cols.append(j); tfs.append(f)
    n_doc, n_vocab = len(docs), len(vocab)
    tf = sp.csr_matrix((np.asarray(tfs, np.float32), (rows, cols)),
                       shape=(n_doc, n_vocab))
    df = np.asarray((tf > 0).sum(axis=0)).ravel()
    dl = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = max(1e-9, dl.mean())
    idf = np.log(1.0 + (n_doc - df + 0.5) / (df + 0.5)).astype(np.float32)
    keep = df <= max_df_ratio * n_doc
    idf = np.where(keep, idf, 0.0).astype(np.float32)

    # BM25 term weight per (doc, term)
    tf = tf.tocoo()
    denom = tf.data + k1 * (1.0 - b + b * dl[tf.row] / avgdl)
    w = idf[tf.col] * tf.data * (k1 + 1.0) / denom
    D = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)),
                      shape=(n_doc, n_vocab))
    D.eliminate_zeros()
    return D, vocab, idf


def query_matrix(queries: list[list[str]], vocab: dict, idf: np.ndarray):
    rows, cols, vals = [], [], []
    for i, toks in enumerate(queries):
        for t in set(toks):
            j = vocab.get(t)
            if j is not None and idf[j] > 0:
                rows.append(i); cols.append(j); vals.append(1.0)
    return sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                         shape=(len(queries), len(vocab)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work_dev")
    ap.add_argument("--ks", default="5,10,15,20,30,40")
    ap.add_argument("--chunk", type=int, default=400)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.work_dir, "models", "meta.json")))
    cfg = Config.from_dict(meta["config"])
    cfg.work_dir = a.work_dir
    report = json.load(open(os.path.join(a.work_dir, "models", "report.json")))

    d = os.path.join(a.work_dir, "train")
    s1 = pd.read_parquet(os.path.join(d, "s1.parquet"),
                         columns=["entity_id", "business_name", "business_address",
                                  "country"])
    idx = pd.read_parquet(os.path.join(d, "idx.parquet"),
                          columns=["entity_id", "business_name", "business_address",
                                   "country"])
    gt_s1, gt_idx = load_gt(cfg)
    fold = crc_fold(s1["entity_id"].tolist())
    hold = np.isin(fold, list(cfg.holdout_folds))

    # evaluate on exactly the holdout entities the report uses
    q_rows = np.flatnonzero(hold)
    truth = defaultdict(set)
    for qq, ii in zip(gt_s1.tolist(), gt_idx.tolist()):
        truth[qq].add(ii)
    n_true = sum(len(truth[q]) for q in q_rows)
    print(f"index records {len(idx):,} | holdout entities {len(q_rows):,} | "
          f"true pairs {n_true:,}")
    print("blocking must be country-consistent, so we restrict per country partition\n")

    # country partitioning (matches the pipeline: matches never cross countries)
    s1c = s1["country"].astype(str).str.strip().str.casefold().to_numpy()
    idc = idx["country"].astype(str).str.strip().str.casefold().to_numpy()

    # ---- incumbent candidate set, for union / complementarity analysis
    import pickle
    with open(os.path.join(d, "partitions.pkl"), "rb") as fh:
        parts = pickle.load(fh)
    inc = defaultdict(set)
    for ck_, (s1_rows, idx_rows) in parts.items():
        for fn in os.listdir(d):
            if fn.startswith(f"pairs_{ck_}_") and fn.endswith(".npz"):
                z = np.load(os.path.join(d, fn))
                key = "q" if "q" in z.files else z.files[0]
                qq = s1_rows[z[key]]
                ii = idx_rows[z["i" if "i" in z.files else z.files[1]]]
                for x, y in zip(qq.tolist(), ii.tolist()):
                    inc[x].add(y)
    inc_hits = sum(len(truth[q] & inc.get(q, set())) for q in q_rows)
    inc_cands = sum(len(inc.get(q, set())) for q in q_rows)
    print(f"incumbent recomputed: recall={inc_hits/max(1,n_true):.4%} "
          f"cands/entity={inc_cands/max(1,len(q_rows)):.2f}\n")

    ks = [int(x) for x in a.ks.split(",")]
    hits = {k: 0 for k in ks}
    cand = {k: 0 for k in ks}
    uni_hits = {k: 0 for k in ks}
    uni_cand = {k: 0 for k in ks}
    rescued = {k: 0 for k in ks}
    missed_total = n_true - inc_hits
    # exact oracle F0.5 ceiling: a perfect matcher restricted to the candidate set.
    # F = 1.25*TP/(0.25*G + TP) with P == TP; G == 0 scores 1.0 (predict empty).
    orc_inc = 0.0
    orc_uni = {k: 0.0 for k in ks}

    def _orc(G_, TP_):
        return 1.0 if G_ == 0 else (1.25 * TP_ / (0.25 * G_ + TP_) if TP_ else 0.0)

    for q_ in q_rows:
        t_ = truth.get(q_, set())
        orc_inc += _orc(len(t_), len(t_ & inc.get(q_, set())))

    for ck in sorted(set(s1c[q_rows])):
        qsel = q_rows[s1c[q_rows] == ck]
        # blank-country index records are visible to every partition, as in prepare.py
        isel = np.flatnonzero((idc == ck) | (idc == ""))
        if not len(qsel) or not len(isel):
            continue
        docs = [tokenize(n, ad) for n, ad in
                zip(idx["business_name"].to_numpy(dtype=object)[isel],
                    idx["business_address"].to_numpy(dtype=object)[isel])]
        D, vocab, idf = build_index(docs)
        qt = [tokenize(n, ad) for n, ad in
              zip(s1["business_name"].to_numpy(dtype=object)[qsel],
                  s1["business_address"].to_numpy(dtype=object)[qsel])]
        Q = query_matrix(qt, vocab, idf)
        Dt = D.T.tocsr()
        kmax = max(ks)
        print(f"  partition {ck!r}: queries={len(qsel):,} docs={len(isel):,} "
              f"vocab={len(vocab):,}")
        for s in range(0, Q.shape[0], a.chunk):
            S = (Q[s:s + a.chunk] @ Dt).toarray()
            if S.size == 0:
                continue
            kk = min(kmax, S.shape[1])
            part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            rowsc = np.arange(S.shape[0])[:, None]
            ordr = np.argsort(-S[rowsc, part], axis=1)
            top = part[rowsc, ordr]
            sc = S[rowsc, top]
            for r in range(S.shape[0]):
                qglob = qsel[s + r]
                tset = truth.get(qglob, set())
                valid = top[r][sc[r] > 0]
                mapped = isel[valid]
                iset = inc.get(qglob, set())
                inc_found = tset & iset
                for k in ks:
                    sub = set(mapped[:k].tolist())
                    cand[k] += len(sub)
                    if tset:
                        hits[k] += len(tset & sub)
                        u = iset | sub
                        uni_hits[k] += len(tset & u)
                        # true pairs the incumbent missed that BM25 top-k recovers
                        rescued[k] += len((tset - inc_found) & sub)
                    uni_cand[k] += len(iset | sub)
                for k in ks:
                    sub = set(mapped[:k].tolist())
                    orc_uni[k] += _orc(len(tset), len(tset & (iset | sub)))

    print("\n=== BM25 top-k (schema-agnostic: name + address + name 4-grams) ===")
    print(f"{'k':>4s} {'recall':>10s} {'cands/entity':>14s}")
    print("-" * 32)
    for k in ks:
        print(f"{k:4d} {hits[k]/max(1,n_true):9.4%} {cand[k]/max(1,len(q_rows)):13.2f}")

    print(f"\n=== INCUMBENT (key-based blocker, same slice/holdout) ===")
    print(f"     {report['holdout_blocking_recall']:9.4%} "
          f"{report['pairs_per_s1']:13.2f}   (report.json)")

    nq = max(1, len(q_rows))
    print("\n=== UNION: incumbent + BM25 top-k  (the question that matters) ===")
    print(f"{'k':>4s} {'union recall':>13s} {'cands/ent':>11s} {'rescued':>9s} "
          f"{'ORACLE F0.5':>12s} {'vs incumbent':>13s}")
    print("-" * 68)
    base_orc = orc_inc / nq
    for k in ks:
        o = orc_uni[k] / nq
        print(f"{k:4d} {uni_hits[k]/max(1,n_true):12.4%} "
              f"{uni_cand[k]/nq:10.2f} {rescued[k]/max(1,missed_total):8.1%} "
              f"{o:12.6f} {o - base_orc:+13.6f}")
    print(f"\nincumbent oracle F0.5 ceiling on this holdout: {base_orc:.6f}")
    print("ORACLE F0.5 is the hard upper bound a perfect matcher could reach on that "
          "candidate set -- this is the number to weigh against the +0.005 promotion "
          "gate in AGENTS.md, not the raw recall.")
    print(f"\nincumbent missed {missed_total:,} true pairs on this holdout.")
    print("Read: BM25 alone losing is not the point. If the UNION lifts recall "
          "meaningfully for a small increase in candidates/entity, the two blockers are "
          "complementary and unioning them is the win. If the rescue rate is near zero, "
          "they fail on the same pairs and BM25 adds nothing -- drop this direction.")


if __name__ == "__main__":
    main()
