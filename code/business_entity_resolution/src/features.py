"""Stage 2: pairwise + contextual features, streamed into an fp16 memmap per partition.

Every feature is country-agnostic (no country id, no one-hot) so the model transfers
to unseen labels such as France. String similarities run in rapidfuzz's C++ core via
process.cpdist(workers=-1) - element-wise, multi-threaded, zero Python loops per pair.
TF-IDF cosines use the hashing trick (no vocabulary in RAM) with IDF fitted on the
*current split's own unlabeled records* (transductive, label-free - adapts to French
tokens at test time).

Memory model: pairs are processed in chunks of cfg.feat_chunk rows; the only full-length
arrays kept in RAM are the handful of columns needed for group-context features.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import distance, fuzz, process
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from blocking import T_AN, T_BM25, T_DENSE, T_NC, T_NH, T_NP, T_NS, T_NT, T_PH
from utils import LOG, group_max, group_rank_desc, popcount16, safe_name, timed

BASE_FEATURES = [
    # name
    "n_ratio", "n_pratio", "n_tsort", "n_tset", "n_jw", "n_lev_full", "n_tset_full",
    "n_concat_ratio", "n_concat_pratio", "n_skel_ratio", "n_alias_best", "n_cos_char",
    "n_cos_word", "n_jacc", "n_first_eq", "n_len_q", "n_len_i", "n_ntok_diff",
    "i_translit", "i_domain", "i_alias",
    # address
    "a_empty_i", "a_ratio", "a_tsort", "a_tset", "a_pratio", "a_core_tset", "a_alpha_tset",
    "a_skel_ratio", "a_cos_word", "a_num_jacc", "a_first_num_eq", "a_qnum_in_i",
    "a_num_conflict", "a_state_eq", "a_len_i", "a_ncomp_i", "i_addr_translit",
    # blocking / structure
    "src_s3", "bscore", "brank", "b_ntypes", "bit_nt", "bit_ns", "bit_ph", "bit_an",
    "bit_nh", "bit_np", "bit_nc", "bit_bm25", "bit_dense", "dense_cos", "emb_cos",
    "n_phon_ratio",
    # frequency / ambiguity
    "q_name_df_s1", "i_name_df_idx", "i_name_df_s1",
]
CONTEXT_FEATURES = [
    "g_max_n_tset", "g_diff_n_tset", "g_rank_n_tset",
    "g_max_a_tset", "g_diff_a_tset", "g_rank_a_tset",
    "g_max_combo", "g_diff_combo", "g_rank_combo", "g_rank_bscore", "q_ncand",
    "i_ncand_s1", "i_rank_combo", "i_diff_combo", "i_rank_bscore",
]
ALL_FEATURES = BASE_FEATURES + CONTEXT_FEATURES
FIDX = {f: j for j, f in enumerate(ALL_FEATURES)}

# similarity features that must not decrease match probability (monotone +1)
MONOTONE_UP = {"n_ratio", "n_tsort", "n_tset", "n_jw", "n_cos_char", "n_cos_word",
               "n_jacc", "a_tset", "a_cos_word", "a_num_jacc", "a_core_tset", "n_phon_ratio"}

STR_COLS = ["n_full", "n_core", "n_concat", "n_alt_a", "n_alt_b", "n_skel", "n_phon",
            "a_full", "a_core", "a_alpha", "a_skel", "a_nums", "a_first_num", "a_state"]
FLAG_COLS = ["n_translit", "n_domain", "n_alias", "a_translit", "a_empty", "a_ncomp"]
MAXLEN = 200


# --------------------------------------------------------------------------- #
# vectorisers (hashing trick + transductive IDF)                               #
# --------------------------------------------------------------------------- #
def _hv(kind, n_features):
    if kind == "char":
        return HashingVectorizer(analyzer="char_wb", ngram_range=(2, 3), n_features=n_features,
                                 alternate_sign=False, norm=None, lowercase=False,
                                 dtype=np.float32)
    return HashingVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b",
                             n_features=n_features, alternate_sign=False, norm=None,
                             lowercase=False, dtype=np.float32)


_IPC_CHUNK = 100_000  # rows per task, capped independent of worker/record count


def _transform(hv, texts, workers):
    """Parallel stateless hashing transform.

    Chunk size is capped at _IPC_CHUNK regardless of `workers` or `n` - NOT n/workers.
    Windows' named-pipe IPC has an undocumented practical size ceiling: a `loky` worker
    sending back one large sparse-matrix chunk (e.g. ~258k rows/worker for a 6.19M-row
    partition split across 24 workers) fails with `OSError: [WinError 1450]
    Insufficient system resources exist to complete the requested service` - observed
    deterministically on this project's larger partitions. Splitting into many smaller
    chunks (more IPC messages, each safely small) avoids it entirely; total throughput
    is essentially unaffected since the work is still spread across all workers.
    """
    n = len(texts)
    if n < 400_000 or workers <= 1:
        return hv.transform(texts).tocsr()
    from joblib import Parallel, delayed
    step = min(_IPC_CHUNK, int(np.ceil(n / workers)))
    parts = Parallel(n_jobs=workers, backend="loky")(
        delayed(hv.transform)(texts[s:s + step]) for s in range(0, n, step))
    return sp.vstack(parts, format="csr")


def tfidf_pair(texts_q, texts_i, kind, n_features, workers, binary=False):
    hv = _hv(kind, n_features)
    Xq = _transform(hv, texts_q, workers)
    Xi = _transform(hv, texts_i, workers)
    if binary:
        for X in (Xq, Xi):
            X.data[:] = 1.0
        return Xq, Xi
    N = Xq.shape[0] + Xi.shape[0]
    df = (np.bincount(Xq.indices, minlength=n_features)
          + np.bincount(Xi.indices, minlength=n_features))
    idf = (np.log((1.0 + N) / (1.0 + df)) + 1.0).astype(np.float32)
    for X in (Xq, Xi):
        X.data = (1.0 + np.log(X.data)) * idf[X.indices]
        sk_normalize(X, norm="l2", copy=False)
    return Xq, Xi


def rowdot(A, B, qi, ii):
    return np.asarray(A[qi].multiply(B[ii]).sum(axis=1)).ravel().astype(np.float32)


def _row_nnz(X):
    return np.diff(X.indptr).astype(np.float32)


# --------------------------------------------------------------------------- #
# per-partition feature builder                                               #
# --------------------------------------------------------------------------- #
class PartitionFeaturizer:
    def __init__(self, Q: pd.DataFrame, I: pd.DataFrame, cfg, emb_q=None, emb_i=None):
        self.cfg = cfg
        self.W = -1
        w = max(1, (os.cpu_count() or 2))
        self.q = {c: Q[c].to_numpy(dtype=object) for c in STR_COLS}
        self.i = {c: I[c].to_numpy(dtype=object) for c in STR_COLS}
        for side, D in ((self.q, Q), (self.i, I)):
            for c in STR_COLS:
                D_c = side[c]
                side[c] = np.array([s[:MAXLEN] if len(s) > MAXLEN else s for s in D_c],
                                   dtype=object)
            for c in FLAG_COLS:   # works for numpy and Arrow-backed dtypes alike
                side[c] = D[c].astype("float32").to_numpy(dtype=np.float32)
        self.i_src3 = (I["src"].astype("int16").to_numpy(dtype=np.int16) == 3).astype(np.float32)
        nf = cfg.hash_features
        with timed("    tfidf matrices"):
            self.m_nchar = tfidf_pair(self.q["n_core"], self.i["n_core"], "char", nf, w)
            self.m_nword = tfidf_pair(self.q["n_core"], self.i["n_core"], "word", nf, w)
            self.m_aword = tfidf_pair(self.q["a_full"], self.i["a_full"], "word", nf, w)
            self.m_nbin = tfidf_pair(self.q["n_core"], self.i["n_core"], "word", nf, w, True)
            self.m_numbin = tfidf_pair(self.q["a_nums"], self.i["a_nums"], "word", nf, w, True)
            self.m_fnum = tfidf_pair(self.q["a_first_num"], self.i["a_first_num"], "word", nf,
                                     w, True)
        self.nb_q, self.nb_i = _row_nnz(self.m_nbin[0]), _row_nnz(self.m_nbin[1])
        self.nn_q, self.nn_i = _row_nnz(self.m_numbin[0]), _row_nnz(self.m_numbin[1])
        self.len_nq = np.fromiter(map(len, self.q["n_core"]), np.float32, len(Q))
        self.len_ni = np.fromiter(map(len, self.i["n_core"]), np.float32, len(I))
        self.len_ai = np.fromiter(map(len, self.i["a_full"]), np.float32, len(I))
        self.ntok_q = np.fromiter((s.count(" ") + 1 if s else 0 for s in self.q["n_core"]),
                                  np.float32, len(Q))
        self.ntok_i = np.fromiter((s.count(" ") + 1 if s else 0 for s in self.i["n_core"]),
                                  np.float32, len(I))
        self.first_q = np.array([s.split(" ", 1)[0] for s in self.q["n_core"]], dtype=object)
        self.first_i = np.array([s.split(" ", 1)[0] for s in self.i["n_core"]], dtype=object)
        # name frequency / ambiguity
        s1_counts = pd.Series(self.q["n_core"]).value_counts()
        idx_counts = pd.Series(self.i["n_core"]).value_counts()
        self.q_name_df_s1 = pd.Series(self.q["n_core"]).map(s1_counts).to_numpy(np.float32)
        self.i_name_df_idx = pd.Series(self.i["n_core"]).map(idx_counts).to_numpy(np.float32)
        self.i_name_df_s1 = (pd.Series(self.i["n_core"]).map(s1_counts).fillna(0)
                             .to_numpy(np.float32))
        self.emb_q, self.emb_i = emb_q, emb_i

    # ------------------------------------------------------------------ #
    def _cp(self, a, b, scorer, scale=100.0, skip=None):
        """Pairwise string scores. Skips empty/gated via `skip` (True=leave NaN)
        and short-circuits exact string equals to 1.0 without calling rapidfuzz.
        """
        n = len(a)
        if n == 0:
            return np.zeros(0, np.float32)
        out = np.full(n, np.nan, dtype=np.float32)
        need = np.ones(n, dtype=bool) if skip is None else ~np.asarray(skip, dtype=bool)
        aa = np.asarray(a, dtype=object)
        bb = np.asarray(b, dtype=object)
        same = need & (aa == bb)
        out[same] = 1.0
        need = need & ~same
        idx = np.flatnonzero(need)
        if len(idx) == 0:
            return out
        r = process.cpdist(aa[idx].tolist(), bb[idx].tolist(),
                           scorer=scorer, workers=self.W, dtype=np.float32)
        out[idx] = (r / scale).astype(np.float32)
        return out

    def chunk(self, qc, ic, P, sl, emb_rows=None):
        """Base features for pairs (qc, ic); P holds blocking arrays; sl the slice."""
        Q, I = self.q, self.i
        n = len(qc)
        F = np.full((n, len(BASE_FEATURES)), np.nan, dtype=np.float32)
        col = {f: j for j, f in enumerate(BASE_FEATURES)}

        def put(name, v):
            F[:, col[name]] = v

        # Two-stage gate: expensive RapidFuzz only for top-brank (and dense hits).
        # brank is 0-based within the key-union list. 0 = off (all pairs get fuzzy).
        fuzz_k = int(getattr(self.cfg, "feat_fuzz_brank_max", 0) or 0)
        bits = P["bits"][sl]
        if fuzz_k > 0:
            br = P["brank"][sl]
            dense_hit = ((bits >> T_DENSE) & 1).astype(bool)
            do_fuzz = (br < fuzz_k) | dense_hit
        else:
            do_fuzz = np.ones(n, dtype=bool)

        qn, inn = Q["n_core"][qc], I["n_core"][ic]
        qn_l, in_l = qn.tolist(), inn.tolist()
        ne = (self.len_nq[qc] == 0) | (self.len_ni[ic] == 0)
        skip_name = ne | ~do_fuzz
        for name, scorer in (("n_ratio", fuzz.ratio), ("n_pratio", fuzz.partial_ratio),
                             ("n_tsort", fuzz.token_sort_ratio),
                             ("n_tset", fuzz.token_set_ratio)):
            put(name, self._cp(qn_l, in_l, scorer, skip=skip_name))
        put("n_jw", self._cp(qn_l, in_l, distance.JaroWinkler.normalized_similarity,
                             1.0, skip=skip_name))
        qf, if_ = Q["n_full"][qc].tolist(), I["n_full"][ic].tolist()
        put("n_lev_full", self._cp(qf, if_, distance.Levenshtein.normalized_similarity,
                                   1.0, skip=skip_name))
        put("n_tset_full", self._cp(qf, if_, fuzz.token_set_ratio, skip=skip_name))
        qc_l, ic_l = Q["n_concat"][qc].tolist(), I["n_concat"][ic].tolist()
        put("n_concat_ratio", self._cp(qc_l, ic_l, fuzz.ratio, skip=skip_name))
        put("n_concat_pratio", self._cp(qc_l, ic_l, fuzz.partial_ratio, skip=skip_name))
        put("n_skel_ratio", self._cp(Q["n_skel"][qc].tolist(), I["n_skel"][ic].tolist(),
                                     fuzz.token_sort_ratio, skip=skip_name))
        put("n_phon_ratio", self._cp(Q["n_phon"][qc].tolist(), I["n_phon"][ic].tolist(),
                                     fuzz.token_set_ratio, skip=skip_name))
        al_a = self._cp(qn_l, I["n_alt_a"][ic].tolist(), fuzz.token_set_ratio, skip=skip_name)
        al_b = self._cp(qn_l, I["n_alt_b"][ic].tolist(), fuzz.token_set_ratio, skip=skip_name)
        put("n_alias_best", np.fmax(al_a, al_b))
        put("n_cos_char", rowdot(*self.m_nchar, qc, ic))
        put("n_cos_word", rowdot(*self.m_nword, qc, ic))
        inter = rowdot(*self.m_nbin, qc, ic)
        union = self.nb_q[qc] + self.nb_i[ic] - inter
        put("n_jacc", np.where(union > 0, inter / np.maximum(union, 1), np.nan))
        put("n_first_eq", (self.first_q[qc] == self.first_i[ic]).astype(np.float32))
        put("n_len_q", self.len_nq[qc])
        put("n_len_i", self.len_ni[ic])
        put("n_ntok_diff", np.abs(self.ntok_q[qc] - self.ntok_i[ic]))
        put("i_translit", I["n_translit"][ic])
        put("i_domain", I["n_domain"][ic])
        put("i_alias", I["n_alias"][ic])

        # ---------------- address ----------------
        a_empty = I["a_empty"][ic]
        put("a_empty_i", a_empty)
        ae = (a_empty > 0) | (Q["a_empty"][qc] > 0)
        skip_addr = ae | ~do_fuzz
        qa, ia = Q["a_full"][qc].tolist(), I["a_full"][ic].tolist()
        for name, scorer in (("a_ratio", fuzz.ratio), ("a_tsort", fuzz.token_sort_ratio),
                             ("a_tset", fuzz.token_set_ratio), ("a_pratio", fuzz.partial_ratio)):
            put(name, self._cp(qa, ia, scorer, skip=skip_addr))
        put("a_core_tset", self._cp(Q["a_core"][qc].tolist(), I["a_core"][ic].tolist(),
                                    fuzz.token_set_ratio, skip=skip_addr))
        put("a_alpha_tset", self._cp(Q["a_alpha"][qc].tolist(), I["a_alpha"][ic].tolist(),
                                     fuzz.token_set_ratio, skip=skip_addr))
        put("a_skel_ratio", self._cp(Q["a_skel"][qc].tolist(), I["a_skel"][ic].tolist(),
                                     fuzz.token_set_ratio, skip=skip_addr))
        v = rowdot(*self.m_aword, qc, ic)
        v[ae] = np.nan
        put("a_cos_word", v)
        inter = rowdot(*self.m_numbin, qc, ic)
        nq_, ni_ = self.nn_q[qc], self.nn_i[ic]
        both = (nq_ > 0) & (ni_ > 0)
        put("a_num_jacc", np.where(both, inter / np.maximum(nq_ + ni_ - inter, 1), np.nan))
        put("a_num_conflict", np.where(both, (inter == 0).astype(np.float32), np.nan))
        fq, fi = Q["a_first_num"][qc], I["a_first_num"][ic]
        has_f = (fq != "") & (fi != "")
        put("a_first_num_eq", np.where(has_f, (fq == fi).astype(np.float32), np.nan))
        hit = rowdot(self.m_fnum[0], self.m_numbin[1], qc, ic)
        put("a_qnum_in_i", np.where((fq != "") & (ni_ > 0), (hit > 0).astype(np.float32), np.nan))
        sq, si = Q["a_state"][qc], I["a_state"][ic]
        has_s = (sq != "") & (si != "")
        put("a_state_eq", np.where(has_s, (sq == si).astype(np.float32), np.nan))
        put("a_len_i", self.len_ai[ic])
        put("a_ncomp_i", I["a_ncomp"][ic])
        put("i_addr_translit", I["a_translit"][ic])

        # ---------------- blocking / structure ----------------
        bits = P["bits"][sl]
        put("src_s3", self.i_src3[ic])
        put("bscore", P["bscore"][sl])
        put("brank", P["brank"][sl].astype(np.float32))
        put("b_ntypes", popcount16(bits))
        for name, t in (("bit_nt", T_NT), ("bit_ns", T_NS), ("bit_ph", T_PH), ("bit_an", T_AN),
                        ("bit_nh", T_NH), ("bit_np", T_NP), ("bit_nc", T_NC),
                        ("bit_bm25", T_BM25), ("bit_dense", T_DENSE)):
            put(name, ((bits >> t) & 1).astype(np.float32))
        put("dense_cos", P["dcos"][sl])
        if self.emb_q is not None and emb_rows is not None:
            eq = np.asarray(self.emb_q[emb_rows[0]], dtype=np.float32)
            ei = np.asarray(self.emb_i[emb_rows[1]], dtype=np.float32)
            put("emb_cos", np.einsum("ij,ij->i", eq, ei))
        put("q_name_df_s1", self.q_name_df_s1[qc])
        put("i_name_df_idx", self.i_name_df_idx[ic])
        put("i_name_df_s1", self.i_name_df_s1[ic])
        return F


def context_features(q, i, n_tset, a_tset, bscore):
    """Group-relative features: how a candidate compares to its rivals on both sides."""
    combo = 0.5 * np.nan_to_num(n_tset) + 0.5 * np.nan_to_num(a_tset)
    out = {}
    for name, v in (("n_tset", n_tset), ("a_tset", a_tset), ("combo", combo)):
        mx = group_max(q, v)
        out[f"g_max_{name}"] = mx
        out[f"g_diff_{name}"] = (np.nan_to_num(v, nan=0) - np.nan_to_num(mx, nan=0)).astype(np.float32)
        out[f"g_rank_{name}"] = group_rank_desc(q, v)
    out["g_rank_bscore"] = group_rank_desc(q, bscore)
    out["q_ncand"] = np.bincount(q)[q].astype(np.float32)
    out["i_ncand_s1"] = np.bincount(i)[i].astype(np.float32)
    out["i_rank_combo"] = group_rank_desc(i, combo)
    out["i_diff_combo"] = (combo - group_max(i, combo)).astype(np.float32)
    out["i_rank_bscore"] = group_rank_desc(i, bscore)
    return out


def feat_path(cfg, split, ck):
    from prepare import split_dir
    return os.path.join(split_dir(cfg, split), f"feat_{safe_name(ck)}.npy")


def run_features(cfg, split: str, s1, idx, parts, emb=None) -> None:
    from blocking import load_pairs
    for ck, (s1_rows, idx_rows) in parts.items():
        path = feat_path(cfg, split, ck)
        if os.path.exists(path) and not cfg.force:
            continue
        P = load_pairs(cfg, split, ck)
        n = len(P["q"])
        tmp = path + ".tmp.npy"
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                       shape=(n, len(ALL_FEATURES)))
        if n == 0:
            mm.flush()
            del mm
            os.replace(tmp, path)
            continue
        with timed(f"features:{split}:{ck!r} pairs={n:,}"):
            Q = s1.iloc[s1_rows].reset_index(drop=True)
            I = idx.iloc[idx_rows].reset_index(drop=True)
            fz = PartitionFeaturizer(Q, I, cfg,
                                     emb[0] if emb is not None else None,
                                     emb[1] if emb is not None else None)
            keep_nt = np.empty(n, np.float32)
            keep_at = np.empty(n, np.float32)
            j_nt, j_at = BASE_FEATURES.index("n_tset"), BASE_FEATURES.index("a_tset")
            step = cfg.feat_chunk
            for s in range(0, n, step):
                sl = slice(s, min(n, s + step))
                qc, ic = P["q"][sl], P["i"][sl]
                emb_rows = (s1_rows[qc], idx_rows[ic]) if emb is not None else None
                F = fz.chunk(qc, ic, P, sl, emb_rows)
                keep_nt[sl] = F[:, j_nt]
                keep_at[sl] = F[:, j_at]
                mm[sl, :len(BASE_FEATURES)] = F.astype(np.float16)
                LOG.info("    chunk %d-%d done", sl.start, sl.stop)
            ctx = context_features(P["q"].astype(np.int64), P["i"].astype(np.int64),
                                   keep_nt, keep_at, P["bscore"].astype(np.float32))
            C = np.column_stack([ctx[c] for c in CONTEXT_FEATURES]).astype(np.float16)
            for s in range(0, n, step):
                mm[s:s + step, len(BASE_FEATURES):] = C[s:s + step]
            mm.flush()
            del mm, C, fz
        os.replace(tmp, path)


def load_features(cfg, split, ck):
    return np.load(feat_path(cfg, split, ck), mmap_mode="r")
