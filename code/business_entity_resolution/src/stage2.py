"""Stage-2 ("collective") features built on top of stage-1 probabilities.

Idea: records of one real entity look like *each other*, and every S2/S3 record has
exactly one owner. So a candidate's evidence should include
  * how it ranks among its S1's candidates   (group context on p1)
  * whether a *different* S1 wants it more   (owner competition on p1)
  * how similar it is to the S1's most confident candidates (cluster support)
All ops are vectorised group-bys; string similarities run through rapidfuzz cpdist.
"""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz, process

from features import FIDX
from utils import group_top2

S2_CORE = ["n_tset", "a_tset", "n_cos_char", "a_cos_word", "a_num_jacc", "a_first_num_eq",
           "a_empty_i", "bscore", "i_name_df_s1", "q_name_df_s1", "n_alias_best",
           "a_state_eq", "emb_cos", "i_domain", "i_translit"]
S2_FEATURES = ["p1_logit", "g_rank_p1", "g_max_p1", "g_diff_p1", "g_second_p1", "g_sum_p1",
               "g_cnt50_p1", "q_ncand", "i_ncand_s1", "i_rank_p1", "i_max_other_p1",
               "i_diff_other_p1", "c2t_n1", "c2t_a1", "c2t_n2", "c2t_a2",
               "support_n", "support_a"] + S2_CORE
S2_MONOTONE_UP = {"p1_logit", "support_a", "support_n"}
_CHUNK = 4_000_000


def _cp_masked(a_idx, b_idx, strings, valid):
    """token_set_ratio(strings[a], strings[b]) where valid, NaN elsewhere (chunked)."""
    out = np.full(len(a_idx), np.nan, dtype=np.float32)
    rows = np.flatnonzero(valid)
    for s in range(0, len(rows), _CHUNK):
        r = rows[s:s + _CHUNK]
        a = strings[a_idx[r]].tolist()
        b = strings[b_idx[r]].tolist()
        out[r] = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1,
                                dtype=np.float32) / 100.0
    return out


def stage2_matrix(q, i, p1, mm, i_ncore, i_afull, out=None) -> np.ndarray:
    """q, i: local indices (int64); p1: stage-1 prob; mm: stage-1 feature memmap.

    out: optional pre-allocated (n, len(S2_FEATURES)) array / memmap (fp16 on disk to
    halve storage at full scale; LightGBM ingests it upcast to fp32, same pattern as the
    stage-1 feature memmap). All internal arithmetic here runs in float64/float32 and is
    only downcast on the final assignment into `X`, so fp16 storage costs no precision
    a tree model would use anyway (every feature is binned to <=255 levels regardless).
    """
    n = len(q)
    if out is None:
        X = np.full((n, len(S2_FEATURES)), np.nan, dtype=np.float32)
    else:
        X = out
        for s in range(0, n, _CHUNK):
            X[s:s + _CHUNK] = np.nan
    col = {f: j for j, f in enumerate(S2_FEATURES)}
    pc = np.clip(p1.astype(np.float64), 1e-6, 1 - 1e-6)
    X[:, col["p1_logit"]] = np.log(pc / (1 - pc))
    rank0, t1, t2 = group_top2(q, p1)
    X[:, col["g_rank_p1"]] = rank0 + 1
    X[:, col["g_max_p1"]] = t1
    X[:, col["g_second_p1"]] = t2
    X[:, col["g_diff_p1"]] = p1 - t1
    X[:, col["g_sum_p1"]] = np.bincount(q, weights=p1)[q]
    X[:, col["g_cnt50_p1"]] = np.bincount(q, weights=(p1 > 0.5).astype(np.float64))[q]
    X[:, col["q_ncand"]] = np.bincount(q)[q]
    X[:, col["i_ncand_s1"]] = np.bincount(i)[i]
    r_i, it1, it2 = group_top2(i, p1)
    other = np.where(r_i == 0, it2, it1)
    X[:, col["i_rank_p1"]] = r_i + 1
    X[:, col["i_max_other_p1"]] = other
    X[:, col["i_diff_other_p1"]] = p1 - np.nan_to_num(other, nan=0.0)

    # top-3 candidates of each S1 by p1 -> two "reference" rows for each pair
    nq = int(q.max()) + 1 if n else 0
    order = np.lexsort((-p1, q))
    qs = q[order]
    from utils import rank_in_sorted_groups
    rk = rank_in_sorted_groups(qs)
    T = np.full((nq, 3), -1, dtype=np.int64)
    m = rk < 3
    T[qs[m], rk[m]] = order[m]
    rows = np.arange(n)
    T0, T1, T2 = T[q, 0], T[q, 1], T[q, 2]
    ref1 = np.where(T0 != rows, T0, T1)
    ref2 = np.where((T0 != rows) & (T1 != rows), T1, T2)
    v1, v2 = ref1 >= 0, ref2 >= 0
    r1, r2 = np.where(v1, ref1, 0), np.where(v2, ref2, 0)
    c_n1 = _cp_masked(i, i[r1], i_ncore, v1)
    c_a1 = _cp_masked(i, i[r1], i_afull, v1 & (i_afull[i] != "") & (i_afull[i[r1]] != ""))
    c_n2 = _cp_masked(i, i[r2], i_ncore, v2)
    c_a2 = _cp_masked(i, i[r2], i_afull, v2 & (i_afull[i] != "") & (i_afull[i[r2]] != ""))
    X[:, col["c2t_n1"]], X[:, col["c2t_a1"]] = c_n1, c_a1
    X[:, col["c2t_n2"]], X[:, col["c2t_a2"]] = c_n2, c_a2
    w1 = np.where(v1, p1[r1], 0.0)
    w2 = np.where(v2, p1[r2], 0.0)
    den = w1 + w2 + 1e-6
    X[:, col["support_n"]] = (w1 * np.nan_to_num(c_n1) + w2 * np.nan_to_num(c_n2)) / den
    X[:, col["support_a"]] = (w1 * np.nan_to_num(c_a1) + w2 * np.nan_to_num(c_a2)) / den

    core_idx = [FIDX[c] for c in S2_CORE]
    for s in range(0, n, _CHUNK):
        blk = np.asarray(mm[s:s + _CHUNK], dtype=np.float32)[:, core_idx]
        for k, c in enumerate(S2_CORE):
            X[s:s + _CHUNK, col[c]] = blk[:, k]
    return X
