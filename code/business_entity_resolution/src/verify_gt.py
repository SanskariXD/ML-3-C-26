#!/usr/bin/env python3
"""CHECK 1 -- verify the claimed generator transformations against ground truth.

Every pattern we assert in research.md gets a detector and a measured rate here. A
claim that does not show up in the data gets deleted from the docs; this file is the
evidence, not the theory.

Also verifies two *structural* properties we plan to exploit, because both are
load-bearing and neither was proven:
  * one-owner      : does each S2/S3 record belong to exactly one S1 entity?
  * count caps     : is n_S2 <= 5 and n_S3 <= 6 really never violated?

Usage:
    python3 verify_gt.py --sample work/sample.json            # quantified rates
    python3 verify_gt.py --sample work/sample.json --show 20  # + manual side-by-side
    python3 verify_gt.py --one-owner student_resource/dataset/train/train_ground_truth.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter

LEGAL = {"inc", "llc", "ltd", "limited", "private", "pvt", "corp", "corporation", "co",
         "company", "incorporated", "plc", "llp", "sarl", "sas", "sasu", "eurl", "sci",
         "sa", "cie", "freres", "fils", "pc", "pllc"}
ADDED = {"service", "services", "center", "centre", "enterprises", "group", "holdings",
         "solutions", "industries", "trading", "associates", "partners"}
STREET_ABBR = {"rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "trl": "trail",
               "cv": "cove", "dr": "drive", "ln": "lane", "blvd": "boulevard", "bd": "boulevard",
               "hwy": "highway", "ct": "court", "pl": "place", "sq": "square", "pkwy": "parkway",
               "ter": "terrace", "cir": "circle", "twp": "township", "r": "rue"}


def strip_acc(s: str) -> str:
    n = unicodedata.normalize("NFKD", s)
    return "".join(c for c in n if not unicodedata.combining(c))


def has_accent(s: str) -> bool:
    n = unicodedata.normalize("NFKD", s)
    return any(unicodedata.combining(c) for c in n)


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", strip_acc(s).lower())


def core(s: str) -> list[str]:
    return [t for t in toks(s) if t not in LEGAL]


def lev(a: str, b: str, cap: int = 3) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def trigrams(s: str) -> set:
    s = re.sub(r"[^a-z0-9]", "", strip_acc(s).lower())
    return {s[i:i + 3] for i in range(max(0, len(s) - 2))} or {s}


def jac(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


# --------------------------------------------------------------- name detectors
def detect_name(n1: str, n2: str) -> set:
    out = set()
    t1, t2 = toks(n1), toks(n2)
    c1, c2 = core(n1), core(n2)
    s1, s2 = set(t1), set(t2)

    if not re.search(r"[a-z]", strip_acc(n2).lower()):
        out.add("non_latin_name")            # Devanagari transliteration
    if has_accent(n2) and not has_accent(n1):
        out.add("diacritic_injected")
    if re.search(r"\.(com|net|in|co\.in|fr)\b|^www|^#", n2.strip().lower()):
        out.add("domain_or_handle_name")
    if re.search(r"[\[\]{}<>|]|^\s*(--|\.\.\.)", n2):
        out.add("bracket_or_junk_noise")

    if s1 and s2:
        if not (s1 & s2):
            out.add("name_zero_token_overlap")
            if jac(trigrams(n1), trigrams(n2)) < 0.20:
                out.add("name_RANDOM_replacement")
        # legal suffix dropped / added
        if set(c1) == set(c2) and s1 != s2:
            l1 = {t for t in t1 if t in LEGAL}
            l2 = {t for t in t2 if t in LEGAL}
            if l1 - l2:
                out.add("legal_suffix_dropped")
            if l2 - l1:
                out.add("legal_suffix_added")
        # appended descriptor word
        extra = s2 - s1
        if extra and s1 <= s2:
            out.add("token_appended")
            if extra & ADDED:
                out.add("token_appended_descriptor")
        # transposition: same multiset, different order
        if sorted(t1) == sorted(t2) and t1 != t2:
            out.add("word_transposition")
        # truncation
        if s2 < s1 and s2:
            out.add("name_truncated")
        # typo: one token replaced by a near-miss
        if len(t1) == len(t2) and s1 != s2:
            d = [(x, y) for x, y in zip(sorted(t1), sorted(t2)) if x != y]
            if len(d) == 1 and 0 < lev(d[0][0], d[0][1]) <= 2:
                out.add("typo_in_name")
    return out


# ------------------------------------------------------------ address detectors
def detect_addr(a1: str, a2: str) -> set:
    out = set()
    if not a2.strip():
        out.add("address_EMPTY")
        return out
    if "<NULL>" in a2:
        out.add("null_literal_token")
    letters = [c for c in a2 if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        out.add("address_UPPERCASED")
    if has_accent(a2) and not has_accent(a1):
        out.add("addr_diacritic_injected")
    if re.search(r"(^|[,\s])#|\bPO BOX\b|\bP\.O\b", a2, re.I):
        out.add("hash_or_pobox_prefix")

    t1, t2 = toks(a1), toks(a2)
    n1 = [t for t in t1 if t.isdigit()]
    n2 = [t for t in t2 if t.isdigit()]
    w1 = [t for t in t1 if not t.isdigit()]
    w2 = [t for t in t2 if not t.isdigit()]

    # street-type abbreviation in either direction
    e1 = {STREET_ABBR.get(t, t) for t in w1}
    e2 = {STREET_ABBR.get(t, t) for t in w2}
    if set(w1) != set(w2) and e1 & e2 and (
            {t for t in w1 if t in STREET_ABBR} or {t for t in w2 if t in STREET_ABBR}):
        out.add("street_type_abbreviated")

    # component reordering: same word multiset, different order
    if sorted(w1) == sorted(w2) and w1 != w2:
        out.add("component_reordered")
    elif set(w1) & set(w2) and w1 and w2:
        # positional drift of the shared tokens = reorder-ish
        sh = [t for t in w1 if t in set(w2)]
        if sh and [t for t in w2 if t in set(w1)] != sh:
            out.add("component_reordered_partial")

    if n1 and n2:
        if set(n1) != set(n2):
            out.add("number_perturbed" if set(n1) & set(n2) else "number_fully_changed")
    if n1 and not n2:
        out.add("number_dropped")
    if set(w2) < set(w1):
        out.add("address_component_dropped")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="work/sample.json")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--one-owner")
    a = ap.parse_args()

    if a.one_owner:
        owner, dup = {}, Counter()
        caps = Counter()
        n = 0
        with open(a.one_owner, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 2 or not p[1]:
                    continue
                n += 1
                ids = [x for x in p[1].split(",") if x]
                s2 = sum(1 for x in ids if x.startswith("S2-"))
                if s2 > 5 or len(ids) - s2 > 6:
                    caps["violation"] += 1
                for x in ids:
                    if x in owner:
                        dup[x] += 1
                    else:
                        owner[x] = p[0]
        print(f"entities with >=1 match : {n:,}")
        print(f"distinct S2/S3 matched  : {len(owner):,}")
        print(f"records claimed by >1 S1: {len(dup):,}   <-- ONE-OWNER "
              f"{'HOLDS' if not dup else 'VIOLATED'}")
        print(f"cap violations (S2>5|S3>6): {caps['violation']:,}   <-- CAPS "
              f"{'HOLD' if not caps['violation'] else 'VIOLATED'}")
        return

    d = json.load(open(a.sample))
    groups, rec = d["groups"], d["rec"]
    cnt = Counter()
    tot = Counter()
    examples: dict[str, tuple] = {}

    for s1, ms in groups:
        if s1 not in rec:
            continue
        n1, a1, _ = rec[s1]
        for m in ms:
            if m not in rec:
                continue
            n2, a2, _ = rec[m]
            src = m[:2]
            tot[src] += 1
            tot["ALL"] += 1
            for tag in detect_name(n1, n2) | detect_addr(a1, a2):
                cnt[f"{tag}|{src}"] += 1
                cnt[f"{tag}|ALL"] += 1
                examples.setdefault(tag, (s1, m, n1, a1, n2, a2))

    tags = sorted({k.split("|")[0] for k in cnt}, key=lambda t: -cnt[f"{t}|ALL"])
    print(f"CHECK 1 -- transformation rates over {tot['ALL']:,} true pairs "
          f"(S2={tot['S2']:,}  S3={tot['S3']:,})\n")
    print(f"{'transformation':32s} {'ALL':>8s} {'S2':>8s} {'S3':>8s}")
    print("-" * 60)
    for t in tags:
        print(f"{t:32s} {cnt[f'{t}|ALL']/tot['ALL']:7.2%} "
              f"{cnt[f'{t}|S2']/tot['S2']:7.2%} {cnt[f'{t}|S3']/tot['S3']:7.2%}")

    if a.show:
        print("\n" + "=" * 78 + "\nONE VERIFIED EXAMPLE PER PATTERN\n" + "=" * 78)
        for t in tags[:a.show]:
            s1, m, n1, a1, n2, a2 = examples[t]
            print(f"\n### {t}   ({s1} -> {m})")
            print(f"  S1 name: {n1!r}\n  {m[:2]} name: {n2!r}")
            print(f"  S1 addr: {a1!r}\n  {m[:2]} addr: {a2!r}")


if __name__ == "__main__":
    main()
