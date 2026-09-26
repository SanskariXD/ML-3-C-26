#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data. Proves the install and the wiring.

Generates a small dataset that mimics the real generator's transformations (street
abbreviation, uppercasing, legal-suffix drop, transposition, typos, reordering,
transliteration-like script swap, empty addresses, distractors), runs the full
pipeline, and asserts the outputs are structurally valid.

    python tests/smoke_test.py          # must print SMOKE TEST PASS

Takes ~1-2 min. Does NOT check score quality -- only that every stage runs and the
submission contract holds.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WORDS = ["apex", "silver", "cornerstone", "quality", "empire", "choice", "rising",
         "dynamic", "prime", "vision", "summit", "harbor", "vertex", "quantum",
         "pioneer", "beacon", "cobalt", "meridian", "atlas", "zenith"]
KIND = ["media", "logistics", "insurance", "nutrients", "developers", "massage",
        "wholesale", "partners", "consultants", "holdings"]
LEGAL = ["Inc", "LLC", "Private Limited", "Pvt Ltd", "Corp", "Co"]
STREETS = [("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave"), ("Trail", "Trl"),
           ("Cove", "Cv"), ("Drive", "Dr"), ("Lane", "Ln"), ("Boulevard", "Blvd")]
CITIES = [("Bangalore", "Bengaluru", "KA"), ("Pune", "Poona", "MH"),
          ("Mumbai", "Mumbai City", "MH"), ("Tyler", "Tyler Township", "TX"),
          ("Salem", "Salem City", "VA"), ("Bronx", "Bronx", "NY")]
DEVA = str.maketrans({"a": "\u093e", "e": "\u0947", "i": "\u093f", "o": "\u094b", "u": "\u0941"})


def typo(s: str, rng: random.Random) -> str:
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def gen(rng: random.Random, n_ent: int):
    """Return (s1, s2, s3, gt) as lists of tuples."""
    s1, s2, s3, gt = [], [], [], []
    uid = [0]

    def nid(p):
        uid[0] += 1
        return f"{p}-{uid[0]:07d}"

    for _ in range(n_ent):
        country = rng.choice(["US", "India"])
        base = f"{rng.choice(WORDS).title()} {rng.choice(KIND).title()}"
        legal = rng.choice(LEGAL)
        name = f"{base} {legal}"
        num = rng.randrange(10, 9999)
        street, abbr = rng.choice(STREETS)
        city, city_alt, state = rng.choice(CITIES)
        addr = f"{num} {street.title()} {street}, {city}, {state}"
        addr = f"{num} Maple {street}, {city}, {state}"
        e1 = nid("S1")
        s1.append((e1, name, addr, country))

        matches = []
        # ---- Source 2: uppercase address, abbreviations, name mutations
        for _ in range(rng.randint(0, 3)):
            n = name
            r = rng.random()
            if r < 0.2:
                n = base                                    # legal suffix dropped
            elif r < 0.35:
                n = " ".join(reversed(base.split())) + f" {legal}"   # transposition
            elif r < 0.5:
                n = typo(base, rng) + f" {legal}"           # typo
            elif r < 0.6:
                n = base.lower().replace(" ", "") + ".com"  # domain form
            elif r < 0.7:
                n = base.translate(DEVA)                    # script swap
            a = f"{num} MAPLE {abbr.upper()}, {city.upper()}, {state}"
            if rng.random() < 0.10:
                a = ""                                      # empty address
            e = nid("S2")
            s2.append((e, n, a, country))
            matches.append(e)
        # ---- Source 3: reordered components, city alias, expanded state
        for _ in range(rng.randint(0, 3)):
            n = name
            r = rng.random()
            if r < 0.2:
                n = base.split()[0]                         # truncation
            elif r < 0.35:
                n = f"{base} Services"                      # appended token
            elif r < 0.5:
                n = typo(base, rng)
            a = f"{state}, {city_alt}, {num} Maple {abbr}"   # reordered + alias
            if rng.random() < 0.10:
                a = ""
            e = nid("S3")
            s3.append((e, n, a, country))
            matches.append(e)
        gt.append((e1, matches))

    # ---- distractors: records owned by nobody (~25%, matching the real data)
    for _ in range(int(0.25 * (len(s2) + len(s3)))):
        country = rng.choice(["US", "India"])
        nm = f"{rng.choice(WORDS).title()} {rng.choice(KIND).title()} {rng.choice(LEGAL)}"
        city, _, state = rng.choice(CITIES)
        ad = f"{rng.randrange(10, 9999)} Oak {rng.choice(STREETS)[0]}, {city}, {state}"
        (s2 if rng.random() < 0.5 else s3).append(
            (nid("S2" if rng.random() < 0.5 else "S3"), nm, ad, country))
    # keep prefixes consistent with the file they live in
    s2 = [("S2-" + e.split("-")[1], n, a, c) for e, n, a, c in s2]
    s3 = [("S3-" + e.split("-")[1], n, a, c) for e, n, a, c in s3]
    return s1, s2, s3, gt


def write_tsv(path: str, rows, header):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


def main() -> int:
    rng = random.Random(7)
    tmp = tempfile.mkdtemp(prefix="ber_smoke_")
    try:
        ds = os.path.join(tmp, "dataset")
        for split in ("train", "test"):
            os.makedirs(os.path.join(ds, split), exist_ok=True)
            s1, s2, s3, gt = gen(rng, 900 if split == "train" else 300)
            hdr = ["entity_id", "business_name", "business_address", "country"]
            write_tsv(os.path.join(ds, split, f"{split}_source1.tsv"), s1, hdr)
            write_tsv(os.path.join(ds, split, f"{split}_source2.tsv"), s2, hdr)
            write_tsv(os.path.join(ds, split, f"{split}_source3.tsv"), s3, hdr)
            if split == "train":
                write_tsv(os.path.join(ds, split, "train_ground_truth.tsv"),
                          [(a, ",".join(b)) for a, b in gt],
                          ["source1_entity_id", "matched_entity_ids"])

        out = os.path.join(tmp, "out")
        cmd = [sys.executable, os.path.join(ROOT, "src", "run.py"), "all",
               "--data-dir", ds, "--work-dir", os.path.join(tmp, "work"),
               "--out-dir", out, "--no-dense", "--bm25-k", "0",
               "--k-dense-script", "0", "--bm25-k-empty", "0"]
        print("running:", " ".join(cmd[:3]), "...")
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-4000:]); print(p.stderr[-4000:])
            print("SMOKE TEST FAIL: pipeline exited", p.returncode)
            return 1

        # ---- assert the submission contract
        mr = os.path.join(out, "matching_results.tsv")
        cp = os.path.join(out, "candidate_pairs.tsv")
        for f in (mr, cp):
            if not os.path.isfile(f):
                print(f"SMOKE TEST FAIL: missing {f}")
                return 1
        test_s1 = {r.split("\t")[0] for r in
                   open(os.path.join(ds, "test", "test_source1.tsv"),
                        encoding="utf-8").read().splitlines()[1:] if r}
        valid = set()
        for n in (2, 3):
            valid |= {r.split("\t")[0] for r in
                      open(os.path.join(ds, "test", f"test_source{n}.tsv"),
                           encoding="utf-8").read().splitlines()[1:] if r}

        def check(path, label):
            seen, n_ids = set(), 0
            lines = open(path, encoding="utf-8").read().splitlines()
            for ln in lines[1:]:
                if not ln:
                    continue
                parts = ln.split("\t")
                eid = parts[0]
                ids = [x for x in (parts[1] if len(parts) > 1 else "").split(",") if x]
                assert eid not in seen, f"{label}: duplicate row {eid}"
                seen.add(eid)
                assert len(ids) == len(set(ids)), f"{label}: duplicate id in {eid}"
                for i in ids:
                    assert i in valid, f"{label}: unknown id {i}"
                n_ids += len(ids)
            assert seen == test_s1, (f"{label}: row set != test source1 "
                                    f"(missing {len(test_s1 - seen)}, extra {len(seen - test_s1)})")
            return len(seen), n_ids

        nm, im = check(mr, "matching_results")
        nc, ic = check(cp, "candidate_pairs")
        print(f"  matching_results: {nm} rows, {im} ids ({im/max(1,nm):.2f}/entity)")
        print(f"  candidate_pairs : {nc} rows, {ic} ids ({ic/max(1,nc):.2f}/entity)")
        print("SMOKE TEST PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
