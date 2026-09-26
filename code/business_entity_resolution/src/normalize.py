"""Text canonicalisation for business names and addresses.

Design rules
  * Script-agnostic: NFKC + anyascii transliterate every script (Devanagari, Kannada,
    Telugu, Tamil, Bengali, Gujarati, accented Latin...) into ASCII, so a Kannada name
    and its Latin S1 counterpart land in the same character space.
  * Country is an OPEN set. Country-specific dictionaries (US / India / France) are
    *profiles*; any unseen label falls back to a generic profile. Nothing filters on
    country and no model feature encodes it.
  * Deterministic & pure: the same input always yields the same output, so train/test
    see identical transformations. Runs in a spawn-safe multiprocessing pool.
  * Consonant "skeletons" collapse vowel/aspiration/nasal variation - the dominant
    noise in transliteration (imdastris ~ industries) and in vowel typos.
"""
from __future__ import annotations

import multiprocessing as mp
import re
import unicodedata

import jellyfish
import numpy as np
import pandas as pd
from anyascii import anyascii

NAME_COLS = ["n_full", "n_core", "n_concat", "n_alt_a", "n_alt_b", "n_skel", "n_phon",
             "n_translit", "n_domain", "n_alias"]
ADDR_COLS = ["a_full", "a_core", "a_alpha", "a_skel", "a_nums", "a_first_num",
             "a_state", "a_translit", "a_empty", "a_ncomp"]
NORM_COLS = NAME_COLS + ADDR_COLS

# --------------------------------------------------------------------------- #
# Generic helpers                                                             #
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1+")
_VOWELS = re.compile(r"[aeiou]")
_NUM = re.compile(r"\d+")
_ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_DIGIT_ALPHA = re.compile(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NON_ALNUM_BAR = re.compile(r"[^a-z0-9|]+")
_PHONE = re.compile(r"\+?\d[\d\-\s().]{6,}\d")
_DOMAIN = re.compile(
    r"(?:https?://)?(?:www\.)?\b([a-z0-9][a-z0-9\-]*)\."
    r"(?:co\.in|com|net|org|biz|info|co|in|us|fr|io|online|shop|store|site|app)\b")
_ALIAS = re.compile(
    r"\b(?:f/?k/?a|a/?k/?a|d/?b/?a|formerly known as|formerly|trading as|"
    r"doing business as|nee)\b")
_LEAD = re.compile(r"^\W*(?:(?:m/s|messrs|mr|mrs|ms)\b\.?\s*)+")

NULLS = {"", "null", "none", "nan", "n/a", "na", "nil", "-", "--", "0", "unknown", "not available"}

_SKEL_TR = str.maketrans({"m": "n", "w": "v", "z": "s", "q": "k", "c": "k", "d": "t",
                          "y": "i", "x": "k"})
_DIGRAPHS = (("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"),
             ("th", "t"), ("dh", "d"), ("bh", "b"), ("zh", "s"))


def skel(tok: str) -> str:
    """Consonant skeleton: 'industries' -> 'intstrs', 'imdastris' -> 'intstrs'."""
    if not tok or tok.isdigit():
        return tok
    t = tok
    for a, b in _DIGRAPHS:
        if a in t:
            t = t.replace(a, b)
    t = t[0] + t[1:].replace("h", "")
    t = t.translate(_SKEL_TR)
    t = t[0] + _VOWELS.sub("", t[1:])
    return _REPEAT.sub(r"\1", t)


def phon(tok: str) -> str:
    """NYSIIS phonetic code: independent of the hand-built skeleton, catches cases it
    misses (transposition, different vowel/consonant confusions). 'industries' and its
    typo 'imdastris' both code to 'INDASTR'. Falls back to the token on any oddity
    (jellyfish requires at least one alphabetic character)."""
    if not tok or tok.isdigit():
        return tok
    try:
        return jellyfish.nysiis(tok)
    except (ValueError, IndexError):
        return tok


def _has_native(s: str) -> bool:
    """True if s contains letters beyond Latin Extended (i.e. needs transliteration)."""
    for ch in s:
        if ord(ch) > 0x24F and ch.isalpha():
            return True
    return False


def _ascii(s: str) -> str:
    return anyascii(unicodedata.normalize("NFKC", s)).lower()


def _join_initials(toks):
    """'l l c' -> 'llc', 's a r l' -> 'sarl'; single initials are kept."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


# --------------------------------------------------------------------------- #
# Names                                                                       #
# --------------------------------------------------------------------------- #
NAME_CANON = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "company": "co", "companies": "co",
    "and": "and", "et": "and", "und": "and",
    "international": "intl", "intl": "intl", "manufacturing": "mfg", "mfg": "mfg",
    "associates": "assoc", "associate": "assoc", "assoc": "assoc", "assocs": "assoc",
    "brothers": "bros", "bros": "bros", "centre": "center", "ctr": "center",
    "technologies": "tech", "technology": "tech", "technical": "tech", "tech": "tech",
    "group": "group", "grp": "group", "management": "mgmt", "mgmt": "mgmt",
    "national": "natl", "natl": "natl", "university": "univ", "univ": "univ",
    "institute": "inst", "inst": "inst", "hospital": "hosp", "hosp": "hosp",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc",
    "enterprise": "enterprises", "entp": "enterprises", "ent": "enterprises",
    "industries": "ind", "industry": "ind", "inds": "ind", "indus": "ind",
    "solutions": "soln", "solution": "soln", "systems": "sys", "system": "sys",
    "engineering": "engg", "engineers": "engg", "engg": "engg",
    "societe": "ste", "compagnie": "cie", "etablissements": "ets", "etablissement": "ets",
    "saint": "st", "sainte": "ste",
    "llc": "llc", "llp": "llp", "plc": "plc", "lp": "lp", "pc": "pc",
}
LEGAL = {
    "inc", "corp", "ltd", "pvt", "co", "llc", "lp", "llp", "pc", "plc", "pllc", "lllp",
    "public", "the", "opc", "huf", "pty", "gmbh", "ag", "bv", "nv", "srl", "spa",
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scs", "sca", "selarl", "scop",
    "cie", "ets", "ste",
}


def norm_name(raw: str):
    if not raw:
        return ("", "", "", "", "", "", "", False, False, False)
    translit = _has_native(raw)
    s = _ascii(raw)
    s = _LEAD.sub(" ", s)
    s = _PHONE.sub(" ", s)
    is_domain = False
    if "." in s:
        s2 = _DOMAIN.sub(r" \1 ", s)
        is_domain = s2 != s
        s = s2
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    s2 = _ALIAS.sub(" | ", s)
    has_alias = s2 != s
    s = _NON_ALNUM_BAR.sub(" ", s2)
    toks = _join_initials(s.split())
    toks = [NAME_CANON.get(t, t) for t in toks]
    while toks and toks[0] in ("mr", "mrs", "ms", "messrs", "|"):
        toks.pop(0)
    while toks and toks[-1] == "|":
        toks.pop()
    parts, cur = [], []
    for t in toks:
        if t == "|":
            if cur:
                parts.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        parts.append(cur)
    all_toks = [t for p in parts for t in p]
    core_toks = [t for t in all_toks if t not in LEGAL] or all_toks
    part_cores = [" ".join([t for t in p if t not in LEGAL] or p) for p in parts]
    core = " ".join(core_toks)
    alt_a = part_cores[0] if part_cores else core
    alt_b = part_cores[-1] if part_cores else core
    return (" ".join(all_toks), core, "".join(core_toks), alt_a, alt_b,
            " ".join(skel(t) for t in core_toks), " ".join(phon(t) for t in core_toks),
            translit, is_domain, has_alias)


# --------------------------------------------------------------------------- #
# Addresses                                                                   #
# --------------------------------------------------------------------------- #
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc", "puerto rico": "pr", "guam": "gu",
}
US_CODES = set(US_STATES.values())

IN_STATES = {
    "andhra pradesh": "andhrapradesh", "arunachal pradesh": "arunachalpradesh",
    "assam": "assam", "bihar": "bihar", "chhattisgarh": "chhattisgarh",
    "chattisgarh": "chhattisgarh", "goa": "goa", "gujarat": "gujarat", "haryana": "haryana",
    "himachal pradesh": "himachalpradesh", "jharkhand": "jharkhand", "karnataka": "karnataka",
    "kerala": "kerala", "keralam": "kerala", "madhya pradesh": "madhyapradesh",
    "maharashtra": "maharashtra", "manipur": "manipur", "meghalaya": "meghalaya",
    "mizoram": "mizoram", "nagaland": "nagaland", "odisha": "odisha", "orissa": "odisha",
    "punjab": "punjab", "rajasthan": "rajasthan", "sikkim": "sikkim",
    "tamil nadu": "tamilnadu", "tamilnadu": "tamilnadu", "telangana": "telangana",
    "tripura": "tripura", "uttar pradesh": "uttarpradesh", "uttarakhand": "uttarakhand",
    "uttaranchal": "uttarakhand", "west bengal": "westbengal", "delhi": "delhi",
    "nct of delhi": "delhi", "chandigarh": "chandigarh", "puducherry": "puducherry",
    "pondicherry": "puducherry", "jammu and kashmir": "jammukashmir", "ladakh": "ladakh",
    "lakshadweep": "lakshadweep", "andaman and nicobar islands": "andamannicobar",
    "dadra and nagar haveli": "dadranagarhaveli", "daman and diu": "damandiu",
}
IN_CODES = {
    "ap": "andhrapradesh", "ar": "arunachalpradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachalpradesh", "jh": "jharkhand", "jk": "jammukashmir",
    "ka": "karnataka", "kl": "kerala", "mp": "madhyapradesh", "mh": "maharashtra",
    "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha",
    "or": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamilnadu",
    "ts": "telangana", "tg": "telangana", "tr": "tripura", "up": "uttarpradesh",
    "uk": "uttarakhand", "ut": "uttarakhand", "wb": "westbengal", "dl": "delhi",
    "ch": "chandigarh", "py": "puducherry", "la": "ladakh",
}
IN_CITY = {
    "bangalore": "bengaluru", "bengaluru": "bengaluru", "bombay": "mumbai",
    "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram", "calicut": "kozhikode",
    "allahabad": "prayagraj", "poona": "pune", "mysore": "mysuru", "trivandrum":
    "thiruvananthapuram", "baroda": "vadodara", "cochin": "kochi", "mangalore": "mangaluru",
    "belgaum": "belagavi", "gulbarga": "kalaburagi", "benares": "varanasi",
    "banaras": "varanasi", "vizag": "visakhapatnam", "trichy": "tiruchirappalli",
    "tiruchirapalli": "tiruchirappalli", "tiruchirapally": "tiruchirappalli",
}
IN_STATE_SKEL = {}   # filled below: skeleton of canonical state -> canonical state

FR_REGIONS = {
    "ile de france": "iledefrance", "hauts de france": "hautsdefrance",
    "nouvelle aquitaine": "nouvelleaquitaine", "auvergne rhone alpes": "auvergnerhonealpes",
    "provence alpes cote d azur": "paca", "provence alpes cote dazur": "paca",
    "grand est": "grandest", "pays de la loire": "paysdelaloire", "bretagne": "bretagne",
    "normandie": "normandie", "occitanie": "occitanie", "centre val de loire":
    "centrevaldeloire", "bourgogne franche comte": "bourgognefranchecomte", "corse": "corse",
}

US_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "aven": "avenue", "blvd": "boulevard", "boul": "boulevard",
    "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court", "crt": "court",
    "cir": "circle", "circ": "circle", "pl": "place", "pkwy": "parkway", "pky": "parkway",
    "hwy": "highway", "trl": "trail", "ter": "terrace", "terr": "terrace", "sq": "square",
    "mt": "mount", "ft": "fort", "ctr": "center", "cntr": "center", "cv": "cove",
    "xing": "crossing", "expy": "expressway", "fwy": "freeway", "jct": "junction",
    "pt": "point", "rte": "route", "rt": "route", "tpke": "turnpike", "aly": "alley",
    "bnd": "bend", "brg": "bridge", "byp": "bypass", "cswy": "causeway", "holw": "hollow",
    "hts": "heights", "lk": "lake", "mdw": "meadow", "mnr": "manor", "plz": "plaza",
    "rdg": "ridge", "spg": "spring", "sta": "station", "vly": "valley", "vw": "view",
    "vlg": "village", "wy": "way", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "twp": "township", "cdp": "", "ext": "extension", "hl": "hill", "is": "island",
}
IN_ABBR = {
    "rd": "road", "st": "street", "ngr": "nagar", "clny": "colony", "col": "colony",
    "sec": "sector", "sect": "sector", "ph": "phase", "blk": "block", "dist": "district",
    "distt": "district", "dt": "district", "nr": "near", "opp": "opposite", "bldg": "building",
    "apts": "apartments", "appt": "apartment", "soc": "society", "chs": "society",
    "indl": "industrial", "estt": "estate", "mkt": "market", "vill": "village",
    "vil": "village", "vpo": "village", "ext": "extension", "extn": "extension",
    "mg": "mahatmagandhi", "cross": "cross", "main": "main", "w": "west", "e": "east",
    "tq": "", "tal": "", "taluk": "", "taluka": "", "tehsil": "", "teh": "", "mandal": "",
    "po": "", "ps": "", "via": "",
}
FR_ABBR = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
    "boul": "boulevard", "bvd": "boulevard", "pl": "place", "ch": "chemin", "che": "chemin",
    "chem": "chemin", "imp": "impasse", "all": "allee", "rte": "route", "fbg": "faubourg",
    "faub": "faubourg", "sq": "square", "qu": "quai", "crs": "cours", "st": "saint",
    "ste": "sainte", "pte": "porte", "prom": "promenade", "res": "residence",
    "lot": "lotissement", "za": "zone", "zi": "zone", "zac": "zone", "cedex": "",
    "bp": "", "cs": "", "sur": "sur",
}
GEN_ABBR = {"rd": "road", "st": "street", "ave": "avenue", "av": "avenue",
            "blvd": "boulevard", "dr": "drive", "ln": "lane", "bd": "boulevard"}

UNIT_WORDS = {
    "unit", "apt", "apartment", "ste", "suite", "fl", "floor", "flr", "pmb", "box", "bx",
    "no", "nos", "number", "num", "hno", "h", "house", "door", "dno", "plot", "plt", "sy",
    "survey", "khasra", "kh", "gat", "cts", "flat", "shop", "room", "rm", "dept", "po",
    "pobox", "p", "o", "ground", "first", "second", "third", "gf", "ff", "sf", "tf",
    "appartement", "batiment", "bat", "escalier", "esc", "etage", "bis", "ter",
}
FR_UNIT = (UNIT_WORDS - {"ste", "ter"}) | {"cedex"}
LANDMARK = {"near", "nr", "opp", "opposite", "behind", "beside", "besides", "next",
            "adjacent", "adj", "infront", "front", "pres", "face"}
ADDR_STOP = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "court", "circle", "place",
    "parkway", "highway", "trail", "terrace", "square", "way", "north", "south", "east",
    "west", "northeast", "northwest", "southeast", "southwest", "near", "opposite", "the",
    "and", "of", "main", "cross", "nagar", "colony", "sector", "phase", "block", "district",
    "road", "marg", "rue", "chemin", "allee", "impasse", "des", "du", "de", "la", "le",
    "les", "sur", "saint", "sainte", "city", "town", "village", "county", "township",
    "building", "complex", "floor", "apartment", "apartments", "society", "india", "usa",
    "france", "behind", "beside", "west", "east",
}


def _profile(country_key: str) -> str:
    c = country_key.strip().lower()
    if c in ("us", "usa", "united states", "united states of america", "u s", "u s a"):
        return "US"
    if c in ("india", "in", "bharat", "ind"):
        return "IN"
    if c in ("france", "fr", "fra", "republique francaise"):
        return "FR"
    return "GEN"


def _multiword_regex(d: dict):
    keys = sorted((k for k in d if " " in k), key=len, reverse=True)
    if not keys:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


_US_MW = _multiword_regex(US_STATES)
_IN_MW = _multiword_regex(IN_STATES)
_FR_MW = _multiword_regex(FR_REGIONS)
for _k, _v in list(IN_STATES.items()):
    IN_STATE_SKEL[skel(_v)] = _v
IN_CANON = set(IN_STATES.values())
FR_CANON = set(FR_REGIONS.values())

PROFILES = {
    "US": dict(abbr=US_ABBR, unit=UNIT_WORDS, mw=_US_MW,
               mw_map={k: v for k, v in US_STATES.items()}),
    "IN": dict(abbr=IN_ABBR, unit=UNIT_WORDS, mw=_IN_MW, mw_map=IN_STATES),
    "FR": dict(abbr=FR_ABBR, unit=FR_UNIT, mw=_FR_MW, mw_map=FR_REGIONS),
    "GEN": dict(abbr=GEN_ABBR, unit=UNIT_WORDS, mw=None, mw_map={}),
}


def _state_of(comp_joined: str, comp_tokens, prof: str, native: bool) -> str:
    if prof == "US":
        if comp_joined in US_CODES:
            return comp_joined
        return US_STATES.get(" ".join(comp_tokens), "")
    if prof == "IN":
        if comp_joined in IN_CODES:
            return IN_CODES[comp_joined]
        v = IN_STATES.get(" ".join(comp_tokens)) or (comp_joined if comp_joined in
                                                     IN_CANON else "")
        if not v and native:
            v = IN_STATE_SKEL.get(skel(comp_joined), "")
        return v
    if prof == "FR":
        if comp_joined in FR_CANON:
            return comp_joined
        return ""
    return ""


_EMPTY_ADDR = ("", "", "", "", "", "", "", False, True, 0)


def norm_addr(raw: str, country: str):
    if not raw or raw.strip().lower() in NULLS:
        return _EMPTY_ADDR
    prof = _profile(country)
    P = PROFILES[prof]
    abbr, unit, mw, mw_map = P["abbr"], P["unit"], P["mw"], P["mw_map"]
    translit_any = False
    all_toks, core_toks, nums = [], [], []
    state = ""
    ncomp = 0
    for comp_raw in raw.split(","):
        native = _has_native(comp_raw)
        c = _ascii(comp_raw).strip(" .-/#")
        if c in NULLS:
            continue
        translit_any |= native
        ncomp += 1
        c = _ORD.sub(r"\1", c)
        nums.extend(_NUM.findall(c))
        c = c.replace("&", " and ")
        c = _DIGIT_ALPHA.sub(" ", c)
        c = _NON_ALNUM.sub(" ", c).strip()
        if not c:
            continue
        if mw is not None:
            c = mw.sub(lambda m: mw_map[m.group(1)].replace(" ", ""), c)
        toks = c.split()
        if prof == "IN":
            toks = [IN_CITY.get(t, t) for t in toks]
        if prof in ("US", "GEN") and len(toks) == 1 and toks[0] in US_STATES:
            toks = [US_STATES[toks[0]]]
        joined = "".join(toks)
        st = _state_of(joined, toks, prof, native)
        if st:
            state = state or st
            all_toks.append(st)
            core_toks.append(st)
            continue
        out = []
        for t in toks:
            t = abbr.get(t, t)
            if not t or t in unit:
                continue
            out.append(t)
        if not out:
            continue
        all_toks.extend(out)
        if out[0] not in LANDMARK:
            core_toks.extend(out)
    if not all_toks and not nums:
        return _EMPTY_ADDR
    nums = list(dict.fromkeys(str(int(n)) for n in nums if len(n) <= 12))[:6]
    alpha = [t for t in all_toks if len(t) >= 3 and t.isalpha() and t not in ADDR_STOP
             and t != state]
    alpha = list(dict.fromkeys(alpha))
    return (" ".join(all_toks), " ".join(core_toks), " ".join(alpha),
            " ".join(skel(t) for t in alpha), " ".join(nums), nums[0] if nums else "",
            state, translit_any, False, ncomp)


# --------------------------------------------------------------------------- #
# Parallel driver                                                             #
# --------------------------------------------------------------------------- #
def _norm_chunk(args):
    names, addrs, countries = args
    cols = [[] for _ in NORM_COLS]
    for n, a, c in zip(names, addrs, countries):
        row = norm_name(n) + norm_addr(a, c)
        for j, v in enumerate(row):
            cols[j].append(v)
    return cols


_STR_OUT = [c for c in NORM_COLS if c not in ("n_translit", "n_domain", "n_alias",
                                               "a_translit", "a_empty", "a_ncomp")]


def _to_frame(cols) -> pd.DataFrame:
    """Chunk result -> compact frame (Arrow-backed strings: ~3-5x less RAM than objects)."""
    out = pd.DataFrame({c: cols[j] for j, c in enumerate(NORM_COLS)})
    for c in _STR_OUT:
        out[c] = out[c].astype("string[pyarrow]")
    for c in ("n_translit", "n_domain", "n_alias", "a_translit", "a_empty"):
        out[c] = out[c].astype(bool)
    out["a_ncomp"] = out["a_ncomp"].astype(np.int16)
    return out


def normalize_frame(df: pd.DataFrame, workers: int, chunk: int = 25_000) -> pd.DataFrame:
    """Normalise names/addresses in a spawn-safe process pool; order preserved.

    Each finished chunk is converted to Arrow strings immediately, so peak RAM stays
    ~ (workers x chunk) Python objects instead of (rows x columns).
    """
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    ctrs = df["country"].tolist()
    tasks = [(names[i:i + chunk], addrs[i:i + chunk], ctrs[i:i + chunk])
             for i in range(0, len(names), chunk)]
    frames = []
    if workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            frames.append(_to_frame(_norm_chunk(t)))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for part in pool.imap(_norm_chunk, tasks, chunksize=1):
                frames.append(_to_frame(part))
    if not frames:
        return _to_frame([[] for _ in NORM_COLS])
    return pd.concat(frames, ignore_index=True)
