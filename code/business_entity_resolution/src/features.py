"""Pairwise features for (S1, target) candidate pairs."""
import polars as pl, numpy as np, math, re
from rapidfuzz import process, fuzz, distance

IND_RE = r"[ऀ-෿]"
WORKERS = 1


def rec_table(split, s):
    p = pl.read_parquet(f"work/p_{split}_s{s}.parquet")
    raw = pl.read_parquet(f"work/{split}_s{s}.parquet", columns=["entity_id", "business_name", "business_address"])
    p = p.with_columns(
        pl.col("nt").list.join(" ").alias("ns"),
        pl.when(pl.col("core").list.len() > 0).then(pl.col("core")).otherwise(pl.col("nt")).alias("core"),
    ).with_columns(
        pl.col("core").list.join(" ").alias("cs"),
        pl.col("core").list.join("").alias("cc"),
        pl.col("nt").list.join("").alias("nc"),
        pl.col("at").list.join(" ").alias("as"),
        pl.col("acore").list.join(" ").alias("acs"),
        pl.col("acore").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).alias("nums"),
    )
    p = p.join(raw.rename({"entity_id": "id"}), on="id", how="left")
    p = p.with_columns((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id")).with_columns(
        pl.col("business_name").str.contains(IND_RE).alias("ind"),
        (pl.col("business_address").str.strip_chars() == "").alias("aempty"),
        pl.col("business_name").str.contains(r"(?i)\.com|www\.|^@").alias("dom"),
        pl.col("business_name").str.contains(r"[A-Z]{4,}").alias("upper"),
    )
    return p


COLS = ["id", "country", "nt", "core", "ns", "cs", "cc", "nc", "at", "acore", "as", "acs", "nums", "ind", "aempty",
        "dom", "al_pre", "al_post", "oov", "ncomp"]


def idf_tables(tgt):
    N = len(tgt)
    n = tgt.select(pl.col("nt").list.unique().alias("t")).explode("t").group_by("t").len("df")
    a = tgt.select(pl.col("at").list.unique().alias("t")).explode("t").group_by("t").len("df")
    nd = dict(zip(n["t"].to_list(), (np.log(N / n["df"].to_numpy())).tolist()))
    ad = dict(zip(a["t"].to_list(), (np.log(N / a["df"].to_numpy())).tolist()))
    return nd, ad, math.log(N)


def _setfeats(A, B, idf, dflt):
    A = set(A); B = set(B)
    inter = A & B
    wa = sum(idf.get(t, dflt) for t in A)
    wb = sum(idf.get(t, dflt) for t in B)
    wi = sum(idf.get(t, dflt) for t in inter)
    # fuzzy alignment of leftovers (typos): count tokens with a close partner
    ra = A - inter; rb = B - inter
    fa = 0.0; fb = 0.0; nfa = 0
    if ra and rb:
        for x in ra:
            best = 0.0
            for y in rb:
                s = distance.JaroWinkler.similarity(x, y)
                if s > best:
                    best = s
            if best >= 0.88:
                fa += idf.get(x, dflt); nfa += 1
    ua = wa - wi - fa  # unexplained weight of A
    ub_list = [idf.get(t, dflt) for t in rb]
    ub = sum(ub_list) - fa  # approx
    jac = len(inter) / max(1, len(A | B))
    return (jac, len(inter), len(A), len(B), wi / max(wa, 1e-9), wi / max(wb, 1e-9), max(ua, 0), max(ub, 0), nfa,
            max(ub_list) if ub_list else 0.0)


def _hn(nums):
    return nums[0] if nums else None


def _hnrel(x, y):
    """relation code between S1 house number x and target house number y."""
    if x == y:
        return 0
    if len(y) < len(x) and x.endswith(y):
        return 1  # target dropped leading digit(s)
    if len(y) < len(x) and x.startswith(y):
        return 2  # target dropped trailing digit(s)
    if len(y) > len(x) and y.endswith(x):
        return 3
    if len(y) > len(x) and y.startswith(x):
        return 4
    d = abs(int(x[:9]) - int(y[:9]))
    if len(x) == len(y) and sum(c1 != c2 for c1, c2 in zip(x, y)) == 1:
        return 5 if d < 10 else 6  # one digit changed (last digit / higher digit)
    if d <= 20:
        return 7
    return 8


def _numfeats2(a, b):
    if not a or not b:
        return (-1, -1, -1, -1, -1)
    sa, sb = set(a), set(b)
    return (_hnrel(a[0], b[0]), len(sb - sa), len(sa - sb), int(a[-1] == b[-1]), int(b[0] in sa))


def _shortun(A, B):
    """unmatched short tokens (<=3 chars) on each side, and unmatched target tokens with no close partner."""
    A = set(A); B = set(B)
    ra = A - B; rb = B - A
    sa = sum(1 for x in ra if len(x) <= 3)
    sb = sum(1 for x in rb if len(x) <= 3)
    nclose = 0
    for y in rb:
        for x in ra:
            if distance.Levenshtein.distance(x, y) <= 1:
                nclose += 1; break
    return (sa, sb, nclose)


def _numfeats(a, b):
    """a: S1 numeric tokens, b: target numeric tokens."""
    if not a or not b:
        return (-1, -1, -1, -1, -1)
    ha, hb = a[0], b[0]
    eq = int(ha == hb)
    anyeq = int(ha in b)
    sub = int(ha in hb or hb in ha)
    try:
        d = abs(int(ha[:9]) - int(hb[:9]))
        ld = math.log1p(d)
    except Exception:
        ld = -1
    sa, sb = set(a), set(b)
    jac = len(sa & sb) / len(sa | sb)
    return (eq, anyeq, sub, ld, jac)


def compute(pairs, R1, RT, nidf, aidf, dflt):
    """pairs: DataFrame with s1, t (+ block cols). R1/RT: record tables."""
    L = pairs.join(R1.select(COLS).rename({c: c + "_1" for c in COLS if c != "id"}), left_on="s1", right_on="id", how="left")
    L = L.join(RT.select(COLS).rename({c: c + "_2" for c in COLS if c != "id"}), left_on="t", right_on="id", how="left")
    out = {}
    # vectorized string similarities
    for f in ["ns", "cs", "cc", "as", "acs"]:
        a = L[f + "_1"].fill_null("").to_list(); b = L[f + "_2"].fill_null("").to_list()
        out[f + "_ratio"] = process.cpdist(a, b, scorer=fuzz.ratio, workers=WORKERS).astype(np.float32)
        if f in ("ns", "cs", "as", "acs"):
            out[f + "_tset"] = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=WORKERS).astype(np.float32)
            out[f + "_tsort"] = process.cpdist(a, b, scorer=fuzz.token_sort_ratio, workers=WORKERS).astype(np.float32)
            out[f + "_part"] = process.cpdist(a, b, scorer=fuzz.partial_ratio, workers=WORKERS).astype(np.float32)
        if f in ("cs", "cc"):
            out[f + "_jw"] = process.cpdist(a, b, scorer=distance.JaroWinkler.normalized_similarity, workers=WORKERS).astype(np.float32)
    # concatenated-name containment (domains, run-together names)
    cc1 = L["cc_1"].fill_null("").to_list(); nc2 = L["nc_2"].fill_null("").to_list(); cc2 = L["cc_2"].fill_null("").to_list()
    out["cc_in_t"] = np.array([int(len(x) > 3 and x in y) for x, y in zip(cc1, nc2)], np.int8)
    out["cc_eq"] = np.array([int(x == y) for x, y in zip(cc1, cc2)], np.int8)
    # token set features
    nt1 = L["nt_1"].to_list(); nt2 = L["nt_2"].to_list()
    co1 = L["core_1"].to_list(); co2 = L["core_2"].to_list()
    at1 = L["acore_1"].to_list(); at2 = L["acore_2"].to_list()
    nm1 = L["nums_1"].to_list(); nm2 = L["nums_2"].to_list()
    names = ["jac", "ninter", "nA", "nB", "covA", "covB", "unA", "unB", "nfuzzy", "maxunB"]
    F = np.array([_setfeats(a or [], b or [], nidf, dflt) for a, b in zip(co1, co2)], np.float32)
    for i, n in enumerate(names):
        out["core_" + n] = F[:, i]
    F = np.array([_setfeats(a or [], b or [], nidf, dflt) for a, b in zip(nt1, nt2)], np.float32)
    for i, n in enumerate(["jac", "covA", "covB", "unA", "unB"]):
        out["nt_" + n] = F[:, names.index(n)]
    F = np.array([_setfeats([x for x in (a or []) if not x.isdigit()], [x for x in (b or []) if not x.isdigit()],
                            aidf, dflt) for a, b in zip(at1, at2)], np.float32)
    for i, n in enumerate(names):
        out["addr_" + n] = F[:, i]
    F = np.array([_numfeats(a or [], b or []) for a, b in zip(nm1, nm2)], np.float32)
    for i, n in enumerate(["hn_eq", "hn_any", "hn_sub", "hn_logd", "num_jac"]):
        out[n] = F[:, i]
    F = np.array([_numfeats2(a or [], b or []) for a, b in zip(nm1, nm2)], np.float32)
    for i, n in enumerate(["hn_rel", "nums_unB", "nums_unA", "num_last_eq", "t_hn_in_s1"]):
        out[n] = F[:, i]
    F = np.array([_shortun(a or [], b or []) for a, b in zip(nt1, nt2)], np.float32)
    for i, n in enumerate(["short_unA", "short_unB", "un_lev1"]):
        out[n] = F[:, i]
    out["n_nums1"] = L["nums_1"].list.len().fill_null(0).to_numpy().astype(np.int16)
    out["n_nums2"] = L["nums_2"].list.len().fill_null(0).to_numpy().astype(np.int16)
    # first token agreement
    out["first_eq"] = np.array([int(bool(a) and bool(b) and a[0] == b[0]) for a, b in zip(co1, co2)], np.int8)
    # alias part similarity
    alp = L["al_post_2"].fill_null("").to_list(); alpre = L["al_pre_2"].fill_null("").to_list()
    ns1 = L["ns_1"].fill_null("").to_list()
    out["has_alias"] = np.array([int(bool(x) or bool(y)) for x, y in zip(alp, alpre)], np.int8)
    out["alias_best"] = np.maximum(process.cpdist(ns1, alp, scorer=fuzz.token_set_ratio, workers=WORKERS),
                                   process.cpdist(ns1, alpre, scorer=fuzz.token_set_ratio, workers=WORKERS)).astype(np.float32)
    # flags
    for c in ["ind_2", "aempty_2", "dom_2", "oov_2", "ncomp_1", "ncomp_2"]:
        out[c] = L[c].fill_null(0).cast(pl.Int16).to_numpy()
    out["is_s3"] = (L["t"] // 10_000_000_000 == 3).to_numpy().astype(np.int8)
    out["len_ns1"] = L["ns_1"].str.len_chars().fill_null(0).to_numpy().astype(np.int16)
    out["len_ns2"] = L["ns_2"].str.len_chars().fill_null(0).to_numpy().astype(np.int16)
    out["len_as2"] = L["as_2"].str.len_chars().fill_null(0).to_numpy().astype(np.int16)
    out["same_country"] = (L["country_1"] == L["country_2"]).to_numpy().astype(np.int8)
    for c in ["bscore", "nkeys", "brank"]:
        if c in L.columns:
            out[c] = L[c].to_numpy().astype(np.float32)
    feats = pl.DataFrame(out)
    return pl.concat([pairs.select("s1", "t"), feats], how="horizontal")
