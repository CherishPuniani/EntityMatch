"""Empty-S1 rescue for countries without training labels (RESEARCH_LOG_TEST_SHIFT.md §23-§24).

The generator's per-S1 copy-count distribution is country-independent in the labelled countries (5.58 % of S1s have no
copy in US and India alike), but France is predicted empty for 6.26 % of its S1s: ~1,770 S1s have copies and get no
link (F = 0 each). For an S1 without links, a wrong link costs only when the S1 is a true singleton, so the break-even
precision is ~48 % (vs ~72 % for additions to linked S1s).

Rule, per S1 of an unlabelled country that has no link in <in_dir>:
  * T = its best candidate (max p2n); T is claimed by no S1 and T's best S1 is this S1;
  * same house number, and every S1 street word (street types / stop words dropped) is fuzzy-present in T's address;
  * p2n >= 0.2, or 0.05 <= p2n < 0.2 when T only adds fillers / legal forms to the S1 name (no compagnie/societe/cie);
  * the name edit is filler-only, legal-form added (S1 had none) or identical name. Random-name candidates and other edits
    are dropped: their rate among empty S1s vs linked S1s (0.51 / 0.83) is at or near the sibling-only level (~0.55),
    while filler-only 1.07 / legal-form added 1.49 show a copy excess (label-free, §24).

usage (from the challenge root, scripts copied to work/):
  PREP=work_unseen rescue_empty_unseen.py <in_dir: matching_results.tsv + candidate_pairs.tsv> <scores_dir: scores.parquet> <out_dir>"""
import os, re, sys, unicodedata
import polars as pl
from rapidfuzz import fuzz

IN, SCD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PREP = os.environ.get("PREP", "work_unseen")
D = "dataset"
B = 10_000_000_000


def tid(e):
    return (pl.when(e.str.starts_with("S2")).then(2 * B).when(e.str.starts_with("S3")).then(3 * B).otherwise(B)
            + e.str.slice(3).cast(pl.Int64))


seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
m = pl.read_csv(f"{IN}/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
L = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
     .filter(pl.col("matched_entity_ids") != "").select(tid(pl.col("source1_entity_id")).alias("s1"), tid(pl.col("matched_entity_ids")).alias("t")))

# normalised records of the unlabelled countries (house number, street/city tokens) from the extended normaliser
P = pl.concat([pl.read_parquet(f"{PREP}/p_test_s{s}.parquet", columns=["id", "country", "acore"]).filter(~pl.col("country").is_in(list(seen))).select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * B).alias("id"), "country",
    pl.col("acore").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first().alias("hn"),
    pl.col("acore").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).alias("st"),
    pl.col("acore").list.len().alias("alen")) for s in (1, 2, 3)])
S1u = P.filter(pl.col("id") < 2 * B).select(pl.col("id").alias("s1"))
emp = S1u.join(L.select("s1").unique(), on="s1", how="anti")
print("unlabelled-country S1:", S1u.height, " without links:", emp.height)

sc = pl.read_parquet(f"{SCD}/scores.parquet", columns=["s1", "t", "p2n"])
tmax = sc.group_by("t").agg(pl.col("p2n").max().alias("tmax"))
b = (sc.join(emp, on="s1", how="semi").sort("p2n", descending=True).group_by("s1").head(1)
     .join(L.select("t").unique(), on="t", how="anti").join(tmax, on="t").filter(pl.col("p2n") >= pl.col("tmax")))
A = P.select(pl.col("id").alias("s1"), pl.col("hn").alias("hn1"), pl.col("st").alias("st1"))
T = P.select(pl.col("id").alias("t"), pl.col("hn").alias("hn2"), pl.col("st").alias("st2"), "alen")
stov = pl.col("st1").list.set_intersection("st2").list.len() / pl.max_horizontal(pl.col("st1").list.len(), 1)
b = b.join(A, on="s1").join(T, on="t").filter((pl.col("alen") > 0) & (pl.col("hn1") == pl.col("hn2")) & (stov >= 0.5) & (pl.col("p2n") >= 0.05))

raw = pl.concat([pl.read_csv(f"{D}/test/test_source{s}.tsv", separator="\t", quote_char=None,
                             schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8})
                 .filter(~pl.col("country").is_in(list(seen)))
                 .select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * B).alias("id"), "entity_id", "business_name", "business_address")
                 for s in (1, 2, 3)])
info = {r[0]: r[1:] for r in raw.iter_rows()}


def nrm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


STOP = {"rue", "r", "avenue", "av", "ave", "boulevard", "bd", "bld", "allee", "all", "impasse", "imp", "place", "pl", "chemin", "ch", "route",
        "rte", "cours", "quai", "de", "du", "des", "la", "le", "les", "l", "d", "et", "bis", "ter", "no", "n", "nº", "n°", "cedex", "parvis",
        "square", "sq", "passage", "voie", "residence", "res"}


def street_words(a):
    a = re.sub(r"[^a-z0-9 ,]", " ", nrm(a))
    seg = [p for p in a.split(",") if re.search(r"\d", p)]
    seg = seg[0] if seg else ""
    return [w for w in seg.split() if not w.isdigit() and w not in STOP and len(w) >= 3 and not re.fullmatch(r"\d+\w*", w)]


def street_ok(s1, t):
    ws = street_words(info[s1][2])
    tw = re.sub(r"[^a-z0-9 ,]", " ", nrm(info[t][2])).replace(",", " ").split()
    return bool(ws) and bool(tw) and all(max(fuzz.ratio(w, v) for v in tw) >= 75 for w in ws)


# name edit, low band (0.05 <= p2n < 0.2): T adds nothing but fillers / legal-form letters
LF1 = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "ei", "s", "a", "r", "l", "u", "e", "c", "i"}
FILL1 = {"developpement", "france", "groupe", "associes", "fils", "freres", "cie", "et", "services", "and", "partners", "co", "compagnie"}


def tok1(x):
    return set(w for w in re.split(r"[^a-z0-9]+", nrm(x)) if w)


def low_band_ok(s1, t):
    a, c = tok1(info[s1][1]), tok1(info[t][1])
    if not (a - LF1) & (c - LF1):
        return False
    return not (c - a - LF1 - FILL1) and not ((c - a) & {"compagnie", "societe", "cie"})


# name edit class of the final filter
LFS = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "ei", "snc", "selarl", "scop", "scm", "scp", "sca"}
FILL2 = {"developpement", "france", "groupe", "associes", "fils", "freres", "cie", "et", "services", "and", "partners", "co", "le", "la", "les", "de", "du", "des"}


def toks2(name):
    s = nrm(name)
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", s).replace(".", "")
    return set(w for w in re.split(r"[^a-z0-9]+", s) if w)


def kind(s1, t):
    a, c = toks2(info[s1][1]), toks2(info[t][1])
    la, lc = a & LFS, c & LFS
    if la and lc and la != lc:
        return "lf_change"
    A2, C2 = a - LFS, c - LFS
    if A2 == C2:
        return "lf_add" if (lc and not la) else "same_name"
    if not A2 & C2:
        return "nooverlap"
    return "filler" if not (C2 - A2 - FILL2) else "other"


b = b.with_columns(pl.struct("s1", "t").map_elements(lambda r: street_ok(r["s1"], r["t"]), return_dtype=pl.Boolean).alias("sok")).filter(pl.col("sok"))
hi = b.filter(pl.col("p2n") >= 0.2).sort("p2n", descending=True).unique("t", keep="first").unique("s1", keep="first")
lo = b.filter(pl.col("p2n") < 0.2).sort("p2n", descending=True).unique("t", keep="first").unique("s1", keep="first")
lo = lo.filter(pl.struct("s1", "t").map_elements(lambda r: low_band_ok(r["s1"], r["t"]), return_dtype=pl.Boolean))
sel = pl.concat([hi, lo]).with_columns(pl.struct("s1", "t").map_elements(lambda r: kind(r["s1"], r["t"]), return_dtype=pl.Utf8).alias("kind"))
print(sel.group_by("kind").agg(pl.len()).sort("kind"))
sel = sel.filter(pl.col("kind").is_in(["filler", "lf_add", "same_name"]))
assert sel["s1"].n_unique() == sel.height and sel["t"].n_unique() == sel.height
R = {info[a][0]: info[c][0] for a, c in sel.select("s1", "t").iter_rows()}
print("links added:", len(R))

os.makedirs(OUT, exist_ok=True)
with open(f"{IN}/matching_results.tsv", encoding="utf-8") as fi, open(f"{OUT}/matching_results.tsv", "w", encoding="utf-8") as fo:
    for i, line in enumerate(fi):
        if i:
            a, _, c = line.rstrip("\n").partition("\t")
            if a in R:
                assert c.strip() == "", (a, c)
                line = f"{a}\t{R[a]}\n"
        fo.write(line)
# candidate file must contain every matched pair (the rescued T is a scored candidate, so normally nothing changes)
patched = 0
with open(f"{IN}/candidate_pairs.tsv", encoding="utf-8") as fi, open(f"{OUT}/candidate_pairs.tsv", "w", encoding="utf-8") as fo:
    for i, line in enumerate(fi):
        if i:
            a, _, c = line.rstrip("\n").partition("\t")
            if a in R and R[a] not in c.split(","):
                patched += 1
                line = f"{a}\t{R[a]},{c}\n" if c else f"{a}\t{R[a]}\n"
        fo.write(line)
print("candidate lines patched:", patched)
