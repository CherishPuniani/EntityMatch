"""Type-word-swap veto for countries without training labels (label-free).
At the S1's own address, a target that replaces one S1 name word by a "type word" of the country's name generator, or
appends one, is a different business: in India/US training data the analogous replacements and appends are 0-6 % true,
while the generator's copy fillers are 95-99 % true (RESEARCH_LOG_TEST_SHIFT.md §19). Type words are found label-free
per country: single added core tokens (>= 4 characters) at the same address with n >= NMIN and replace:append >= RATIO
(copy fillers sit at 2-4, appended descriptors near 0). Morphological variants (prefix, or edit distance <= 2) are exempt.
usage: apply_twveto.py <in_dir with scores.parquet> <out_dir>
env: PREP = dir with the p_test_s*.parquet of the normaliser used for the unseen countries; FEATU = comma list of their
feature dirs (acs_tset); hn_eq / aempty_2 are name-independent."""
import polars as pl, glob, os, sys
sys.path.insert(0, "work")
from ids import to_str
IN, OUT = sys.argv[1], sys.argv[2]
PREP = os.environ["PREP"]; FEATU = os.environ["FEATU"].split(",")
RATIO = float(os.environ.get("RATIO", "10")); NMIN = int(os.environ.get("NMIN", "100")); TH = 0.7
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
un = s1te.filter(~pl.col("country").is_in(list(seen))).select("s1", "country")
sc = pl.read_parquet(f"{IN}/scores.parquet")
T = []
for s in (1, 2, 3):
    d = pl.read_parquet(f"{PREP}/p_test_s{s}.parquet", columns=["id", "country", "nt", "core"]).filter(~pl.col("country").is_in(list(seen)))
    T.append(d.select((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "nt", "core"))
T = pl.concat(T)
x = sc.join(un, on="s1").select("s1", "t", "p2n", "country")
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_eq", "aempty_2", "acs_tset"]) for d in FEATU for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
x = x.join(f, on=["s1", "t"]).filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1) & (pl.col("acs_tset") >= 90))
x = x.join(T.select(pl.col("id").alias("s1"), pl.col("nt").alias("nt1"), pl.col("core").alias("c1")), on="s1").join(
    T.select(pl.col("id").alias("t"), pl.col("nt").alias("nt2"), pl.col("core").alias("c2")), on="t")
L2 = lambda e: e.list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
x = x.with_columns(L2(pl.col("c2").list.set_difference("nt1")).alias("added"), L2(pl.col("c1").list.set_difference("nt2")).alias("dropped"))
x = x.filter(pl.col("added").list.len() == 1).with_columns(
    pl.col("added").list.first().alias("a"), pl.col("dropped").list.len().alias("nd"), pl.col("dropped").list.first().alias("d"))
voc = (x.filter(pl.col("a").str.len_chars() >= 4).group_by("country", "a")
       .agg(pl.len().alias("n"), (pl.col("nd") == 0).sum().alias("n_add"), (pl.col("nd") > 0).sum().alias("n_repl"))
       .filter((pl.col("n") >= NMIN) & (pl.col("n_repl") >= RATIO * pl.col("n_add"))))
print("type-word vocabulary:", voc.height, "tokens;", voc.sort("n", descending=True)["a"].head(60).to_list())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


cand = x.join(voc.select("country", "a"), on=["country", "a"]).filter(pl.col("nd") <= 1)
pairs = cand.select("a", "d").unique()
morph = {(a, d): d is not None and (a.startswith(d) or d.startswith(a) or lev(a, d) <= 2) for a, d in pairs.iter_rows()}
cand = cand.with_columns(pl.struct("a", "d").map_elements(lambda r: morph[(r["a"], r["d"])], return_dtype=pl.Boolean).alias("morph"))
veto = cand.filter(~pl.col("morph")).select("s1", "t").unique()
print("same-address type-word pairs:", cand.height, "| morphological variants exempt:", cand["morph"].sum(), "| vetoed:", veto.height,
      "| of which currently accepted (p2n >= 0.7):", cand.filter(~pl.col("morph") & (pl.col("p2n") >= TH)).height)
res = sc.join(veto.with_columns(pl.lit(True).alias("v")), on=["s1", "t"], how="left").with_columns(
    pl.when(pl.col("v").fill_null(False)).then(0.0).otherwise(pl.col("p2n")).alias("p2n")).drop("v")


def select(df, col):
    return df.with_columns(pl.col(col).rank("ordinal", descending=True).over("t").alias("rt")).filter((pl.col(col) >= TH) & (pl.col("rt") == 1))


old, new = select(sc, "p2n"), select(res, "p2n")
rem = old.join(new, on=["s1", "t"], how="anti").join(s1te.select("s1", "country"), on="s1")
addl = new.join(old, on=["s1", "t"], how="anti").join(s1te.select("s1", "country"), on="s1")
print("accepted links removed:", rem.group_by("country").len().rows(), "| re-assigned targets added:", addl.group_by("country").len().rows())
os.makedirs(OUT, exist_ok=True)
S1 = s1te.select(pl.col("entity_id").alias("source1_entity_id"))


def write(pairs, pcol, colname, path):
    agg = (pairs.sort(["s1", pcol], descending=[False, True]).with_columns(to_str("s1"), to_str("t"))
           .group_by("s1", maintain_order=True).agg(pl.col("t").str.join(",").alias(colname)).rename({"s1": "source1_entity_id"}))
    o = S1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col(colname).fill_null(""))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{colname}\n")
        for a, b in o.iter_rows():
            fh.write(f"{a}\t{b}\n")


write(new, "p2n", "matched_entity_ids", f"{OUT}/matching_results.tsv")
write(res, "p2n", "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
res.write_parquet(f"{OUT}/scores.parquet")
k = s1te.join(new.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
print("matches per S1 / empty:", k.group_by("country").agg(pl.col("len").mean().alias("k"), (pl.col("len") == 0).mean().alias("empty")).sort("country").rows())
