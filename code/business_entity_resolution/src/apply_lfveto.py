"""Legal-form-change veto for countries without training labels, decided label-free per country.
A pair "changes legal form" when both names carry a legal form and the canonical sets differ. For each country, the
change rate is measured on its confident same-number pairs (c_true: true copies) and on its adjacent-number pairs
(c_sib: siblings, see RESEARCH_LOG_TEST_SHIFT.md). If the country's generator never changes the legal form of a copy
(c_true < CT_MAX) but siblings often do (c_sib > CS_MIN), legal-form-change pairs are vetoed for that country.
Training countries are left to the learned model (their copies swap legal forms routinely).
usage: apply_lfveto.py <in_dir with scores.parquet> <out_dir>"""
import polars as pl, glob, os, sys
sys.path.insert(0, "work")
from ids import to_str
IN, OUT = sys.argv[1], sys.argv[2]
CT_MAX = float(os.environ.get("CT_MAX", "0.005")); CS_MIN = float(os.environ.get("CS_MIN", "0.2")); TH = 0.7
sys.path.insert(0, 'work')
from lfutil import lf
recs = []
for s in (1, 2, 3):
    d = pl.read_parquet(f"work/test_s{s}.parquet", columns=["entity_id", "business_name", "country"]).with_columns(
        (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"))
    recs.append(d.select("id", "country", lf("business_name").alias("lf")))
R = pl.concat(recs)
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
sc = pl.read_parquet(f"{IN}/scores.parquet")
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_rel", "hn_eq", "aempty_2"]) for d in ["work/feat_test", "work/feat_test_dense"]
               for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
x = sc.filter(pl.col("p2") >= 0.5).join(f, on=["s1", "t"], how="left").join(R.rename({"id": "s1", "lf": "lf1"}), on="s1").join(
    R.select(pl.col("id").alias("t"), pl.col("lf").alias("lf2")), on="t")
x = x.with_columns(((pl.col("lf1") != "") & (pl.col("lf2") != "") & (pl.col("lf1") != pl.col("lf2"))).alias("chg"))
rates = {}
for c in x["country"].unique().to_list():
    xc = x.filter(pl.col("country") == c)
    ct = xc.filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1) & (pl.col("p2") >= 0.95))["chg"].mean()
    cs = xc.filter(pl.col("hn_rel").is_in([5, 7]))["chg"].mean()
    rates[c] = (ct, cs, c not in seen and ct < CT_MAX and cs > CS_MIN)
print("legal-form change rate (true-copy proxy, sibling proxy, veto):", rates)
veto = x.filter(pl.col("chg") & pl.col("country").replace_strict({c: v[2] for c, v in rates.items()}, default=False)).select("s1", "t")
res = sc.join(veto.with_columns(pl.lit(True).alias("v")), on=["s1", "t"], how="left").with_columns(
    pl.when(pl.col("v").fill_null(False)).then(0.0).otherwise(pl.col("p2n")).alias("p2n")).drop("v")
def select(df, col):
    return df.with_columns(pl.col(col).rank("ordinal", descending=True).over("t").alias("rt")).filter((pl.col(col) >= TH) & (pl.col("rt") == 1))
old, new = select(sc, "p2n"), select(res, "p2n")
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
rem = old.join(new, on=["s1", "t"], how="anti").join(s1te.select("s1", "country"), on="s1")
print("vetoed pairs", len(veto), "| accepted links removed by country:", rem.group_by("country").len().rows())
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
