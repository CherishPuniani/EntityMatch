"""France S1s predicted empty with a plausible top candidate: stratum of the top candidate, whether the correction
touched it, examples."""
import polars as pl, glob, sys
sys.path.insert(0, "work")
from lfutil import lf
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(70)
sc = pl.read_parquet("output_shift_routed/scores.parquet")
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
sc = sc.join(s1te, on="s1")
sc = sc.with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
k = sc.group_by("s1").agg(((pl.col("p2n") >= 0.7) & (pl.col("rt") == 1)).sum().alias("k"))
top = sc.sort("p2n", descending=True).group_by("s1", maintain_order=True).first()
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_rel", "hn_eq", "aempty_2", "cs_tset", "acs_tset", "t_ns1"]) for d in ["work/feat_test", "work/feat_test_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
e = top.join(k.filter(pl.col("k") == 0), on="s1").filter((pl.col("p2n") >= 0.1) & (pl.col("p2n") < 0.7)).join(f, on=["s1", "t"], how="left")
e = e.with_columns(pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq")).when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum")).otherwise(pl.concat_str(pl.lit("r"), pl.col("hn_rel").cast(pl.Int32).cast(pl.Utf8))).alias("st"),
                   (pl.col("p2n") < pl.col("p2") - 1e-6).alias("corrected"), (pl.col("rt") > 1).alias("owned_elsewhere"))
print(e.group_by("country", "st").agg(pl.len(), pl.col("corrected").mean(), pl.col("owned_elsewhere").mean(), pl.col("p2n").mean(), pl.col("cs_tset").mean(), pl.col("acs_tset").mean()).sort("country", "len", descending=[False, True]))
e.write_parquet("work2/fr_uncertain_empty.parquet")
def recs(split):
    out = []
    for s in (1, 2, 3):
        d = pl.read_parquet(f"work/{split}_s{s}.parquet", columns=["entity_id", "business_name", "business_address"]).with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"))
        out.append(d.select("id", pl.col("business_name").alias("nm"), pl.col("business_address").alias("ad")))
    return pl.concat(out)
R = recs("test")
x = e.filter((pl.col("country") == "France") & ~pl.col("corrected")).sample(12, seed=7)
for s1, p in x.select("s1", "p2n").iter_rows():
    cand = sc.filter(pl.col("s1") == s1).filter(pl.col("p2") > 0.03).join(R, left_on="t", right_on="id").sort("p2", descending=True).head(6)
    s = R.filter(pl.col("id") == s1)
    print("== S1:", s["nm"][0], "|", s["ad"][0])
    for tt, p2, p2n, rt, nm, ad in cand.select("t", "p2", "p2n", "rt", "nm", "ad").iter_rows():
        print("   p2 %.3f p2n %.3f rank_t %d  %s | %s" % (p2, p2n, rt, nm, ad))
