"""fr6 vs fr5 (France): added/removed links by class; copy-count test (other links of the S1 in the NEW output, excluding
the changed link) for added links per class; reference = fr5-accepted known-filler links."""
import polars as pl, glob, sys
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
A, B = sys.argv[1], sys.argv[2]; FD = sys.argv[3]


def links(o):
    m = pl.read_csv(f"{o}/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    return (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").select(
        (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
        (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000) + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))


fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
a = links(A).join(fr, on="s1", how="semi"); b = links(B).join(fr, on="s1", how="semi")
add = b.join(a, on=["s1", "t"], how="anti"); rem = a.join(b, on=["s1", "t"], how="anti")
FC = ["s1", "t", "hn_rel", "hn_eq", "aempty_2", "cs_tset", "acs_tset"]
ch = pl.concat([add.with_columns(pl.lit("added").alias("k")), rem.with_columns(pl.lit("removed").alias("k"))])
f = pl.concat([pl.read_parquet(p, columns=FC).join(ch.select("s1", "t"), on=["s1", "t"], how="semi") for d in FD.split(",") for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
ch = ch.join(f, on=["s1", "t"], how="left").with_columns(pl.concat_str([
    pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq")).when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum")).when(pl.col("hn_rel").is_in([5, 7])).then(pl.lit("adj")).otherwise(pl.lit("othnum")),
    pl.when(pl.col("cs_tset") >= 99.9).then(pl.lit("n=")).when(pl.col("cs_tset") >= 80).then(pl.lit("n~")).otherwise(pl.lit("n≠"))], separator="|").alias("c"))
print(f"added {add.height} removed {rem.height}")
print(ch.group_by("k", "c").len().pivot(on="k", index="c", values="len").fill_null(0).sort("added", descending=True))
oth = b.group_by("s1").len()
for k in ("added", "removed"):
    x = ch.filter(pl.col("k") == k)
    g = x.select("s1", "c").unique().join(oth, on="s1", how="left").with_columns(pl.col("len").fill_null(0))
    # other links excluding the changed ones of that S1
    nch = x.group_by("s1").len().rename({"len": "nch"})
    g = g.join(nch, on="s1").with_columns((pl.col("len") - pl.when(pl.lit(k) == "added").then(pl.col("nch")).otherwise(0)).alias("other"))
    print(k, "copy-count by class (other links in new output):")
    print(g.group_by("c").agg(pl.len().alias("S1s"), pl.col("other").mean().round(3).alias("mean_other")).filter(pl.col("S1s") >= 200).sort("S1s", descending=True))
