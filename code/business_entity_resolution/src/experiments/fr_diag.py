"""Label-free France diagnostics for submitted (corrected) score files: acceptance by stratum / name edit.
usage: fr_diag.py name=scores.parquet [name=scores.parquet ...]   (files with s1, t, p2n)"""
import polars as pl, glob, sys
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
fr = s1.filter(pl.col("country") == "France").select("s1"); n = len(fr)
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_rel", "hn_eq", "aempty_2", "core_unA", "core_unB"]) for d in ["work/feat_test", "work/feat_test_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))]).join(fr, on="s1")
f = f.with_columns(pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_rel").is_in([4, 5, 7])).then(pl.lit("adjacent"))
    .when(pl.col("hn_eq") == 1).then(pl.when((pl.col("core_unA") > 0) & (pl.col("core_unB") > 0)).then(pl.lit("same:drop+add")).when(pl.col("core_unB") > 0).then(pl.lit("same:add"))
    .when(pl.col("core_unA") > 0).then(pl.lit("same:drop")).otherwise(pl.lit("same:identical"))).otherwise(pl.lit("other")).alias("st")).select("s1", "t", "st")
rows = []
for arg in sys.argv[1:]:
    name, path = arg.split("=")
    sc = pl.read_parquet(path, columns=["s1", "t", "p2n"]).join(fr, on="s1").with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
    sc = sc.with_columns(((pl.col("p2n") >= 0.7) & (pl.col("rt") == 1)).alias("acc")).join(f, on=["s1", "t"], how="left")
    by = {r[0]: r[1] for r in sc.filter(pl.col("p2n") >= 0.05).group_by("st").agg(pl.col("acc").mean()).iter_rows()}
    k = sc.group_by("s1").agg(pl.col("acc").sum().alias("k"))
    rows.append(dict(name=name, matches_per_S1=round(k["k"].mean(), 3), empty_pct=round((k["k"] == 0).mean() * 100, 2),
                     **{f"acc:{s}": round(by.get(s, float('nan')), 3) for s in ["same:identical", "same:drop", "same:add", "same:drop+add", "empty", "adjacent", "other"]}))
pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(20)
print(pl.DataFrame(rows))
print("US/India reference (same-address pairs, train true rate): identical 99.9 %, drop 99.6-99.9 %, add 99.9 %, drop+add 97.6-97.9 %")
