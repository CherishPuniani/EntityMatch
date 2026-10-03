"""Same-number pairs by name-edit type: target adds / drops core tokens relative to the S1. Train: true rate and p2;
test: p2 per country (and, for France, under the re-featurised scores)."""
import polars as pl, glob, sys
def run(split, country, sp, fd, pcol="p2"):
    cols = ["s1", "t", pcol] + (["y"] if split == "train" else [])
    sc = pl.read_parquet(sp, columns=cols).rename({pcol: "p"}).filter(pl.col("p") >= 0.05)
    T = []
    for s in (1, 2, 3):
        d = pl.read_parquet(f"work/p_{split}_s{s}.parquet", columns=["id", "country", "nt", "core"]).filter(pl.col("country") == country)
        T.append(d.select((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "nt", "core"))
    T = pl.concat(T)
    f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_eq", "aempty_2", "acs_tset"]) for d in fd for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
    x = sc.join(f, on=["s1", "t"]).filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1))
    x = x.join(T.select(pl.col("id").alias("s1"), pl.col("nt").alias("nt1"), pl.col("core").alias("c1")), on="s1").join(
        T.select(pl.col("id").alias("t"), pl.col("nt").alias("nt2"), pl.col("core").alias("c2")), on="t")
    x = x.with_columns((pl.col("c2").list.set_difference("nt1").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.len() > 0).alias("adds"),
                       (pl.col("c1").list.set_difference("nt2").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.len() > 0).alias("drops"))
    x = x.with_columns(pl.when(pl.col("adds") & pl.col("drops")).then(pl.lit("drop+add")).when(pl.col("adds")).then(pl.lit("add")).when(pl.col("drops")).then(pl.lit("drop")).otherwise(pl.lit("same")).alias("edit"))
    agg = [pl.len().alias("n"), pl.col("p").mean().alias("mean_p"), (pl.col("p") >= 0.7).mean().alias("acc_rate")] + ([pl.col("y").mean().alias("true_rate")] if split == "train" else [])
    print(split, country, sp.split("/")[-2] if "/" in sp else sp); print(x.group_by("edit").agg(agg).sort("edit"))
pl.Config.set_tbl_width_chars(160)
run("train", "US", "work2/oof_ce3.parquet", ["work/feat_trainD", "work/feat_trainD_dense"])
run("train", "India", "work2/oof_ce3.parquet", ["work/feat_trainD", "work/feat_trainD_dense"])
run("test", "US", "output_shift_routed/scores.parquet", ["work/feat_test", "work/feat_test_dense"])
run("test", "France", "output_shift_routed/scores.parquet", ["work/feat_test", "work/feat_test_dense"])
run("test", "France", "work2/test_scores_tree_frnorm2.parquet", ["work/feat_test", "work/feat_test_dense"])
