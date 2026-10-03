"""What are the extra uncertain test pairs? Per-S1 rates of pairs by p2 band x feature stratum, train OOF vs test."""
import polars as pl, glob
COLS = ["s1", "t", "hn_rel", "hn_eq", "cs_tset", "ns_ratio", "as_tset", "acs_tset", "aempty_2", "core_covA", "core_covB", "dom_2", "ind_2", "num_jac", "rs"]
def feats(dirs):
    return pl.concat([pl.read_parquet(p, columns=COLS) for d in dirs for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
s1tr = pl.read_parquet("work/p_train_s1.parquet", columns=["id", "country"]).with_columns(
    (pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
def strat(df):
    return df.with_columns(
        pl.when(pl.col("aempty_2") == 1).then(pl.lit("t_addr_empty"))
        .when(pl.col("hn_eq") == 1).then(pl.lit("hn_equal"))
        .when(pl.col("hn_rel").is_null()).then(pl.lit("hn_missing"))
        .otherwise(pl.concat_str(pl.lit("hn_rel="), pl.col("hn_rel").cast(pl.Utf8))).alias("stratum"),
        pl.when(pl.col("p2") < 0.1).then(pl.lit("a<0.1")).when(pl.col("p2") < 0.5).then(pl.lit("b0.1-0.5"))
        .when(pl.col("p2") < 0.7).then(pl.lit("c0.5-0.7")).when(pl.col("p2") < 0.9).then(pl.lit("d0.7-0.9")).otherwise(pl.lit("e>0.9")).alias("band"))
tr = pl.read_parquet("work/oof_dense_s1.parquet", columns=["s1", "t", "y", "p2"]).filter((pl.col("p2") > 0.02))
tr = tr.join(feats(["work/feat_trainD", "work/feat_trainD_dense"]), on=["s1", "t"], how="left").join(s1tr, on="s1")
te = pl.read_parquet("work/test_scores_dense_s1.parquet", columns=["s1", "t", "p2"]).filter(pl.col("p2") > 0.02)
te = te.join(feats(["work/feat_test", "work/feat_test_dense"]), on=["s1", "t"], how="left").join(s1te, on="s1")
ntr = {c: n for c, n in pl.read_parquet("work/keep_trainD.parquet").rename({"id": "s1"}).join(s1tr, on="s1").group_by("country").len().iter_rows()}
nte = {c: n for c, n in s1te.group_by("country").len().iter_rows()}
tr, te = strat(tr), strat(te)
tr.write_parquet("work2/diag_tr_unc.parquet"); te.write_parquet("work2/diag_te_unc.parquet")
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
a = tr.group_by("country", "band", "stratum").agg(pl.len().alias("n_tr"), pl.col("y").mean().alias("ypos_tr"), pl.col("p2").mean().alias("p_tr"))
a = a.with_columns((pl.col("n_tr") * 1000 / pl.col("country").replace_strict(ntr)).alias("per1k_tr"))
b = te.group_by("country", "band", "stratum").agg(pl.len().alias("n_te"), pl.col("p2").mean().alias("p_te"))
b = b.with_columns((pl.col("n_te") * 1000 / pl.col("country").replace_strict(nte)).alias("per1k_te"))
m = b.join(a, on=["country", "band", "stratum"], how="full", coalesce=True).with_columns((pl.col("per1k_te") / pl.col("per1k_tr")).alias("ratio"))
m = m.filter(pl.col("band").is_in(["b0.1-0.5", "c0.5-0.7", "d0.7-0.9"])).sort("country", "band", "per1k_te", descending=[False, False, True])
print(m.select("country", "band", "stratum", "per1k_tr", "per1k_te", "ratio", "ypos_tr", "p_tr", "p_te"))
