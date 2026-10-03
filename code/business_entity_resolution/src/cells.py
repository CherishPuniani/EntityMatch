"""Per-cell label-shift estimate: train positives per S1 vs test pairs per S1, cell = country x hn stratum x p2 bin.
usage: cells.py <train_oof.parquet> <test_scores.parquet> <out_prefix>"""
import polars as pl, glob, sys, numpy as np
COLS = ["s1", "t", "hn_rel", "hn_eq", "aempty_2"]
import os
BINS = [float(x) for x in os.environ.get("BINS", "0.3,0.5,0.6,0.7,0.8,0.9,0.95").split(",")]
def feats(dirs):
    return pl.concat([pl.read_parquet(p, columns=COLS) for d in dirs for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
def strat(df):
    return df.with_columns(
        pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq"))
        .when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum"))
        .otherwise(pl.concat_str(pl.lit("r"), pl.col("hn_rel").cast(pl.Int32).cast(pl.Utf8))).alias("st"),
        pl.col("p2").cut(BINS, left_closed=True).cast(pl.Utf8).alias("bin"))
s1tr = pl.read_parquet("work/p_train_s1.parquet", columns=["id", "country"]).with_columns(
    (pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
keep = pl.read_parquet("work/keep_trainD.parquet").rename({"id": "s1"})
trp, tep, out = sys.argv[1], sys.argv[2], sys.argv[3]
tr = pl.read_parquet(trp, columns=["s1", "t", "y", "p2"]).filter(pl.col("p2") >= 0.3)
tr = strat(tr.join(feats(["work/feat_trainD", "work/feat_trainD_dense"]), on=["s1", "t"], how="left").join(s1tr, on="s1"))
te = pl.read_parquet(tep, columns=["s1", "t", "p2"]).filter(pl.col("p2") >= 0.3)
te = strat(te.join(feats(["work/feat_test", "work/feat_test_dense"]), on=["s1", "t"], how="left").join(s1te, on="s1"))
ntr = dict(keep.join(s1tr, on="s1").group_by("country").len().iter_rows())
nte = dict(s1te.group_by("country").len().iter_rows())
a = tr.group_by("country", "st", "bin").agg(pl.len().alias("n_tr"), pl.col("y").sum().alias("pos_tr"))
# pooled (all training countries) rates, used for countries without labels
pool = a.group_by("st", "bin").agg(pl.col("n_tr").sum(), pl.col("pos_tr").sum()).with_columns(
    (pl.col("n_tr") / sum(ntr.values())).alias("nrate_tr"), (pl.col("pos_tr") / sum(ntr.values())).alias("prate_tr"))
a = a.with_columns((pl.col("n_tr") / pl.col("country").replace_strict(ntr)).alias("nrate_tr"),
                   (pl.col("pos_tr") / pl.col("country").replace_strict(ntr)).alias("prate_tr"))
b = te.group_by("country", "st", "bin").agg(pl.len().alias("n_te")).with_columns(
    (pl.col("n_te") / pl.col("country").replace_strict(nte)).alias("nrate_te"))
seen = set(ntr)
bs = b.filter(pl.col("country").is_in(list(seen))).join(a, on=["country", "st", "bin"], how="left")
bu = b.filter(~pl.col("country").is_in(list(seen))).join(pool, on=["st", "bin"], how="left")
m = pl.concat([bs, bu.select(bs.columns)]).with_columns(pl.col(["n_tr", "pos_tr"]).fill_null(0))
# smoothed train precision and estimated test precision (positives per S1 invariant, excess = negatives)
m = m.with_columns(((pl.col("pos_tr") + 1) / (pl.col("n_tr") + 2)).alias("prec_tr"))
m = m.with_columns(pl.min_horizontal(pl.col("prec_tr"), (pl.col("prate_tr") + 1e-7) / pl.col("nrate_te")).alias("prec_te"),
                   (pl.col("nrate_te") / (pl.col("nrate_tr") + 1e-9)).alias("ratio"))
m = m.with_columns((pl.col("prec_te") / pl.col("prec_tr")).alias("mult"))
m.write_parquet(f"{out}_cells.parquet")
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_width_chars(220)
if os.environ.get("QUIET"): sys.exit(0)
print(m.filter(pl.col("mult") < 0.9).sort("country", "st", "bin").select("country", "st", "bin", "n_te", "nrate_tr", "nrate_te", "ratio", "prec_tr", "prec_te", "mult"))
# excess accepted (p2>=0.7) pairs per 1k S1 = n_te - pos_tr-equivalent
acc = m.filter(pl.col("bin").str.slice(1, 3).cast(pl.Float64, strict=False) >= 0.7)
print(acc.group_by("country").agg((pl.col("nrate_te") * 1000).sum().alias("acc_te_per1k"),
                                   (pl.col("nrate_te") * (1 - pl.col("prec_te")) * 1000).sum().alias("estFP_per1k"),
                                   (pl.col("nrate_tr") * (1 - pl.col("prec_tr")) * 1000).sum().alias("trainFP_per1k")))
