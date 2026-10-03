"""Sibling-cluster signature for adjacent-number pairs: S1's own number corroborated by an accepted exact-number copy
(eqacc) and the target's number shared by another candidate (clus). Train OOF vs test densities per 1k S1."""
import polars as pl, glob, sys
trp, tep = sys.argv[1], sys.argv[2]
at_tr = pl.read_parquet("work/rec_attrs_trainD.parquet", columns=["id", "hn_h"])
at_te = pl.read_parquet("work/rec_attrs_test.parquet", columns=["id", "hn_h"])
def prep(sc, fd, at, s1c):
    f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_rel", "hn_eq", "aempty_2"]) for d in fd for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
    d = sc.filter(pl.col("p2") >= 0.3).join(f, on=["s1", "t"], how="left").join(at.rename({"id": "t", "hn_h": "hn_t"}), on="t", how="left")
    d = d.with_columns(((pl.col("hn_eq") == 1) & (pl.col("p2") >= 0.7)).any().over("s1").alias("eqacc"),
                       (pl.len().over("s1", "hn_t") - 1).alias("clus"))
    d = d.filter(pl.col("hn_rel").is_in([5, 7])).join(s1c, on="s1")
    return d.with_columns(pl.when(pl.col("p2") >= 0.95).then(pl.lit("hi")).when(pl.col("p2") >= 0.7).then(pl.lit("acc_mid")).otherwise(pl.lit("rej")).alias("b"),
                          (pl.col("clus") >= 1).alias("cl"))
s1tr = pl.read_parquet("work/p_train_s1.parquet", columns=["id", "country"]).with_columns((pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
keep = pl.read_parquet("work/keep_trainD.parquet").rename({"id": "s1"})
tr = prep(pl.read_parquet(trp, columns=["s1", "t", "y", "p2"]), ["work/feat_trainD", "work/feat_trainD_dense"], at_tr, s1tr.join(keep, on="s1"))
te = prep(pl.read_parquet(tep, columns=["s1", "t", "p2"]), ["work/feat_test", "work/feat_test_dense"], at_te, s1te)
ntr = dict(keep.join(s1tr, on="s1").group_by("country").len().iter_rows()); nte = dict(s1te.group_by("country").len().iter_rows())
a = tr.group_by("country", "b", "eqacc", "cl").agg(pl.len().alias("ntr"), pl.col("y").mean().alias("prec_tr"), pl.col("y").sum().alias("pos"))
a = a.with_columns((pl.col("ntr") * 1000 / pl.col("country").replace_strict(ntr)).alias("tr1k"), (pl.col("pos") * 1000 / pl.col("country").replace_strict(ntr)).alias("pos1k"))
b = te.group_by("country", "b", "eqacc", "cl").agg(pl.len().alias("nte"))
b = b.with_columns((pl.col("nte") * 1000 / pl.col("country").replace_strict(nte)).alias("te1k"))
m = b.join(a, on=["country", "b", "eqacc", "cl"], how="left").with_columns((pl.col("pos1k") / pl.col("te1k")).alias("prec_te_est"))
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
print(m.sort("country", "b", "eqacc", "cl").select("country", "b", "eqacc", "cl", "nte", "tr1k", "te1k", "prec_tr", "prec_te_est"))
