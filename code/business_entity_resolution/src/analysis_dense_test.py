"""Label-free per-country readout of the dense channel on test: how many dense-only candidates exist and how many
are accepted (p2 >= 0.7 and best S1), per country, next to the same statistics on train OOF (where labels exist).
usage: analysis_dense_test.py <test_scores_dense.parquet> <pairfeats_test.parquet> <oof_dense.parquet> <pairfeats_trainD.parquet>"""
import sys, polars as pl


def acc(sc, pf, s1file, idcol="entity_id"):
    s1 = pl.read_parquet(s1file, columns=[idcol, "country"]).select(
        (pl.col(idcol).str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")
    d = pl.read_parquet(sc).join(pl.read_parquet(pf, columns=["s1", "t", "dense_only", "dscore"]), on=["s1", "t"], how="left")
    d = d.with_columns(((pl.col("p2") >= 0.7) & (pl.col("p2").rank("ordinal", descending=True).over("t") == 1)).alias("acc"))
    d = d.join(s1, on="s1")
    g = d.group_by("country").agg(
        pl.col("s1").n_unique().alias("S1"),
        (pl.col("dense_only") == 1).sum().alias("dense_only_pairs"),
        ((pl.col("dense_only") == 1) & pl.col("acc")).sum().alias("dense_only_accepted"),
        pl.col("acc").sum().alias("all_accepted"))
    return g.with_columns((1000 * pl.col("dense_only_accepted") / pl.col("S1")).round(1).alias("dense_acc_per_1000_S1"),
                          (pl.col("dense_only_accepted") / pl.col("dense_only_pairs")).round(4).alias("dense_acc_rate")).sort("country")


with pl.Config(tbl_width_chars=200):
    print("TEST"); print(acc(sys.argv[1], sys.argv[2], "work/test_s1.parquet"))
    print("TRAIN OOF (kept S1)"); t = acc(sys.argv[3], sys.argv[4], "work/train_s1.parquet"); print(t)
    o = pl.read_parquet(sys.argv[3], columns=["s1", "t", "y", "p2"]).join(pl.read_parquet(sys.argv[4], columns=["s1", "t", "dense_only"]), on=["s1", "t"])
    o = o.filter(pl.col("dense_only") == 1).with_columns(((pl.col("p2") >= 0.7) & (pl.col("p2").rank("ordinal", descending=True).over("t") == 1)).alias("acc"))
    print(f"train dense-only accepted: {o['acc'].sum()} of which true {o.filter(pl.col('acc'))['y'].sum()} "
          f"(precision {o.filter(pl.col('acc'))['y'].mean():.4f})")
