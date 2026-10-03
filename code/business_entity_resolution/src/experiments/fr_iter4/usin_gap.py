"""US/India: per class (address stratum x name class), links per 1k S1: train truth, OOF accepted (p2>=0.7), test accepted (v3).
Test deviating from OOF = test-specific behaviour (shift); OOF < truth = in-train misses."""
import polars as pl, glob
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
FC = ["s1", "t", "hn_rel", "hn_eq", "aempty_2", "cs_tset"]


def cls(df):
    return df.with_columns(pl.concat_str([
        pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq"))
        .when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum"))
        .when(pl.col("hn_rel").is_in([5, 7])).then(pl.lit("adj")).otherwise(pl.lit("othnum")),
        pl.when(pl.col("cs_tset") >= 99.9).then(pl.lit("n=")).when(pl.col("cs_tset") >= 80).then(pl.lit("n~")).otherwise(pl.lit("n≠"))], separator="|").alias("c"))


def feats(dirs, keys):
    return pl.concat([pl.read_parquet(p, columns=FC).join(keys, on=["s1", "t"], how="semi") for d in dirs for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])


if __name__ == "__main__":
    o = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y", "p2"]).filter((pl.col("y") == 1) | (pl.col("p2") >= 0.7))
    ctr = pl.read_parquet(f"{R}/work/p_train_s1.parquet", columns=["id", "country"]).select((pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")
    keep = pl.read_parquet(f"{R}/work/keep_trainD.parquet").rename({"id": "s1"})
    ntr = dict(keep.join(ctr, on="s1").group_by("country").len().iter_rows())
    o = cls(o.join(ctr, on="s1").join(feats(["work/feat_trainD", "work/feat_trainD_dense"], o.select("s1", "t")), on=["s1", "t"], how="left"))
    tr = o.group_by("country", "c").agg((pl.col("y").sum()).alias("truth"), (pl.col("p2") >= 0.7).sum().alias("oof_acc"),
                                         ((pl.col("p2") >= 0.7) & (pl.col("y") == 1)).sum().alias("oof_tp"))
    m = pl.read_csv(f"{R}/output_final_v3/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    L = (m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "").select(
        (pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"),
        (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000) + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))
    c = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")
    nte = dict(c.group_by("country").len().iter_rows())
    L = cls(L.join(c, on="s1").filter(pl.col("country") != "France").join(feats(["work/feat_test", "work/feat_test_dense"], L), on=["s1", "t"], how="left"))
    te = L.group_by("country", "c").len().rename({"len": "test_acc"})
    t = tr.join(te, on=["country", "c"], how="full", coalesce=True).fill_null(0).with_columns(
        *[(pl.col(x) * 1000 / pl.col("country").replace_strict(ntr, default=1)).round(1) for x in ["truth", "oof_acc", "oof_tp"]],
        (pl.col("test_acc") * 1000 / pl.col("country").replace_strict(nte, default=1)).round(1))
    t = t.with_columns((pl.col("test_acc") - pl.col("oof_acc")).round(1).alias("test-oof"), (pl.col("oof_acc") - pl.col("oof_tp")).round(1).alias("oof_fp"),
                       (pl.col("truth") - pl.col("oof_tp")).round(1).alias("oof_fn"))
    print(t.filter((pl.col("truth") > 1) | (pl.col("test_acc") > 1)).sort("country", "c"))
