"""Extended retrieval feasibility (train): for true pairs missed by blocking (non-empty target address), how many share
(country, house number, >=1 non-numeric address token) with the S1; pool sizes of the key (country, hn, token)."""
import polars as pl
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
keep = pl.read_parquet(f"{R}/work/keep_trainD.parquet").rename({"id": "s1"})
gt = pl.read_parquet(f"{R}/work/gt_pairs_int.parquet").join(keep, on="s1")
cand = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t"])
miss = gt.join(cand, on=["s1", "t"], how="anti")
P = pl.concat([pl.read_parquet(f"{R}/work/p_train_s{s}.parquet", columns=["id", "country", "acore"]).select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "country",
    pl.col("acore").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first().alias("hn"),
    pl.col("acore").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$") & (pl.element().str.len_chars() >= 3))).list.unique().alias("toks")) for s in (1, 2, 3)])
m = miss.join(P.rename({"id": "s1", "hn": "hn1", "toks": "k1"}), on="s1").join(P.select(pl.col("id").alias("t"), pl.col("hn").alias("hn2"), pl.col("toks").alias("k2")), on="t")
m = m.filter(pl.col("k2").list.len() > 0)
m = m.with_columns((pl.col("hn1") == pl.col("hn2")).fill_null(False).alias("hn_eq"), pl.col("k1").list.set_intersection("k2").list.len().alias("ntok"))
print("missed non-empty:", m.group_by("country").agg(pl.len(), pl.col("hn_eq").mean().round(3), (pl.col("hn_eq") & (pl.col("ntok") >= 1)).mean().round(3).alias("hn+tok"),
                                                     (pl.col("hn_eq") & (pl.col("ntok") >= 2)).mean().round(3).alias("hn+2tok")).rows())
# pool sizes: targets per (country, hn, token)
T = P.filter(pl.col("id") >= 20_000_000_000).filter(pl.col("hn").is_not_null()).explode("toks").drop_nulls("toks")
pool = T.group_by("country", "hn", "toks").len()
print("targets per (hn, token) key: ", pool["len"].describe().rows())
S = P.filter(pl.col("id") < 20_000_000_000).join(keep.rename({"s1": "id"}), on="id").filter(pl.col("hn").is_not_null()).explode("toks").drop_nulls("toks")
pairs = S.join(pool, on=["country", "hn", "toks"]).group_by("id").agg(pl.col("len").sum())
print("S1 x key pool pairs (upper bound, with duplicates):", pairs["len"].sum(), "| per S1 median", pairs["len"].median())
