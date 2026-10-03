"""Deterministically drop 18.6% of train S1 so that targets-per-S1 matches the test set (4.68 -> 5.75).
The dropped entities' S2/S3 records stay in the target pool as orphan clusters, as observed in test."""
import polars as pl
s = pl.read_parquet("work/p_train_s1.parquet", columns=["id"]).select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("id"))
k = s.filter((pl.col("id").hash(seed=2024) % 1000) < 814)
k.write_parquet("work/keep_trainD.parquet"); print(len(s), len(k))
