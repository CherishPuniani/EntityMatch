"""test CE = mean of the two half-models (m0 from the shipped run, m1 scored in this session)."""
import polars as pl
a = pl.read_parquet("work/ce/ce_test_qwen_dense_s1.parquet")            # s1, t, ce  (model h0)
b = pl.read_parquet("work2/sc_test_final_m1.parquet").rename({"ce0": "ce1"})  # s1, t, ce1 (model h1)
m = a.join(b, on=["s1", "t"], how="inner"); assert len(m) == len(a) == len(b), (len(a), len(b), len(m))
print("corr m0/m1", m.select(pl.corr("ce", "ce1")).item(), "mean", m["ce"].mean(), m["ce1"].mean(), "std", m["ce"].std(), m["ce1"].std())
m.select("s1", "t", ((pl.col("ce") + pl.col("ce1")) / 2).alias("ce")).write_parquet("work2/ce_test_avg.parquet")
