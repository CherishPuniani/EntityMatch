import polars as pl, glob
SP = "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad"; R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
sc = pl.read_parquet(f"{SP}/fr6/output_fr6a/scores.parquet", columns=["s1", "t", "p2n"]).join(fr, on="s1").with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
acc = sc.filter((pl.col("p2n") >= 0.9) & (pl.col("p2n") < 0.999) & (pl.col("rt") == 1))
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_eq", "aempty_2", "cs_tset", "acs_tset"]).join(acc.select("s1", "t"), on=["s1", "t"], how="semi") for d in [f"{SP}/fr6/work_unseen/feat_testun", f"{SP}/fr6/work_unseen/feat_testun_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
x = acc.join(f, on=["s1", "t"]).filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1) & (pl.col("cs_tset") < 80))
raw = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name", "business_address") for s in (1, 2, 3)])
info = {r[0]: (r[1], r[2]) for r in raw.iter_rows()}
print("accepted eq|n≠ 0.9-0.999:", x.height)
for s1, t, p in x.sample(14, seed=8).select("s1", "t", "p2n").iter_rows():
    a, b = info[s1], info[t]; print(f"{p:.3f} S1 {a[0][:34]:34s} | {a[1][:50]}\n      T  {b[0][:34]:34s} | {b[1][:50]}")
