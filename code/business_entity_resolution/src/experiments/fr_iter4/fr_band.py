"""Random French pairs by decision band (final v3): rejected-uncertain (0.1<=p2n<0.7) and accepted-low (0.7<=p2n<0.95)."""
import polars as pl, sys
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
LO, HI, N, SEED = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
COUNTRY = sys.argv[5] if len(sys.argv) > 5 else "France"
recs = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == COUNTRY).select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name", "business_address")
    for s in (1, 2, 3)])
fr = recs.filter(pl.col("id") < 20_000_000_000).select(pl.col("id").alias("s1"))
sc = pl.read_parquet(f"{R}/output_final_v3/scores.parquet").join(fr, on="s1")
sc = sc.with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
x = sc.filter((pl.col("p2n") >= LO) & (pl.col("p2n") < HI))
print(f"{COUNTRY}: pairs with {LO}<=p2n<{HI}: {len(x)} ({len(x) * 1000 / len(fr):.1f} per 1k S1)")
info = {r[0]: (r[1], r[2]) for r in recs.iter_rows()}
for s1, t, p2, p2n, rt in x.sample(min(N, len(x)), seed=SEED).select("s1", "t", "p2", "p2n", "rt").iter_rows():
    a, b = info[s1], info.get(t, ("?", "?"))
    print(f"p2={p2:.3f} p2n={p2n:.3f} rt={rt}\n   S1 {a[0][:40]:40s} | {a[1][:75]}\n   T  {b[0][:40]:40s} | {b[1][:75]}")
