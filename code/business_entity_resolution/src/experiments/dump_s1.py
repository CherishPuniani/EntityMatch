"""Print random S1s of one country with all candidates (p2 >= 0.03) and the submitted decision. Diagnostic only."""
import polars as pl, sys
country, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sc_path = sys.argv[4] if len(sys.argv) > 4 else "output_shift_routed/scores.parquet"
def recs():
    out = []
    for s in (1, 2, 3):
        d = pl.read_parquet(f"work/test_s{s}.parquet", columns=["entity_id", "business_name", "business_address", "country"]).filter(pl.col("country") == country)
        out.append(d.select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), pl.col("business_name").alias("nm"), pl.col("business_address").alias("ad")))
    return pl.concat(out)
R = recs()
sc = pl.read_parquet(sc_path, columns=["s1", "t", "p2", "p2n"])
sc = sc.join(R.select(pl.col("id").alias("s1")), on="s1").with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
s1s = sc.select("s1").unique().sample(n, seed=seed)["s1"].to_list()
for s1 in s1s:
    c = sc.filter((pl.col("s1") == s1) & (pl.col("p2") >= 0.03)).join(R, left_on="t", right_on="id").sort("p2", descending=True)
    s = R.filter(pl.col("id") == s1)
    print(f"== {s['nm'][0]} | {s['ad'][0]}")
    for t, p2, p2n, rt, nm, ad in c.select("t", "p2", "p2n", "rt", "nm", "ad").iter_rows():
        acc = "ACC" if (p2n >= 0.7 and rt == 1) else "   "
        print(f"   {acc} p2 {p2:.3f}{'→%.3f' % p2n if abs(p2n - p2) > 1e-6 else '      '} rt{rt} {nm} | {ad}")
