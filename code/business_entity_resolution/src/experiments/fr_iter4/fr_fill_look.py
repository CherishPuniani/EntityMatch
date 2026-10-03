"""Random rejected same-address filler pairs (v3) for groupe / developpement / france, with the S1's accepted links."""
import polars as pl, sys
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
import filler_train as ft
R = ft.R
TOKS = sys.argv[1].split(","); N = int(sys.argv[2]); SEED = int(sys.argv[3])
x = ft.load("test", "France", f"{R}/output_final_v3/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n")
x = x.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
rej = x.filter(pl.col("add1").is_in(TOKS) & (pl.col("p") < 0.7))
recs = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name", "business_address") for s in (1, 2, 3)])
info = {r[0]: (r[1], r[2]) for r in recs.iter_rows()}
sc = pl.read_parquet(f"{R}/output_final_v3/scores.parquet").filter(pl.col("p2n") >= 0.7)
for s1, t, p, rt in rej.sample(N, seed=SEED).select("s1", "t", "p", "rt").iter_rows():
    a, b = info[s1], info[t]
    acc = sc.filter(pl.col("s1") == s1)["t"].to_list()
    print(f"p2n={p:.3f} rt={rt}  S1 {a[0][:36]:36s} | {a[1][:60]}\n                  T  {b[0][:36]:36s} | {b[1][:60]}")
    for tt in acc[:4]:
        c = info.get(tt, ("?", "?"))
        print(f"                  ✓  {c[0][:36]:36s} | {c[1][:60]}")
