"""French uncertain same-address candidates with (near-)exact core name: examples + what differs."""
import polars as pl, glob, sys
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
CLS = sys.argv[1]; N = int(sys.argv[2]); SEED = int(sys.argv[3])
sc = pl.read_parquet(f"{R}/output_final_v3/scores.parquet").rename({"p2n": "p"})
fr = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
sc = sc.join(fr, on="s1").with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
u = sc.filter((pl.col("p") >= 0.1) & (pl.col("p") < 0.7) & (pl.col("rt") == 1))
FC = ["s1", "t", "hn_rel", "hn_eq", "aempty_2", "cs_tset", "ns_tset", "acs_tset", "t_ns1", "t_best", "t_rank"]
f = pl.concat([pl.read_parquet(p, columns=FC).join(u.select("s1", "t"), on=["s1", "t"], how="semi") for d in ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"] for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])
u = u.join(f, on=["s1", "t"]).filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1))
u = u.filter(pl.col("cs_tset") >= 99.9) if CLS == "eq" else u.filter((pl.col("cs_tset") >= 80) & (pl.col("cs_tset") < 99.9))
print(CLS, "pairs:", u.height, "| p2 (before correction) mean", round(u["p2"].mean(), 3), "| acs_tset<90 share", round((u["acs_tset"] < 90).mean(), 3))
print(u.select(pl.col("acs_tset").cut([50, 80, 90, 99.9]).value_counts(sort=True)).unnest("acs_tset"))
recs = pl.concat([pl.read_parquet(f"{R}/work/test_s{s}.parquet").filter(pl.col("country") == "France").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name", "business_address") for s in (1, 2, 3)])
info = {r[0]: (r[1], r[2]) for r in recs.iter_rows()}
acc = sc.filter(pl.col("p") >= 0.7)
for s1, t, p, p2, a_ in u.sample(N, seed=SEED).select("s1", "t", "p", "p2", "acs_tset").iter_rows():
    a, b = info[s1], info[t]
    print(f"p2n={p:.3f} p2={p2:.3f} acs={a_:.0f}  S1 {a[0][:34]:34s} | {a[1][:62]}\n{'':30s}T  {b[0][:34]:34s} | {b[1][:62]}")
    for tt in acc.filter(pl.col("s1") == s1)["t"].to_list()[:3]:
        c = info.get(tt, ("?", "?")); print(f"{'':30s}✓  {c[0][:34]:34s} | {c[1][:62]}")
