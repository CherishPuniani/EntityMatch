"""Evaluate a dense-retrieval probe against the lexical blocking candidates on the same query S1s.
usage: analysis_dense.py <probe.parquet: s1,t,dscore,drank> <pairs_trainD.parquet> [miss_trainD.parquet]"""
import sys, polars as pl

pr = pl.read_parquet(sys.argv[1])
q = pr.select("s1").unique()
bl = pl.read_parquet(sys.argv[2], columns=["s1", "t", "rrank"]).join(q, on="s1")
gt = pl.read_parquet("work/gt_pairs_int.parquet").join(q, on="s1")
j = gt.join(bl, on=["s1", "t"], how="left").join(pr.select("s1", "t", "drank"), on=["s1", "t"], how="left")
G = len(gt)
print(f"queries {len(q)}  true pairs {G}  blocking@30 recall {j['rrank'].is_not_null().mean():.4f}")
for K in (1, 5, 10, 20, 50, 100):
    d = (j["drank"] <= K).fill_null(False)          # nulls = not retrieved (polars mean() would skip them)
    u = j["rrank"].is_not_null() | d
    newc = pr.filter(pl.col("drank") <= K).join(bl, on=["s1", "t"], how="anti")
    print(f"K={K:3d}: dense recall {d.mean():.4f} | union recall {u.mean():.4f} (+{u.mean() - j['rrank'].is_not_null().mean():.4f}) "
          f"| new candidates/S1 {len(newc)/len(q):.1f} | misses recovered {(d & j['rrank'].is_null()).sum()}/{j['rrank'].is_null().sum()}")
if len(sys.argv) > 3:
    m = pl.read_parquet(sys.argv[3], columns=["s1", "t", "cat"]).join(q, on="s1").join(pr.select("s1", "t", "drank"), on=["s1", "t"], how="left")
    print(m.group_by("cat").agg(pl.len().alias("missed"), (pl.col("drank") <= 10).sum().alias("dense@10"),
                                (pl.col("drank") <= 50).sum().alias("dense@50"), (pl.col("drank") <= 100).sum().alias("dense@100"))
          .sort("missed", descending=True))
