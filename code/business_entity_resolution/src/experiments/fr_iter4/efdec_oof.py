"""OOF validation of expected-F0.5 decoding vs threshold 0.7 (exact per-entity F0.5 of compare_oof.py, paired bootstrap)."""
import sys, os, time, numpy as np, polars as pl
os.chdir("/home2/home/amritanshu_t/amazon-mlc-26/run"); sys.path.insert(0, "work")
os.environ.setdefault("KEEP", "work/keep_trainD.parquet")
sys.argv = [sys.argv[0], "work2/oof_ce4q.parquet:p2", "work2/oof_ce4q.parquet:p2"]
import compare_oof as C
sys.path.insert(0, "/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad")
import efdec
df = C.load("work2/oof_ce4q.parquet:p2")
da = C.per_entity(df, 0.7)
print("threshold 0.7 F05", round(da["f"].mean(), 5), da.group_by("country").agg(pl.col("f").mean()).sort("country").rows())
for w in [float(x) for x in os.environ.get("WS", "1.0").split(",")]:
    t0 = time.time()
    sel = efdec.decode(df.select("s1", "t", "p"), w)
    x = df.join(sel.with_columns(pl.lit(1.0).alias("q")), on=["s1", "t"], how="left").with_columns(pl.col("q").fill_null(0.0)).drop("p").rename({"q": "p"})
    db = C.per_entity(x, 0.5)
    diff = (db["f"] - da["f"]).to_numpy(); rng = np.random.default_rng(0); n = len(diff)
    bs = [diff[rng.integers(0, n, n)].mean() for _ in range(200)]
    print(f"w={w} EF-decode F05 {db['f'].mean():.5f} delta {diff.mean():+.5f} CI [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}] "
          f"links {len(sel)} vs {int((da['k']).sum())} | by country {db.group_by('country').agg(pl.col('f').mean()).sort('country').rows()} ({time.time() - t0:.0f}s)")
