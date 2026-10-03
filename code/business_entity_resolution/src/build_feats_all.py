"""Compute pair features for all S1 of a split, chunked to disk."""
import polars as pl, numpy as np, sys, time, os
sys.path.insert(0, "work")
import features as F
split = sys.argv[1]; Kuse = int(sys.argv[2]); outdir = sys.argv[3]
wid = int(sys.argv[4]) if len(sys.argv) > 4 else 0; nw = int(sys.argv[5]) if len(sys.argv) > 5 else 1
os.makedirs(outdir, exist_ok=True)
t0 = time.time()
ptag = os.environ.get("PTAG", split)
pairs = pl.read_parquet(f"work/pairs_{ptag}.parquet")
if os.environ.get("NOFILTER") != "1":          # NOFILTER=1: extra candidate channels without a blocking rank
    pairs = pairs.filter(pl.col("rrank") <= Kuse)
pairs = pairs.sort("s1")
print("pairs", len(pairs), flush=True)
R1 = F.rec_table(split, 1)
RT = pl.concat([F.rec_table(split, 2), F.rec_table(split, 3)])
nidf, aidf, dflt = F.idf_tables(RT)
print("tables", time.time() - t0, flush=True)
CH = 2_000_000
for j, i in enumerate(range(0, len(pairs), CH)):
    fn = f"{outdir}/part_{j:04d}.parquet"
    if os.path.exists(fn) or j % nw != wid:
        continue
    p = pairs.slice(i, CH)
    f = F.compute(p, R1, RT, nidf, aidf, dflt)
    EXTRA = ["t_ns1", "t_best", "t_rank", "t_bbest", "t_brank", "rs", "rrank", "npool", "wmax"] + [f"w{k}" for k in range(8)]
    f = f.join(p.select(["s1", "t"] + EXTRA), on=["s1", "t"], how="left")
    f = f.with_columns([pl.col(c).cast(pl.Float32) for c, dt in f.schema.items() if c not in ("s1", "t")])
    f.write_parquet(fn + ".tmp"); os.rename(fn + ".tmp", fn)
    print(j, i, time.time() - t0, flush=True)
print("done", time.time() - t0, flush=True)
