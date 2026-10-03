"""Score test pairs with the no-block stage-1 model and one or two stage-2 models (tree-only, and optionally with
the three CE blocks). Run from a workspace whose work/ holds the features and rec_attrs to use.
env: FEAT, EXTRA_PAIRFEATS, S1DIR (no-block models), S2DIRS (comma list of stage-2 model dirs), CE_SCORES,
CE2_SCORES, CE3_SCORES, OUTS (comma list of output parquet paths, one per S2DIR)"""
import polars as pl, numpy as np, lightgbm as lgb, sys, os, pickle, time, gc
sys.path.insert(0, "work")
os.environ.setdefault("TAG", "predict")
import stages as S
from train_stage2_ce import ce_context, CE_COLS
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
f = S.stage1_context(S.load_feats(os.environ["FEAT"], "test")); log(f"loaded {f.shape}")
c1 = pickle.load(open(f"{os.environ['S1DIR']}/cols.pkl", "rb"))
X = f.select([pl.col(c).cast(pl.Float32) for c in c1["F1"]]).to_numpy()
p1 = np.mean([lgb.Booster(model_file=f"{os.environ['S1DIR']}/s1_f{k}.txt").predict(X, num_threads=30) for k in range(4)], axis=0)
del X; gc.collect(); f = f.with_columns(pl.Series("p1", p1.astype(np.float32))); log("stage 1")
base = f
for d, out in zip(os.environ["S2DIRS"].split(","), os.environ["OUTS"].split(",")):
    cols = pickle.load(open(f"{d}/cols.pkl", "rb")); g = base
    if "ce_x" in cols["F2"]:
        g = ce_context(g.join(pl.read_parquet(os.environ["CE2_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left")).rename({c: c + "_x" for c in CE_COLS})
    if "ce_b" in cols["F2"]:
        g = ce_context(g.join(pl.read_parquet(os.environ["CE3_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left")).rename({c: c + "_b" for c in CE_COLS})
    if "ce" in cols["F2"]:
        g = ce_context(g.join(pl.read_parquet(os.environ["CE_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
    g = S.stage2_context(g, "p1")
    X = g.select([pl.col(c).cast(pl.Float32) for c in cols["F2"]]).to_numpy()
    p2 = np.mean([lgb.Booster(model_file=f"{d}/s2_f{k}.txt").predict(X, num_threads=30) for k in range(4)], axis=0)
    g.select("s1", "t", "p1").with_columns(pl.Series("p2", p2.astype(np.float32))).write_parquet(out); log(f"wrote {out}")
    del X, g; gc.collect()
