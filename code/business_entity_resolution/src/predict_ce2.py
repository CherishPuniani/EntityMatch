"""Test stage-2 scores for a stage-2 model with one (train_stage2_ce.py) or two (train_stage2_ce2.py) CE blocks.
env: MODELDIR (stage-2 models + cols.pkl), BASE_TAG (stage-1 test p1 source), CE_SCORES (qwen), CE2_SCORES (xlmr,
optional), FEAT, EXTRA_PAIRFEATS, SC_OUT (s1,t,p1,p2 parquet)"""
import polars as pl, numpy as np, lightgbm as lgb, sys, os, pickle, time, gc
sys.path.insert(0, "work")
import stages as S
from train_stage2_ce import ce_context, CE_COLS
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
MD = os.environ["MODELDIR"]; BASE = os.environ.get("BASE_TAG", "dense_s1")
cols = pickle.load(open(f"{MD}/cols.pkl", "rb"))
f = S.stage1_context(S.load_feats(os.environ.get("FEAT", "work/feat_test"), "test"))
bs = pl.read_parquet(f"work/test_scores_{BASE}.parquet", columns=["s1", "t", "p1"])
n0 = len(f); f = f.join(bs, on=["s1", "t"], how="inner", maintain_order="left"); assert len(f) == n0 == len(bs)
if "ce_x" in cols["F2"]:
    f = ce_context(f.join(pl.read_parquet(os.environ["CE2_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
    f = f.rename({c: c + "_x" for c in CE_COLS})
if "ce_b" in cols["F2"]:
    f = ce_context(f.join(pl.read_parquet(os.environ["CE3_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
    f = f.rename({c: c + "_b" for c in CE_COLS})
if "ce_c" in cols["F2"]:
    f = ce_context(f.join(pl.read_parquet(os.environ["CE4_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
    f = f.rename({c: c + "_c" for c in CE_COLS})
f = ce_context(f.join(pl.read_parquet(os.environ["CE_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
log(f"CE coverage {f['ce'].is_not_null().mean():.4f}")
f = S.stage2_context(f, "p1")
res = f.select("s1", "t", "p1")
f = f.select([pl.col(c).cast(pl.Float32) for c in cols["F2"]]); gc.collect()
models = [lgb.Booster(model_file=f"{MD}/s2_f{k}.txt") for k in range(4)]
p2 = np.zeros(len(f), np.float32); CH = 5_000_000
for i in range(0, len(f), CH):
    X = f.slice(i, CH).to_numpy()
    p2[i:i + CH] = np.mean([m.predict(X, num_threads=30) for m in models], axis=0)
res.with_columns(pl.Series("p2", p2)).write_parquet(os.environ["SC_OUT"])
log("done")
