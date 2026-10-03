"""Stage 2 with TWO cross-encoders (Qwen3-Reranker-0.6B and XLM-R-base), within-S1 features of each (XLM-R block
suffixed _x). Same OOF p1, folds, subsample, hyper-parameters and seed as train_stage2_ce.py; outputs go to work2/.
env: BASE_TAG, TAG, CE_SCORES (qwen, s1,t,ce), CE2_SCORES (xlmr, s1,t,ce), SUB, SEED, NR2, FEAT, ATTRS, EXTRA_PAIRFEATS"""
import polars as pl, numpy as np, lightgbm as lgb, sys, time, os, pickle, gc
sys.path.insert(0, "work")
import stages as S
from train_stage2_ce import ce_context, CE_COLS
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)
NF = 4
SUB = float(os.environ.get("SUB", "1.0")); NR2 = int(os.environ.get("NR2", "600")); SEED = int(os.environ.get("SEED", "1"))
TAG = os.environ["TAG"]; BASE = os.environ.get("BASE_TAG", "dense_s1")
MD = f"work2/models_{TAG}"; os.makedirs(MD, exist_ok=True)
W = CE_COLS[:5]; WX = [c + "_x" for c in W]; WB = [c + "_b" for c in W] if os.environ.get("CE3_SCORES") else []; WC = [c + "_c" for c in W] if os.environ.get("CE4_SCORES") else []
f = S.load_feats(os.environ.get("FEAT", "work/feat_trainD"), "train", os.environ.get("ATTRS", "trainD"))
f = S.stage1_context(f)
oof = pl.read_parquet(f"work/oof_{BASE}.parquet", columns=["s1", "t", "y", "fold", "p1"])
n0 = len(f)
f = f.join(oof, on=["s1", "t"], how="inner", maintain_order="left"); assert len(f) == n0 == len(oof)
f = f.with_columns(((pl.col("s1").hash(seed=23) % 1000) < SUB * 1000).alias("sub"))
# XLM-R block first (ce_context works on the column "ce"), renamed with suffix _x
f = ce_context(f.join(pl.read_parquet(os.environ["CE2_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
f = f.rename({c: c + "_x" for c in CE_COLS})
if WB:   # third block: CE trained on the gated band itself (in-band hard negatives), suffix _b
    f = ce_context(f.join(pl.read_parquet(os.environ["CE3_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
    f = f.rename({c: c + "_b" for c in CE_COLS})
if WC:   # optional fourth block: second in-band CE (different seed), suffix _c
    f = ce_context(f.join(pl.read_parquet(os.environ["CE4_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
    f = f.rename({c: c + "_c" for c in CE_COLS})
f = ce_context(f.join(pl.read_parquet(os.environ["CE_SCORES"], columns=["s1", "t", "ce"]), on=["s1", "t"], how="left", maintain_order="left"))
log(f"CE coverage qwen {f['ce'].is_not_null().mean():.4f} xlmr {f['ce_x'].is_not_null().mean():.4f}")
f = S.stage2_context(f, "p1")
y = f["y"].to_numpy(); fold = f["fold"].to_numpy(); sub = f["sub"].to_numpy()
base_cols = pickle.load(open(f"work/models_{BASE}/cols.pkl", "rb"))
F2 = list(base_cols["F2"]) + W + WX + WB + WC
log(f"loaded {f.shape}; {len(F2)} stage-2 features")
X = f.select([pl.col(c).cast(pl.Float32) for c in F2]).to_numpy()
keep = f.select("s1", "t", "y", "fold", "p1"); del f; gc.collect()
params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, num_threads=30, verbose=-1, seed=SEED, max_bin=255)
p2 = np.zeros(X.shape[0], np.float32); imp = np.zeros(len(F2))
for k in range(NF):
    tr = (fold != k) & sub
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=F2), NR2)
    p2[fold == k] = m.predict(X[fold == k], num_threads=30)
    imp += m.feature_importance("gain"); m.save_model(f"{MD}/s2_f{k}.txt"); log(f"stage2 fold {k}")
o = np.argsort(-imp)
log("stage2 importance: " + str([(F2[i], round(imp[i] / imp.sum(), 4)) for i in o[:25]]))
pickle.dump({"F1": base_cols["F1"], "F2": F2}, open(f"{MD}/cols.pkl", "wb"))
keep.with_columns(pl.Series("p2", p2)).write_parquet(f"work2/oof_{TAG}.parquet")
log("done")
