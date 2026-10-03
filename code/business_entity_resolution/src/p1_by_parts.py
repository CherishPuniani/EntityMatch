"""Memory-light stage-1 scoring from saved fold models, streaming feature parts.
Feature parts are slices of s1-sorted pairs, and every stage-1 context feature is a window over s1 (or pair-local),
so processing whole-S1 groups part by part is exact; rows of the S1 that straddles a part boundary are carried over.
train: each row scored by the model of its own fold (OOF p1, identical to train_stages.py);
test : mean of the 4 fold models (identical to predict_test.py).
usage: p1_by_parts.py <train|test> <featdir> <attrs_tag> <model_tag> <out.parquet>"""
import sys, glob, polars as pl, numpy as np, lightgbm as lgb
sys.path.insert(0, "work")
import stages as S

split, featdir, attrs, tag, out = sys.argv[1:6]
at = pl.read_parquet(f"work/rec_attrs_{attrs}.parquet")
models = [lgb.Booster(model_file=f"work/models_{tag}/s1_f{k}.txt") for k in range(4)]
F1 = models[0].feature_name()
res = []; carry = None
parts = sorted(glob.glob(f"{featdir}/part_*.parquet"))
for i, p in enumerate(parts):
    f = pl.read_parquet(p)
    if carry is not None:
        f = pl.concat([carry, f])
    if i < len(parts) - 1:
        last = f["s1"][-1]
        carry = f.filter(pl.col("s1") == last); f = f.filter(pl.col("s1") != last)
    f = f.join(at.rename({c: c + "_s" for c in S.ATTR}), left_on="s1", right_on="id", how="left", maintain_order="left")
    f = f.join(at.rename({c: c + "_t" for c in S.ATTR}), left_on="t", right_on="id", how="left", maintain_order="left")
    f = S.stage1_context(f)
    X = f.select(F1).to_numpy().astype(np.float32)
    if split == "train":
        fold = (f["s1"].hash(seed=11) % 4).to_numpy()
        p1 = np.zeros(len(f), np.float32)
        for k in range(4):
            mk = fold == k
            p1[mk] = models[k].predict(X[mk], num_threads=8)
    else:
        p1 = np.mean([m.predict(X, num_threads=8) for m in models], axis=0).astype(np.float32)
    res.append(f.select("s1", "t").with_columns(pl.Series("p1", p1)))
    print(i, len(f), flush=True)
pl.concat(res).write_parquet(out)
print("done", sum(len(r) for r in res))
