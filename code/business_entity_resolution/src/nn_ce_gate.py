"""Selective CE scoring: gate candidate pairs on the baseline stage-1 probability, then assemble OOF CE scores.

usage: nn_ce_gate.py make <lo> <hi> <train_p1.parquet> <test_p1.parquet> <outdir>
         writes <outdir>/gate_train_h{0,1}.parquet (train pairs of S1s in half h with lo <= OOF p1 <= hi) and
         <outdir>/gate_test.parquet (test pairs, fold-averaged p1 in the same band). Same rule for train and test.
       nn_ce_gate.py assemble <outdir> <tag>
         train: half-h pairs take the score of the CE trained on half 1-h  -> <outdir>/ce_train_<tag>.parquet
         test : mean of the available CEs (sc_test_<tag>_m0/_m1)            -> <outdir>/ce_test_<tag>.parquet
"""
import sys, polars as pl

mode = sys.argv[1]
if mode == "make":
    lo, hi, trp, tep, out = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    gtp = pl.read_parquet("work/gt_pairs_int.parquet").with_columns(pl.lit(1).cast(pl.Int8).alias("y"))
    oof = pl.read_parquet(trp, columns=["s1", "t", "p1"]).join(gtp, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    oof = oof.with_columns(((pl.col("s1").hash(seed=11) % 4) // 2).cast(pl.Int8).alias("half"))
    band = (pl.col("p1") >= lo) & (pl.col("p1") <= hi)
    gt = oof.filter(band)
    for h in (0, 1):
        gt.filter(pl.col("half") == h).select("s1", "t", "y").write_parquet(f"{out}/gate_train_h{h}.parquet")
    te = pl.read_parquet(tep, columns=["s1", "t", "p1"]).filter(band)
    te.select("s1", "t").write_parquet(f"{out}/gate_test.parquet")
    pos = oof.filter(pl.col("y") == 1)
    print(f"gate [{lo},{hi}]: train {len(gt)}/{len(oof)} pairs ({len(gt)/len(oof):.4f}), positives in band "
          f"{pos.filter(band).height/len(pos):.4f}; test {len(te)} pairs")
elif mode == "assemble":
    import os
    out, tag = sys.argv[2], sys.argv[3]

    def put(df, path):                     # atomic write: readers never see a partial file
        df.write_parquet(path + ".tmp"); os.replace(path + ".tmp", path)
    tr = pl.concat([pl.read_parquet(f"{out}/sc_train_h{h}_{tag}.parquet").select("s1", "t", pl.col("ce0").alias("ce"))
                    for h in (0, 1)])
    put(tr, f"{out}/ce_train_{tag}.parquet")
    te = [pl.read_parquet(f"{out}/sc_test_{tag}_m{k}.parquet").rename({"ce0": f"c{k}"}) for k in (0, 1)
          if os.path.exists(f"{out}/sc_test_{tag}_m{k}.parquet")]
    if not te:
        print("train", len(tr), "(no test scores yet)"); sys.exit(0)
    t = te[0] if len(te) == 1 else te[0].join(te[1], on=["s1", "t"])
    cs = [c for c in t.columns if c.startswith("c") and c[1:].isdigit()]
    put(t.select("s1", "t", (pl.sum_horizontal(cs) / len(cs)).alias("ce")), f"{out}/ce_test_{tag}.parquet")
    print("train", len(tr), "test", len(t), "test models", len(cs),
          "agreement corr" if len(cs) == 2 else "", t.select(pl.corr("c0", "c1")).item() if len(cs) == 2 else "")
