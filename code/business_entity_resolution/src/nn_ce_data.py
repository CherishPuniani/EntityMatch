"""Build cross-encoder training / eval pair sets from the blocking candidates (hard negatives), cross-fitted in two
halves aligned with the stage models' folds: half = (hash(s1, seed=11) % 4) // 2, so a CE trained on half h never
saw labels of the S1s it later scores (half 1-h).

Per sampled S1: up to POS_MAX true candidates, the NEG_TOP highest-ranked negatives by blocking-ranker score, and
NEG_RAND random lower-ranked negatives. Negatives are real retrieved confusers, never random targets.
usage: nn_ce_data.py <pairs_trainD.parquet> <outdir> <n_s1_per_half> [eval_s1_per_half]
"""
import sys, polars as pl, numpy as np

POS_MAX, NEG_TOP, NEG_RAND = 4, 6, 1
pairs = pl.read_parquet(sys.argv[1], columns=["s1", "t", "rs", "rrank"])
out = sys.argv[2]; n_s1 = int(sys.argv[3]); n_ev = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
gt = pl.read_parquet("work/gt_pairs_int.parquet").with_columns(pl.lit(1).cast(pl.Int8).alias("y"))
pairs = pairs.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
pairs = pairs.with_columns(((pl.col("s1").hash(seed=11) % 4) // 2).cast(pl.Int8).alias("half"),
                           (pl.col("s1").hash(seed=5) % 1_000_003).alias("h5"),
                           (pl.struct("s1", "t").hash(seed=6) % 1_000_003).alias("h6"))
s1s = pairs.select("s1", "half", "h5").unique("s1")
for h in (0, 1):
    pool = s1s.filter(pl.col("half") == h).sort("h5")
    tr_ids = pool.head(n_s1).select("s1")
    ev_ids = pool.tail(n_ev).select("s1")          # disjoint from training S1s; used only for learning curves
    p = pairs.join(tr_ids, on="s1")
    neg = p.filter(pl.col("y") == 0).with_columns(pl.col("rrank").rank("ordinal").over("s1").alias("nr"))
    sel = pl.concat([
        p.filter(pl.col("y") == 1).sort("h6").group_by("s1", maintain_order=True).head(POS_MAX),
        neg.filter(pl.col("nr") <= NEG_TOP).drop("nr"),
        neg.filter(pl.col("nr") > NEG_TOP).sort("h6").group_by("s1", maintain_order=True).head(NEG_RAND).drop("nr")])
    sel = sel.select("s1", "t", "y").sort(pl.struct("s1", "t").hash(seed=7))
    ev = pairs.join(ev_ids, on="s1").filter(pl.col("rrank") <= 10).select("s1", "t", "y")
    sel.write_parquet(f"{out}/train_h{h}.parquet"); ev.write_parquet(f"{out}/eval_h{h}.parquet")
    print(f"half {h}: train S1 {len(tr_ids)} pairs {len(sel)} pos {sel['y'].mean():.3f} | eval S1 {len(ev_ids)} pairs {len(ev)} pos {ev['y'].mean():.3f}")
