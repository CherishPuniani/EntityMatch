"""Dense-retrieval candidate channel (Qwen3-Embedding) on top of lexical blocking.

usage:
  dense_cands.py unassigned <split> <scores.parquet: s1,t,p2> <out.parquet>
      targets NOT assigned by the base model (assigned = p2 >= 0.7 and this S1 is the target's best S1). The truth
      is a partition, so a true target missed by blocking is (almost) never one confidently owned by another S1;
      indexing only unassigned targets cuts encoding cost ~60 % and deepens the effective K. Train uses OOF p2,
      test uses test p2 — the same rule on both sides.
  dense_cands.py build <split> <search.parquet: s1,t,dscore,drank> <blocking pairs.parquet> <K> <tag>
      -> work/pairs_<tag>_dense.parquet : new pairs (dense rank <= K, not in blocking), blocking columns null
      -> work/dense/pairfeats_<tag>.parquet : s1, t, dscore, drank, dense_only for every union pair
"""
import sys, polars as pl

mode = sys.argv[1]
if mode == "unassigned":
    split, sc, out = sys.argv[2], sys.argv[3], sys.argv[4]
    d = pl.read_parquet(sc, columns=["s1", "t", "p2"])
    asg = d.with_columns(pl.col("p2").rank("ordinal", descending=True).over("t").alias("rt")).filter(
        (pl.col("p2") >= 0.7) & (pl.col("rt") == 1)).select(pl.col("t").alias("id")).unique()
    tg = pl.concat([pl.read_parquet(f"work/{split}_s{s}.parquet", columns=["entity_id"]).select(
        (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id")) for s in (2, 3)])
    un = tg.join(asg, on="id", how="anti")
    un.write_parquet(out)
    print(f"{split}: targets {len(tg)} assigned {len(asg)} unassigned {len(un)} ({len(un)/len(tg):.3f})")
elif mode == "build":
    split, srch, bp, K, tag = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
    s = pl.read_parquet(srch)
    bl = pl.read_parquet(bp, columns=["s1", "t"])
    s = s.join(bl.select("s1").unique(), on="s1")                 # only S1s of this candidate population
    new = s.filter(pl.col("drank") <= K).join(bl, on=["s1", "t"], how="anti")
    F32 = ["bscore", "nkeys", "wmax"] + [f"w{k}" for k in range(8)] + ["brank", "npool", "brel", "rs", "rrank",
                                                                          "t_ns1", "t_best", "t_rank", "t_bbest", "t_brank"]
    new.select("s1", "t").with_columns([pl.lit(None, pl.Float32).alias(c) for c in F32]).write_parquet(
        f"work/pairs_{tag}_dense.parquet")
    pf = pl.concat([bl.join(s.select("s1", "t", "dscore", "drank"), on=["s1", "t"], how="left").with_columns(
                        pl.lit(0, pl.Int8).alias("dense_only")),
                    new.select("s1", "t", "dscore", "drank").with_columns(pl.lit(1, pl.Int8).alias("dense_only"))])
    pf = pf.with_columns(pl.col("dscore").cast(pl.Float32), pl.col("drank").cast(pl.Float32), pl.col("dense_only").cast(pl.Float32))
    pf.write_parquet(f"work/dense/pairfeats_{tag}.parquet")
    print(f"{tag}: blocking pairs {len(bl)} | dense hits on blocking pairs {pf['dscore'].is_not_null().sum() - len(new)} "
          f"| new pairs {len(new)} ({len(new)/bl['s1'].n_unique():.2f} per S1)")
    if split == "train":
        gt = pl.read_parquet("work/gt_pairs_int.parquet").join(bl.select("s1").unique(), on="s1")
        a = len(gt.join(bl, on=["s1", "t"])); b = len(gt.join(new.select("s1", "t"), on=["s1", "t"]))
        print(f"true pairs {len(gt)}: blocking {a} ({a/len(gt):.4f}) + dense-new {b} -> union recall {(a+b)/len(gt):.4f}")
