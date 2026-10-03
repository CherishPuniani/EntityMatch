"""Branch-sibling vocabulary veto (re-implementation of the team's v5 idea, SOLUTION_REPORT §26.4).
Among labelled train candidate pairs whose S1 core name is fully contained in the target (core_covA >= 0.99) and whose
target has an address, collect the target's extra core tokens (absent from the S1 name). Tokens seen >= MINC times
with a true-match rate < MAXR form the vocabulary. Veto: drop a predicted link if the S1 name is contained, the target
has an address and adds a vocabulary token.
usage: sibvocab.py learn <frac 0..1 of S1 used (hash seed 77)> <out.parquet>
       sibvocab.py eval <vocab.parquet> <oof.parquet>          (held-out S1 = those not used for learning)
       sibvocab.py flags <vocab.parquet> test <out.parquet>    (s1, t pairs of test that the rule would veto)"""
import polars as pl, glob, sys, os
MINC = int(os.environ.get("MINC", "800")); MAXR = float(os.environ.get("MAXR", "0.005"))
def toks(split):
    out = []
    for s in (1, 2, 3):
        d = pl.read_parquet(f"work/p_{split}_s{s}.parquet", columns=["id", "nt", "core"])
        out.append(d.select((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "nt", "core"))
    return pl.concat(out)
def cand(split, fd):
    f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "core_covA", "aempty_2"]) for d in fd for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
    f = f.filter((pl.col("core_covA") >= 0.99) & (pl.col("aempty_2") == 0)).select("s1", "t")
    T = toks(split)
    f = f.join(T.select(pl.col("id").alias("s1"), pl.col("nt").alias("nt1")), on="s1").join(T.select(pl.col("id").alias("t"), pl.col("core").alias("c2")), on="t")
    return f.with_columns(pl.col("c2").list.set_difference("nt1").alias("extra")).select("s1", "t", "extra")
mode = sys.argv[1]
TR = ["work/feat_trainD", "work/feat_trainD_dense"]; TE = ["work/feat_test", "work/feat_test_dense"]
if mode == "learn":
    frac = float(sys.argv[2])
    c = cand("train", TR).filter((pl.col("s1").hash(seed=77) % 1000) < frac * 1000)
    gt = pl.read_parquet("work/gt_pairs_int.parquet").with_columns(pl.lit(1).alias("y"))
    c = c.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    v = c.explode("extra").drop_nulls("extra").group_by("extra").agg(pl.len().alias("n"), pl.col("y").mean().alias("rate"))
    minc = MINC * frac
    voc = v.filter((pl.col("n") >= minc) & (pl.col("rate") < MAXR)).sort("n", descending=True)
    print(f"pairs {len(c)} (pos {c['y'].mean():.4f}); vocabulary {len(voc)} tokens (n >= {minc:.0f}, rate < {MAXR})")
    print(voc.head(40).rows())
    voc.write_parquet(sys.argv[3])
elif mode in ("eval", "flags"):
    voc = pl.read_parquet(sys.argv[2])["extra"].to_list()
    split = "train" if mode == "eval" else "test"
    c = cand(split, TR if split == "train" else TE)
    c = c.filter(pl.col("extra").list.eval(pl.element().is_in(voc)).list.any()).select("s1", "t")
    if mode == "flags":
        c.write_parquet(sys.argv[4]); print("test pairs flagged", len(c)); sys.exit(0)
    o = pl.read_parquet(sys.argv[3], columns=["s1", "t", "y", "p2"])
    held = (pl.col("s1").hash(seed=77) % 1000) >= float(os.environ.get("FRAC", "0.25")) * 1000
    acc = o.with_columns(pl.col("p2").rank("ordinal", descending=True).over("t").alias("rt")).filter((pl.col("p2") >= 0.7) & (pl.col("rt") == 1) & held)
    hit = acc.join(c, on=["s1", "t"])
    print(f"held-out accepted links {len(acc)}; vetoed {len(hit)} of which true {int(hit['y'].sum())} (FP share {1 - hit['y'].mean():.3f})")
    o.join(c.with_columns(pl.lit(True).alias("v")), on=["s1", "t"], how="left").with_columns(
        pl.when(pl.col("v").fill_null(False) & held).then(0.0).otherwise(pl.col("p2")).alias("p2")).drop("v").write_parquet("work2/oof_sibveto_eval.parquet")
