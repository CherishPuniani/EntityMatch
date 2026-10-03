"""Evaluate decision policies on full OOF predictions (exact macro F0.5 over ALL train S1)."""
import polars as pl, numpy as np, sys
sys.path.insert(0, "work")

oof = pl.read_parquet(sys.argv[1]); col = sys.argv[2]
extra = sys.argv[3] if len(sys.argv) > 3 else ""
s1 = pl.read_parquet("work/p_train_s1.parquet", columns=["id", "country"]).with_columns(
    (pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
import os
if os.environ.get("KEEP"):
    s1 = s1.join(pl.read_parquet(os.environ["KEEP"]).rename({"id": "s1"}), on="s1")
gt = pl.read_parquet("work/gt_pairs_int.parquet").join(s1.select("s1"), on="s1")
G = gt.group_by("s1").len("G")
base = s1.join(G, on="s1", how="left").with_columns(pl.col("G").fill_null(0))
print(f"S1 {len(base)} truepairs {len(gt)} cand-recall {oof['y'].sum()/len(gt):.4f}")


def score(sel, name):
    """sel: DataFrame s1,t,y of predicted pairs."""
    agg = sel.group_by("s1").agg(pl.len().alias("k"), pl.col("y").sum().alias("tp"))
    d = base.join(agg, on="s1", how="left").with_columns(pl.col("k").fill_null(0), pl.col("tp").fill_null(0))
    d = d.with_columns(pl.when(pl.col("G") == 0).then((pl.col("k") == 0).cast(pl.Float64))
                       .when(pl.col("k") == 0).then(0.0)
                       .otherwise(5.0 * pl.col("tp") / (pl.col("G") + 4.0 * pl.col("k"))).alias("f"))
    P = d["tp"].sum() / max(d["k"].sum(), 1); R = d["tp"].sum() / d["G"].sum()
    by = d.group_by("country").agg(pl.col("f").mean()).sort("country")
    sing = d.filter(pl.col("G") == 0)["f"].mean(); ns = d.filter(pl.col("G") > 0)["f"].mean()
    print(f"{name:34s} F05={d['f'].mean():.5f} sing={sing:.4f} nonsing={ns:.4f} P={P:.4f} R={R:.4f} "
          + " ".join(f"{c}={v:.5f}" for c, v in by.iter_rows()), flush=True)
    return d["f"].mean()


df = oof.select("s1", "t", "y", pl.col(col).alias("p"))
excl = df.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
for t in ([0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85] if not extra else [0.7]):
    score(df.filter(pl.col("p") >= t), f"{col} thresh {t}")
    score(excl.filter((pl.col("p") >= t) & (pl.col("rt") == 1)), f"{col} thresh+excl {t}")

if extra == "rank":
    r = df.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("rs"),
                        pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
    for a in [0.4, 0.5, 0.6, 0.7]:
        for b in [0.6, 0.7, 0.8]:
            score(r.filter((pl.col("rt") == 1) & (((pl.col("rs") == 1) & (pl.col("p") >= a)) | ((pl.col("rs") > 1) & (pl.col("p") >= b)))),
                  f"{col} top>={a} rest>={b}")
if extra == "expf":
    import policy
    sub = df.filter(pl.col("p") >= 0.01)
    for mu in [0.0, 0.1]:
        pred = policy.apply_policy(sub.select("s1", "t", "p"), "expf", mu=mu, lo=0.01)
        rows = [(s, t) for s, ts in pred.items() for t in ts]
        sel = pl.DataFrame(rows, schema={"s1": pl.Int64, "t": pl.Int64}, orient="row").join(df.select("s1", "t", "y"), on=["s1", "t"])
        score(sel, f"{col} expf mu={mu}")
