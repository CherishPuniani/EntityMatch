"""Fixed-population comparison of OOF predictions: exact per-entity macro F0.5 (all kept train S1, singletons and
S1s without candidates included), threshold + target exclusivity, paired bootstrap over S1s, diagnostic slices and
sequential loss attribution.

usage: KEEP=work/keep_trainD.parquet compare_oof.py <base.parquet>:<col> <new.parquet>:<col> [th=0.7]
Each file must contain s1, t, y and the probability column.
"""
import sys, os, re, numpy as np, polars as pl

TH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
IND = r"[ऀ-෿]"
s1 = pl.read_parquet("work/p_train_s1.parquet", columns=["id", "country"]).with_columns(
    (pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
if os.environ.get("KEEP"):
    s1 = s1.join(pl.read_parquet(os.environ["KEEP"]).rename({"id": "s1"}), on="s1")
gt = pl.read_parquet("work/gt_pairs_int.parquet").join(s1.select("s1"), on="s1")
# entity-level slice attributes from the true targets
tr = []
for s in (2, 3):
    tr.append(pl.read_parquet(f"work/train_s{s}.parquet").select(
        (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("t"),
        pl.col("business_name").str.contains(IND).alias("ind"),
        (pl.col("business_address").str.strip_chars() == "").alias("emp")))
tr = pl.concat(tr)
ga = gt.join(tr, on="t", how="left").group_by("s1").agg(pl.len().alias("G"), pl.col("ind").any().alias("has_ind"),
                                                          pl.col("emp").any().alias("has_empty"))
base = s1.join(ga, on="s1", how="left").with_columns(pl.col("G").fill_null(0), pl.col("has_ind").fill_null(False),
                                                     pl.col("has_empty").fill_null(False))
base = base.with_columns(pl.when(pl.col("G") == 0).then(pl.lit("0")).when(pl.col("G") == 1).then(pl.lit("1"))
                         .when(pl.col("G") <= 3).then(pl.lit("2-3")).otherwise(pl.lit("4+")).alias("Gb"))


def load(spec):
    path, col = spec.rsplit(":", 1)
    return pl.read_parquet(path, columns=["s1", "t", "y", col]).rename({col: "p"})


def per_entity(df, th):
    sel = df.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt")).filter(
        (pl.col("p") >= th) & (pl.col("rt") == 1))
    agg = sel.group_by("s1").agg(pl.len().alias("k"), pl.col("y").sum().alias("tp"))
    d = base.join(agg, on="s1", how="left").with_columns(pl.col("k").fill_null(0), pl.col("tp").fill_null(0))
    return d.with_columns(pl.when(pl.col("G") == 0).then((pl.col("k") == 0).cast(pl.Float64))
                          .when(pl.col("k") == 0).then(0.0)
                          .otherwise(5.0 * pl.col("tp") / (pl.col("G") + 4.0 * pl.col("k"))).alias("f")).sort("s1")


def summary(name, df, d):
    P = d["tp"].sum() / max(d["k"].sum(), 1); R = d["tp"].sum() / d["G"].sum()
    cand = df["y"].sum() / d["G"].sum()
    by = {c: v for c, v in d.group_by("country").agg(pl.col("f").mean()).iter_rows()}
    print(f"{name:10s} F05={d['f'].mean():.5f}  P={P:.4f} R={R:.4f} cand_recall={cand:.4f} sing={d.filter(pl.col('G')==0)['f'].mean():.4f} "
          + " ".join(f"{c}={v:.5f}" for c, v in sorted(by.items())))


def attribution(df, d, name):
    """sequential counterfactuals: drop all FPs, then add every in-candidate TP."""
    sel = df.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt")).filter(
        (pl.col("p") >= TH) & (pl.col("rt") == 1))
    tpo = sel.filter(pl.col("y") == 1).group_by("s1").agg(pl.len().alias("k"))
    ora = df.filter(pl.col("y") == 1).group_by("s1").agg(pl.len().alias("k"))
    out = []
    for kk in (tpo, ora):
        x = base.join(kk, on="s1", how="left").with_columns(pl.col("k").fill_null(0))
        x = x.with_columns(pl.when(pl.col("G") == 0).then(1.0).when(pl.col("k") == 0).then(0.0)
                           .otherwise(5.0 * pl.col("k") / (pl.col("G") + 4.0 * pl.col("k"))).alias("f"))
        out.append(x["f"].mean())
    print(f"{name:10s} attribution: actual {d['f'].mean():.5f} | no-FP {out[0]:.5f} (FP cost {out[0]-d['f'].mean():.5f}) | "
          f"+all in-cand TP {out[1]:.5f} (in-cand FN cost {out[1]-out[0]:.5f}) | blocking loss {1-out[1]:.5f}")


if __name__ == "__main__":
    A = load(sys.argv[1]); B = load(sys.argv[2])
    for th in (0.6, 0.65, 0.7, 0.75, 0.8):
        print(f"th {th}: base {per_entity(A, th)['f'].mean():.5f}  new {per_entity(B, th)['f'].mean():.5f}")
    da, db = per_entity(A, TH), per_entity(B, TH)
    summary("base", A, da); summary("new", B, db)
    attribution(A, da, "base"); attribution(B, db, "new")
    diff = (db["f"] - da["f"]).to_numpy()
    rng = np.random.default_rng(0); n = len(diff)
    bs = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(200)])
    print(f"DELTA @th{TH}: {diff.mean():+.5f}  bootstrap SE {bs.std():.5f}  95% CI [{np.percentile(bs,2.5):+.5f}, {np.percentile(bs,97.5):+.5f}]"
          f"  entities improved {int((diff>0).sum())} worsened {int((diff<0).sum())}")
    j = da.select("s1", "country", "Gb", "has_ind", "has_empty", pl.col("f").alias("fa")).with_columns(db["f"].alias("fb"))
    for sl in ("country", "Gb", "has_ind", "has_empty"):
        t = j.group_by(sl).agg(pl.len().alias("n"), pl.col("fa").mean(), pl.col("fb").mean(),
                               ((pl.col("fb") - pl.col("fa")).sum() / len(j)).alias("contrib")).sort(sl)
        print(f"slice {sl}: " + " | ".join(f"{r[0]} n={r[1]} {r[2]:.5f}->{r[3]:.5f} ({r[4]:+.5f})" for r in t.iter_rows()))
