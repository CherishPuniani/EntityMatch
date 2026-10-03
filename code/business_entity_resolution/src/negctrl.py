"""Negative control for the unseen-country correction: India OOF plays the unseen country, US OOF the reference.
Reports India's true macro F0.5 before/after (a) the sigma-scaled r4/r5/r7 correction alone, (b) the same gated by
the legal-form mixture test (only applied if c_true < 0.5 % and c_sib > 20 %)."""
import polars as pl, glob, sys
sys.path.insert(0, "work")
from lfutil import lf
BINS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]; TH = 0.7; STRATA_U = ["r4", "r5", "r7"]
o = pl.read_parquet("work/oof_dense_s1_ce_qwen.parquet", columns=["s1", "t", "y", "p2"])
s1c = pl.read_parquet("work/p_train_s1.parquet", columns=["id", "country"]).with_columns((pl.col("id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1")).select("s1", "country")
keep = pl.read_parquet("work/keep_trainD.parquet").rename({"id": "s1"}).join(s1c, on="s1")
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_rel", "hn_eq", "aempty_2"]) for d in ["work/feat_trainD", "work/feat_trainD_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
x = o.filter(pl.col("p2") >= 0.3).join(f, on=["s1", "t"], how="left").join(s1c, on="s1")
x = x.with_columns(pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq"))
                   .when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum"))
                   .otherwise(pl.concat_str(pl.lit("r"), pl.col("hn_rel").cast(pl.Int32).cast(pl.Utf8))).alias("st"),
                   pl.col("p2").cut(BINS, left_closed=True).cast(pl.Utf8).alias("bin"))
n = dict(keep.group_by("country").len().iter_rows())
ref = x.filter(pl.col("country") == "US").group_by("st", "bin").agg(pl.len().alias("n_tr"), pl.col("y").sum().alias("pos_tr")).with_columns(
    (pl.col("n_tr") / n["US"]).alias("nrate_tr"), (pl.col("pos_tr") / n["US"]).alias("prate_tr"), ((pl.col("y_") if False else pl.col("pos_tr")) + 1).truediv(pl.col("n_tr") + 2).alias("prec_tr"))
tst = x.filter(pl.col("country") == "India").group_by("st", "bin").agg(pl.len().alias("n_te")).with_columns((pl.col("n_te") / n["India"]).alias("nrate_te"))
c = tst.join(ref, on=["st", "bin"], how="left").fill_null(0)
top = c.filter(pl.col("bin") == "[0.95, inf)")
t = top.filter(pl.col("st").is_in(["r1", "r2", "r6"])); sig = float(t["nrate_te"].sum() / t["nrate_tr"].sum())
c = c.with_columns(pl.when(pl.col("st").is_in(STRATA_U)).then(pl.min_horizontal(pl.col("prec_tr"), sig * (pl.col("prate_tr") + 1e-7) / pl.col("nrate_te"))).otherwise(pl.col("prec_tr")).alias("prec_te"))
c = c.with_columns((pl.col("prec_te") / pl.col("prec_tr")).alias("m"))
# legal-form mixture gate, measured on the "unseen" country exactly as apply_lfveto.py does
recs = []
for s in (1, 2, 3):
    d = pl.read_parquet(f"work/train_s{s}.parquet", columns=["entity_id", "business_name", "country"]).filter(pl.col("country") == "India")
    recs.append(d.select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), lf("business_name").alias("lf")))
R = pl.concat(recs)
xi = x.filter((pl.col("country") == "India") & (pl.col("p2") >= 0.5)).join(R.rename({"id": "s1", "lf": "l1"}), on="s1").join(R.rename({"id": "t", "lf": "l2"}), on="t")
xi = xi.with_columns(((pl.col("l1") != "") & (pl.col("l2") != "") & (pl.col("l1") != pl.col("l2"))).alias("chg"))
ct = xi.filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1) & (pl.col("p2") >= 0.95))["chg"].mean()
cs = xi.filter(pl.col("hn_rel").is_in([5, 7]))["chg"].mean()
gate = ct < 0.005 and cs > 0.2
print(f"India-as-unseen: sigma {sig:.3f}; legal-form gate c_true {ct:.4f} c_sib {cs:.4f} -> {'PASS' if gate else 'FAIL'}")
gt = pl.read_parquet("work/gt_pairs_int.parquet").join(keep.filter(pl.col("country") == "India").select("s1"), on="s1")
G = gt.group_by("s1").len().rename({"len": "G"})
oi = o.join(keep.filter(pl.col("country") == "India").select("s1"), on="s1")
adj = x.filter(pl.col("country") == "India").join(c.select("st", "bin", "m"), on=["st", "bin"], how="left").select("s1", "t", (pl.col("p2") * pl.col("m").fill_null(1.0)).alias("p2n"))
oi = oi.join(adj, on=["s1", "t"], how="left").with_columns(pl.coalesce("p2n", "p2").alias("p2n"))
def F(col):
    sel = oi.with_columns(pl.col(col).rank("ordinal", descending=True).over("t").alias("rt")).filter((pl.col(col) >= TH) & (pl.col("rt") == 1))
    a = sel.group_by("s1").agg(pl.len().alias("k"), pl.col("y").sum().alias("tp"))
    d = keep.filter(pl.col("country") == "India").select("s1").join(G, on="s1", how="left").join(a, on="s1", how="left").fill_null(0)
    return d.with_columns(pl.when(pl.col("G") == 0).then((pl.col("k") == 0).cast(pl.Float64)).when(pl.col("k") == 0).then(0.0)
                          .otherwise(5 * pl.col("tp") / (pl.col("G") + 4 * pl.col("k"))).alias("f"))["f"].mean()
b, a = F("p2"), F("p2n")
print(f"India macro F0.5: before {b:.5f} | sigma correction alone {a:.5f} ({a-b:+.5f}) | with legal-form gate {(a if gate else b):.5f} ({(a-b) if gate else 0:+.5f})")
