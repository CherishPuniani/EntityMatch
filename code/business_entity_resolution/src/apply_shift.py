"""Apply the per-cell label-shift correction to routed test scores and write a submission.
env: RMIN (min test/train density ratio to act, default 2), LO/HI (p2 band corrected, default 0.5/0.95), MINN (min test
pairs in cell, default 30), OUT (dir), SCORES (routed scores), CF (cells for training countries), CT (cells for unseen)"""
import polars as pl, glob, os, sys
sys.path.insert(0, "work")
from ids import to_str
RMIN = float(os.environ.get("RMIN", "2")); LO = float(os.environ.get("LO", "0.5")); HI = float(os.environ.get("HI", "0.95"))
MINN = int(os.environ.get("MINN", "30")); TH = 0.7; OUT = os.environ.get("OUT", "work2/out_shift")
BINS = [float(x) for x in os.environ.get("BINS", "0.3,0.5,0.6,0.7,0.8,0.9,0.95").split(",")]
SC = os.environ.get("SCORES", "work/test_scores_final_routed.parquet")
COLS = ["s1", "t", "hn_rel", "hn_eq", "aempty_2"]
s1te = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).with_columns(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
STRATA = os.environ.get("STRATA", "r5,r7").split(",")
STRATA_U = os.environ.get("STRATA_U", "r4,r5,r7").split(",")   # unseen countries (r4: see RESEARCH_LOG_TEST_SHIFT D6)
cf = pl.read_parquet(os.environ.get("CF", "work2/c_final_cells.parquet")).filter(pl.col("country").is_in(list(seen)))
ct = pl.read_parquet(os.environ.get("CT", "work2/c_tree_cells.parquet")).filter(~pl.col("country").is_in(list(seen)))
# Countries without labels: their copy-typo rate is estimated label-free from sibling-free number strata (r1: dropped
# leading digits, r2: dropped trailing digits, r6: higher digit changed) at p2 >= 0.95, relative to the pooled training
# countries; expected true adjacent-number (r5/r7) pairs are scaled by it and the correction covers every bin >= LO.
TYPO = os.environ.get("TYPO", "r1,r2,r6").split(",")
top = ct.filter(pl.col("bin") == "[0.95, inf)")
sig = {}
for c in ct["country"].unique().to_list():
    t = top.filter((pl.col("country") == c) & pl.col("st").is_in(TYPO))
    sig[c] = float(t["nrate_te"].sum() / t["nrate_tr"].sum())
print("unseen-country typo-rate scale:", sig)
if os.environ.get("NOSIG") == "1":
    sig = {c: 1.0 for c in sig}
# Gate (negative control in RESEARCH_LOG_TEST_SHIFT D8): the sigma-scaled correction is only trusted for an unseen
# country whose own data show that copies never change the legal form while adjacent-number candidates often do.
# A country that fails the gate gets no correction at all (India-as-unseen: sigma alone would cost -0.0197 F).
sys.path.insert(0, "work")
from lfutil import lf
_recs = []
for _s in (1, 2, 3):
    _d = pl.read_parquet(f"work/test_s{_s}.parquet", columns=["entity_id", "business_name", "country"]).filter(pl.col("country").is_in(list(sig)))
    _recs.append(_d.select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + _s * 10_000_000_000).alias("id"), "country", lf("business_name").alias("lf")))
_R = pl.concat(_recs)
_sc = pl.read_parquet(SC, columns=["s1", "t", "p2"]).filter(pl.col("p2") >= 0.5)
_f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_rel", "hn_eq", "aempty_2"]) for d in ["work/feat_test", "work/feat_test_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
_x = _sc.join(_R.select(pl.col("id").alias("s1"), "country", pl.col("lf").alias("l1")), on="s1").join(_R.select(pl.col("id").alias("t"), pl.col("lf").alias("l2")), on="t").join(_f, on=["s1", "t"], how="left")
_x = _x.with_columns(((pl.col("l1") != "") & (pl.col("l2") != "") & (pl.col("l1") != pl.col("l2"))).alias("chg"))
gate = {}
for c in sig:
    xc = _x.filter(pl.col("country") == c)
    c_true = xc.filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1) & (pl.col("p2") >= 0.95))["chg"].mean() or 0.0
    c_sib = xc.filter(pl.col("hn_rel").is_in([5, 7]))["chg"].mean() or 0.0
    gate[c] = bool(c_true < 0.005 and c_sib > 0.2)
    print(f"unseen {c}: legal-form change true-copy proxy {c_true:.4f}, sibling proxy {c_sib:.4f} -> gate {'PASS' if gate[c] else 'FAIL (no correction)'}")
del _x, _f, _sc, _R
ct = ct.filter(pl.col("country").is_in([c for c, g in gate.items() if g]))
ct = ct.with_columns(pl.col("country").replace_strict(sig, default=1.0).alias("sig"))
ct = ct.with_columns(pl.when(pl.col("st").is_in(STRATA_U)).then(
    pl.min_horizontal(pl.col("prec_tr"), pl.col("sig") * (pl.col("prate_tr") + 1e-7) / pl.col("nrate_te"))).otherwise(pl.col("prec_te")).alias("prec_te"))
ct = ct.with_columns((pl.col("prec_te") / pl.col("prec_tr")).alias("mult"),
                     pl.when(pl.col("st").is_in(STRATA_U)).then(pl.lit(99.0)).otherwise(pl.col("ratio")).alias("ratio"))
cells = pl.concat([cf.with_columns(pl.lit(False).alias("allbins")), ct.select(cf.columns).with_columns(pl.col("st").is_in(STRATA_U).alias("allbins"))])
cells = cells.with_columns(pl.when((pl.col("st").is_in(STRATA) | pl.col("allbins")) & (pl.col("ratio") >= RMIN) & (pl.col("n_te") >= MINN)).then(pl.col("mult")).otherwise(1.0).alias("m"))
sc = pl.read_parquet(SC, columns=["s1", "t", "p2"])
f = pl.concat([pl.read_parquet(p, columns=COLS) for d in ["work/feat_test", "work/feat_test_dense"] for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
x = sc.filter(pl.col("p2") >= LO).join(f, on=["s1", "t"], how="left").join(s1te.select("s1", "country"), on="s1")
x = x.with_columns(
    pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq"))
    .when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum"))
    .otherwise(pl.concat_str(pl.lit("r"), pl.col("hn_rel").cast(pl.Int32).cast(pl.Utf8))).alias("st"),
    pl.col("p2").cut(BINS, left_closed=True).cast(pl.Utf8).alias("bin"))
x = x.join(cells.select("country", "st", "bin", "m", "allbins"), on=["country", "st", "bin"], how="left").with_columns(
    pl.when((pl.col("p2") < HI) | pl.col("allbins").fill_null(False)).then(pl.col("m").fill_null(1.0)).otherwise(1.0).alias("m"))
adj = x.select("s1", "t", (pl.col("p2") * pl.col("m")).alias("p2n"))
res = sc.join(adj, on=["s1", "t"], how="left").with_columns(pl.coalesce("p2n", "p2").alias("p2n"))
def select(df, col):
    return df.with_columns(pl.col(col).rank("ordinal", descending=True).over("t").alias("rt")).filter((pl.col(col) >= TH) & (pl.col("rt") == 1))
old, new = select(res, "p2"), select(res, "p2n")
for nm, d in (("old", old), ("new", new)):
    k = s1te.join(d.group_by("s1").len(), on="s1", how="left").with_columns(pl.col("len").fill_null(0))
    print(nm, k.group_by("country").agg(pl.col("len").mean().alias("k"), (pl.col("len") == 0).mean().alias("empty")).sort("country").rows())
rem = old.join(new, on=["s1", "t"], how="anti").join(s1te.select("s1", "country"), on="s1")
add = new.join(old, on=["s1", "t"], how="anti").join(s1te.select("s1", "country"), on="s1")
print("removed per 1k S1:", {c: round(n * 1000 / m, 2) for (c, n), m in zip(rem.group_by("country").len().sort("country").rows(),
      [x for _, x in s1te.group_by("country").len().sort("country").rows()])}, "added:", len(add))
print("removed total", len(rem), "cells acting:", cells.filter(pl.col("m") < 1).height)
os.makedirs(OUT, exist_ok=True)
S1 = s1te.select(pl.col("entity_id").alias("source1_entity_id"))
def write(pairs, pcol, colname, path):
    agg = (pairs.sort(["s1", pcol], descending=[False, True]).with_columns(to_str("s1"), to_str("t"))
           .group_by("s1", maintain_order=True).agg(pl.col("t").str.join(",").alias(colname)).rename({"s1": "source1_entity_id"}))
    o = S1.join(agg, on="source1_entity_id", how="left").with_columns(pl.col(colname).fill_null(""))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{colname}\n")
        for a, b in o.iter_rows():
            fh.write(f"{a}\t{b}\n")
write(new, "p2n", "matched_entity_ids", f"{OUT}/matching_results.tsv")
if os.environ.get("CANDS", "1") == "1":
    write(res, "p2n", "candidate_entity_ids", f"{OUT}/candidate_pairs.tsv")
res.select("s1", "t", "p2", "p2n").write_parquet(f"{OUT}/scores.parquet")
print("wrote", OUT)
