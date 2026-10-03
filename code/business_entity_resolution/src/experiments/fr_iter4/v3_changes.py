"""Label-free check of the US/India links output_final_v3 changes vs output_shift_frnorm2 (test), and the labelled
analogue on OOF (ce3 -> ce4q decisions at p2 >= 0.7). Strata as in cells.py; lf = legal-form set change."""
import polars as pl, glob, sys
sys.path.insert(0, "/home2/home/amritanshu_t/amazon-mlc-26/Amazon-mlc-26/business_entity_resolution/src")
from lfutil import lf
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
FC = ["s1", "t", "hn_rel", "hn_eq", "aempty_2"]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)


def enc(col, base):
    return pl.col(col).str.slice(3).cast(pl.Int64) + base


def links(p):
    d = pl.read_csv(p, separator="\t", schema_overrides={"matched_entity_ids": pl.Utf8})
    return (d.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .select(enc("source1_entity_id", 10_000_000_000).alias("s1"),
                    (pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(20_000_000_000).otherwise(30_000_000_000)
                     + pl.col("matched_entity_ids").str.slice(3).cast(pl.Int64)).alias("t")))


def feats(dirs, keys):
    return pl.concat([pl.read_parquet(p, columns=FC).join(keys, on=["s1", "t"], how="semi")
                      for d in dirs for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])


def strat(df):
    return df.with_columns(
        pl.when(pl.col("aempty_2") == 1).then(pl.lit("empty")).when(pl.col("hn_eq") == 1).then(pl.lit("eq"))
        .when(pl.col("hn_rel").is_null() | (pl.col("hn_rel") < 0)).then(pl.lit("nonum"))
        .when(pl.col("hn_rel").is_in([5, 7])).then(pl.lit("adj(r5/r7)")).otherwise(pl.lit("other_num")).alias("st"))


def names(split):
    out = []
    for s, base in [(1, 10_000_000_000), (2, 20_000_000_000), (3, 30_000_000_000)]:
        d = pl.read_parquet(f"{R}/work/{split}_s{s}.parquet", columns=["entity_id", "business_name"])
        out.append(d.select(enc("entity_id", base).alias("id"), lf("business_name").alias("lf")))
    return pl.concat(out)


def summarize(d, lab):
    d = d.with_columns(((pl.col("lf_s") != pl.col("lf_t")) & (pl.col("lf_s") != "") & (pl.col("lf_t") != "")).alias("lfchg"))
    g = d.group_by("country", "kind").agg(
        pl.len().alias("n"), *[(pl.col("st") == s).mean().round(3).alias(s) for s in ["eq", "adj(r5/r7)", "other_num", "nonum", "empty"]],
        pl.col("lfchg").mean().round(3).alias("lf_change"), *([pl.col("y").mean().round(3).alias("true_rate")] if "y" in d.columns else []))
    print(f"\n== {lab}"); print(g.sort("country", "kind"))
    if "y" in d.columns:
        h = d.group_by("country", "kind", "st").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true_rate"))
        print(h.filter(pl.col("n") >= 50).sort("country", "kind", "st"))


# ---- test: final decisions (after correction, veto, exclusivity)
a = links(f"{R}/output_shift_frnorm2/matching_results.tsv"); b = links(f"{R}/output_final_v3/matching_results.tsv")
c = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).select(enc("entity_id", 10_000_000_000).alias("s1"), "country")
d = pl.concat([b.join(a, on=["s1", "t"], how="anti").with_columns(pl.lit("added").alias("kind")),
               a.join(b, on=["s1", "t"], how="anti").with_columns(pl.lit("removed").alias("kind")),
               a.join(b, on=["s1", "t"], how="semi").with_columns(pl.lit("kept").alias("kind"))]).join(c, on="s1")
d = d.filter(pl.col("country").is_in(["US", "India"]))
d = strat(d.join(feats(["work/feat_test", "work/feat_test_dense"], d.select("s1", "t")), on=["s1", "t"], how="left"))
nm = names("test")
d = d.join(nm.rename({"id": "s1", "lf": "lf_s"}), on="s1", how="left").join(nm.rename({"id": "t", "lf": "lf_t"}), on="t", how="left")
summarize(d, "TEST (no labels): links added / removed by v3 vs frnorm2, and links kept by both")

# ---- OOF: ce3 -> ce4q decision changes at p2 >= 0.7 (no exclusivity), with labels
o = pl.read_parquet(f"{R}/work2/oof_ce3.parquet", columns=["s1", "t", "y", "p2"]).join(
    pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "p2"]).rename({"p2": "p2q"}), on=["s1", "t"])
o = o.filter((pl.col("p2") >= 0.7) != (pl.col("p2q") >= 0.7)).with_columns(
    pl.when(pl.col("p2q") >= 0.7).then(pl.lit("added")).otherwise(pl.lit("removed")).alias("kind"))
ctr = pl.read_parquet(f"{R}/work/p_train_s1.parquet", columns=["id", "country"]).select(enc("id", 10_000_000_000).alias("s1"), "country")
o = strat(o.join(ctr, on="s1").join(feats(["work/feat_trainD", "work/feat_trainD_dense"], o.select("s1", "t")), on=["s1", "t"], how="left"))
nt = names("train")
o = o.join(nt.rename({"id": "s1", "lf": "lf_s"}), on="s1", how="left").join(nt.rename({"id": "t", "lf": "lf_t"}), on="t", how="left")
summarize(o, "OOF (labels): decisions changed ce3 -> ce4q at p2 >= 0.7")
