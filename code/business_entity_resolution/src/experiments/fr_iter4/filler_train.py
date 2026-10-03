"""Labelled check (US/India OOF) of the French pattern: same house number, target adds a token (filler / country word)
and possibly drops an S1 content token. True rate and model acceptance by added token and edit type; test France
(no labels) acceptance for the same classes."""
import polars as pl, glob
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)


def load(split, country, sp, fd, pcol):
    cols = ["s1", "t", pcol] + (["y"] if split == "train" else [])
    sc = pl.read_parquet(sp, columns=cols).rename({pcol: "p"})
    pref = "work_unseen" if country == "France" else "work"
    T = []
    for s in (1, 2, 3):
        d = pl.read_parquet(f"{R}/{pref}/p_{split}_s{s}.parquet", columns=["id", "country", "nt", "core"]).filter(pl.col("country") == country)
        T.append(d.select((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "nt", "core"))
    T = pl.concat(T)
    s1 = T.filter(pl.col("id") < 20_000_000_000).select(pl.col("id").alias("s1"), pl.col("nt").alias("nt1"), pl.col("core").alias("c1"))
    sc = sc.join(s1, on="s1")
    f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_eq", "aempty_2", "acs_tset"]).join(sc.select("s1", "t"), on=["s1", "t"], how="semi")
                   for d in fd for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])
    x = sc.join(f, on=["s1", "t"]).filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1) & (pl.col("acs_tset") >= 90))
    x = x.join(T.select(pl.col("id").alias("t"), pl.col("nt").alias("nt2"), pl.col("core").alias("c2")), on="t")
    L3 = lambda e: e.list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
    x = x.with_columns(L3(pl.col("nt2").list.set_difference("nt1")).alias("added"),
                       L3(pl.col("c1").list.set_difference("nt2")).alias("dropped"),
                       pl.col("c1").list.set_intersection("nt2").list.len().alias("shared_core"))
    x = x.with_columns(pl.when(pl.col("added").list.len() == 0).then(pl.lit("none")).when(pl.col("added").list.len() == 1)
                       .then(pl.col("added").list.first()).otherwise(pl.lit("2+")).alias("add1"),
                       (pl.col("dropped").list.len() > 0).alias("drop"))
    return x


def report(x, lab, lab_col):
    agg = [pl.len().alias("n"), pl.col("p").mean().round(3).alias("mean_p"), (pl.col("p") >= 0.7).mean().round(3).alias("acc")]
    if lab_col: agg.append(pl.col("y").mean().round(3).alias("true"))
    g = x.filter(pl.col("add1") != "none").group_by("add1", "drop").agg(agg).filter(pl.col("n") >= 200).sort("n", descending=True)
    print(f"\n== {lab}: same number+street pairs adding exactly one token (top by n)"); print(g.head(30))


if __name__ == "__main__":
  tr = load("train", "India", f"{R}/work2/oof_ce4q.parquet", ["work/feat_trainD", "work/feat_trainD_dense"], "p2")
  report(tr, "TRAIN India (labels)", True)
  tr = load("train", "US", f"{R}/work2/oof_ce4q.parquet", ["work/feat_trainD", "work/feat_trainD_dense"], "p2")
  report(tr, "TRAIN US (labels)", True)
  te = load("test", "France", f"{R}/output_final_v3/scores.parquet", ["work_unseen/feat_testun", "work_unseen/feat_testun_dense"], "p2n")
  report(te, "TEST France (no labels; p = corrected p2n)", False)
