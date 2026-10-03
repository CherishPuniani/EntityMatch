"""Adjacent-number candidates (hn_rel 5/7): cluster size = number of the S1's candidates sharing the target's house number.
Train (labels): true rate by cluster size (all / accepted). Test: accepted links by cluster size per country."""
import polars as pl, glob, sys
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)


def hn(split):
    out = []
    for s in (2, 3):
        d = pl.read_parquet(f"{R}/work/p_{split}_s{s}.parquet", columns=["id", "at"])
        out.append(d.select((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("t"),
                            pl.col("at").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).list.first().alias("hn")))
    return pl.concat(out)


def clusters(sc, split, fd):
    f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_rel", "hn_eq", "cs_tset"]) for d in fd for p in sorted(glob.glob(f"{R}/{d}/part_*.parquet"))])
    x = sc.join(f, on=["s1", "t"]).join(hn(split), on="t", how="left")
    x = x.filter(pl.col("hn").is_not_null() & (pl.col("hn_eq") != 1))
    x = x.with_columns(pl.len().over("s1", "hn").alias("csize"), (pl.col("cs_tset") >= 80).sum().over("s1", "hn").alias("csize_sim"))
    return x.filter(pl.col("hn_rel").is_in([5, 7]))


mode = sys.argv[1]
if mode == "train":
    o = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y", "p2"])
    x = clusters(o, "train", ["work/feat_trainD", "work/feat_trainD_dense"])
    x = x.with_columns(pl.col("csize_sim").clip(0, 3).alias("cs"))
    print(x.group_by("cs").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true"), (pl.col("p2") >= 0.7).sum().alias("acc"),
                               pl.col("y").filter(pl.col("p2") >= 0.7).mean().round(3).alias("true_acc")).sort("cs"))
else:
    sc = pl.read_parquet(f"{R}/output_final_v3/scores.parquet", columns=["s1", "t", "p2n"])
    c = pl.read_parquet(f"{R}/work/test_s1.parquet", columns=["entity_id", "country"]).select((pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"), "country")
    sc = sc.join(c, on="s1").filter(pl.col("country") != "France")
    x = clusters(sc, "test", ["work/feat_test", "work/feat_test_dense"]).with_columns(pl.col("csize_sim").clip(0, 3).alias("cs"))
    x = x.with_columns(pl.col("p2n").rank("ordinal", descending=True).over("t").alias("rt"))
    n = dict(c.group_by("country").len().iter_rows())
    g = x.group_by("country", "cs").agg(pl.len().alias("n"), ((pl.col("p2n") >= 0.7) & (pl.col("rt") == 1)).sum().alias("acc"))
    print(g.with_columns((pl.col("acc") * 1000 / pl.col("country").replace_strict(n)).round(1).alias("acc_per1k"), (pl.col("n") * 1000 / pl.col("country").replace_strict(n)).round(1).alias("n_per1k")).sort("country", "cs"))
