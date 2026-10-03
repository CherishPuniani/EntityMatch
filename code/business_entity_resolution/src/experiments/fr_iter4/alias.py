"""Alias transitivity: a candidate whose name equals the registered (pre-DBA) name carried by another candidate of the same S1
('Solbelo DBA Jf Pharmacie' -> 'Solbelo'). Train: true rate + model acceptance; test: counts + acceptance."""
import polars as pl, sys, re
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
split = sys.argv[1]
norm = lambda c: pl.col(c).fill_null("").str.replace_all(r"[^a-z0-9]", "")
P = []
for s in (1, 2, 3):
    d = pl.read_parquet(f"{R}/work/p_{split}_s{s}.parquet", columns=["id", "country", "nf", "al_pre", "al_post"])
    P.append(d.select((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "country", norm("nf").alias("nk"), norm("al_pre").alias("ak"), norm("al_post").alias("pk")))
P = pl.concat(P)
if split == "train":
    sc = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t", "y", "p2"]).rename({"p2": "p"})
else:
    sc = pl.read_parquet(sys.argv[2], columns=["s1", "t", "p2n"]).rename({"p2n": "p"})
sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("t").alias("rt"))
x = sc.join(P.rename({"id": "t"}), on="t")
# registered names stated by the S1's candidates that are strong (p >= 0.9, the S1's best claimant) or by the S1 itself
src = x.filter((pl.col("p") >= 0.9) & (pl.col("rt") == 1) & (pl.col("ak").str.len_chars() >= 4)).select("s1", pl.col("ak").alias("reg"), pl.col("t").alias("t_src"))
s1self = P.filter(pl.col("id") < 20_000_000_000).filter(pl.col("ak").str.len_chars() >= 4).select(pl.col("id").alias("s1"), pl.col("ak").alias("reg"), pl.lit(0, dtype=pl.Int64).alias("t_src"))
reg = pl.concat([src, s1self]).unique(["s1", "reg"])
hit = x.join(reg, left_on=["s1", "nk"], right_on=["s1", "reg"]).filter(pl.col("t") != pl.col("t_src")).unique(["s1", "t"])
agg = [pl.len().alias("n"), ((pl.col("p") >= 0.7) & (pl.col("rt") == 1)).mean().round(3).alias("acc"), (pl.col("rt") == 1).mean().round(3).alias("best")]
if split == "train":
    agg += [pl.col("y").mean().round(3).alias("true"), pl.col("y").filter(pl.col("rt") == 1).mean().round(3).alias("true_best")]
print(split, "candidates named by another candidate's registered (pre-DBA) name:")
print(hit.group_by("country").agg(agg).sort("country"))
