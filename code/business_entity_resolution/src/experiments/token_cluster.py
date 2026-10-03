"""Added-name-token 'cluster rate': for candidate pairs at the S1's own house number, tokens the target adds to the
S1 name; cluster = another candidate of the same S1 adds the same token. Copy-noise fillers are one-offs, words that
define a different business are shared by that business's copies. Train (labelled) validates, test France applies."""
import polars as pl, glob, sys
split, country = sys.argv[1], sys.argv[2]
PFX = {"train": ("work/oof_ce3.parquet" if False else "work2/oof_ce3.parquet", ["work/feat_trainD", "work/feat_trainD_dense"]),
       "test": ("output_shift_routed/scores.parquet", ["work/feat_test", "work/feat_test_dense"])}
sp, fd = PFX[split]
cols = ["s1", "t", "p2"] + (["y"] if split == "train" else [])
sc = pl.read_parquet(sp, columns=cols).filter(pl.col("p2") >= 0.05)
T = []
for s in (1, 2, 3):
    d = pl.read_parquet(f"work/p_{split}_s{s}.parquet", columns=["id", "country", "nt", "core"]).filter(pl.col("country") == country)
    T.append(d.select((pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "nt", "core"))
T = pl.concat(T)
f = pl.concat([pl.read_parquet(p, columns=["s1", "t", "hn_eq", "aempty_2"]) for d in fd for p in sorted(glob.glob(f"{d}/part_*.parquet"))])
x = sc.join(T.select(pl.col("id").alias("s1"), pl.col("nt").alias("nt1")), on="s1").join(f, on=["s1", "t"]).filter((pl.col("hn_eq") == 1) & (pl.col("aempty_2") != 1))
x = x.join(T.select(pl.col("id").alias("t"), pl.col("core").alias("c2")), on="t").with_columns(pl.col("c2").list.set_difference("nt1").alias("added"))
e = x.select("s1", "t", "p2", *(["y"] if split == "train" else []), "added").explode("added").drop_nulls("added")
e = e.filter(pl.col("added").str.len_chars() >= 3)
e = e.with_columns((pl.len().over("s1", "added") > 1).alias("clustered"))
agg = e.group_by("added").agg(pl.len().alias("n"), pl.col("clustered").mean().alias("cluster_rate"), pl.col("p2").mean().alias("mean_p2"),
                              *([pl.col("y").mean().alias("true_rate")] if split == "train" else [])).filter(pl.col("n") >= (300 if split == "train" else 150)).sort("n", descending=True)
pl.Config.set_tbl_rows(45); pl.Config.set_tbl_width_chars(160)
print(split, country, "same-number pairs with an added token:", e.select("s1", "t").n_unique())
print(agg.head(40))
if split == "train":
    print("corr(cluster_rate, true_rate) over tokens:", agg.select(pl.corr("cluster_rate", "true_rate")).item())
    print(agg.with_columns(pl.col("cluster_rate").cut([0.2, 0.4, 0.6]).alias("cr")).group_by("cr").agg(pl.len(), pl.col("true_rate").mean(), pl.col("n").sum()).sort("cr"))
