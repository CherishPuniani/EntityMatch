"""Train: do entity-ID numbers or row positions carry pair information?"""
import polars as pl, numpy as np
D = "/home2/home/amritanshu_t/amazon-mlc-26/run/dataset/train"
rd = lambda f: pl.read_csv(f"{D}/{f}", separator="\t", quote_char=None, schema_overrides={"entity_id": pl.Utf8}).with_row_index("row")
s1, s2, s3 = rd("train_source1.tsv"), rd("train_source2.tsv"), rd("train_source3.tsv")
gt = pl.read_csv(f"{D}/train_ground_truth.tsv", separator="\t").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id": "a", "matched_entity_ids": "b"}).drop_nulls("b")
num = lambda c: pl.col(c).str.slice(3).cast(pl.Int64)
tg = pl.concat([s2.select("entity_id", "row", pl.lit(2).alias("src")), s3.select("entity_id", "row", pl.lit(3).alias("src"))])
x = gt.join(s1.select(pl.col("entity_id").alias("a"), pl.col("row").alias("ra")), on="a").join(tg.rename({"entity_id": "b", "row": "rb"}), on="b")
x = x.with_columns(num("a").alias("ia"), num("b").alias("ib"))
print("pairs", x.height)
n1, n2, n3 = s1.height, s2.height, s3.height
for src, n in ((2, n2), (3, n3)):
    y = x.filter(pl.col("src") == src)
    ra = (y["ra"].to_numpy() / n1); rb = (y["rb"].to_numpy() / n)
    print(f"S{src}: corr(row S1, row S{src}) = {np.corrcoef(ra, rb)[0, 1]:.4f} | corr(id S1, id) = {np.corrcoef(y['ia'].to_numpy(), y['ib'].to_numpy())[0, 1]:.4f}")
    print("   |row diff| (normalised) quantiles:", np.quantile(np.abs(ra - rb), [0.01, 0.1, 0.5]).round(4))
# copies of the same S1 in the same source: row distance vs random
for src, n in ((2, n2), (3, n3)):
    y = x.filter(pl.col("src") == src).group_by("a").agg(pl.col("rb").sort()).filter(pl.col("rb").list.len() >= 2)
    d = y.select(pl.col("rb").list.diff().list.drop_nulls().alias("d")).explode("d")["d"].to_numpy()
    print(f"S{src}: row gap between copies of the same S1 (median / 10%): {np.median(d):.0f} / {np.quantile(d, 0.1):.0f}  (random expectation ~{n / 3:.0f})")
    yi = x.filter(pl.col("src") == src).group_by("a").agg(pl.col("ib").sort()).filter(pl.col("ib").list.len() >= 2)
    di = yi.select(pl.col("ib").list.diff().list.drop_nulls().alias("d")).explode("d")["d"].to_numpy()
    print(f"     id gap between copies (median / 10%): {np.median(di):.0f} / {np.quantile(di, 0.1):.0f}  | id range {tg['entity_id'].str.slice(3).cast(pl.Int64).min()}..{tg['entity_id'].str.slice(3).cast(pl.Int64).max()}")
print("id digit patterns: ia % 1000 == ib % 1000 share:", round((x["ia"] % 1000 == x["ib"] % 1000).mean(), 5))
