"""Train/test overlap: exact (name, address, country) matches between test and train records; IDs shared; and whether
train ground-truth pairs transfer to test pairs through exact duplicates."""
import polars as pl
D = "/home2/home/amritanshu_t/amazon-mlc-26/run/dataset"
rd = lambda f: pl.read_csv(f"{D}/{f}", separator="\t", quote_char=None, schema_overrides={"entity_id": pl.Utf8}).with_columns(
    pl.concat_str([pl.col("business_name").fill_null("").str.to_lowercase().str.strip_chars(), pl.col("business_address").fill_null("").str.to_lowercase().str.strip_chars(), "country"], separator="|").alias("k"))
tr = pl.concat([rd(f"train/train_source{i}.tsv").with_columns(pl.lit(i).alias("s")) for i in (1, 2, 3)])
te = pl.concat([rd(f"test/test_source{i}.tsv").with_columns(pl.lit(i).alias("s")) for i in (1, 2, 3)])
print("shared entity IDs train/test:", te.join(tr, on="entity_id", how="semi").height)
for i in (1, 2, 3):
    t = te.filter(pl.col("s") == i)
    print(f"test source{i}: {t.height} records, exact (name,address,country) found in train: {t.join(tr, on='k', how='semi').height}, "
          f"by country: {t.join(tr, on='k', how='semi').group_by('country').len().rows()}")
# transfer: test S1 dup of train S1 Y, test target dup of train target W with (Y, W) a gt pair
gt = pl.read_csv(f"{D}/train/train_ground_truth.tsv", separator="\t").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").rename({"source1_entity_id": "Y", "matched_entity_ids": "W"}).drop_nulls("W")
m1 = te.filter(pl.col("s") == 1).select(pl.col("entity_id").alias("X"), "k").join(tr.filter(pl.col("s") == 1).select(pl.col("entity_id").alias("Y"), "k"), on="k")
mt = te.filter(pl.col("s") > 1).select(pl.col("entity_id").alias("Z"), "k").join(tr.filter(pl.col("s") > 1).select(pl.col("entity_id").alias("W"), "k"), on="k")
tp = m1.select("X", "Y").join(gt, on="Y").join(mt.select("Z", "W"), on="W").select("X", "Z").unique()
print("test S1 duplicated in train:", m1["X"].n_unique(), "| test targets duplicated in train:", mt["Z"].n_unique(), "| transferable test pairs:", tp.height)
