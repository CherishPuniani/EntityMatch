"""Ground-truth pair table (train only), string and integer ids."""
import polars as pl, sys
sys.path.insert(0, "work")
from ids import to_int
gt = pl.read_parquet("work/gt.parquet").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("t"))
gt = gt.explode("t").filter(pl.col("t") != "").select(pl.col("source1_entity_id").alias("s1"), "t")
gt.write_parquet("work/gt_pairs.parquet")
gt.with_columns(to_int("s1"), to_int("t")).write_parquet("work/gt_pairs_int.parquet")
print(gt.shape)
