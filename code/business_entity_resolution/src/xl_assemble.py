"""XLM-R CE scores on exactly the final model's gated pairs (train: OOF by the other-half model; test: mean of both)."""
import polars as pl
q = pl.read_parquet("work/ce/ce_train_qwen_dense_s1.parquet", columns=["s1", "t"])
tr = pl.concat([pl.read_parquet(f).select("s1", "t", pl.col("ce0").alias("ce")) for f in
                ["work/ce/sc_train_h0_xlmr.parquet", "work/ce/sc_train_h1_xlmr.parquet", "work2/xl_sc_train_h0.parquet", "work2/xl_sc_train_h1.parquet"]]).unique(["s1", "t"])
tr = q.join(tr, on=["s1", "t"], how="left"); print("train", len(tr), "missing", tr["ce"].null_count()); tr.write_parquet("work2/ce_train_xlmr_final.parquet")
qt = pl.read_parquet("work/ce/ce_test_qwen_dense_s1.parquet", columns=["s1", "t"])
new = pl.read_parquet("work2/xl_sc_test.parquet").select("s1", "t", ((pl.col("ce0") + pl.col("ce1")) / 2).alias("ce"))
te = pl.concat([pl.read_parquet("work/ce/ce_test_xlmr.parquet").select("s1", "t", "ce"), new]).unique(["s1", "t"])
te = qt.join(te, on=["s1", "t"], how="left"); print("test", len(te), "missing", te["ce"].null_count()); te.write_parquet("work2/ce_test_xlmr_final.parquet")
