import polars as pl, sys
for split in ["train","test"]:
    for s in [1,2,3]:
        p=f"dataset/{split}/{split}_source{s}.tsv"
        df=pl.read_csv(p,separator="\t",quote_char=None,infer_schema=False,missing_utf8_is_empty_string=True)
        print(p,df.shape,df.columns)
        df.write_parquet(f"work/{split}_s{s}.parquet")
gt=pl.read_csv("dataset/train/train_ground_truth.tsv",separator="\t",quote_char=None,infer_schema=False,missing_utf8_is_empty_string=True)
print(gt.shape); gt.write_parquet("work/gt.parquet")
