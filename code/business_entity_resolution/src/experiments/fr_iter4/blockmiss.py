"""Train: true pairs missing from the candidate set (blocking loss). Profile + examples."""
import polars as pl
R = "/home2/home/amritanshu_t/amazon-mlc-26/run"
keep = pl.read_parquet(f"{R}/work/keep_trainD.parquet").rename({"id": "s1"})
gt = pl.read_parquet(f"{R}/work/gt_pairs_int.parquet").join(keep, on="s1")
cand = pl.read_parquet(f"{R}/work2/oof_ce4q.parquet", columns=["s1", "t"])
miss = gt.join(cand, on=["s1", "t"], how="anti")
print("gt pairs", gt.height, "missing from candidates", miss.height, round(miss.height / gt.height, 4))
P = pl.concat([pl.read_parquet(f"{R}/work/p_train_s{s}.parquet", columns=["id", "country", "core", "acore", "nt"]).select(
    (pl.col("id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "country", "core", "acore") for s in (1, 2, 3)])
raw = pl.concat([pl.read_parquet(f"{R}/work/train_s{s}.parquet").select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + s * 10_000_000_000).alias("id"), "business_name", "business_address") for s in (1, 2, 3)])
m = miss.join(P.rename({"id": "s1", "core": "c1", "acore": "a1"}), on="s1").join(P.select(pl.col("id").alias("t"), pl.col("core").alias("c2"), pl.col("acore").alias("a2")), on="t")
m = m.with_columns((pl.col("a2").list.len() == 0).alias("t_addr_empty"),
                   (pl.col("c1").list.sort() == pl.col("c2").list.sort()).alias("same_core"),
                   (pl.col("c1").list.set_intersection("c2").list.len() > 0).alias("share_core_tok"),
                   (pl.col("a1").list.set_intersection("a2").list.len() / pl.max_horizontal(pl.col("a1").list.len(), 1)).alias("addr_ov"))
print(m.group_by("country").agg(pl.len(), pl.col("t_addr_empty").mean().round(3), pl.col("same_core").mean().round(3),
                                pl.col("share_core_tok").mean().round(3), (pl.col("addr_ov") >= 0.8).mean().round(3).alias("addr_ov>=0.8")))
# how many S1s share the exact core name (per country) for missed empty-address pairs
cnt = P.filter(pl.col("id") < 20_000_000_000).with_columns(pl.col("core").list.sort().list.join(" ").alias("ck")).group_by("country", "ck").len().rename({"len": "n_s1_same_name"})
mm = m.filter(pl.col("t_addr_empty") & pl.col("same_core")).with_columns(pl.col("c1").list.sort().list.join(" ").alias("ck")).join(cnt, on=["country", "ck"], how="left")
print("missed, empty-address, same core name: n_s1 sharing the name:", mm["n_s1_same_name"].describe().rows())
info = {r[0]: (r[1], r[2]) for r in raw.join(pl.concat([m.select(pl.col("s1").alias("id")), m.select(pl.col("t").alias("id"))]).unique(), on="id").iter_rows()}
for s1, t in m.filter(~pl.col("t_addr_empty")).sample(12, seed=3).select("s1", "t").iter_rows():
    a, b = info[s1], info[t]
    print(f"S1 {a[0][:40]:40s} | {a[1][:60]}\n T {b[0][:40]:40s} | {b[1][:60]}")
