"""Re-gate CE scoring for a new stage-1 model (e.g. after adding a candidate channel) and reuse existing scores.
A CE score depends only on the pair and on which CE scored it (train: the model of the other half; test: fixed
models), so pairs already scored keep their score and only newly gated pairs are scored.

usage: nn_ce_regate.py plan <lo> <hi> <train_p1.parquet> <test_p1.parquet> <model_tag> <new_tag>
          -> work/ce/todo_{train_h0,train_h1,test}_<new_tag>.parquet   (pairs still to score)
       nn_ce_regate.py merge <model_tag> <new_tag> <n_test_models>
          -> work/ce/ce_train_<model_tag>_<new_tag>.parquet, work/ce/ce_test_<model_tag>_<new_tag>.parquet
"""
import sys, os, polars as pl

D = "work/ce"
mode = sys.argv[1]


def put(df, path):
    df.write_parquet(path + ".tmp"); os.replace(path + ".tmp", path)


if mode == "plan":
    lo, hi, trp, tep, M, tag = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
    band = (pl.col("p1") >= lo) & (pl.col("p1") <= hi)
    tr = pl.read_parquet(trp, columns=["s1", "t", "p1"]).filter(band).with_columns(
        ((pl.col("s1").hash(seed=11) % 4) // 2).cast(pl.Int8).alias("half"))
    te = pl.read_parquet(tep, columns=["s1", "t", "p1"]).filter(band)
    for h in (0, 1):
        g = tr.filter(pl.col("half") == h).select("s1", "t")
        put(g, f"{D}/gate_train_h{h}_{tag}.parquet")
        old = pl.read_parquet(f"{D}/gate_train_h{h}.parquet", columns=["s1", "t"])   # base gate: scored by the base round
        todo = g.join(old, on=["s1", "t"], how="anti")
        put(todo, f"{D}/todo_train_h{h}_{tag}.parquet")
        print(f"train h{h}: gated {len(g)}, already scored {len(g) - len(todo)}, to score {len(todo)}")
    put(te.select("s1", "t"), f"{D}/gate_test_{tag}.parquet")
    old = pl.read_parquet(f"{D}/gate_test.parquet", columns=["s1", "t"])
    todo = te.select("s1", "t").join(old, on=["s1", "t"], how="anti")
    put(todo, f"{D}/todo_test_{tag}.parquet")
    print(f"test: gated {len(te)}, already scored {len(te) - len(todo)}, to score {len(todo)}")
elif mode == "merge":
    M, tag, nt = sys.argv[2], sys.argv[3], int(sys.argv[4])
    parts = []
    for h in (0, 1):
        g = pl.read_parquet(f"{D}/gate_train_h{h}_{tag}.parquet")
        sc = pl.concat([pl.read_parquet(f"{D}/sc_train_h{h}_{M}.parquet", columns=["s1", "t", "ce0"]),
                        pl.read_parquet(f"{D}/sc_todo_train_h{h}_{tag}_{M}.parquet", columns=["s1", "t", "ce0"])]).unique(["s1", "t"])
        parts.append(g.join(sc, on=["s1", "t"], how="left").rename({"ce0": "ce"}))
    tr = pl.concat(parts); assert tr["ce"].null_count() == 0
    put(tr, f"{D}/ce_train_{M}_{tag}.parquet")
    g = pl.read_parquet(f"{D}/gate_test_{tag}.parquet")
    cols = []
    for k in range(nt):
        sc = pl.concat([pl.read_parquet(f"{D}/sc_test_{M}_m{k}.parquet", columns=["s1", "t", "ce0"]),
                        pl.read_parquet(f"{D}/sc_todo_test_{tag}_{M}_m{k}.parquet", columns=["s1", "t", "ce0"])]).unique(["s1", "t"])
        g = g.join(sc.rename({"ce0": f"c{k}"}), on=["s1", "t"], how="left"); cols.append(f"c{k}")
    te = g.select("s1", "t", (pl.sum_horizontal(cols) / len(cols)).alias("ce")); assert te["ce"].null_count() == 0
    put(te, f"{D}/ce_test_{M}_{tag}.parquet")
    print(f"merged: train {len(tr)} test {len(te)} ({nt} test model(s))")
