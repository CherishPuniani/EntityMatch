"""Dense retrieval with a Qwen3 embedding model (last-token pooling, MRL-truncated, cosine), per country.

usage: nn_dense.py encode <split> <out.npy>           encode all S2/S3 targets of a split
       nn_dense.py search <split> <targets.npy> <s1_ids.parquet|all> <K> <out.parquet>
Queries use the model's instruction format; documents are encoded without instruction (Qwen3-Embedding usage).
"""
import sys, os, time, numpy as np, polars as pl, torch, torch.nn.functional as F
sys.path.insert(0, "work")
from nn_common import records, ser, QWEN_EMBED
from transformers import AutoTokenizer, AutoModel

DIM = int(os.environ.get("DENSE_DIM", "512"))
BS = int(os.environ.get("DENSE_BS", "1024"))
TASK = "Given a business record (name and address), retrieve records of the same business despite typos, abbreviations, transliteration and missing fields"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)


def load():
    tok = AutoTokenizer.from_pretrained(QWEN_EMBED, padding_side="left")
    m = AutoModel.from_pretrained(QWEN_EMBED, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    return tok, m


@torch.no_grad()
def encode(tok, m, texts, maxlen=96, tok_budget=int(os.environ.get("DENSE_TOKENS", "65536"))):
    """pre-tokenize once, then length-sorted batches under a token budget (little padding, big GEMMs)."""
    enc = []
    for i in range(0, len(texts), 200_000):
        enc += tok(texts[i:i + 200_000], truncation=True, max_length=maxlen)["input_ids"]
    lens = np.array([len(e) for e in enc]); order = np.argsort(lens, kind="stable")
    out = np.zeros((len(texts), DIM), np.float16); pad = tok.pad_token_id
    i = 0; nb = 0
    while i < len(order):
        L = lens[order[min(len(order) - 1, i)]]
        n = max(1, tok_budget // max(L, 1))
        idx = order[i:i + n]; L = lens[idx].max()
        ids = np.full((len(idx), L), pad, np.int64); am = np.zeros((len(idx), L), np.int64)
        for r, j in enumerate(idx):
            e = enc[j]; ids[r, L - len(e):] = e; am[r, L - len(e):] = 1
        h = m(input_ids=torch.from_numpy(ids).cuda(), attention_mask=torch.from_numpy(am).cuda()).last_hidden_state[:, -1, :DIM].float()
        out[idx] = F.normalize(h, dim=-1).half().cpu().numpy()
        i += len(idx); nb += 1
        if nb % 200 == 0:
            log(f"encoded {i}/{len(texts)}")
    return out


def texts_of(df, query):
    t = [ser(n, a, tl) for n, a, tl in zip(df["name"].to_list(), df["addr"].to_list(), df["tl"].to_list())]
    return [f"Instruct: {TASK}\nQuery:{x}" for x in t] if query else t


if __name__ == "__main__":
    mode, split = sys.argv[1], sys.argv[2]
    tok, m = load()
    if mode == "encode":
        tg = records(split, (2, 3))
        if os.environ.get("COUNTRY"):
            tg = tg.filter(pl.col("country") == os.environ["COUNTRY"])
        if os.environ.get("IDS"):                    # restrict the index, e.g. to targets not yet assigned
            tg = tg.join(pl.read_parquet(os.environ["IDS"]).select(pl.col("id")), on="id")
        if os.environ.get("LIMIT"):
            tg = tg.head(int(os.environ["LIMIT"]))
        e = encode(tok, m, texts_of(tg, False))
        np.save(sys.argv[3], e)
        tg.select("id", "country").write_parquet(sys.argv[3].replace(".npy", "_ids.parquet"))
        log(f"saved {e.shape}")
    elif mode == "search":
        # <targets.npy[,more.npy]> : several encoded matrices are searched as one index; TGT_IDS restricts it
        tpaths, qpath, K, out = sys.argv[3].split(","), sys.argv[4], int(sys.argv[5]), sys.argv[6]
        mats = [np.load(p, mmap_mode="r") for p in tpaths]
        tid = pl.concat([pl.read_parquet(p.replace(".npy", "_ids.parquet")).with_row_index("row").with_columns(
            pl.lit(k).alias("mat")) for k, p in enumerate(tpaths)])
        if os.environ.get("TGT_IDS"):
            tid = tid.join(pl.read_parquet(os.environ["TGT_IDS"]).select("id"), on="id")
        q = records(split, (1,), translit=False)
        if qpath != "all":
            qi = pl.read_parquet(qpath); q = q.join(qi.select(pl.col(qi.columns[0]).alias("id")), on="id")
        q = q.with_columns(pl.lit("").alias("tl"))
        QE = encode(tok, m, texts_of(q, True))
        log(f"queries {len(q)}; index {len(tid)} targets")
        del m; torch.cuda.empty_cache()
        res = []
        for c in q["country"].unique().to_list():
            rows = tid.filter(pl.col("country") == c)
            if len(rows) == 0:
                continue
            T = torch.cat([torch.from_numpy(np.ascontiguousarray(mats[k][rows.filter(pl.col("mat") == k)["row"].to_numpy()]))
                           for k in range(len(mats))]).cuda()
            rid = np.concatenate([rows.filter(pl.col("mat") == k)["id"].to_numpy() for k in range(len(mats))])
            qm = (q["country"] == c).to_numpy(); qi = q["id"].to_numpy()[qm]; Q = torch.from_numpy(QE[qm]).cuda()
            k = min(K, len(rid))
            for i in range(0, len(Q), 1024):
                sc, j = torch.topk(Q[i:i + 1024] @ T.T, k, dim=1)
                sc = sc.float().cpu().numpy(); j = j.cpu().numpy()
                res.append(pl.DataFrame({"s1": np.repeat(qi[i:i + 1024], k), "t": rid[j.ravel()],
                                         "dscore": sc.ravel().astype(np.float32),
                                         "drank": np.tile(np.arange(1, k + 1, dtype=np.int16), len(sc))}))
            log(f"country {c}: {qm.sum()} queries x {len(rid)} targets")
            del T; torch.cuda.empty_cache()
        pl.concat(res).write_parquet(out)
        log("done")
