"""Ditto-style pairwise cross-encoder on a Qwen3 backbone (default: Qwen3-Reranker-0.6B).

The reranker's yes/no LM-head rows are collapsed into one linear head initialised to w_yes - w_no, so at step 0
score = logit(yes) - logit(no), identical to the official Qwen3-Reranker score; it is then fully fine-tuned
with BCE on (S1, target) candidate pairs. Records are serialized Ditto-style (nn_common.ser) and tokenized once
per record; each record is truncated to REC_MAX tokens before the pair is assembled.

usage:
  nn_ce.py train <pairs.parquet: s1,t,y> <split> <outdir> [eval_pairs.parquet]
  nn_ce.py score <pairs.parquet: s1,t> <split> <modeldir> <out.parquet>
  nn_ce.py zeroshot <pairs.parquet: s1,t,y> <split>          (official template, no fine-tuning)
"""
import sys, os, time, math, json, numpy as np, polars as pl, torch
sys.path.insert(0, "work")
from nn_common import records, ser, QWEN_RERANKER
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

BACKBONE = os.environ.get("CE_BACKBONE", QWEN_RERANKER)
IS_QWEN = "qwen" in BACKBONE.lower()     # decoder (last-token pooling, left pad) vs bidirectional encoder ([CLS], right pad)

REC_MAX = int(os.environ.get("REC_MAX", "48"))
BS = int(os.environ.get("CE_BS", "64"))
LR = float(os.environ.get("CE_LR", "2e-5"))
EPOCHS = float(os.environ.get("CE_EPOCHS", "1"))
EVAL_EVERY = int(os.environ.get("CE_EVAL_EVERY", "1000"))
SEED = int(os.environ.get("CE_SEED", "0"))
INSTR = "Do the Query and the Document describe the same real-world business?"
SYS = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct '
       'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n')
PRE = "<|im_start|>user\n<Instruct>: " + INSTR + "\n<Query>: "
MID = "\n<Document>: "
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)


class CE(torch.nn.Module):
    def __init__(self, path=BACKBONE, head=None):
        super().__init__()
        if not IS_QWEN:      # classic Ditto: encoder + linear head on the [CLS]/<s> representation
            self.body = AutoModel.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa", add_pooling_layer=False)
            self.head = torch.nn.Linear(self.body.config.hidden_size, 1)
            if head is not None:
                self.head.load_state_dict(head)
            return
        lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa")
        tok = AutoTokenizer.from_pretrained(path)
        self.body = lm.model
        self.head = torch.nn.Linear(lm.config.hidden_size, 1)
        with torch.no_grad():
            if head is None:
                W = lm.get_output_embeddings().weight
                y, n = tok.convert_tokens_to_ids(["yes", "no"])
                self.head.weight.copy_((W[y] - W[n])[None].float()); self.head.bias.zero_()
            else:
                self.head.load_state_dict(head)
        del lm

    def forward(self, ids, mask):
        h = self.body(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1 if IS_QWEN else 0]  # qwen: left padded
        return self.head(h.float()).squeeze(-1)


class Tok:
    """Pre-tokenize records once; assemble pairs from cached ids."""
    def __init__(self, split, ids, template="compact"):
        self.tok = AutoTokenizer.from_pretrained(BACKBONE)
        self.pad = self.tok.pad_token_id
        if IS_QWEN:
            pre = (SYS if template == "official" else "") + PRE
            self.pre, self.mid, self.suf = [self.tok(x, add_special_tokens=False)["input_ids"] for x in (pre, MID, SUF)]
        else:                # <s> A </s></s> B </s>
            c, e = self.tok.cls_token_id, self.tok.sep_token_id
            self.pre, self.mid, self.suf = [c], [e, e], [e]
        need = pl.DataFrame({"id": np.unique(ids)})
        srcs = sorted(set((need["id"] // 10_000_000_000).to_list()))
        rec = records(split, srcs, ids=need["id"])
        txt = [ser(n, a, tl) for n, a, tl in zip(rec["name"].to_list(), rec["addr"].to_list(), rec["tl"].to_list())]
        enc = self.tok(txt, add_special_tokens=False, truncation=True, max_length=REC_MAX)["input_ids"]
        self.idx = {i: k for k, i in enumerate(rec["id"].to_list())}
        self.enc = enc
        log(f"tokenized {len(enc)} records, mean len {np.mean([len(e) for e in enc]):.1f}")

    def pair(self, a, b):
        return self.pre + self.enc[self.idx[a]] + self.mid + self.enc[self.idx[b]] + self.suf

    def batch(self, seqs):
        L = max(len(s) for s in seqs)
        ids = np.full((len(seqs), L), self.pad, np.int64); m = np.zeros((len(seqs), L), np.int64)
        for i, s in enumerate(seqs):
            if IS_QWEN:
                ids[i, L - len(s):] = s; m[i, L - len(s):] = 1
            else:
                ids[i, :len(s)] = s; m[i, :len(s)] = 1
        return torch.from_numpy(ids).cuda(non_blocking=True), torch.from_numpy(m).cuda(non_blocking=True)


def batches(lens, bs, shuffle, rng):
    """length-bucketed batches: sort within chunks of 100 batches, then shuffle batch order."""
    n = len(lens); order = rng.permutation(n) if shuffle else np.arange(n)
    out = []
    for c in range(0, n, bs * 100):
        ch = order[c:c + bs * 100]
        ch = ch[np.argsort(lens[ch], kind="stable")]
        out += [ch[i:i + bs] for i in range(0, len(ch), bs)]
    if shuffle:
        rng.shuffle(out)
    return out


@torch.no_grad()
def predict(model, T, s1, t, bs=512):
    model.eval()
    seqs = [T.pair(a, b) for a, b in zip(s1, t)]
    lens = np.array([len(s) for s in seqs])
    out = np.zeros(len(seqs), np.float32)
    done = 0
    for bi, b in enumerate(batches(lens, bs, False, None)):
        ids, m = T.batch([seqs[i] for i in b])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[b] = model(ids, m).float().cpu().numpy()
        done += len(b)
        if bi % 2000 == 0:
            log(f"scored {done}/{len(seqs)}")
    return out


def metrics(y, s):
    from sklearn.metrics import roc_auc_score, log_loss
    p = 1 / (1 + np.exp(-s.astype(np.float64)))
    return dict(auc=round(roc_auc_score(y, s), 5), logloss=round(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), 5),
                acc=round(float(((p > 0.5) == y).mean()), 5), n=int(len(y)), pos=round(float(y.mean()), 4))


def train(pairs_path, split, outdir, eval_path=None):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    P = pl.read_parquet(pairs_path)
    E = pl.read_parquet(eval_path) if eval_path else None
    ids = np.concatenate([P["s1"].to_numpy(), P["t"].to_numpy()] + ([E["s1"].to_numpy(), E["t"].to_numpy()] if E is not None else []))
    T = Tok(split, ids)
    model = CE().cuda()
    seqs = [T.pair(a, b) for a, b in zip(P["s1"].to_list(), P["t"].to_list())]
    y = P["y"].to_numpy().astype(np.float32)
    lens = np.array([len(s) for s in seqs])
    log(f"train pairs {len(seqs)} pos {y.mean():.3f} mean tokens {lens.mean():.1f} max {lens.max()}")
    steps_ep = math.ceil(len(seqs) / BS); total = int(steps_ep * EPOCHS)
    model.body.get_input_embeddings().weight.requires_grad_(False)     # (tied) input embedding: frozen, lookup only
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable params {sum(p.numel() for p in params)/1e6:.1f}M of {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01, betas=(0.9, 0.98), fused=True)
    warm = max(100, total // 30)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, 1 - s / total))
    lossf = torch.nn.BCEWithLogitsLoss()
    hist = []; step = 0; run = 0.0
    if E is not None:
        ev = predict(model, T, E["s1"].to_list(), E["t"].to_list())
        hist.append(dict(step=0, **metrics(E["y"].to_numpy(), ev))); log(f"eval {hist[-1]}")
    while step < total:
        for b in batches(lens, BS, True, rng):
            if step >= total:
                break
            model.train()
            ids, m = T.batch([seqs[i] for i in b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(ids, m)
            loss = lossf(s.float(), torch.from_numpy(y[b]).cuda())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
            step += 1; run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 200 == 0:
                log(f"step {step}/{total} loss {run:.4f} lr {sch.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated()/2**30:.1f}G")
            if E is not None and (step % EVAL_EVERY == 0 or step == total):
                ev = predict(model, T, E["s1"].to_list(), E["t"].to_list())
                hist.append(dict(step=step, **metrics(E["y"].to_numpy(), ev))); log(f"eval {hist[-1]}")
    os.makedirs(outdir, exist_ok=True)
    model.body.save_pretrained(outdir, safe_serialization=True)
    torch.save(model.head.state_dict(), f"{outdir}/head.pt")
    json.dump(dict(hist=hist, n_train=len(seqs), steps=total, bs=BS, lr=LR, rec_max=REC_MAX, backbone=BACKBONE,
                   seed=SEED, pairs=pairs_path), open(f"{outdir}/train_log.json", "w"), indent=1)
    log("saved")


def load_trained(d):
    m = CE(head=torch.load(f"{d}/head.pt"))
    kw = {} if IS_QWEN else {"add_pooling_layer": False}
    m.body = AutoModel.from_pretrained(d, dtype=torch.float32, attn_implementation="sdpa", **kw)
    return m.cuda()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        train(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
    elif mode == "score":
        P = pl.read_parquet(sys.argv[2]); split = sys.argv[3]
        T = Tok(split, np.concatenate([P["s1"].to_numpy(), P["t"].to_numpy()]))
        dirs = sys.argv[4].split(",")
        res = {}
        for k, d in enumerate(dirs):
            model = load_trained(d).to(torch.bfloat16)
            res[f"ce{k}"] = predict(model, T, P["s1"].to_list(), P["t"].to_list(), bs=1024)
            del model; torch.cuda.empty_cache()
        P.select("s1", "t").with_columns([pl.Series(k, v) for k, v in res.items()]).write_parquet(sys.argv[5] + ".tmp")
        os.replace(sys.argv[5] + ".tmp", sys.argv[5])     # atomic: downstream queues poll for this file
        log("done")
    elif mode == "zeroshot":
        P = pl.read_parquet(sys.argv[2]); split = sys.argv[3]
        T = Tok(split, np.concatenate([P["s1"].to_numpy(), P["t"].to_numpy()]), template=os.environ.get("TPL", "official"))
        model = CE().cuda().to(torch.bfloat16)
        s = predict(model, T, P["s1"].to_list(), P["t"].to_list())
        print(json.dumps(metrics(P["y"].to_numpy(), s)))
