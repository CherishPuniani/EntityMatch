"""Learn a native-script (Indic) -> Latin word dictionary from TRAINING ground-truth pairs.
Native-script names are word-for-word transliterations of the S1 name, so aligned tokens give the mapping.
Also learns native-script state names in addresses."""
import polars as pl, collections, json


def isindic(s):
    return any(0x0900 <= ord(c) <= 0x0DFF for c in s)


recs = {}
for s in [1, 2, 3]:
    df = pl.read_parquet(f"work/train_s{s}.parquet").filter(pl.col("country") == "India")
    for eid, n, a, c in df.iter_rows():
        recs[eid] = (n, a)
gt = pl.read_parquet("work/gt.parquet").rows()
name = collections.defaultdict(collections.Counter)
addr = collections.defaultdict(collections.Counter)
for s1, m in gt:
    if s1 not in recs:
        continue
    n1, a1 = recs[s1]
    comps1 = [x.strip().lower() for x in a1.split(",")]
    for x in (m or "").split(","):
        if not x or x not in recs:
            continue
        n2, a2 = recs[x]
        if isindic(n2):
            a, b = n1.split(), n2.split()
            if len(a) == len(b):
                for u, v in zip(a, b):
                    if isindic(v):
                        name[v][u.lower()] += 1
        if isindic(a2):
            for comp in a2.split(","):
                comp = comp.strip()
                if isindic(comp):
                    for c1 in comps1:
                        addr[comp][c1] += 1
nd = {v: c.most_common(1)[0][0] for v, c in name.items()}
ad = {v: c.most_common(1)[0][0] for v, c in addr.items()}
json.dump({"name": nd, "addr": ad}, open("work/indic_dict.json", "w"), ensure_ascii=False)
print(len(nd), len(ad))
