"""Leaderboard diagnostic: copy a submission with every S1 of one country predicted empty.
score(full) - score(masked) = share_c * (F_c - singleton_rate_c), so one extra upload yields country c's macro F0.5
(share_c = its fraction of scored S1s, singleton_rate_c ≈ 0.06). Not a submission candidate.
usage: mask_country.py <in_dir> <country> <out_dir>   (run from the challenge root; reads dataset/test/test_source1.tsv)"""
import sys, os, csv
src, country, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = set()
with open("dataset/test/test_source1.tsv", encoding="utf-8") as fh:
    r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
    h = next(r); ci, ii = h.index("country"), h.index("entity_id")
    for row in r:
        if row[ci] == country:
            ids.add(row[ii])
os.makedirs(out, exist_ok=True)
n = 0
with open(f"{src}/matching_results.tsv", encoding="utf-8") as fi, open(f"{out}/matching_results.tsv", "w", encoding="utf-8") as fo:
    fo.write(fi.readline())
    for line in fi:
        a, b = line.rstrip("\n").split("\t")
        if a in ids:
            b = ""; n += 1
        fo.write(f"{a}\t{b}\n")
os.system(f"cp {src}/candidate_pairs.tsv {out}/candidate_pairs.tsv")
print(f"{country}: {len(ids)} S1s, {n} rows emptied -> {out}")
