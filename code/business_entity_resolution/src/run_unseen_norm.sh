#!/bin/bash
# Re-featurise the candidate pairs of countries WITHOUT training labels (read from the data; France in this test set)
# with an extended normaliser, and score them with the tree-only model (dense_s1). Run from the challenge root after
# run_all.sh and run_neural.sh. Output: work2/test_scores_tree_unseen.parquet (s1, t, p1, p2), same candidate pairs.
#
# Normaliser extensions (norm.py is patched in an isolated copy; US/India keep the features their models were trained
# on, which is why this step is limited to unseen countries):
#   * dotted legal forms collapse to one token (s.a.r.l. -> sarl, e.u.r.l. -> eurl, s.a.s.u -> sasu, s.c.i. -> sci);
#     before, their single letters r/u/e/i survived as fake name tokens (≈ 5.5 % of French copies);
#   * bd / boul / bld -> blvd;
#   * French regions and departments map to a region code handled like US/Indian state codes (dropped from the core
#     address); S1 records carry the region, copies the region, the department (Nord, Gironde, ...) or neither.
#   * (frnorm4-6a, RESEARCH_LOG_TEST_SHIFT.md §20-§21; LB 0.984851 -> 0.98667) the tree only knows training tokens (its
#     IDF-weighted name features encode token identity), so French words with a known role are mapped onto the training
#     token with that role: copy fillers groupe / developpement / france -> services; legal forms sarl -> llc,
#     compagnie -> cie; associes -> partners; et -> and. French type words centre / service leave the filler list.
# Candidate pairs are rebuilt from the feat_test* parts (they carry the blocking columns), so the scored pair set is
# exactly the pipeline's candidate set.
set -euo pipefail
ROOT=$(pwd); PY=${PY:-python3}; export PYTHONHASHSEED=0
log() { echo "[$(date +%T)] $*"; }
W=work_unseen; FX=fx_unseen
rm -rf $W $FX; mkdir -p $W $FX work2
for f in work/*.py; do ln -s ../$f $W/$(basename $f); done
for f in test_s1.parquet test_s2.parquet test_s3.parquet train_s1.parquet indic_dict.json; do ln -s ../work/$f $W/$f; done
ln -s ../work/models_dense_s1 $W/models_dense_s1; ln -s ../work/dense $W/dense
rm $W/norm.py; cp work/norm.py $W/norm.py
$PY - << 'PYEOF'
p = "work_unseen/norm.py"; s = open(p).read()
old = "    f = fold(lat)\n"
assert s.count(old) == 1
s = s.replace(old, old + "    f = re.sub(r\"\\b((?:[a-z]\\.){2,}[a-z]?)(?![a-z])\", lambda m: m.group(1).replace('.', ''), f)\n")
old = '"boulevard": "blvd",'
assert old in s; s = s.replace(old, '"boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd",')
# frnorm4-6a: French words mapped onto the training-vocabulary token with the same role; French type words leave NOISE
old = "lambda m: m.group(1).replace('.', ''), f)\n"
assert s.count(old) == 1
s = s.replace(old, old + "    f = re.sub(r\"\\b(groupe|developpement|france)\\b\", \"services\", f)\n    f = re.sub(r\"\\bsarl\\b\", \"llc\", f)\n"
              "    f = re.sub(r\"\\bcompagnie\\b\", \"cie\", f)\n    f = re.sub(r\"\\bassocies\\b\", \"partners\", f)\n    f = re.sub(r\"\\bet\\b\", \"and\", f)\n")
old = "ALIAS_RE = re.compile("
assert s.count(old) == 1
s = s.replace(old, 'NOISE |= {"groupe", "developpement", "france"}\nNOISE -= {"centre", "service"}\n' + old)
FR = {"hauts-de-france": "frhdf", "hauts de france": "frhdf", "nord": "frhdf", "pas-de-calais": "frhdf", "pas de calais": "frhdf",
      "aisne": "frhdf", "oise": "frhdf", "somme": "frhdf",
      "nouvelle-aquitaine": "frnaq", "nouvelle aquitaine": "frnaq", "gironde": "frnaq", "landes": "frnaq", "dordogne": "frnaq",
      "charente": "frnaq", "charente-maritime": "frnaq", "correze": "frnaq", "creuse": "frnaq", "lot-et-garonne": "frnaq",
      "pyrenees-atlantiques": "frnaq", "deux-sevres": "frnaq", "haute-vienne": "frnaq",
      "pays de la loire": "frpdl", "pays-de-la-loire": "frpdl", "loire-atlantique": "frpdl", "loire atlantique": "frpdl",
      "maine-et-loire": "frpdl", "mayenne": "frpdl", "sarthe": "frpdl", "vendee": "frpdl",
      "ile-de-france": "fridf", "auvergne-rhone-alpes": "frara", "bourgogne-franche-comte": "frbfc", "bretagne": "frbre",
      "centre-val de loire": "frcvl", "corse": "frcor", "grand est": "fres", "normandie": "frnor", "occitanie": "frocc",
      "provence-alpes-cote d'azur": "frpac"}
old = "ALL_STATES.update(IN_STATES)\n"
assert s.count(old) == 1
s = s.replace(old, old + "FR_ADMIN = " + repr(FR) + "\nALL_STATES.update(FR_ADMIN)\n")
old = "STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())"
assert s.count(old) == 1; s = s.replace(old, old + " | set(FR_ADMIN.values())")
open(p, "w").write(s)
PYEOF
ln -s ../$W $FX/work
cd $FX
log "prep of unseen-country records"
$PY - << 'PYEOF'
import polars as pl, sys
sys.path.insert(0, "work")
from multiprocessing import Pool
import prep
from norm import name_forms
assert name_forms("Lycée Sainte Développement SARL et Compagnie & Associés")[1] == ["lycee", "sainte", "services", "llc", "and", "cie", "and", "partners"]
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
for s in (1, 2, 3):
    raw = pl.read_parquet(f"work/test_s{s}.parquet")
    un = raw.filter(~pl.col("country").is_in(list(seen)))
    rows = un.rows(); ch = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
    with Pool(24) as p: res = p.map(prep.proc, ch, chunksize=1)
    new = pl.DataFrame([r for part in res for r in part], schema=prep.cols, orient="row")
    orig = pl.read_parquet(f"../work/p_test_s{s}.parquet")
    out = pl.concat([orig.join(new.select("id"), on="id", how="anti"), new]); assert len(out) == len(orig)
    out.write_parquet(f"work/p_test_s{s}.parquet"); print(s, "unseen-country records re-prepared:", len(new), flush=True)
PYEOF
$PY work/rec_attrs.py test > /dev/null
log "unseen-country candidate pairs from the feature parts"
$PY - << 'PYEOF'
import polars as pl, glob
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).filter(~pl.col("country").is_in(list(seen))).select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
B = ["bscore", "nkeys", "brank", "t_ns1", "t_best", "t_rank", "t_bbest", "t_brank", "rs", "rrank", "npool", "wmax"] + [f"w{k}" for k in range(8)]
for src, dst in [("../work/feat_test", "work/pairs_testun.parquet"), ("../work/feat_test_dense", "work/pairs_testun_dense.parquet")]:
    d = pl.concat([pl.read_parquet(p, columns=["s1", "t"] + B) for p in sorted(glob.glob(src + "/part_*.parquet"))]).join(s1, on="s1")
    d.write_parquet(dst); print(dst, len(d))
PYEOF
log "features"
for ((w=0; w<4; w++)); do NOFILTER=1 PTAG=testun POLARS_MAX_THREADS=6 $PY work/build_feats_all.py test 0 work/feat_testun $w 4 > /dev/null 2>&1 & done; wait
for ((w=0; w<4; w++)); do NOFILTER=1 PTAG=testun_dense POLARS_MAX_THREADS=6 $PY work/build_feats_all.py test 0 work/feat_testun_dense $w 4 > /dev/null 2>&1 & done; wait
log "tree model (dense_s1) on unseen-country pairs"
TAG=dense_s1 TH=0.7 EXCL=1 OUT=out_unseen FEAT=work/feat_testun,work/feat_testun_dense EXTRA_PAIRFEATS=work/dense/pairfeats_test.parquet POLARS_MAX_THREADS=24 $PY work/predict_test.py
cp work/test_scores_dense_s1.parquet ../work2/test_scores_tree_unseen.parquet
log "done"
