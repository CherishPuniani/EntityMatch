#!/bin/bash
# France v4 ("frnorm4") = frnorm2 normaliser + French filler/type-word fix, built entirely under $BASE (/tmp; the home
# quota is full). Stages: chk = assembly path reproduces output_final_v3 byte-for-byte; refeat = re-featurise and
# tree-score unseen-country pairs; asm = assemble output_fr6_noveto (US/India = ce4q, unchanged).
#   filler fix (label-free evidence, RESEARCH_LOG §19): groupe / developpement / france behave like the generator's
#   copy fillers (replace:append 2.5-2.9 like services/cie; copy-count test) -> NOISE; centre / service behave like
#   French type words (ratio 57 / 18; copy-count test) -> removed from NOISE for French records.
set -euo pipefail
ROOT=/home2/home/amritanshu_t/amazon-mlc-26/run
BASE=/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/fr6
PY=/home2/home/amritanshu_t/amazon-mlc-26/venv/bin/python
export PYTHONHASHSEED=0 POLARS_MAX_THREADS=24 ROOT BASE
log() { echo "[$(date +%T)] $*"; }
cd $ROOT

assemble() {  # $1 unseen-country tree scores, $2 intermediates dir, $3 output dir
  local U=$1 D=$2 O=$3; mkdir -p $D
  $PY - << EOF
import polars as pl
s1 = pl.read_parquet('work/test_s1.parquet', columns=['entity_id','country']).with_columns((pl.col('entity_id').str.slice(3).cast(pl.Int64)+10_000_000_000).alias('s1'))
seen = set(pl.read_parquet('work/train_s1.parquet', columns=['country'])['country'].unique().to_list())
un = s1.filter(~pl.col('country').is_in(list(seen))).select('s1')
U = pl.read_parquet('$U', columns=['s1','t','p2'])
T = pl.read_parquet('work/test_scores_dense_s1.parquet', columns=['s1','t','p2'])
A = pl.read_parquet('work2/test_scores_ce4q.parquet', columns=['s1','t','p2'])
assert len(U) == len(T.join(un, on='s1')) and len(U.join(T.join(un, on='s1'), on=['s1','t'], how='anti')) == 0
pl.concat([T.join(un, on='s1', how='anti'), U]).write_parquet('$D/test_scores_tree_final.parquet')
pl.concat([A.join(un, on='s1', how='anti'), U]).write_parquet('$D/test_scores_final_ce4q.parquet')
EOF
  QUIET=1 $PY work2/src/cells.py work/oof_dense_s1.parquet $D/test_scores_tree_final.parquet $D/c_tree_final > $D/cells_tree_final.log 2>&1
  SCORES=$D/test_scores_final_ce4q.parquet CF=work2/c_ce4q_cells.parquet CT=$D/c_tree_final_cells.parquet OUT=$D/out_final CANDS=0 \
    $PY work2/src/apply_shift.py > $D/apply_shift.log 2>&1
  $PY work2/src/apply_lfveto.py $D/out_final $O > $D/lfveto.log 2>&1
  python3 utils/validate_submission.py --matching $O/matching_results.tsv --candidate $O/candidate_pairs.tsv --test-dir dataset/test --check-ids | tail -1
  sha256sum $O/matching_results.tsv $O/candidate_pairs.tsv
}

refeat() {
  local W=$BASE/work_unseen FX=$BASE/fx_unseen
  mkdir $W $FX
  for f in $ROOT/work/*.py; do [ "$(basename $f)" = norm.py ] || ln -s $f $W/$(basename $f); done
  for f in test_s1.parquet test_s2.parquet test_s3.parquet train_s1.parquet indic_dict.json; do ln -s $ROOT/work/$f $W/$f; done
  ln -s $ROOT/work/models_dense_s1 $W/models_dense_s1; ln -s $ROOT/work/dense $W/dense
  cp $ROOT/work/norm.py $W/norm.py
  $PY - << 'PYEOF'
import os
p = os.environ["BASE"] + "/work_unseen/norm.py"; s = open(p).read()
# --- frnorm2 (identical to src/run_unseen_norm.sh)
old = "    f = fold(lat)\n"
assert s.count(old) == 1
s = s.replace(old, old + "    f = re.sub(r\"\\b((?:[a-z]\\.){2,}[a-z]?)(?![a-z])\", lambda m: m.group(1).replace('.', ''), f)\n")
old = '"boulevard": "blvd",'
assert old in s; s = s.replace(old, '"boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd",')
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
# --- frnorm4: French copy fillers become NOISE, French type words stop being NOISE
old = "ALIAS_RE = re.compile("
assert s.count(old) == 1
s = s.replace(old, 'NOISE |= {"groupe", "developpement", "france"}\nNOISE -= {"centre", "service"}\n' + old)
# --- frnorm5: map French copy fillers / legal form onto the training-vocabulary token with the same role (the tree's
# IDF-weighted token features only know training tokens): groupe/developpement/france -> services, sarl -> llc
old = "    f = re.sub(r\"\\b((?:[a-z]\\.){2,}[a-z]?)(?![a-z])\", lambda m: m.group(1).replace('.', ''), f)\n"
assert s.count(old) == 1
s = s.replace(old, old + "    f = re.sub(r\"\\b(groupe|developpement|france)\\b\", \"services\", f)\n    f = re.sub(r\"\\bsarl\\b\", \"llc\", f)\n    f = re.sub(r\"\\bcompagnie\\b\", \"cie\", f)\n    f = re.sub(r\"\\bassocies\\b\", \"partners\", f)\n    f = re.sub(r\"\\bet\\b\", \"and\", f)\n")
open(p, "w").write(s)
print("norm.py patched (frnorm2 + frnorm5)")
PYEOF
  ln -s $W $FX/work
  cd $FX
  log "prep of unseen-country records"
  $PY - << 'PYEOF'
import polars as pl, sys, os
sys.path.insert(0, "work")
from multiprocessing import Pool
import prep
from norm import NOISE
assert "france" in NOISE and "centre" not in NOISE
from norm import name_forms
assert name_forms("Lycée Sainte Développement SARL et Compagnie & Associés")[1] == ["lycee", "sainte", "services", "llc", "and", "cie", "and", "partners"], name_forms("Lycée Sainte Développement SARL")
ROOT = os.environ["ROOT"]
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
for s in (1, 2, 3):
    raw = pl.read_parquet(f"work/test_s{s}.parquet")
    un = raw.filter(~pl.col("country").is_in(list(seen)))
    rows = un.rows(); ch = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
    with Pool(24) as p: res = p.map(prep.proc, ch, chunksize=1)
    new = pl.DataFrame([r for part in res for r in part], schema=prep.cols, orient="row")
    orig = pl.read_parquet(f"{ROOT}/work/p_test_s{s}.parquet")
    out = pl.concat([orig.join(new.select("id"), on="id", how="anti"), new]); assert len(out) == len(orig)
    out.write_parquet(f"work/p_test_s{s}.parquet"); print(s, "unseen-country records re-prepared:", len(new), flush=True)
PYEOF
  $PY work/rec_attrs.py test > /dev/null
  log "unseen-country candidate pairs from the original feature parts"
  $PY - << 'PYEOF'
import polars as pl, glob, os
ROOT = os.environ["ROOT"]
seen = set(pl.read_parquet("work/train_s1.parquet", columns=["country"])["country"].unique().to_list())
s1 = pl.read_parquet("work/test_s1.parquet", columns=["entity_id", "country"]).filter(~pl.col("country").is_in(list(seen))).select(
    (pl.col("entity_id").str.slice(3).cast(pl.Int64) + 10_000_000_000).alias("s1"))
B = ["bscore", "nkeys", "brank", "t_ns1", "t_best", "t_rank", "t_bbest", "t_brank", "rs", "rrank", "npool", "wmax"] + [f"w{k}" for k in range(8)]
for src, dst in [(f"{ROOT}/work/feat_test", "work/pairs_testun.parquet"), (f"{ROOT}/work/feat_test_dense", "work/pairs_testun_dense.parquet")]:
    d = pl.concat([pl.read_parquet(p, columns=["s1", "t"] + B) for p in sorted(glob.glob(src + "/part_*.parquet"))]).join(s1, on="s1")
    d.write_parquet(dst); print(dst, len(d))
PYEOF
  log "features"
  for ((w=0; w<4; w++)); do NOFILTER=1 PTAG=testun POLARS_MAX_THREADS=6 $PY work/build_feats_all.py test 0 work/feat_testun $w 4 > $BASE/feat_testun_$w.log 2>&1 & done; wait
  for ((w=0; w<4; w++)); do NOFILTER=1 PTAG=testun_dense POLARS_MAX_THREADS=6 $PY work/build_feats_all.py test 0 work/feat_testun_dense $w 4 > $BASE/feat_testun_dense_$w.log 2>&1 & done; wait
  log "tree model (dense_s1) on unseen-country pairs"
  TAG=dense_s1 TH=0.7 EXCL=1 OUT=out_unseen FEAT=work/feat_testun,work/feat_testun_dense EXTRA_PAIRFEATS=$BASE/pairfeats_fr6.parquet $PY work/predict_test.py > $BASE/predict_unseen.log 2>&1
  cp work/test_scores_dense_s1.parquet $BASE/test_scores_tree_unseen_fr6.parquet
  cd $ROOT
}

for stage in ${@:-chk refeat asm}; do
  case $stage in
    chk) log "chk: assembly with the frnorm2 French scores must reproduce output_final_v3 (6be4fe53..., 7d75bdd2...)"
         assemble work2/test_scores_tree_unseen.parquet $BASE/chk $BASE/chk/output ;;
    refeat) log "refeat: frnorm6"; refeat ;;
    asm) log "asm: output_fr6_noveto"; assemble $BASE/test_scores_tree_unseen_fr6.parquet $BASE/asm $BASE/output_fr6_noveto ;;
  esac
done
log "fr6 chain done"
