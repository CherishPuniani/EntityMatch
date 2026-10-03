#!/bin/bash
# France-only re-featurisation with two normaliser fixes (dotted legal forms collapsed; bd/boul/bld -> blvd).
# Isolated: run/work3 holds symlinks to read-only inputs plus the new files; scripts run from run/fx (work -> ../work3).
set -euo pipefail
cd /home2/home/amritanshu_t/amazon-mlc-26/run
PY=$(pwd)/../venv/bin/python; export PYTHONHASHSEED=0
log() { echo "[$(date +%T)] $*"; }
rm -rf work4 fx2; mkdir -p work4 fx2
for f in work/*.py; do ln -s ../$f work4/$(basename $f); done
for f in test_s1.parquet test_s2.parquet test_s3.parquet train_s1.parquet indic_dict.json; do ln -s ../work/$f work4/$f; done
ln -s ../work/models_dense_s1 work4/models_dense_s1; ln -s ../work/dense work4/dense
rm work4/norm.py; cp work/norm.py work4/norm.py
python3 - << 'PYEOF'
p = "work4/norm.py"; s = open(p).read()
old = "    f = fold(lat)\n"
new = ("    f = fold(lat)\n"
       "    # dotted legal forms / acronyms (s.a.r.l., e.u.r.l., s.a.s.u, s.c.i., l.l.c.) -> one token (French copies use them)\n"
       "    f = re.sub(r\"\\b((?:[a-z]\\.){2,}[a-z]?)(?![a-z])\", lambda m: m.group(1).replace('.', ''), f)\n")
assert s.count(old) == 1; s = s.replace(old, new)
old2 = '"boulevard": "blvd",'
assert old2 in s; s = s.replace(old2, '"boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd",')
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
old3 = "ALL_STATES.update(IN_STATES)\n"
assert s.count(old3) == 1
s = s.replace(old3, old3 + "# French regions and departments (department -> its region), handled like US/Indian states\nFR_ADMIN = " + repr(FR) + "\nALL_STATES.update(FR_ADMIN)\n")
old4 = "STATE_CODES = set(US_STATES.values()) | set(IN_STATES.values())"
assert s.count(old4) == 1
s = s.replace(old4, old4 + " | set(FR_ADMIN.values())")
open(p, "w").write(s)
PYEOF
ln -s ../work4 fx2/work
cd fx2
$PY -c "
import sys; sys.path.insert(0,'work'); import norm
for n in ['Trousse Club S.A.R.L.','SAS MVF Club','Mérignac Comite E.U.R.L','Centre Médical Doo S.A.S.U','Ets Orgues S.C.I.','Roussell Virginia P.L.L.C.','St. Louis Bakery','A.B. Smith & Co']:
    print(repr(n), norm.name_forms(n)[1])
print(norm.addr_forms('155 BD CONSTANTIN DESCAT, TOURCOING, Nord')[2], norm.addr_forms('155 Boulevard Constantin Descat, Tourcoing, Hauts-de-France')[2], norm.addr_forms('42 RUE de Coulmiers, Nantes, Pays de la Loire')[2])
"
log "France-only prep"
$PY - << 'PYEOF'
import polars as pl, sys, os
sys.path.insert(0, "work")
from multiprocessing import Pool
import prep
for s in (1, 2, 3):
    raw = pl.read_parquet(f"work/test_s{s}.parquet")
    fr = raw.filter(pl.col("country") == "France")
    rows = fr.rows(); ch = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
    with Pool(24) as p: res = p.map(prep.proc, ch, chunksize=1)
    new = pl.DataFrame([r for part in res for r in part], schema=prep.cols, orient="row")
    orig = pl.read_parquet(f"../work/p_test_s{s}.parquet")
    out = pl.concat([orig.join(new.select("id"), on="id", how="anti"), new])
    assert len(out) == len(orig)
    out.write_parquet(f"work/p_test_s{s}.parquet")
    ch_ = orig.join(new, on="id", suffix="_n").filter(pl.col("nt") != pl.col("nt_n"))
    print(s, "France records", len(new), "name tokens changed", len(ch_), flush=True)
PYEOF
log "rec_attrs test"
$PY work/rec_attrs.py test > ../logs2/frnorm2_recattrs.log 2>&1
log "France pair subsets"
$PY -c "
import polars as pl
s1 = pl.read_parquet('work/test_s1.parquet', columns=['entity_id','country']).filter(pl.col('country')=='France').select((pl.col('entity_id').str.slice(3).cast(pl.Int64)+10_000_000_000).alias('s1'))
a = pl.read_parquet('../work/pairs_test.parquet').join(s1, on='s1'); a.write_parquet('work/pairs_testfr.parquet')
b = pl.read_parquet('../work/pairs_test_dense.parquet').join(s1, on='s1'); b.write_parquet('work/pairs_testfr_dense.parquet')
print('france pairs', len(a), 'dense', len(b))"
log "features (blocking pairs)"
for ((w=0; w<4; w++)); do PTAG=testfr POLARS_MAX_THREADS=6 $PY work/build_feats_all.py test 30 work/feat_testfr $w 4 > ../logs2/frnorm2_feat_$w.log 2>&1 & done; wait
log "features (dense pairs)"
for ((w=0; w<4; w++)); do NOFILTER=1 PTAG=testfr_dense POLARS_MAX_THREADS=6 $PY work/build_feats_all.py test 0 work/feat_testfr_dense $w 4 > ../logs2/frnorm2_featd_$w.log 2>&1 & done; wait
log "tree model on France pairs"
TAG=dense_s1 TH=0.7 EXCL=1 OUT=out_fr FEAT=work/feat_testfr,work/feat_testfr_dense EXTRA_PAIRFEATS=work/dense/pairfeats_test.parquet POLARS_MAX_THREADS=24 $PY work/predict_test.py > ../logs2/frnorm2_predict.log 2>&1
tail -2 ../logs2/frnorm2_predict.log
log done
