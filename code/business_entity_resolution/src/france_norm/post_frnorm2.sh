#!/bin/bash
set -euo pipefail
cd /home2/home/amritanshu_t/amazon-mlc-26/run
PY=../venv/bin/python; export PYTHONHASHSEED=0 POLARS_MAX_THREADS=24; S=work2/src
until grep -q "\] done" logs2/frnorm2.out; do sleep 10; if grep -q -i "error\|Traceback" logs2/frnorm2.out logs2/frnorm2_predict.log 2>/dev/null; then echo FAILED; exit 1; fi; done
echo "[$(date +%T)] merge"
$PY -c "
import polars as pl
fr = pl.read_parquet('work4/test_scores_dense_s1.parquet', columns=['s1','t','p1','p2'])
old = pl.read_parquet('work/test_scores_dense_s1.parquet', columns=['s1','t','p1','p2'])
frs = fr.select('s1').unique()
oldfr = old.join(frs, on='s1')
m = oldfr.join(fr, on=['s1','t'], how='full', coalesce=True, suffix='_n')
print('France pairs old', len(oldfr), 'new', len(fr), 'unmatched', m['p2'].null_count(), m['p2_n'].null_count())
print('France mean p2 old %.4f new %.4f; p2>=0.7 old %d new %d' % (oldfr['p2'].mean(), fr['p2'].mean(), (oldfr['p2']>=0.7).sum(), (fr['p2']>=0.7).sum()))
tree = pl.concat([old.join(frs, on='s1', how='anti'), fr]); tree.write_parquet('work2/test_scores_tree_frnorm2.parquet')
ce3 = pl.read_parquet('work2/test_scores_ce3.parquet', columns=['s1','t','p2'])
pl.concat([ce3.join(frs, on='s1', how='anti'), fr.select('s1','t','p2')]).write_parquet('work2/test_scores_frnorm2_routed.parquet')"
echo "[$(date +%T)] cells (tree, France re-featurised)"
QUIET=1 $PY $S/cells.py work/oof_dense_s1.parquet work2/test_scores_tree_frnorm2.parquet work2/c_trfr2 > logs2/cells_trfr2.log 2>&1
SCORES=work2/test_scores_frnorm2_routed.parquet CF=work2/c_ce3_cells.parquet CT=work2/c_trfr2_cells.parquet OUT=work2/out_frnorm2_shift CANDS=0 $PY $S/apply_shift.py | tee logs2/shift_frnorm2.log
$PY $S/apply_lfveto.py work2/out_frnorm2_shift output_shift_frnorm2 | tee logs2/lfveto_frnorm2.log
python3 utils/validate_submission.py --matching output_shift_frnorm2/matching_results.tsv --candidate output_shift_frnorm2/candidate_pairs.tsv --test-dir dataset/test --check-ids | tail -1
sha256sum output_shift_frnorm2/matching_results.tsv | tee -a logs2/hashes.txt
$PY work2/src/diag_expf2.py output_shift_frnorm2/scores.parquet 2>&1 | head -3
echo "[$(date +%T)] post done"
