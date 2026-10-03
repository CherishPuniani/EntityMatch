#!/bin/bash
# France-ensemble variant: US/India = ce3 (validated), unlabelled countries = mean(tree-only dense_s1, ce3) p2.
set -euo pipefail
PY=../venv/bin/python; export PYTHONHASHSEED=0 POLARS_MAX_THREADS=24; S=work2/src
echo "[$(date +%T)] averaged OOF / test scores"
$PY -c "
import polars as pl
a = pl.read_parquet('work/oof_dense_s1.parquet', columns=['s1','t','y','fold','p1','p2'])
b = pl.read_parquet('work2/oof_ce3.parquet', columns=['s1','t','p2']).rename({'p2':'q'})
m = a.join(b, on=['s1','t'], how='inner'); assert len(m) == len(a) == len(b)
m.with_columns(((pl.col('p2') + pl.col('q')) / 2).alias('p2')).drop('q').write_parquet('work2/oof_avg.parquet')
a = pl.read_parquet('work/test_scores_dense_s1.parquet', columns=['s1','t','p1','p2'])
b = pl.read_parquet('work2/test_scores_ce3.parquet', columns=['s1','t','p2']).rename({'p2':'q'})
m = a.join(b, on=['s1','t'], how='inner'); assert len(m) == len(a) == len(b)
m.with_columns(((pl.col('p2') + pl.col('q')) / 2).alias('p2')).drop('q').write_parquet('work2/test_scores_avg.parquet')
s1 = pl.read_parquet('work/test_s1.parquet', columns=['entity_id','country']).with_columns((pl.col('entity_id').str.slice(3).cast(pl.Int64)+10_000_000_000).alias('s1'))
seen = set(pl.read_parquet('work/train_s1.parquet', columns=['country'])['country'].unique().to_list())
un = s1.filter(~pl.col('country').is_in(list(seen))).select('s1')
A = pl.read_parquet('work2/test_scores_ce3.parquet', columns=['s1','t','p2']); B = pl.read_parquet('work2/test_scores_avg.parquet', columns=['s1','t','p2'])
pl.concat([A.join(un, on='s1', how='anti'), B.join(un, on='s1')]).write_parquet('work2/test_scores_frens.parquet')"
KEEP=work/keep_trainD.parquet $PY work/compare_oof.py work2/oof_ce3.parquet:p2 work2/oof_avg.parquet:p2 > logs2/cmp_ce3_avg.log 2>&1 &
QUIET=1 $PY $S/cells.py work2/oof_avg.parquet work2/test_scores_avg.parquet work2/c_avg > logs2/cells_avg.log 2>&1
wait
grep -E "^new|DELTA" logs2/cmp_ce3_avg.log
SCORES=work2/test_scores_frens.parquet CF=work2/c_ce3_cells.parquet CT=work2/c_avg_cells.parquet OUT=work2/out_frens_shift CANDS=0 $PY $S/apply_shift.py | tee logs2/shift_frens.log
$PY $S/apply_lfveto.py work2/out_frens_shift output_shift_frens | tee logs2/lfveto_frens.log
python3 utils/validate_submission.py --matching output_shift_frens/matching_results.tsv --candidate output_shift_frens/candidate_pairs.tsv --test-dir dataset/test --check-ids | tail -1
sha256sum output_shift_frens/matching_results.tsv | tee -a logs2/hashes.txt
echo "[$(date +%T)] done"
