#!/bin/bash
# fr6a = fr6 token mappings, original dense pair features (predict + assemble only)
set -euo pipefail
source <(sed -n '/^assemble() {/,/^}/p' /tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/fr6/run_fr6.sh)
ROOT=/home2/home/amritanshu_t/amazon-mlc-26/run; BASE=/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/fr6; PY=/home2/home/amritanshu_t/amazon-mlc-26/venv/bin/python
export PYTHONHASHSEED=0 POLARS_MAX_THREADS=24 ROOT BASE
cd $BASE/fx_unseen
echo "[$(date +%T)] predict fr6a"
TAG=dense_s1 TH=0.7 EXCL=1 OUT=out_unseen_a FEAT=work/feat_testun,work/feat_testun_dense EXTRA_PAIRFEATS=work/dense/pairfeats_test.parquet $PY work/predict_test.py > $BASE/predict_unseen_a.log 2>&1
cp work/test_scores_dense_s1.parquet $BASE/test_scores_tree_unseen_fr6a.parquet
cd $ROOT
echo "[$(date +%T)] asm fr6a"
assemble $BASE/test_scores_tree_unseen_fr6a.parquet $BASE/asm_a $BASE/output_fr6a_noveto
PREP=$BASE/work_unseen FEATU=$BASE/work_unseen/feat_testun,$BASE/work_unseen/feat_testun_dense $PY /tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad/fr4/apply_twveto.py $BASE/output_fr6a_noveto $BASE/output_fr6a | grep -v "type-word vocabulary"
python3 utils/validate_submission.py --matching $BASE/output_fr6a/matching_results.tsv --candidate $BASE/output_fr6a/candidate_pairs.tsv --test-dir dataset/test --check-ids | tail -1
echo "[$(date +%T)] fr6a done"
