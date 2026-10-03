#!/bin/bash
# End-to-end pipeline: raw TSVs -> blocking -> features -> 2-stage model -> output/*.tsv
# Run from the challenge root (the directory that contains dataset/). Intermediate files go to work/.
# Resources used for the reference run: 32 CPU cores, 160 GB RAM, ~3 hours wall-clock.
set -euo pipefail
ROOT=$(pwd)
SRC=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
NW=${NW:-5}                      # parallel feature workers (~26 GB RAM each)
[[ "$NW" =~ ^[1-9][0-9]*$ ]] || { echo "NW must be a positive integer" >&2; exit 1; }
for split in train test; do
  for source in 1 2 3; do
    input="dataset/$split/${split}_source$source.tsv"
    [ -f "$input" ] || { echo "Missing $input; run from the challenge root with the dataset installed." >&2; exit 1; }
  done
done
for input in dataset/train/train_ground_truth.tsv utils/validate_submission.py; do
  [ -f "$input" ] || { echo "Missing $input; see docs/reproduction.md." >&2; exit 1; }
done
mkdir -p work output
cp "$SRC"/*.py work/             # scripts import each other from work/
export PYTHONHASHSEED=0

$PY work/load.py                              # TSV -> parquet (no quoting, exact row counts)
$PY work/learn_dict.py                        # learn Indic-script -> Latin word dictionary from TRAIN pairs
$PY work/prep.py                              # normalized name/address forms for all records
$PY work/gt_pairs.py                          # ground-truth pair table (int ids)
$PY work/raw_pool.py train 0.05 work/rawpool_train5.parquet   # raw blocking pool on 5% of train S1
$PY work/train_ranker.py work/rawpool_train5.parquet work/ranker.txt  # learned blocking ranker
$PY work/make_keep.py                         # train S1 dropout (18.6%) to mimic the test S1/target ratio
$PY work/run_block.py train 300 30 work/keep_trainD.parquet trainD   # top-30 candidates per kept train S1
$PY work/run_block.py test 300 30             # top-30 candidates per S1 (test)
$PY work/rec_attrs.py train work/keep_trainD.parquet trainD
$PY work/rec_attrs.py test
feature_workers() {
  local split=$1 dir=$2 tag=$3 w pid failed=0
  local pids=()
  for ((w=0; w<NW; w++)); do
    PTAG=$tag POLARS_MAX_THREADS=6 $PY work/build_feats_all.py "$split" 30 "$dir" "$w" "$NW" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}
feature_workers train work/feat_trainD trainD
feature_workers test work/feat_test test
TAG=final SUB=0.5 FEAT=work/feat_trainD ATTRS=trainD $PY work/train_stages.py   # 2-stage LightGBM, 4 S1-grouped folds
KEEP=work/keep_trainD.parquet $PY work/eval_oof.py work/oof_final.parquet p2       # validation macro F0.5 (exact metric)
TAG=final TH=0.7 EXCL=1 OUT=output $PY work/predict_test.py
$PY utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
