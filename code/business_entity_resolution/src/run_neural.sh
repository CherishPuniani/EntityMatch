#!/bin/bash
# Neural extension on top of run_all.sh (run that first; it leaves work/models_final, oof_final, test_scores_final).
#   A. Ditto-style Qwen3-Reranker-0.6B cross-encoder (CE), 2-way cross-fitted, gated on stage-1 p1, stacked into stage 2
#   B. Qwen3-Embedding-0.6B dense-retrieval candidate channel over not-yet-assigned targets, stage 1+2 retrained (SUB 1.0)
#   C. CE features on top of B (re-gated; only newly gated pairs are scored)
#   D. country routing: countries absent from the training labels use the tree-only p2 (see route_by_country.py)
# Needs one CUDA GPU (developed on an RTX 6000 Ada, 48 GB; peak 29 GB) and ~160 GB RAM. Run from the challenge root.
# Wall-clock on the reference node: CE training 2 x ~40 min, CE scoring ~600 pairs/s, dense encode ~2,000 rec/s,
# dense retrain ~2 h CPU. Every step is resumable (logs/<step>.done markers).
set -euo pipefail
PY=${PY:-python3}; NW=${NW:-5}; BASE=${BASE_TAG:-final}
[[ "$NW" =~ ^[1-9][0-9]*$ ]] || { echo "NW must be a positive integer" >&2; exit 1; }
for input in "work/oof_$BASE.parquet" "work/test_scores_$BASE.parquet" work/pairs_trainD.parquet utils/validate_submission.py; do
  [ -f "$input" ] || { echo "Missing $input; complete run_all.sh first (see docs/reproduction.md)." >&2; exit 1; }
done
Q=Qwen/Qwen3-Reranker-0.6B
mkdir -p work/ce work/dense logs
export PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { local name=$1; shift; if [ -f "logs/$name.done" ]; then echo "skip $name"; return; fi
  echo "[$(date +%T)] $name"; "$@" > "logs/$name.log" 2>&1; touch "logs/$name.done"; }
feats() { local split=$1 dir=$2 tag=$3; local pids=()
  for ((w=0; w<NW; w++)); do NOFILTER=1 PTAG=$tag POLARS_MAX_THREADS=6 $PY work/build_feats_all.py $split 0 $dir $w $NW & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done; }

# ---- A. cross-encoder: hard-negative data (40k S1 per half), training, gated OOF/test scoring
step ce_data $PY work/nn_ce_data.py work/pairs_trainD.parquet work/ce 40000 3000
for h in 0 1; do
  $PY -c "import polars as pl; pl.read_parquet('work/ce/eval_h$h.parquet').head(10000).write_parquet('work/ce/eval_h${h}_10k.parquet')"
  step ce_train_h$h env CE_BACKBONE=$Q CE_BS=64 CE_LR=2e-5 CE_EPOCHS=1 CE_EVAL_EVERY=1500 \
      $PY work/nn_ce.py train work/ce/train_h$h.parquet train work/ce/qwen_h$h work/ce/eval_h${h}_10k.parquet
done
step ce_gate $PY work/nn_ce_gate.py make 0.005 0.995 work/oof_$BASE.parquet work/test_scores_$BASE.parquet work/ce
step ce_sc_tr0 env CE_BACKBONE=$Q $PY work/nn_ce.py score work/ce/gate_train_h0.parquet train work/ce/qwen_h1 work/ce/sc_train_h0_qwen.parquet
step ce_sc_tr1 env CE_BACKBONE=$Q $PY work/nn_ce.py score work/ce/gate_train_h1.parquet train work/ce/qwen_h0 work/ce/sc_train_h1_qwen.parquet
step ce_sc_te0 env CE_BACKBONE=$Q $PY work/nn_ce.py score work/ce/gate_test.parquet test work/ce/qwen_h0 work/ce/sc_test_qwen_m0.parquet
step ce_assemble $PY work/nn_ce_gate.py assemble work/ce qwen
# A-only system (blocking candidates + CE), kept as a fallback submission
step s2_ce env CE_WITHIN=1 TAG=ce_qwen_within BASE_TAG=$BASE SEED=1 SUB=0.5 CE_SCORES=work/ce/ce_train_qwen.parquet $PY work/train_stage2_ce.py

# ---- B. dense channel (Qwen3-Embedding-0.6B) over targets not assigned by the base model
step d_un_train $PY work/dense_cands.py unassigned train work/oof_$BASE.parquet work/dense/unassigned_train.parquet
step d_un_test $PY work/dense_cands.py unassigned test work/test_scores_$BASE.parquet work/dense/unassigned_test.parquet
step d_enc_train env IDS=work/dense/unassigned_train.parquet $PY work/nn_dense.py encode train work/dense/train_t_u.npy
step d_search_train $PY work/nn_dense.py search train work/dense/train_t_u.npy work/keep_trainD.parquet 20 work/dense/search_trainD.parquet
step d_enc_test env IDS=work/dense/unassigned_test.parquet $PY work/nn_dense.py encode test work/dense/test_t_u.npy
step d_search_test $PY work/nn_dense.py search test work/dense/test_t_u.npy all 20 work/dense/search_test.parquet
step d_cands_train $PY work/dense_cands.py build train work/dense/search_trainD.parquet work/pairs_trainD.parquet 10 trainD
step d_cands_test $PY work/dense_cands.py build test work/dense/search_test.parquet work/pairs_test.parquet 10 test
step d_feats_train feats train work/feat_trainD_dense trainD_dense
step d_feats_test feats test work/feat_test_dense test_dense
step d_train env TAG=dense SUB=1.0 FEAT=work/feat_trainD,work/feat_trainD_dense ATTRS=trainD \
    EXTRA_PAIRFEATS=work/dense/pairfeats_trainD.parquet $PY work/train_stages.py
step d_predict env TAG=dense TH=0.7 EXCL=1 OUT=output_dense FEAT=work/feat_test,work/feat_test_dense \
    EXTRA_PAIRFEATS=work/dense/pairfeats_test.parquet $PY work/predict_test.py

# ---- C. CE on top of B: re-gate on the dense model's p1, score only newly gated pairs, stage 2 with CE
step f_plan $PY work/nn_ce_regate.py plan 0.005 0.995 work/oof_dense.parquet work/test_scores_dense.parquet qwen dense
step f_sc_tr0 env CE_BACKBONE=$Q $PY work/nn_ce.py score work/ce/todo_train_h0_dense.parquet train work/ce/qwen_h1 work/ce/sc_todo_train_h0_dense_qwen.parquet
step f_sc_tr1 env CE_BACKBONE=$Q $PY work/nn_ce.py score work/ce/todo_train_h1_dense.parquet train work/ce/qwen_h0 work/ce/sc_todo_train_h1_dense_qwen.parquet
step f_sc_te0 env CE_BACKBONE=$Q $PY work/nn_ce.py score work/ce/todo_test_dense.parquet test work/ce/qwen_h0 work/ce/sc_todo_test_dense_qwen_m0.parquet
step f_merge $PY work/nn_ce_regate.py merge qwen dense 1
DF="FEAT=work/feat_trainD,work/feat_trainD_dense ATTRS=trainD EXTRA_PAIRFEATS=work/dense/pairfeats_trainD.parquet"
step f_train env $DF CE_WITHIN=1 TAG=dense_ce_qwen BASE_TAG=dense SEED=1 SUB=1.0 CE_SCORES=work/ce/ce_train_qwen_dense.parquet $PY work/train_stage2_ce.py
step f_eval env KEEP=work/keep_trainD.parquet $PY work/compare_oof.py work/oof_$BASE.parquet:p2 work/oof_dense_ce_qwen.parquet:p2
step f_predict env FEAT=work/feat_test,work/feat_test_dense EXTRA_PAIRFEATS=work/dense/pairfeats_test.parquet TAG=dense_ce_qwen \
    BASE_TAG=dense CE_SCORES=work/ce/ce_test_qwen_dense.parquet OUT=output_dense_ce_qwen $PY work/predict_test_ce.py

# ---- D. country routing (training countries: neural system; unseen countries: dense tree-only model) -> output/
step route $PY work/route_by_country.py work/test_scores_dense_ce_qwen.parquet work/test_scores_dense.parquet output
$PY utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
