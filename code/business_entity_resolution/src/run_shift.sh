#!/bin/bash
# Test-shift extension on top of run_all.sh + run_neural.sh (see RESEARCH_LOG_TEST_SHIFT.md). Run from the challenge
# root. Needs the artefacts of run_neural.sh: work/oof_dense_s1*.parquet, work/test_scores_dense_s1.parquet,
# work/models_dense_s1*, work/ce/{qwen,xlmr}_h{0,1}, work/ce/gate_*_dense_s1.parquet, work/ce/ce_*_qwen_dense_s1.parquet.
#   A. second Qwen3 half-model on test (test CE = mean of both halves)
#   B. XLM-R-base CE scores on the final gate (classic Ditto backbone, trained in run_neural's E4 recipe)
#   C. in-band CEs: XLM-R-base and Qwen3-Reranker-0.6B trained on the labelled gated band of their half
#      (in-distribution hard negatives), cross-fitted; test = mean of both half-models
#   D. stage 2 with four CE blocks (Qwen3, XLM-R, in-band XLM-R, in-band Qwen3), same folds / seed / SUB 1.0
#   E. countries without training labels: re-featurised with an extended normaliser and scored by the tree
#      (run_unseen_norm.sh); label-free test-shift correction + legal-form veto (gated)
#   F. type-word veto (unlabelled countries), expected-F0.5 decoding (labelled countries), duplicate consistency
#   G. empty-S1 rescue (unlabelled countries) -> output/
# Every step is resumable (logs/<step>.done markers). GPU: one CUDA device (peak ~26 GB when two jobs share it).
set -euo pipefail
PY=${PY:-python3}; SRC=$(cd "$(dirname "$0")" && pwd)
# The supplied run_neural.sh uses the dense tag, while this historical extension
# expects dense_s1. Require its actual inputs rather than silently assuming equivalence.
for input in work/oof_dense_s1.parquet work/oof_dense_s1_ce_qwen.parquet \
    work/test_scores_dense_s1.parquet work/models_dense_s1/cols.pkl \
    work/ce/gate_train_h0_dense_s1.parquet work/ce/gate_train_h1_dense_s1.parquet \
    work/ce/ce_train_qwen_dense_s1.parquet work/ce/ce_test_qwen_dense_s1.parquet \
    work/ce/qwen_h0/head.pt work/ce/qwen_h1/head.pt utils/validate_submission.py; do
  [ -f "$input" ] || { echo "Missing $input; this extension requires the dense_s1 artifacts. See docs/reproduction.md." >&2; exit 1; }
done
mkdir -p work work2 logs output
cp "$SRC"/*.py work/
export PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { local name=$1; shift; if [ -f "logs/$name.done" ]; then echo "skip $name"; return; fi
  echo "[$(date +%T)] $name"; "$@" > "logs/$name.log" 2>&1; touch "logs/$name.done"; }
Q=Qwen/Qwen3-Reranker-0.6B; X=FacebookAI/xlm-roberta-base

# ---- A. Qwen3 half-model 1 on the test gate; test CE = mean(m0, m1)
step sh_pairs $PY -c "import polars as pl; pl.read_parquet('work/ce/ce_test_qwen_dense_s1.parquet').select('s1','t').write_parquet('work2/ce_test_pairs_final.parquet')"
step sh_q_m1 env CE_BACKBONE=$Q $PY work/nn_ce.py score work2/ce_test_pairs_final.parquet test work/ce/qwen_h1 work2/sc_test_final_m1.parquet
step sh_q_avg $PY work/ce_avg.py

# ---- B. XLM-R-base CE (E4 recipe: trained on run_neural's hard-negative pairs) scored on the final gate. The shipped
#      run reused E4's scores for the 90 % of pairs already gated by the base model and scored only the rest
#      (xl_assemble.py); this script scores the whole gate, which gives the same numbers up to GPU non-determinism.
for h in 0 1; do
  [ -f work/ce/xlmr_h$h/head.pt ] || step sh_x_train_h$h env CE_BACKBONE=$X CE_BS=64 CE_LR=3e-5 CE_EPOCHS=1 CE_EVAL_EVERY=1500 \
      $PY work/nn_ce.py train work/ce/train_h$h.parquet train work/ce/xlmr_h$h work/ce/eval_h${h}_10k.parquet
done
step sh_x_tr0 env CE_BACKBONE=$X $PY work/nn_ce.py score work/ce/gate_train_h0_dense_s1.parquet train work/ce/xlmr_h1 work2/xlf_sc_train_h0.parquet
step sh_x_tr1 env CE_BACKBONE=$X $PY work/nn_ce.py score work/ce/gate_train_h1_dense_s1.parquet train work/ce/xlmr_h0 work2/xlf_sc_train_h1.parquet
step sh_x_te env CE_BACKBONE=$X $PY work/nn_ce.py score work2/ce_test_pairs_final.parquet test work/ce/xlmr_h0,work/ce/xlmr_h1 work2/xlf_sc_test.parquet
step sh_x_asm $PY -c "
import polars as pl
q = pl.read_parquet('work/ce/ce_train_qwen_dense_s1.parquet', columns=['s1', 't'])
x = pl.concat([pl.read_parquet(f'work2/xlf_sc_train_h{h}.parquet').select('s1', 't', pl.col('ce0').alias('ce')) for h in (0, 1)])
q.join(x, on=['s1', 't'], how='left').write_parquet('work2/ce_train_xlmr_final.parquet')
t = pl.read_parquet('work2/xlf_sc_test.parquet')
t.select('s1', 't', ((pl.col('ce0') + pl.col('ce1')) / 2).alias('ce')).write_parquet('work2/ce_test_xlmr_final.parquet')"

# ---- C. in-band XLM-R: train on the labelled gated band of half h, score half 1-h (OOF) and test (both models)
step sh_b_lab $PY -c "
import polars as pl
gt = pl.read_parquet('work/gt_pairs_int.parquet').with_columns(pl.lit(1).cast(pl.Int8).alias('y'))
for h in (0, 1):
    g = pl.read_parquet(f'work/ce/gate_train_h{h}_dense_s1.parquet').join(gt, on=['s1', 't'], how='left').with_columns(pl.col('y').fill_null(0))
    g.sort(pl.struct('s1', 't').hash(seed=7)).write_parquet(f'work2/gate_lab_h{h}.parquet')"
for h in 0 1; do
  step sh_b_train_h$h env CE_BACKBONE=$X CE_BS=64 CE_LR=3e-5 CE_EPOCHS=1 CE_EVAL_EVERY=5000 \
      $PY work/nn_ce.py train work2/gate_lab_h$h.parquet train work2/xlb_h$h work/ce/eval_h${h}_10k.parquet
done
step sh_b_tr0 env CE_BACKBONE=$X $PY work/nn_ce.py score work/ce/gate_train_h0_dense_s1.parquet train work2/xlb_h1 work2/xlb_sc_train_h0.parquet
step sh_b_tr1 env CE_BACKBONE=$X $PY work/nn_ce.py score work/ce/gate_train_h1_dense_s1.parquet train work2/xlb_h0 work2/xlb_sc_train_h1.parquet
step sh_b_te env CE_BACKBONE=$X $PY work/nn_ce.py score work2/ce_test_pairs_final.parquet test work2/xlb_h0,work2/xlb_h1 work2/xlb_sc_test.parquet
step sh_b_asm $PY -c "
import polars as pl
q = pl.read_parquet('work/ce/ce_train_qwen_dense_s1.parquet', columns=['s1', 't'])
b = pl.concat([pl.read_parquet(f'work2/xlb_sc_train_h{h}.parquet').select('s1', 't', pl.col('ce0').alias('ce')) for h in (0, 1)])
q.join(b, on=['s1', 't'], how='left').write_parquet('work2/ce_train_xlb.parquet')
t = pl.read_parquet('work2/xlb_sc_test.parquet')
t.select('s1', 't', ((pl.col('ce0') + pl.col('ce1')) / 2).alias('ce')).write_parquet('work2/ce_test_xlb.parquet')"

# in-band Qwen3-Reranker-0.6B (same data and cross-fitting; ~1.6 h per half on one RTX 6000 Ada)
for h in 0 1; do
  step sh_q_train_h$h env CE_BACKBONE=$Q CE_BS=64 CE_LR=2e-5 CE_EPOCHS=1 CE_EVAL_EVERY=5000 \
      $PY work/nn_ce.py train work2/gate_lab_h$h.parquet train work2/qib_h$h work/ce/eval_h${h}_10k.parquet
done
step sh_q_tr0 env CE_BACKBONE=$Q $PY work/nn_ce.py score work/ce/gate_train_h0_dense_s1.parquet train work2/qib_h1 work2/qib_sc_train_h0.parquet
step sh_q_tr1 env CE_BACKBONE=$Q $PY work/nn_ce.py score work/ce/gate_train_h1_dense_s1.parquet train work2/qib_h0 work2/qib_sc_train_h1.parquet
step sh_q_te env CE_BACKBONE=$Q $PY work/nn_ce.py score work2/ce_test_pairs_final.parquet test work2/qib_h0,work2/qib_h1 work2/qib_sc_test.parquet
step sh_q_asm $PY -c "
import polars as pl
q = pl.read_parquet('work/ce/ce_train_qwen_dense_s1.parquet', columns=['s1', 't'])
b = pl.concat([pl.read_parquet(f'work2/qib_sc_train_h{h}.parquet').select('s1', 't', pl.col('ce0').alias('ce')) for h in (0, 1)])
q.join(b, on=['s1', 't'], how='left').write_parquet('work2/ce_train_qib.parquet')
t = pl.read_parquet('work2/qib_sc_test.parquet')
t.select('s1', 't', ((pl.col('ce0') + pl.col('ce1')) / 2).alias('ce')).write_parquet('work2/ce_test_qib.parquet')"

# ---- D. stage 2 with four CE blocks (work2/models_ce4q, work2/oof_ce4q.parquet), test scores
DF="FEAT=work/feat_trainD,work/feat_trainD_dense ATTRS=trainD EXTRA_PAIRFEATS=work/dense/pairfeats_trainD.parquet"
step sh_s2 env $DF TAG=ce4q BASE_TAG=dense_s1 SEED=1 SUB=1.0 CE_SCORES=work/ce/ce_train_qwen_dense_s1.parquet \
    CE2_SCORES=work2/ce_train_xlmr_final.parquet CE3_SCORES=work2/ce_train_xlb.parquet CE4_SCORES=work2/ce_train_qib.parquet $PY work/train_stage2_ce2.py
step sh_eval env KEEP=work/keep_trainD.parquet $PY work/compare_oof.py work/oof_dense_s1_ce_qwen.parquet:p2 work2/oof_ce4q.parquet:p2
step sh_pred env FEAT=work/feat_test,work/feat_test_dense EXTRA_PAIRFEATS=work/dense/pairfeats_test.parquet MODELDIR=work2/models_ce4q \
    BASE_TAG=dense_s1 CE_SCORES=work2/ce_test_avg.parquet CE2_SCORES=work2/ce_test_xlmr_final.parquet CE3_SCORES=work2/ce_test_xlb.parquet \
    CE4_SCORES=work2/ce_test_qib.parquet SC_OUT=work2/test_scores_ce4q.parquet TAG=ce4q $PY work/predict_ce2.py

# ---- E. countries without training labels: extended normaliser + tree (run_unseen_norm.sh); correction + veto -> output/
step sh_unseen env PY="$PY" bash "$SRC/run_unseen_norm.sh"   # PY must be an absolute path or on PATH (the script cd-s)
step sh_route $PY -c "
import polars as pl
s1 = pl.read_parquet('work/test_s1.parquet', columns=['entity_id', 'country']).with_columns((pl.col('entity_id').str.slice(3).cast(pl.Int64) + 10_000_000_000).alias('s1'))
seen = set(pl.read_parquet('work/train_s1.parquet', columns=['country'])['country'].unique().to_list())
un = s1.filter(~pl.col('country').is_in(list(seen))).select('s1')
U = pl.read_parquet('work2/test_scores_tree_unseen.parquet', columns=['s1', 't', 'p2'])
T = pl.read_parquet('work/test_scores_dense_s1.parquet', columns=['s1', 't', 'p2'])
A = pl.read_parquet('work2/test_scores_ce4q.parquet', columns=['s1', 't', 'p2'])
pl.concat([T.join(un, on='s1', how='anti'), U]).write_parquet('work2/test_scores_tree_final.parquet')
pl.concat([A.join(un, on='s1', how='anti'), U]).write_parquet('work2/test_scores_final_ce4q.parquet')"
step sh_cells env QUIET=1 $PY work/cells.py work2/oof_ce4q.parquet work2/test_scores_ce4q.parquet work2/c_ce4q
step sh_cells_tree env QUIET=1 $PY work/cells.py work/oof_dense_s1.parquet work2/test_scores_tree_final.parquet work2/c_tree_final
step sh_shift env SCORES=work2/test_scores_final_ce4q.parquet CF=work2/c_ce4q_cells.parquet CT=work2/c_tree_final_cells.parquet \
    OUT=work2/out_final_ce4q CANDS=0 $PY work/apply_shift.py
step sh_lfveto $PY work/apply_lfveto.py work2/out_final_ce4q work2/out_lf_ce4q
# ---- F. (RESEARCH_LOG_TEST_SHIFT.md §20-§22) type-word-swap veto for unlabelled countries; expected-F0.5 decoding for
#         labelled countries; duplicate-record consistency
step sh_twveto env PREP=work_unseen FEATU=work_unseen/feat_testun,work_unseen/feat_testun_dense $PY work/apply_twveto.py work2/out_lf_ce4q work2/out_tw_ce4q
step sh_efdec $PY work/final_ef.py work2/out_tw_ce4q work2/out_ef_ce4q
step sh_dupfix $PY work/dupfix.py work2/out_ef_ce4q work2/out_dup_ce4q
# ---- G. (RESEARCH_LOG_TEST_SHIFT.md §23-§24) empty-S1 rescue for unlabelled countries -> output/
step sh_rescue env PREP=work_unseen $PY work/rescue_empty_unseen.py work2/out_dup_ce4q work2/out_ef_ce4q output
$PY utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
