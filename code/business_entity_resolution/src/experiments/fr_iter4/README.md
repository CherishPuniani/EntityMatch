# Iterations 4–5 (2026-09-27): analysis scripts as run

> Historical notes preserved below. Referenced research logs and run artifacts are not included;
> stated outcomes have not been reproduced. See the [experiment guide](../README.md) and
> [current reproduction guide](../../../../../docs/reproduction.md).

Results: `RESEARCH_LOG_TEST_SHIFT.md` §19–§22. These scripts are kept as a record; they contain absolute paths to the run
root (`/home2/home/amritanshu_t/amazon-mlc-26/run`) and to the session scratchpad
(`/tmp/claude-1022/-home2-home-amritanshu-t-amazon-mlc-26-Amazon-mlc-26/fe28c95e-d5a1-48c7-beb4-ebf7a1a8677e/scratchpad`,
subfolders `fr4/ fr5/ fr6/ final7/ final8/ xr/`). The production path is `../run_unseen_norm.sh`, `../apply_twveto.py`,
`../efdec.py`, `../final_ef.py`, `../dupfix.py` (wired into `../run_shift.sh`, step F).

| Script | Question |
|---|---|
| `filler_train.py`, `filler_sig.py` | same-address single-token additions: true rate (train) / acceptance (test) by token; replace:append signature |
| `copycount.py`, `acc_swap.py`, `cc_uncert.py`, `cc_acc.py`, `diff56.py`, `veto_audit.py` | copy-count test (label-free copy vs sibling check) |
| `fr_sarl.py`, `fr_sarl2.py`, `fr_lfdrop.py` | SARL IDF artefact |
| `fr_look.py`, `fr_band.py`, `fr_eqname.py`, `fr_fill_look.py`, `fr_accneq.py` | example dumps of French pairs by decision band |
| `run_fr4.sh`, `run_fr5.sh`, `run_fr6.sh`, `run_fr6a.sh`, `mk_pairfeats.py` | French rebuilds under /tmp (fr6 = rejected dense-nulling variant) |
| `fr4_eval.py`, `fr5_eval.py`, `why_rej.py`, `impact.py` | class acceptance before/after, bounded LB impact |
| `usin_gap.py`, `usin_tok.py`, `empty1.py`, `efdec_oof.py`, `rescue.py`, `sibclu.py` | US/India loss decomposition, decoding validation, sibling-cluster test |
| `blockmiss.py`, `xret.py`, `xr_build.py`, `orphan.py`, `alias.py` | candidate-recall ideas (all negative) |
| `leak1.py`, `leak2.py`, `dups.py` | ID / row order / train–test overlap (none) and duplicate consistency |
