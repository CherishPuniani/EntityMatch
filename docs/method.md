# Method and design decisions

## Task and metric

The code treats Source 1 as the query source and Sources 2 and 3 as the target pool. Records have `entity_id`, `business_name`, `business_address`, and `country` fields. A prediction is a variable-size target-ID set, including the empty set.

`metric.py` computes per-entity F0.5 as `5 * TP / (G + 4 * K)`, where `G` is the true match-set size and `K` is the predicted size. It returns 1 for empty truth and empty prediction, and 0 when only one is empty. Macro evaluation averages over the specified Source 1 population. False additions therefore matter strongly, and empty entities must remain in the evaluation.

## 1. Normalize multilingual records

`load.py` converts the seven input TSVs to Parquet with tab separation and quoting disabled. `learn_dict.py` learns Indic-to-Latin word mappings from aligned training pairs. `norm.py` adds rule-based fallback transliteration, Unicode folding, alias handling, legal-form and filler vocabularies, and address abbreviations. `prep.py` retains multiple text and token views rather than one aggressively stripped string.

This gives the matcher evidence for spelling changes, native-script names, reordered words, legal suffixes, and incomplete addresses. Missing addresses are represented explicitly in the neural serialization; they do not automatically imply disagreement.

## 2. Retrieve a manageable candidate set

`block.py` creates compound keys from rare name tokens, address tokens, and numbers, with country-aware keys. `raw_pool.py` samples a training candidate pool and `train_ranker.py` learns a LightGBM blocking ranker from key-type evidence. `run_block.py` keeps the top 30 candidates per query in the baseline.

Candidate retrieval imposes an upper bound on matching recall: the matcher cannot accept a target it never sees. The code includes recall-at-K and complete-match-set diagnostics. `make_keep.py` removes a deterministic subset of training Source 1 records to approximate the target-to-query ratio at test time.

## 3. Score lexical pairs with two-stage trees

`features.py` combines character similarity, exact and fuzzy token overlap, IDF-weighted evidence, address and house-number relations, aliases, and script indicators. `rec_attrs.py` adds record hashes and frequencies. `stages.py` adds context within a query and across queries competing for the same target.

`train_stages.py` assigns four folds by a hash of Source 1 ID. Stage 1 produces out-of-fold pair probabilities. Stage 2 uses those probabilities and contextual features such as within-query rank, target competition, and corroboration from similar candidates. Test prediction averages four fold models. The baseline accepts probabilities at least 0.7 and, when exclusivity is enabled, only the highest-ranked query for a target.

The split groups by Source 1 ID; it does not establish a fully independent split across every related target or business cluster. The transliteration dictionary and retrieval preparation are built before matcher cross-validation. OOF scores should therefore be interpreted as scores from this implemented validation procedure, not proof of a completely isolated preprocessing evaluation.

## 4. Add neural evidence selectively

`nn_ce_data.py` builds positive and retrieved-negative pairs. `nn_ce.py` fine-tunes pairwise cross-encoders using explicit name/address serialization. The Qwen3 reranker is trained in two halves; training pairs are scored by the model trained on the other half. Gates restrict expensive neural scoring to stage-1 probabilities between 0.005 and 0.995. `train_stage2_ce.py` stacks those scores into the tree matcher, with an option to restrict context to the same Source 1 entity.

`nn_dense.py` embeds records with Qwen3-Embedding and searches by country. `dense_cands.py` unions a dense candidate channel with lexical candidates; the tree models are then retrained on the union. Additional scripts stack XLM-R and in-band Qwen3/XLM-R scores. The in-band models train on the gated uncertainty region rather than the original negative sampling distribution.

## 5. Handle distribution shift and decode match sets

The later pipeline routes countries represented in training labels to the neural system and unseen countries to a tree model with extended normalization. `run_unseen_norm.sh` patches an isolated normalizer copy, including French legal forms, street abbreviations, administrative regions, and vocabulary mappings.

`cells.py` compares training positive rates and test candidate rates in country/house-number/probability strata. `apply_shift.py` adjusts selected strata; legal-form and type-word vetoes remove some likely sibling-business links. These rules use unlabeled test distributions and hand-crafted vocabulary assumptions. Their correctness for a new dataset is not established by this snapshot.

`efdec.py` chooses a match-set size to maximize expected F0.5 under independent Bernoulli candidate truths, subject to target exclusivity and probability gates. `dupfix.py` enforces consistency for exact duplicate targets. `rescue_empty_unseen.py` adds selected same-address candidates for unseen-country queries that otherwise have no links.

## Research scope

The available experiments cover feature variants, French normalization, uncertain-pair analysis, copy-versus-sibling signatures, candidate-recall probes, expected-metric decoding, duplicate checks, and empty-query rescue. The experiment directory records those investigations, including approaches described as rejected. Historical outcomes are not independently verified without their logs and intermediate files.
