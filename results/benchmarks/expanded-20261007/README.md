# Completed S1Q benchmark evidence

This directory contains aggregate metrics only. It contains no questions, individual predictions, checkpoints, or private paper files.

## Reading the results

Each ranking fixes the experiment batches, precision, dataset list, method group and complete common model coverage in `ranking_scopes.json`. Accuracy is the mean of dataset accuracy within a model, then the mean across models. It is not pooled accuracy across decisions, and no missing cells are imputed. Old and expanded protocols have separate scope IDs.

The current public method is **S1Q**; the internal historical ID `s1q-mac` is normalized to `s1q`. `s1q-local` and `s1q2-beta05` are S1Qv1 and S1Qv2. Additional enhancement / repair methods are preserved in `metrics.csv` but are not substituted for the fixed main comparison group.

All resource-independent accuracy rows here use floating-point quantize/dequantize execution. SmoothQuant*, AWQ*, GPTQ-block*, and SpinQuant* denote repository adaptations, not official method reproductions. A method appears in a comparison only when explicitly declared by that scope and completely measured. This evidence establishes neither integer-kernel acceleration nor state-of-the-art superiority. NanoJev measures reference-action compatibility and is excluded from text decision rankings.

Point rankings do not establish statistical superiority; no aggregate confidence interval is fabricated. Earlier exploratory Dev and external sub-cohort reuse is identified in the input protocol. See `provenance.json` for input hashes and audited-summary agreement.

Bold values mark the best quantized displayed value, including ties after rounding; accuracy is higher-is-better and NLL/Brier/ECE15 are lower-is-better. Exact ranks use unrounded values.

## expanded-v3-w4a4-22source-20261007

Precision: W4A4; comparison group: all12 (12 methods); 10 complete common models; 22 required datasets/cohorts; 220 model × dataset cells.

Required datasets: agnews, amazon, arc, boolq, csqa, emotion, imdb, jevbench, mmlu, mnli, openbookqa, paws, qnli, sciq, semif, sst5, toolace, trec, tweet-offensive, wanli, wildjailbreak, yelp.

| Rank | Method | Accuracy % | NLL | Brier | ECE15 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | S1Q (current) | **54.11** | 1.079 | **0.581** | **0.166** |
| 2 | S1Q-AC | 53.20 | **1.075** | 0.586 | 0.168 |
| 3 | S1Q-Margin | 52.61 | 1.095 | 0.591 | 0.167 |
| 4 | S1Q-Joint | 52.16 | 1.103 | 0.595 | 0.168 |
| 5 | S1Qv2 | 42.52 | 1.319 | 0.702 | 0.206 |
| 6 | SpinQuant Had* | 42.28 | 1.312 | 0.700 | 0.206 |
| 7 | AWQ* | 42.19 | 1.305 | 0.702 | 0.208 |
| 8 | S1Qv1 | 41.60 | 1.332 | 0.709 | 0.211 |
| 9 | SpinQuant no-Had* | 37.15 | 1.485 | 0.776 | 0.249 |
| 10 | SmoothQuant* | 36.57 | 1.478 | 0.783 | 0.256 |
| 11 | GPTQ-block* | 35.22 | 1.666 | 0.840 | 0.304 |
| 12 | RTN | 35.09 | 1.669 | 0.843 | 0.302 |

## Visualization

The figure displays `expanded-v3-w4a4-22source-20261007` only.

![Common-coverage accuracy, family rankings and probability metrics](benchmark_ranking.png)

[Vector export](benchmark_ranking.svg)

Accuracy whiskers are conditional 95% intervals for evaluated samples of the fixed admitted models and source suites, using shared within-source whole-cluster draws. Models and sources are not resampled. Intervals have no multiple-comparisons adjustment and do not establish a universal winner. Paired differences use shared-draw paired intervals; marginal bar intervals must not be independently subtracted. The CI file SHA and interpretation are recorded in `figure_manifest.json`.

## Accuracy by model and dataset

[Complete accuracy tables](accuracy_table.md) · [Aggregate cells CSV](accuracy_by_dataset.csv) · [Table provenance](accuracy_table_manifest.json). Tables contain every declared source, Native reference rows and the same complete common method/model coverage as the primary scope.

## Files

- `metrics.csv` and `metrics.json`: all supplied aggregate rows, including legacy enhancement variants and separate protocols.
- `rankings.csv`: exact equal-weight rankings with coverage, probability metrics and family slices.
- `model_scores.csv`: each model's equally weighted dataset averages for every declared comparison group.
- `ranking_scopes.json`: exact scope, required datasets, per-cell sample sizes and exclusions.
- `provenance.json`: input hashes, row counts and sanitized checkpoint / quantization identities.
- `figure_manifest.json`: figure-to-scope binding and output hashes.
