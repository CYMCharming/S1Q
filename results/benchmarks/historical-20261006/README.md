# Completed S1Q benchmark evidence

This directory contains aggregate metrics only. It contains no questions, individual predictions, checkpoints, or private paper files.

## Reading the results

Each ranking fixes the experiment batches, precision, dataset list, method group and complete common model coverage in `ranking_scopes.json`. Accuracy is the mean of dataset accuracy within a model, then the mean across models. It is not pooled accuracy across decisions, and no missing cells are imputed. Old and expanded protocols have separate scope IDs.

The current public method is **S1Q**; the internal historical ID `s1q-mac` is normalized to `s1q`. `s1q-local` and `s1q2-beta05` are S1Qv1 and S1Qv2. Additional enhancement / repair methods are preserved in `metrics.csv` but are not substituted for the fixed main comparison group.

All resource-independent accuracy rows here use floating-point quantize/dequantize execution. SmoothQuant*, AWQ*, GPTQ-block*, and SpinQuant* denote repository adaptations, not official method reproductions. A method appears in a comparison only when explicitly declared by that scope and completely measured. This evidence establishes neither integer-kernel acceleration nor state-of-the-art superiority. NanoJev measures reference-action compatibility and is excluded from text decision rankings.

Point rankings do not establish statistical superiority; no aggregate confidence interval is fabricated. Earlier exploratory Dev and external sub-cohort reuse is identified in the input protocol. See `provenance.json` for input hashes and audited-summary agreement.

Bold values mark the best quantized displayed value, including ties after rounding; accuracy is higher-is-better and NLL/Brier/ECE15 are lower-is-better. Exact ranks use unrounded values.

## original-w4a4-threecohort-20261004

Precision: W4A4; comparison group: all11 (11 methods); 3 complete common models; 3 required datasets/cohorts; 9 model × dataset cells.

Required datasets: development, wanli, mmlu-pro.

| Rank | Method | Accuracy % | NLL | Brier | ECE15 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | S1Q (current) | **42.72** | **1.541** | **0.706** | 0.178 |
| 2 | S1Q-Joint | 42.50 | 1.575 | 0.708 | 0.173 |
| 3 | S1Q-Margin | 42.23 | 1.589 | 0.714 | 0.187 |
| 4 | S1Q-AC | 41.44 | 1.568 | 0.716 | 0.184 |
| 5 | SpinQuant Had* | 39.83 | 1.578 | 0.725 | **0.155** |
| 6 | S1Qv1 | 37.57 | 1.676 | 0.750 | 0.188 |
| 7 | AWQ* | 37.07 | 1.657 | 0.754 | 0.190 |
| 8 | S1Qv2 | 36.30 | 1.631 | 0.751 | 0.188 |
| 9 | SpinQuant no-Had* | 33.98 | 1.768 | 0.810 | 0.216 |
| 10 | RTN | 30.14 | 2.100 | 0.887 | 0.274 |
| 11 | GPTQ-block* | 29.18 | 2.067 | 0.886 | 0.274 |

Excluded incomplete models: kev-9b. Missing cells are recorded explicitly in `ranking_scopes.json`.

## newfamily-w4a4-threecohort-20261006

Precision: W4A4; comparison group: all11 (11 methods); 6 complete common models; 3 required datasets/cohorts; 18 model × dataset cells.

Required datasets: development, wanli, mmlu-pro.

| Rank | Method | Accuracy % | NLL | Brier | ECE15 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | S1Q-Margin | **36.81** | **1.569** | **0.734** | **0.134** |
| 2 | S1Q (current) | 36.45 | 1.615 | 0.741 | 0.163 |
| 3 | S1Q-AC | 36.29 | 1.592 | 0.737 | 0.148 |
| 4 | S1Q-Joint | 35.31 | 1.592 | 0.743 | 0.142 |
| 5 | AWQ* | 30.59 | 1.702 | 0.798 | 0.193 |
| 6 | SpinQuant Had* | 30.37 | 1.747 | 0.808 | 0.203 |
| 7 | S1Qv1 | 30.24 | 1.749 | 0.806 | 0.198 |
| 8 | S1Qv2 | 29.98 | 1.745 | 0.809 | 0.199 |
| 9 | GPTQ-block* | 28.96 | 2.027 | 0.910 | 0.300 |
| 10 | SpinQuant no-Had* | 27.75 | 1.855 | 0.859 | 0.244 |
| 11 | RTN | 26.37 | 2.055 | 0.930 | 0.323 |

## historical-w4a4-threecohort-20261006

Precision: W4A4; comparison group: all11 (11 methods); 9 complete common models; 3 required datasets/cohorts; 27 model × dataset cells.

Required datasets: development, wanli, mmlu-pro.

| Rank | Method | Accuracy % | NLL | Brier | ECE15 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | S1Q-Margin | **38.62** | **1.575** | **0.727** | **0.151** |
| 2 | S1Q (current) | 38.54 | 1.590 | 0.729 | 0.168 |
| 3 | S1Q-AC | 38.00 | 1.584 | 0.730 | 0.160 |
| 4 | S1Q-Joint | 37.71 | 1.586 | 0.731 | 0.153 |
| 5 | SpinQuant Had* | 33.52 | 1.690 | 0.780 | 0.187 |
| 6 | AWQ* | 32.75 | 1.687 | 0.783 | 0.192 |
| 7 | S1Qv1 | 32.68 | 1.725 | 0.787 | 0.195 |
| 8 | S1Qv2 | 32.09 | 1.707 | 0.790 | 0.195 |
| 9 | SpinQuant no-Had* | 29.83 | 1.826 | 0.843 | 0.235 |
| 10 | GPTQ-block* | 29.03 | 2.040 | 0.902 | 0.291 |
| 11 | RTN | 27.62 | 2.070 | 0.916 | 0.306 |

Excluded incomplete models: kev-9b. Missing cells are recorded explicitly in `ranking_scopes.json`.

## original-w3a4-dev-stress-20261004

Precision: W3A4; comparison group: all11 (11 methods); 2 complete common models; 1 required datasets/cohorts; 2 model × dataset cells.

Required datasets: development.

| Rank | Method | Accuracy % | NLL | Brier | ECE15 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | S1Q (current) | **52.09** | **1.029** | **0.597** | **0.137** |
| 2 | S1Q-Joint | 48.55 | 1.034 | 0.614 | 0.143 |
| 3 | S1Q-Margin | 47.27 | 1.137 | 0.647 | 0.165 |
| 4 | S1Q-AC | 46.46 | 1.059 | 0.629 | 0.160 |
| 5 | S1Qv2 | 44.37 | 1.100 | 0.663 | 0.194 |
| 6 | S1Qv1 | 42.28 | 1.200 | 0.701 | 0.213 |
| 7 | SpinQuant Had* | 41.80 | 1.453 | 0.767 | 0.259 |
| 8 | AWQ* | 41.16 | 1.171 | 0.692 | 0.213 |
| 9 | RTN | 39.23 | 1.840 | 0.861 | 0.342 |
| 10 | GPTQ-block* | 39.07 | 1.868 | 0.868 | 0.364 |
| 11 | SpinQuant no-Had* | 38.91 | 1.530 | 0.801 | 0.285 |

## Visualization

The figure displays `historical-w4a4-threecohort-20261006` only.

![Common-coverage accuracy, family rankings and probability metrics](benchmark_ranking.png)

[Vector export](benchmark_ranking.svg)

## Files

- `metrics.csv` and `metrics.json`: all supplied aggregate rows, including legacy enhancement variants and separate protocols.
- `rankings.csv`: exact equal-weight rankings with coverage, probability metrics and family slices.
- `model_scores.csv`: each model's equally weighted dataset averages for every declared comparison group.
- `ranking_scopes.json`: exact scope, required datasets, per-cell sample sizes and exclusions.
- `provenance.json`: input hashes, row counts and sanitized checkpoint / quantization identities.
- `figure_manifest.json`: figure-to-scope binding and output hashes.
