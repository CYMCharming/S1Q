# S1Q expanded decision benchmark

**Completed: 10 text decision models × 22 source suites × 12 W4A4 quantization methods, plus Native references.** The final aggregate has zero metric-audit errors and complete common coverage. This document records the measured results, frozen protocol and actual admission counts. Historical three-cohort results remain a separate snapshot.

## What was added

The new benchmark contains **22 source suites, 2,565 complete requests and 2,871 decisions before model-specific admission**. A request can contain multiple questions, so request and decision counts differ. These are bounded, screened subsets rather than complete upstream benchmark or leaderboard evaluations.

The suites comprise **18 standard dataset sources, two authored scenario/rule suites, and two domain decision suites**. SemIf and JevBench are authored decision benchmarks; counting all 22 as conventional independent classification datasets would be misleading. Different task templates, JevBench difficulties and editions of a composite Decision Index are not additional independent datasets.

| Source suite | Prepared requests | Prepared decisions | Evaluation target / source category |
| --- | ---: | ---: | --- |
| AG News | 120 | 360 | Topic decisions, including the source's multiple question types; standard dataset |
| Amazon Reviews | 128 | 128 | Source-native sentiment score decisions; standard dataset |
| ARC | 128 | 128 | Science multiple-choice questions; standard dataset |
| BoolQ | 120 | 120 | Boolean reading-comprehension decisions; standard dataset |
| CommonsenseQA | 128 | 128 | Commonsense multiple-choice questions; standard dataset |
| Emotion | 120 | 120 | Emotion classification; standard dataset |
| IMDb | 53 | 53 | Sentiment decisions; standard dataset |
| JevBench | 48 | 48 | Authored scenarios; **easy subset only** after excluding previously evaluated requests |
| MMLU | 99 | 99 | Knowledge multiple-choice questions; standard dataset, distinct from the old MMLU-Pro short cohort |
| MNLI | 122 | 122 | Natural-language inference; standard dataset |
| OpenBookQA | 128 | 128 | Science multiple-choice questions; standard dataset |
| PAWS | 128 | 128 | Paraphrase decisions; standard dataset |
| QNLI | 128 | 128 | Question/passage entailment; standard dataset |
| SciQ | 92 | 92 | Science multiple-choice questions; standard dataset |
| SemIf | 128 | 128 | Authored rule/scenario decisions; 32 source groups |
| SST-5 | 128 | 128 | Source-native sentiment score decisions; standard dataset |
| ToolACE | 68 | 68 | **Tool-choice classification**, using native references; does not measure tool-execution success |
| TREC | 121 | 121 | Question-type classification; standard dataset |
| TweetEval Offensive | 128 | 128 | Offensive-language decisions; standard dataset |
| WANLI | 256 | 256 | Natural-language inference; new requests from the frozen upstream pool |
| WildJailBreak | 128 | 128 | Native-reference harmful/benign classification; does not measure generated-response safety or jailbreak attack success |
| Yelp | 66 | 132 | Source-native score and Boolean sentiment questions; standard dataset |
| **Total** | **2,565** | **2,871** | **22 source suites before model-specific admission** |

The frozen data-integrity audit verifies all 22 hashes and exact request/gold preservation. A separately reproduced preparation produced 22 byte-identical dataset files. The prepared manifest SHA-256 is `41e4ccc75326baaca3212085dd66e0127fb8bc122d5b06054a170a0a2a0a231d` (`expanded-native-decision-20261007-v2`). The completed [ranking scope](../results/benchmarks/expanded-20261007/ranking_scopes.json), [interval admission counts](../results/benchmarks/expanded-20261007/accuracy-uncertainty/accuracy_uncertainty.json) and [220-cell Native/source audit](../results/benchmarks/expanded-20261007/audits/native-source-binding-10models.json) confirm that **every evaluated model actually admitted all 2,565 requests and 2,871 decisions**. Thus prepared and actual admitted counts match in this batch; they remain distinct recorded quantities.

## Upstream evidence and frozen sources

The source request files are taken from the projects' released decision evaluation suites:

- [Kev](https://github.com/jaredpalmer/kev/tree/0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91): pinned decision-v7, transfer-v4, public-pool-v6 and external WANLI / SemIf source files. These files already encode the state, question type, candidates, scoring distribution and reference labels used by decision models.
- [Intern-Decision evaluation](https://github.com/InternLM/Intern-Decision/blob/3572c8a68b5df5dafe02d0e093989ba8ec0183bc/docs/EVALUATION.md): pinned `accuracy-v1` manifest and source files for JevBench, ToolACE and WildJailBreak. The manifest identifies AG News and Typed Decisions as well; duplicate source families or difficulty tiers are not counted twice.
- [StartLux-Decision model card](https://huggingface.co/startlux-models/StartLux-Decision-0.8B): its reported decision-suite coverage guides the benchmark scope. Reported composite Decision Index editions are not independent datasets or additional S1Q observations.

Raw source requests stay local. Public outputs should contain aggregate metrics, preparation code, source revisions, file hashes, admission counts, protocol metadata and figures. Private paper files, model checkpoints, individual predictions and raw dataset records are not part of this release.

## Screening and leakage boundary

The expanded requests are selected **without looking at model predictions**. The frozen selection policy uses SHA-256 ordering of complete source groups, prefers upstream test records where available, and discloses unexamined upstream development supplements. It preserves the complete original state/scenario, question schema, candidate order and reference labels. Requests are not shortened, rewritten or converted into easier candidate subsets.

The prepared suite admits clean requests with at most 10 choices and at most 1,000 characters under the documented preparation rule. File deduplication and whole-group exclusion prevent repeated upstream requests, calibration overlap, and reuse of earlier locally evaluated request/state/scenario identities. Group metadata is retained for paired cluster analysis. Model-specific tokenizer and API admission was measured for every completed run and is recorded in its source scope.

The old **Mixed Dev** cohort is read only as a legacy diagnostic by the existing runner. It is explicitly excluded from the new ranking dataset list and from its 22-source average. Previously inspected WANLI requests are also excluded from the new WANLI selection. Upstream files named `development` may supply new, disclosed requests; that does not make them the old S1Q Mixed Dev cohort.

This boundary establishes disjointness from the recorded local calibration and previous inference requests. It does **not** certify absence from model pretraining, absence from all upstream training data, or independence from every external publication. Public JevBench tasks remain public authored diagnostics.

JevBench deserves a specific qualification: the source pool has 231 tasks. The preparation excludes 85 previously evaluated tasks and 98 overlong tasks, leaving **48 easy tasks**. There is no new original/hard JevBench score in this expansion.

Typed Decisions is excluded from the main ranking because its 2,000 supplied decisions use **teacher-agreement labels**, not independent task reference labels. All 400 source requests also exceed this preparation's character limit, leaving zero admitted requests. It has **not been evaluated** under this expanded protocol; neither a native-label score nor a teacher-agreement diagnostic should be fabricated.

## Frozen methods and fair aggregation

The comparison uses the recorded frozen recipes and the same admitted calibration requests, declared protected parameter categories and retained parameter counts, selected Linear scope and quantization setting for each matched model run. New evaluation labels are not used for quantizer fitting or method selection. Each run records its model revision, method list, precision, group size, budgets, source hashes, implementation hashes and original/native admission exclusions. These declarations do not establish bytewise protected-tensor identity; no protected-tensor content-hash audit is claimed.

The expanded principal comparison group has **12 methods**: RTN, S1Qv1, S1Qv2, SmoothQuant*, AWQ*, GPTQ-block*, SpinQuant no-Had*, SpinQuant Had*, S1Q-Joint, S1Q-AC, S1Q-Margin and current **S1Q**. Historical comparisons retain their measured 11-method group; a missing historical SmoothQuant result is not invented. The public current method is called S1Q; the historical internal alias `s1q-mac` is normalized to `s1q`. The asterisk marks repository adaptations of baseline methods, rather than official complete reproductions.

Accuracy here is labeled decision accuracy. A source's accuracy aggregates its admitted decisions; multi-question requests retain their original questions and cluster identity. The main average gives **each of the 22 source suites equal weight within a model**, then gives each model equal weight. It is not pooled accuracy over all requests or decisions. A small suite and WANLI therefore receive equal source weight; exact denominators remain visible.

Each ranking fixes its batches, precision, required 22-source list, method group and common complete model coverage. All compared methods must evaluate the same admitted requests within a model/source cell. Missing or failed cells are recorded explicitly, never imputed as zero. Comparisons with a different method or model coverage have a separate scope. Old three-cohort scores and new 22-source scores are never combined into a single average.

NLL, Brier and ECE15 are reported separately, with lower values preferred. They are not arbitrarily blended with accuracy into one numerical composite. Point ranks are descriptive; statistical superiority requires the corresponding paired uncertainty analysis. Quantization accuracy uses floating-point QDQ execution and does not imply integer-kernel acceleration or measured packed-model memory savings.

## Completed expanded comparison

The principal matrix contains **10 text decision models × 22 source suites × 12 quantization methods at W4A4**, plus Native references: Kev-0.8B/4B/9B, Laya, Intern-Decision-0.8B/2B/4B and StartLux-Decision-0.8B/2B/4B. **Kev-27B and NanoJev are not in this new matrix.** All methods share the verified source admissions within every model/source cell. The report uses scope `expanded-v3-w4a4-22source-20261007`, group `all12`, with each source and then each model averaged equally.

| Method | Accuracy % ↑ | Rank | NLL ↓ | Rank | Brier ↓ | Rank | ECE15 ↓ | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native reference | 76.38 | — | 0.734 | — | 0.345 | — | 0.131 | — |
| S1Q (current) | **54.11** | 1 | 1.079 | 2 | **0.581** | 1 | **0.166** | 1 |
| S1Q-AC | 53.20 | 2 | **1.075** | 1 | 0.586 | 2 | 0.168 | 3 |
| S1Q-Margin | 52.61 | 3 | 1.095 | 3 | 0.591 | 3 | 0.167 | 2 |
| S1Q-Joint | 52.16 | 4 | 1.103 | 4 | 0.595 | 4 | 0.168 | 4 |
| S1Qv2 | 42.52 | 5 | 1.319 | 7 | 0.702 | 7 | 0.206 | 6 |
| SpinQuant Had* | 42.28 | 6 | 1.312 | 6 | 0.700 | 5 | 0.206 | 5 |
| AWQ* | 42.19 | 7 | 1.305 | 5 | 0.702 | 6 | 0.208 | 7 |
| S1Qv1 | 41.60 | 8 | 1.332 | 8 | 0.709 | 8 | 0.211 | 8 |
| SpinQuant no-Had* | 37.15 | 9 | 1.485 | 10 | 0.776 | 9 | 0.249 | 9 |
| SmoothQuant* | 36.57 | 10 | 1.478 | 9 | 0.783 | 10 | 0.256 | 10 |
| GPTQ-block* | 35.22 | 11 | 1.666 | 11 | 0.840 | 11 | 0.304 | 12 |
| RTN | 35.09 | 12 | 1.669 | 12 | 0.843 | 12 | 0.302 | 11 |

Bold marks the best displayed quantized value, including rounding ties. Native is an unranked reference. Ranks use unrounded scores and each metric's own direction; the table is ordered by accuracy. [Unrounded rankings](../results/benchmarks/expanded-20261007/rankings.csv) · [Per-model/source accuracy](../results/benchmarks/expanded-20261007/accuracy_table.md) · [Accuracy CSV](../results/benchmarks/expanded-20261007/accuracy_by_dataset.csv).

Current **S1Q ranks first in accuracy, Brier and ECE15 among these twelve implementations**; S1Q-AC has the lowest NLL. Its accuracy gain over RTN is **19.01 percentage points**, but Native remains **22.27 points higher**. These results do not establish lossless quantization, superiority over the official baseline implementations, or the best method for every individual model/source.

The conditional paired accuracy intervals below use 2,000 shared within-source cluster draws, seed 20261007. Differences are oriented as current S1Q minus the named reference:

| Current S1Q minus reference | Difference pp | Conditional paired 95% interval pp | Includes zero? |
| --- | ---: | ---: | --- |
| S1Q − RTN | +19.014 | [+18.010, +19.998] | No |
| S1Q − S1Q-AC | +0.909 | [+0.253, +1.614] | No |
| S1Q − S1Q-Margin | +1.495 | [+0.811, +2.217] | No |

All three intervals exclude zero **conditional on these evaluated samples and fixed models/source suites**. There is no multiple-comparisons adjustment; this is not a universal superiority claim. The [interval report](../results/benchmarks/expanded-20261007/accuracy-uncertainty/accuracy_uncertainty.md) and [paired CSV](../results/benchmarks/expanded-20261007/accuracy-uncertainty/pairwise-accuracy-ci.csv) retain unrounded estimates and all pairs. See [the CI protocol](#conditional-accuracy-intervals) for scope binding and interpretation.

The [completed PNG](../results/benchmarks/expanded-20261007/benchmark_ranking.png) and [SVG](../results/benchmarks/expanded-20261007/benchmark_ranking.svg) show common-coverage accuracy with conditional intervals, family-specific ranks and probability metrics. Its [figure manifest](../results/benchmarks/expanded-20261007/figure_manifest.json) binds the scope, aggregate data and CI file. The expanded scope explicitly sets `show_all_probability_methods: true`, so the NLL, Brier and ECE15 panels show **all 12 main methods in the accuracy-rank order**. These metrics retain their own lower-is-better interpretation; they are not a combined winner score. The historical plot keeps its previous layout when this flag is absent. The new caption identifies SemIf / JevBench as authored suites, ToolACE as tool-choice classification and WildJailBreak as harmful/benign classification.

The public [aggregate report](../results/benchmarks/expanded-20261007/README.md) contains 2,990 metric rows and 90 ranking rows in two declared comparison scopes (`all12` and the six S1Q variants). The metric export includes 130 old Mixed Dev diagnostic cells; these are excluded from the new 22-source ranking. The supplemental [source breakdown CSV](../results/benchmarks/expanded-20261007/metrics_by_source.csv) has 6,240 rows, and the [paired-by-cohort CSV](../results/benchmarks/expanded-20261007/paired_by_cohort.csv) has 34,080 rows. Their [audit manifest](../results/benchmarks/expanded-20261007/supplemental_table_manifest.json) records hashes and definitions. Task/source subtypes are breakdowns, not additional independent datasets. The cohort-wise paired intervals use seed 20261004 and their own cohort estimator; they must not replace the fixed 22-source global intervals above. Recorded quantization process durations are not integer-kernel inference benchmarks.

The [Native control audit](../results/benchmarks/expanded-20261007/audits/native-control-evidence.json) records the main/SmoothQuant numerical-equivalence gate for all ten models; the [Native/source binding audit](../results/benchmarks/expanded-20261007/audits/native-source-binding-10models.json) covers the 220 Native source cells. The [audit manifest](../results/benchmarks/expanded-20261007/audits/manifest.json) also links the narrower Laya/WildJailBreak API check and recorded StartLux runtime settings. These source/shape and numerical checks do not establish bytewise protected-tensor identity or saved candidate-key snapshots.

## Reproduce the frozen request selection

The public [metadata directory](../configs/benchmarks/expanded-20261007/) includes the canonical source manifest, frozen exclusion registry, data-integrity and reproduction audits, source provenance, evaluator code hashes and the 22-source comparison scope. Its benchmark identifier is **`expanded-native-decision-20261007-v2`**. The `v3` in the evaluation run/batch identifier records a run attempt; it does not change the source selection or its benchmark identifier.

Use local checkouts of Kev at `0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91` and Intern-Decision at `3572c8a68b5df5dafe02d0e093989ba8ec0183bc`. The Intern source argument points to the **`benchmarks/accuracy-v1` directory**, not its repository root. With source files available under the example paths:

```bash
python scripts/prepare_expanded_benchmarks.py \
  --source-root work/upstream/kev \
  --intern-source-root work/upstream/Intern-Decision/benchmarks/accuracy-v1 \
  --exclusion-registry configs/benchmarks/expanded-20261007/exclusion_registry.json \
  --output work/data/expanded-20261007 \
  --seed 20261007 --count 128 \
  --benchmark-id expanded-native-decision-20261007-v2
```

The preparation flag is **`--output`**. It recreates all 22 selected dataset files locally, without using later model results or newly changing the frozen exclusion registry. WANLI retains its documented 256-request limit. Confirm that each recreated file SHA-256 matches the corresponding entry in [the canonical manifest](../configs/benchmarks/expanded-20261007/manifest.json) before model evaluation. The generated manifest records frozen-registry reproduction evidence, so its provenance section and overall hash may differ from the original manifest; the selected dataset file bytes must match. The canonical manifest hash remains `41e4ccc75326baaca3212085dd66e0127fb8bc122d5b06054a170a0a2a0a231d`, and the runner records whichever verified local manifest it actually reads.

## Evaluate the frozen W4A4 recipes

This expansion fixes **W4A4, group size 128, BF16, 128 calibration requests, 128 reservoir rows, 256 old development requests, and evaluation seed 20261004**. The preparation sampling seed above is separately fixed to 20261007. The original calibration and development files are local-only prerequisites; a differently prepared split does not reproduce this comparison:

| Local input | Required SHA-256 | Role |
| --- | --- | --- |
| `work/data-validation/calibration.jsonl` | `be17333bff1f4b3c284ed9f37d97f55acf4edd8e4847732bdf4e65f7206b9f4d` | Fixed label-free quantizer calibration |
| `work/data-validation/development.jsonl` | `6308e110d61479226e1c893b89bd71f9989563a24faf0d5315dcfc28ba06397c` | Legacy diagnostic; excluded from the new 22-source average |

The first command evaluates the frozen eleven-method group. SmoothQuant* is evaluated in a separate output directory by the second command, then admitted to the 12-method comparison only after the merge gate verifies native logits and softmax probabilities within `1e-6`, matching requests, calibration hashes and declared quantization scope. Install the pinned model runtime and checkpoint/source prerequisites described in the [reproduction guide](reproduce-current.md). Each output path must be new; use the model's corresponding pinned checkpoint/source options where required.

```bash
python scripts/run_expanded_benchmarks.py \
  --model laya --data-dir work/data-validation \
  --benchmark-manifest work/data/expanded-20261007/manifest.json \
  --output-dir work/runs/expanded-20261007/laya-main \
  --methods rtn,s1q-local,s1q2-beta05,awq-adapted,gptq-blockdiag-adapted,spinquant-nohad-adapted,spinquant-had-adapted,s1q-joint,s1q-ac,s1q-margin,s1q \
  --bits 4 --activation-bits 4 --group-size 128 --dtype bf16 \
  --calibration-count 128 --development-count 256 \
  --reservoir-size 128 --seed 20261004

python scripts/run_expanded_smoothquant.py \
  --model laya --data-dir work/data-validation \
  --benchmark-manifest work/data/expanded-20261007/manifest.json \
  --output-dir work/runs/expanded-20261007/laya-smoothquant \
  --bits 4 --activation-bits 4 --group-size 128 --dtype bf16 \
  --calibration-count 128 --development-count 256 \
  --reservoir-size 128 --seed 20261004
```

SmoothQuant* delegates to the existing repository channel-balancing implementation with **fixed `smooth_alpha=0.5`**. Its process-local registration is restored after completion or failure; the eleven-method recipe and quantization math are not edited. This W4A4 adaptation is not an official SmoothQuant W8A8 reproduction.

For StartLux-Decision runs, set **`STARTLUX_ALLOW_SLOW=1`** in the process environment for both commands, substitute the relevant pinned `startlux-decision-0.8b`, `startlux-decision-2b` or `startlux-decision-4b` model, and use matching model-specific output directories. For Bash, `STARTLUX_ALLOW_SLOW=1 python scripts/run_expanded_benchmarks.py ...` enables the recorded reference runtime. This runtime setting does not establish a throughput or integer-kernel acceleration result.

No new evaluation gold labels may be used to choose recipes, hyperparameters, model subsets or dataset subsets. Keep negative results and failures visible. Only verified completed runs with matching manifest/source identities enter the final aggregate; an intermediate single-model table is a validation artifact, not completion of the ten-model matrix.

## Audit and generate aggregate tables

The read-only merge gate checks every decision in all 22 source suites **and the old development diagnostic**, including identical native admissions, gold indices, option dimensions and recorded source/configuration identities. It checks both native logits and their softmax probabilities to an absolute tolerance of `1e-6`. This is a numerical native-equivalence check; it is not a protected-tensor content-hash check. A failed gate prevents adding the independent control to the comparison.

```bash
python scripts/audit_smoothquant_merge.py \
  --main-run work/runs/expanded-20261007/laya-main \
  --control-run work/runs/expanded-20261007/laya-smoothquant \
  --manifest work/data/expanded-20261007/manifest.json \
  --development-data work/data-validation/development.jsonl \
  --output work/audits/expanded-20261007/laya-smoothquant-merge.json

python scripts/derive_smoothquant_run.py \
  --main-run work/runs/expanded-20261007/laya-main \
  --control-run work/runs/expanded-20261007/laya-smoothquant \
  --manifest work/data/expanded-20261007/manifest.json \
  --development-data work/data-validation/development.jsonl \
  --output work/runs/expanded-derived-20261007/laya
```

The derived run is a new local evidence directory. The tool reruns the admission gate, preserves the original eleven-method payloads byte for byte, retains both original pre-evaluation configurations and records the independent SmoothQuant control. It does not pretend that all twelve methods were evaluated together in one process. The original two runs are kept intact. These private evidence directories contain individual predictions and must not be published.

Repeat the evaluation, gate and derivation for each target text model. Keep the derived-runs root separate from the original main/control roots to avoid duplicate model cells. Only after the full intended matrix is complete and all audits pass, generate the public report:

```bash
python scripts/summarize_optimization.py \
  --runs-root work/runs/expanded-derived-20261007 \
  --output-dir work/summaries/expanded-20261007 \
  --seed 20261004

python scripts/build_public_benchmark_report.py \
  --input expanded-20261007-v3=work/summaries/expanded-20261007 \
  --dataset-list configs/benchmarks/expanded-20261007/expanded-public-scope-template.json \
  --output-dir results/benchmarks/expanded-20261007
```

The summarizer must report zero audit errors. Before treating a reproduced run as the ten-model result, inspect `ranking_scopes.json` for the complete intended model list and 22 sources; the public reporter deliberately records and excludes incomplete models instead of inventing values. NanoJev action compatibility has a separate interpretation and does not fill a text decision-model cell.

The resulting public directory contains aggregate CSV/JSON, fixed-scope rankings, PNG/SVG figures and source-by-source accuracy tables. `accuracy_table.md` splits the frozen source order into 8/8/6 columns for readability. Every part's Avg still uses **all 22 source suites**, and the CSV retains unrounded scores and actual admitted denominators. **Bold identifies the best displayed quantized accuracy including rounding ties**; Native rows remain plain reference values. NLL, Brier and ECE15 minima are marked independently in the main ranking table. Historical metrics and rankings remain in their separate directory and scope.

## Conditional accuracy intervals

After the initial public export above, compute intervals from the immutable derived runs using that exact generated `ranking_scopes.json` and unrounded `rankings.csv`. The source scope ID remains `expanded-v3-w4a4-22source-20261007`; the twelve-method group is `all12`.

```bash
python scripts/summarize_accuracy_uncertainty.py \
  --runs-root work/runs/expanded-derived-20261007 \
  --ranking-scopes results/benchmarks/expanded-20261007/ranking_scopes.json \
  --suite-id expanded-v3-w4a4-22source-20261007 \
  --comparison-group all12 --samples 2000 --seed 20261007 \
  --rankings-csv results/benchmarks/expanded-20261007/rankings.csv \
  --output-dir results/benchmarks/expanded-20261007/accuracy-uncertainty

python scripts/build_public_benchmark_report.py \
  --input expanded-20261007-v3=work/summaries/expanded-20261007 \
  --dataset-list configs/benchmarks/expanded-20261007/expanded-public-scope-template.json \
  --output-dir results/benchmarks/expanded-20261007 \
  --accuracy-uncertainty results/benchmarks/expanded-20261007/accuracy-uncertainty/accuracy_uncertainty.json
```

The CI output directory must be new. The second command reuses the original report arguments and output directory. It checks the existing scope and ranking files against the regenerated bytes, verifies their SHA-256 bindings in the CI report, and checks every method's **unrounded** accuracy within `1e-10` before drawing. A changed scope, method/model/source list, admitted denominator or point estimate rejects the CI overlay. The point rankings and accuracy tables remain unchanged; the figure manifest records the CI file SHA and interpretation. If the scope changes, first produce a fresh matching public export and CI rather than attaching an older interval file.

Whole saved clusters are resampled with replacement **within each source**. Each source's draw multiplicities are shared across all methods and models, preserving paired comparisons and cross-model overlap. The estimator still computes correct/admitted decisions for each model/source, then averages sources and models equally. Models and source suites themselves are fixed, not resampled. Different model admission subsets retain their recorded denominators; this does not establish a common population across different admission rules.

The percentile **95% intervals are conditional on the evaluated samples and these fixed models/source suites**. They do not measure generalization uncertainty over unseen architectures or datasets. Native receives a reference interval in the aggregate output but no quantizer rank or main accuracy bar. Family ranks and NLL/Brier/ECE15 panels remain point estimates.

`method-accuracy-ci.csv` records method intervals; `pairwise-accuracy-ci.csv` records differences computed from the **same bootstrap replicates**. A pair's interval must not be formed by subtracting independent endpoints of the marginal bar intervals. The comparisons have **no multiple-comparisons adjustment**, and point ranks or overlapping/non-overlapping bar intervals do not establish a universal or statistically significant champion. Only aggregate intervals, counts and hashes are written; bootstrap arrays and individual predictions remain private.

## Supplemental classification diagnostics

Accuracy can conceal a method's preference for the majority class. Supplemental diagnostics therefore report the admitted class distribution, the majority-class reference accuracy, balanced accuracy, macro-F1, per-class precision/recall/F1, predicted-class counts and confusion matrices. These remain **per model/source/method diagnostics**: they do not replace the frozen 22-source accuracy estimator, change its weights, select a new method recipe or create a combined F1 leaderboard.

The current diagnostic release admits four audited, stable semantic class schemas:

| Source | Semantic classes | Interpretation |
| --- | --- | --- |
| WildJailBreak | harmful, benign | Harmful/benign classification; inspect both class recalls and predicted counts. This is not generated-response safety or jailbreak success. |
| TweetEval Offensive | not_offensive, offensive | Offensive-language classification with the evaluator's Boolean false/true candidate order. |
| MNLI | entailment, neutral, contradiction | Three-way natural-language inference. |
| WANLI | supported, insufficient, contradicted | Three-way inference; each question's original candidate permutation is mapped back to these semantic classes. |

Answer positions such as A/B/C/D in ARC or MMLU are question-specific options, not semantic classes for a source-wide confusion matrix. Mixed question schemas in AG News/Yelp, authored action/rule scenarios and question-specific tool candidates also do not qualify for this diagnostic release. All 22 sources remain in the main accuracy comparison; the explicit diagnostic allowlist does not remove them from that ranking.

For a full intended ten-model diagnostic scope, use the same fixed model list as the final main comparison:

```bash
python scripts/summarize_classification_diagnostics.py \
  --runs-root work/runs/expanded-derived-20261007 \
  --manifest work/data/expanded-20261007/manifest.json \
  --models kev-0.8b kev-4b kev-9b laya \
    intern-decision-0.8b intern-decision-2b intern-decision-4b \
    startlux-decision-0.8b startlux-decision-2b startlux-decision-4b \
  --output-dir results/benchmarks/expanded-20261007/classification-diagnostics
```

The output directory must be fresh. The tool requires Native and all twelve W4A4 methods to be complete for each included model, checks source byte hashes, source-derived gold indices, prediction keys and logit widths, and keeps one matched admission set per source. Candidate-to-class mapping follows the frozen source and evaluator ordering contract (choice criteria insertion order; Boolean false/true); optional saved options are checked when present. Current saved JSONLs have no option-key snapshots, so their observed semantic ordering cannot be independently established from logits alone. An absent or incomplete model is listed as pending and has no numeric cells. Inspect `status`, `included_models` and `pending_models` before calling the diagnostic scope complete.

`classification-source-shapes.csv` records the frozen subset's gold counts/proportions, screening counts and majority-class accuracy. `classification-diagnostics.csv` records model-specific admitted counts, accuracy, balanced accuracy, macro-F1, gold/predicted counts and confusion matrices; `classification-per-class.csv` records class precision/recall/F1. The JSON and Markdown companions record definitions, scope, exclusions and provenance. Confusion-matrix rows are gold classes and columns are predicted classes in the declared semantic order.

Balanced accuracy averages recall over **gold-present classes**; a gold-absent class has an undefined (`null`) recall, and the number of present classes is reported. Macro-F1 averages all fixed semantic classes, using `2TP / (gold_count + pred_count)` per class and zero F1 for an empty gold/prediction class. Precision is undefined when a class is never predicted. These conventions remain explicit rather than silently treating missing classes as extra observations.

The majority-class reference is the largest gold-class share of the relevant frozen or admitted subset. Compare overall accuracy with that reference, balanced accuracy, all class recalls and predicted counts before interpreting a gain. A high point accuracy alone can coexist with poor minority-class recall. These counts describe the screened, short-request frozen subset and each model's admitted subset, **not full upstream prevalence or leaderboard performance**. The diagnostic metrics do not carry confidence intervals unless separately computed, and no new winner or causal explanation is inferred from them alone.

The completed [classification report](../results/benchmarks/expanded-20261007/classification-diagnostics/classification_diagnostics.md) contains **520 cells: 10 models × 4 audited semantic-class sources × (12 quantizers + Native)**, with no pending models. [Diagnostic CSV](../results/benchmarks/expanded-20261007/classification-diagnostics/classification-diagnostics.csv) · [Per-class CSV](../results/benchmarks/expanded-20261007/classification-diagnostics/classification-per-class.csv) · [Frozen class distributions](../results/benchmarks/expanded-20261007/classification-diagnostics/classification-source-shapes.csv). Public artifacts contain aggregate counts, metrics, schemas and hashes only. No private paper files, raw requests, individual predictions, checkpoints or bootstrap arrays are included.
