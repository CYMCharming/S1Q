# S1Q evaluation design

This page preserves the v0.1 data/evaluation design and its historical final cohorts. The current S1Q uses the label-free optimization workflow described in [Method](method.md) and [Current reproduction](reproduce-current.md). The newer batch's inspected development and additional evaluation cohorts must not be described as fresh v0.1 final tests.

Status: working protocol, 2026-10-01 (Asia/Shanghai). Pilot results were inspected
before the Fisher candidate expansion; those comparisons remain exploratory.
The external confirmation cohort and fixed-profile evaluation are specified
separately below. This document does not report experimental success.

## Research questions

1. At the same weight format, group size, head precision and calibration budget,
   does S1Q preserve decisions and probabilities better than round-to-nearest
   (RTN) and ordinary activation reconstruction?
2. Do any gains transfer across an encoder decision model (Laya), a decoder
   Transformer decision model (NanoJev), and hybrid recurrent/attention decision
   models (Kev-0.8B, Kev-4B and Kev-9B)?
3. How do low-bit weights change confident mistakes, action thresholds, candidate
   order robustness and actual resource use?

The reference is each released checkpoint with its published inference semantics.
Record the exact model/base/tokenizer commits, source commit, built-in temperature,
precision and serialization. Use the same reference dtype for all methods within
a model. A separate FP32 versus BF16 check measures precision-conversion error.
Official hosted Jev is optional contextual evidence; the requested open models are
the main experimental objects. No commercial API call is required by this protocol.

## Data selection

| Suite | Role and practical format | Ground truth and limitations | License / publication handling |
| --- | --- | --- | --- |
| Kev `decision-v7` | Immediate shared suite; UTF-8 labelled request JSONL. Calibration 968 records / 1,148 questions; development 1,204 / 1,468; test 1,176 / 1,440 at the inspected manifest. | Choice, Noul, Score from public NLP sources and generated policy/rule tasks. Source families overlap Kev training; these are held-out records, not proof of an unseen domain. | Mixed underlying dataset licenses; public repository code license does not replace them. Publish downloader, hashes and IDs initially, not copied JSONL. |
| Kev `transfer-v4` | Shared source-transfer evaluation; development/test JSONL, 764 records each before S1Q screening. | Source families held out from Kev's supervised decision training. Do not call this OOD for Laya/NanoJev unless their training membership is known. | Inherited source licenses; same local-only data policy. |
| Kev `transfer-v9` | Secondary stress suite; 1,264 development and 1,264 test records. Adds 10-way options, long distractor states and evidence-removed inputs. | Evidence-removed cases test confidence/abstention, not ordinary label accuracy. Keep each case with its intact control. The suite has **no calibration examples**. | Inherited mixed licenses; exclude over-length cases from common comparison and report them in native-length analysis. |
| NanoJev unified hard games | Native NanoJev evaluation: `unified/hard/{calibration,dev,test,ood}.jsonl` at dataset commit `7afc5257c0f3ff0ba08512729888a51d94b40e7e`. | The upstream package covers Maze, Snake, ViZDoom Basic and Predict Position, but final S1Q native test/OOD cohorts contain only shooting tasks. All evaluated native labels are explicitly supplied `reference_argmax_compatibility` gold from recorded RL reference-policy action argmaxes. Maze/Snake teacher-only questions without supported hard gold are excluded. These are action-compatibility targets, not human annotation, optimal-action gold or observed-success probabilities. | The inspected dataset card does not declare a dataset license. Download for the experiment; do not redistribute source records or teacher receipts. |
| JevBench public original + hard | External confirmation cohort at `bb05a335bc809e61b20c0f745d25499a82b326fc`: 72 original +111 public hard tasks before screening. | Original authored scenarios; hard labels were authored and cross-reviewed with model assistance. All 183 have explicit expected targets; ten hard probability tasks additionally supply state-derivable distributions. Small public cohort, not the full private leaderboard. | All 183 inspected item provenance entries declare MIT. `THIRD-PARTY.md` explicitly covers original items; the hard specification and per-item provenance cover hard items. Preserve the upstream notice. |
| Banking77 | Optional large-choice stress; 10,003 training and 3,080 test banking utterances; CSV upstream / Parquet HF mirror. | 77 human-annotated intents. The full candidate set is unsuitable for the <=10-choice shared cohort. Report separately if native interfaces admit all 77 candidates. | CC BY 4.0, attribution to Casanueva et al.; publish IDs/recipe rather than change label/candidate task definition. |
| BoolQ | Optional binary Noul evaluation; 9,427 training and 3,270 labelled development examples. | Passage-supported yes/no label; official test has no public labels. Partition labelled development once or use it as one external final set. | CC BY-SA 3.0; retain attribution and source license. |
| ChaosNLI | Optional human-disagreement evaluation; SNLI/MNLI/alphaNLI JSONL with 100 annotations per example. | The empirical human distribution enables probability-distribution evaluation beyond one hard label. Analyze SNLI/MNLI label ordering explicitly. | CC BY-NC 4.0; optional academic research set, never silently include in commercial redistributable data. |

Final NanoJev scope in the v0.1.0 snapshot: test has 1,023 shooting decisions (321 Basic, 702 Predict Position), and OOD has 1,024 (398 Basic, 626 Predict Position). The reference policies are `edbeeching/doom_basic_1111` and `thainv0212/sonic_doom`. The converter copies the dataset's explicit `gold`; it does not generate an argmax fallback. Accuracy is reference-policy argmax action agreement, with no claim about optimal actions or observed episode-success probabilities. This documentation clarification leaves the original experiment and aggregate snapshot unchanged.

Primary sources: [Kev suite documentation](https://github.com/jaredpalmer/kev#benchmarks),
[decision-v7 manifest](https://github.com/jaredpalmer/kev/blob/main/evals/v7/decision-v7/manifest.json),
[transfer-v9 manifest](https://github.com/jaredpalmer/kev/blob/main/evals/v9/transfer-v9/manifest.json),
[NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data),
[JevBench third-party terms](https://github.com/fstandhartinger/jevbench/blob/main/THIRD-PARTY.md),
[hard-tier provenance specification](https://github.com/fstandhartinger/jevbench/blob/main/datasets/HARD-TIER.md),
[Banking77 upstream](https://github.com/PolyAI-LDN/task-specific-datasets),
[BoolQ upstream](https://github.com/google-research-datasets/boolean-questions),
[ChaosNLI upstream](https://github.com/easonnie/ChaosNLI).

There are several unrelated repositories called JevBench. All reports must name
the owner/repository and full commit, rather than only writing “JevBench”. The
[model-collapse benchmark](https://github.com/model-collapse/jev-bench) is another
typed-choice/ordinal harness with model-panel labels; it is not the same data as
fstandhartinger's suite. It is a supplementary candidate, not silently interchangeable.

## Executable preparation

`src/s1q/data.py` downloads fixed source revisions or reads a local verified checkout.
Its current Kev pin is `0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91`. Each source file is
verified against its upstream SHA-256 manifest. Source data, prepared JSONL and
prediction dumps should remain in ignored experiment directories.

```python
from s1q.data import prepare_kev_data, prepare_nano_data

prepare_kev_data("work/data/shared-pilot", source_root="work/upstream/kev",
                 calibration=64, temperature_calibration=64,
                 development=64, test=256,
                 max_choices=10, max_chars=1000)

# Native game preparation preserves target semantics and skips teacher-only questions.
prepare_nano_data("work/data/nano-pilot", calibration=64,
                  temperature_calibration=64, development=64, test=256)
```

The pilot limits count records, not questions. Sampling cycles over source families
and uses deterministic SHA-256 ordering with seed `20261001`. A source group is
retained whole or skipped whole; pilot counts can fall below the requested limit
for large episodes. The manifests expose resulting record/question/group counts,
source coverage, exclusion reasons, selected IDs and source/output hashes.

`scripts/prepare_jevbench.py` prepares only `external_test.jsonl` plus an audit
manifest and the upstream MIT notice. `configs/jevbench-data-checksums.json` pins
the exact Git commit, raw Git blob SHA-256/size, source counts and scoring semantics.
Git object reads avoid Windows checkout newline conversion; a non-Git source
directory must contain byte-exact exported files. No model prediction is read.

```sh
python scripts/prepare_jevbench.py --source-dir work/jevbench \
  --output-dir work/data/jevbench-external
```

The complete public original+hard source cohort contains 183 tasks. The declared
<=10-choice / <=1,000-complete-request-character filter admits **85 tasks in 49
scenario clusters** (72 original and 13 hard; 33 Noul, 40 Choice, 12 Score) and
rejects 98 hard tasks for length. The manifest names every rejected ID and reason.
All original paraphrase pairs remain in one cluster. These counts precede actual
model tokenizer admission. No mathematical-probability item fits this short
cohort; do not claim that it evaluates recovery of known probability distributions.
A separately named native-length cohort can use `--max-chars 0`: all 183 tasks
are admissible by candidate count, with ten mathematical probability targets.

For this external confirmation, freeze the quantizer profiles, source/code
commits, calibration IDs, temperature policy and compared methods **before the
first JevBench prediction**. Use the full admitted cohort, without an outcome-based
subsample. It is external to S1Q's method selection and pilot inspection; this
does not establish absence from a model's unknown pretraining data. It must never
feed quantizer statistics, Fisher weights, clipping, temperature fitting, thresholds
or bit/profile selection. The original scenarios have explicit reviewed rubrics;
the hard scenarios have model-assisted authorship and cross-review. Call results
“S1Q screened public JevBench authored cohort”, not the official full benchmark
score or rank. `--check-disjoint-with <prepared.jsonl>` can be repeated to verify
exact-request and group disjointness from existing calibration/development pools.

## Split discipline and eligibility

The upstream calibration partition is split by a group hash into two disjoint
pools: one for quantizer statistics/search and one for temperature fitting or
deployment-threshold estimation. Development chooses hyperparameters and bit width.
Test is used after the recipe and code commit are frozen. Freeze a transfer-test
configuration separately; do not repeatedly choose variants on that split.

Use source record/episode/scenario IDs for all split assignment, sampling and
bootstrap resampling. A question's paraphrases, candidate permutations, paired
controls and repeated states from one episode must remain in the same group.
The executable preparation rejects group reuse or exact full-request reuse across
prepared partitions. This does not certify the absence of paraphrase duplicates,
unknown foundation-model pretraining contamination or prior developer exposure.
Existing Kev test suites have already been used by upstream model developers;
they are external regression data for S1Q, not a newly private dataset.

NanoJev has identical inference requests across distinct gameplay episodes.
Within each upstream split, preparation joins complete episode groups connected
by identical requests into transitive components; it never assigns buckets per
row. It reserves admitted upstream pools before pilot sampling, in fixed priority
test, OOD, development, calibration. A lower-priority component overlapping a
retained higher-priority request or upstream group is removed whole and logged.
Test is preserved; OOD/development/calibration are filtered, never resplit into
new held-out partitions. Only the remaining calibration components are divided
into quantizer and temperature pools. Manifests record joined groups, repeated
requests, rejected component IDs/row counts and cross-source overlap details;
no repeated source observation is silently deduplicated. Bootstrap uses the
resulting `_meta.group_id`, preserving all connected episodes and questions.

The shared cohort currently admits only clean variants, <=10 candidate options
and <=1,000 characters of the complete JSON request. The character bound is a
screening heuristic. **Before confirming a common cohort, run every model's actual
tokenizer/sequence builder and intersect admission IDs.** Reject cases that require
any candidate or state truncation. Freeze the intersection before comparing
quantized predictions. Report how many records/questions and which source families
remain. An optional tokenizer predicate is supported by preparation and its use is
recorded in the manifest; disclose the predicate/version separately.

Native-length and native-option suites should accompany the common cohort. The
shared short cohort cannot establish long-context ability. Do not compare a
truncated Laya prompt against a full Kev/NanoJev prompt under the same task name.
Do not delete difficult items because a quantized model fails: invalid outputs are
failures and retain their IDs in the denominator.

Only `state`, `questions.type`, `questions.instructions` and `questions.criteria`
are allowed into model input. `label`, `gold`, `gold_probs`, `rationale`, teacher
answers, episode outcomes and every metadata field remain outside inference.
`model_request()` provides this allowlist. Label descriptions/order and state
serialization are fixed per adapter; do not rewrite descriptions per precision.

## Measurements

Report both target performance and fidelity to the unquantized reference.

| Quantity | Definition / reporting rule |
| --- | --- |
| Accuracy | Exact argmax correctness for Choice/Noul; report per source/type and macro average over source families. For expert policy labels call it action-target agreement. |
| NLL | `-log(max(p_y, 1e-15))`, with flooring disclosed and zero-probability counts retained. Requires a valid complete distribution. |
| Multiclass Brier | `sum_k (p_k - 1[k=y])**2`, averaged per question; range [0,2]. Binary uses the same two-class sum, not the alternative one-class convention. |
| ECE | Descriptive top-label ECE, 15 equal-width confidence bins; disclose bins and sample counts. Report per suite/source where large enough. Do not make ECE alone the optimization/success criterion. |
| Score error | Expected level `sum_i i*p_i`; normalized MAE dividing by `K-1`, plus exact argmax accuracy. Use ranked probability score for complete ordinal distributions. |
| FP agreement | Fraction of identical top candidates after aligning candidate keys. Agreement is fidelity, not objective accuracy. |
| Probability drift | Mean total variation and Jensen-Shannon divergence to the matched FP distribution; mean/max absolute probability shift; top-two margin drift. |
| Confident errors | Fraction of all labelled questions that are wrong with max probability >=0.9; also report conditional error among accepted questions. |
| Selective prediction | Risk-coverage curve and area under it. A deployment threshold is selected on calibration and fixed before test; test-optimized coverage at 5% risk is only descriptive/oracle evidence. |
| Robustness | Original versus candidate-permuted answer flip rate and aligned probability drift; bundled versus isolated question differences. Compare quantized increase above FP's own bias. |

For human distributions, use cross-entropy, JS/TV distance and expected Brier
against empirical annotator proportions; the majority label is an additional
view. For evidence-removed questions, report max-confidence distribution, high
confidence share and changes versus the intact control. Avoid grading evidence-free
answers as though the answer remained identifiable from the input.

Temperature scaling is a separate optional stage. Current executable raw tables
use unscaled decision logits and a common complete candidate distribution,
including `[false, true]` for binary questions; they do not reproduce upstream
served temperatures or rounded responses. Record published checkpoint temperatures
in model metadata and identify any optional served-convention evaluation separately.
Show every quantizer with the same post-hoc opportunity in a separate table;
fitting temperature only for S1Q would confound the comparison. Positive scalar temperature preserves
argmax, so it cannot repair flipped decisions. This follows the established
[temperature-scaling literature](https://arxiv.org/abs/1706.04599).

Use paired, source-stratified cluster bootstrap intervals (e.g. 2,000 resamples)
on metric differences, resampling groups rather than individual questions. Publish
uncertainty, sample sizes and all failed runs. A 64-record pilot is a smoke/search
run; a 256-record test is preliminary and cannot establish small improvements.
Final tables should use the full admitted suites and repeat calibration sampling
with at least three seeds when claiming a reliable method gain.

## Controlled quantizer comparisons

Start with W4A16 backbone weight-only quantization and a W8A16 control. Keep typed
decision heads, normalization, softmax and Kev recurrent-state arithmetic in the
reference precision initially. Report this as mixed precision, with actual
quantized parameter coverage and average effective bits including scales/zeros.
Protected embeddings or projections must be disclosed and held equal for RTN/S1Q.
W8A8 and W4A8/W4A4 are later experiments if real kernels and calibrated accuracy
justify them; retaining floating activation/state paths is not full W4A4.

Mandatory comparisons are unquantized, RTN with identical format, S1Q, and a
plain activation-statistics/reconstruction variant. For research claims, add an
established PTQ method where compatible (GPTQ or AWQ) and document adaptations.
Do not label a simplified activation-scaling heuristic as a full AWQ reproduction.

Ablate output-decision weighting, clipping/scale selection, protected heads,
calibration composition and temperature. Also distinguish a random task-calibration
pool from low-margin selection because uncertainty-aware calibration selection
already exists in [DPQ](https://arxiv.org/abs/2608.21019).

Declare provisional noninferiority margins before final tests, e.g. no more than
1 percentage point accuracy loss and 0.01 Brier increase versus the reference.
These are study design choices, not universal deployment guarantees. Method
selection uses development performance under a fixed bit budget; a negative result
is reportable. Do not decide the thresholds from favorable test numbers.

## Efficiency and release evidence

Quantize-dequantize simulation verifies numerical behavior only. A tensor stored
in BF16/FP32 after rounding is not a packed low-bit deployment artifact. Measure
packed checkpoint bytes, scales/zeros, live device memory and actual backend dtype.
Report forward-only and full public prediction latency separately, p50/p95,
throughput at declared batch/question/state lengths and peak memory. Synchronize
GPU timings, warm up, alternate candidate order, run on the same otherwise comparable
GPU and exclude model loading from steady-state latency. Publish environment and
code hashes. A W4 simulation running float matrix multiplication cannot support
an acceleration claim.

Publish the experiment recipe, model/data pins, manifests, aggregate results,
ablation tables and truthful limitations in `CYMCharming/S1Q`. Respect each model's
base and derivative licenses when publishing quantized weights; do not replace
upstream provenance with S1Q's repository license. Link Jev, Kev, NanoJev, Laya and
relevant benchmark authors prominently. Promotion should refer to reproduced
results and useful integrations rather than an unsupported priority claim.
