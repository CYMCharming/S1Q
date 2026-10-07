# S1Q expanded decision benchmark

**Status: the expanded evaluation is running. New accuracy values, rankings, probability metrics and winner claims remain pending.** This document records the frozen evaluation protocol and prepared data sizes, not completed model results. Historical three-cohort results are a separate snapshot.

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

The frozen data-integrity audit verifies all 22 hashes and exact request/gold preservation. A separately reproduced preparation produced 22 byte-identical dataset files. The prepared manifest SHA-256 is `41e4ccc75326baaca3212085dd66e0127fb8bc122d5b06054a170a0a2a0a231d` (`expanded-native-decision-20261007-v2`). Actual model-admitted counts, exclusions and decision denominators will accompany completed results; prepared counts must not be silently substituted for those counts.

## Upstream evidence and frozen sources

The source request files are taken from the projects' released decision evaluation suites:

- [Kev](https://github.com/jaredpalmer/kev/tree/0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91): pinned decision-v7, transfer-v4, public-pool-v6 and external WANLI / SemIf source files. These files already encode the state, question type, candidates, scoring distribution and reference labels used by decision models.
- [Intern-Decision evaluation](https://github.com/InternLM/Intern-Decision/blob/3572c8a68b5df5dafe02d0e093989ba8ec0183bc/docs/EVALUATION.md): pinned `accuracy-v1` manifest and source files for JevBench, ToolACE and WildJailBreak. The manifest identifies AG News and Typed Decisions as well; duplicate source families or difficulty tiers are not counted twice.
- [StartLux-Decision model card](https://huggingface.co/startlux-models/StartLux-Decision-0.8B): its reported decision-suite coverage guides the benchmark scope. Reported composite Decision Index editions are not independent datasets or additional S1Q observations.

Raw source requests stay local. Public outputs should contain aggregate metrics, preparation code, source revisions, file hashes, admission counts, protocol metadata and figures. Private paper files, model checkpoints, individual predictions and raw dataset records are not part of this release.

## Screening and leakage boundary

The expanded requests are selected **without looking at model predictions**. The frozen selection policy uses SHA-256 ordering of complete source groups, prefers upstream test records where available, and discloses unexamined upstream development supplements. It preserves the complete original state/scenario, question schema, candidate order and reference labels. Requests are not shortened, rewritten or converted into easier candidate subsets.

The prepared suite admits clean requests with at most 10 choices and at most 1,000 characters under the documented preparation rule. File deduplication and whole-group exclusion prevent repeated upstream requests, calibration overlap, and reuse of earlier locally evaluated request/state/scenario identities. Group metadata is retained for paired cluster analysis. Model-specific tokenizer and API admission still need to be measured for each run.

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

## Expanded results — pending

The new main table, method ranking and scientific figure will be generated only from completed, verified runs with a matching frozen manifest and complete marker. They must use the actual admitted counts and explicit common coverage. A partially completed run, a pilot, a startup smoke check, an earlier attempt or the old three-cohort snapshot cannot fill a missing new result.

| Method | New 22-source accuracy | NLL / Brier / ECE15 | Rank |
| --- | --- | --- | --- |
| RTN | Pending | Pending | Pending |
| S1Qv1 | Pending | Pending | Pending |
| S1Qv2 | Pending | Pending | Pending |
| SmoothQuant* | Pending | Pending | Pending |
| AWQ* | Pending | Pending | Pending |
| GPTQ-block* | Pending | Pending | Pending |
| SpinQuant no-Had* | Pending | Pending | Pending |
| SpinQuant Had* | Pending | Pending | Pending |
| S1Q-Joint | Pending | Pending | Pending |
| S1Q-AC | Pending | Pending | Pending |
| S1Q-Margin | Pending | Pending | Pending |
| **S1Q (current)** | Pending | Pending | Pending |

The final chart will show the new scope's common-coverage accuracy, family-specific ranks and probability metrics, backed by its aggregate CSV and an explicit figure manifest. Its caption will identify SemIf / JevBench as authored suites, ToolACE as tool-choice classification and WildJailBreak as harmful/benign classification. The old Mixed Dev / WANLI / MMLU-Pro caption will not be reused.

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

The summarizer must report zero audit errors. Before treating this as the planned ten-model result, inspect `ranking_scopes.json` for the complete intended model list and 22 sources; the public reporter deliberately records and excludes incomplete models instead of inventing values. NanoJev action compatibility has a separate interpretation and does not fill a text decision-model cell.

The resulting public directory contains aggregate CSV/JSON, fixed-scope rankings, PNG/SVG figures and source-by-source accuracy tables. `accuracy_table.md` splits the frozen source order into 8/8/6 columns for readability. Every part's Avg still uses **all 22 source suites**, and the CSV retains unrounded scores and actual admitted denominators. **Bold identifies the best displayed quantized accuracy including rounding ties**; Native rows remain plain reference values. NLL, Brier and ECE15 minima are marked independently in the main ranking table. Historical metrics and rankings remain in their separate directory and scope.
