# S1Q: Low-Bit Quantization for System One Decision Models

[中文说明](README.zh-CN.md) · [Method](docs/method.md) · [Current reproduction](docs/reproduce-current.md) · [Component figures](results/benchmarks/components-20261008/README.md) · [Expanded results](results/benchmarks/expanded-20261007/README.md) · [Expanded protocol](docs/expanded-benchmarks.md) · [Historical aggregate](results/benchmarks/historical-20261006/README.md) · [Related work](docs/related-work.md) · [Model audit](docs/model-audit.md) · [Legacy results](docs/results.md) · [Packed artifacts](docs/artifacts.md)

**S1Q now denotes the current decision-margin-aware, activation-compensated method.** Its October 2026 experiment identifier was `s1q-mac`; that identifier remains supported to reproduce frozen runs. The public method name is **S1Q**. Earlier S1Q and S1Q2 recipes remain available as historical controls.

S1Q quantizes open Jev-like **System One decision models** that map a state and typed questions directly to finite choice, Boolean or discrete-level distributions. The implementation supports pinned Kev, NanoJev, Laya, Intern-Decision and StartLux-Decision adapters; it preserves native candidate order and decision readouts. The new Intern/StartLux adapters cover released 0.8B, 2B and 4B sizes in a text-only contract, with separate native parity gates. See [model extensions](docs/decision-model-extensions.md) for interface, provenance and runtime scope. Adapter support alone does not imply a completed benchmark. It is an independent research project and is not affiliated with [TypeSafe Jev](https://docs.typesafe.ai/introduction).

## Method

S1Q combines three calibration steps:

1. Backpropagate each teacher decision's top-one versus runner-up logit margin separately. Use aligned, bounded token gradient energy to sample calibration rows that matter to candidate comparisons. No calibration gold labels are used.
2. Search channel scales and clipping using the **actual quantized weight and activation reconstruction**. Both W and A error enter the local output objective.
3. Fit a bounded ridge correction for activation error before quantizing weights. Select corrected or uncorrected candidates using a separate half of the saved token reservoir.

The gradient statistic is a **margin-Jacobian token proxy, not a categorical Fisher matrix**. [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html) already uses end-loss gradients to guide reconstruction, and [RSQ](https://openreview.net/pdf?id=kBezrKXHVS) already prioritizes important tokens in quantization. Ridge-based activation-error compensation has close prior art in [ERQ (ICML 2024)](https://proceedings.mlr.press/v235/zhong24a.html), and matching native outputs has close prior art in [GPTAQ](https://arxiv.org/abs/2504.02692). Gradient guidance, token importance, scaling, clipping and compensation are not individually new. The research contribution being evaluated is their concrete decision-aware integration across native typed-decision architectures. See the exact formulas, assumptions and limitations in [Method](docs/method.md).

## Component ablation (October 8)

The [complete component analysis](results/benchmarks/components-20261008/README.md) now covers **all 10 models × all 22 source suites at W4A4**. It compares the four available component configurations using the same source-then-model average; no favorable model or source subset is selected.

| Configuration | Margin sampling | Activation compensation | Candidates/layer | Accuracy % ↑ |
| --- | ---: | ---: | ---: | ---: |
| S1Q-Joint | No | No | 22 | 52.16 |
| S1Q-AC | No | Yes | 44 | 53.20 |
| S1Q-Margin | Yes | No | 22 | 52.61 |
| S1Q (current) | Yes | Yes | 44 | **54.11** |

S1Q improves over Joint on **6/10 model averages**; all four negative results remain visible. Activation compensation increases the search budget from **22 to 44 candidates per layer**, so this is a component/recipe comparison rather than an equal-compute causal isolation or a synergy test. S1Q-AC retains the best overall NLL. The new OmniQuant/AdaRound/HQQ matrix remains incomplete and does not replace the completed twelve-method ranking below.

Against S1Q-AC, which also evaluates **44 candidates/layer**, S1Q gains **+0.909 pp** (conditional paired 95% CI [0.253, 1.614]). The added margin sampling requires backward passes: matched candidate counts do not mean equal total computation. These are the existing fixed-scope shared-cluster intervals, without a multiple-comparisons adjustment.

![Complete W4A4 component comparison and gains for every model](results/benchmarks/components-20261008/ablation_overview.png)

[Vector figure](results/benchmarks/components-20261008/ablation_overview.svg) · [All 220 model/source gains](results/benchmarks/components-20261008/source_robustness.png) · [All twelve methods](results/benchmarks/components-20261008/overall_comparison.png) · [Complete component/source CSV](results/benchmarks/components-20261008/component_by_source.csv) · [Model/configuration CSV](results/benchmarks/components-20261008/component_accuracy.csv) · [Paired intervals](results/benchmarks/components-20261008/component_paired_ci.csv) · [Figure provenance](results/benchmarks/components-20261008/figure_manifest.json).

The PNG/PDF/SVG and TikZ outputs are standalone public benchmark figures; the manuscript remains private. [Regenerate from published aggregates](scripts/build_component_analysis.py): install dependencies with `python -m pip install -e '.[plots]'`, then run `python scripts/build_component_analysis.py --source-root results/benchmarks/expanded-20261007 --output-dir work/component-figures` (requires a fresh output directory).

### Illustrative cases selected after evaluation

The [selected component cases](results/benchmarks/component-cases-20261008/README.md) show favorable outcomes while retaining all four configurations:

| Case and averaging scope | Joint | +AC | +Margin | S1Q | S1Q gain over best alternative (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| StartLux-Decision-4B; all 22 sources | 56.80 | 60.86 | 60.80 | **65.10** | +4.24 |
| SST-5; all 10 models | 31.41 | 30.86 | 30.08 | **35.55** | +4.14 |

Values are accuracy percentages at W4A4. Selection is **post hoc**: separately among all 10 model averages and all 22 source averages, these cases have the largest S1Q gap over the strongest Joint/+AC/+Margin alternative. The linked bundle publishes **all 32 candidate rankings and the complete negative breakdown**. These examples do not replace the complete 10-model/22-source primary evidence above; no selected-case confidence intervals or significance claims are made. The component search-budget qualifications above still apply.

![Post hoc illustrative component cases with all four configurations](results/benchmarks/component-cases-20261008/case_studies.png)

## Expanded benchmark results (October 7)

**Completed: 10 text decision models × 22 source suites × 12 quantization methods at W4A4, plus Native references.** Every model admitted the same 2,565 requests and 2,871 decisions, confirmed against its frozen source scope. Models are Kev-0.8B/4B/9B, Laya, Intern-Decision-0.8B/2B/4B and StartLux-Decision-0.8B/2B/4B. **Kev-27B and NanoJev are not in this new matrix.** Accuracy averages sources equally within each model, then models equally; it is not pooled decision accuracy.

Current **S1Q ranks first in accuracy, Brier and ECE15 among these 12 implementations**; S1Q-AC has the lowest NLL. S1Q reaches **54.11%**, versus RTN's 35.09%, a **+19.01 percentage-point** gain. Native averages 76.38%: the remaining **22.27-point accuracy loss** rules out a lossless-quantization claim. This is a fixed-scope result, not superiority over official baseline implementations or every model/dataset.

| Accuracy rank | Method | Accuracy % ↑ | NLL ↓ | Brier ↓ | ECE15 ↓ |
| ---: | --- | ---: | ---: | ---: | ---: |
| — | Native reference | 76.38 | 0.734 | 0.345 | 0.131 |
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

Bold marks the best displayed quantized value, including ties; Native is an unranked reference. [All metric ranks and paired differences](docs/expanded-benchmarks.md#completed-expanded-comparison) · [Per-model/source accuracy tables](results/benchmarks/expanded-20261007/accuracy_table.md) · [Unrounded rankings CSV](results/benchmarks/expanded-20261007/rankings.csv).

The shared-cluster conditional paired 95% intervals for current S1Q's accuracy gains are **+19.014 [18.010, 19.998] pp versus RTN**, **+0.909 [0.253, 1.614] pp versus S1Q-AC**, and **+1.495 [0.811, 2.217] pp versus S1Q-Margin**. These intervals exclude zero under the fixed evaluated-sample protocol; they have no multiple-comparisons adjustment and do not establish a universal winner. [Interval evidence](results/benchmarks/expanded-20261007/accuracy-uncertainty/accuracy_uncertainty.md) · [Reproduce the CI pipeline](docs/expanded-benchmarks.md#conditional-accuracy-intervals).

![Completed W4A4 accuracy with conditional intervals, family ranks and all twelve methods' probability metrics](results/benchmarks/expanded-20261007/benchmark_ranking.png)

[Vector figure](results/benchmarks/expanded-20261007/benchmark_ranking.svg) · [Figure provenance](results/benchmarks/expanded-20261007/figure_manifest.json) · [Aggregate report](results/benchmarks/expanded-20261007/README.md) · [Supplemental class diagnostics](results/benchmarks/expanded-20261007/classification-diagnostics/classification_diagnostics.md).

Supplemental [source breakdowns (6,240 rows)](results/benchmarks/expanded-20261007/metrics_by_source.csv) and [cohort-wise paired comparisons (34,080 rows)](results/benchmarks/expanded-20261007/paired_by_cohort.csv) have a [hash/interpretation manifest](results/benchmarks/expanded-20261007/supplemental_table_manifest.json). Subtypes are not extra datasets; old development breakdowns stay outside the main ranking. Cohort-wise intervals (seed 20261004) differ from the fixed 22-source global intervals (seed 20261007). [Native control gate](results/benchmarks/expanded-20261007/audits/native-control-evidence.json) · [220-cell Native/source audit](results/benchmarks/expanded-20261007/audits/native-source-binding-10models.json).

The 22 screened, short-request sources comprise 18 standard dataset sources, two authored scenario/rule suites and two domain decision suites; they are bounded subsets, not complete upstream leaderboard evaluations. JevBench contributes only 48 new easy tasks; ToolACE measures tool-choice classification, and WildJailBreak measures harmful/benign classification. Typed Decisions was excluded because of teacher labels and the length bound. Source selection preserved questions/candidates, excluded recorded calibration and earlier evaluation identities, and kept old Mixed Dev outside the new average. The 520 supplemental class-diagnostic cells cover four audited semantic-class sources and do not change the main ranking. See [source roles, filtering and ordering-contract limitations](docs/expanded-benchmarks.md).

The original eleven-method runs and independent SmoothQuant* control were combined only after native-equivalence and scope checks; neither original run was changed. * SmoothQuant, AWQ, GPTQ-block and SpinQuant denote repository adaptations/proxies. Accuracy uses floating-point QDQ; this batch establishes no integer-kernel speedup or measured packed-checkpoint compression. StartLux uses the recorded `STARTLUX_ALLOW_SLOW=1` reference runtime.

## Historical three-cohort snapshot

The [historical October 6 aggregate](results/benchmarks/historical-20261006/README.md) retains **537 completed aggregate rows**: 321 from the original optimization batch and 216 from the six new Intern-Decision / StartLux-Decision models. It includes [metrics CSV](results/benchmarks/historical-20261006/metrics.csv), [JSON](results/benchmarks/historical-20261006/metrics.json), [scoped rankings](results/benchmarks/historical-20261006/rankings.csv), [coverage and sample counts](results/benchmarks/historical-20261006/ranking_scopes.json), and [source hashes](results/benchmarks/historical-20261006/provenance.json). Its eight declared scopes keep precision, model coverage and evaluation groups explicit. The historical principal group contains 11 measured methods; SmoothQuant is not retroactively inserted.

The figure below displays the **historical W4A4 three-cohort snapshot**, with nine models shared by all 11 methods. It is separate from the completed 22-source protocol and its average. [Vector figure](results/benchmarks/historical-20261006/benchmark_ranking.svg) · [Figure provenance](results/benchmarks/historical-20261006/figure_manifest.json).

![Historical W4A4 common-coverage accuracy, family ranks and probability metrics](results/benchmarks/historical-20261006/benchmark_ranking.png)

## Current evidence

The following is **historical October 4 development evidence**, separate from the completed expanded comparison. That frozen batch contains five models, W4A4 and two-model W3A4 stress tests, ablations and adapted quantization controls. Representative **development** accuracy (%):

| Model | Precision | Native | RTN | Earlier S1Q2 | Current S1Q |
|---|---|---:|---:|---:|---:|
| Kev-0.8B | W4A4 | 80.06 | 43.41 | 63.99 | 61.41 |
| Kev-4B | W4A4 | 85.21 | 35.05 | 55.63 | 76.85 |
| Kev-9B | W4A4 | 85.53 | 40.19 | 41.48 | 71.70 |
| Laya | W4A4 | 65.59 | 50.16 | 55.95 | 59.81 |
| NanoJev | W4A4 | 81.96 | 36.47 | 79.22 | 82.35 |
| Kev-0.8B | W3A4 | 80.06 | 40.51 | 44.69 | 47.27 |
| Kev-4B | W3A4 | 85.53 | 37.94 | 44.05 | 56.91 |

These are exploratory, inspected development cohorts, not a fresh final test or universal method ranking. Text models use 311 decisions; NanoJev uses 255 reference-policy compatibility decisions and measures recorded action agreement rather than game success. Native Kev-4B differs between W3/W4 runs because their recorded runtimes differ; compare methods within each run. S1Q improves some difficult low-bit settings, but the Kev-0.8B W4A4 negative result is retained. WANLI and MMLU-Pro additional cohorts support some transfer gains and substantial remaining accuracy loss. Kev-27B has a separate small non-gradient pilot; **the full current S1Q has not been evaluated on Kev-27B**.

AWQ, GPTQ and SpinQuant controls in this batch are **adaptations/proxies**, not official reproductions. In particular GPTQ is block diagonal and SpinQuant uses restricted block rotations, with separate Hadamard/no-Hadamard variants. They cannot establish superiority over the original methods' published results. SmoothQuant is implemented but was not run in this frozen batch.

## Quick start

Use Python 3.12 and a PyTorch build appropriate for your GPU. From a repository checkout:

```bash
python -m pip install -e '.[models,test]'
python scripts/setup_sources.py --families kev laya NanoJev
python scripts/prepare_data.py shared --output work/data/shared \
  --source-root work/upstream/kev --calibration 128 \
  --temperature-calibration 128 --development 256 --test 1024

CUDA_VISIBLE_DEVICES=0 s1q optimize --model kev-4b \
  --data-dir work/data/shared --output-dir work/runs/kev4-s1q-w4a4 \
  --methods s1q,rtn,s1q-local,s1q2-beta05,awq-adapted \
  --bits 4 --activation-bits 4 --group-size 128 \
  --calibration-count 128 --development-count 256 \
  --reservoir-size 128 --seed 20261004
```

Preparing the data creates a test file; `s1q optimize` reads calibration and development only. New runs write the configuration before evaluation, source/code hashes, native eligibility, matched predictions and metric reports. Use a new output directory for each run. The [reproduction guide](docs/reproduce-current.md) describes NanoJev, W3A4/W4A8, additional cohorts and ablation commands. Reproduction creates a new run; exact published values also depend on its recorded packages, cohort admission and checkpoint hashes.

A reusable API accepts already native-admitted calibration requests:

```python
from s1q.api import quantize_s1q

adapter.model.eval()
session, metadata = quantize_s1q(adapter, calibration_requests,
                                bits=4, activation_bits=4, group_size=128)
with session:
    logits = adapter.infer(unlabeled_request)
    session.export("work/linears-s1q-w4a4.pt")
# Native backbone weights and hooks are restored on exit.
```

The exported artifact contains packed selected linear weights and their scales, **not a standalone complete checkpoint**. Retained native parameters, model code, tokenizer and heads are still required. The October batch did not export current-method complete checkpoints or measure integer-kernel speedups.

## Execution and research boundaries

- Weights use symmetric groupwise codes; activations use dynamic per-token symmetric A4/A8 **floating QDQ simulation**. Matrix multiplications remain floating point.
- Decision heads, embeddings, normalization, biases and non-Linear recurrent operations retain native precision equally across matched methods.
- Calibration uses backward passes to measure margin sensitivity, but the selected S1Q does not train model parameters. The optional gain-repair experiment is a separate optimization-based control and is not part of current S1Q.
- The reservoir fit/selection split separates token rows from the same calibration requests. It is not an independent request-level validation set. Layer fitting uses native teacher inputs; it does not replay accumulated quantized upstream inputs.
- No INT4 throughput, new-method GPU-memory reduction, universally improved confidence calibration, or first-ever System One quantization claim is made.

The historical [v0.1.0 release](https://github.com/CYMCharming/S1Q/releases/tag/v0.1.0), packed artifacts, [results](docs/results.md) and [method document](docs/legacy-method-v0.1.md) describe earlier recipes. `s1q run` retains that legacy workflow. Manuscripts remain private; public benchmark figures are generated from released aggregate metrics. No new model weights, raw datasets or raw prediction cohorts are redistributed.

## Attribution and citation

Upstream decision models: [Kev](https://github.com/jaredpalmer/kev), [NanoJev](https://github.com/TianyuCodings/NanoJev), [Laya](https://github.com/NandhaKishorM/laya), [Intern-Decision](https://github.com/InternLM/Intern-Decision), [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision). Model/data licenses remain upstream-specific; see [NOTICE](NOTICE), [model audit](docs/model-audit.md), [model extensions](docs/decision-model-extensions.md) and [artifact attribution](docs/artifacts.md).

Quantization ingredients are attributed to [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438), [GPTQ](https://arxiv.org/abs/2210.17323), [SpinQuant](https://arxiv.org/abs/2405.16406), GuidedQuant, RSQ, ERQ and GPTAQ. Native model adapters, experiment code and the decision-aware integration are implemented here; official baseline equivalence has not been claimed.

When using S1Q, cite the software and pin the exact commit or release:

```bibtex
@software{chen2026s1q,
  author = {Chen, Yuanming},
  title = {S1Q: Low-Bit Quantization for System One Decision Models},
  year = {2026},
  url = {https://github.com/CYMCharming/S1Q}
}
```

Maintained by [CYMCharming](https://github.com/CYMCharming).
