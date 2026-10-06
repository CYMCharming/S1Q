# S1Q: Low-Bit Quantization for System One Decision Models

[中文说明](README.zh-CN.md) · [Method](docs/method.md) · [Current reproduction](docs/reproduce-current.md) · [Related work](docs/related-work.md) · [Model audit](docs/model-audit.md) · [Legacy results](docs/results.md) · [Packed artifacts](docs/artifacts.md)

**S1Q now denotes the current decision-margin-aware, activation-compensated method.** Its October 2026 experiment identifier was `s1q-mac`; that identifier remains supported to reproduce frozen runs. The public method name is **S1Q**. Earlier S1Q and S1Q2 recipes remain available as historical controls.

S1Q quantizes open Jev-like **System One decision models** that map a state and typed questions directly to finite choice, Boolean or discrete-level distributions. The implementation supports pinned Kev, NanoJev and Laya adapters; it preserves native candidate order and decision heads. It is an independent research project and is not affiliated with [TypeSafe Jev](https://docs.typesafe.ai/introduction).

## Method

S1Q combines three calibration steps:

1. Backpropagate each teacher decision's top-one versus runner-up logit margin separately. Use aligned, bounded token gradient energy to sample calibration rows that matter to candidate comparisons. No calibration gold labels are used.
2. Search channel scales and clipping using the **actual quantized weight and activation reconstruction**. Both W and A error enter the local output objective.
3. Fit a bounded ridge correction for activation error before quantizing weights. Select corrected or uncorrected candidates using a separate half of the saved token reservoir.

The gradient statistic is a **margin-Jacobian token proxy, not a categorical Fisher matrix**. [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html) already uses end-loss gradients to guide reconstruction, and [RSQ](https://openreview.net/pdf?id=kBezrKXHVS) already prioritizes important tokens in quantization. Ridge-based activation-error compensation has close prior art in [ERQ (ICML 2024)](https://proceedings.mlr.press/v235/zhong24a.html), and matching native outputs has close prior art in [GPTAQ](https://arxiv.org/abs/2504.02692). Gradient guidance, token importance, scaling, clipping and compensation are not individually new. The research contribution being evaluated is their concrete decision-aware integration across native typed-decision architectures. See the exact formulas, assumptions and limitations in [Method](docs/method.md).

## Current evidence

The frozen October 4 batch contains five models, W4A4 and two-model W3A4 stress tests, ablations and adapted quantization controls. Representative **development** accuracy (%):

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

The historical [v0.1.0 release](https://github.com/CYMCharming/S1Q/releases/tag/v0.1.0), packed artifacts, [results](docs/results.md) and [method document](docs/legacy-method-v0.1.md) describe earlier recipes. `s1q run` retains that legacy workflow. Manuscripts and their tables/figures are **not included in this code update**. No new model weights, raw datasets or raw prediction cohorts are redistributed.

## Attribution and citation

Upstream decision models: [Kev](https://github.com/jaredpalmer/kev), [NanoJev](https://github.com/TianyuCodings/NanoJev), [Laya](https://github.com/NandhaKishorM/laya). Model/data licenses remain upstream-specific; see [NOTICE](NOTICE), [model audit](docs/model-audit.md) and [artifact attribution](docs/artifacts.md).

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
