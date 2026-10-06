# Related work and novelty audit

Original audit: 2026-10-01; current-method additions: 2026-10-06 (Asia/Shanghai). Primary papers, model cards, source
repositories and the authors' experiment reports were inspected. This is a
bounded search, not proof that no other relevant work exists.

## Naming and scope

The project title is **S1Q: Low-Bit Quantization for System One Decision Models**.
Use “System One” as the motivating product/model paradigm and define the studied
objects operationally: models taking a state plus typed questions and returning
distributions over supplied decisions, without free-text generation. The sampled
models do not share one architecture. “Jev-like” is a functional/inspirational
description; official Jev is not an open checkpoint reproduced by this project.

Attribute the term and API to [TypeSafe's documentation](https://docs.typesafe.ai/introduction).
Describe independent implementation choices through [Kev](https://github.com/jaredpalmer/kev),
[NanoJev](https://github.com/TianyuCodings/NanoJev) and
[Laya's model card](https://huggingface.co/convaiinnovations/laya).

Searches for `"S1Q" quantization` and `"S1Q" "System One"` did not identify an
established matching machine-learning method in the inspected results. Many hits
use S1Q as unrelated mathematical notation. This supports retaining the working
name; it is not a trademark search or an exhaustive name-collision guarantee.

## Direct prior quantization of these models

**Do not claim “the first quantization of Jev-like/System One models.”** Published
low-bit Laya artifacts and Kev experiments already contradict that broad statement.

| Primary source | Verified scope | Consequence for S1Q |
| --- | --- | --- |
| [androidli/laya-multilingual-onnx-int4](https://huggingface.co/androidli/laya-multilingual-onnx-int4) | Public weight-only INT4 ONNX projections/embedding conversion, scripts and a 60-item accuracy check. | Prior 4-bit Laya work; cite it. Its small single-workload evaluation leaves room for broader probability/decision analysis. |
| [laya-mlx engineering report](https://github.com/mizorewww/laya-mlx/blob/main/docs/ENGINEERING_10X_RESEARCH.md) | Encoder-only affine 8/4-bit quantization experiments with higher-precision heads and probability/answer agreement diagnostics. | Protected heads and decision-drift measurement are not new alone. The report also cautions that storage savings did not establish speedup. |
| [Kev issue #162](https://github.com/jaredpalmer/kev/issues/162), opened 2026-09-27 | Kev-4B MLX 8-bit group64 versus BF16 on five development suites, pointer head FP32; accuracy, probability metrics and paired intervals. | Prior decision/calibration-aware empirical comparison for Kev. S1Q must extend, rather than ignore, this evidence. |
| [DreamBlooms/kev-0.8b-GGUF](https://huggingface.co/DreamBlooms/kev-0.8b-GGUF) and [dohnuts.cpp](https://github.com/DreamBlooms/dohnuts.cpp) | Public merged Q8_0 Kev-0.8B backbone plus FP32 pointer head; source conversion path. Model card says it has not been benchmarked against the PyTorch reference. | Prior open quantized Kev artifact, but not a substitute for a matched reference evaluation. |
| [llama.cpp Laya support PR #29363](https://github.com/ggml-org/llama.cpp/pull/29363) | Model conversion/runtime proposal with Laya quantization support visible in its commit description. | Additional engineering prior art; PR state is mutable and is not evidence of an upstream released runtime by itself. |
| [ZeroDegress/NanoJev-bf16](https://huggingface.co/ZeroDegress/NanoJev-bf16) | FP32-to-BF16 weight conversion, explicitly described as a precision variant. | This is not calibrated integer PTQ. Still acknowledge prior reduced-precision NanoJev deployment. |

The audit has not reproduced any source's published numbers. Different checkpoint
revisions, suites, temperatures and runtimes cannot be placed in a causal S1Q
before/after table. They are related work, not our own results.

## Existing quantized decision-model releases

The ecosystem already includes the [self-contained Kev-27B NF4 release](https://huggingface.co/TheCulliganMan/kev-27b-nf4), including its quantized backbone, original adapter and pointer head; [Laya Q8_0 GGUF](https://huggingface.co/ggml-org/Laya-GGUF/tree/main); and [jevos OpenVINO INT8](https://github.com/feder-cr/jev) for native yes/no decisions. These directly preclude a first-ever Jev-like/System One quantization or first compressed native decision-model claim. Their checkpoints, formats, runtimes and evaluation settings differ from this study. They are related ecosystem work, not matched S1Q baselines, and their reported scores/resource measurements are not mixed into our tables.

## Established quantization ingredients

For the current S1Q, close references include [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html), which uses end-loss gradient guidance while retaining cross-weight dependencies, and [RSQ](https://openreview.net/pdf?id=kBezrKXHVS), which uses important-token feature scaling and quantization statistics. Gradient-guided and token-aware quantization are established ideas. S1Q's specific adaptation uses native top-two decision-margin gradients and applies bounded token importance once through a priority reservoir; it is not a first gradient-guided quantizer or a full Fisher estimator.

[ERQ (ICML 2024)](https://proceedings.mlr.press/v235/zhong24a.html) already fits a closed-form ridge correction for activation error before weight quantization; its [expanded journal paper](https://arxiv.org/abs/2407.06794) is also relevant. [GPTAQ](https://arxiv.org/abs/2504.02692) calibrates against native full-precision outputs while addressing preceding-layer asymmetry. The S1Q ridge formula is not claimed as new. Our implementation uses native teacher layer inputs rather than replaying quantized preceding layers, so it does not implement GPTAQ's full asymmetric calibration. The current token importance is a margin-Jacobian norm proxy, not the categorical Fisher estimator used by an earlier recipe. See [Method](method.md) and [the archived v0.1 method](legacy-method-v0.1.md).

| Method | Relevant idea | Appropriate S1Q usage |
| --- | --- | --- |
| [GPTQ](https://arxiv.org/abs/2210.17323) | Approximate second-order local reconstruction for low-bit weights. | Established weight-only reference or ingredient; account for recurrent/hybrid model projections and LoRA merging. |
| [AWQ](https://arxiv.org/abs/2306.00978) | Activation statistics identify sensitive weight channels; equivalent channel scaling protects them. | Motivation/reference for activation-aware scale search. A reduced heuristic must be called AWQ-inspired, not an exact reproduction. |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | Equivalent rescaling moves activation outliers' difficulty into weights for W8A8. | A later activation-quantization path; verify architecture-specific equivalent transformations. |
| [OmniQuant](https://arxiv.org/abs/2308.13137) | Learned clipping and equivalent transformations within PTQ reconstruction. | Clipping search is prior art; an output-decision objective requires clear distinction and ablation. |
| [QuaRot](https://arxiv.org/abs/2404.00456) | Rotation-based removal of outliers for 4-bit weight/activation/cache inference. | Possible W4A4 ingredient for supported Transformer paths; do not assume it commutes with GELU, DeltaNet state updates or an arbitrary decision head. |
| [SpinQuant](https://arxiv.org/abs/2405.16406) | Learned rotations improve low-bit model behavior. | Rotation learning is established; any typed-decision modification must be specified and tested. |
| [QQQ](https://arxiv.org/abs/2406.09904) | W4A8 adaptive smoothing and Hessian compensation with specialized GEMM. | Illustrates that numerical formats and real kernels are separate requirements for acceleration. |

RTN is a necessary simple baseline, but beating RTN alone does not establish
state-of-the-art quantization. Equalize bit format, group size, quantized layer
coverage, protected heads, calibration budget, dtype and post-hoc calibration.
Mixing published ingredients is an adaptation until a new objective/procedure and
its benefit are demonstrated. Attribute mathematical components and reused code,
respect licenses and keep the public method description faithful to implementation.

## Uncertainty and calibration prior art

[Doubt-Preserving Quantization (DPQ)](https://arxiv.org/abs/2608.21019),
published 2026-08-21, uses full-precision uncertainty to select a calibration
mixture before quantization. Its [implementation](https://github.com/xi-xiaoran/DPQ)
and [paper](https://arxiv.org/html/2608.21019v1) explicitly examine probability drift,
top-two margins, abstention boundaries and full-precision agreement. Consequently,
low-margin calibration selection, preserving confidence, and reporting JS distance
to FP cannot each be presented as S1Q's first novel idea.

Temperature scaling and calibrated neural classification have a long history;
[Guo et al.](https://arxiv.org/abs/1706.04599) are a baseline reference. Distinguish
**PTQ calibration data** used to select quantization parameters from **probability
calibration** measured against labels. Matching FP probabilities does not prove
that the FP probabilities were calibrated to truth. Post-hoc temperature fitting
must not be credited as an exclusive quantizer improvement when other methods
were denied the same fitting data.

## What S1Q can defensibly contribute

The current contribution is a reproducible cross-architecture study and a concrete
typed-decision adaptation: aligned top-two margin-Jacobian token sampling,
joint W/A local reconstruction and bounded activation compensation. The frozen
October batch includes unquantized, RTN, ordinary joint reconstruction and full
S1Q controls, with both positive and negative results. These experiments support
some difficult low-bit gains; they do not establish a new ridge solver, universal
decision preservation, consistent near-boundary improvement or superiority over
official quantizer implementations. The estimator is not a full Fisher matrix.

If the implementation modifies channel scale/clipping objectives using output
decision sensitivity, show the exact loss, candidate masking/order handling,
search space and protection policy. Compare output weighting against the same
unweighted quantizer and distinguish it from DPQ-style data selection. If the
experiment merely combines existing protected heads and activation-weighted
rounding, describe it as an adapted recipe and study rather than asserting a
fundamentally new quantization algorithm.

Recommended public wording before final results:

> S1Q is an open research framework for low-bit quantization of System One decision
> models. It studies decision and probability preservation across Kev, NanoJev
> and Laya, with matched unquantized and quantization baselines.

Recommended wording after reproduced positive results:

> We adapt activation-aware post-training quantization to typed decision outputs
> and evaluate its effect on accuracy, probability fidelity, calibration and
> resource use across multiple decision-model architectures.

A narrow “to our knowledge, first systematic study across these named families”
statement would need a fresh documented audit at paper submission and a clearly
defined meaning of “systematic”. It is optional and currently unverified. Prefer
concrete coverage and results to a global priority claim. Prominent links, adapter
examples, reproducible quantized artifacts and an honest comparison with existing
Jev ecosystem work provide discoverability without exaggeration.
