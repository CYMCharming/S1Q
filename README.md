# S1Q: Low-Bit Quantization for System One Decision Models

[中文说明](README.zh-CN.md) · [Results](docs/results.md) · [Method](docs/method.md) · [Evaluation protocol](docs/evaluation-design.md) · [Weights](docs/artifacts.md) · [Runtime](docs/runtime.md) · [Related work](docs/related-work.md)

S1Q is a research implementation for quantizing open Jev-like **typed decision models**: **Kev-0.8B, Kev-4B, Kev-9B, NanoJev, and Laya**. These models directly score choices, Boolean questions, or ordered levels. Evaluation therefore measures decision accuracy **and** probability quality, rather than language-model perplexity alone.

S1Q adapts activation-aware channel scaling and groupwise clipping to native decision-model backbones. It also tests teacher categorical-Fisher weighting of output channels and uses held-out development decisions to select among weight-only, activation-quantized, and selective higher-precision configurations. Native decision heads and recurrent state arithmetic retain their original precision. No claim is made that these individual quantization ideas are new.

**Research snapshot:** five real A100/A800 GPU model experiments, development ablations, source-transfer/game OOD evaluation, and a separate frozen public JevBench cohort. This repository does **not** claim to be the first quantization of System One models: prior Laya INT4/INT8 and Kev INT8 work exists. See the primary-source [prior-art audit](docs/related-work.md).

## Results at a glance

Main-cohort accuracy (%), with identical quantized layer scope and activation precision for the matched RTN comparison. Four models use 914 typed decisions from the screened Kev suites; NanoJev uses 1,023 native game decisions. These cohorts are exploratory after pilot inspection; the separate 85-task JevBench cohort is evaluated with already frozen profiles. See [full results, paired confidence intervals, probability metrics and negative outcomes](docs/results.md).

| Model | Selected configuration | Native | Matched RTN | S1Q | Complete parameter storage / native |
|---|---|---:|---:|---:|---:|
| Kev-0.8B | W4, decision-Fisher weighted | 82.93 | 80.63 | 81.18 | 51.6% |
| Kev-4B | W4, activation-aware | 85.45 | 84.79 | 84.79 | 37.8% |
| Kev-9B | W4A8 simulation | 86.98 | 84.35 | 86.54 | 36.0% |
| NanoJev | W4, activation-aware | 79.86 | 79.47 | 80.45 | 36.0% |
| Laya | W4, activation-aware | 66.85 | 63.68 | 64.55 | 29.4% |

Storage includes retained native components and is a parameter-byte estimate. It is not a measured speedup. W4A4 had substantial development-set degradation; this study supports a W4 backbone starting point with model-specific checks, rather than a universal four-bit activation recipe. S1Q is not consistently superior: Kev-0.8B loses 2.38 accuracy points to RTN on source transfer, and several confidence intervals include zero.

## What is implemented

- Revision-pinned native model adapters; labels and targets are removed before inference.
- Symmetric groupwise W4/W8 RTN, activation-aware scale/clipping search, optional decision-Fisher weighting.
- W4A8/W4A4 **simulation** on selected linear inputs, with explicit mixed-precision accounting.
- Disjoint quantization calibration, temperature calibration, development and final-test data; request/episode overlap checks.
- Accuracy, NLL, multiclass Brier, ECE, selective coverage, probability drift, harmful decision flips and cluster-bootstrap intervals.
- Packed INT4/INT8 linear artifacts, restoration, and a low-storage reference runtime that dequantizes one layer at a time.

**Execution boundary:** current matrix multiplications use floating point. Activation quantization is simulated. Packed storage and GPU memory reduction are distinct from native INT4-GEMM acceleration; this project makes no speedup claim. Nonlinear operators, embeddings, decision heads and hybrid recurrent state updates are not all reduced to four bits.

## Quick start

Use an isolated Python environment and a PyTorch build compatible with your GPU. Python 3.12 is recommended for the current Kev source.

```bash
python -m pip install -e '.[models,test]'
python scripts/setup_sources.py
python scripts/prepare_data.py shared --output work/data/shared \
  --source-root work/upstream/kev --calibration 128 \
  --temperature-calibration 128 --development 256 --test 1024

CUDA_VISIBLE_DEVICES=0 s1q run --model kev-0.8b \
  --data work/data/shared --output results/kev-0.8b-reproduction \
  --bits 4 --fisher
```

Available model names: `kev-0.8b`, `kev-4b`, `kev-9b`, `nanojev`, `laya`.

NanoJev's released checkpoint is specialized for games. Prepare its native hard-label dataset separately:

```bash
python scripts/prepare_data.py nanojev --output work/data/nanojev
s1q run --model nanojev --data work/data/nanojev \
  --output results/nanojev-reproduction --bits 4 --fisher
```

Upstream game samples that only contain Jev teacher probabilities are excluded from hard-label accuracy. Expert-action agreement is reported separately from observed-outcome probability calibration. Whole episode components sharing identical requests are kept together; lower-priority overlaps are excluded using a published fixed policy.

## Evaluate or inspect packed weights

The [v0.1.0 weight release](https://github.com/CYMCharming/S1Q/releases/tag/v0.1.0) provides the exact evaluated packed linears for Kev-0.8B, Kev-4B, Kev-9B and Laya, with byte hashes and native checkpoint pins. NanoJev has a local reproduction recipe and full results; its audited model card does not provide a separate explicit fine-tuned weight license, so its packed weights are not redistributed. See [artifact usage and attribution](docs/artifacts.md).

```bash
python scripts/fetch_artifacts.py --model kev-0.8b --output-dir artifacts/released
```

```python
from s1q.models import load_model
from s1q.quantization import load_quantized_artifact

adapter = load_model("kev-0.8b")
session = load_quantized_artifact(adapter.backbone, "artifacts/released/kev-0.8b-s1q-fisher-w4.pt")
# The artifact contains selected linears; the same pinned native model is required.
logits = adapter.infer(request)
session.restore()
```

For persistent packed storage, restore and delete the session, then use `s1q.packed.apply_packed`. The [benchmark script](scripts/benchmark_packed.py) compares its predictions with the same artifact's dense reference and measures persistent and peak GPU allocation separately.

## Model ecosystem and attribution

S1Q studies independent open implementations inspired by [TypeSafe Jev](https://docs.typesafe.ai/introduction). It is not affiliated with TypeSafe, and does not quantize the closed hosted Jev model.

| Project | Native implementation | Upstream |
|---|---|---|
| Kev | Qwen hybrid backbone, merged LoRA, pointer head | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) |
| NanoJev | Qwen3-0.6B, dynamic candidate decision heads | [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) |
| Laya | ModernBERT encoder, typed decision head | [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) |

The quantization design draws on [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438), and sensitivity-weighted reconstruction literature. Related decision-distribution work and close Fisher/gradient approaches are explicitly discussed in [Related work](docs/related-work.md) and [Method](docs/method.md).

## Reproducibility and scope

Every completed run records model/source revisions, package versions, dataset hashes, eligibility exclusions, development selection, per-question predictions and matched RTN results where precision policies differ. The inspected exploratory Laya pilot motivated a Fisher extension; its inspected cohorts are not presented as fresh confirmation. External JevBench evaluation hashes its inputs before inference and reuses frozen model profiles and temperatures.

Rebuild the report with `python scripts/summarize_results.py`. Install `python -m pip install -e '.[plots]'` for `python scripts/plot_results.py`; the published figure snapshot used Matplotlib 3.8.4. Independent external evaluation is in [evaluate_external.py](scripts/evaluate_external.py); it does not fit temperatures or tune quantization on the external cohort. The [runtime audit](docs/runtime.md) distinguishes recorded experimental packages from the fresh supported installation profile.

Dataset source JSONL and full upstream clones are not committed. Model and dataset licenses remain upstream-specific; see [NOTICE](NOTICE) and the evaluation documentation. Publication of recipes, identifiers and derived metrics does not relicense source material.

## Citation

Until a paper is available, cite the software and pin the exact release/commit:

```bibtex
@software{chen2026s1q,
  author = {Chen, Yuanming},
  title = {S1Q: Low-Bit Quantization for System One Decision Models},
  year = {2026},
  url = {https://github.com/CYMCharming/S1Q}
}
```

Maintained by [CYMCharming](https://github.com/CYMCharming).
