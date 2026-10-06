# Reproducing the current S1Q

The public name is **S1Q**. New runs use `--methods s1q`; frozen October 4 runs used the compatible key `s1q-mac`. The public and frozen aliases produce identical quantized codes/scales for identical inputs. The legacy `s1q run` workflow remains available for v0.1 experiments.

## Installation and native data

From a checkout, with Python 3.12 and an appropriate CUDA PyTorch build:

```bash
python -m pip install -e '.[models,test]'
python scripts/setup_sources.py --families kev laya NanoJev
python scripts/prepare_data.py shared --output work/data/shared \
  --source-root work/upstream/kev --calibration 128 \
  --temperature-calibration 128 --development 256 --test 1024
python scripts/prepare_data.py nanojev --output work/data/nanojev \
  --calibration 128 --temperature-calibration 128 --development 256 --test 1024
```

The preparer preserves upstream/native schema, hard-label eligibility, source-group disjointness and source hashes. It writes several splits; the current optimization runner reads only `calibration.jsonl` and `development.jsonl`, plus explicitly named additional cohorts. Preparing a test file does not authorize using it for method development. Model weights are acquired by the pinned native loader or provided through `--checkpoint-dir` and an optional checksum manifest; no model weights are committed here.

The reviewed model/runtime pins live in [models.py](../src/s1q/models.py), [model audit](model-audit.md) and [runtime notes](runtime.md). Install-time packages and the exact frozen experimental environments differ. Exact numerical reproduction requires the recorded checkpoint/source/data hashes and environment, not just a method name. Some native requests may be rejected before quantization; the accepted rows are then fixed identically for all methods. NanoJev uses recorded action-argmax compatibility labels.

## Current method and ablations

```bash
CUDA_VISIBLE_DEVICES=0 s1q optimize --model kev-4b \
  --data-dir work/data/shared --output-dir work/runs/kev4-current-w4a4 \
  --methods rtn,s1q-local,s1q2-beta05,s1q-joint,s1q-ac,s1q-margin,s1q \
  --bits 4 --activation-bits 4 --group-size 128 \
  --calibration-count 128 --development-count 256 \
  --reservoir-size 128 --seed 20261004
```

Use model names `kev-0.8b`, `kev-4b`, `kev-9b`, `laya` with shared text data, or `nanojev` with its separate data. Change `--bits 3` for W3A4, or `--activation-bits 8` for W4A8; give each run a new output directory. W4A8 is a supported recipe, not a claimed completed result in the current October 4 evidence. Current S1Q requires gradients through the native forward during calibration, even though it does not fit model parameters.

The default `s1q optimize` methods are `s1q,rtn`. The script entry point can also be used from an installed checkout:

```bash
python scripts/run_optimization.py --model kev-4b \
  --data-dir work/data/shared --output-dir work/runs/kev4-script \
  --methods s1q,rtn --bits 4 --activation-bits 4
```

Without `--methods`, that historical script still runs its older full comparison set. The package CLI is the preferred current-method entry point.

## Adapted controls and extra evaluation

Append any supported controls to the method list: `awq-adapted`, `gptq-blockdiag-adapted`, `spinquant-nohad-adapted`, `spinquant-had-adapted`; extension keys are `awq-ac`, `awq-mac`, `gptq-ac`, `gptq-mac`, `spinquant-nohad-mac` and `spinquant-had-mac`. All are explicitly adapted/proxy controls. SmoothQuant is separately available through the baseline API, but was not run by this frozen optimization batch. The optional `-repair` keys are extra optimization controls, not part of final S1Q.

To evaluate a prepared native-schema cohort without using it for calibration:

```bash
s1q optimize --model kev-4b --data-dir work/data/shared \
  --output-dir work/runs/kev4-extra --methods s1q,rtn \
  --extra-evaluation wanli=work/data/extra/wanli-native.jsonl \
  --extra-evaluation mmlupro=work/data/extra/mmlupro-native.jsonl
```

Additional cohorts require the same native schema as prepared rows and explicit labels for evaluation. Names must be unique and filename-safe. Request/group overlap with calibration or another cohort fails explicitly. The frozen study used a prespecified 256-row WANLI sample and 156 eligible short MMLU-Pro rows; supplying different cohorts produces a new experiment, not the frozen scores.

## Outputs and audit

A run records:

- `frozen-config-before-evaluation.json`: methods and settings before model load/evaluation.
- `identity.json` and `report.json`: native checkpoint metadata, hashes, eligibility, calibration scope, per-method status, probabilities and storage estimates.
- `native-<cohort>.jsonl` and `<method>-<cohort>.jsonl`: locally saved matched predictions.

The calibration fit/select split is within the saved token reservoir, not independent requests. Extra evaluation labels never select quantization candidates. Gain-repair uses only the calibration teacher and is separately marked.

`python scripts/summarize_optimization.py --help` exposes an offline raw-logit audit/report generator. It recomputes the full metric set and paired source-cluster intervals without loading a model. Do not publish source datasets or raw predictions automatically: review upstream licensing and disclosure scope first. Store fresh runs under ignored `work/` by default.

The current inference path uses floating QDQ. Exporting `session.artifact()`/`session.export(path)` packs selected linears, not a complete model or integer GEMM runtime. An export from a new run must be independently checked against its QDQ predictions before claiming deployment equivalence. Historical release artifacts and resource measurements belong to v0.1.
