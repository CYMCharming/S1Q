# Released quantization artifacts

**Historical v0.1 artifacts:** the downloadable weights below belong to earlier S1Q recipes, not the current decision-margin/activation-compensation method. Current S1Q supports packed selected-linear export through [the API](../src/s1q/api.py); this code update does not publish newly evaluated model weights. See [Current reproduction](reproduce-current.md).

The public weight release covers **Kev-0.8B, Kev-4B, Kev-9B, and Laya**. S1Q publishes code, recipes, and evaluation results for NanoJev, but does not redistribute its fine-tuned packed weights because the audited release does not state a separate explicit fine-tuned weight license. Download the original NanoJev checkpoint from its upstream repository and reproduce its recipe locally. See the attribution details below.

These files contain selected quantized backbone linear weights and their scales. They require the identical pinned native checkpoint, tokenizer, architecture, and retained decision head. They are **not standalone model checkpoints**. The group size, activation scope, protected layers, loaded floating-point dtype, and selected method can differ across entries; read the catalog rather than assuming every file is an entirely W4A4 model.

## Download and identify a release

From a checkout of S1Q with its `LICENSE` and `NOTICE` files:

```bash
python -m pip install -e '.[models]'
python scripts/setup_sources.py
python scripts/fetch_artifacts.py --model kev-0.8b --output-dir artifacts/released
```

Use the package versions in the corresponding run's `environment.json` for faithful reproduction. Kev's pinned upstream implementation requires the Qwen3.5 support in Transformers 5.17 or newer; the tested runtime, rather than a minimum version bound alone, is the reproduction reference. Loading the native model also downloads its required pinned upstream files unless they are already cached.

`configs/release-artifacts.json` uses the `s1q-release-v1` schema. Its `artifacts` array contains downloadable assets. Each entry records:

| Field | Meaning |
| --- | --- |
| `model`, `run`, `profile` | Model identity, completed experiment directory, and frozen selected configuration |
| `model_metadata` | Exact native model/source revisions, loaded dtype, and model-specific context/head metadata; Kev also identifies its base revision |
| `bits`, `group_size` | Weight quantizer precision and groups along the input dimension |
| `scope`, `storage` | Retained components and the estimated parameter storage accounting |
| `summary_sha256` | SHA256 of the completed evaluation summary corresponding to this artifact |
| `filename`, `artifact_bytes`, `artifact_sha256` | Reconstructed artifact identity, exact byte count, and SHA256 |
| `parts` | Ordered release asset names, byte counts, and individual SHA256 values |

Parts are concatenated in **catalog order**. The downloader verifies every part and the whole artifact before committing the final file, and refuses to overwrite an existing file. A failed owned temporary download is removed. Successful verification establishes equality to the catalog's bytes; obtain the catalog from the matching S1Q source/release rather than treating an arbitrary replacement catalog as an independent authenticity guarantee.

A `recipe_only` entry, when present, describes a nonredistributed model such as NanoJev. Its pins, internal evaluated artifact hash, and scope support reproduction; it is outside `artifacts` and has no downloadable S1Q weight parts. `fetch_artifacts.py` selects downloadable entries only.

For example, inspect the model and precision before loading:

```python
import json
from pathlib import Path

catalog = json.loads(Path("configs/release-artifacts.json").read_text())
entry = next(item for item in catalog["artifacts"] if item["model"] == "kev-0.8b")
artifact_path = Path("artifacts/released") / entry["filename"]
print(entry["model_metadata"], entry["profile"], entry["scope"])
```

## Reproduce the dense evaluation path

`load_quantized_artifact` reconstructs the transformed floating-point weights and reinstalls the diagonal input hooks used during evaluation. Its context manager restores the original weights on exit. Loading just a transformed `state_dict` without these hooks is incorrect.

```python
import torch
from s1q.models import load_model
from s1q.quantization import load_quantized_artifact

dtype = entry["model_metadata"]["dtype"].removeprefix("torch.")
adapter = load_model(entry["model"], device="cuda", dtype=dtype,
                     source_dir="work/upstream")
actual = {k: v for k, v in adapter.metadata.items() if k != "device"}
expected = {k: v for k, v in entry["model_metadata"].items() if k != "device"}
if actual != expected:
    raise RuntimeError("Native checkpoint or inference configuration differs from release")
adapter.model.eval()

request = {"state": "The target is left of the aim.", "questions": {
    "action": {"type": "choice", "instructions": "Choose the next action.",
               "criteria": {"left": "Move the aim left.",
                            "right": "Move the aim right.", "wait": "Wait."}}}}
with torch.inference_mode(), load_quantized_artifact(adapter.backbone, artifact_path):
    logits = adapter.infer(request)
    print(torch.softmax(logits[0], dim=-1).cpu().tolist())
```

This prints probabilities at temperature one in native candidate insertion order. Boolean order is `[no, yes]`. Gold labels are unnecessary for inference. Probability metrics in the completed experiments may additionally use a fitted, frozen temperature; those values are in the run's selected `metrics.json`/`summary.json`, not supplied implicitly by the weight artifact. Do not fit a new temperature on a test set to reproduce a calibrated result.

## Keep weights packed between calls

For persistent packed storage, start with a fresh native model and apply the same verified artifact:

```python
from s1q.packed import apply_packed, packed_report

adapter = load_model(entry["model"], device="cuda", dtype=dtype,
                     source_dir="work/upstream")
adapter.model.eval()
adapter.backbone = apply_packed(adapter.backbone, artifact_path)
with torch.inference_mode():
    logits = adapter.infer(request)
print(packed_report(adapter.model))
```

The audited native backbones are modules that are replaced in place, so their enclosing decision models keep the same backbone reference. This loader does not retain original dense linears for restoration. If converting an existing dense quantization session instead, first call `session.restore()`, delete the session and its retained module references, then apply packed storage. Rebuild the native model to return from packed storage to a different configuration.

`PackedLinear` unpacks one weight matrix and executes floating-point linear algebra for each call. It has no native INT4/INT8 GEMM kernel. Optional A4/A8 simulates selected linear inputs; callers accessing a reconstructed `.weight` directly bypass that activation hook. Check architecture coverage and prediction parity against the dense path before making accuracy or memory claims. The [packed benchmark](../scripts/benchmark_packed.py) compares the same artifact's paths and measures persistent and peak CUDA allocation separately. Small files do not establish faster inference or a reduction in peak memory.

The measured peak covers inference after conversion. This reference loader first constructs the full native checkpoint; its initial allocation is excluded from that peak. The persistent packed-memory figure therefore does not establish that the model can initially load on a device of that size.

## NanoJev local reproduction

The native pins are `C-Tianyu/NanoJev@047b927b30882a1138fc504821b82ac145a4b81a` and `TianyuCodings/NanoJev@76fdfc9ecdca45a9bcef17991a07d3041a87685a`. The dataset pin and audited split hashes are recorded in `configs/nano-data-checksums.json`. S1Q's adapter downloads the original checkpoint directly from upstream; S1Q does not mirror it.

```bash
python scripts/prepare_data.py nanojev --output work/data/nanojev
s1q run --model nanojev --data work/data/nanojev --output results/nanojev-reproduction --bits 4 --fisher
```

This reruns calibration/development selection on the declared native dataset and exports a local artifact when the run completes. Match the published run's environment and configuration for reproduction. Serialization or numerical differences across environments can change the local file hash; a recipe-only hash identifies the evaluated internal file and is not a promise of a public download. Test and external cohorts must not be used to change the recipe.

## Upstream attribution and distribution boundaries

Quantized matrices are modified derivatives of the upstream weights. The S1Q source license does not relicense them. Preserve the applicable license text, copyright/attribution notices, any upstream `NOTICE`, and the notice that S1Q quantized selected backbone linears while retaining native components. The model/source/base pins in the catalog identify their origin. [Apache-2.0 redistribution conditions](https://www.apache.org/licenses/LICENSE-2.0#redistribution) describe the required license copy, modification notices, and retained attribution.

- **Kev:** [jaredpalmer/kev](https://github.com/jaredpalmer/kev), its Apache-2.0 adapter/head releases, and the corresponding Apache-2.0 Qwen3.5 base. Merged adapter modifications are included in the quantized backbone; the native pointer head is obtained separately from the pinned checkpoint.
- **Laya:** [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) and [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), with Apache-2.0 runtime/model. S1Q retains its native decision head and tokenizer.
- **NanoJev:** [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) source is MIT, copyright 2026 OpenJev contributors. Its [pinned model card](https://huggingface.co/C-Tianyu/NanoJev/blob/047b927b30882a1138fc504821b82ac145a4b81a/README.md) states: “Source code carries the included MIT license; the Qwen base model retains its upstream license.” The card has no `license` metadata field or separate explicit fine-tuned weight grant. Its [pinned Qwen3-0.6B base license](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE) is Apache-2.0, copyright 2024 Alibaba Cloud. These are distinct facts; the source MIT license must not be described as an independently verified MIT license for the fine-tuned weights. S1Q therefore publishes NanoJev recipes/results rather than redistributed packed weights.

Full immutable references and native implementation details are in [the model audit](model-audit.md). The closed hosted [TypeSafe Jev](https://docs.typesafe.ai/introduction) inspired this model ecosystem; it is not a distributed checkpoint in this release, and these independent implementations should not be presented as identical to it or endorsed by its authors. Model weights, source code, and datasets have separate provenance and terms.
