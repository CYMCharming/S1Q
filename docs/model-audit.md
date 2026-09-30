# Open System One model audit

Verified on 2026-10-01 using immutable upstream source checkouts, Git remote
references, and model configuration files. This audit concerns open Jev-inspired
implementations, not the closed TypeSafe Jev weights. Architecture and benchmark
claims from an earlier conversation were not assumed to be correct.

## Pinned releases

| Model | Hugging Face repository | Weight commit | Architecture | Distribution |
|---|---|---|---|---|
| Kev-0.8B | `jaredpalmer/kev-0.8b` | `9a45d25eb2ab761841196625383fa1dff0e56c1e` | Qwen3.5-0.8B-Base, LoRA r16, pointer head | Apache-2.0 adapter/head and Apache-2.0 base |
| Kev-4B | `jaredpalmer/kev-4b` | `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` | Qwen3.5-4B-Base, LoRA r16, pointer head | Apache-2.0 adapter/head and Apache-2.0 base |
| Kev-9B | `jaredpalmer/kev-9b` | `2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c` | Qwen3.5-9B-Base, LoRA r16, pointer head | Apache-2.0 adapter/head and Apache-2.0 base |
| NanoJev | `C-Tianyu/NanoJev` | `047b927b30882a1138fc504821b82ac145a4b81a` | Qwen3-0.6B with scalar and attention-based candidate-set head | Source MIT; model card retains Qwen base license without a separate explicit Hub weight-license field |
| Laya English | `convaiinnovations/laya` root | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | ModernBERT-large plus two transformer head layers, option scorer, act head | Apache-2.0 model and runtime |

Code revisions used by the adapters:

- [Kev](https://github.com/jaredpalmer/kev/tree/0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91): `0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91`.
- [NanoJev](https://github.com/TianyuCodings/NanoJev/tree/76fdfc9ecdca45a9bcef17991a07d3041a87685a): `76fdfc9ecdca45a9bcef17991a07d3041a87685a`.
- [Laya](https://github.com/NandhaKishorM/laya/tree/6d942c92081fbc139e736bbd9ac0023223c29b7f): `6d942c92081fbc139e736bbd9ac0023223c29b7f`.

Kev checkpoint metadata pins its base independently of its adapter. Published
suite manifests identify these base revisions:

| Base | Revision |
|---|---|
| `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| `Qwen/Qwen3.5-9B-Base` | `68c46c4b3498877f3ef123c856ecfde50c39f404` |

The loaded `head.pt` metadata is the final authority; the adapter records its
actual `base` and `base_revision` in the experiment manifest. NanoJev's run
configuration records its original Qwen3-0.6B base revision as
`c1899de289a04d12100db370d81485cdf75e47ca`; inference loads the saved full decision
checkpoint and local backbone configuration, so it need not download that base.

NanoJev's `unified-games-v1` is an **annotated tag**. Its model tag object is
`456cea98b6eb4990c48a52ddcbb3d097189c19cc`, but the actual model commit is the
`047b927...` revision above. Its dataset tag object is
`fd4d07b63aec3ba8dda483e9434c9855964b5931`; the actual dataset commit is
`7afc5257c0f3ff0ba08512729888a51d94b40e7e`. Pin the commit, not the tag object.

## Kev adapter and quantization boundaries

[Checkpoint loading](https://github.com/jaredpalmer/kev/blob/0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91/kev/checkpoint.py)
is the canonical loader. A released directory contains an adapter, tokenizer and
`head.pt`; it is not a conventional standalone causal-LM checkpoint. Load with:

```python
from kev.checkpoint import Checkpoint, LoadOptions
tok, model = Checkpoint("jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e").load(
    "cuda", LoadOptions(dtype=torch.bfloat16, merge=True, temperature=1.0,
                        backend="torch", fused=False, cuda_graphs=False))
```

Merge the LoRA first, then quantize `model.lm`. Leave `model.head` in FP32. The
released head temperatures are 2.35 / 2.41 / 2.30 for 0.8B / 4B / 9B. Setting
`temperature=1.0` obtains raw logits; native served-temperature and separately
refitted-temperature results should be reported explicitly.

The [model implementation](https://github.com/jaredpalmer/kev/blob/0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91/kev/model.py)
contains full attention and Gated DeltaNet layers. Hybrid backbones run each
question as a causal row, since recurrent layers cannot honor the packed branch
mask. Candidate pointer logits use the decision token and option-end hidden
states. Backbone projections include attention `q_proj/k_proj/v_proj/o_proj`,
MLP `gate_proj/up_proj/down_proj`, and DeltaNet
`in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj`. The first W4 study can target
merged linear weights while retaining convolution, normalizations and recurrent
state arithmetic. Replacing DeltaNet output/inputs with W4A4 requires measuring
the recurrence's accumulated error, not merely validating a dense-layer formula.

Use public `state/questions` requests with `api.SystemOneRequest` and
`api.to_record`, then `model.encode(..., strict=True)` and `model.forward(enc)`.
Labels/targets are excluded. Training-context defaults are 384 state tokens,
1024 per causal row and 2048 packed tokens. Over-budget requests are rejected;
the benchmark must account for those rejections. Do not mix this path with the
optional fused Triton, CUDA graph, or shared-prefix serving path when judging
quantization-induced differences.

The pinned project requires Python >=3.12,<3.14, Transformers >=5.17,<6, PEFT
>=0.21, and Torch >=2.6,<2.9. An existing Transformers 5.16 environment is below
the source's declared requirement. Verify compatibility or install the declared
version before interpreting failures as quantization problems.

## Laya adapter and quantization boundaries

The [runtime](https://github.com/NandhaKishorM/laya/blob/6d942c92081fbc139e736bbd9ac0023223c29b7f/laya/agent.py)
loads `rl_agent_config.json`, `model.safetensors`, `encoder/` and `tokenizer/`:

```python
agent = laya.load("convaiinnovations/laya",
                  revision="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
                  device="cuda", fast=False, compile=False)
```

Use `agent.model.encoder` as the quantization backbone. Keep its decision head,
type embeddings, option scorer and act head unchanged for the first study.
`agent.predict` returns already calibrated, rounded answers, so S1Q reads raw
`agent.model(...)` logits. The adapter uses the native schema checking,
normalization, sequence builder and collator and records shipped temperatures.
Explicit autocast precision avoids accidental changes between methods.

[Sequence construction](https://github.com/NandhaKishorM/laya/blob/6d942c92081fbc139e736bbd9ac0023223c29b7f/laya/common.py)
places an option marker before each candidate and includes the state in each
question row. The English root defaults to 512 total tokens and a 192-token
question/head budget. It can truncate the state or collapse different candidate
descriptions into identical token spans. S1Q detects both with native statistics
and rejects them by default. Banking77 in Kev's suite can therefore cause
coverage differences; publish the retained cohort/rejection counts, and use
appropriate lower-cardinality tasks or a separately documented budget setting.

The same Hub root also bundles `multilingual` and `typed-decisions` subfolders;
these are different checkpoints and should receive separate result rows if
used. English root weights have LFS SHA256
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`
and size 842,609,210 bytes. Base-English task quality must be established before
claiming that a preserved low-bit score demonstrates a useful decision engine.

## NanoJev adapter and evaluation scope

The [native inference implementation](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/predict_toy_decisions.py)
expects local `best.safetensors`, `config.json`, `tokenizer/` and
`backbone_config/`. S1Q downloads precisely these files from the pinned commit
and constructs the full model before strictly loading all tensors. Selected
weights SHA256 is
`f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b`,
size 2,385,039,280 bytes. This checkpoint stores FP32 parameters and was trained
with BF16 forward autocast; its maximum candidate path length is 8192.

The [decision model](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/scripts/train_toy_decisions.py)
uses `model.backbone` and an EOS representation for each candidate path. Choice
adds an attention-based set head; Boolean emits logits `[0,z]` from one semantic
path; Score emits an ordered distribution. Keep the LayerNorm, scalar head and
candidate-set head high precision initially. This is an attention-only Qwen3
backbone, simpler for common LLM quantizers than current hybrid Kev.

Native schemas differ: NanoJev uses `boolean` where Kev/Laya use `noul`, requires
nonempty text in candidate descriptions, supports 2-255 Choice candidates and
2-10 Score levels, and rejects over-length inputs. S1Q maps only the type alias;
it does not silently rewrite null criteria or fabricate probabilities. Generic
Kev tasks with unsupported descriptions must be excluded or explicitly adapted
as a separately documented presentation experiment.

The [release notes](https://github.com/TianyuCodings/NanoJev/blob/76fdfc9ecdca45a9bcef17991a07d3041a87685a/docs/UNIFIED_DEVELOPMENT_RELEASE.md)
identify a game-specialized `unified-games-v1` release. Use
`C-Tianyu/NanoJev-Data@7afc5257c0f3ff0ba08512729888a51d94b40e7e`,
`unified/hard/{train,dev,calibration,test,ood}.jsonl`, and preserve episode/family
groups. Hard/soft versions share questions, so they are not independent test
sets. Counts per variant are 10,898 train, 1,715 dev, 1,709 calibration, 2,496 test
and 1,942 OOD rows before the upstream invalid-target quarantine. Full gameplay
and frozen single-decision prediction measure different outcomes; label them
separately.

## Novelty and practical conclusions

An unrestricted "first quantization of System One models" claim is already
contradicted by Laya's
[ONNX exporter](https://github.com/NandhaKishorM/laya/blob/6d942c92081fbc139e736bbd9ac0023223c29b7f/scripts/export_onnx.py),
which implements per-channel INT8 quantization and records a small decision-flip
comparison. Its "weight-only" description should not replace inspection of the
generated ONNX operations: [ONNX Runtime's dynamic quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#dynamic-quantization) also quantizes
activations at runtime for relevant integer MatMul operations. S1Q can fairly
describe a decision-aware low-bit study across these open implementations, with
novelty tied to its explicit objective, method and validated findings. A
first-INT4/multi-model claim still requires a broader literature/project search.

Start with W4 weight-only backbone experiments and RTN at identical granularity,
group size, clipping convention and protected modules. Treat W4A8 and W4A4 as
separate measured extensions. Embeddings, heads, normalization, softmax and
hybrid recurrence remain high precision unless an ablation justifies changing
them. Report quantized-parameter coverage: W4 on selected backbone linears is a
mixed-precision model, not an all-parameter four-bit model. Fake quantization
alone demonstrates numerical behavior; it does not establish storage reduction
or accelerated integer inference.

## S1Q adapter contract

`src/s1q/models.py` exposes `load_model` (alias `load_adapter`) for the five
requested model names. Adapters expose `.model`, `.backbone`, `.linear_modules()`,
`.metadata`, `.stats`, `.infer(record)` and `.probs(record, temperature=1.0)`.
Inference logits remain tensors on the inference device, enabling calibration
hooks and differentiable objectives when a caller enables gradients. Callers
should use `torch.inference_mode()` for ordinary evaluation. Each request uses
question insertion order; probability labels use candidate insertion order or
`[false,true]` / score-index order. Rejected requests raise an exception and
increment counters; no output distribution is invented.

The source adapter passed local Python syntax compilation. GPU loading and
native-parity checks are pending the experiment runner; no accuracy, quality,
memory-saving or speedup claim is supported by this audit alone.
