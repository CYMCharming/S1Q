# Intern-Decision and StartLux-Decision adapters

The model registry adds the released 0.8B, 2B and 4B checkpoints from
[Intern-Decision](https://huggingface.co/internlm/Intern-Decision-4B) and
[StartLux-Decision](https://huggingface.co/startlux-models/StartLux-Decision-4B).
Each size uses its own immutable Hub revision. Registered support, a native
interface check and a completed quantization benchmark are separate statuses.
Do not infer an accuracy result from the presence of a model configuration.

## Native decision contract

Intern uses its publisher's original compiler, chat template and complete
masked assistant skeleton. Each question reads its candidate-symbol logits
immediately before the trained decision marker. Only language-decoder Linear
weights enter quantization; embeddings, normalization, the output head, vision
tower and projector remain native. This adapter evaluates text records only.

StartLux uses the publisher's original eager row compiler, causal forward and
float32 26-letter output readout. It bypasses the inference-only `no_grad`
decorator to allow S1Q's native-margin calibration gradients. Graph replay and
shared-prefix caches are disabled throughout the matched run. Hierarchical
wide-option inference and images are outside this contract; unsupported native
records are rejected rather than silently truncated or reinterpreted.

Intern's native loader retains the full released model, including protected
vision and output-head parameters. StartLux's native `images=False` loader
keeps the text decoder and 26 float32 output rows while removing the vision
tower and full vocabulary head. Its parameter-byte estimates therefore refer
to that loaded text runtime; they are not a compression ratio for the complete
released multimodal checkpoint. Neither adapter evaluates image inputs.

The adapters preserve option order and map boolean outputs to the existing
canonical `[false, true]` order. `infer` returns raw candidate logits, and
benchmark probabilities use T=1, consistent with the other S1Q adapters.
Publisher serving-temperature metadata is recorded separately; evaluation
labels are not used to refit it. The benchmark uses its established first-index
argmax convention. Intern's serving wrapper declares lexical tie-breaking;
the parity gate records any observed decision tie rather than silently changing
the older evaluation rule.

## Provenance and Python compatibility

A local checkpoint override requires a repo/revision-matched byte manifest
covering its original native Python files. The acquisition tool also checks
published file sizes and LFS SHA256 values. A self-supplied manifest alone does
not prove its authority. The direct Hub-loading route records an immutable
Hub request; it is not described as independently file-hash verified.

Intern's publisher requires Python 3.12+. On Python 3.11, a fixed four-expression
loader changes only the outer quote delimiter of reviewed PEP-701 f-strings in
memory. It refuses a different expression pattern, leaves original checkpoint
files untouched, and records publisher/executed source hashes and the exact
transform. On Python 3.12+ the shim is not applied. This is a disclosed S1Q
compatibility path, not a statement that the publisher supports Python 3.11.

## Verification and use

Run `scripts/check_symbol_model_parity.py` before a new size's benchmark. It
compares native candidate probabilities and public typed outputs at T=1,
including multi-question, boolean, score and reordered-choice records. It also
requires positive native-margin gradient energy in every selected Linear.
The check writes evidence; it does not fit quantization settings or establish
benchmark accuracy.

For a verified local checkpoint, the four component configurations plus RTN are:

```sh
s1q optimize --model intern-decision-0.8b \
  --checkpoint-dir /path/to/verified/Intern-Decision-0.8B \
  --checkpoint-manifest /path/to/verified/Intern-Decision-0.8B/checkpoint-manifest.json \
  --data-dir /path/to/prepared/data --output-dir /path/to/new/run \
  --methods rtn,s1q-joint,s1q-ac,s1q-margin,s1q \
  --bits 4 --activation-bits 4 --group-size 128 \
  --calibration-count 128 --development-count 256 --reservoir-size 128 \
  --seed 20261004 --dtype bf16
```

Use `startlux-decision-0.8b`, or the registered `2b`/`4b` names, with that size's
own verified checkpoint. The generic acquisition helper
`scripts/fetch_decision_extensions.py` accepts a frozen official metadata JSON;
the host-specific transfer scripts used in this project remain private.

Calibration and evaluation must remain disjoint. Additional cohorts enter only
evaluation, and every method within a model must share admitted records,
precision, Linear scope and hardware. AWQ/GPTQ/Spin baselines remain repository
adaptations. Floating QDQ does not establish an integer-kernel speedup or a
measured complete packed checkpoint. Small-budget smoke scores are separate
from full-budget results.

Intern weights are released under Apache-2.0 with their upstream notices.
StartLux weights are CC BY-NC 4.0; its inference code is Apache-2.0. These
adapters do not redistribute model weights or change their licenses.
