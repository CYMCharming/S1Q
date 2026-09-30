# Upstream licenses and S1Q modification notice

This directory accompanies the S1Q release's modified backbone linear-weight payloads. The files below are byte-for-byte copies of the pinned upstream license files, including their original copyright notices. [`manifest.json`](manifest.json) records the exact source URL, commit, byte count, SHA256, and collection transport for every copy. Git files were read from their committed blobs, avoiding checkout newline conversion; Hugging Face files were downloaded through HTTPS with normal certificate verification.

| File | Pinned origin | Applies to |
| --- | --- | --- |
| [Kev-source-LICENSE.txt](Kev-source-LICENSE.txt) | `jaredpalmer/kev@0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91` | Apache-2.0 Kev implementation/adapter attribution |
| [Laya-source-LICENSE.txt](Laya-source-LICENSE.txt) | `NandhaKishorM/laya@6d942c92081fbc139e736bbd9ac0023223c29b7f` | Apache-2.0 Laya implementation attribution |
| [Qwen3.5-0.8B-Base-LICENSE.txt](Qwen3.5-0.8B-Base-LICENSE.txt) | `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | Kev-0.8B base-weight derivative |
| [Qwen3.5-4B-Base-LICENSE.txt](Qwen3.5-4B-Base-LICENSE.txt) | `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` | Kev-4B base-weight derivative |
| [Qwen3.5-9B-Base-LICENSE.txt](Qwen3.5-9B-Base-LICENSE.txt) | `Qwen/Qwen3.5-9B-Base@68c46c4b3498877f3ef123c856ecfde50c39f404` | Kev-9B base-weight derivative |
| [NanoJev-source-LICENSE.txt](NanoJev-source-LICENSE.txt) | `TianyuCodings/NanoJev@76fdfc9ecdca45a9bcef17991a07d3041a87685a` | MIT source-code attribution; does not establish a fine-tuned weight license |

The original copyright lines are retained exactly: Kev's license states “Copyright 2026 Jared Palmer”; all three Qwen base licenses state “Copyright 2026 Alibaba Cloud”; NanoJev's source license states “Copyright (c) 2026 OpenJev contributors”. Laya's supplied license file ends at the Apache terms and has no separate named copyright line; it has been preserved without inventing one. No separate `NOTICE` file was found in the audited Kev or Laya source trees.

The released native model revisions and complete artifact hashes are in [`../configs/release-artifacts.json`](../configs/release-artifacts.json). Laya's [pinned model card](https://huggingface.co/convaiinnovations/laya/blob/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/README.md) independently declares `apache-2.0`. Kev's catalog entries identify both the native adapter checkpoint and its separate pinned Qwen base. These notices describe provenance and do not change upstream licensing.

## Notice of S1Q modifications

Copyright 2026 Yuanming Chen / CYMCharming — S1Q modifications only.

S1Q merged Kev's native LoRA adapters into the loaded backbones, then quantized selected backbone `nn.Linear` weight matrices into signed groupwise low-bit codes, group scales, and input-channel compensation scales. Laya's selected encoder linears were quantized in the same artifact format. The catalog records each evaluated profile, weight precision, group size, and activation simulation setting. These released payloads are modified weights; they are not the original upstream checkpoints.

Original native decision heads, biases, embeddings, normalization, convolution/recurrent components, tokenizers, and unselected weights are required from the pinned upstream checkpoints and are not bundled into these linear-only payloads. The runtime may reconstruct floating-point weights and simulate selected activation precision; no native integer-kernel performance is implied.

NanoJev has public code, recipes, and results, but no S1Q-redistributed weight asset. Its audited model card distinguishes source MIT licensing from the Qwen base's upstream terms and gives no separate explicit fine-tuned weight redistribution grant. The copied MIT source license is not an implied grant for those weights. See [`../docs/artifacts.md`](../docs/artifacts.md) for loading, attribution boundaries, and local reproduction.

Distribute this directory together with the S1Q [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE). Preserve applicable upstream license and attribution notices when redistributing derivatives.
