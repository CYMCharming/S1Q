# S1Q: current method and evaluation contract

This document describes the current **S1Q**, whose frozen October 2026 run key was `s1q-mac`. That key is retained as a reproduction alias, not a separate public method name. Earlier S1Q/Fisher and S1Q2 recipes are historical controls; their method description is preserved in [legacy-method-v0.1.md](legacy-method-v0.1.md).

## Scope

A System One decision model maps a state and typed question to a finite decision distribution without an autoregressive answer trace. Candidate cardinality varies by question. S1Q uses native Kev, NanoJev and Laya adapters, preserves candidate insertion order, and keeps native decision heads. It quantizes selected backbone `torch.nn.Linear` weights; embeddings, norms, biases and non-Linear recurrent state arithmetic remain native. Kev adapters are merged before quantization. All matched controls share that boundary.

The evaluated implementation uses symmetric groupwise W4/W3 codes with group size 128, and dynamic per-token symmetric A4 QDQ. A8 and native-activation paths are supported but must be reported separately from the completed A4 batch. QDQ reconstructs values into floating point; the implementation does not supply integer activation GEMMs.

## Decision-sensitive calibration

For teacher logits z of one question, choose detached indices a and b of its largest and second-largest logits. Define

\[
m = z_a-z_b,\qquad \ell = m/(\operatorname{stopgrad}|m|+1).
\]

For each selected layer's native output token y_t, backpropagate each question separately and accumulate

\[
e_t = \sum_q\|\partial \ell_q/\partial y_t\|_2^2.
\]

The input token row and gradient come from the **same native forward call**. Separate backward passes avoid cancellations between questions. Normalize energies to mean one per request and layer; shrink by 50% toward uniform and bound the resulting importance in [0.5, 4], restoring mean one under those bounds. All-zero gradients use uniform importance. Calibration gold labels and targets are stripped before the teacher forward.

Use these importances once in a weighted-priority reservoir of 128 token rows. Do not multiply the reconstruction loss by the importance again: priority sampling already changes the distribution. Shuffle the saved rows with fixed seed 20261004 and split 64/64 for correction fitting and candidate selection. The actual sizes follow the available rows, with a minimum of four required. These are disjoint **token** halves from the same calibration requests, not independent request-level validation data.

This is an isotropic **margin-Jacobian token proxy**. It discards gradient direction and cross-token terms and is not the categorical Fisher information matrix. Request-wise normalization also cancels a common margin-denominator factor for a single-question request: the algorithm does not globally prioritize low-confidence requests. The proxy motivates prioritizing candidate comparisons, but does not prove that full-model decision flips will decrease.

**Close precedence:** [GuidedQuant](https://proceedings.mlr.press/v267/kim25d.html) incorporates end-loss gradient information into quantization objectives while retaining dependencies among weights within output channels. [RSQ](https://openreview.net/pdf?id=kBezrKXHVS) prioritizes important tokens, scales token features and computes GPTQ statistics from them, with attention concentration as its adopted importance signal. S1Q does not claim the first gradient-guided or token-aware quantizer. Its concrete distinction is a detached native top-two candidate-margin signal, aligned per-question token energies, bounded request/layer normalization and a priority reservoir applied once before joint W/A selection and compensation. That narrower adaptation must be validated empirically rather than inferred from the existence of those steps.

## Joint weight/activation reconstruction

For native input rows X, weight W and positive channel scale s, write B=W diag(s). The equivalent native computation satisfies

\[
XW^\top=(X\operatorname{diag}(s)^{-1})B^\top.
\]

Let Z=Q_A(X diag(s)^{-1}) be the **actual activation QDQ values**. Each candidate is evaluated with quantized reconstructed weights, not an activation-only or weight-only surrogate:

\[
L=\|Z_{\rm select}\widehat B^\top-X_{\rm select}W^\top\|_F^2
  /\max(\|X_{\rm select}W^\top\|_F^2,\epsilon).
\]

The S1Q scale families are root-mean-square and mean-absolute fit-token magnitudes, each raised to alpha in {0, .25, .5, .75, .9, 1}. Scales are normalized and bounded by the shared baseline helper. The duplicate mean-absolute alpha=0 candidate is omitted. Groupwise clipping ratios are {1, .95}. With correction strengths {0, 1}, this gives 44 candidates per layer under quantized activations. The final choice minimizes the selection-half loss. Local candidate improvement does not guarantee end-to-end accuracy improvement.

## Bounded activation compensation

On fit rows, the activation-only residual is

\[
E=X_{\rm fit}W^\top-Z_{\rm fit}B^\top.
\]

The correction solves the established ridge objective

\[
\min_D\ \|Z_{\rm fit}D^\top-E\|_F^2+\lambda\|D\|_F^2,
\quad
D=E^\top(Z_{\rm fit}Z_{\rm fit}^\top+\lambda I)^{-1}Z_{\rm fit},
\]

with lambda = .1 times the mean diagonal of ZZ^T. The implementation uses a Cholesky solve in calibration-row space and reports any regularization escalation. Bound ||D||_F to at most .1||B||_F, then quantize B+gamma D with gamma in {0, 1}. The correction folds into the quantized target; there is no extra inference residual branch. Native-activation and zero-strength controls are exact transformed-weight no-ops.

**Attribution:** [ERQ (ICML 2024)](https://proceedings.mlr.press/v235/zhong24a.html) already uses closed-form ridge regression to reduce activation quantization error before weight quantization; its [expanded journal version](https://arxiv.org/abs/2407.06794) develops this line further. S1Q does not claim that solver as new. [GPTAQ](https://arxiv.org/abs/2504.02692) matches full-precision outputs while handling asymmetric error accumulated from prior quantized layers. This S1Q implementation collects native teacher inputs and does **not** replay quantized preceding layers; it is not an implementation of GPTAQ.

The proposed adaptation is the aligned, bounded candidate-margin sampling integrated with joint W/A candidate selection and compensation for native typed-decision backbones. Equivalent scaling/clipping, gradient sensitivity and ridge fitting all have prior art. The value of this integration is an empirical question tested by matched controls; current data do not support global novelty or universal state-of-the-art claims.

## Factorial ablations and baseline extensions

The fixed four-way comparison is:

| Control | Margin-priority token reservoir | Compensation |
|---|---|---|
| Joint (`s1q-joint`) | No | No |
| AC (`s1q-ac`) | No | Yes |
| Margin (`s1q-margin`) | Yes | No |
| **S1Q** (`s1q`, frozen alias `s1q-mac`) | Yes | Yes |

All four use actual W/A local reconstruction and the same scale/clipping search. The newer joint objective is already a substantial change from earlier S1Q, so its gain must not be credited solely to margin sensitivity or compensation. Gains from each added mechanism require the corresponding paired comparison.

AWQ-AC/MAC use an adapted activation-aware scale search plus this W/A objective and correction; they differ from the original control in more than one operation. GPTQ-AC/MAC use a block-diagonal Hessian derived from actual quantized fit inputs. SpinQuant extensions use limited block rotations with separate Hadamard/no-Hadamard paths. These are project adaptations, **not official algorithm reproductions**. The rotation artifacts have separate representation limitations. No universal plug-in improvement has been established; several GPTQ and Hadamard-Spin extensions degrade accuracy.

Optional `-repair` controls fit bounded row gains with teacher KL and margin loss, then fold gains into group scales if their same-subset fitting loss improves. This is additional optimization, not the selected S1Q and not training-free PTQ. Same-subset acceptance is not held-out acceptance.

## Evaluation and execution limits

The runner freezes requested methods/settings before native evaluation, then fixes native-admitted rows across all methods. Calibration/evaluation request and source-group overlap is rejected. Extra named evaluation cohorts never enter quantization calibration. No development accuracy or gold label selects the per-layer candidates. Model/data/code hashes, failures, raw logits and native retention scope are recorded locally.

Report accuracy with source-cluster paired intervals, NLL, Brier, ECE, native flips and probability drift. Coverage-at-risk measured with evaluation labels is an oracle descriptive statistic, not a deployment threshold guarantee. NanoJev hard labels denote recorded reference-policy action compatibility, not game reward or optimal actions.

The October batch is exploratory after prior development inspection; additional WANLI/MMLU-Pro cohorts were frozen before evaluation but the main-method naming decision followed those results. Multi-seed replication, stronger official baselines, additional independent data and deployment kernels remain open work. Representative gains coexist with negative results. Preserve both.

Packed selected linear payloads are available through the session export API and require the original native checkpoint. Packing is storage compression; floating QDQ accuracy results do not demonstrate INT4 throughput or complete-model memory reduction. Historical v0.1 packed-resource measurements cannot be relabeled as current S1Q measurements.
