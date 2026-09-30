# S1Q: method and evaluation contract

S1Q is an experimental post-training quantization framework for open System One decision models. It preserves each upstream model's native typed question format and decision head, and applies a small search over symmetric groupwise backbone weight quantization. The decision-aware variant weights this search with an estimate of the teacher's categorical Fisher information at intermediate linear outputs. This document describes the implemented estimator and its assumptions; it does not establish a new general quantization principle or claim a performance improvement before the corresponding experiments are complete.

## Scope and terminology

Here, **System One decision model** means a model that maps a state and typed question to a finite decision distribution, without requiring an autoregressive answer trace. The audited implementations support boolean, choice, and discrete score questions with a variable number of candidates. `Jev-like` is useful as a descriptive relationship to Jev, but is not a universally standardized architecture name. S1Q evaluates the concrete pinned implementations listed in [the model audit](model-audit.md), rather than treating all models in this group as architecturally interchangeable.

The supported quantization boundary is the backbone's `torch.nn.Linear` modules. Kev's adapters are merged before quantization. Native decision heads, token embeddings, normalization, convolution, and hybrid recurrent operations retain their loaded precision. Kev's Qwen3.5 hybrid backbone, Laya's ModernBERT encoder, and NanoJev's Qwen3 backbone require different native loaders and forward paths. The adapter passes native raw logits to evaluation and preserves candidate insertion order. Boolean logits are consistently interpreted as `[no, yes]`; NanoJev's native single boolean logit is represented exactly as `[0, z]`.

The current reference implementation concentrates on W4 and W8 weights with floating-point activations. An optional higher-precision protection budget makes a mixed-precision artifact, so its realized bit rate must be reported. An experiment with activation fake quantization must be labeled separately and specify which activations were quantized. A W4 weight result alone is not evidence for W4A4.

## Backbone input statistics

For a linear layer with weight matrix `W` of shape `[out_features, in_features]`, collect its unmodified teacher input activations `x` on the calibration split. Flatten all leading dimensions into token rows and compute

\[
m_j = \mathbb{E}[x_j^2],\qquad
h_j = \frac{m_j}{\max(\operatorname{mean}_k m_k, 10^{-30})}.
\]

`h` is normalized once per layer. Flattening includes any padded rows emitted by the native forward; the collector does not infer a token mask from arbitrary intermediate tensors. Report the observed row counts and calibration record counts, and use the same preprocessing and padding policy for all candidates. These statistics are inputs to a diagonal approximation, rather than a full input covariance matrix.

## Label-free categorical Fisher estimator

Let a record contain `Q` questions and let `z_q` be the raw teacher logits for question `q`. Questions may have different candidate counts. At an explicitly recorded temperature `T > 0`, define

\[
u_q = z_q/T,\qquad p_q = \operatorname{softmax}(u_q),\qquad
F_q = \operatorname{diag}(p_q) - p_qp_q^\top.
\]

`p_q` is detached from autograd. This is the categorical Fisher in logit coordinates for the teacher distribution, and also the Hessian of categorical cross-entropy with respect to those coordinates. It is not an estimate of a downstream task reward, and it does not use the gold label. For score questions, this treats the finite levels as categories; it does not encode distance between ordinal levels.

For each independent probe, sample `epsilon_q ~ N(0, I)` from a seeded CPU generator and set

\[
a_q = \sqrt{p_q}\odot\epsilon_q,\qquad
v_q = a_q-p_q\sum_k a_{qk}.
\]

Then `E[v_q] = 0`, `E[v_q v_q^T] = F_q`, and `sum_k v_qk = 0` up to floating-point error. The last identity makes a common additive shift of all candidate logits irrelevant, as required by a categorical distribution. The probe objective used for a record is

\[
\ell_\mathrm{probe} = \frac{1}{\sqrt Q}\sum_{q=1}^Q
\operatorname{stopgrad}(v_q)^\top u_q.
\]

The scalar is a device for computing a random gradient projection. It is not a training loss to minimize. With independent probes for each question, its gradient covariance is the average per-question Fisher propagated through the model Jacobians. The `1/sqrt(Q)` factor prevents records with more independent questions from automatically multiplying the expected gradient variance. Temperature affects both `p_q` and the `1/T` derivative through `u_q`; it must not be changed after choosing a configuration without reporting a new experiment.

For a selected linear output `y` of shape `[..., out_features]`, the collector accumulates

\[
\widehat f_i = \frac{1}{N_\mathrm{rows}}
\sum_{r,\,\mathrm{probe}}\left(\frac{\partial\ell_\mathrm{probe}}
{\partial y_{ri}}\right)^2.
\]

The denominator counts flattened output rows across successful backward probes. A record with more token rows therefore contributes more rows to this average. With a finite number of random projections, this estimate is noisy. It is a diagonal hidden-output Fisher proxy, not the full parameter Fisher or a full Hessian.

The quantization search normalizes the estimate to mean one and then floors the row importance:

\[
r_i = \max\left(\frac{\widehat f_i}
{\max(\operatorname{mean}_k\widehat f_k,10^{-30})},\,10^{-4}\right),
\]

with the implementation's conservative uniform convention for an all-zero finite estimate. There is no second normalization after the floor. This minimum prevents a sampled zero from making a weight row completely free in the local objective. The helper `DecisionGradientStatistics.row_weights` exposes optional shrinkage toward uniform weights for analysis, but the integrated quantization path consumes the raw mean-square estimate and applies its own normalization. A shrinkage experiment must specify its coefficient and actually pass the resulting weights through the search; calling the helper alone does not change the artifact.

### Autograd and failure policy

`DecisionFisherCollector` freezes model parameters during collection, so it does not allocate or update full weight gradients. Forward hooks enable gradients on selected linear outputs where needed, without detaching an existing graph. The collector temporarily enters evaluation mode and enables autograd for its own pass, then restores parameter `requires_grad` flags and module training flags. This still requires activation graphs and backward computation; it is more expensive than an activation-only forward pass.

The adapters must return differentiable native raw logits. A detached backend, a fused path without the required derivatives, an unobserved selected module, nonfinite logits or gradients, and an unsupported native input cause an explicit failure. There is no silent switch to activation-only scoring. A failed record discards all of its partially collected gradient statistics. Metadata records successful records, probes, failures, seed, temperature, question normalization, row reduction, and the no-fallback policy. Preserve this metadata with each artifact.

The implementation is in [`decision_stats.py`](../src/s1q/decision_stats.py), with covariance, common-shift invariance, graph preservation, state restoration, label independence, attachment, and failed-record rollback tests in [`test_decision_stats.py`](../tests/test_decision_stats.py).

## Groupwise weight quantization and local search

The scale search uses the input statistics to build an equivalent channel transformation. For exponent `alpha`, let

\[
s_j = (\max(\sqrt{h_j},10^{-8}))^\alpha.
\]

For positive observed channels, divide these values by `max(sqrt(min(s) * max(s)), 10^-12)`, then clamp to `[10^-4, 10^4]`. The minimum and maximum for normalization consider only observed channels. Unobserved channels receive `s_j = 1`. At `alpha = 0`, every channel receives scale one. Define

\[
W' = W\operatorname{diag}(s),\qquad x' = x\oslash s.
\]

This transformation leaves the floating-point linear map unchanged before rounding. Groups run along the input dimension independently for each output row; the default group size is 128, with a zero-padded last group when necessary. For bit width `b` and clipping candidate `c`, set

\[
q_{\max}=2^{b-1}-1,\qquad
d_g=\frac{c\max_{j\in g}|W'_{ij}|}{q_{\max}},\qquad
q_{ij}=\operatorname{clip}\left(\operatorname{round}(W'_{ij}/d_g),
-q_{\max},q_{\max}\right).
\]

An all-zero group uses scale one. There is no affine zero point. Thus W4 uses signed values `[-7, 7]` and W8 uses `[-127, 127]`; the unused most-negative code is intentional. Dequantize and undo the channel scale:

\[
\widehat W'_{ij}=d_gq_{ij},\qquad
\widehat W_{ij}=\widehat W'_{ij}/s_j.
\]

For each layer, the decision-aware local objective is

\[
E(W,\widehat W) = \sum_{i,j}r_i h_j
(\widehat W_{ij}-W_{ij})^2.
\]

The activation-only variant sets `r_i = 1`. RTN uses `alpha = 0` and `c = 1` without a calibration search. The default search grid is `alpha in {0, 0.25, 0.5, 0.75}` and `c in {0.9, 0.95, 1}`. Ties choose smaller `alpha`, then larger `c`, so an equal objective does not add an unnecessary transformation or clipping step. The search reconstructs weights in FP32. The reference execution path writes the transformed dequantized matrix `W_hat'` into the model's loaded weight dtype and installs an input hook that computes `x/s` in the activation dtype. This may introduce additional dtype rounding compared with the ideal FP32 effective-weight objective; the actual native forward is the evaluation authority.

For comparable layer protection scores, relative error divides the same weighted numerator by `sum_ij r_i h_j W_ij^2`. A zero-energy denominator returns zero. A higher-precision protection budget ranks the RTN relative errors and protects `ceil(fraction * eligible_layer_count)` layers, breaking tied scores by module name. This fraction counts layers, not weight elements, so realized parameter-weighted precision can differ across models. This is an additional mixed-precision component, and its benefit must be separated from the effect of channel scaling and decision weights. The optional `reservoir_relative_error` diagnostic evaluates unweighted linear-output MSE on sampled input rows; it is not the Fisher-weighted search score and does not select the current candidate.

The objective approximates propagated decision divergence by multiplying an output-gradient diagonal with an input second-moment diagonal. It drops dependencies between input channels, between output channels, between token positions, and between quantization errors in different layers. It also factorizes activation and gradient moments instead of retaining their joint moments. Consequently, a smaller local objective does not guarantee better task accuracy, probability calibration, or distribution transfer. The exact categorical Fisher covariance of the random projection does not remove these later approximations.

## Suggested collection order

Collect input statistics and decision-gradient statistics on the same frozen floating-point teacher and the same accepted calibration records, before mutating any weights. `DecisionFisherCollector.collect(record)` returns detached teacher logits after its backward probes. `attach_input_statistics(input_statistics, gradient_statistics, require_all=True)` creates a new mapping of input statistics with `output_mean_square` attached; it does not mutate the original activation-only mapping. A missing layer match raises by default. Quantize from the pristine teacher for every candidate artifact rather than collecting statistics from a partly quantized model.

Gold labels must be stripped before model input construction and are not used by either statistics collector. Label-based development metrics may choose among a finite, predeclared set of complete configurations on the development split. They must not redefine the local Fisher estimator or use test records for clipping, protection, temperature fitting, or selecting a seed.

For a native adapter already loaded in its declared precision, a minimal calibration pass is:

```python
from s1q.decision_stats import DecisionFisherCollector, attach_input_statistics
from s1q.quantization import CalibrationCollector, selected_linear_modules

modules = selected_linear_modules(adapter.backbone)
with CalibrationCollector(adapter.backbone, reservoir_size=32, seed=seed) as inputs:
    with DecisionFisherCollector(adapter, modules=modules, seed=seed,
                                 temperature=1.0, probes_per_record=2) as fisher:
        for record in calibration_records:
            fisher.collect(record)

input_statistics = inputs.statistics()
gradient_statistics = fisher.statistics()
decision_statistics = attach_input_statistics(input_statistics, gradient_statistics)
collection_metadata = fisher.metadata()
```

Use native admitted calibration records and preserve failures rather than catching an exception and silently changing the cohort. This example collects both moments on the same forward passes. The experiment runner may collect them in separate passes; record the actual accepted records and Fisher subset size.

## Calibration, development, and test roles

Every run should persist exact model/source revisions, dataset revisions and file checksums, accepted record identifiers, rejection reasons, question/candidate counts, calibration size, random seeds, precision, group size, search grid, protected modules, temperature policy, package versions, and collection metadata. Dataset provenance and split definitions are documented separately; generated conversions must point back to their upstream source.

- **Calibration:** gather unlabeled activation and teacher-gradient statistics. If post-quantization temperature fitting is used, fit it on a designated labeled calibration subset and state that additional use.
- **Development:** compare the fixed candidate configurations and choose one deterministic rule before final evaluation. Do not select a different configuration for every test set.
- **Test and transfer:** evaluate the frozen selection rule. Report the floating-point teacher, RTN, and each required ablation on a common set of inputs accepted by all compared models or explicitly report model-specific rejection cohorts.

Kev's published decision and transfer releases are useful for reproducing that model's upstream task evaluation; they do not by themselves establish evaluation independence from model development. NanoJev is a gameplay specialization: its native game release provides an appropriate specialization benchmark, and unrelated decision corpora should be labeled as transfer. Preserve game episode/family grouping and quarantine the upstream documented invalid rows. Converting another model's record format must not invent missing criteria, duplicate candidate descriptions, or truncate a record silently. Native validation and overflow rules can create different accepted cohorts, which must be reported.

Report question accuracy alongside proper probability metrics such as NLL and Brier score, decision-type breakdowns, and floating-point versus quantized disagreement. For discrete score questions, include an ordinal error metric in addition to categorical accuracy. State whether probabilities use raw logits, the unchanged upstream temperature, or a newly fitted temperature. A newly fitted calibration temperature must not be credited to weight quantization alone. Pair models on identical question identifiers for confidence intervals or paired tests, and distinguish record counts from question counts.

### Development disclosure

Early Laya test and transfer results were consulted before adding the decision-gradient variant. They motivated the next method revision and are exploratory evidence. Reusing those cohorts cannot provide a fresh confirmation of the revision even if its numeric hyperparameters are selected only on calibration/development data. Final confirmatory evidence requires a predeclared configuration evaluated once on an untouched cohort. The planned independent external JevBench original/hard cohort must have its conversion, duplicate handling, and selection rule frozen before evaluation. Upstream model training overlap may remain unknown even when a cohort is untouched by S1Q development; state that limitation.

[`evaluate_external.py`](../scripts/evaluate_external.py) consumes a completed run's selection, environment, exported artifact, and already fitted temperatures. Before inference it writes hashes for those inputs and the prepared external manifest/data to `freeze.json`. Native floating-point admission establishes the paired cohort; a failure after admission aborts evaluation. It loads the selected artifact through the original dense reconstruction and input-hook path, and evaluates RTN on exactly the exported layer mask and activation scope. RTN calibrated metrics are available only if a matching frozen temperature report exists in the completed run; the script never fits missing temperatures on the external cohort. It publishes exclusions, candidate-key order, paired raw logits, decision-type/tier/family metrics, separate authored soft-target distances, and cluster-bootstrap accuracy differences. Separate cross-model comparison must intersect the frozen admission identifiers before claiming a common cohort.

## Required ablations and baselines

The minimum useful comparison separates the parts of the adaptation:

| Variant | Weight quantizer | Input-weighted local search | Teacher decision Fisher | Protected layers |
| --- | --- | --- | --- | --- |
| Floating-point teacher | none | none | none | all loaded precision |
| RTN | symmetric groupwise | no | no | none |
| Activation-only local search | same | yes | no | none |
| Decision-aware local search | same | yes | yes | none |
| Protected decision-aware search | same | yes | yes | predeclared budget |

Add matched protection to RTN/activation-only variants if making a claim about decision weights under mixed precision. Vary Fisher probe count, calibration size, temperature, and protection fraction on development data. Report whether the extra backward cost is justified by the result. Head-only quantization and whole-model low-bit quantization are separate scope experiments, not implied by a backbone-only artifact. Official AWQ or GPTQ implementations are valuable additional baselines when compatible with the native architecture; S1Q's local search should not be relabeled as either official algorithm.

## Storage and execution claims

Reference fake quantization evaluates the numerically reconstructed floating-point weights. It is appropriate for measuring accuracy effects but does not reduce their in-memory tensor dtype or implement low-bit matrix multiplication. Optional A4/A8 uses per-token symmetric dynamic fake quantization of the selected linear's already compensated input, with range `[-7,7]` or `[-127,127]`. It leaves activations outside that hook, including protected layers, untouched. This is a simulation of the selected inputs and does not establish an entirely integer W4A4 model.

A packed artifact can reduce weight storage, while metadata, scales, protected modules, and untouched components still consume space. The research quantization-session artifact contains only selected quantized weights, scales, and scope metadata; it requires the identical base model for untouched weights, biases, tokenizer, and heads. It is a patch rather than a standalone full-model checkpoint. Report both eligible weight storage and the total deployment footprint with required base components, and verify packed reconstruction against the reference artifact.

There are no custom native low-bit kernels in this method. A packed reference loader may unpack to floating-point for execution. Neither packed file size nor nominal W4 precision proves lower peak inference memory, lower latency, or higher throughput. Those claims require a separately identified runtime, its actual kernels and precision, and end-to-end measurements with warm-up, synchronization, batch/question counts, and sequence lengths. Kev's hybrid operations and native decision heads must be included in such measurements.

## Relationship to prior work and permissible claims

S1Q borrows established quantization ingredients. [AWQ](https://arxiv.org/abs/2306.00978) motivates activation-guided channel transformations; this implementation's second-moment search is a compact adaptation, not the official AWQ algorithm. [GPTQ](https://arxiv.org/abs/2210.17323) uses approximate second-order weight reconstruction, whereas the present local objective retains only diagonals. [SmoothQuant](https://arxiv.org/abs/2211.10438) addresses weight/activation difficulty through equivalent scaling, but this project's weight-only experiments do not implement its W8A8 deployment path.

Fisher and end-loss guidance are also established. [FIT](https://arxiv.org/abs/2210.08502) uses Fisher information to measure quantization sensitivity. [GuidedQuant](https://arxiv.org/abs/2505.07004) incorporates end-loss gradients and retains within-output-channel weight dependencies. [C-PTQ](https://arxiv.org/abs/2607.21076) explicitly uses Fisher-weighted channel sensitivity to guide multimodal model quantization. Fisher-weighted error alone therefore cannot be claimed as S1Q's novel general method. The proposed contribution to assess is the adaptation to native variable-candidate typed decisions, its label-free categorical estimator and question normalization, hybrid/model-specific boundaries, and a reproducible cross-model evaluation. Whether this adaptation is sufficiently novel for a paper remains an empirical and literature-review question.

Laya already exposes an INT8 ONNX quantization path in its audited upstream source. That path uses ONNX Runtime dynamic quantization; [the runtime documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#dynamic-quantization) distinguishes dynamic activation quantization from static calibration. This existing implementation rules out an unqualified claim that S1Q is the first quantization of System One models. The narrower claim that a particular low-bit, cross-model benchmark is first also requires a fuller current literature and repository audit. Use the neutral title **S1Q: Low-Bit Quantization for System One Decision Models** until evidence supports a more specific claim.
