# Contributing

Contributions should include a reproducible model/source revision, a data recipe,
an explicit quantized layer scope and a fair baseline at the same precision policy.
Use an untouched external cohort after inspecting pilot tests. Report negative
results and distinguish source-family holdout from unknown pretraining overlap.

Run `python -m pytest tests -q` with the pinned upstream checkouts installed by
`scripts/setup_sources.py`. Tests use tiny synthetic tensors and native schema
builders; no production weights are needed. GPU results should record library
versions, hardware, dataset hashes and per-question paired predictions.

Do not describe fake activation quantization or dequantized floating inference as
native INT4 execution. Any acceleration contribution must measure real end-to-end
latency, memory and output parity with the actual kernel in use.
