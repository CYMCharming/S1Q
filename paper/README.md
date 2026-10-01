# S1Q research manuscript

**S1Q: Low-Bit Quantization for System One Decision Models**  
Yuanming Chen (CYMCharming), October 1, 2026

This is a complete English research draft, not a conference acceptance or an arXiv submission. It reports completed experiments from release v0.1.0, with explicit exploratory-cohort and fixed-profile external-evaluation boundaries.

- [Read the 11-page PDF](s1q.pdf).
- [Edit the standalone LaTeX source](s1q.tex).
- [Inspect table-input hashes and document provenance](provenance.json).
- [Read the underlying experimental report](../docs/results.md).

The draft contains an abstract, related work, quantizer equations, a conditional categorical-Fisher probe proposition and proof, an algorithm, evaluation definitions, five-model results, paired confidence intervals, transfer and external cohorts, full development-search tables, probability metrics, storage and CUDA memory measurements, limitations, source pins, and runtime notes. Its organization follows established quantization papers such as GPTQ, AWQ, SmoothQuant, and OmniQuant; it does not reproduce their prose or claim to be an official implementation of those methods.

The central claim is a reproducible native decision-model quantization framework and empirical study. Results are mixed: most main-cohort differences against matched RTN have intervals containing zero, transfer includes losses, raw NLL worsens on four of five main cohorts, and every external improvement interval includes zero. Fisher weighting is established prior art and is selected for only one model. Prior quantized Kev/Laya artifacts also preclude a broad first-quantization claim. Official strong baselines, multi-seed stability studies, larger independent evaluation cohorts, and efficient integer kernels remain work for stronger claims.

## Compile

The source is self-contained: references are inline and diagrams use TikZ/PGFPlots. No model weights or result files are required to compile it. With a LaTeX distribution providing the packages declared in the source, run from this directory:

```sh
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error s1q.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error s1q.tex
```

The exported PDF was compiled twice and all 11 rendered pages were visually checked. The provenance file binds the source and exported PDF to the recorded table inputs. PDF byte hashes can differ across recompilations because of build metadata; input and source hashes identify this draft snapshot.

## Cite the draft

The manuscript has no assigned DOI, arXiv identifier, or publication venue. Cite it as a research draft and pin the repository commit; cite the software release separately when reproducing experiments.

```bibtex
@misc{chen2026s1qmanuscript,
  author = {Chen, Yuanming},
  title = {S1Q: Low-Bit Quantization for System One Decision Models},
  year = {2026},
  month = oct,
  note = {Research manuscript draft; experimental snapshot v0.1.0},
  howpublished = {\url{https://github.com/CYMCharming/S1Q/tree/main/paper}}
}
```
