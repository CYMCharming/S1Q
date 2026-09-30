# Runtime compatibility and reproduction

Use Python 3.12 in a fresh environment for the five-model quick start. The S1Q
core package permits Python 3.10+, while the pinned Kev project declares
Python >=3.12,<3.14. Installing the `models` extra targets the shared inference
dependencies; it is not an installation of Kev's training or serving stack.

The pinned [Kev inference requirements](https://github.com/jaredpalmer/kev/blob/0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91/space/requirements.txt)
require Transformers >=5.17,<6, PEFT >=0.21, Accelerate >=1.15, and Pydantic >=2.9.
S1Q now declares these dependencies explicitly. Its `models` extra also follows
Kev's declared PyTorch >=2.6,<2.9 range. Transformers 5.17's package metadata
requires Hugging Face Hub >=1.5,<2 and Safetensors >=0.8; these floors are explicit
as well. [Transformers package metadata](https://pypi.org/pypi/transformers/5.17.0/json)

The full [Kev project metadata](https://github.com/jaredpalmer/kev/blob/0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91/pyproject.toml)
also specifies NumPy >=2.5.3 and scikit-learn >=1.9.1 for its broader project.
S1Q imports the pinned native inference sources directly and does not import
Kev's training/data modules. It therefore keeps its core NumPy floor at 1.24
and does not require scikit-learn for this inference workflow. Laya's pinned
project has a lower PyTorch/Transformers floor; the combined quick start uses
the stricter Kev inference requirements.

## Fresh reference installation

The direct versions in [.github/constraints.txt](../.github/constraints.txt)
define the Python 3.12 CPU CI reference profile. Use a PyTorch wheel index
appropriate for the target hardware; the CPU profile is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c .github/constraints.txt torch \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c .github/constraints.txt -e '.[models,test]'
python -m pip check
python scripts/setup_sources.py
python -m pytest tests -q
```

These are direct dependency pins, not a complete transitive lock file. CI logs
`pip freeze` and checks imports of the pinned Kev, Laya and NanoJev sources
before running the tests. Source schema and tiny-forward tests require no model
weights; full GPU model evaluations are separate recorded experiments. Install
upstream sources through `setup_sources.py` rather than an unpinned PyPI Laya or
Kev package.

The reference profile was checked in a newly created, isolated Python 3.12.3
environment on 2026-10-01, with system site packages disabled. All direct pins
matched the constraint file, PyTorch reported `2.7.1+cpu` with no CUDA runtime,
`pip check` passed, and the mandatory native imports passed. The latest source
snapshot passed **129 tests and 28 subtests** in 7.95 seconds. This check reused
the verified pinned upstream source archives; it did not reload model weights
or change either GPU experiment environment.

## Recorded experimental environments

Published runs preserve their actual package versions in `environment.json`.
Those records take precedence when reproducing a particular result. A new
installation meeting the supported floors may produce different logits or
timings from an older experimental environment.

| Recorded host | GPU | PyTorch | Transformers | PEFT | NumPy | Datasets |
|---|---|---|---|---|---|---|
| GPU29 | A100 80GB PCIe | 2.7.1+cu118 | 5.16.1 | 0.20.0 | 2.5.2 | 5.0.1 |
| GPU24 | A800 80GB PCIe | 2.9.1+cu126 | 5.17.0 | 0.21.0 | 1.26.4 | 2.14.6 |

The GPU29 row is recorded in
[Kev-0.8B validation v2](../results/kev-0.8b-validation-v2/environment.json) and
[Kev-4B validation v2](../results/kev-4b-validation-v2/environment.json).
The GPU24 row is recorded in
[Laya validation v2](../results/laya-validation-v2/environment.json).
Both recorded environments use Hugging Face Hub 1.29.0. Some older run records
omit NumPy; missing fields should not be inferred from another run.

GPU29's Transformers/PEFT versions are below the current pinned Kev declarations;
its NumPy version is below Kev's broader project floor. Native inference worked
in that recorded experimental environment. GPU24's PyTorch is above Kev's
declared upper bound, while its Laya workload used Laya's native inference path.
Its Datasets version is also below the shared S1Q `models` extra floor. These
observed runs do not establish general compatibility outside the declared
ranges. The dependency audit did not upgrade these running environments.

To replay their exact environments, preserve the recorded installed packages
and install only S1Q itself with `python -m pip install -e . --no-deps`.
Running `pip install -e '.[models,test]'` in those environments can change their
package versions; create a new environment for the supported reference profile.

## Bit-width scope

The main study compares fixed W4 backbone configurations. It does not claim
global bit-width optimality. W8 is available as a supplementary fallback, with
a matched W8 RTN baseline. Any fallback policy should be fixed on development
data before a new confirmation test, and all W4 outcomes should remain visible,
including models for which W4 degrades decision or probability quality.
