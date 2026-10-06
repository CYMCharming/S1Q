"""Repository entry point for matched optimization experiments.

The package owns the implementation so the same workflow is available through
`s1q optimize` after installation. Historical method keys remain supported.
"""
from s1q.optimization import BASES, NEW, main

if __name__ == "__main__":
    main(default_methods=",".join(BASES + NEW))
