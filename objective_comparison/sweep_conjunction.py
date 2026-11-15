"""Tune the 504-way conjunction RNN using the shared validation sweep engine.

Requires sweep_factorized.py, factorized_vs_conjunction.py and
controlled_comparison.py in the same directory. No final test data are used.

Smoke test:
    python sweep_conjunction.py --quick --device cuda

Full sequential grid:
    python sweep_conjunction.py --device cuda --patience-evals 0

For the cluster, prefer run_conjunction_sweep_h100.slurm, which runs each
configuration in a separate, serialized array allocation.
"""

from sweep_factorized import main


if __name__ == "__main__":
    main(default_objective="conjunction")
