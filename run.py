"""Single entry point for the porous-airfoil software package.

Set ``RUN_MODE`` in ``porous_airfoil/config.py`` or set the environment
variable ``POROUS_RUN_MODE`` before running:

    python run.py
"""

from __future__ import annotations

from porous_airfoil.workflows import run_selected_workflow


if __name__ == "__main__":
    run_selected_workflow()
