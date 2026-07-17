"""Module entry point for ``python -m porous_airfoil``."""

from __future__ import annotations

from .workflows import run_selected_workflow


def main() -> None:
    run_selected_workflow()


if __name__ == "__main__":
    main()
