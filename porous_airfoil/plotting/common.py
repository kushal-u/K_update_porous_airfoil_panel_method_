"""Shared plotting, path, and formatting helpers.

This module centralises non-physical utilities that were previously duplicated
across the contour, streamline, geometry, and aerodynamic plotting scripts.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable, Mapping, Any

import matplotlib as mpl
import pandas as pd


def windows_long_path(path: str | Path) -> str:
    """Return a filesystem path that is safe for long Windows paths."""
    path = Path(path)
    if os.name != "nt":
        return str(path)

    absolute = str(path.resolve())
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute.lstrip("\\")
    return "\\\\?\\" + absolute if len(absolute) >= 240 else absolute


def safe_mkdir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    target = Path(path)
    os.makedirs(windows_long_path(target), exist_ok=True)
    return target


def save_figure_all_formats(
    fig: mpl.figure.Figure,
    output_stem: str | Path,
    formats: Iterable[str] = ("png", "pdf", "svg"),
    png_dpi: int = 1000,
    pad_inches: float = 0.04,
) -> list[Path]:
    """Save one figure in each requested format."""
    output_stem = Path(output_stem)
    safe_mkdir(output_stem.parent)
    saved: list[Path] = []

    for extension in formats:
        extension = extension.lstrip(".")
        path = output_stem.with_suffix(f".{extension}")
        kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "pad_inches": pad_inches,
        }
        if extension.lower() == "png":
            kwargs["dpi"] = int(png_dpi)
        fig.savefig(windows_long_path(path), **kwargs)
        saved.append(path)
        print(f"[Plot Saved] {path}")

    return saved


def write_csv(path: str | Path, rows: list[Mapping[str, Any]] | Mapping[str, Any]) -> Path:
    """Write dictionaries to CSV using long-path-safe file handling."""
    path = Path(path)
    safe_mkdir(path.parent)
    records = [rows] if isinstance(rows, Mapping) else rows
    with open(windows_long_path(path), "w", encoding="utf-8", newline="") as handle:
        pd.DataFrame(records).to_csv(handle, index=False)
    return path


def format_float_label(value: float, decimals: int = 2) -> str:
    """Return a folder-safe signed floating-point label."""
    sign = "p" if float(value) >= 0.0 else "m"
    magnitude = f"{abs(float(value)):.{int(decimals)}f}".replace(".", "p")
    return f"{sign}{magnitude}"


def diameter_label_mm(diameter_m: float, decimals: int = 2) -> str:
    """Return a folder-safe pore-diameter label in millimetres."""
    return f"{1000.0 * float(diameter_m):.{int(decimals)}f}mm".replace(".", "p")
