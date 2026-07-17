"""Lightweight XFOIL wrapper used by the porous-airfoil pipeline.

This version avoids common Windows/XFOIL issues by:
1. running XFOIL inside a short system temporary folder,
2. disabling XFOIL graphics,
3. giving XFOIL only simple filenames such as cp.txt and polar.txt,
4. sending QUIT twice so XFOIL fully exits,
5. writing a debug log for every XFOIL call.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd


# =============================================================================
# BASIC HELPERS
# =============================================================================
def _normalise_airfoil_name(airfoil_name: str) -> str:
    name = str(airfoil_name).strip().upper()

    if name.startswith("NACA"):
        name = name[4:].strip()

    if len(name) != 4 or not name.isdigit():
        raise ValueError("Only NACA 4-digit airfoils are supported by this wrapper.")

    return name


def _safe_case_name(prefix: str, value: float) -> str:
    sign = "p" if float(value) >= 0.0 else "m"
    magnitude = f"{abs(float(value)):.3f}".replace(".", "p")
    return f"{prefix}_{sign}{magnitude}"


def _short_text(text: str, max_chars: int = 5000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def _write_debug_log(
    debug_log_path: Path,
    script: str,
    completed: subprocess.CompletedProcess[str] | None = None,
    note: str = "",
) -> None:
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(debug_log_path, "w", encoding="utf-8", errors="ignore") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("XFOIL DEBUG LOG\n")
        handle.write("=" * 80 + "\n\n")

        if note:
            handle.write("[NOTE]\n")
            handle.write(note + "\n\n")

        handle.write("[SCRIPT SENT TO XFOIL]\n")
        handle.write(script)
        handle.write("\n\n")

        if completed is not None:
            handle.write("[RETURN CODE]\n")
            handle.write(str(completed.returncode))
            handle.write("\n\n")

            handle.write("[STDOUT]\n")
            handle.write(completed.stdout or "")
            handle.write("\n\n")

            handle.write("[STDERR]\n")
            handle.write(completed.stderr or "")
            handle.write("\n")


def _run_xfoil_script(
    xfoil_exe_path: str | Path,
    script_lines: Iterable[str],
    working_dir: Path,
    timeout: float,
    debug_log_path: Path,
    note: str = "",
) -> subprocess.CompletedProcess[str]:
    exe = Path(xfoil_exe_path).expanduser()

    if not exe.exists():
        raise FileNotFoundError(f"XFOIL executable not found: {exe}")

    working_dir.mkdir(parents=True, exist_ok=True)

    script = "\n".join(str(line) for line in script_lines) + "\n"

    try:
        completed = subprocess.run(
            [str(exe)],
            input=script,
            cwd=str(working_dir),
            text=True,
            capture_output=True,
            timeout=float(timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _write_debug_log(
            debug_log_path=debug_log_path,
            script=script,
            completed=None,
            note=(
                f"XFOIL timed out after {timeout} seconds.\n"
                f"Working directory: {working_dir}"
            ),
        )
        raise RuntimeError(
            f"XFOIL timed out after {timeout} seconds. "
            f"Debug log: {debug_log_path}"
        ) from exc

    _write_debug_log(
        debug_log_path=debug_log_path,
        script=script,
        completed=completed,
        note=f"Working directory: {working_dir}",
    )

    return completed


def _xfoil_startup_lines(naca: str) -> list[str]:
    """
    Standard XFOIL startup.

    PLOP/G disables graphics. This is important on Windows because XFOIL can
    otherwise wait forever when trying to open a plot window.
    """
    return [
        "PLOP",
        "G",
        "",
        f"NACA {naca}",
        "PANE",
    ]


# =============================================================================
# XFOIL CP RUN
# =============================================================================
def run_xfoil_cp(
    xfoil_exe_path: str | Path,
    airfoil_name: str,
    aoa_deg: float,
    output_dir: str | Path,
    mach: float = 0.0,
    timeout: float = 120.0,
    reynolds: float | None = None,
    allow_inviscid_fallback: bool = True,
) -> Path:
    """Run XFOIL at one AoA and save a Cp distribution file."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    naca = _normalise_airfoil_name(airfoil_name)

    cp_path = output_dir / (
        f"xfoil_cp_NACA{naca}_{_safe_case_name('alpha', aoa_deg)}.txt"
    )

    debug_log_path = output_dir / (
        f"xfoil_debug_cp_NACA{naca}_{_safe_case_name('alpha', aoa_deg)}.log"
    )

    with tempfile.TemporaryDirectory(prefix="xfoil_cp_") as tmp:
        tmp_dir = Path(tmp)
        tmp_cp_name = "cp.txt"
        tmp_cp = tmp_dir / tmp_cp_name

        if tmp_cp.exists():
            tmp_cp.unlink()

        viscous_requested = reynolds is not None and float(reynolds) > 0.0

        script_lines: list[str] = _xfoil_startup_lines(naca)

        script_lines.extend(
            [
                "OPER",
                f"MACH {float(mach):.8g}",
            ]
        )

        if viscous_requested:
            script_lines.extend(
                [
                    f"VISC {float(reynolds):.8g}",
                    "ITER 300",
                    "INIT",
                ]
            )

        # IMPORTANT:
        # CPWR must receive the filename on the same line.
        # The blank line after CPWR exits the OPER menu.
        # QUIT then exits the main XFOIL menu.
        script_lines.extend(
            [
                f"ALFA {float(aoa_deg):.8g}",
                f"CPWR {tmp_cp_name}",
                "",
                "QUIT",
            ]
        )

        completed = _run_xfoil_script(
            xfoil_exe_path=xfoil_exe_path,
            script_lines=script_lines,
            working_dir=tmp_dir,
            timeout=timeout,
            debug_log_path=debug_log_path,
            note=(
                "Primary Cp attempt. "
                + ("Viscous mode enabled." if viscous_requested else "Inviscid mode.")
            ),
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "XFOIL Cp run failed.\n"
                f"Debug log: {debug_log_path}\n\n"
                f"stdout:\n{_short_text(completed.stdout)}\n\n"
                f"stderr:\n{_short_text(completed.stderr)}"
            )

        if not tmp_cp.exists() or tmp_cp.stat().st_size == 0:
            raise RuntimeError(
                "XFOIL Cp file was not created.\n"
                "XFOIL reached CPWR but did not write cp.txt.\n"
                f"Debug log: {debug_log_path}"
            )

        shutil.move(str(tmp_cp), str(cp_path))

    return cp_path


# =============================================================================
# XFOIL POLAR SWEEP
# =============================================================================
def run_xfoil_polar_sweep(
    xfoil_exe_path: str | Path,
    airfoil_name: str,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    output_dir: str | Path,
    mach: float = 0.0,
    timeout: float = 120.0,
    reynolds: float | None = None,
) -> Path:
    """Run an XFOIL AoA sweep and save a polar file."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    naca = _normalise_airfoil_name(airfoil_name)

    polar_path = output_dir / (
        f"xfoil_polar_NACA{naca}_"
        f"{_safe_case_name('a0', alpha_start)}_"
        f"{_safe_case_name('a1', alpha_end)}_"
        f"step_{str(float(alpha_step)).replace('.', 'p')}.txt"
    )

    debug_log_path = output_dir / (
        f"xfoil_debug_polar_NACA{naca}_"
        f"{_safe_case_name('a0', alpha_start)}_"
        f"{_safe_case_name('a1', alpha_end)}.log"
    )

    with tempfile.TemporaryDirectory(prefix="xfoil_polar_") as tmp:
        tmp_dir = Path(tmp)
        tmp_polar_name = "polar.txt"
        tmp_polar = tmp_dir / tmp_polar_name

        if tmp_polar.exists():
            tmp_polar.unlink()

        script_lines: list[str] = _xfoil_startup_lines(naca)

        script_lines.extend(
            [
                "OPER",
                f"MACH {float(mach):.8g}",
            ]
        )

        if reynolds is not None and float(reynolds) > 0.0:
            script_lines.extend(
                [
                    f"VISC {float(reynolds):.8g}",
                    "ITER 300",
                    "INIT",
                ]
            )

        script_lines.extend(
            [
                "PACC",
                tmp_polar_name,
                "",
                f"ASEQ {float(alpha_start):.8g} {float(alpha_end):.8g} {float(alpha_step):.8g}",
                "PACC",
                "",
                "QUIT",
            ]
        )

        completed = _run_xfoil_script(
            xfoil_exe_path=xfoil_exe_path,
            script_lines=script_lines,
            working_dir=tmp_dir,
            timeout=timeout,
            debug_log_path=debug_log_path,
            note="AoA polar sweep attempt.",
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "XFOIL polar sweep failed.\n"
                f"Debug log: {debug_log_path}\n\n"
                f"stdout:\n{_short_text(completed.stdout)}\n\n"
                f"stderr:\n{_short_text(completed.stderr)}"
            )

        if not tmp_polar.exists() or tmp_polar.stat().st_size == 0:
            raise RuntimeError(
                "XFOIL polar file was not created.\n"
                "This usually means XFOIL did not converge for the requested sweep, "
                "or XFOIL stopped at a prompt.\n"
                f"Debug log: {debug_log_path}"
            )

        shutil.move(str(tmp_polar), str(polar_path))

    return polar_path


# =============================================================================
# LOAD CP FILE
# =============================================================================
def load_xfoil_cp(cp_path: str | Path) -> dict[str, np.ndarray]:
    """Load an XFOIL Cp file and split points into upper/lower branches."""
    cp_path = Path(cp_path)

    rows: list[tuple[float, float, float]] = []

    with open(cp_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()

            if len(parts) < 3:
                continue

            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
                cp_val = float(parts[2])
            except ValueError:
                continue

            rows.append((x_val, y_val, cp_val))

    if not rows:
        raise ValueError(f"No numeric Cp rows found in XFOIL Cp file: {cp_path}")

    data = np.asarray(rows, dtype=float)

    x = data[:, 0]
    y = data[:, 1]
    cp = data[:, 2]

    upper_mask = y >= 0.0
    lower_mask = ~upper_mask

    def sort_by_x(
        values_x: np.ndarray,
        values_cp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(values_x)
        return values_x[order], values_cp[order]

    xu, cpu = sort_by_x(x[upper_mask], cp[upper_mask])
    xl, cpl = sort_by_x(x[lower_mask], cp[lower_mask])

    return {
        "x": x,
        "y": y,
        "cp": cp,
        "x_upper": xu,
        "cp_upper": cpu,
        "x_lower": xl,
        "cp_lower": cpl,
    }


# =============================================================================
# LOAD POLAR FILE
# =============================================================================
def load_xfoil_polar_dataframe(polar_path: str | Path) -> pd.DataFrame:
    """Load an XFOIL polar text file into a dataframe with standard columns."""
    polar_path = Path(polar_path)

    rows: list[list[float]] = []
    numeric_line = re.compile(r"^\s*[-+]?\d")

    with open(polar_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not numeric_line.match(line):
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            values: list[float] = []

            for item in parts[:7]:
                try:
                    values.append(float(item))
                except ValueError:
                    values = []
                    break

            if values:
                rows.append(values)

    if not rows:
        raise ValueError(f"No numeric polar rows found in XFOIL file: {polar_path}")

    max_cols = max(len(row) for row in rows)

    padded_rows = [
        row + [np.nan] * (max_cols - len(row))
        for row in rows
    ]

    names = [
        "alpha_deg",
        "CL",
        "CD",
        "CDp",
        "CM",
        "Top_Xtr",
        "Bot_Xtr",
    ][:max_cols]

    return pd.DataFrame(padded_rows, columns=names)