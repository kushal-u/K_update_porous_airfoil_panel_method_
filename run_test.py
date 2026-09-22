"""
Validation Test Script: Custom Integral Boundary-Layer Solver vs. XFOIL
======================================================================
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from porous_airfoil.config import (
    AOA_DEG,
    CHORD,
    MU_INF,
    N_CRIT,
    N_PANELS,
    P_INF,
    REYNOLDS_EXTERNAL,
    RHO_INF,
)
from porous_airfoil.porous import build_base_problem
from porous_airfoil.boundary_layer import solve_boundary_layer


def run_xfoil(
    airfoil_name: str = "0012",
    alpha: float = 4.0,
    reynolds: float = 1.0e6,
    n_crit: float = 9.0,
) -> dict | None:
    """Automate XFOIL execution using relative paths for WSL-Windows compatibility."""
    xfoil_bin = shutil.which("xfoil")
    if xfoil_bin is None:
        local_xfoil = Path("./xfoil/xfoil.exe").resolve()
        if local_xfoil.exists():
            xfoil_bin = str(local_xfoil)
        else:
            print("\n[WARNING] 'xfoil' executable not found.")
            return None

    # Use relative local paths so Windows xfoil.exe in WSL can write files
    bl_dump_file = "xfoil_bl.dat"
    polar_file = "xfoil_polar.dat"

    for f in [bl_dump_file, polar_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    code = airfoil_name.lower().replace("naca", "").replace(" ", "").strip()

    # Direct XFOIL batch command stream
    xfoil_input = f"""naca {code}
oper
visc {reynolds}
ncri {n_crit}
iter 200
pacc
{polar_file}

alfa {alpha}
dump {bl_dump_file}

quit
"""

    try:
        process = subprocess.run(
            [xfoil_bin],
            input=xfoil_input,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.SubprocessError as e:
        print(f"[ERROR] Failed to execute XFOIL: {e}")
        return None

    xfoil_res = {}

    # 1. Parse Boundary Layer Dump File
    if os.path.exists(bl_dump_file):
        try:
            data = np.loadtxt(bl_dump_file, skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            xfoil_res.update(
                {
                    "s": data[:, 0],
                    "x": data[:, 1],
                    "y": data[:, 2],
                    "Ue": data[:, 3],
                    "delta_star": data[:, 4],
                    "theta": data[:, 5],
                    "Cf": data[:, 6],
                    "Hk": data[:, 7],
                    "N": data[:, 8] if data.shape[1] > 8 else np.zeros(data.shape[0]),
                }
            )
        except Exception as e:
            print(f"[WARNING] Failed to parse BL dump file: {e}")

    # 2. Parse Polar Output File
    if os.path.exists(polar_file):
        try:
            with open(polar_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        try:
                            if abs(float(parts[0]) - alpha) < 1e-2:
                                xfoil_res["CL"] = float(parts[1])
                                xfoil_res["CD"] = float(parts[2])
                                xfoil_res["xtr_upper"] = float(parts[5])
                                xfoil_res["xtr_lower"] = float(parts[6])
                                break
                        except ValueError:
                            continue
        except Exception as e:
            print(f"[WARNING] Failed to parse polar file: {e}")

    # 3. Fallback: Parse CL/CD directly from stdout
    if "CL" not in xfoil_res or "CD" not in xfoil_res:
        for line in process.stdout.splitlines():
            clean_line = line.replace("=", " ")
            parts = clean_line.split()
            if "CL" in parts and "CD" in parts:
                for idx, p in enumerate(parts):
                    if p == "CL" and idx + 1 < len(parts):
                        xfoil_res["CL"] = float(parts[idx + 1])
                    if p == "CD" and idx + 1 < len(parts):
                        xfoil_res["CD"] = float(parts[idx + 1])
            if "Top" in parts and "Xtr" in parts and "Bot" in parts:
                for idx, p in enumerate(parts):
                    if p == "Top" and idx + 2 < len(parts) and parts[idx + 1] == "Xtr":
                        xfoil_res["xtr_upper"] = float(parts[idx + 2])
                    if p == "Bot" and idx + 2 < len(parts) and parts[idx + 1] == "Xtr":
                        xfoil_res["xtr_lower"] = float(parts[idx + 2])

    return xfoil_res if xfoil_res else None


def fmt_val(val: float, fmt: str = ".4f") -> str:
    """Format numerical values, cleanly identifying NaN as Laminar/TE."""
    if np.isnan(val):
        return "Laminar (1.0000)"
    return f"{val:{fmt}}"


def main() -> None:
    raw_airfoil_name = "naca0012"
    airfoil_code = raw_airfoil_name.lower().replace("naca", "").replace(" ", "").replace("-", "").strip()

    alpha = AOA_DEG
    Re = REYNOLDS_EXTERNAL

    print("=" * 65)
    print(f" RUNNING SOLID AIRFOIL TEST & VALIDATION (NACA {airfoil_code})")
    print(f" Parameters: Re = {Re:.1e}, Alpha = {alpha} deg, Ncrit = {N_CRIT}")
    print("=" * 65)

    # 1. Custom Solver Execution
    flow, geom, ref, aero_solver = build_base_problem(
        airfoil_code, CHORD, N_PANELS, alpha, Re, RHO_INF, MU_INF, P_INF
    )

    solid_aero = aero_solver.solve()
    bl_upper, bl_lower, Cd_profile = solve_boundary_layer(
        geom, solid_aero, flow, N_crit=N_CRIT
    )

    print("\n[Custom Solver Finished]")
    print(f"  Lift Coeff (CL)        : {solid_aero.CL:.4f}")
    print(f"  Profile Drag (CD)      : {Cd_profile:.6f}")
    print(f"  Upper Transition (xtr) : {fmt_val(bl_upper.xtr)}")
    print(f"  Lower Transition (xtr) : {fmt_val(bl_lower.xtr)}")

    # 2. XFOIL Reference Execution
    xfoil_res = run_xfoil(
        airfoil_name=airfoil_code,
        alpha=alpha,
        reynolds=Re,
        n_crit=N_CRIT,
    )

    # 3. Print Summary Table
    print("\n" + "=" * 65)
    print(f"{'Metric':<22} | {'Custom Solver':<18} | {'XFOIL':<18}")
    print("-" * 65)
    cl_xf = xfoil_res.get("CL", float("nan")) if xfoil_res else float("nan")
    cd_xf = xfoil_res.get("CD", float("nan")) if xfoil_res else float("nan")
    xtr_u_xf = xfoil_res.get("xtr_upper", float("nan")) if xfoil_res else float("nan")
    xtr_l_xf = xfoil_res.get("xtr_lower", float("nan")) if xfoil_res else float("nan")

    print(f"{'Lift Coeff (CL)':<22} | {solid_aero.CL:<18.4f} | {fmt_val(cl_xf, '.4f'):<18}")
    print(f"{'Profile Drag (CD)':<22} | {Cd_profile:<18.6f} | {fmt_val(cd_xf, '.6f'):<18}")
    print(f"{'Upper x_tr/c':<22} | {fmt_val(bl_upper.xtr):<18} | {fmt_val(xtr_u_xf):<18}")
    print(f"{'Lower x_tr/c':<22} | {fmt_val(bl_lower.xtr):<18} | {fmt_val(xtr_l_xf):<18}")
    print("=" * 65)

# 4. Plot Comparisons
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"Boundary Layer Validation: Custom Solver vs XFOIL (NACA {airfoil_code}, $\\alpha={alpha}^\\circ$, $Re={Re:.1e}$)",
        fontsize=13,
    )

    # --- UPDATED SUBPLOT 1: Surface Pressure Distribution ---
    axs[0, 0].plot(geom.XC, solid_aero.Cp, "b.-", label="Custom Panel Method", lw=1.2)
    if xfoil_res and "x" in xfoil_res:
        if "Cp" in xfoil_res:
            axs[0, 0].plot(xfoil_res["x"], xfoil_res["Cp"], "k--", label="XFOIL", alpha=0.7)
        elif "Ue" in xfoil_res:  # Fallback if your dictionary only stores edge velocity ratio
            axs[0, 0].plot(xfoil_res["x"], 1.0 - (xfoil_res["Ue"] ** 2), "k--", label="XFOIL", alpha=0.7)
            
    axs[0, 0].set_ylabel("$C_p$")
    axs[0, 0].invert_yaxis()
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].legend()
    axs[0, 0].set_title("Surface Pressure Distribution")
    # ---------------------------------------------------------

    axs[0, 1].plot(bl_upper.x, bl_upper.Cf, "r-", label="Custom Upper", lw=1.5)
# ... [Rest of your plotting code remains unchanged] ...

    axs[0, 1].plot(bl_upper.x, bl_upper.Cf, "r-", label="Custom Upper", lw=1.5)
    axs[0, 1].plot(bl_lower.x, bl_lower.Cf, "b-", label="Custom Lower", lw=1.5)
    if xfoil_res and "x" in xfoil_res:
        axs[0, 1].plot(xfoil_res["x"], xfoil_res["Cf"], "k--", label="XFOIL", alpha=0.7)
    axs[0, 1].set_ylabel("Skin Friction $C_f$")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)
    axs[0, 1].legend()
    axs[0, 1].set_title("Skin Friction Coefficient")

    axs[1, 0].plot(bl_upper.x, bl_upper.delta_star, "r-", label="Custom Upper", lw=1.5)
    axs[1, 0].plot(bl_lower.x, bl_lower.delta_star, "b-", label="Custom Lower", lw=1.5)
    if xfoil_res and "x" in xfoil_res:
        axs[1, 0].plot(xfoil_res["x"], xfoil_res["delta_star"], "k--", label="XFOIL", alpha=0.7)
    axs[1, 0].set_xlabel("$x / c$")
    axs[1, 0].set_ylabel("Displacement Thickness $\\delta^*$ [m]")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].legend()
    axs[1, 0].set_title("Displacement Thickness")

    axs[1, 1].plot(bl_upper.x, bl_upper.theta, "r-", label="Custom Upper", lw=1.5)
    axs[1, 1].plot(bl_lower.x, bl_lower.theta, "b-", label="Custom Lower", lw=1.5)
    if xfoil_res and "x" in xfoil_res:
        axs[1, 1].plot(xfoil_res["x"], xfoil_res["theta"], "k--", label="XFOIL", alpha=0.7)
    axs[1, 1].set_xlabel("$x / c$")
    axs[1, 1].set_ylabel("Momentum Thickness $\\theta$ [m]")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)
    axs[1, 1].legend()
    axs[1, 1].set_title("Momentum Thickness")

    plt.tight_layout()
    plot_file = "validation_comparison.png"
    plt.savefig(plot_file, dpi=300)
    print(f"\n[Plot Saved] Validation plot saved as '{plot_file}'")
    plt.show()


if __name__ == "__main__":
    main()