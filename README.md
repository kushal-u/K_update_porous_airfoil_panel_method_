# Porous Airfoil Panel Method

A Python research pipeline for analysing porous NACA airfoils with a source–vortex panel method coupled to independent internal porous passages.

The project supports:

- fixed-angle porous-airfoil calculations;
- angle-of-attack sweeps;
- optional XFOIL comparison;
- pressure, velocity and pressure-coefficient contours;
- journal-quality lift and pitching-moment figures;
- porous-network geometry figures;
- external and approximate internal-channel streamline visualisation;
- optional ParaView export.

All workflows are launched from the project root through:

```bash
python3 run.py
```

Do not run the files inside `porous_airfoil/plotting/` directly. They are internal package modules used by `run.py`.

## Repository structure

```text
porous_airfoil_panel_method_restructured/
├── ARCHITECTURE.md
├── LICENSE
├── README.md
├── outputs/
├── porous_airfoil/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── porous.py
│   ├── solver.py
│   ├── workflows.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── paraview.py
│   │   └── xfoil.py
│   └── plotting/
│       ├── __init__.py
│       ├── aerodynamic.py
│       ├── common.py
│       ├── contours.py
│       ├── geometry.py
│       └── streamlines.py
├── pyproject.toml
├── requirements.txt
├── run.py
└── xfoil/
    └── xfoil/
        ├── Xfoil699src.zip
        ├── pplot.exe
        ├── pxplot.exe
        └── xfoil.exe
```

The bundled XFOIL executable is therefore located at:

```text
xfoil/xfoil/xfoil.exe
```

The Python wrapper that communicates with XFOIL is separate:

```text
porous_airfoil/integrations/xfoil.py
```

## Main modules

| Module | Responsibility |
|---|---|
| `porous_airfoil/config.py` | User-editable settings, run mode, model selection, flow conditions, numerical tolerances and paths |
| `porous_airfoil/solver.py` | NACA geometry generation and source–vortex panel-method solver |
| `porous_airfoil/porous.py` | Porous passages, internal resistance, transpiration mapping and coupled solution |
| `porous_airfoil/workflows.py` | Workflow selection and execution |
| `porous_airfoil/integrations/xfoil.py` | XFOIL execution and result parsing |
| `porous_airfoil/integrations/paraview.py` | VTK/PVD export for ParaView |
| `porous_airfoil/plotting/aerodynamic.py` | Fixed-case plots, AoA-sweep plots and journal CL/CM figures |
| `porous_airfoil/plotting/contours.py` | Final pressure, velocity and Cp difference contours |
| `porous_airfoil/plotting/geometry.py` | Porous-airfoil model geometry figures and coordinate CSV files |
| `porous_airfoil/plotting/streamlines.py` | Continuous external and approximate internal-channel streamlines |
| `porous_airfoil/plotting/common.py` | Shared plotting, path and file-saving utilities |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the software dependency structure.

## WSL / Ubuntu setup

Move to the project root:

```bash
cd "/mnt/c/Users/s358959/Downloads/porous_airfoil_panel_method_restructured/porous_airfoil_panel_method_restructured"
```

Install virtual-environment support if required:

```bash
sudo apt update
sudo apt install -y python3-venv python3-full
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For future terminal sessions, return to the project root and reactivate the environment:

```bash
source .venv/bin/activate
```

## Configuration

Normal user settings are stored in:

```text
porous_airfoil/config.py
```

Important settings include:

- `RUN_MODE`
- `SELECTED_MODEL`
- `AIRFOIL_NAME`
- `CHORD`
- `N_PANELS`
- `REYNOLDS_EXTERNAL`
- `AOA_DEG`
- `PORE_DIAMETER`
- `FIXED_AOA_PORE_DIAMETERS`
- `AOA_SWEEP_START_DEG`
- `AOA_SWEEP_END_DEG`
- `AOA_SWEEP_STEP_DEG`
- `AOA_SWEEP_PORE_DIAMETERS`
- coupling tolerances and relaxation;
- contour-grid resolution;
- XFOIL and ParaView switches.

The default results folder is:

```text
outputs/
```

All default paths are anchored to the project root rather than the terminal's current working directory.

## Available workflows

| Run mode | Purpose | Main output location |
|---|---|---|
| `fixed` | Fixed-AoA porous cases for the selected model and configured diameters | `outputs/` |
| `aoa_sweep` | AoA sweep for the selected model or all models | `outputs/aoa_sweep/` |
| `aero_plots` | Journal CL and CM figures generated from sweep CSV files | `outputs/journal_aero_graphs_three_diameters/` |
| `geometry` | Porous-model layout figures and exact channel-coordinate CSV files | `outputs/results_discussion_porous_models/` |
| `contours` | Journal velocity, pressure and Cp difference contours | `outputs/final_difference_contours_journal/` |
| `streamlines` | External and approximate internal-channel streamline figures | `outputs/streamline_results/` |

## Running one workflow

The easiest method is to select the run mode temporarily in the terminal.

```bash
POROUS_RUN_MODE=fixed python3 run.py
```

Other workflows:

```bash
POROUS_RUN_MODE=aoa_sweep python3 run.py
POROUS_RUN_MODE=aero_plots python3 run.py
POROUS_RUN_MODE=geometry python3 run.py
POROUS_RUN_MODE=contours python3 run.py
POROUS_RUN_MODE=streamlines python3 run.py
```

Alternatively, set `RUN_MODE` in `porous_airfoil/config.py` and then run:

```bash
python3 run.py
```

The equivalent package command is:

```bash
python3 -m porous_airfoil
```

## Running all plotting workflows

When new AoA-sweep data is required:

```bash
POROUS_RUN_MODE=aoa_sweep python3 run.py &&
POROUS_RUN_MODE=aero_plots python3 run.py &&
POROUS_RUN_MODE=geometry python3 run.py &&
POROUS_RUN_MODE=contours python3 run.py &&
POROUS_RUN_MODE=streamlines python3 run.py
```

When valid AoA-sweep CSV files already exist:

```bash
POROUS_RUN_MODE=aero_plots python3 run.py &&
POROUS_RUN_MODE=geometry python3 run.py &&
POROUS_RUN_MODE=contours python3 run.py &&
POROUS_RUN_MODE=streamlines python3 run.py
```

The `&&` operator starts the next workflow only after the previous command finishes successfully.

The contour and streamline workflows recompute the required flow fields; they are not simple replotting steps.

## XFOIL configuration

The bundled executable is expected at:

```text
xfoil/xfoil/xfoil.exe
```

Check whether the project can find it:

```bash
python3 -c "from porous_airfoil.config import XFOIL_EXE_PATH; print(XFOIL_EXE_PATH); print(XFOIL_EXE_PATH is not None and XFOIL_EXE_PATH.exists())"
```

Expected final line:

```text
True
```

A temporary executable override can be supplied with:

```bash
export XFOIL_EXE="$PWD/xfoil/xfoil/xfoil.exe"
```

Disable XFOIL for one run with:

```bash
USE_XFOIL=0 POROUS_RUN_MODE=fixed python3 run.py
```

XFOIL text files and debug logs are written under:

```text
outputs/_xfoil/
```

## Output structure

The main generated result groups are:

```text
outputs/
├── _xfoil/
├── aoa_sweep/
├── final_difference_contours_journal/
├── journal_aero_graphs_three_diameters/
├── results_discussion_porous_models/
└── streamline_results/
```

The contents of `outputs/` are generated data and figures, not source code.

## Model names

The current compatible internal identifiers are:

```text
all
model_1_4_chordwise
model_2_9_perpendicular
model_3_combined_independent
model_4_saved_case_1
```

These identifiers are retained to remain compatible with existing folders, CSV files and plotting logic. Publication labels are normally shown as Horizontal, Vertical and Isotropic or Horizontal + Vertical.

## Notes on streamline figures

The external streamlines are integrated from the panel-method velocity field.

The internal-channel streamline representation uses the solved passage flow rate and an approximate channel velocity corridor. It is intended for flow-direction visualisation and is not a fully resolved internal CFD solution.

## Licence

This project is released under the MIT License. See `LICENSE`.