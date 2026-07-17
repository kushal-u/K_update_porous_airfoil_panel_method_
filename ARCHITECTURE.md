# Project Architecture

## Overview

The project is organised as one Python package with a single entry point:

```bash
python3 run.py
```

`run.py` reads the selected workflow from `porous_airfoil/config.py` and runs the required calculation or plotting module.

## Project structure

```text
porous-airfoil-panel-method/
├── run.py
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── pyproject.toml
├── outputs/
├── xfoil/
│   └── xfoil/
│       └── xfoil.exe
└── porous_airfoil/
    ├── config.py
    ├── solver.py
    ├── porous.py
    ├── workflows.py
    ├── integrations/
    │   ├── xfoil.py
    │   └── paraview.py
    └── plotting/
        ├── common.py
        ├── aerodynamic.py
        ├── geometry.py
        ├── contours.py
        └── streamlines.py
```

## Main modules

### `config.py`

Contains the main user settings, including the airfoil, Reynolds number, angle of attack, pore diameter, selected model, plotting resolution and workflow mode.

### `solver.py`

Contains the external aerodynamic panel-method solver, airfoil geometry and velocity-field calculations.

### `porous.py`

Contains the porous-channel models, internal-flow calculations and coupling with the panel method.

### `workflows.py`

Controls the complete pipeline and connects the numerical solver, plotting modules, XFOIL and ParaView exports.

### `integrations/`

* `xfoil.py` runs XFOIL and reads its results.
* `paraview.py` exports results for ParaView.

### `plotting/`

* `aerodynamic.py` creates aerodynamic coefficient plots.
* `geometry.py` creates porous-airfoil geometry figures.
* `contours.py` creates pressure, velocity and difference contours.
* `streamlines.py` creates streamline figures.
* `common.py` contains shared plotting and file-saving functions.

## Workflow

```text
run.py
  ↓
workflows.py
  ↓
solver.py + porous.py
  ↓
plotting/ or integrations/
  ↓
outputs/
```

All generated figures, CSV files, XFOIL results and ParaView files are saved under:

```text
outputs/
```

The numerical solver is kept separate from plotting and external-software integrations so the code remains easier to understand and maintain.
