# Software Architecture

## 1. Purpose

The project is organised as a single Python package with one root entry point. The architecture separates:

- external aerodynamic calculations;
- porous-network and coupling calculations;
- workflow orchestration;
- external-software integrations;
- plotting and publication post-processing;
- generated outputs.

The aim is to keep the numerical implementation understandable while avoiding duplicated runners and fragile import paths.

## 2. Current repository structure

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

The root `xfoil/` directory stores external XFOIL files. The Python XFOIL adapter is stored separately in `porous_airfoil/integrations/xfoil.py`.

## 3. Dependency direction

```text
run.py
  │
  ▼
porous_airfoil.workflows
  ├──────────────► porous_airfoil.plotting.*
  ├──────────────► porous_airfoil.integrations.*
  └──────────────► porous_airfoil.porous
                            │
                            ▼
                    porous_airfoil.solver
```

The intended direction is from the application layer toward the numerical core.

Key rule:

```text
solver.py must not import workflows, plotting or external integrations.
```

This keeps the aerodynamic core reusable and prevents circular dependencies.

## 4. Entry points

### `run.py`

The normal project entry point:

```bash
python3 run.py
```

It delegates to:

```python
porous_airfoil.workflows.run_selected_workflow()
```

### `porous_airfoil/__main__.py`

Provides the equivalent package entry point:

```bash
python3 -m porous_airfoil
```

Files inside `porous_airfoil/plotting/` and `porous_airfoil/integrations/` are internal modules and should not normally be executed directly.

## 5. Core numerical layer

### `porous_airfoil/solver.py`

Owns the external aerodynamic implementation:

- NACA four-digit airfoil generation;
- boundary ordering and panel geometry;
- surface interpolation and surface-point mapping;
- source and vortex influence matrices;
- source–vortex panel-method solution;
- aerodynamic coefficients;
- panel pressures and tangential velocities;
- external velocity-field evaluation;
- optional Numba acceleration for the field kernel.

This module should remain independent of publication formatting, CSV discovery, XFOIL process execution and ParaView file writing.

### `porous_airfoil/porous.py`

Owns the porous-network and coupled-solver implementation:

- pore, chamber, passage and model definitions;
- passage geometry and channel length;
- pore pressure sampling;
- internal-flow resistance calculation;
- flow-rate solution for each passage;
- conversion of passage flow to normal transpiration;
- independent passage-network assembly;
- aerodynamic/porous fixed-point coupling;
- fixed porous-model construction;
- reusable fixed-case and AoA-sweep calculations;
- result tables used by workflows and plotting modules.

The active model identifiers are retained for compatibility with existing output folders and CSV files.

## 6. Configuration and workflow layer

### `porous_airfoil/config.py`

Contains user-editable study settings and repository-relative paths.

Important path values include:

```text
PROJECT_ROOT
OUTPUT_ROOT
XFOIL_FOLDER
XFOIL_OUTPUT_ROOT
XFOIL_EXE_PATH
```

The current bundled XFOIL directory is:

```text
PROJECT_ROOT/xfoil/xfoil/
```

The current executable is:

```text
PROJECT_ROOT/xfoil/xfoil/xfoil.exe
```

Environment variables may override selected settings without editing the source file.

### `porous_airfoil/workflows.py`

Coordinates complete application workflows without containing the panel-method equations or figure-specific drawing algorithms.

Supported run modes:

```text
fixed
aoa_sweep
aero_plots
geometry
contours
streamlines
```

Responsibilities include:

- reading and validating `RUN_MODE`;
- validating `SELECTED_MODEL`;
- building shared base problems;
- selecting porous models;
- dispatching fixed-AoA calculations;
- dispatching AoA sweeps;
- dispatching plotting workflows;
- creating and coordinating output locations.

## 7. External integration layer

### `porous_airfoil/integrations/xfoil.py`

Adapter between the Python pipeline and the external XFOIL executable.

Responsibilities:

- launch XFOIL with `subprocess`;
- use short temporary working directories;
- send XFOIL command scripts;
- disable graphics for automated runs;
- write debug logs;
- copy generated Cp and polar files into the output tree;
- parse Cp and polar data into Python and pandas structures.

The external executable and source archive are not Python modules and remain under:

```text
xfoil/xfoil/
```

### `porous_airfoil/integrations/paraview.py`

Adapter for ParaView-compatible VTK output.

Responsibilities:

- airfoil surface VTP export;
- solid reference surface export;
- porous network VTP export;
- structured flow-field VTS export;
- fixed-case PVD collections;
- AoA-sweep PVD collections.

This module writes output files but does not change the aerodynamic or porous solution.

## 8. Plotting and post-processing layer

### `porous_airfoil/plotting/aerodynamic.py`

Responsible for aerodynamic result figures:

- fixed-case pressure-coefficient plots;
- porous/solid/XFOIL comparison plots;
- standard pressure and velocity field comparisons;
- AoA-sweep CL, CD and CM plots;
- journal CL and CM figures for selected diameters;
- sweep CSV discovery and normalisation.

### `porous_airfoil/plotting/contours.py`

Responsible for final contour figures:

- porous and solid field evaluation;
- porous-minus-solid velocity difference;
- porous-minus-solid pressure difference;
- porous pressure relative to a reference pressure;
- pressure-coefficient difference;
- common symmetric colour ranges;
- portrait multi-model publication figures;
- contour summary CSV files.

This module may solve the selected contour cases because the required fields depend on the aerodynamic solution.

### `porous_airfoil/plotting/geometry.py`

Responsible for porous-model documentation figures:

- individual porous-airfoil layouts;
- combined model comparison figure;
- channel centre lines and pore mouths;
- exact pore and channel coordinate CSV files.

The geometry workflow builds the airfoil and porous-model definitions but does not run the coupled aerodynamic solution.

### `porous_airfoil/plotting/streamlines.py`

Responsible for streamline visualisation:

- external velocity-field interpolation;
- continuous streamline integration with SciPy;
- forward, backward and bidirectional integration;
- far-field seed generation;
- exact pore-mouth seed generation;
- approximate internal-channel velocity corridors based on solved passage flow;
- direction arrows;
- pore and seed-location CSV files.

The internal streamline corridor is a post-processing visualisation. It is not a fully resolved internal CFD model.

### `porous_airfoil/plotting/common.py`

Contains shared non-physical utilities:

- safe directory creation;
- Windows long-path handling;
- common figure saving;
- common CSV writing where applicable;
- shared formatting helpers.

Shared utilities belong here only when they do not encode aerodynamic or porous physics.

## 9. Output layer

All standard generated data is written beneath:

```text
outputs/
```

Current main groups:

```text
outputs/
├── _xfoil/
├── aoa_sweep/
├── final_difference_contours_journal/
├── journal_aero_graphs_three_diameters/
├── results_discussion_porous_models/
└── streamline_results/
```

The `outputs/` directory contains generated files and should not be imported by source modules.

## 10. Execution flow by workflow

### Fixed case

```text
run.py
  → workflows.run_fixed_cases()
  → solver builds solid baseline
  → porous model is built
  → coupled porous solution is calculated
  → aerodynamic plots/CSV files are written
  → optional XFOIL and ParaView exports are written
```

### AoA sweep

```text
run.py
  → workflows.run_aoa_sweep()
  → each configured AoA and diameter is solved
  → sweep CSV files are written
  → standard sweep plots are written
  → optional XFOIL sweep comparison is added
```

### Journal aerodynamic plots

```text
run.py
  → workflows dispatches aero_plots
  → aerodynamic.py reads existing sweep CSV files
  → CL and CM publication figures are written
```

### Geometry

```text
run.py
  → workflows dispatches geometry
  → geometry.py builds model definitions
  → layout figures and coordinate CSV files are written
```

### Contours

```text
run.py
  → workflows dispatches contours
  → contours.py solves the selected cases
  → porous and solid fields are evaluated
  → difference fields and publication figures are written
```

### Streamlines

```text
run.py
  → workflows dispatches streamlines
  → streamlines.py solves the selected cases
  → velocity interpolators are constructed
  → external and internal visualisation streamlines are integrated
  → figures and seed-location CSV files are written
```
