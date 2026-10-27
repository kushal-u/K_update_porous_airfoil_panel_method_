# Porous Airfoil Panel Method

A Python pipeline for analysing porous airfoils using a two-dimensional source–vortex panel method coupled with independent internal rectangular channels.

The pipeline calculates aerodynamic coefficients, surface pressure, internal channel flow and surface transpiration. It also generates aerodynamic graphs, geometry figures, pressure and velocity contours, and streamline visualisations for reports and publications.

## Porous-airfoil models

Three configurations are available:

| Model | Configuration | Channels |
| --- | --- | --- |
| Horizontal | Four independent chordwise channels | 4 |
| Vertical | Four independent lower-to-upper channels | 4 |
| Combined horizontal and vertical | Both layouts, with no hydraulic connections at internal intersections | 8 |

A solid-airfoil panel solution provides the reference at the same angle of attack and external Reynolds number. Optional XFOIL results can be included for comparison.

## Method overview

The external flow is represented by constant-strength source panels and a global vortex strength, with a Kutta condition at the trailing edge. Channel flows impose normal velocities over the surface openings, modifying the external pressure field that drives the internal flow.

The internal model uses rectangular channels per unit span. For channel height `H` and signed flow rate per unit span `q_prime`:

- Mean channel velocity: `U = q_prime / H`.
- Hydraulic diameter: `D_h = 2H`.
- Laminar flow follows the parallel-wall Poiseuille relation.
- Turbulent friction follows the Blasius correlation, with a smooth blend between the laminar and turbulent regimes.

The coupled system is nonlinear. A reduced formulation eliminates the panel unknowns from the iterative channel-flow solve. Damped Newton iterations, a least-squares fallback and continuation strategies are used to seek a consistent aerodynamic and hydraulic solution. The channel energy closure and solver controls are defined in `config.py`.

The external solver is inviscid. Reported drag and lift-to-drag ratios are based on pressure forces; they do not include skin-friction drag or a boundary-layer model.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/kushal-u/porous-airfoil-panel-method.git
cd porous-airfoil-panel-method
```

Create and activate a Python virtual environment on Linux or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

XFOIL is optional and must be supplied separately as an executable compatible with the environment used to run the project.

## Running the pipeline

Run from the repository root with the virtual environment active:

```bash
python3 run.py
```

Enabled stages run in the following order:

| Stage | Configuration switch | Supplied default |
| --- | --- | --- |
| Fixed-angle cases | `RUN_FIXED_CASES` | `False` |
| Angle-of-attack sweep | `RUN_AOA_SWEEP` | `True` |
| Aerodynamic plots | `RUN_AERODYNAMIC_PLOTS` | `True` |
| Porous-model geometry plots | `RUN_GEOMETRY` | `True` |
| Pressure and velocity contours | `RUN_CONTOURS` | `True` |
| Streamline plots | `RUN_STREAMLINES` | `True` |

`ENABLE_PLOTS` is the master switch for figure generation. Setting it to `False` disables plotting while retaining enabled numerical calculations and CSV outputs.

Results are written to `outputs/` by default. Change `OUTPUT_ROOT` in `config.py` to use another location.

## Configuration

Edit `porous_airfoil/config.py` before starting a run. It contains the geometry, flow conditions, channel dimensions, solver controls, stage switches and export settings.

The supplied numerical settings include:

```python
AIRFOIL_NAME = "0018"
CHORD = 1.0
N_PANELS = 1000
REYNOLDS_EXTERNAL = 1.0e6
AOA_DEG = 4.0

PORE_HEIGHTS_M = (0.0080, 0.0090, 0.0100)

AOA_SWEEP_START_DEG = -5.0
AOA_SWEEP_END_DEG = 15.0
AOA_SWEEP_STEP_DEG = 1.0
```

Lengths are in metres and angles are in degrees. `PORE_HEIGHTS_M` defines rectangular channel heights, so the values above correspond to 8, 9 and 10 mm. Legacy settings containing `diameter` refer to channel height in the current formulation.

Fixed-case heights, sweep heights and aerodynamic plotting selections inherit the shared height list unless their overrides are set. Geometry, contour and streamline stages use `CHANNEL_HEIGHT` and `AOA_DEG` unless visualisation overrides are supplied.

Select a model using `SELECTED_MODEL`:

| Value | Selection |
| --- | --- |
| `"all"` | All three configurations |
| `"model_1_4_chordwise"` | Horizontal channels |
| `"model_2_9_perpendicular"` | Vertical channels |
| `"model_3_combined_independent"` | Combined independent channels |

Additional controls include:

| Purpose | Settings |
| --- | --- |
| Detailed fixed-case data | `EXPORT_SURFACE_DATA`, `EXPORT_PASSAGE_DATA` |
| Fixed-case figures | `MAKE_CP_PLOTS`, `MAKE_NETWORK_PLOTS` |
| Sweep figures | `MAKE_AOA_SWEEP_PLOTS` |
| ParaView exports | `EXPORT_PARAVIEW`, `EXPORT_AOA_SWEEP_PARAVIEW` |
| Performance reporting | `PRINT_AERODYNAMIC_METRICS`, `SAVE_AERODYNAMIC_METRICS`, `PRINT_METRICS_PER_CASE` |
| Timing | `PRINT_ELAPSED_TIME`, `SAVE_TIMING_REPORT` |
| Handling stage errors | `CONTINUE_AFTER_STAGE_ERROR` |

Restart the program after changing configuration values.

## Project structure

| File or directory | Purpose |
| --- | --- |
| `run.py` | Main pipeline entry point |
| `porous_airfoil/config.py` | Simulation settings and shared data containers |
| `porous_airfoil/solver.py` | Airfoil generation, panel geometry and source–vortex solver |
| `porous_airfoil/porous_hydraulics.py` | Channel pressure–flow relations, surface openings and passage networks |
| `porous_airfoil/porous_coupling.py` | Reduced coupled problem, analytical Jacobian and nonlinear solver |
| `porous_airfoil/porous_models.py` | Model builders, XFOIL helpers and result/export utilities |
| `porous_airfoil/porous.py` | Continuation, fixed-case runs, sweeps and compatibility imports |
| `porous_airfoil/reporting.py` | Aerodynamic comparisons, timing and report generation |
| `porous_airfoil/workflows.py` | Pipeline orchestration |
| `porous_airfoil/plotting/` | Aerodynamic, geometry, contour and streamline plotting |
| `porous_airfoil/integrations/` | XFOIL and ParaView integrations |
| `outputs/` | Generated results |
| `xfoil/` | Optional XFOIL executable location |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Package metadata and build configuration |

When updating from the original single-file implementation, replace `porous.py` and add `porous_hydraulics.py`, `porous_coupling.py` and `porous_models.py` alongside it. Ensure that `reporting.py` is also present. Existing workflow imports through `porous.py` remain supported.

## Main outputs

Depending on the enabled stages, the pipeline produces:

- Lift, pressure-drag and pitching-moment coefficient curves.
- Fixed-case and angle-of-attack sweep CSV files.
- Surface pressure and internal channel-flow data.
- Porous-airfoil geometry figures.
- Pressure and velocity contours, including differences from the solid reference.
- External and internal channel streamline visualisations.
- Optional XFOIL comparison data and ParaView exports.
- Aerodynamic change metrics and elapsed-time reports.

Key output files, relative to the default `outputs/` directory, include:

| Output | Contents |
| --- | --- |
| `all_models_summary.csv` | Combined fixed-case results, when enabled |
| `aoa_sweep/selected_models_aoa_sweep_summary.csv` | Combined sweep results for the selected models and heights |
| `pipeline_stage_status.csv` | Stage status, elapsed time and error messages |
| `reports/run_timing.csv` | Timing records for pipeline and nested calls |
| `reports/latest_run_timing.json` | Latest top-level timing record |

Figures are exported in PNG, PDF and SVG where supported and configured. Per-model folders contain additional summaries, data and figures.

Aerodynamic changes are evaluated against the solid panel solution at the same operating point. Relative changes are suppressed when the reference coefficient is too small, and pressure-based lift-to-drag ratios are only reported when the pressure-drag threshold is satisfied.

Invalid or unconverged cases are retained in the summaries with their status and `NaN` porous performance values. Check `coupling_converged` and the diagnostic columns before interpreting results. Nested timing records overlap and should not be added to the total pipeline time.

## Optional XFOIL comparisons

Configure XFOIL in `porous_airfoil/config.py` using `USE_XFOIL`, `XFOIL_EXECUTABLE` and `XFOIL_FOLDER`. The code can also discover the executable on the system path.

Set `USE_XFOIL = False` to disable comparisons. If an executable cannot be found, the pipeline skips XFOIL comparisons and continues with the panel-method calculations. XFOIL files are stored under `outputs/_xfoil/` by default.

## Troubleshooting

If the program reports:

```text
ModuleNotFoundError: No module named 'porous_airfoil.reporting'
```

ensure that the reporting module is saved as `porous_airfoil/reporting.py`, alongside `workflows.py` and `porous.py`.

For other missing package modules, check that all split files are present with their exact filenames. Run the pipeline from the repository root using `python3 run.py`.

If a stage fails and `CONTINUE_AFTER_STAGE_ERROR` is enabled, later stages can still run. Inspect `pipeline_stage_status.csv` to distinguish successful stages from failed or skipped ones.

## License

See [LICENSE](LICENSE) for the project licence.
