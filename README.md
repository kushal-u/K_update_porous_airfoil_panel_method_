# Porous Airfoil Panel Method

A Python pipeline for analysing a porous airfoil using a source–vortex panel
method coupled with independent internal rectangular channels.

The project calculates aerodynamic coefficients, surface pressure, internal
channel flow and the resulting surface transpiration. It also generates
publication-ready aerodynamic graphs, geometry plots, contour plots and
streamline visualisations.

## Porous-airfoil models

The pipeline includes three porous configurations:

1. **Horizontal model** – four independent chordwise channels.
2. **Vertical model** – four independent lower-to-upper channels.
3. **Horizontal and Veritcal model** – the horizontal and vertical channels combined while
   remaining hydraulically independent.

A solid airfoil solution is calculated as the reference case. Optional XFOIL
results can also be included for comparison.

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/kushal-u/porous-airfoil-panel-method.git
cd porous-airfoil-panel-method
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Running the pipeline

Run the complete pipeline with:

```bash
python3 run.py
```

This runs the following stages in order:

1. Angle-of-attack sweep
2. Aerodynamic plots
3. Porous-model geometry plots
4. Pressure and velocity contours
5. Streamline plots

Results are written to the `outputs/` directory.

## Configuration

The main simulation settings are defined in:

```text
porous_airfoil/config.py
```

This file contains the airfoil geometry, flow conditions, panel count, channel
dimensions, angle-of-attack range, selected models and output options.

## Project structure

```text
porous-airfoil-panel-method/
├── run.py                         # Main entry point
├── porous_airfoil/
│   ├── config.py                  # Simulation settings
│   ├── solver.py                  # Source–vortex panel method
│   ├── porous.py                  # Internal flow and coupled solution
│   ├── workflows.py               # Analysis pipeline
│   ├── plotting/                  # Graph and field plotting
│   └── integrations/              # XFOIL and ParaView support
├── outputs/                       # Generated results
├── xfoil/                         # Optional XFOIL executable
├── requirements.txt
└── pyproject.toml
```

## Main outputs

The pipeline produces:

- lift and moment coefficient curves
- AoA-sweep CSV files
- porous-airfoil geometry figures
- pressure and velocity difference contours
- external and channel streamline plots
- optional XFOIL comparison data
- optional ParaView export files

Figures are exported in formats suitable for reports and journal publications,
including PNG, PDF and SVG where configured.

## License

See [LICENSE](LICENSE) for the project licence.
