"""
paraview_export.py

ParaView / VTK export utilities for porous-airfoil panel-method results.

This module writes ParaView-compatible ASCII VTK XML files without requiring
the vtk Python package.

Main single-case exports:
    paraview_results.pvd
    paraview_airfoil_surface.vtp
    paraview_porous_network.vtp
    paraview_flow_field_porous.vts

Optional solid exports:
    paraview_solid_airfoil_surface.vtp
    paraview_flow_field_solid.vts

AoA sweep parent collections:
    paraview_aoa_sweep_all_parts.pvd
    paraview_aoa_sweep_surface.pvd
    paraview_aoa_sweep_solid_surface.pvd
    paraview_aoa_sweep_network.pvd
    paraview_aoa_sweep_porous_flow.pvd
    paraview_aoa_sweep_solid_flow.pvd

For AoA sweeps, ParaView treats the angle of attack as the timestep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# BASIC VTK WRITING HELPERS
# =============================================================================
def _vtk_float_ascii(values: Any) -> str:
    """
    Convert scalar/vector numeric arrays to VTK ASCII format.

    Supports:
        1D array: scalar data
        2D array: vector/component data
    """
    arr = np.asarray(values)

    if arr.ndim == 1:
        return " ".join(f"{float(v):.16e}" for v in arr)

    if arr.ndim == 2:
        return "\n".join(
            " ".join(f"{float(v):.16e}" for v in row)
            for row in arr
        )

    raise ValueError("VTK ASCII writer only supports 1D or 2D arrays.")


def _vtk_int_ascii(values: Any) -> str:
    arr = np.asarray(values, dtype=int).ravel()
    return " ".join(str(int(v)) for v in arr)


def _layout_id(layout_kind: str) -> int:
    """
    Numeric layout ID for ParaView coloring.

        1 = internal_chordwise
        2 = vertical
        3 = surface_pair
        9 = other/unknown
    """
    kind = str(layout_kind).lower()

    if kind in {"internal_chordwise", "internal_horizontal", "internal_chordwise_horizontal"}:
        return 1
    if kind in {"vertical", "perpendicular"}:
        return 2
    if kind == "surface_pair":
        return 3

    return 9


def _side_id(side: str) -> int:
    """
    Numeric side ID for ParaView coloring.

        upper = +1
        lower = -1
    """
    return 1 if str(side).lower() == "upper" else -1


def _relative_vtk_path(path: Path, output_dir: Path) -> str:
    """
    Return a ParaView-friendly relative path for .pvd collection entries.

    This is important for AoA sweeps because the .pvd file is usually stored
    in the parent AoA folder, while each VTP/VTS file is stored inside an
    alpha-specific subfolder.
    """
    path = Path(path)
    output_dir = Path(output_dir)

    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_data_array_float(
    f,
    name: str,
    values: Any,
    number_of_components: int | None = None,
    indent: str = "        ",
) -> None:
    if number_of_components is None:
        f.write(
            f'{indent}<DataArray type="Float64" Name="{name}" format="ascii">\n'
            f"{indent}  {_vtk_float_ascii(values)}\n"
            f"{indent}</DataArray>\n"
        )
    else:
        f.write(
            f'{indent}<DataArray type="Float64" Name="{name}" '
            f'NumberOfComponents="{number_of_components}" format="ascii">\n'
            f"{_vtk_float_ascii(values)}\n"
            f"{indent}</DataArray>\n"
        )


def _write_data_array_int(
    f,
    name: str,
    values: Any,
    indent: str = "        ",
) -> None:
    f.write(
        f'{indent}<DataArray type="Int32" Name="{name}" format="ascii">\n'
        f"{indent}  {_vtk_int_ascii(values)}\n"
        f"{indent}</DataArray>\n"
    )


# =============================================================================
# 1. POROUS AIRFOIL SURFACE EXPORT
# =============================================================================
def export_airfoil_surface_vtp(
    output_dir: Path,
    geom: Any,
    solid_result: Any,
    porous_result: Any,
) -> Path:
    """
    Export porous-airfoil surface panel data as VTP PolyData.

    Geometry:
        one VTK line cell per surface panel

    Main fields:
        porous_Cp
        solid_Cp
        porous_Vt_m_s
        solid_Vt_m_s
        normal_transpiration_m_s
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "paraview_airfoil_surface.vtp"

    aero = porous_result.aero_result

    n_points = len(geom.XB)
    n_lines = geom.num_pan

    points = np.column_stack(
        [
            geom.XB,
            geom.YB,
            np.zeros_like(geom.XB),
        ]
    )

    connectivity: list[int] = []
    offsets: list[int] = []

    for i in range(n_lines):
        connectivity.extend([i, i + 1])
        offsets.append(len(connectivity))

    side_id = np.zeros(geom.num_pan, dtype=int)
    side_id[np.asarray(geom.upper_panel_ids, dtype=int)] = 1
    side_id[np.asarray(geom.lower_panel_ids, dtype=int)] = -1

    cell_int_arrays = {
        "panel_id": np.arange(geom.num_pan, dtype=int),
        "surface_side_id": side_id,
    }

    cell_float_arrays = {
        "XC_m": geom.XC,
        "YC_m": geom.YC,
        "XC_over_c": geom.XC / geom.chord,
        "YC_over_c": geom.YC / geom.chord,
        "panel_length_m": geom.S,
        "porous_Cp": aero.Cp,
        "solid_Cp": solid_result.Cp,
        "porous_Vt_m_s": aero.Vt,
        "solid_Vt_m_s": solid_result.Vt,
        "normal_transpiration_m_s": porous_result.normal_transpiration,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfLines="{n_lines}">\n')

        f.write('      <CellData Scalars="porous_Cp">\n')
        for name, values in cell_int_arrays.items():
            _write_data_array_int(f, name, values)
        for name, values in cell_float_arrays.items():
            _write_data_array_float(f, name, values)
        f.write("      </CellData>\n")

        f.write("      <Points>\n")
        _write_data_array_float(
            f,
            name="Points",
            values=points,
            number_of_components=3,
            indent="        ",
        )
        f.write("      </Points>\n")

        f.write("      <Lines>\n")
        _write_data_array_int(f, "connectivity", connectivity)
        _write_data_array_int(f, "offsets", offsets)
        f.write("      </Lines>\n")

        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")

    print(f"[ParaView Saved] {save_path}")
    return save_path


# =============================================================================
# 2. SOLID AIRFOIL SURFACE EXPORT
# =============================================================================
def export_solid_airfoil_surface_vtp(
    output_dir: Path,
    geom: Any,
    solid_result: Any,
) -> Path:
    """
    Export solid-airfoil surface panel data as VTP PolyData.

    This is useful for:
        1. the single solid baseline reference
        2. the AoA sweep solid-airfoil surface result at every angle

    Geometry:
        one VTK line cell per surface panel

    Main fields:
        solid_Cp
        solid_Vt_m_s
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "paraview_solid_airfoil_surface.vtp"

    n_points = len(geom.XB)
    n_lines = geom.num_pan

    points = np.column_stack(
        [
            geom.XB,
            geom.YB,
            np.zeros_like(geom.XB),
        ]
    )

    connectivity: list[int] = []
    offsets: list[int] = []

    for i in range(n_lines):
        connectivity.extend([i, i + 1])
        offsets.append(len(connectivity))

    side_id = np.zeros(geom.num_pan, dtype=int)
    side_id[np.asarray(geom.upper_panel_ids, dtype=int)] = 1
    side_id[np.asarray(geom.lower_panel_ids, dtype=int)] = -1

    cell_int_arrays = {
        "panel_id": np.arange(geom.num_pan, dtype=int),
        "surface_side_id": side_id,
    }

    cell_float_arrays = {
        "XC_m": geom.XC,
        "YC_m": geom.YC,
        "XC_over_c": geom.XC / geom.chord,
        "YC_over_c": geom.YC / geom.chord,
        "panel_length_m": geom.S,
        "solid_Cp": solid_result.Cp,
        "solid_Vt_m_s": solid_result.Vt,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfLines="{n_lines}">\n')

        f.write('      <CellData Scalars="solid_Cp">\n')
        for name, values in cell_int_arrays.items():
            _write_data_array_int(f, name, values)
        for name, values in cell_float_arrays.items():
            _write_data_array_float(f, name, values)
        f.write("      </CellData>\n")

        f.write("      <Points>\n")
        _write_data_array_float(
            f,
            name="Points",
            values=points,
            number_of_components=3,
            indent="        ",
        )
        f.write("      </Points>\n")

        f.write("      <Lines>\n")
        _write_data_array_int(f, "connectivity", connectivity)
        _write_data_array_int(f, "offsets", offsets)
        f.write("      </Lines>\n")

        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")

    print(f"[ParaView Saved] {save_path}")
    return save_path


# =============================================================================
# 3. POROUS INTERNAL NETWORK EXPORT
# =============================================================================
def export_porous_network_vtp(
    output_dir: Path,
    network: Any,
    geom: Any,
    result: Any,
) -> Path:
    """
    Export porous internal passages as VTP PolyData.

    Geometry:
        one VTK line cell per internal passage

    Main fields:
        Q_m3_s
        abs_Q_m3_s
        dp_total_Pa
        signed_dp_p1_minus_p2_Pa
        reynolds_internal_equivalent
        Rs_Pa_s_per_m3
        passage_length_m
        diameter_m
        channel_direction_unit
        flow_direction_unit
        flow_Q_vector_m3_s
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "paraview_porous_network.vtp"

    points: list[list[float]] = []
    connectivity: list[int] = []
    offsets: list[int] = []

    point_pore_id: list[int] = []
    point_passage_id: list[int] = []
    point_side_id: list[int] = []
    point_x_frac: list[float] = []
    point_pressure: list[float] = []

    cell_passage_id: list[int] = []
    cell_layout_id: list[int] = []
    cell_pore1_side_id: list[int] = []
    cell_pore2_side_id: list[int] = []

    cell_x1_frac: list[float] = []
    cell_x2_frac: list[float] = []
    cell_Q: list[float] = []
    cell_abs_Q: list[float] = []
    cell_dp_total: list[float] = []
    cell_signed_dp: list[float] = []
    cell_p1: list[float] = []
    cell_p2: list[float] = []
    cell_Re: list[float] = []
    cell_Rs: list[float] = []
    cell_diameter: list[float] = []
    cell_length: list[float] = []

    cell_channel_unit: list[list[float]] = []
    cell_flow_unit: list[list[float]] = []
    cell_flow_q_vector: list[list[float]] = []

    for passage_id, (passage, state) in enumerate(
        zip(network.passages, result.passage_states),
        start=1,
    ):
        sp1 = passage.pore_surface_point(geom, passage.pore1)
        sp2 = passage.pore_surface_point(geom, passage.pore2)

        p0 = len(points)
        p1 = p0 + 1

        x1, y1 = float(sp1.x), float(sp1.y)
        x2, y2 = float(sp2.x), float(sp2.y)

        points.append([x1, y1, 0.0])
        points.append([x2, y2, 0.0])

        connectivity.extend([p0, p1])
        offsets.append(len(connectivity))

        point_pore_id.extend([1, 2])
        point_passage_id.extend([passage_id, passage_id])
        point_side_id.extend(
            [
                _side_id(passage.pore1.side),
                _side_id(passage.pore2.side),
            ]
        )
        point_x_frac.extend(
            [
                float(passage.pore1.x_frac),
                float(passage.pore2.x_frac),
            ]
        )
        point_pressure.extend(
            [
                float(state.p1),
                float(state.p2),
            ]
        )

        dx = x2 - x1
        dy = y2 - y1
        length_surface = float(np.hypot(dx, dy))

        if length_surface > 1e-16:
            unit_12 = np.array([dx / length_surface, dy / length_surface, 0.0])
        else:
            unit_12 = np.array([0.0, 0.0, 0.0])

        q = float(state.Q)

        if q >= 0.0:
            flow_unit = unit_12
        else:
            flow_unit = -unit_12

        flow_q_vector = flow_unit * abs(q)

        cell_passage_id.append(passage_id)
        cell_layout_id.append(_layout_id(getattr(passage, "layout_kind", "")))
        cell_pore1_side_id.append(_side_id(passage.pore1.side))
        cell_pore2_side_id.append(_side_id(passage.pore2.side))

        cell_x1_frac.append(float(passage.pore1.x_frac))
        cell_x2_frac.append(float(passage.pore2.x_frac))
        cell_Q.append(q)
        cell_abs_Q.append(abs(q))
        cell_dp_total.append(float(state.dp_total))
        cell_signed_dp.append(float(state.p1 - state.p2))
        cell_p1.append(float(state.p1))
        cell_p2.append(float(state.p2))
        cell_Re.append(float(state.reynolds_equivalent))
        cell_Rs.append(float(state.Rs))
        cell_diameter.append(float(passage.pore1.diameter))
        cell_length.append(float(passage.passage_length(geom)))

        cell_channel_unit.append(unit_12.tolist())
        cell_flow_unit.append(flow_unit.tolist())
        cell_flow_q_vector.append(flow_q_vector.tolist())

    points_array = np.asarray(points, dtype=float)

    n_points = len(points)
    n_lines = len(network.passages)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfLines="{n_lines}">\n')

        f.write('      <PointData Scalars="pore_pressure_Pa">\n')
        _write_data_array_int(f, "pore_id", point_pore_id)
        _write_data_array_int(f, "passage_id", point_passage_id)
        _write_data_array_int(f, "surface_side_id", point_side_id)
        _write_data_array_float(f, "x_frac", point_x_frac)
        _write_data_array_float(f, "pore_pressure_Pa", point_pressure)
        f.write("      </PointData>\n")

        f.write('      <CellData Scalars="Q_m3_s" Vectors="flow_Q_vector_m3_s">\n')

        cell_int_arrays = {
            "passage_id": cell_passage_id,
            "layout_id": cell_layout_id,
            "pore1_side_id": cell_pore1_side_id,
            "pore2_side_id": cell_pore2_side_id,
        }

        for name, values in cell_int_arrays.items():
            _write_data_array_int(f, name, values)

        cell_float_arrays = {
            "x1_frac": cell_x1_frac,
            "x2_frac": cell_x2_frac,
            "Q_m3_s": cell_Q,
            "abs_Q_m3_s": cell_abs_Q,
            "dp_total_Pa": cell_dp_total,
            "signed_dp_p1_minus_p2_Pa": cell_signed_dp,
            "p1_Pa": cell_p1,
            "p2_Pa": cell_p2,
            "reynolds_internal_equivalent": cell_Re,
            "Rs_Pa_s_per_m3": cell_Rs,
            "diameter_m": cell_diameter,
            "passage_length_m": cell_length,
        }

        for name, values in cell_float_arrays.items():
            _write_data_array_float(f, name, values)

        _write_data_array_float(
            f,
            "channel_direction_unit",
            np.asarray(cell_channel_unit, dtype=float),
            number_of_components=3,
        )
        _write_data_array_float(
            f,
            "flow_direction_unit",
            np.asarray(cell_flow_unit, dtype=float),
            number_of_components=3,
        )
        _write_data_array_float(
            f,
            "flow_Q_vector_m3_s",
            np.asarray(cell_flow_q_vector, dtype=float),
            number_of_components=3,
        )

        f.write("      </CellData>\n")

        f.write("      <Points>\n")
        _write_data_array_float(
            f,
            name="Points",
            values=points_array,
            number_of_components=3,
            indent="        ",
        )
        f.write("      </Points>\n")

        f.write("      <Lines>\n")
        _write_data_array_int(f, "connectivity", connectivity)
        _write_data_array_int(f, "offsets", offsets)
        f.write("      </Lines>\n")

        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")

    print(f"[ParaView Saved] {save_path}")
    return save_path


# =============================================================================
# 4. FLOW FIELD CALCULATION
# =============================================================================
def _compute_flow_field(
    solver: Any,
    aero_result: Any,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int,
    ny: int,
) -> dict[str, np.ndarray]:
    """
    Compute a structured 2D flow field around the airfoil.

    The velocity vector is exported for ParaView streamlines.
    Pressure and Cp are computed from Bernoulli using the panel-method velocity.
    """
    geom = solver.geom
    flow = solver.flow
    chord = geom.chord

    x = np.linspace(xlim[0] * chord, xlim[1] * chord, nx)
    y = np.linspace(ylim[0] * chord, ylim[1] * chord, ny)

    XX, YY = np.meshgrid(x, y)

    blank_inside_mask = np.zeros_like(XX, dtype=bool)
    inside_airfoil = solver.build_inside_mask(XX, YY)

    U, V, _, _ = solver.velocity_field(
        XX,
        YY,
        aero_result,
        inside=blank_inside_mask,
    )

    speed = np.sqrt(U**2 + V**2)

    q_inf = 0.5 * flow.rho_inf * flow.v_inf**2
    pressure = flow.p_inf + q_inf - 0.5 * flow.rho_inf * speed**2
    Cp = (pressure - flow.p_inf) / max(q_inf, 1e-16)

    valid_fluid = (~inside_airfoil).astype(float)

    velocity = np.column_stack(
        [
            U.ravel(order="C"),
            V.ravel(order="C"),
            np.zeros(U.size),
        ]
    )

    points = np.column_stack(
        [
            XX.ravel(order="C"),
            YY.ravel(order="C"),
            np.zeros(XX.size),
        ]
    )

    return {
        "X": XX,
        "Y": YY,
        "points": points,
        "U": U,
        "V": V,
        "velocity": velocity,
        "speed": speed,
        "pressure": pressure,
        "Cp": Cp,
        "inside_airfoil": inside_airfoil.astype(float),
        "valid_fluid": valid_fluid,
    }


def _write_structured_grid_vts(
    save_path: Path,
    nx: int,
    ny: int,
    points: np.ndarray,
    point_float_arrays: dict[str, Any],
    point_vector_arrays: dict[str, Any] | None = None,
    active_scalar: str = "speed_m_s",
    active_vector: str = "velocity_m_s",
) -> Path:
    """
    Write a VTK StructuredGrid file.
    """
    point_vector_arrays = point_vector_arrays or {}

    extent = f"0 {nx - 1} 0 {ny - 1} 0 0"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="{extent}">\n')
        f.write(f'    <Piece Extent="{extent}">\n')

        f.write(
            f'      <PointData Scalars="{active_scalar}" Vectors="{active_vector}">\n'
        )

        for name, values in point_vector_arrays.items():
            _write_data_array_float(
                f,
                name=name,
                values=np.asarray(values, dtype=float),
                number_of_components=3,
                indent="        ",
            )

        for name, values in point_float_arrays.items():
            _write_data_array_float(
                f,
                name=name,
                values=np.asarray(values).ravel(order="C"),
                indent="        ",
            )

        f.write("      </PointData>\n")

        f.write("      <Points>\n")
        _write_data_array_float(
            f,
            name="Points",
            values=points,
            number_of_components=3,
            indent="        ",
        )
        f.write("      </Points>\n")

        f.write("    </Piece>\n")
        f.write("  </StructuredGrid>\n")
        f.write("</VTKFile>\n")

    print(f"[ParaView Saved] {save_path}")
    return save_path


# =============================================================================
# 5. POROUS OR SOLID FLOW FIELD EXPORT
# =============================================================================
def export_flow_field_vts(
    output_dir: Path,
    solver: Any,
    aero_result: Any,
    file_stem: str,
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
    nx: int = 300,
    ny: int = 300,
) -> Path:
    """
    Export one flow field as a VTS StructuredGrid.

    This function is used for both:
        1. porous flow field
        2. solid-airfoil flow field

    Main fields:
        velocity_m_s
        U_m_s
        V_m_s
        speed_m_s
        pressure_Pa
        Cp
        inside_airfoil
        valid_fluid
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{file_stem}.vts"

    field = _compute_flow_field(
        solver=solver,
        aero_result=aero_result,
        xlim=xlim,
        ylim=ylim,
        nx=nx,
        ny=ny,
    )

    point_vectors = {
        "velocity_m_s": field["velocity"],
    }

    point_scalars = {
        "U_m_s": field["U"],
        "V_m_s": field["V"],
        "speed_m_s": field["speed"],
        "pressure_Pa": field["pressure"],
        "Cp": field["Cp"],
        "inside_airfoil": field["inside_airfoil"],
        "valid_fluid": field["valid_fluid"],
    }

    return _write_structured_grid_vts(
        save_path=save_path,
        nx=nx,
        ny=ny,
        points=field["points"],
        point_float_arrays=point_scalars,
        point_vector_arrays=point_vectors,
        active_scalar="speed_m_s",
        active_vector="velocity_m_s",
    )


# =============================================================================
# 6. POROUS MINUS SOLID DIFFERENCE FIELD EXPORT
# =============================================================================
def export_flow_field_delta_vts(
    output_dir: Path,
    solver: Any,
    porous_result: Any,
    solid_result: Any,
    file_stem: str = "paraview_flow_field_delta",
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
    nx: int = 300,
    ny: int = 300,
) -> Path:
    """
    Export porous-minus-solid difference field.

    Main fields:
        delta_velocity_m_s
        delta_U_m_s
        delta_V_m_s
        delta_speed_m_s
        delta_pressure_Pa
        delta_Cp
        inside_airfoil
        valid_fluid
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{file_stem}.vts"

    porous = _compute_flow_field(
        solver=solver,
        aero_result=porous_result,
        xlim=xlim,
        ylim=ylim,
        nx=nx,
        ny=ny,
    )

    solid = _compute_flow_field(
        solver=solver,
        aero_result=solid_result,
        xlim=xlim,
        ylim=ylim,
        nx=nx,
        ny=ny,
    )

    delta_U = porous["U"] - solid["U"]
    delta_V = porous["V"] - solid["V"]
    delta_speed = porous["speed"] - solid["speed"]
    delta_pressure = porous["pressure"] - solid["pressure"]
    delta_Cp = porous["Cp"] - solid["Cp"]

    delta_velocity = np.column_stack(
        [
            delta_U.ravel(order="C"),
            delta_V.ravel(order="C"),
            np.zeros(delta_U.size),
        ]
    )

    point_vectors = {
        "delta_velocity_m_s": delta_velocity,
    }

    point_scalars = {
        "delta_U_m_s": delta_U,
        "delta_V_m_s": delta_V,
        "delta_speed_m_s": delta_speed,
        "delta_pressure_Pa": delta_pressure,
        "delta_Cp": delta_Cp,
        "inside_airfoil": porous["inside_airfoil"],
        "valid_fluid": porous["valid_fluid"],
    }

    return _write_structured_grid_vts(
        save_path=save_path,
        nx=nx,
        ny=ny,
        points=porous["points"],
        point_float_arrays=point_scalars,
        point_vector_arrays=point_vectors,
        active_scalar="delta_speed_m_s",
        active_vector="delta_velocity_m_s",
    )


# =============================================================================
# 7. PARAVIEW COLLECTION FILE
# =============================================================================
def export_paraview_collection(
    output_dir: Path,
    files: list[Path],
    collection_name: str = "paraview_results.pvd",
    timesteps: list[float] | None = None,
    parts: list[int] | None = None,
) -> Path:
    """
    Write a ParaView .pvd collection file.

    Parameters
    ----------
    output_dir:
        Folder where the .pvd file is written.

    files:
        VTK files to include. Files may be in subfolders of output_dir.

    collection_name:
        Name of the .pvd collection file.

    timesteps:
        Optional timestep value for each file. For AoA sweeps, this should be
        the angle of attack in degrees.

    parts:
        Optional part index for each file. Use the same part index across
        timesteps when each file belongs to the same physical dataset.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / collection_name

    if timesteps is not None and len(timesteps) != len(files):
        raise ValueError("timesteps must have the same length as files.")

    if parts is not None and len(parts) != len(files):
        raise ValueError("parts must have the same length as files.")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")

        for i, path in enumerate(files):
            timestep = 0.0 if timesteps is None else float(timesteps[i])
            part = i if parts is None else int(parts[i])
            rel_path = _relative_vtk_path(Path(path), output_dir)

            f.write(
                f'    <DataSet timestep="{timestep:.8g}" group="" '
                f'part="{part}" file="{rel_path}"/>\n'
            )

        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")

    print(f"[ParaView Saved] {save_path}")
    return save_path


# =============================================================================
# 8. SOLID REFERENCE EXPORT
# =============================================================================
def export_solid_reference_paraview_files(
    output_root: Path,
    geom: Any,
    aero_solver: Any,
    solid_result: Any,
    field_nx: int = 300,
    field_ny: int = 300,
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
) -> dict[str, Path]:
    """
    Export the single solid-airfoil baseline reference.

    This is used for the main fixed-AoA model run.

    Files:
        solid_reference_paraview/paraview_solid_airfoil_surface.vtp
        solid_reference_paraview/paraview_flow_field_solid.vts
        solid_reference_paraview/paraview_solid_reference.pvd
    """
    solid_dir = output_root / "solid_reference_paraview"
    solid_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}
    files: list[Path] = []

    exported["solid_surface"] = export_solid_airfoil_surface_vtp(
        output_dir=solid_dir,
        geom=geom,
        solid_result=solid_result,
    )
    files.append(exported["solid_surface"])

    exported["solid_flow"] = export_flow_field_vts(
        output_dir=solid_dir,
        solver=aero_solver,
        aero_result=solid_result,
        file_stem="paraview_flow_field_solid",
        xlim=xlim,
        ylim=ylim,
        nx=field_nx,
        ny=field_ny,
    )
    files.append(exported["solid_flow"])

    exported["collection"] = export_paraview_collection(
        output_dir=solid_dir,
        files=files,
        collection_name="paraview_solid_reference.pvd",
    )

    return exported


# =============================================================================
# 9. MASTER SINGLE-CASE EXPORT FUNCTION
# =============================================================================
def export_paraview_files(
    output_dir: Path,
    network: Any,
    geom: Any,
    aero_solver: Any,
    solid_result: Any,
    porous_result: Any,
    field_nx: int = 300,
    field_ny: int = 300,
    xlim: tuple[float, float] = (-0.5, 1.5),
    ylim: tuple[float, float] = (-0.5, 0.5),
    include_solid_exports: bool = False,
    include_difference_exports: bool = False,
    collection_name: str = "paraview_results.pvd",
) -> dict[str, Path]:
    """
    Export ParaView files for one porous model or one AoA sweep case.

    Normal fixed-AoA model case:
        include_solid_exports=False

    AoA sweep case:
        include_solid_exports=True

    Always exported:
        paraview_airfoil_surface.vtp
        paraview_porous_network.vtp
        paraview_flow_field_porous.vts
        paraview_results.pvd

    Extra exports when include_solid_exports=True:
        paraview_solid_airfoil_surface.vtp
        paraview_flow_field_solid.vts

    Returns
    -------
    dict[str, Path]
        Paths keyed by:
            surface
            solid_surface
            network
            porous_flow
            solid_flow
            delta_flow (only when include_difference_exports=True)
            collection
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}
    files: list[Path] = []

    exported["surface"] = export_airfoil_surface_vtp(
        output_dir=output_dir,
        geom=geom,
        solid_result=solid_result,
        porous_result=porous_result,
    )
    files.append(exported["surface"])

    if include_solid_exports:
        exported["solid_surface"] = export_solid_airfoil_surface_vtp(
            output_dir=output_dir,
            geom=geom,
            solid_result=solid_result,
        )
        files.append(exported["solid_surface"])

    exported["network"] = export_porous_network_vtp(
        output_dir=output_dir,
        network=network,
        geom=geom,
        result=porous_result,
    )
    files.append(exported["network"])

    exported["porous_flow"] = export_flow_field_vts(
        output_dir=output_dir,
        solver=aero_solver,
        aero_result=porous_result.aero_result,
        file_stem="paraview_flow_field_porous",
        xlim=xlim,
        ylim=ylim,
        nx=field_nx,
        ny=field_ny,
    )
    files.append(exported["porous_flow"])

    if include_solid_exports:
        exported["solid_flow"] = export_flow_field_vts(
            output_dir=output_dir,
            solver=aero_solver,
            aero_result=solid_result,
            file_stem="paraview_flow_field_solid",
            xlim=xlim,
            ylim=ylim,
            nx=field_nx,
            ny=field_ny,
        )
        files.append(exported["solid_flow"])

    if include_difference_exports:
        exported["delta_flow"] = export_flow_field_delta_vts(
            output_dir=output_dir,
            solver=aero_solver,
            porous_result=porous_result.aero_result,
            solid_result=solid_result,
            file_stem="paraview_flow_field_delta",
            xlim=xlim,
            ylim=ylim,
            nx=field_nx,
            ny=field_ny,
        )
        files.append(exported["delta_flow"])

    exported["collection"] = export_paraview_collection(
        output_dir=output_dir,
        files=files,
        collection_name=collection_name,
    )

    return exported


# =============================================================================
# 10. AOA SWEEP COLLECTION EXPORT
# =============================================================================
def export_aoa_sweep_collections(
    output_dir: Path,
    cases: list[dict[str, Path]],
    alpha_values: list[float],
) -> dict[str, Path]:
    """
    Write parent .pvd files for all AoA sweep ParaView exports.

    Each case should usually be produced with:

        export_paraview_files(..., include_solid_exports=True)

    ParaView treats the AoA value as the timestep, so opening one .pvd file
    lets you step through alpha = -5, -4, ..., 15 deg as a time series.

    Written collections:
        paraview_aoa_sweep_surface.pvd
        paraview_aoa_sweep_solid_surface.pvd
        paraview_aoa_sweep_network.pvd
        paraview_aoa_sweep_porous_flow.pvd
        paraview_aoa_sweep_solid_flow.pvd
            paraview_aoa_sweep_all_parts.pvd
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(cases) != len(alpha_values):
        raise ValueError("cases and alpha_values must have the same length.")

    written: dict[str, Path] = {}

    collection_specs = {
        "surface": "paraview_aoa_sweep_surface.pvd",
        "solid_surface": "paraview_aoa_sweep_solid_surface.pvd",
        "network": "paraview_aoa_sweep_network.pvd",
        "porous_flow": "paraview_aoa_sweep_porous_flow.pvd",
        "solid_flow": "paraview_aoa_sweep_solid_flow.pvd",
    }

    for key, collection_name in collection_specs.items():
        files = [case[key] for case in cases if key in case]
        timesteps = [
            alpha
            for case, alpha in zip(cases, alpha_values)
            if key in case
        ]

        if not files:
            continue

        written[key] = export_paraview_collection(
            output_dir=output_dir,
            files=files,
            collection_name=collection_name,
            timesteps=timesteps,
            parts=[0] * len(files),
        )

    ordered_keys = [
        "surface",
        "solid_surface",
        "network",
        "porous_flow",
        "solid_flow",
    ]

    all_files: list[Path] = []
    all_timesteps: list[float] = []
    all_parts: list[int] = []

    for alpha, case in zip(alpha_values, cases):
        for part_id, key in enumerate(ordered_keys):
            if key not in case:
                continue

            all_files.append(case[key])
            all_timesteps.append(float(alpha))
            all_parts.append(part_id)

    if all_files:
        written["all_parts"] = export_paraview_collection(
            output_dir=output_dir,
            files=all_files,
            collection_name="paraview_aoa_sweep_all_parts.pvd",
            timesteps=all_timesteps,
            parts=all_parts,
        )

    return written