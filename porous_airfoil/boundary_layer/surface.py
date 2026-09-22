"""Surface geometry and stagnation-point mapping for boundary-layer marching."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .state import BLStation


@dataclass
class BLSurfacePath:
    """Ordered marching stations along upper or lower surface from stagnation."""

    side: str
    stations: list[BLStation]

    @property
    def s(self) -> np.ndarray:
        return np.array([st.s for st in self.stations], dtype=float)

    @property
    def x(self) -> np.ndarray:
        return np.array([st.x for st in self.stations], dtype=float)

    @property
    def y(self) -> np.ndarray:
        return np.array([st.y for st in self.stations], dtype=float)

    @property
    def Ue(self) -> np.ndarray:
        return np.array([st.Ue for st in self.stations], dtype=float)

    @property
    def dUe_ds(self) -> np.ndarray:
        return np.array([st.dUe_ds for st in self.stations], dtype=float)

    @property
    def Cp(self) -> np.ndarray:
        return np.array([st.Cp for st in self.stations], dtype=float)

    @property
    def panel_ids(self) -> np.ndarray:
        return np.array([st.panel_id for st in self.stations], dtype=int)


def find_stagnation_point(geom: object, Vt: np.ndarray) -> tuple[int, float, float, float]:
    """Find the stagnation point where tangential velocity Vt crosses zero.

    Returns
    -------
    tuple[int, float, float, float]
        (stagnation_panel_id, x_stag, y_stag, s_stag_global)
    """
    Vt = np.asarray(Vt, dtype=float)

    # Stagnation is near the minimum |Vt| / maximum Cp
    min_idx = int(np.argmin(np.abs(Vt)))

    # Compute panel-centre arc lengths along the closed contour
    s_centres = np.zeros(geom.num_pan, dtype=float)
    s_cum = 0.0
    for i in range(geom.num_pan):
        s_centres[i] = s_cum + 0.5 * geom.S[i]
        s_cum += geom.S[i]

    # Find sign transition in adjacent panel tangential velocities
    zero_cross_idx = min_idx
    for i in range(max(0, min_idx - 5), min(geom.num_pan - 1, min_idx + 5)):
        if Vt[i] * Vt[i + 1] <= 0.0:
            zero_cross_idx = i
            break

    i1 = zero_cross_idx
    i2 = min(i1 + 1, geom.num_pan - 1)

    v1, v2 = Vt[i1], Vt[i2]
    s1, s2 = s_centres[i1], s_centres[i2]

    if abs(v2 - v1) > 1e-12:
        frac = -v1 / (v2 - v1)
        frac = float(np.clip(frac, 0.0, 1.0))
    else:
        frac = 0.5

    s_stag = s1 + frac * (s2 - s1)
    x_stag = float((1.0 - frac) * geom.XC[i1] + frac * geom.XC[i2])
    y_stag = float((1.0 - frac) * geom.YC[i1] + frac * geom.YC[i2])

    return zero_cross_idx, x_stag, y_stag, s_stag


def build_bl_surface_paths(geom: object, aero_result: object) -> tuple[BLSurfacePath, BLSurfacePath]:
    """Construct upper and lower surface paths marching from stagnation (s=0) to TE.

    Upper surface marches from stagnation towards TE_upper ($s \rightarrow s_{TE,u}$).
    Lower surface marches from stagnation towards TE_lower ($s \rightarrow s_{TE,l}$).
    Edge velocity is prescribed as $U_e = |V_t|$.
    """
    Vt = np.asarray(aero_result.Vt, dtype=float)
    Cp = np.asarray(aero_result.Cp, dtype=float)

    stag_id, x_stag, y_stag, _ = find_stagnation_point(geom, Vt)

    upper_ids = np.sort(np.asarray(geom.upper_panel_ids, dtype=int))
    lower_ids = np.sort(np.asarray(geom.lower_panel_ids, dtype=int))

    # Split panel IDs into upper and lower paths originating at stagnation
    upper_march_ids = upper_ids[upper_ids >= stag_id]
    if len(upper_march_ids) == 0:
        upper_march_ids = upper_ids

    lower_march_ids = lower_ids[lower_ids <= stag_id][::-1]
    if len(lower_march_ids) == 0:
        lower_march_ids = lower_ids[::-1]

    def _build_path(panel_sequence: np.ndarray, side_label: str) -> BLSurfacePath:
        stations: list[BLStation] = []

        # Stagnation station (s = 0)
        stations.append(
            BLStation(
                panel_id=stag_id,
                x=x_stag,
                y=y_stag,
                s=0.0,
                Ue=0.0,
                Cp=1.0,
                dUe_ds=0.0,
            )
        )

        s_acc = 0.0
        prev_x, prev_y = x_stag, y_stag

        for pid in panel_sequence:
            px = float(geom.XC[pid])
            py = float(geom.YC[pid])
            ds = float(np.hypot(px - prev_x, py - prev_y))
            if ds < 1e-12:
                continue

            s_acc += ds
            ue_val = abs(float(Vt[pid]))
            cp_val = float(Cp[pid])

            stations.append(
                BLStation(
                    panel_id=int(pid),
                    x=px,
                    y=py,
                    s=s_acc,
                    Ue=ue_val,
                    Cp=cp_val,
                    dUe_ds=0.0,
                )
            )
            prev_x, prev_y = px, py

        # Compute dUe/ds using central differences
        n = len(stations)
        s_arr = np.array([st.s for st in stations])
        ue_arr = np.array([st.Ue for st in stations])
        due_ds = np.zeros(n, dtype=float)

        if n > 1:
            due_ds[1:-1] = (ue_arr[2:] - ue_arr[:-2]) / (s_arr[2:] - s_arr[:-2])
            due_ds[0] = (ue_arr[1] - ue_arr[0]) / (s_arr[1] - s_arr[0])
            due_ds[-1] = (ue_arr[-1] - ue_arr[-2]) / (s_arr[-1] - s_arr[-2])

        # Assign updated derivatives
        updated_stations = []
        for i, st in enumerate(stations):
            updated_stations.append(
                BLStation(
                    panel_id=st.panel_id,
                    x=st.x,
                    y=st.y,
                    s=st.s,
                    Ue=st.Ue,
                    Cp=st.Cp,
                    dUe_ds=float(due_ds[i]),
                )
            )

        return BLSurfacePath(side=side_label, stations=updated_stations)

    upper_path = _build_path(upper_march_ids, "upper")
    lower_path = _build_path(lower_march_ids, "lower")

    return upper_path, lower_path