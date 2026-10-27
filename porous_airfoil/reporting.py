"""Aerodynamic comparisons and wall-clock reporting shared by every workflow."""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from collections.abc import Mapping
import csv
import json
import time
import uuid

import numpy as np
import pandas as pd
from . import config as settings

_run_id = ContextVar('porous_run_id', default=None)


def format_elapsed(seconds: float) -> str:
    hours, remainder = divmod(max(0.0, float(seconds)), 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    return f'{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}'


def timed_stage(stage: str):
    """Time the full call, including fields, figures and exports; log failures.

    Nested stages share a run ID. pipeline is the total wall-clock record;
    nested elapsed times overlap and must not be added to the pipeline time.
    """
    def decorate(operation):
        @wraps(operation)
        def run(*args, **kwargs):
            token = None
            run_id = _run_id.get()
            if run_id is None:
                run_id = uuid.uuid4().hex
                token = _run_id.set(run_id)
            started = datetime.now(timezone.utc)
            tick = time.perf_counter()
            status, message, result = 'completed', '', None
            try:
                result = operation(*args, **kwargs)
                if isinstance(result, dict):
                    status = result.get('status', status)
                elif isinstance(result, pd.DataFrame) and 'coupling_converged' in result:
                    failed = int((~result['coupling_converged'].fillna(False).astype(bool)).sum())
                    if failed:
                        status = 'completed_with_unconverged_cases'
                        message = f'{failed} cases retained as NaN'
                return result
            except BaseException as exc:
                status = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
                message = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                elapsed = time.perf_counter()-tick
                finished = datetime.now(timezone.utc)
                record = dict(run_id=run_id, stage=stage, status=status,
                              start_utc=started.isoformat(), finish_utc=finished.isoformat(),
                              elapsed_seconds=elapsed, elapsed_hh_mm_ss=format_elapsed(elapsed),
                              message=message)
                if isinstance(result, dict):
                    result.setdefault('elapsed_seconds', elapsed)
                if settings.PRINT_ELAPSED_TIME:
                    print(f'[Timing] {stage}: {format_elapsed(elapsed)} ({elapsed:.3f} s); {status}')
                try:
                    if settings.SAVE_TIMING_REPORT:
                        root = Path(settings.REPORT_OUTPUT_DIR)
                        root.mkdir(parents=True, exist_ok=True)
                        path = root/'run_timing.csv'
                        header = not path.exists() or path.stat().st_size == 0
                        with path.open('a', newline='', encoding='utf-8') as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(record))
                            if header:
                                writer.writeheader()
                            writer.writerow(record)
                        if stage == 'pipeline' or token is not None:
                            (root/'latest_run_timing.json').write_text(
                                json.dumps(record, indent=2)+'\n', encoding='utf-8')
                except OSError as exc:
                    print(f'[Timing] Could not save timing report: {exc}')
                finally:
                    if token is not None:
                        _run_id.reset(token)
        return run
    return decorate


def _number(result, coefficient):
    value = result.get(coefficient, float('nan')) if isinstance(result, Mapping) else getattr(result, coefficient, float('nan'))
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


def _relative_change(delta, baseline):
    if not np.isfinite(delta) or not np.isfinite(baseline) or abs(baseline) <= settings.METRICS_MIN_ABS_BASELINE:
        return float('nan')
    return 100.0*delta/abs(baseline)


def aerodynamic_change_metrics(solid_result, porous_result=None, converged=True):
    """Signed coefficient changes, percentages and explicitly pressure-only L/D.

    Percent change is 100*(porous-solid)/abs(solid). Positive drag_reduction
    means a decrease in pressure drag. Near-zero baselines give NaN relative
    changes, while absolute changes remain available. Failed candidates never
    contribute porous performance metrics.
    """
    valid = bool(converged and porous_result is not None)
    values = {'metrics_valid': valid,
              'metric_reference': 'solid_panel_same_AoA_Re',
              'relative_change_definition': '100*(porous-solid)/abs(solid)',
              'aerodynamic_force_definition': 'closed_contour_pressure_integral',
              'drag_definition': 'inviscid_pressure_only'}
    for key in ('CL', 'CD', 'CM'):
        baseline = _number(solid_result, key)
        porous = _number(porous_result, key) if valid else float('nan')
        delta = porous-baseline
        values.update({f'solid_{key}': baseline, f'porous_{key}': porous,
                       f'delta_{key}': delta,
                       f'{key}_change_percent': _relative_change(delta, baseline)})
    values['lift_change_percent'] = values['CL_change_percent']
    values['drag_reduction_percent'] = -values['CD_change_percent']
    values['moment_change_percent'] = values['CM_change_percent']
    # Backwards-compatible legacy lift-gain formula, safely guarded at CL=0.
    cl0 = values['solid_CL']
    values['gain_percent_vs_solid_panel'] = (
        100.0*(values['porous_CL']/cl0-1.0)
        if np.isfinite(cl0) and abs(cl0) > settings.METRICS_MIN_ABS_BASELINE else float('nan'))
    for prefix in ('solid', 'porous'):
        cl, cd = values[f'{prefix}_CL'], values[f'{prefix}_CD']
        values[f'{prefix}_pressure_LD'] = (
            cl/cd if np.isfinite(cl) and np.isfinite(cd)
            and cd > settings.METRICS_MIN_POSITIVE_CD else float('nan'))
    ld0, ld = values['solid_pressure_LD'], values['porous_pressure_LD']
    values['delta_pressure_LD'] = ld-ld0
    values['pressure_LD_change_percent'] = _relative_change(ld-ld0, ld0)
    values['pressure_LD_status'] = ('defined' if np.isfinite(ld) and np.isfinite(ld0)
                                     else 'unavailable_nonpositive_or_small_CD_or_invalid_case')
    return values


def print_case_metrics(metrics, label=''):
    if not settings.PRINT_AERODYNAMIC_METRICS:
        return
    print(f'[Aerodynamic change] {label}'.rstrip())
    for key in ('CL', 'CD', 'CM'):
        print(f"  {key}: solid={metrics['solid_'+key]: .8g}, porous={metrics['porous_'+key]: .8g}, "
              f"change={metrics['delta_'+key]:+.6g}, change[%]={metrics[key+'_change_percent']:+.4g}")
    print(f"  Pressure drag reduction[%]={metrics['drag_reduction_percent']:+.4g}; "
          f"pressure L/D change[%]={metrics['pressure_LD_change_percent']:+.4g}")


def enrich_aerodynamic_frame(frame):
    """Support both fixed-case solid_* and sweep solid_panel_* CSV schemas."""
    frame = frame.copy()
    records = []
    for _, row in frame.iterrows():
        baseline = {key: row.get('solid_panel_'+key, row.get('solid_'+key, np.nan))
                    for key in ('CL', 'CD', 'CM')}
        porous = {key: row.get('porous_'+key, np.nan) for key in ('CL', 'CD', 'CM')}
        flag = row.get('coupling_converged', True)
        valid = str(flag).lower() in ('true', '1', '1.0') if isinstance(flag, str) else bool(pd.notna(flag) and flag)
        records.append(aerodynamic_change_metrics(baseline, porous, valid))
    if records:
        metrics = pd.DataFrame(records, index=frame.index)
        for column in metrics:
            frame[column] = metrics[column]
    return frame


def report_aerodynamic_frame(frame, output_dir, stem='aerodynamic_change_metrics'):
    frame = enrich_aerodynamic_frame(frame)
    if settings.SAVE_AERODYNAMIC_METRICS:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root/(stem+'.csv')
        frame.to_csv(path, index=False)
        print(f'[CSV Saved] {path}')
    if settings.PRINT_AERODYNAMIC_METRICS and not frame.empty:
        display = frame
        angle_column = 'alpha_deg' if 'alpha_deg' in frame else 'aoa_deg'
        if settings.METRICS_PRINT_AOA_DEG is not None and angle_column in frame:
            display = frame.loc[np.isclose(pd.to_numeric(frame[angle_column], errors='coerce'),
                                           settings.METRICS_PRINT_AOA_DEG, rtol=0.0, atol=1e-8)]
        columns = ['model_name', angle_column, 'channel_height_m', 'coupling_converged',
                   'delta_CL', 'CL_change_percent', 'delta_CD', 'drag_reduction_percent',
                   'delta_CM', 'CM_change_percent', 'pressure_LD_change_percent']
        columns = [key for key in columns if key in display]
        print('[Aerodynamic change] Relative to the solid panel baseline at the same AoA/Re.')
        if not display.empty:
            print(display[columns].to_string(index=False, float_format=lambda value: f'{value:.6g}'))
        else:
            print(f'  No rows at the requested reporting AoA {settings.METRICS_PRINT_AOA_DEG:g} deg; full results are in the CSV.')
        print('  CD and L/D describe inviscid pressure forces. NaN percentages indicate an undefined baseline or invalid case.')
    return frame
