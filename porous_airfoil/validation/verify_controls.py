"""Integration smoke checks for supplied modules; missing XFOIL/ParaView are isolated."""
from pathlib import Path
import ast
import importlib
import json
import shutil
import sys
import types
import logging
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
import tempfile
TEST_ROOT=Path(tempfile.mkdtemp(prefix='porous_controls_'))
if TEST_ROOT.exists(): shutil.rmtree(TEST_ROOT)
TEST_ROOT.mkdir(exist_ok=True)

def package_variant(name, overrides):
 folder=TEST_ROOT/name
 if folder.exists():shutil.rmtree(folder)
 folder.mkdir(parents=True)
 for source_name in ('config.py','solver.py','porous.py','reporting.py','workflows.py','run_configured.py'):
  shutil.copy2(ROOT/source_name,folder/source_name)
 shutil.copytree(ROOT/'plotting',folder/'plotting',ignore=shutil.ignore_patterns('__pycache__'))
 config=folder/'config.py';source=config.read_text();lines=source.splitlines(keepends=True)
 for node in reversed(ast.parse(source).body):
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
   var=node.targets[0].id
   if var in overrides:lines[node.lineno-1:node.end_lineno]=[f'{var} = {overrides[var]}\n']
 config.write_text(''.join(lines))
 pkg=types.ModuleType(name);pkg.__path__=[str(folder)];sys.modules[name]=pkg
 integration=types.ModuleType(name+'.integrations');integration.__path__=[];sys.modules[integration.__name__]=integration
 def unavailable(*a,**kw):raise AssertionError('Disabled external integration was invoked')
 for mod,names in {'xfoil':['load_xfoil_cp','load_xfoil_polar_dataframe','run_xfoil_cp','run_xfoil_polar_sweep'],
                   'paraview':['export_aoa_sweep_collections','export_paraview_files','export_solid_reference_paraview_files']}.items():
  m=types.ModuleType(name+'.integrations.'+mod)
  for function in names:setattr(m,function,unavailable)
  sys.modules[m.__name__]=m
 return importlib.import_module(name+'.config')

base={'OUTPUT_ROOT':repr(TEST_ROOT/'outputs'),'USE_XFOIL':'False','PNG_DPI':'60','AERODYNAMIC_PNG_DPI':'60',
      'N_PANELS':'160','CONTOUR_NX':'48','CONTOUR_NY':'32','COUPLING_VERBOSE':'False',
      'AOA_SWEEP_START_DEG':'0.0','AOA_SWEEP_END_DEG':'2.0','METRICS_PRINT_AOA_DEG':'1.0',
      'PLOT_STREAMLINE_EXTERNAL_STREAMLINE_COUNT_CLOSE':'6','PLOT_STREAMLINE_EXTERNAL_STREAMLINE_COUNT_NORMAL':'6',
      'PLOT_STREAMLINE_STREAMLINE_MAX_STEP_C':'0.03','PLOT_STREAMLINE_STREAMLINE_RTOL':'1e-4',
      'PLOT_STREAMLINE_STREAMLINE_ATOL':'1e-6','PLOT_STREAMLINE_STREAMLINE_MIN_POINTS':'4'}
# Path repr must be an expression, not bare PosixPath.
base['OUTPUT_ROOT']='Path('+repr(str(TEST_ROOT/'outputs'))+')'
# 1. One edited master list propagates to every consumer.
c=package_variant('controls_binding',{**base,'PORE_HEIGHTS_M':'(0.0070,0.0085,0.0100)','SELECTED_MODEL':repr('model_1_4_chordwise')})
assert c.FIXED_AOA_CHANNEL_HEIGHTS==c.AOA_SWEEP_CHANNEL_HEIGHTS==c.PLOT_AERO_SELECT_PORE_DIAMETERS_M==c.PORE_HEIGHTS_M
modules={name:importlib.import_module('controls_binding.plotting.'+name) for name in ['aerodynamic','geometry','contours','streamlines']}
assert modules['aerodynamic'].SELECT_PORE_DIAMETERS_M==c.PORE_HEIGHTS_M
for mod,attribute in [('geometry','EXPORT_PORE_DIAMETER_M'),('contours','FINAL_CONTOUR_PORE_DIAMETER_M'),('streamlines','STREAMLINE_PORE_DIAMETER_M')]:
 assert getattr(modules[mod],attribute)==.007
assert len(modules['aerodynamic'].MODEL_PANELS)==1
# 2. Only one stage's override changes its selection.
c2=package_variant('controls_override',{**base,'PORE_HEIGHTS_M':'(.007,.008,.009)',
  'CONTOUR_HEIGHT_OVERRIDE_M':'.010','STREAMLINE_AOA_OVERRIDE_DEG':'7.0',
  'PLOT_STREAMLINE_EXPORT_NORMAL_VIEW':'True','PLOT_STREAMLINE_EXPORT_CLOSE_UP_VIEW':'False'})
assert c2.PLOT_CONTOUR_FINAL_CONTOUR_PORE_DIAMETER_M==.010
assert c2.PLOT_STREAMLINE_STREAMLINE_PORE_DIAMETER_M==.007
assert c2.PLOT_STREAMLINE_STREAMLINE_AOA_DEG==7.
st=importlib.import_module('controls_override.plotting.streamlines')
assert [v.folder_name for v in st.streamline_views()]==['normal']
# 3. Master plot switch prevents importing every disabled stage.
c3=package_variant('controls_disabled',{**base,'OUTPUT_ROOT':'Path('+repr(str(TEST_ROOT/'disabled'))+')',
 'RUN_FIXED_CASES':'False','RUN_AOA_SWEEP':'False','ENABLE_PLOTS':'False'})
w3=importlib.import_module('controls_disabled.workflows')
r3=w3.run_all_workflows()
assert not any('controls_disabled.plotting.'+x in sys.modules for x in ['geometry','contours','streamlines'])
status=pd.read_csv(Path(c3.OUTPUT_ROOT)/'pipeline_stage_status.csv')
assert set(status.status)=={'skipped'}
assert r3['elapsed_seconds']>=0.
# The core imports aerodynamic.py for its fixed-case API; no main/field solve
# or stage import for geometry/contours/streamlines occurs when disabled.
# 4. Metrics signs, zero references, invalid cases and efficiency guards.
rpt=importlib.import_module('controls_binding.reporting')
m=rpt.aerodynamic_change_metrics({'CL':1.,'CD':.1,'CM':-.1},{'CL':1.2,'CD':.08,'CM':-.08})
assert np.isclose(m['CL_change_percent'],20.) and np.isclose(m['drag_reduction_percent'],20.)
assert np.isclose(m['pressure_LD_change_percent'],50.) and np.isclose(m['CM_change_percent'],20.)
z=rpt.aerodynamic_change_metrics({'CL':0.,'CD':0.,'CM':0.},{'CL':.2,'CD':.01,'CM':-.01})
assert np.isnan(z['CL_change_percent']) and z['delta_CL']==.2 and np.isnan(z['pressure_LD_change_percent'])
f=rpt.aerodynamic_change_metrics({'CL':1.,'CD':.1,'CM':-.1},{'CL':2.,'CD':.1,'CM':-.1},False)
assert np.isnan(f['delta_CL']) and np.isnan(f['porous_CL'])
negative=rpt.aerodynamic_change_metrics({'CL':1.,'CD':-.01,'CM':0.},{'CL':1.2,'CD':.01,'CM':0.})
assert np.isnan(negative['pressure_LD_change_percent'])
# 5. Full supplied pipeline, genuine solver and genuine plot/field functions.
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
c4=package_variant('controls_pipeline',base)
w4=importlib.import_module('controls_pipeline.workflows')
r4=w4.run_all_workflows()
print('PIPELINE RESULTS',r4['status'],flush=True)
assert r4['status'] in ('completed','completed_with_unconverged_cases'),r4
sweep=r4['aoa_sweep'];assert len(sweep)==27
assert sweep.coupling_converged.any()
failed=sweep.loc[~sweep.coupling_converged]
assert failed[['porous_CL','porous_CD','porous_CM','CL_change_percent']].isna().all().all()
for field in ['delta_CL','CL_change_percent','delta_CD','drag_reduction_percent','delta_CM','pressure_LD_change_percent','elapsed_seconds']:
 assert field in sweep,field
assert np.all(sweep.elapsed_seconds>0.)
assert np.allclose(sweep['delta_CL'],sweep['porous_CL']-sweep['solid_panel_CL'],equal_nan=True)
timing=pd.read_csv(Path(c4.REPORT_OUTPUT_DIR)/'run_timing.csv')
for stage in ['pipeline','aoa_sweep','aero_plots','geometry','contours','streamlines']:
 assert stage in set(timing.stage),stage
assert len(set(timing.run_id))==1
pipeline_time=timing[timing.stage=='pipeline'].elapsed_seconds.iloc[-1]
assert pipeline_time>=timing[timing.stage=='aoa_sweep'].elapsed_seconds.iloc[-1]
assert pipeline_time>=timing[timing.stage=='contours'].elapsed_seconds.iloc[-1]
assert (Path(c4.REPORT_OUTPUT_DIR)/'latest_run_timing.json').exists()
# All generated image/export sizes and report schemas are checked.
images=list(Path(c4.OUTPUT_ROOT).rglob('*.png'))
assert len(images)>=12,len(images)
assert all(path.stat().st_size>1000 for path in images)
for stage,rootpath,pattern in [('contours',c4.PLOT_CONTOUR_FINAL_CONTOUR_OUTPUT_ROOT,'all_final_contour_cases_summary.csv'),('streamlines',c4.PLOT_STREAMLINE_STREAMLINE_OUTPUT_ROOT,'all_streamline_cases_summary.csv')]:
 paths=list(Path(rootpath).rglob(pattern));assert len(paths)==1
 frame=pd.read_csv(paths[0]);assert len(frame)==3
 assert frame.coupling_converged.all()
 assert all(key in frame for key in ['solid_CD','porous_CD','CD_change_percent','solve_elapsed_seconds','elapsed_seconds'])
 assert (frame.elapsed_seconds>=frame.solve_elapsed_seconds).all()
# Metrics from standalone aerodynamic plotting preserve the third 10 mm case.
plot_metrics=pd.read_csv(Path(c4.PLOT_AERO_OUTPUT_DIR)/'aerodynamic_change_metrics.csv')
assert set(np.round(plot_metrics.channel_height_m,4))=={.008,.009,.010}
summary={'master_height_binding':'passed','per_stage_overrides':'passed','disabled_stages':'passed',
 'metric_edge_cases':'passed','pipeline_status':r4['status'],'sweep_cases':len(sweep),
 'converged_cases':int(sweep.coupling_converged.sum()),'unconverged_coarse_mesh_cases':len(failed),'png_files':len(images),
 'pipeline_elapsed_seconds':float(pipeline_time),'stage_timings':timing[['stage','status','elapsed_seconds']].to_dict('records'),
 'external_integration_scope':'XFOIL and ParaView unavailable; disabled and verified never invoked'}
(TEST_ROOT/'controls_validation_results.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
