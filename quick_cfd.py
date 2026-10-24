#!/usr/bin/env python3
"""GXXX airfoil points -> parameterized LF1/LF2 mesh -> solver -> SI exports."""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import html
import json
import os
import subprocess
import sys
import time
from pipeline import configuration as cfg
from pipeline.geometry import prepare
from pipeline.export import export_case, mesh_svg
import numpy as np

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def configure(args):
    c = cfg.load(args.config)
    if args.stop_time is not None and args.physical_time is not None:
        raise ValueError('choose --stop-time or --physical-time, not both')
    if len(args.levels) != len(set(args.levels)):
        raise ValueError('levels must be distinct')
    for attr, section, key in [('mach','freestream','Mach'), ('aoa','freestream','AOA_degrees'),
                              ('pressure','freestream','p_Pa'), ('temperature','freestream','T_K'),
                              ('chord','freestream','chord_m'), ('turbulence','physics','turbulence'),
                              ('viscosity','physics','viscosity'), ('backend','solver','backend'),
                              ('cfl','solver','CFL'), ('mode','solver','mode'), ('stop_time','solver','stop_time_star')]:
        value = getattr(args, attr, None)
        if value is not None:
            c[section][key] = value
    if args.cfl is not None and args.dt is not None:
        raise ValueError('choose --cfl or --dt, not both')
    if args.cfl is not None:
        c['solver']['time_step'] = dict(mode='cfl', dt_seconds=None)
    if args.dt is not None:
        c['solver']['time_step'] = dict(mode='fixed', dt_seconds=args.dt)
    if args.physical_time is not None:
        c['solver']['stop_time_star'] = args.physical_time/cfg.scales(c)['time']
    for attr, key in [('fields','fields'), ('formats','formats'), ('region','region_xy_c'), ('load_fields','load_fields')]:
        value = getattr(args, attr, None)
        if value is not None:
            c['output'][key] = value
    if args.surface is not None:
        c['output']['surface'] = args.surface
    if args.loads is not None:
        c['output']['loads'] = args.loads
    if args.volume is not None:
        c['output']['volume'] = args.volume
    return cfg.validate(c)


def executable(backend):
    cpu = ROOT/'build_cpu/quick_airfoil'
    return cpu if backend=='cpu' and cpu.exists() else ROOT/'build/quick_airfoil'


def build(args):
    folder=ROOT/('build_cpu' if args.cpu else 'build')
    if args.jobs < 1:
        raise ValueError('--jobs must be positive')
    cache = folder/'CMakeCache.txt'
    if cache.exists():
        homes = [line.split('=',1)[1] for line in cache.read_text().splitlines() if line.startswith('CMAKE_HOME_DIRECTORY:INTERNAL=')]
        if homes and Path(homes[0]).resolve() != ROOT:
            backup = ROOT/(folder.name+'_previous_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            folder.rename(backup)
            print('Preserved previous-location build at', backup)
    subprocess.run(['cmake','-S',str(ROOT),'-B',str(folder),'-DCMAKE_BUILD_TYPE=Release',
                    '-DCFD_CUDA='+('OFF' if args.cpu else 'ON'), '-DCMAKE_CUDA_ARCHITECTURES='+args.cuda_arch],check=True)
    subprocess.run(['cmake','--build',str(folder),'--parallel',str(args.jobs),'--target','quick_airfoil'],check=True)


def run_one(exe, config, label, geometry, folder, name, mesh_only=False):
    folder.mkdir(parents=True)
    (folder/'internal').mkdir()
    dump(folder/'config.resolved.json', config)
    settings=folder/'settings.txt'
    settings.write_text(cfg.settings(config,label,name,'mesh' if mesh_only else None))
    prefix=folder/'internal/flow'
    cmd=[str(exe),'--settings',str(settings),'--geometry',str(geometry/'airfoil.csv'),
         '--patches',str(geometry/'patches.csv'),'--out',str(prefix)]
    dump(folder/'command.json',cmd)
    start=time.perf_counter()
    with (folder/'solver.log').open('w') as log:
        process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
                               env={**os.environ,'OMP_NUM_THREADS':'1'})
    wall=time.perf_counter()-start
    result_path=Path(str(prefix)+'.result.json')
    if process.returncode!=0 or not result_path.exists():
        dump(folder/'invocation.json',dict(returncode=process.returncode,process_seconds=wall,failed=True))
        raise RuntimeError(f'{label} solver failed; inspect {folder/"solver.log"}')
    result=json.loads(result_path.read_text())
    expected_order=config['solver']['spatial_order']
    expected_metric=config['solver']['convergence']['metric']
    if result.get('spatial_order',1)!=expected_order or result.get('convergence_metric','all_loads')!=expected_metric:
        raise RuntimeError(f'{label} executable did not honor requested spatial order or convergence metric; rebuild the solver')
    result['mesh_workflow']='cartesian_amr_2d'
    pseudo=(config['solver']['preconditioning']!='none' or config['solver']['local_pseudo_time'] or config['solver']['adaptive_cfl']['enabled'])
    result.update(process_seconds=wall,returncode=process.returncode,profile=label,
                  time_kind='pseudo' if pseudo else 'physical',
                  preconditioning=config['solver']['preconditioning'],
                  physical_time_seconds=(result['physical_time']*cfg.scales(config)['time'] if not pseudo else None))
    if result['time_kind']=='pseudo':
        result['pseudo_time_star']=result['physical_time']
    result['gpu']=json.loads(Path(str(prefix)+'.gpu.json').read_text())
    if mesh_only:
        out=folder/'output';out.mkdir()
        polygon=np.loadtxt(geometry/'airfoil.csv',delimiter=',',skiprows=1)
        mesh_svg(prefix,config,polygon,out/'mesh.svg')
        result['export']=dict(mesh_only=True)
    else:
        result['export']=export_case(folder,config,geometry)
    dump(folder/'result.json',result)
    return result


def index_page(out, summary):
    rows=[]
    for c in summary['cases']:
        label=c['profile'];state='mesh-only' if c['mode']=='mesh' else ('steady' if c['converged'] else c['stop_reason'])
        rows.append(f'<tr><td>{label}</td><td>{c["fluid_control_volumes"]}</td><td>{html.escape(state)}</td>'
                    f'<td>{c["mesh_setup_seconds"]:.4f}</td><td>{c["gpu"]["advance_cuda_event_seconds"]:.4f}</td>'
                    f'<td>{c["process_seconds"]:.4f}</td><td><a href="{label}/result.json">result</a> · '
                    f'<a href="{label}/output/">outputs</a></td></tr>')
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Quick CFD airfoil result</title>
<style>body{font:16px/1.8 system-ui;max-width:1100px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}img{width:100%}</style>'''
    text+=f'<h1>{html.escape(summary["geometry"]["name"])}: LF1 / LF2</h1><p>This result uses the dedicated two-dimensional Cartesian-AMR airfoil workflow. Each grid starts independently from uniform freestream at t=0. Mechanical viscous model with kappa=0. Coordinates lie in the x-y plane; positive Mz is counterclockwise. Coarse-grid coefficients support qualitative studies and version comparisons.</p>'
    text+='<table><tr><th>profile</th><th>Fluid CVs</th><th>status</th><th>mesh(s)</th><th>GPU event(s)</th><th>process(s)</th><th>Data</th></tr>'+''.join(rows)+'</table>'
    text+='<p>Total pipeline time, including geometry processing, solver processes, and Python exports: '+format(summary['pipeline_seconds'],'.4f')+' seconds. GPU event time is elapsed stream time, including possible dispatch gaps; it is zero for CPU runs. Full timing and configuration: <a href="pipeline.json">pipeline.json</a>.</p>'
    for c in summary['cases']:
        text+=f'<h2>{c["profile"]} mesh close-up</h2><img src="{c["profile"]}/output/mesh.svg"><p>Background member rectangles and the original airfoil outline are shown. Cut-cell integration uses actual fluid areas and wall fragments.</p>'
    (out/'index.html').write_text(text+'</html>')


def run(args):
    start=time.perf_counter()
    c=configure(args)
    exe=Path(args.executable).resolve() if args.executable else executable(c['solver']['backend'])
    if not exe.exists():
        raise ValueError('solver not built: run python3 quick_cfd.py build (or build --cpu)')
    source=Path(args.points).resolve()
    if not source.is_file():
        raise ValueError(f'input file not found: {source}')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=Path(args.out).resolve() if args.out else ROOT/'runs'/(source.stem+'_'+stamp)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'output folder is not empty: {out}; choose a new folder')
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'config.resolved.json',c)
    prepared=time.perf_counter();geometry=prepare(source,out/'geometry',args.point_format)
    prep_seconds=time.perf_counter()-prepared
    vertices=np.loadtxt(out/'geometry/airfoil.csv',delimiter=',',skiprows=1)
    b=c['mesh']['bounds_xy_c']
    if not (np.all(vertices[:,0]>b[0]) and np.all(vertices[:,0]<b[1]) and np.all(vertices[:,1]>b[2]) and np.all(vertices[:,1]<b[3])):
        raise ValueError('airfoil must lie strictly inside configured domain')
    summary=dict(mesh_workflow='cartesian_amr_2d',geometry=geometry,config=c,scales=cfg.scales(c),geometry_preparation_seconds=prep_seconds,
                 executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').iterdir() if p.is_file()},
                 pipeline_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'quick_cfd.py',ROOT/'config/default.json',*(ROOT/'pipeline').glob('*.py')]},
                 initial_condition='each level independently uniform freestream t0; no LF1-to-LF2 warm start',cases=[])
    for label in args.levels:
        print(f'{geometry["name"]} {label}: mesh -> '+('output' if args.command=='mesh' else 'solve -> output'),flush=True)
        result=run_one(exe,c,label,out/'geometry',out/label,geometry['name'],args.command=='mesh')
        summary['cases'].append(result)
        summary['pipeline_seconds']=time.perf_counter()-start
        dump(out/'pipeline.json',summary)
        print(f'{label}: {result["fluid_control_volumes"]} CV, mesh={result["mesh_setup_seconds"]:.4f}s, '
              f'process={result["process_seconds"]:.3f}s, converged={result["converged"]}, reason={result["stop_reason"]}',flush=True)
    index_page(out,summary)
    print(out/'index.html')
    if args.command=='run' and c['solver']['mode']=='steady' and not all(r['converged'] for r in summary['cases']):
        return 3
    return 0


def export(args):
    out=Path(args.run).resolve()
    # Re-export options without changing the solved physics/scales.
    original=json.loads((out/'config.resolved.json').read_text())
    if args.config:
        patch=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        cfg.merge(original['output'],patch.get('output',{}),'output.')
    for attr,key in [('fields','fields'),('formats','formats'),('region','region_xy_c'),('load_fields','load_fields')]:
        value=getattr(args,attr,None)
        if value is not None:original['output'][key]=value
    if args.surface is not None:original['output']['surface']=args.surface
    if args.loads is not None:original['output']['loads']=args.loads
    if args.volume is not None:original['output']['volume']=args.volume
    cfg.validate(original)
    summary=json.loads((out/'pipeline.json').read_text())
    exports=[]
    for case in summary['cases']:
        if case['mode']=='mesh':raise ValueError('mesh-only run has no solved flow to export')
        result=export_case(out/case['profile'],original,out/'geometry')
        exports.append(dict(profile=case['profile'],metadata=result))
        print(case['profile'],'export',result['selected_cells'],'cells')
    dump(out/'export.config.json',original['output'])
    history_file=out/'export.history.json'
    history=json.loads(history_file.read_text()) if history_file.exists() else []
    history.append(dict(timestamp=datetime.now().isoformat(),options=original['output'],cases=exports,solver_invoked=False))
    dump(history_file,history)
    return 0


def output_options(p):
    p.add_argument('--fields',nargs='+',choices=sorted(cfg.FIELDS))
    p.add_argument('--formats',nargs='+',choices=['npz','csv'])
    p.add_argument('--region',nargs=4,type=float,metavar=('XMIN','XMAX','YMIN','YMAX'),help='centroid selection bounds, in chord units')
    p.add_argument('--surface',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--loads',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--volume',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--load-fields',nargs='+',choices=sorted(cfg.LOAD_FIELDS))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build',help='standalone CMake build');p.add_argument('--cpu',action='store_true');p.add_argument('--jobs',type=int,default=8);p.add_argument('--cuda-arch',default='89',help='CMake CUDA architectures; default 89 for RTX4090')
    p=sub.add_parser('init-config',help='write an editable complete configuration');p.add_argument('--out',required=True)
    for command in ['run','mesh']:
        p=sub.add_parser(command,help='both LF1 and LF2 by default')
        p.add_argument('--points',required=True);p.add_argument('--config');p.add_argument('--out')
        p.add_argument('--executable',help='explicit solver executable, for independent builds')
        p.add_argument('--point-format',choices=['gxxx','polygon'],default='gxxx',help='polygon requires an ordered x,y,patch CSV')
        p.add_argument('--levels',nargs='+',choices=['LF1','LF2'],default=['LF1','LF2'])
        for flag in ['mach','aoa','pressure','temperature','chord','cfl','dt']:
            p.add_argument('--'+flag,type=float)
        p.add_argument('--stop-time',type=float,help='nondimensional t*=t a_inf/chord cap')
        p.add_argument('--physical-time',type=float,help='physical time cap in seconds')
        p.add_argument('--mode',choices=['steady','time'])
        p.add_argument('--backend',choices=['cuda','cpu'])
        p.add_argument('--turbulence',choices=['none','sa-neg']);p.add_argument('--viscosity',choices=['constant','sutherland'])
        output_options(p)
    p=sub.add_parser('export',help='reselect exports from an existing run, without solving')
    p.add_argument('--run',required=True);p.add_argument('--config');output_options(p)
    args=parser.parse_args()
    if args.command=='build':build(args);return 0
    if args.command=='init-config':
        target=Path(args.out)
        if target.exists():raise ValueError('config already exists; choose another path')
        target.parent.mkdir(parents=True,exist_ok=True);dump(target,cfg.load());print(target);return 0
    return export(args) if args.command=='export' else run(args)


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,subprocess.CalledProcessError) as error:
        print('ERROR:',error,file=sys.stderr)
        sys.exit(2)
