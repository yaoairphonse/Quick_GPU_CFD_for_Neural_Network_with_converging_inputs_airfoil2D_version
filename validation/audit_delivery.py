#!/usr/bin/env python3
"""Independent checks of packaged references, steady gates, SI exports and provenance."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from pipeline import configuration as cfg

def read(p):return json.loads(Path(p).read_text())
def main():
    provenance=read(R/'CORE_PROVENANCE.json')
    for name,sha in provenance['files_sha256'].items():
        assert hashlib.sha256((R/'src'/name).read_bytes()).hexdigest()==sha,name
    checks=read(R/'validation/ten_short_checks.json');assert checks['passed'] and checks['completed']==20
    assert read(R/'validation/controls_checks.json')['passed']
    rows=[]
    suite=read(R/'runs/G0001_default/pipeline.json')
    for row in suite['cases']:
        label=row['profile'];folder=R/'runs/G0001_default'/label;raw=folder/'internal/flow'
        prefix=(R/'validation/default_reference_LF1/flow' if label=='LF1' else R/'validation/reference_archive/LF2_flow')
        a=np.load(str(prefix)+'.final.state.npy');b=np.load(str(raw)+'.final.state.npy')
        old=read(str(prefix)+'.result.json');new=read(folder/'result.json')
        l1=float(np.abs(a-b).sum()/np.abs(a).sum());linf=float(np.abs(a-b).max()/max(1,np.abs(a).max()));sa=float(np.abs(a[4]-b[4]).max()/np.abs(a[4]).max())
        assert l1<=2e-11 and linf<=2e-10 and sa<=2e-10
        assert all(old[k]==new[k] for k in ['steps','physical_time','rejected']) and new['rejected']==0
        assert new['converged'] and new['hold']==5
        loads=read(folder/'output/loads.json')
        for key,coefficient in [('CL','Cl'),('CD','Cd'),('CM_z','Cm')]:
            assert abs(loads['total'][key]-new[coefficient])<=2e-11
            assert abs(old[coefficient]-new[coefficient])/max(.01,abs(old[coefficient]))<=2e-10
        history=np.genfromtxt(str(raw)+'.history.csv',delimiter=',',names=True)[-5:]
        assert all(np.all(history[k]<=.001) for k in ['res_rho','res_mx','res_my','res_E','res_sa'])
        assert all(np.all(history[k]<=.005) for k in ['delta_Cl','delta_Cd','delta_Cm'])
        assert np.all(history['time']>=3) and history[-1]['hold']==5
        flat=np.load(folder/'output/aerodynamics.npz')
        assert set(flat.files)==set(cfg.DEFAULT_LOAD_FIELDS)
        for field,part,key in [('fx','total','force_x'),('fy','total','force_y'),('drag','total','drag'),('lift','total','lift'),('viscous_force_x','viscous','force_x'),('viscous_force_y','viscous','force_y'),('viscous_drag','viscous','drag'),('viscous_lift','viscous','lift'),('moment_z','total','moment_z')]:
            assert flat[field].shape==(1,) and flat[field][0]==loads[part][key]
        metadata=read(folder/'output/metadata.json')
        assert metadata['positivity_passed'] and metadata['surface_load_integration_passed'] and metadata['SI_EOS_relative_error']<=2e-12
        # Independently sum the exported per-fragment force and moment columns.
        surface=np.load(folder/'output/surface.npz')
        for axis in ['x','y']:
            value=(surface['pressure_force_'+axis+'_N_per_m']+surface['viscous_force_'+axis+'_N_per_m']).sum()
            assert abs(value-loads['total']['force_'+axis])<=2e-8
        assert abs(surface['moment_z_N_m_per_m'].sum()-loads['total']['moment_z'])<=2e-8
        rows.append(dict(profile=label,passed=True,cells=new['fluid_control_volumes'],steps=new['steps'],physical_time_star=new['physical_time'],relative_L1=l1,relative_Linf=linf,SA_relative_Linf=sa,
                         mesh_seconds=new['mesh_setup_seconds'],GPU_event_seconds=new['gpu']['advance_cuda_event_seconds'],advance_wall_seconds=new['gpu']['advance_wall_seconds'],process_seconds=new['process_seconds'],export_seconds=new['export']['seconds'],Cl=new['Cl'],Cd=new['Cd'],Cm=new['Cm']))
    for label in ['LF1','LF2']:
        out=R/'validation/demo_short'/label/'output'
        table=np.load(out/'field.npz');assert set(table.files)=={'cell_id','fluid_area_m2','x','y','p','mach'}
        flat=np.load(out/'aerodynamics.npz');assert set(flat.files)=={'drag','lift','fx','fy','viscous_force_x','viscous_force_y','viscous_drag','viscous_lift','moment_z'}
        assert not (out/'field.csv').exists() and (out/'probes.npz').exists()
    cpu=read(R/'validation/cpu_standalone/LF1/result.json');assert cpu['rejected']==0 and not cpu['failed']
    assert 'cuda' not in (R/'validation/cpu_linkage.log').read_text().lower()
    for label in ['LF1','LF2']:
        mesh=read(R/'validation/mesh_only'/label/'result.json');assert mesh['mode']=='mesh' and mesh['steps']==0 and mesh['gpu']['advance_cuda_event_seconds']==0
    summary=dict(passed=True,core_hashes_unchanged=True,short_cases=20,short_max_relative_Linf=max(x['relative_Linf'] for x in checks['cases']),controls=read(R/'validation/controls_checks.json'),full_steady=rows,
                 default_pipeline_seconds=suite['pipeline_seconds'],geometry_preparation_seconds=suite['geometry_preparation_seconds'],reexport_without_solver_passed=True,independent_CPU_build_passed=True,mesh_only_passed=True,
                 physical_scope='mechanical SA-neg/laminar, kappa=0; coarse high-Re qualitative tool, not wall-resolved certification')
    (R/'validation/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
