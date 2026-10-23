from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parents[1]
cases={
 'subsonic_outlet':dict(freestream=dict(Mach=.7,AOA_degrees=3.,p_Pa=30000.,T_K=250.,chord_m=.8),physics=dict(turbulence='none',viscosity='constant',constant_mu_Pa_s=1.7e-5),boundaries=dict(xmax='pressure-outlet',outlet_pressure_Pa=30000.),solver=dict(mode='time',stop_time_star=.01),output=dict(fields=['x','y','p','rho','u','v','mach','temperature'],region_xy_c=[-.5,2,-.5,.5],sample_points_xy_c=[[1.2,0]])),
 'fixed_dt':dict(solver=dict(mode='time',stop_time_star=.01,time_step=dict(mode='fixed',dt_seconds=1e-7))),
 'cpu_ausm':dict(physics=dict(turbulence='none',viscosity='sutherland'),solver=dict(backend='cpu',flux='ausm',mode='time',stop_time_star=.01,CFL=.1))}
rows=[]
for name,config in cases.items():
 path=R/'validation'/('control_'+name+'.json');path.write_text(json.dumps(config,indent=2)+'\n')
 out=R/'validation'/('control_'+name)
 cmd=[sys.executable,str(R/'quick_cfd.py'),'run','--points',str(R/'examples/G0001_airfoil_points.csv'),'--config',str(path),'--levels','LF2','--out',str(out)]
 with (R/'validation'/('control_'+name+'.log')).open('w') as log:z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert z.returncode==0,(name,z.returncode)
 result=json.loads((out/'LF2/result.json').read_text());meta=json.loads((out/'LF2/output/metadata.json').read_text())
 assert not result['failed'] and result['rejected']==0 and result['physical_time']==.01 and not result['converged']
 assert meta['positivity_passed'] and meta['SI_EOS_relative_error']<=2e-12 and meta['surface_load_integration_passed']
 row=dict(name=name,passed=True,steps=result['steps'],physical_time=result['physical_time'],SI_EOS_relative_error=meta['SI_EOS_relative_error'],resolved_config=str(out/'config.resolved.json'))
 rows.append(row);print(row,flush=True)
(R/'validation/controls_checks.json').write_text(json.dumps(dict(passed=True,cases=rows),indent=2)+'\n')
