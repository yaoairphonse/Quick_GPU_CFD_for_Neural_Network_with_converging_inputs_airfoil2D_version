#!/usr/bin/env python3
"""Reproduce the registered 10-shape/two-mesh wrapper equivalence audit."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.geometry import prepare
from pipeline import configuration as cfg
D=ROOT/'validation/ten_short'
EXE=ROOT/'build/baseline_airfoil_suite'

def main():
    checks=[]
    for source in sorted((ROOT/'examples').glob('G*_airfoil_points.csv')):
        name=source.stem.split('_')[0]
        dest=D/name
        cmd=[sys.executable,str(ROOT/'quick_cfd.py'),'run','--points',str(source),'--mode','time','--stop-time','.01','--out',str(dest)]
        with (ROOT/'validation'/('short_'+name+'.log')).open('w') as log:
            z=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        assert z.returncode==0,(name,z.returncode)
        # The frozen legacy driver takes the old geometry/test10 path layout.
        refroot=D/'reference_inputs';poly=refroot/'geometry/test10';poly.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dest/'geometry/airfoil.csv',poly/(name+'.csv'))
        shutil.copy2(dest/'geometry/patches.csv',poly/(name+'.patches.csv'))
        summary=json.loads((dest/'pipeline.json').read_text())
        for label,mesh in [('LF1','coarse'),('LF2','fine')]:
            prefix=dest/label/'reference';new=dest/label/'internal/flow'
            args=[str(EXE),'--root',str(refroot),'--case',name,'--out',str(prefix),'--mode','smoke','--backend','cuda','--mesh',mesh,'--viscosity','sutherland','--turbulence','sa-neg','--sa-layout','fused']
            (dest/label/'reference.command.json').write_text(json.dumps(args,indent=2)+'\n')
            with Path(str(prefix)+'.log').open('w') as log:
                z=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'})
            assert z.returncode==0,(name,label)
            a=np.load(str(prefix)+'.final.state.npy');b=np.load(str(new)+'.final.state.npy')
            l1=float(np.abs(a-b).sum()/np.abs(a).sum());linf=float(np.abs(a-b).max()/max(1,np.abs(a).max()))
            sa=float(np.abs(a[4]-b[4]).max()/np.abs(a[4]).max())
            ra=json.loads(Path(str(prefix)+'.result.json').read_text());rb=json.loads(Path(str(new)+'.result.json').read_text())
            counters=all(ra[k]==rb[k] for k in ['steps','physical_time','rejected'])
            geom=all(np.array_equal(np.load(str(prefix)+'.'+key+'.npy'),np.load(str(new)+'.'+key+'.npy')) for key in ['cells','faces','walls','agglomeration'])
            loads=max(abs(ra[k]-rb[k])/max(.01,abs(ra[k])) for k in ['Cl','Cd','Cm'])
            passed=l1<=2e-11 and linf<=2e-10 and sa<=2e-10 and loads<=2e-10 and counters and geom and rb['rejected']==0
            row=dict(case=name,profile=label,passed=bool(passed),relative_L1=l1,relative_Linf=linf,SA_relative_Linf=sa,loads_relative_max=loads,counters_equal=counters,geometry_exact=geom,steps=rb['steps'],cells=rb['fluid_control_volumes'])
            checks.append(row)
            (ROOT/'validation/ten_short_checks.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in checks),completed=len(checks),cases=checks),indent=2)+'\n')
            print(name,label,passed,'CV',row['cells'],'SA',sa,flush=True)
            assert passed,row
    assert len(checks)==20
if __name__=='__main__':main()
