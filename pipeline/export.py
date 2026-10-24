"""Selectable field/surface exports and body-force integration in SI units."""
from pathlib import Path
import json
import math
import time
import numpy as np
from .configuration import scales, DEFAULT_LOAD_FIELDS

UNITS = dict(x='m', y='m', p='Pa', rho='kg/m^3', u='m/s', v='m/s', mach='1', temperature='K')


def write_table(prefix, data, formats):
    prefix = Path(prefix)
    if 'npz' in formats:
        np.savez(str(prefix)+'.npz', **data)
    if 'csv' in formats:
        np.savetxt(str(prefix)+'.csv', np.column_stack(list(data.values())), delimiter=',',
                   header=','.join(data), comments='', fmt='%.17g')


def field_data(raw, config):
    a = scales(config)
    U = np.load(str(raw)+'.final.state.npy')
    cells = np.load(str(raw)+'.cells.npy')
    rho = U[0]
    u, v = U[1]/rho, U[2]/rho
    p = (config['freestream']['gamma']-1)*(U[3]-.5*rho*(u*u+v*v))
    if not np.isfinite(U).all() or np.any(rho <= 0) or np.any(p <= 0):
        raise ValueError('nonphysical or nonfinite state: exports are not certified')
    data = dict(x=cells[:,9]*a['length'], y=cells[:,10]*a['length'],
                p=p*a['pressure'], rho=rho*a['rho'], u=u*a['velocity'], v=v*a['velocity'],
                mach=np.hypot(u,v)/np.sqrt(config['freestream']['gamma']*p/rho),
                temperature=p/rho*a['temperature'])
    return data, cells


def body_loads(raw, config):
    a = scales(config)
    loads = json.loads(Path(str(raw)+'.final.viscous_loads.json').read_text())
    angle = math.radians(config['freestream']['AOA_degrees'])
    co, si = math.cos(angle), math.sin(angle)
    xr, yr = config['solver']['moment_reference_xy_c']
    def part(fx, fy, mz):
        moment = mz-xr*fy+yr*fx
        force_scale = a['pressure']*a['length']
        return dict(force_x=fx*force_scale, force_y=fy*force_scale,
                    drag=(co*fx+si*fy)*force_scale, lift=(-si*fx+co*fy)*force_scale,
                    moment_z=moment*force_scale*a['length'],
                    CD=(co*fx+si*fy)*a['pressure']/a['q_inf'],
                    CL=(-si*fx+co*fy)*a['pressure']/a['q_inf'], CM_z=moment*a['pressure']/a['q_inf'])
    def split(row):
        row=np.asarray(row)
        return dict(pressure=part(*row[:3]), viscous=part(*row[3:]), total=part(*(row[:3]+row[3:])))
    result = split(loads['total'])
    result['patches'] = {str(row['patch']):split(row['loads']) for row in loads['patches']}
    result['units'] = dict(force_x='N/m span', force_y='N/m span', drag='N/m span', lift='N/m span',
                           moment_z='N m/m span', CD='1', CL='1', CM_z='1')
    result['coordinates'] = 'x-y plane; positive Mz is counterclockwise; drag along freestream, lift rotated +90 degrees'
    result['moment_reference_m'] = [xr*a['length'], yr*a['length']]
    result['integration'] = 'all body wall fragments; pressure and viscous parts; independent of field ROI and selected surface patches; absolute static pressure'
    result['q_inf_Pa'] = a['q_inf']
    return result


def surface_data(raw, config, field):
    a = scales(config)
    walls = np.load(str(raw)+'.walls.npy')
    stress = np.load(str(raw)+'.final.wall_stress.npy')
    owner = walls[:,0].astype(int)
    nw = len(walls)
    x, y = .5*(walls[:,4]+walls[:,6])*a['length'], .5*(walls[:,5]+walls[:,7])*a['length']
    data = dict(wall_id=np.arange(nw), patch_id=walls[:,3], source_edge_id=walls[:,1], owner_cell=owner,
                donor_cell=stress[:,2], nx=walls[:,8], ny=walls[:,9], length_m=walls[:,10]*a['length'])
    data['p_owner_Pa'] = field['p'][owner]
    data['p_wall_Pa'] = stress[:,8]*a['pressure']
    options = dict(x=x, y=y, p_wall_Pa=data['p_wall_Pa'], rho_owner_kg_m3=field['rho'][owner],
                   u_wall_m_s=np.zeros(nw), v_wall_m_s=np.zeros(nw), mach_wall=np.zeros(nw),
                   temperature_owner_K=field['temperature'][owner])
    names = dict(x='x', y='y', p='p_wall_Pa', rho='rho_owner_kg_m3', u='u_wall_m_s', v='v_wall_m_s',
                 mach='mach_wall', temperature='temperature_owner_K')
    for key in config['output']['fields']:
        data[names[key]] = options[names[key]]
    data['Cp'] = (data['p_wall_Pa']-config['freestream']['p_Pa'])/a['q_inf']
    data['mu_donor_Pa_s'] = stress[:,4]*a['viscosity']
    data['normal_distance_m'] = stress[:,3]*a['length']
    data['wall_shear_signed_Pa'] = stress[:,7]*a['pressure']
    scale = a['pressure']*a['length']
    for label, cx, cy in [('pressure',9,10), ('viscous',11,12)]:
        data[label+'_force_x_N_per_m'] = stress[:,cx]*scale
        data[label+'_force_y_N_per_m'] = stress[:,cy]*scale
    fx = (stress[:,9]+stress[:,11])*scale
    fy = (stress[:,10]+stress[:,12])*scale
    xr,yr = np.asarray(config['solver']['moment_reference_xy_c'])*a['length']
    data['moment_z_N_m_per_m'] = (x-xr)*fy-(y-yr)*fx
    mask = np.isin(walls[:,3],config['output']['surface_patches'])
    return {key:value[mask] for key,value in data.items()}


def probes(raw, config, field, cells, polygon):
    points = config['output']['sample_points_xy_c']
    if not points:
        return None
    b = config['mesh']['bounds_xy_c']
    chosen=[]; distances=[]
    for px, py in points:
        if not (b[0] < px < b[1] and b[2] < py < b[3]):
            raise ValueError('probe point is outside the open computational domain')
        inside=False; near=False
        for left,right in zip(polygon,np.roll(polygon,1,axis=0)):
            delta=right-left
            t=np.clip(np.dot(np.array([px,py])-left,delta)/np.dot(delta,delta),0,1)
            near |= np.linalg.norm(np.array([px,py])-left-t*delta) < 1e-12
            if (left[1]>py)!=(right[1]>py):
                inside ^= px < (right[0]-left[0])*(py-left[1])/(right[1]-left[1])+left[0]
        if inside or near:
            raise ValueError('probe point lies inside/on the airfoil; use surface output for wall data')
        distance=np.hypot(cells[:,9]-px,cells[:,10]-py)
        k=int(np.argmin(distance));chosen.append(k);distances.append(distance[k])
    a=scales(config)
    return dict(query_x_m=np.asarray(points)[:,0]*a['length'],query_y_m=np.asarray(points)[:,1]*a['length'],
                nearest_cell=np.asarray(chosen),distance_m=np.asarray(distances)*a['length'],
                **{key:field[key][chosen] for key in config['output']['fields']})


def mesh_svg(raw, config, polygon, output):
    members=np.load(str(raw)+'.agglomeration.npy')
    # Crop around the airfoil; member rectangles show mesh layout, not exact cut polygons.
    x0,x1,y0,y1=-.6,2.5,-.65,.65
    rects=[]
    for row in members:
        if row[3]<x0 or row[2]>x1 or row[5]<y0 or row[4]>y1:continue
        rects.append(f'<rect x="{row[2]}" y="{-row[5]}" width="{row[3]-row[2]}" height="{row[5]-row[4]}"/>')
    points=' '.join(f'{x},{-y}' for x,y in polygon)
    output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {-y1} {x1-x0} {y1-y0}"><rect x="{x0}" y="{-y1}" width="{x1-x0}" height="{y1-y0}" fill="white"/><g fill="none" stroke="#6d9eb0" stroke-width=".0009">'+''.join(rects)+f'</g><polygon points="{points}" fill="white" stroke="#173745" stroke-width=".002"/></svg>')


def export_case(folder, config, geometry_folder):
    start = time.perf_counter()
    folder = Path(folder)
    raw, out = folder/'internal/flow', folder/'output'
    field, cells = field_data(raw, config)
    options = config['output']
    polygon = np.loadtxt(Path(geometry_folder)/'airfoil.csv', delimiter=',', skiprows=1)
    mask = np.ones(len(cells), dtype=bool)
    if options['region_xy_c'] is not None:
        x0,x1,y0,y1 = options['region_xy_c']
        mask = (cells[:,9]>=x0)&(cells[:,9]<=x1)&(cells[:,10]>=y0)&(cells[:,10]<=y1)
    a = scales(config)
    selected = dict(cell_id=np.arange(len(cells))[mask], fluid_area_m2=cells[mask,8]*a['length']**2,
                    **{key:field[key][mask] for key in options['fields']})
    sampled = probes(raw, config, field, cells, polygon)
    surface = surface_data(raw, config, field) if options['surface'] else None
    loads = body_loads(raw, config)
    # Certify all fragments, even when the selected surface contains a subset.
    stress = np.load(str(raw)+'.final.wall_stress.npy')
    walls = np.load(str(raw)+'.walls.npy')
    fx = (stress[:,9]+stress[:,11])*a['pressure']*a['length']
    fy = (stress[:,10]+stress[:,12])*a['pressure']*a['length']
    xr,yr = np.asarray(config['solver']['moment_reference_xy_c'])*a['length']
    moment = (.5*(walls[:,4]+walls[:,6])*a['length']-xr)*fy-(.5*(walls[:,5]+walls[:,7])*a['length']-yr)*fx
    reconstructed = [fx.sum(),fy.sum(),moment.sum()]
    certified = [loads['total'][key] for key in ['force_x','force_y','moment_z']]
    if not np.allclose(reconstructed,certified,rtol=2e-11,atol=2e-8):
        raise ValueError('surface-to-load integration mismatch')
    eos = float(np.max(np.abs(field['p']-field['rho']*config['freestream']['R_J_kgK']*field['temperature'])/field['p']))
    if eos > 2e-12:
        raise ValueError('SI equation-of-state mismatch')
    out.mkdir(exist_ok=True)
    # Only regenerate these product-owned files, keeping internal solver data intact.
    for stem in ['field','surface','probes','loads','aerodynamics']:
        for suffix in ['csv','npz','json']:
            file = out/(stem+'.'+suffix)
            if file.exists():
                file.unlink()
    if options.get('volume',True):
        write_table(out/'field', selected, options['formats'])
    if surface is not None:
        write_table(out/'surface', surface, options['formats'])
    if sampled is not None:
        write_table(out/'probes', sampled, options['formats'])
    if options['loads']:
        (out/'loads.json').write_text(json.dumps(loads,indent=2)+'\n')
        flat = {key:loads['total'][value] for key,value in dict(fx='force_x',fy='force_y',drag='drag',lift='lift',moment_z='moment_z',moment='moment_z',CD='CD',CL='CL',CM_z='CM_z').items()}
        for part in ['pressure','viscous']:
            for key in ['force_x','force_y','drag','lift','moment_z']:
                flat[part+'_'+key] = loads[part][key]
        write_table(out/'aerodynamics', {key:np.asarray([flat[key]]) for key in options.get('load_fields',DEFAULT_LOAD_FIELDS)}, options['formats'])
        rows = [[loads[part][key] for key in ['force_x','force_y','drag','lift','moment_z','CD','CL','CM_z']] for part in ['pressure','viscous','total']]
        if 'csv' in options['formats']:
            (out/'loads.csv').write_text('component,force_x_N_per_m,force_y_N_per_m,drag_N_per_m,lift_N_per_m,moment_z_N_m_per_m,CD,CL,CM_z\n'+''.join(part+','+','.join(format(v,'.17g') for v in row)+'\n' for part,row in zip(['pressure','viscous','total'],rows)))
    mesh_svg(raw, config, polygon, out/'mesh.svg')
    metadata = dict(fields=options['fields'], volume_output=options.get('volume',True), load_fields=options.get('load_fields',DEFAULT_LOAD_FIELDS), load_units='forces N/m span; moments N m/m span; CL/CD/CM dimensionless', units={key:UNITS[key] for key in options['fields']},
                    selected_cells=int(mask.sum()), total_cells=len(cells), region_xy_c=options['region_xy_c'],
                    selection='fluid control-volume centroid inside inclusive region; cell-average value',
                    surface_sampling='p_wall and Cp from the pressure used by wall flux/loads; p_owner, rho, temperature from owner cell; u=v=Mach=0 at stationary no-slip wall',
                    probes='nearest fluid-control-volume centroid, no interpolation; requested points inside/on body are rejected',
                    patch_markers={'1':'upper','2':'lower','3':'vertical trailing edge'},
                    load_scope='all airfoil fragments; independent of ROI/selected surface patches',
                    positivity_passed=True, SI_EOS_relative_error=eos, surface_load_integration_passed=True,
                    seconds=time.perf_counter()-start)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata
