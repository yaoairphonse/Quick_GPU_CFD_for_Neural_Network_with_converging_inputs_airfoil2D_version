"""JSON configuration and conversion from physical to solver units."""
from copy import deepcopy
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {'x', 'y', 'p', 'rho', 'u', 'v', 'mach', 'temperature'}
DEFAULT_LOAD_FIELDS = ['drag', 'lift', 'fx', 'fy', 'pressure_force_x', 'pressure_force_y', 'pressure_drag', 'pressure_lift', 'viscous_force_x', 'viscous_force_y', 'viscous_drag', 'viscous_lift', 'moment_z', 'pressure_moment_z', 'viscous_moment_z', 'CD', 'CL', 'CM_z']
LOAD_FIELDS = set(DEFAULT_LOAD_FIELDS) | {'moment'}


def merge(base, patch, path=''):
    if not isinstance(patch, dict):
        raise ValueError(f'{path or "config"} must be an object')
    for key, value in patch.items():
        if key not in base:
            raise ValueError(f'unknown setting: {path}{key}')
        if isinstance(base[key], dict):
            merge(base[key], value, f'{path}{key}.')
        else:
            base[key] = value
    return base


def load(path=None):
    config = json.loads((ROOT / 'config/default.json').read_text())
    if path:
        merge(config, json.loads(Path(path).read_text(encoding='utf-8-sig')))
    return config


def positive(value, name, maximum=None, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{name} must be numeric')
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum}')


def integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')


def bounds(value, name):
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value):
        raise ValueError(f'{name} requires [xmin,xmax,ymin,ymax]')
    if value[0] >= value[1] or value[2] >= value[3]:
        raise ValueError(f'{name} bounds must have positive width/height')


def validate(c):
    f, p, s, m, o = (c[k] for k in ['freestream', 'physics', 'solver', 'mesh', 'output'])
    # The public airfoil driver remains the dedicated Cartesian-AMR workflow.
    # The separate 3D hybrid workflow has its own geometry and table schema.
    m.setdefault('method', 'cartesian_amr_2d')
    if m['method'] != 'cartesian_amr_2d':
        raise ValueError('this driver requires mesh.method=cartesian_amr_2d')
    # Older saved runs predate the optional steady-development controls.
    s.setdefault('preconditioning','none');s.setdefault('mach_floor',.02);s.setdefault('local_pseudo_time',False)
    s.setdefault('spatial_order',1)
    s.setdefault('adaptive_cfl',dict(enabled=False,minimum=.03,maximum=.45,growth=1.1,shrink=.5))
    for k in ['Mach', 'p_Pa', 'T_K', 'chord_m', 'R_J_kgK']:
        positive(f[k], 'freestream.' + k)
    positive(f['gamma'] - 1, 'gamma - 1')
    if not isinstance(f['AOA_degrees'], (int, float)) or not math.isfinite(f['AOA_degrees']):
        raise ValueError('AOA_degrees must be finite')
    if p['turbulence'] not in ['none', 'sa-neg'] or p['viscosity'] not in ['constant', 'sutherland']:
        raise ValueError('physics: turbulence none|sa-neg; viscosity constant|sutherland')
    for k in ['mu0_Pa_s', 'T0_K']:
        positive(p['sutherland'][k], k)
    positive(p['sutherland']['S_K'], 'S_K', allow_zero=True)
    if p['constant_mu_Pa_s'] is not None:
        positive(p['constant_mu_Pa_s'], 'constant_mu_Pa_s')
    positive(p['sa_nu_inf_over_nu'], 'sa_nu_inf_over_nu')
    positive(p['wall_probe_min_h'], 'wall_probe_min_h', 2, True)
    if s['backend'] not in ['cpu', 'cuda'] or s['flux'] not in ['hllc', 'ausm'] or s['sa_layout'] not in ['reference', 'fused'] or s['mode'] not in ['steady', 'time']:
        raise ValueError('invalid backend, flux, sa_layout, or solver mode')
    if s['preconditioning'] not in ['none', 'turkel']:
        raise ValueError('solver.preconditioning must be none|turkel')
    integer(s['spatial_order'], 'solver.spatial_order', 1, 2)
    positive(s['mach_floor'], 'solver.mach_floor', 1)
    ac=s['adaptive_cfl']
    if not isinstance(s['local_pseudo_time'],bool) or not isinstance(ac['enabled'],bool):
        raise ValueError('local_pseudo_time and adaptive_cfl.enabled must be booleans')
    for key in ['minimum','maximum']:positive(ac[key],'adaptive_cfl.'+key,.5)
    if (ac['enabled'] and (ac['minimum']>s['CFL'] or ac['maximum']<s['CFL'])) or ac['growth']<=1 or not 0<ac['shrink']<1:
        raise ValueError('invalid adaptive CFL bounds or factors')
    if (s['local_pseudo_time'] or ac['enabled']) and (s['mode']!='steady' or s['time_step']['mode']!='cfl'):
        raise ValueError('local/adaptive stepping is restricted to steady pseudo-time with automatic CFL')
    if s['preconditioning'] != 'none' and (s['mode'] != 'steady' or s['time_step']['mode'] != 'cfl'):
        raise ValueError('preconditioning requires steady pseudo-time and automatic CFL')
    for k, maximum in [('CFL', .5), ('viscous_CFL', .2), ('SA_CFL', .1)]:
        positive(s[k], k, maximum)
    positive(s['stop_time_star'], 'stop_time_star')
    positive(s['budget_seconds'] - 1, 'budget_seconds - 1')
    ts = s['time_step']
    if ts['mode'] not in ['cfl', 'fixed']:
        raise ValueError('time_step.mode must be cfl|fixed')
    if ts['mode'] == 'fixed':
        positive(ts['dt_seconds'], 'fixed dt_seconds')
    elif ts['dt_seconds'] is not None:
        raise ValueError('use null dt_seconds in CFL mode')
    cv = s['convergence']
    cv.setdefault('metric','all_loads');cv.setdefault('cp_change',.005)
    if cv['metric'] not in ['all_loads','cl_cm_cp']:
        raise ValueError('convergence.metric must be all_loads|cl_cm_cp')
    positive(cv['cp_change'], 'convergence.cp_change')
    for k in ['raw_tolerance', 'coefficient_change', 'check_period_star']:
        positive(cv[k], 'convergence.' + k)
    if cv['check_period_star'] <= .01:
        raise ValueError('check_period_star must exceed the initial .01 diagnostic time')
    positive(cv['min_time_star'], 'min_time_star', allow_zero=True)
    integer(cv['hold'], 'hold', 1, 10000)
    bounds(cv['residual_region_xy_c'], 'residual_region_xy_c')
    if len(s['moment_reference_xy_c']) != 2 or not all(math.isfinite(x) for x in s['moment_reference_xy_c']):
        raise ValueError('moment_reference_xy_c must have two finite numbers')
    bounds(m['bounds_xy_c'], 'mesh.bounds_xy_c')
    for n in m['base_shape']:
        integer(n, 'base_shape', 2, 4096)
    if len(m['base_shape']) != 2:
        raise ValueError('base_shape requires [nx,ny]')
    integer(m['wall_buffer'], 'wall_buffer', 2, 100)
    integer(m['max_cells'], 'max_cells', 100, 20000000)
    for label, profile in m['profiles'].items():
        integer(profile['levels'], label + '.levels', 0, 10)
        for box in profile['boxes']:
            if len(box) != 5:
                raise ValueError('box requires [xmin,xmax,ymin,ymax,level]')
            bounds(box[:4], label + '.box')
            integer(box[4], label + '.box.level', 0, profile['levels'])
    for side in ['xmin', 'xmax', 'ymin', 'ymax']:
        if c['boundaries'][side] not in ['farfield', 'pressure-outlet', 'transmissive']:
            raise ValueError('boundary must be farfield|pressure-outlet|transmissive')
    positive(c['boundaries']['outlet_pressure_Pa'], 'outlet_pressure_Pa')
    if not o['fields'] or not set(o['fields']) <= FIELDS or len(o['fields']) != len(set(o['fields'])):
        raise ValueError('output.fields must contain distinct supported field names')
    if not o.get('load_fields', DEFAULT_LOAD_FIELDS) or not set(o.get('load_fields', DEFAULT_LOAD_FIELDS)) <= LOAD_FIELDS:
        raise ValueError('unsupported output.load_fields')
    if not o['formats'] or not set(o['formats']) <= {'csv', 'npz'}:
        raise ValueError('output.formats must contain csv and/or npz')
    if o['region_xy_c'] is not None:
        bounds(o['region_xy_c'], 'output.region_xy_c')
    if not set(o['surface_patches']) <= {1, 2, 3}:
        raise ValueError('surface_patches must use 1 upper, 2 lower, 3 trailing edge')
    if not isinstance(o.get('volume',True),bool) or not isinstance(o['surface'], bool) or not isinstance(o['loads'], bool):
        raise ValueError('output.volume/surface/loads must be booleans')
    for point in o['sample_points_xy_c']:
        if len(point) != 2 or not all(math.isfinite(x) for x in point):
            raise ValueError('sample point requires two finite coordinates')
    return c


def scales(c):
    f, p = c['freestream'], c['physics']
    rho = f['p_Pa'] / (f['R_J_kgK'] * f['T_K'])
    a = math.sqrt(f['gamma'] * f['R_J_kgK'] * f['T_K'])
    law = p['sutherland']
    mu = law['mu0_Pa_s'] * (f['T_K'] / law['T0_K']) ** 1.5 * (law['T0_K'] + law['S_K']) / (f['T_K'] + law['S_K'])
    if p['viscosity'] == 'constant' and p['constant_mu_Pa_s'] is not None:
        mu = p['constant_mu_Pa_s']
    return dict(rho=rho, velocity=a, length=f['chord_m'], pressure=rho*a*a,
                temperature=f['gamma']*f['T_K'], time=f['chord_m']/a,
                viscosity=rho*a*f['chord_m'], mu_inf=mu,
                q_inf=.5*rho*(f['Mach']*a)**2, Re_chord=rho*f['Mach']*a*f['chord_m']/mu)


def settings(c, label, name, mode=None):
    f, p, s, m = (c[k] for k in ['freestream', 'physics', 'solver', 'mesh'])
    scale = scales(c)
    profile = m['profiles'][label]
    cv = s['convergence']
    mu = scale['mu_inf'] / scale['viscosity']
    b = m['bounds_xy_c']
    r = cv['residual_region_xy_c']
    out = dict(name=name, mode=mode or s['mode'], backend=s['backend'], mesh_preset=label,
               preconditioning=s['preconditioning'], mach_floor=s['mach_floor'],
               local_pseudo_time=int(s['local_pseudo_time']),adaptive_cfl=int(s['adaptive_cfl']['enabled']),
               cfl_min=s['adaptive_cfl']['minimum'],cfl_max=s['adaptive_cfl']['maximum'],
               cfl_growth=s['adaptive_cfl']['growth'],cfl_shrink=s['adaptive_cfl']['shrink'],
               viscosity=p['viscosity'], turbulence=p['turbulence'], sa_layout=s['sa_layout'],
               stop_time=s['stop_time_star'], budget=s['budget_seconds'], mu_star=mu,
               mach=f['Mach'], aoa=f['AOA_degrees'], gamma=f['gamma'], events=1,
               nx=m['base_shape'][0], ny=m['base_shape'][1], levels=profile['levels'],
               buffer=m['wall_buffer'], max_cells=m['max_cells'], xmin=b[0], xmax=b[1], ymin=b[2], ymax=b[3],
               cfl=s['CFL'], spatial_order=s['spatial_order'], fixed_dt=(s['time_step']['dt_seconds']/scale['time'] if s['time_step']['mode']=='fixed' else 0),
               flux=s['flux'], moment_x=s['moment_reference_xy_c'][0], moment_y=s['moment_reference_xy_c'][1],
               raw_tol=cv['raw_tolerance'], load_tol=cv['coefficient_change'], cp_tol=cv['cp_change'], convergence_metric=cv['metric'], hold=cv['hold'],
               tmin=cv['min_time_star'], period=cv['check_period_star'],
               res_xmin=r[0], res_xmax=r[1], res_ymin=r[2], res_ymax=r[3],
               outlet_p_star=c['boundaries']['outlet_pressure_Pa']/scale['pressure'],
               sa_nu_inf=p['sa_nu_inf_over_nu']*mu, sutherland_star=p['sutherland']['S_K']/scale['temperature'],
               viscous_cfl=s['viscous_CFL'], sa_cfl=s['SA_CFL'], wall_probe_min_h=p['wall_probe_min_h'])
    for index, side in enumerate(['xmin', 'xmax', 'ymin', 'ymax']):
        out['bc'+str(index)] = c['boundaries'][side]
    return '\n'.join(f'{k} {v}' for k, v in out.items()) + '\n' + ''.join('box '+' '.join(map(str, box))+'\n' for box in profile['boxes'])
