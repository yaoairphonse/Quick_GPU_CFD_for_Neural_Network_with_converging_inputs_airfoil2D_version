"""Compare a 2D cut-cell run with XFOIL using common upper/lower x/c stations.

XFOIL is an independent numerical reference, not an experimental truth source.
The run must use the same airfoil polygon, Mach, Reynolds number, and incidence.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def reference_polar(path):
    lines = Path(path).read_text().splitlines()
    condition = next((match for line in lines
                      if (match := re.search(r'Mach\s*=\s*([0-9.]+)\s+Re\s*=\s*([0-9.]+)\s*e\s*([+-]?\d+)', line))), None)
    if condition is None:
        raise ValueError('missing Mach and Reynolds number in XFOIL polar')
    rows = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 5:
            try:
                rows.append([float(value) for value in fields[:5]])
            except ValueError:
                pass
    if len(rows) != 1:
        raise ValueError('expected one XFOIL polar row at the matched incidence')
    alpha, cl, cd, cdp, cm = rows[0]
    return dict(alpha=alpha, Mach=float(condition[1]), Re_chord=float(condition[2])*10**int(condition[3]),
                CL=cl, CD=cd, CDp=cdp, CM=cm)


def interpolated(x, value, stations):
    order = np.argsort(x)
    x, value = np.asarray(x)[order], np.asarray(value)[order]
    unique, inverse = np.unique(x, return_inverse=True)
    if len(unique) < 2 or stations[0] < unique[0] or stations[-1] > unique[-1]:
        raise ValueError('surface does not cover the comparison stations')
    if len(unique) != len(x):
        totals = np.bincount(inverse, weights=value)
        value = totals / np.bincount(inverse)
        x = unique
    return np.interp(stations, x, value)


def compare(cp_path, polar_path, run_dir):
    ref = np.loadtxt(cp_path, comments='#')
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError('XFOIL Cp file must have x/c and Cp columns')
    leading = int(np.argmin(ref[:, 0]))
    branches = {1: ref[:leading+1], 2: ref[leading:]}
    stations = np.linspace(.02, .98, 97)
    polar = reference_polar(polar_path)
    config = json.loads((run_dir/'config.resolved.json').read_text())
    pipeline = json.loads((run_dir/'pipeline.json').read_text())
    reynolds = pipeline['scales']['Re_chord']
    if abs(polar['alpha']-config['freestream']['AOA_degrees']) > .001 or \
       abs(polar['Mach']-config['freestream']['Mach']) > .005 or \
       abs(polar['Re_chord']-reynolds)/reynolds > .005:
        raise ValueError('XFOIL and cut-cell Mach/Re/incidence differ')
    result = {'reference': {'kind': 'XFOIL numerical, free transition', **polar},
              'case': {'Mach': config['freestream']['Mach'], 'AOA_degrees': config['freestream']['AOA_degrees']},
              'stations': {'count': len(stations), 'x_over_c_min': stations[0], 'x_over_c_max': stations[-1]},
              'levels': {}}
    for level in ('LF1', 'LF2'):
        folder = run_dir/level
        surface = np.load(folder/'output/surface.npz')
        loads = json.loads((folder/'output/loads.json').read_text())['total']
        run = json.loads((folder/'result.json').read_text())
        errors = []
        patches = {}
        for patch in (1, 2):
            mask = surface['patch_id'] == patch
            expected = interpolated(branches[patch][:, 0], branches[patch][:, 1], stations)
            computed = interpolated(surface['x'][mask]/config['freestream']['chord_m'], surface['Cp'][mask], stations)
            error = computed - expected
            patches['upper' if patch == 1 else 'lower'] = {
                'Cp_rms': float(np.sqrt(np.mean(error**2))),
                'Cp_max_abs': float(np.max(np.abs(error))),
            }
            errors.extend(error)
        # XFOIL reports the opposite moment sign from this solver's CCW-positive Mz.
        cm_reference = -polar['CM']
        result['levels'][level] = {
            'converged': run['converged'], 'spatial_order': run.get('spatial_order', 1),
            'fluid_control_volumes': run['fluid_control_volumes'], 'steps': run['steps'],
            'rejected': run['rejected'], 'process_seconds': run['process_seconds'],
            'CL': loads['CL'], 'CL_relative_error': (loads['CL']-polar['CL'])/polar['CL'],
            'CM_z_CCW': loads['CM_z'], 'CM_relative_error': (loads['CM_z']-cm_reference)/cm_reference,
            'Cp_rms': float(np.sqrt(np.mean(np.square(errors)))), 'patches': patches,
        }
    return result


def plot_cp(cp_path, run_dir, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ref = np.loadtxt(cp_path, comments='#')
    leading = int(np.argmin(ref[:, 0]))
    branches = {1: ref[:leading+1], 2: ref[leading:]}
    stations = np.linspace(.01, .99, 197)
    chord = json.loads((run_dir/'config.resolved.json').read_text())['freestream']['chord_m']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for patch, ax in zip((1, 2), axes):
        r = branches[patch]
        ax.plot(stations, interpolated(r[:, 0], r[:, 1], stations), color='#222222', lw=1.8, label='XFOIL')
        for level, color in (('LF1', '#1769aa'), ('LF2', '#d27416')):
            with np.load(run_dir/level/'output/surface.npz') as surface:
                mask = surface['patch_id'] == patch
                cp = interpolated(surface['x'][mask]/chord, surface['Cp'][mask], stations)
            ax.plot(stations, cp, color=color, lw=1.5, label=level)
        ax.set(title='Upper surface' if patch == 1 else 'Lower surface', xlabel='x/c', xlim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('$C_p$')
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cp', type=Path, required=True)
    parser.add_argument('--polar', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    result = compare(args.cp, args.polar, args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_cp(args.cp, args.run, args.plot)
    print(json.dumps(result['levels'], indent=2))


if __name__ == '__main__':
    main()
