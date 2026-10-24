"""Check restart continuity, invalid input rejection, and the saved CRM state."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--executable', required=True)
parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--integrator', choices=['rk3', 'dplur'], default='rk3')
parser.add_argument('--work', required=True)
parser.add_argument('--crm-mesh')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
work = Path(args.work).resolve()
work.mkdir(parents=True, exist_ok=True)
mesh = root / 'local_development/sphere3d/case_M165/mesh.txt'
base = [args.executable, '--backend', args.backend,
        '--integrator', args.integrator, '--budget', '60',
        '--binary-output', '1', '--check-period', '1000']
if args.integrator == 'dplur':
    base += ['--cfl', '1', '--cfl-max', '20', '--dplur-sweeps', '5']


def run(options, expected=(0, 3)):
    result = subprocess.run(base + options, capture_output=True, text=True)
    if result.returncode not in expected:
        raise RuntimeError(result.stdout + result.stderr)
    return result


def sphere(name, steps, restart=None):
    options = ['--mesh', str(mesh), '--out', str(work / name),
               '--iterations', str(steps)]
    if restart:
        options += ['--restart', str(restart)]
    run(options)
    folder = work / name
    state = folder / 'state.bin'
    best = folder / 'best_state.bin'
    result = json.loads((folder / 'result.json').read_text())
    assert result['best_checkpoint_available']
    assert state.stat().st_size == best.stat().st_size
    return state


whole = sphere('whole_20', 20)
first = sphere('first_10', 10)
continued = sphere('continued_10', 10, first)
a = np.fromfile(whole, dtype='<f8').reshape(-1, 6)
b = np.fromfile(continued, dtype='<f8').reshape(-1, 6)
errors = (np.max(np.abs(a - b), axis=0) /
          np.maximum(np.max(np.abs(a), axis=0), 1e-30))
assert np.max(errors) < 2e-13, errors
assert json.loads((continued.parent / 'result.json').read_text())['restarted']
metadata_only = work / 'metadata_only'
run(['--mesh', str(mesh), '--out', str(metadata_only), '--iterations', '0',
     '--field-output', 'none'])
metadata = json.loads((metadata_only / 'result.json').read_text())
assert not metadata['best_checkpoint_available']
assert not (metadata_only / 'state.bin').exists()
assert not (metadata_only / 'best_state.bin').exists()
run(['--mesh', str(mesh), '--out', str(first.parent), '--restart', str(first),
     '--iterations', '0'], expected=(2,))
invalid = work / 'truncated.bin'
invalid.write_bytes(first.read_bytes()[:-8])
run(['--mesh', str(mesh), '--out', str(work / 'invalid_size'),
     '--restart', str(invalid), '--iterations', '0'], expected=(2,))
for name, column, value in [('nan', 5, float('nan')), ('density', 0, -1.),
                            ('energy', 4, -1.)]:
    bad = b.copy()
    bad[0, column] = value
    invalid = work / f'{name}.bin'
    bad.tofile(invalid)
    run(['--mesh', str(mesh), '--out', str(work / f'invalid_{name}'),
         '--restart', str(invalid), '--iterations', '0'], expected=(2,))

report = {'backend': args.backend, 'integrator': args.integrator,
          'sphere_20_vs_10_plus_10_relative_Linf':
          errors.tolist(), 'invalid_checkpoints_rejected': True,
          'input_directory_overwrite_rejected': True,
          'binary_checkpoint_write_verified': True,
          'metadata_only_checkpoint_flag_verified': True}
if args.crm_mesh:
    saved = root / 'local_development/crm10m/result'
    out = work / 'crm_load_only'
    run(['--mesh', args.crm_mesh, '--out', str(out), '--restart',
         str(saved / 'state.bin'), '--iterations', '0', '--mach', '.85',
         '--aoa', '2.5', '--re', '5000000', '--mu-star', '.00000017',
         '--reference-area', '3.9092552169415784', '--moment-x',
         '4.807469180565627', '--moment-z', '.6451776649746193',
         '--cfl', '.182326', '--step-scale', '1'])
    original = json.loads((saved / 'result.json').read_text())
    loaded = json.loads((out / 'result.json').read_text())
    residual_error = max(abs(x-y)/max(abs(x), 1e-30)
                         for x, y in zip(original['raw_residual'],
                                         loaded['raw_residual']))
    assert residual_error < 2e-11, residual_error

    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()

    assert sha(out / 'state.bin') == sha(saved / 'state.bin')
    report['CRM_10067380_cells_roundtrip_bitwise_equal'] = True
    report['CRM_initial_residual_relative_difference'] = residual_error
report['passed'] = True
(work / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
