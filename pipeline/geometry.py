"""Reorder the GXXX x-paired point format without moving or dropping points."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    with source.open(encoding="utf-8-sig") as stream:
        if [x.strip().lower() for x in stream.readline().strip().split(",")] != ["x", "y"]:
            raise ValueError("CSV first row must be x,y")
    points = np.loadtxt(source, delimiter=',', skiprows=1, ndmin=2)
    if points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('expected CSV header x,y followed by two finite numeric columns')
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError('duplicate points are not accepted; coordinates are never silently removed')
    xs = np.unique(points[:, 0])
    if len(xs) < 3 or xs[0] != 0 or xs[-1] != 1 or np.sum(points[:, 0] == 0) != 1:
        raise ValueError('GXXX format requires normalized x/c in [0,1], one leading-edge point at x=0')
    upper, lower = [], []
    for x in xs[1:-1]:
        ys = np.sort(points[points[:, 0] == x, 1])
        if len(ys) != 2 or ys[0] >= ys[1]:
            raise ValueError(f'x={x}: expected distinct lower and upper points, exactly two')
        lower.append((x, ys[0])); upper.append((x, ys[1]))
    te = points[points[:, 0] == 1]
    te = te[np.argsort(te[:, 1])]
    if len(te) < 2:
        raise ValueError('GXXX format requires a finite-thickness vertical trailing edge with >=2 points')
    vertices = np.array([te[-1], *upper[::-1], points[points[:, 0] == 0][0], *lower, *te[:-1]])
    if len(vertices) != len(points) or set(map(tuple, vertices)) != set(map(tuple, points)):
        raise ValueError('polygon reordering did not preserve the complete coordinate set')
    area = float(np.sum(vertices[:, 0]*np.roll(vertices[:, 1], -1)-vertices[:, 1]*np.roll(vertices[:, 0], -1))/2)
    if area <= 0:
        raise ValueError('airfoil polygon has nonpositive area')
    patches = [1]*(len(upper)+1)+[2]*(len(lower)+1)+[3]*(len(te)-1)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination/'input_points.csv')
    np.savetxt(destination/'airfoil.csv', vertices, delimiter=',', header='x,y', comments='', fmt='%.17g')
    (destination/'patches.csv').write_text('edge,patch\n'+''.join(f'{i},{p}\n' for i,p in enumerate(patches)))
    match = re.search(r'G\d+', source.stem)
    name = match.group() if match else re.sub(r'[^A-Za-z0-9_-]', '_', source.stem)
    manifest = dict(name=name, source_file=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    polygon_sha256=hashlib.sha256((destination/'airfoil.csv').read_bytes()).hexdigest(),
                    vertices=len(vertices), area_c2=area, perimeter_c=float(np.linalg.norm(vertices-np.roll(vertices, 1, axis=0), axis=1).sum()),
                    all_input_coordinates_preserved=True, patches={'upper':1, 'lower':2, 'trailing_edge':3})
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
