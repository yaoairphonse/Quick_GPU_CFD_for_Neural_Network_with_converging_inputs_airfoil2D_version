"""Build an offline CRM tetrahedral volume and exact plane-section inspector.

The 3D CRM solver uses body-fitted tetrahedra. There is no Cartesian parent
cell, so the 2D cut-cell fluid-area fraction has no 3D counterpart here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path

import numpy as np


CELL_DTYPE = np.dtype([
    ("x", "<f8", (3,)), ("volume", "<f8"), ("dwall", "<f8"),
    ("first", "<i4"), ("count", "<i4"),
    ("term_first", "<i4"), ("term_count", "<i4"),
])
BOUNDARY_DTYPE = np.dtype([("node", "<i4", (3,)), ("bc", "<i4"), ("patch", "<i4")])
EDGE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
FACE_TRIPLES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
WINDOWS = (
    ("Fuselage crown", 0.10, (4.0, 6.4), (1.05, 1.80)),
    ("Wing midspan", 2.10, (4.75, 5.90), (0.61, 0.79)),
    ("Outer wing", 4.00, (6.20, 6.85), (0.85, 1.02)),
)


def mesh_arrays(mesh_path: Path, connectivity_path: Path):
    with mesh_path.open("rb") as stream:
        magic, count, _, _, _ = struct.unpack("<8s4Q", stream.read(40))
    if magic.rstrip(b"\0") != b"QCFD3D1":
        raise ValueError("expected QCFD3D1 mesh")
    with connectivity_path.open("rb") as stream:
        header = struct.unpack("<8sII10Q", stream.read(96))
    if (header[0].rstrip(b"\0") != b"QCFDNC1" or header[1] != 1
            or header[2] != 0x01020304 or header[4] != count
            or header[12] != connectivity_path.stat().st_size):
        raise ValueError("incompatible or truncated QCFDNC1 connectivity")
    cells = np.memmap(mesh_path, dtype=CELL_DTYPE, mode="r", offset=40, shape=(count,))
    points = np.memmap(connectivity_path, dtype="<f8", mode="r", offset=header[7], shape=(header[3], 3))
    tetra = np.memmap(connectivity_path, dtype="<i4", mode="r", offset=header[8], shape=(count, 4))
    boundary = np.memmap(connectivity_path, dtype=BOUNDARY_DTYPE, mode="r",
                         offset=header[11], shape=(header[5],))
    return cells, points, tetra, boundary


def intersect_y(vertices: np.ndarray, y: float):
    """Return an exact tetrahedron/plane intersection polygon in x,z."""
    result = []
    for a, b in EDGE_PAIRS:
        pa, pb = vertices[a], vertices[b]
        da, db = pa[1] - y, pb[1] - y
        if abs(da) < 1e-12 and abs(db) < 1e-12:
            result.extend((pa[[0, 2]], pb[[0, 2]]))
        elif abs(da) < 1e-12:
            result.append(pa[[0, 2]])
        elif abs(db) < 1e-12:
            result.append(pb[[0, 2]])
        elif da * db < 0:
            p = pa + (-da / (db - da)) * (pb - pa)
            result.append(p[[0, 2]])
    unique = []
    for p in result:
        if not any(np.linalg.norm(p - q) < 1e-10 for q in unique):
            unique.append(p)
    if len(unique) < 3:
        return None
    polygon = np.asarray(unique)
    center = polygon.mean(axis=0)
    polygon = polygon[np.argsort(np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0]))]
    area = 0.5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                     - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))
    return polygon if area > 1e-15 else None


def aspect_ratio(vertices: np.ndarray, volume: float):
    longest = max(np.linalg.norm(vertices[b] - vertices[a]) for a, b in EDGE_PAIRS)
    largest_face_area = max(0.5 * np.linalg.norm(np.cross(vertices[b] - vertices[a],
                                                         vertices[c] - vertices[a]))
                            for a, b, c in FACE_TRIPLES)
    return float(longest * largest_face_area / (3 * volume))


def wall_segments(points, boundary, y):
    segments = []
    for triangle in points[boundary["node"][boundary["bc"] == 1]]:
        if not (triangle[:, 1].min() <= y <= triangle[:, 1].max()):
            continue
        intersections = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = triangle[a], triangle[b]
            da, db = pa[1] - y, pb[1] - y
            if da * db < 0:
                p = pa + (-da / (db - da)) * (pb - pa)
                intersections.append(p[[0, 2]])
            elif abs(da) < 1e-12:
                intersections.append(pa[[0, 2]])
        if len(intersections) >= 2 and np.linalg.norm(intersections[0] - intersections[1]) > 1e-10:
            segments.append(np.round(intersections[:2], 10).tolist())
    return segments


def make_data(cells, points, tetra, boundary, chunk):
    bounds = (0, 11, 0, 6, 0, 1.5)
    volume_edges = np.arange(-19, 6, 1, dtype=float)
    counts = np.zeros(len(volume_edges) - 1, dtype=np.int64)
    volume_sums = np.zeros(len(counts), dtype=float)
    near_count = 0
    near_volume = 0.0
    windows = [dict(name=name, y=y, x=list(xrange), z=list(zrange), cells=[], wall=[])
               for name, y, xrange, zrange in WINDOWS]
    for first in range(0, len(cells), chunk):
        last = min(first + chunk, len(cells))
        block = cells[first:last]
        center, volume = block["x"], block["volume"]
        near = ((center[:, 0] >= bounds[0]) & (center[:, 0] <= bounds[1]) &
                (center[:, 1] >= bounds[2]) & (center[:, 1] <= bounds[3]) &
                (center[:, 2] >= bounds[4]) & (center[:, 2] <= bounds[5]))
        v = volume[near]
        if np.any(~np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("non-positive or non-finite mesh volume")
        near_count += len(v)
        near_volume += float(v.sum())
        histogram, _ = np.histogram(np.log10(v), bins=volume_edges)
        weighted, _ = np.histogram(np.log10(v), bins=volume_edges, weights=v)
        counts += histogram
        volume_sums += weighted

        vertices = np.asarray(points[tetra[first:last]])
        for window in windows:
            y, (xmin, xmax), (zmin, zmax) = window["y"], window["x"], window["z"]
            selected = ((vertices[:, :, 1].min(axis=1) <= y) &
                        (vertices[:, :, 1].max(axis=1) >= y) &
                        (vertices[:, :, 0].max(axis=1) >= xmin) &
                        (vertices[:, :, 0].min(axis=1) <= xmax) &
                        (vertices[:, :, 2].max(axis=1) >= zmin) &
                        (vertices[:, :, 2].min(axis=1) <= zmax))
            for index in np.flatnonzero(selected):
                polygon = intersect_y(vertices[index], y)
                if polygon is None:
                    continue
                volume_cell = float(volume[index])
                window["cells"].append([
                    first + int(index), np.round(polygon, 10).tolist(),
                    round(math.log10(volume_cell), 5),
                    round(aspect_ratio(vertices[index], volume_cell), 3),
                    round(float(block["dwall"][index]), 7),
                ])
        print(f"read {last:,} / {len(cells):,} tetrahedra", flush=True)
    for window in windows:
        window["wall"] = wall_segments(points, boundary, window["y"])
        window["count"] = len(window["cells"])
        window["wall_segments"] = len(window["wall"])
    return dict(mesh="NASA DPW4 CRM medium body-fitted tetrahedral half-model",
                total_cells=len(cells), nearfield_bounds=bounds,
                nearfield_cells=near_count, nearfield_volume=near_volume,
                volume_bin_log10_edges=volume_edges.tolist(), volume_bin_counts=counts.tolist(),
                volume_bin_sums=volume_sums.tolist(), windows=windows,
                definition="Each tetrahedron is a full fluid control volume; no Cartesian parent-cell occupancy fraction exists.")


def write_preview(data, destination: Path):
    """Export a static check of the same exact polygons used by the HTML."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for axis, window in zip(axes, data["windows"]):
        polygons = [cell[1] for cell in window["cells"]]
        values = np.array([cell[2] for cell in window["cells"]])
        collection = PolyCollection(polygons, array=values, cmap="turbo",
                                    edgecolors="none", linewidths=0,
                                    norm=matplotlib.colors.Normalize(vmin=-13, vmax=-4))
        axis.add_collection(collection)
        axis.add_collection(LineCollection(window["wall"], colors="black", linewidths=0.45))
        axis.set_xlim(window["x"]); axis.set_ylim(window["z"])
        axis.set_aspect("equal", adjustable="box")
        axis.set(xlabel="x / MAC", ylabel="z / MAC",
                 title=f"{window['name']}: y/MAC={window['y']:.2f}, {window['count']:,} exact tetrahedron sections")
    fig.colorbar(collection, ax=axes, label="log10(tetrahedron volume / MAC³)", shrink=0.75)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CRM 3D volume mesh inspector</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#183246;font:14px/1.45 system-ui}
header{background:#173745;color:white;padding:14px 22px}h1{font-size:21px;margin:0}h2{font-size:17px}
main{max-width:1500px;margin:auto;padding:16px}.grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}
.card{background:white;border:1px solid #d6e0e7;border-radius:8px;padding:14px;margin-bottom:15px}
.note{background:#fff2d6;border-left:4px solid #e6a236;padding:10px;margin:0 0 16px}
canvas{width:100%;height:600px;border:1px solid #d6e0e7;display:block;cursor:crosshair}
select,input{font:inherit;margin:5px}label{display:block;margin:8px 0}#readout{font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;padding:4px;border-bottom:1px solid #e2e9ee}th:first-child,td:first-child{text-align:left}
.bar{display:inline-block;height:8px;background:#2474a6;vertical-align:middle;margin-left:5px;min-width:1px}.vol{background:#e68138}
@media(max-width:850px){.grid{grid-template-columns:1fr}canvas{height:450px}}
</style>
<header><h1>CRM 3D volume mesh inspector</h1>NASA DPW4 10M tetrahedral mesh · offline, exact plane intersections</header>
<main><div class="note">The solver mesh uses body-fitted tetrahedra. Each tetrahedron has its own full fluid volume; there is no Cartesian parent cube and no 2D-style cut-cell occupancy fraction. The table compares <b>fraction of cells</b> with <b>fraction of total fluid volume</b> in the stated near-field region.</div>
<div class="grid"><section><div class="card"><label>Exact spanwise cut <select id="station"></select></label>
<label>Color <select id="metric"><option value="volume">tetrahedron volume (log scale)</option><option value="aspect">longest edge / minimum altitude (log scale)</option></select></label>
<label><input type="checkbox" id="wall" checked> Draw actual wall intersection</label>
<canvas id="view"></canvas><p>Each filled polygon is the exact intersection of one original 3D tetrahedron with the selected y plane. Drag to pan; scroll to zoom; click to inspect the underlying 3D cell. White inside the flow region can indicate an uncovered slice and merits inspection. The aircraft solid and outside the mesh are also white.</p>
<p><a href="__VIEWER__">Open the existing rotatable 3D tetrahedron viewer</a> for the local y/c ≈ 2.103 window.</p></div></section>
<aside><div class="card"><h2>Slice</h2><div id="sliceInfo"></div><h2>Clicked cell</h2><div id="readout">Click a colored polygon.</div></div>
<div class="card"><h2>Near-field volume allocation</h2><div id="summary"></div><table><thead><tr><th>V / MAC³</th><th>cells</th><th>cell %</th><th>volume %</th></tr></thead><tbody id="bins"></tbody></table>
<p>Blue bar: cell share. Orange bar: volume share. Near-field membership uses cell centroid, so bins crossing the region boundary are not clipped.</p></div></aside></div></main>
<script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);
const station=el('station'),metric=el('metric'),canvas=el('view'),ctx=canvas.getContext('2d');
D.windows.forEach((w,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${w.name}: y/MAC = ${w.y.toFixed(2)}`;station.append(o)});
let scale=1,tx=0,ty=0,drag=false,last=[0,0],moved=false,screenPolys=[];
function windowData(){return D.windows[+station.value]}
function resize(){let d=devicePixelRatio||1;canvas.width=Math.round(canvas.clientWidth*d);canvas.height=Math.round(canvas.clientHeight*d);ctx.setTransform(d,0,0,d,0,0);draw()}
function bounds(){let w=windowData(),dx=w.x[1]-w.x[0],dz=w.z[1]-w.z[0],s=Math.min((canvas.clientWidth-40)/dx,(canvas.clientHeight-40)/dz)*scale;return {w,s,ox:(canvas.clientWidth-dx*s)/2+tx,oz:(canvas.clientHeight-dz*s)/2+ty}}
function project(p,b){return [b.ox+(p[0]-b.w.x[0])*b.s,canvas.clientHeight-b.oz-(p[1]-b.w.z[0])*b.s]}
function color(c){let t=metric.value==='volume'?(c[2]+13)/9:(Math.log10(Math.max(1,c[3]))/3.5);t=Math.max(0,Math.min(1,t));let hue=(1-t)*235;return `hsl(${hue.toFixed(1)} 85% 49%)`}
function draw(){let b=bounds();ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.clientWidth,canvas.clientHeight);screenPolys=[];
 for(let c of b.w.cells){let poly=c[1].map(p=>project(p,b)),fill=color(c);ctx.beginPath();ctx.moveTo(poly[0][0],poly[0][1]);for(let j=1;j<poly.length;j++)ctx.lineTo(poly[j][0],poly[j][1]);ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=fill;ctx.lineWidth=.25;ctx.stroke();screenPolys.push([poly,c])}
 if(el('wall').checked){ctx.strokeStyle='#101820';ctx.lineWidth=1.8;for(let seg of b.w.wall){let a=project(seg[0],b),z=project(seg[1],b);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(z[0],z[1]);ctx.stroke()}}
 ctx.fillStyle='#172e41';ctx.fillText(`x/MAC ${b.w.x[0]}–${b.w.x[1]}   z/MAC ${b.w.z[0]}–${b.w.z[1]}`,12,18);
 el('sliceInfo').textContent=`${b.w.count.toLocaleString()} original tetrahedra intersect this plotted section; ${b.w.wall_segments.toLocaleString()} original wall-triangle segments.`}
function inside(p,poly){let x=p[0],y=p[1],hit=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){let a=poly[i],b=poly[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])hit=!hit}return hit}
canvas.onpointerdown=e=>{drag=true;moved=false;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};
canvas.onpointerup=()=>{drag=false};canvas.onpointermove=e=>{if(!drag)return;let dx=e.clientX-last[0],dy=e.clientY-last[1];if(Math.abs(dx)+Math.abs(dy)>1)moved=true;tx+=dx;ty-=dy;last=[e.clientX,e.clientY];draw()};
canvas.onwheel=e=>{e.preventDefault();scale=Math.max(.5,Math.min(15,scale*Math.exp(-e.deltaY*.001)));draw()};
canvas.onclick=e=>{if(moved)return;let rect=canvas.getBoundingClientRect(),p=[e.clientX-rect.left,e.clientY-rect.top];for(let i=screenPolys.length-1;i>=0;i--){let [poly,c]=screenPolys[i];if(inside(p,poly)){el('readout').textContent=`cell ID: ${c[0]}\nvolume / MAC³: ${Math.pow(10,c[2]).toExponential(4)}\nequivalent cube edge / MAC: ${Math.pow(10,c[2]/3).toExponential(4)}\nlongest edge / min altitude: ${c[3]}\nwall distance / MAC: ${c[4]}`;return}}el('readout').textContent='No intersected tetrahedron at this pixel.'};
station.onchange=()=>{scale=1;tx=ty=0;draw()};metric.onchange=draw;el('wall').onchange=draw;
el('summary').textContent=`${D.nearfield_cells.toLocaleString()} cell centroids in x=[0,11], y=[0,6], z=[0,1.5] MAC; their total volume is ${D.nearfield_volume.toExponential(3)} MAC³. Full half-model: ${D.total_cells.toLocaleString()} cells.`;
let rows=[];for(let i=0;i<D.volume_bin_counts.length;i++){let n=D.volume_bin_counts[i],v=D.volume_bin_sums[i];if(!n)continue;let cn=100*n/D.nearfield_cells,cv=100*v/D.nearfield_volume;let fmt=x=>x<.01?x.toFixed(4):x.toFixed(2);rows.push(`<tr><td>10<sup>${D.volume_bin_log10_edges[i]}</sup>–10<sup>${D.volume_bin_log10_edges[i+1]}</sup></td><td>${n.toLocaleString()}</td><td>${fmt(cn)}<span class="bar" style="width:${Math.max(1,cn*1.5)}px"></span></td><td>${fmt(cv)}<span class="bar vol" style="width:${Math.max(1,cv*1.5)}px"></span></td></tr>`)}el('bins').innerHTML=rows.join('');
addEventListener('resize',resize);resize();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--connectivity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=200_000)
    args = parser.parse_args()
    cells, points, tetra, boundary = mesh_arrays(args.mesh, args.connectivity)
    data = make_data(cells, points, tetra, boundary, args.chunk)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "crm_volume_inspector.json").write_text(json.dumps({k: v for k, v in data.items() if k != "windows"}
                                                           | {"windows": [{k: v for k, v in w.items() if k not in ("cells", "wall")}
                                                                          for w in data["windows"]]}, indent=2) + "\n")
    viewer = Path(__file__).resolve().parent.parent / "crm10m" / "mesh3d_viewer.html"
    html = HTML.replace("__VIEWER__", os.path.relpath(viewer, args.out.resolve()))
    (args.out / "crm_volume_inspector.html").write_text(html.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    write_preview(data, args.out / "crm_volume_sections.png")
    print(json.dumps({"output": str(args.out / "crm_volume_inspector.html"),
                      "nearfield_cells": data["nearfield_cells"],
                      "slices": [(w["name"], w["count"]) for w in data["windows"]]}, indent=2))


if __name__ == "__main__":
    main()
