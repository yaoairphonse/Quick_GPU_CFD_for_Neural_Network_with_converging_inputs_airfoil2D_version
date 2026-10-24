#!/usr/bin/env python3
"""Render published CFD samples using CV averages on background member rectangles.

Install docs/requirements.txt, then run this script from any working directory.
The solid mask preserves the original polygon. No field interpolation is applied.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/quick-cfd-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import numpy as np

ROOT=Path(__file__).resolve().parent
BOUNDS=(-.35,1.8,-.65,.65)

def render():
    samples={level:np.load(ROOT/f'data/G0001_{level}.npz') for level in ['LF1','LF2']}
    variables=[('mach','Mach number','turbo',1),('p','Static pressure (kPa)','viridis',1e-3),
               ('speed','Velocity magnitude (m/s)','plasma',1),('temperature','Temperature (K)','magma',1)]
    limits={}
    for key,_,_,scale in variables:
        values=np.concatenate([d[key][d['members'][:,1].astype(int)]*scale for d in samples.values()])
        limits[key]=(float(values.min()),float(values.max()))
    for level,d in samples.items():
        members=d['members'];owners=members[:,1].astype(int)
        rectangles=np.array([[[r[2],r[4]],[r[3],r[4]],[r[3],r[5]],[r[2],r[5]]] for r in members])
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for ax,(key,label,cmap,scale) in zip(axes.flat,variables):
            collection=PolyCollection(rectangles,array=d[key][owners]*scale,cmap=cmap,edgecolors='none',rasterized=True)
            collection.set_clim(*limits[key]);ax.add_collection(collection)
            ax.add_patch(Polygon(d['polygon'],facecolor='#edf2f5',edgecolor='#1b2d3c',linewidth=.9))
            ax.set(xlim=BOUNDS[:2],ylim=BOUNDS[2:],xlabel='x / c',ylabel='y / c',title=label)
            ax.set_aspect('equal');fig.colorbar(collection,ax=ax,shrink=.8,pad=.02)
        fig.suptitle(f'G0001 | {level} | M∞ = 1.65 | AOA = 1.5° | SA-neg / Sutherland',fontsize=15)
        fig.supxlabel(f'{int(d["cells"])} fluid CVs; steady criteria passed; t* = {float(d["time"]):g}. CV averages on member rectangles; solid masked.',fontsize=10)
        fig.savefig(ROOT/f'assets/G0001_{level}_flow.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
    for level,d in samples.items():
        for patch,label,color in [(1,'upper','#1766ab'),(2,'lower','#c65b18')]:
            mask=d['patch']==patch;order=np.argsort(d['surface_x'][mask])
            ax.plot(d['surface_x'][mask][order],d['cp'][mask][order],color=color,
                    linestyle='-' if level=='LF2' else '--',linewidth=1.3,label=f'{level} {label}')
    ax.invert_yaxis();ax.set(xlabel='x / c',ylabel='Cp = (p_owner − p∞) / q∞',title='G0001 surface pressure coefficient | M∞ = 1.65 | AOA = 1.5°')
    ax.grid(alpha=.2);ax.legend(ncol=2);fig.supxlabel('Owner-cell pressure at wall-fragment midpoints; LF1/LF2 comparison does not establish grid independence.',fontsize=10)
    fig.savefig(ROOT/'assets/G0001_surface_comparison.png',dpi=170);plt.close(fig)
    print('Rendered LF1/LF2 flow fields and surface Cp comparison.')

if __name__=='__main__':render()
