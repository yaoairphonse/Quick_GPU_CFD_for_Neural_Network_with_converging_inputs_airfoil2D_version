"""Read the public NASA DPW4 cell-based CRM tetrahedral mesh with system CGNS.

The ParaView CGNS library prefixes its C API with vtkcgns_. No pip dependency is
needed. Geometry is normalized by the published 275.8-inch reference chord.
"""
from pathlib import Path
import ctypes as C
import argparse,json,time
import numpy as np

def read(source,destination,chord=275.8):
    start=time.perf_counter();out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    lib=C.CDLL('/lib/x86_64-linux-gnu/libvtkcgns-pv5.11.so.1');lib.vtkcgns_cg_get_error.restype=C.c_char_p
    def call(name,*args):
        code=getattr(lib,'vtkcgns_'+name)(*args)
        if code:raise RuntimeError(lib.vtkcgns_cg_get_error().decode())
    ptr=lambda a:a.ctypes.data_as(C.c_void_p)
    file=C.c_int();call('cg_open',str(source).encode(),0,C.byref(file))
    name=C.create_string_buffer(128);sizes=np.zeros(9,dtype=np.int64)
    call('cg_zone_read',file,1,1,name,ptr(sizes));npnt,ncell=int(sizes[0]),int(sizes[1])
    coords=np.empty((3,npnt),dtype=np.float64);lower=np.ones(3,dtype=np.int64);upper=np.ones(3,dtype=np.int64);upper[0]=npnt
    for axis,key in enumerate(['CoordinateX','CoordinateY','CoordinateZ']):call('cg_coord_read',file,1,1,key.encode(),4,ptr(lower),ptr(upper),ptr(coords[axis]))
    xyz=np.ascontiguousarray(coords.T/chord);xyz.tofile(out/'points.bin')
    count=C.c_int();call('cg_nsections',file,1,1,C.byref(count));sections=[];triangles=[];triangle_ids=[];tetra=None
    for s in range(1,count.value+1):
        kind=C.c_int();first=C.c_int64();last=C.c_int64();boundary=C.c_int();parent=C.c_int()
        call('cg_section_read',file,1,1,s,name,C.byref(kind),C.byref(first),C.byref(last),C.byref(boundary),C.byref(parent))
        size=C.c_int64();call('cg_ElementDataSize',file,1,1,s,C.byref(size))
        data=np.empty(size.value,dtype=np.int64);parents=np.empty(4*(last.value-first.value+1),dtype=np.int64) if parent.value else None
        call('cg_elements_read',file,1,1,s,ptr(data),ptr(parents) if parents is not None else None)
        description=dict(name=name.value.decode(),kind=kind.value,first=first.value,last=last.value)
        sections.append(description);print(description,flush=True)
        if kind.value==10:
            if tetra is not None:raise ValueError('multiple tetrahedron sections are not supported')
            tetra=np.asarray(data.reshape(-1,4)-1,dtype=np.int32);tetra.tofile(out/'tetra.bin')
        elif kind.value==5:
            triangles.append(np.asarray(data.reshape(-1,3)-1,dtype=np.int32));triangle_ids.append(np.arange(first.value,last.value+1,dtype=np.int64))
        else:raise ValueError(f'unsupported CGNS element type {kind.value}')
    tri=np.concatenate(triangles);ids=np.concatenate(triangle_ids);order=np.argsort(ids);sorted_ids=ids[order]
    bctype=np.zeros(len(tri),dtype=np.int32);patch=np.zeros_like(bctype);nb=C.c_int();call('cg_nbocos',file,1,1,C.byref(nb));bocos=[]
    for b in range(1,nb.value+1):
        bc=C.c_int();ptype=C.c_int();npoints=C.c_int64();normal=np.zeros(3,dtype=np.int32);nnormal=C.c_int64();ntype=C.c_int();ndataset=C.c_int()
        call('cg_boco_info',file,1,1,b,name,C.byref(bc),C.byref(ptype),C.byref(npoints),ptr(normal),C.byref(nnormal),C.byref(ntype),C.byref(ndataset))
        data=np.empty(npoints.value,dtype=np.int64);normal_data=np.empty(nnormal.value,dtype=np.float64)
        call('cg_boco_read',file,1,1,b,ptr(data),ptr(normal_data) if nnormal.value else None)
        info=dict(name=name.value.decode(),kind=bc.value,point_set=ptype.value,count=npoints.value,first_values=data[:4].tolist());bocos.append(info)
        # CGNS codes: BCWallViscous=22, BCWallViscousHeatFlux=23,
        # BCSymmetryPlane=16, BCFarfield=7. Inspect names as an additional check.
        lname=info['name'].lower()
        native=3 if bc.value==16 or 'reflection' in lname else 2 if bc.value==7 or 'box' in lname or 'far' in lname else 1 if bc.value in [20,21,22,23,24,25] else 0
        if not native:raise ValueError(f'unrecognized boundary: {info}')
        if ptype.value in [4,6]:element_ids=np.arange(data[0],data[1]+1,dtype=np.int64)
        elif ptype.value in [2,7]:element_ids=data
        else:raise ValueError(f'unsupported boundary point set: {info}')
        loc=np.searchsorted(sorted_ids,element_ids)
        if np.any(loc>=len(ids)) or not np.array_equal(sorted_ids[loc],element_ids):raise ValueError(f'BC references non-surface elements: {info}')
        bctype[order[loc]]=native;patch[order[loc]]=b
    call('cg_close',file)
    if np.any(bctype==0):raise ValueError('surface elements without boundary conditions')
    np.column_stack((tri,bctype,patch)).astype(np.int32).tofile(out/'boundary.bin')
    mesh=dict(source=str(source),points=npnt,cells=ncell,tetrahedra=len(tetra),boundary_triangles=len(tri),
              reference_chord_source_units=chord,reference_area_normalized=594720/(chord*chord),
              symmetry='half aircraft; reflect-plane boundary retained',bounds_normalized=[xyz.min(axis=0).tolist(),xyz.max(axis=0).tolist()],
              sections=sections,boundaries=bocos,read_convert_seconds=time.perf_counter()-start)
    (out/'input.json').write_text(json.dumps(mesh,indent=2)+'\n')
    (out/'input_counts.txt').write_text(f'{npnt} {len(tetra)} {len(tri)}\n')
    print(json.dumps({k:v for k,v in mesh.items() if k not in ['sections','boundaries']},indent=2));return mesh

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();read(a.source,a.out)
