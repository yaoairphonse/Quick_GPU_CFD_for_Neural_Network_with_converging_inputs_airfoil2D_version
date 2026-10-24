"""Build a closed, full-3D cubed-sphere shell of triangular-face polyhedral CVs.

Every physical face is shared exactly once. The inner surface is a faceted sphere;
the volume/closure audit uses that discrete geometry, not the analytic sphere.
"""
from pathlib import Path
import argparse,json
import numpy as np

def build(destination, subdivisions=12, radial_layers=24, outer_radius=8.0):
    n,nr=subdivisions,radial_layers
    if n<2 or nr<2 or outer_radius<=1:raise ValueError('invalid shell dimensions')
    points=[];point_ids={}
    radii=1+(outer_radius-1)*np.expm1(3.5*np.linspace(0,1,nr+1))/np.expm1(3.5)
    def vertex(axis,sign,i,j,k):
        cube=[0,0,0];cube[axis]=sign*n
        other=[a for a in range(3) if a!=axis];cube[other[0]]=2*i-n;cube[other[1]]=2*j-n
        key=(*cube,k)
        if key not in point_ids:
            direction=np.asarray(cube,dtype=float);direction/=np.linalg.norm(direction)
            point_ids[key]=len(points);points.append(direction*radii[k])
        return point_ids[key]
    quads=[];centers=[]
    for axis in range(3):
        for sign in [-1,1]:
            for k in range(nr):
                for j in range(n):
                    for i in range(n):
                        a,b,c,d=[vertex(axis,sign,ii,jj,k) for ii,jj in [(i,j),(i+1,j),(i+1,j+1),(i,j+1)]]
                        e,f,g,h=[vertex(axis,sign,ii,jj,k+1) for ii,jj in [(i,j),(i+1,j),(i+1,j+1),(i,j+1)]]
                        quads.append([(a,b,c,d),(e,f,g,h),(a,b,f,e),(b,c,g,f),(c,d,h,g),(d,a,e,h)])
                        centers.append(np.mean([points[v] for v in [a,b,c,d,e,f,g,h]],axis=0))
    points=np.asarray(points);centers=np.asarray(centers);faces=[];face_lookup={};cell_faces=[[] for _ in quads]
    volumes=np.zeros(len(quads));moments=np.zeros_like(centers);wall_area=0
    for ci,cell in enumerate(quads):
        for quad in cell:
            # The diagonal through the smallest global vertex is the same on both sides.
            k=int(np.argmin(quad));q=quad[k:]+quad[:k]
            for ids in [(q[0],q[1],q[2]),(q[0],q[2],q[3])]:
                vertices=points[list(ids)];fc=vertices.mean(axis=0)
                av=.5*np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0])
                if np.dot(av,fc-centers[ci])<0:av=-av
                tv=np.dot(av,fc)/3
                volumes[ci]+=tv;moments[ci]+=tv*(.75*fc)
                key=tuple(sorted(ids))
                if key in face_lookup:
                    fi=face_lookup[key];assert faces[fi]['right']==-1;faces[fi]['right']=ci
                    assert np.linalg.norm(av+faces[fi]['av'])<1e-11
                else:
                    fi=len(faces);face_lookup[key]=fi
                    is_inner=np.all(np.abs(np.linalg.norm(vertices,axis=1)-1)<1e-12)
                    faces.append(dict(left=ci,right=-1,bc=1 if is_inner else 2,center=fc,av=av,vertices=list(ids)))
                cell_faces[ci].append(fi)
    assert np.all(volumes>0);centers=moments/volumes[:,None]
    closure=np.zeros_like(centers);outer_volume=inner_volume=0
    for f in faces:
        av=f['av'];fc=f['center'];l,r=f['left'],f['right'];f['area']=float(np.linalg.norm(av));f['normal']=av/f['area']
        closure[l]+=av
        if r>=0:
            closure[r]-=av;f['bc']=0;delta=centers[r]-centers[l]
        else:
            delta=fc-centers[l]
            if f['bc']==1:wall_area+=f['area'];inner_volume-=np.dot(av,fc)/3
            else:outer_volume+=np.dot(av,fc)/3
        f['distance']=float(np.dot(delta,f['normal']))
        assert f['distance']>0
    terms=[];cells=[];adjacency=[]
    for ci,indices in enumerate(cell_faces):
        ds=[];sources=[]
        for fi in indices:
            f=faces[fi];neighbor=f['right'] if f['left']==ci else f['left']
            if neighbor>=0:ds.append(centers[neighbor]-centers[ci]);sources.append((neighbor,fi))
            else:ds.append(f['center']-centers[ci]);sources.append((-f['bc'],fi))
        ds=np.asarray(ds);weights=1/np.einsum('ij,ij->i',ds,ds)
        matrix=np.einsum('i,ij,ik->jk',weights,ds,ds)
        coefficients=(np.linalg.inv(matrix)@(weights[:,None]*ds).T).T
        start=len(terms)
        terms.extend((neighbor,fi,*coeff) for (neighbor,fi),coeff in zip(sources,coefficients))
        cells.append((*centers[ci],volumes[ci],max(float(np.linalg.norm(centers[ci])-1),1e-12),len(adjacency),len(indices),start,len(sources)))
        adjacency.extend(indices)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with (destination/'mesh.txt').open('w') as stream:
        stream.write(f'{len(cells)} {len(faces)} {len(adjacency)} {len(terms)}\n')
        for c in cells:stream.write(' '.join(map(str,c))+'\n')
        for f in faces:stream.write(' '.join(map(str,(f['left'],f['right'],f['bc'],*f['center'],*f['normal'],f['area'],f['distance'])))+'\n')
        stream.write(' '.join(map(str,adjacency))+'\n')
        for t in terms:stream.write(' '.join(map(str,t))+'\n')
    metadata=dict(cells=len(cells),faces=len(faces),subdivisions=n,radial_layers=nr,outer_radius=outer_radius,
        minimum_volume=float(volumes.min()),closure_linf=float(np.linalg.norm(closure,axis=1).max()),
        volume_balance_error=float(abs(volumes.sum()-(outer_volume-inner_volume))),wall_area=wall_area,
        analytic_sphere_area=4*np.pi,relative_sphere_area_error=abs(wall_area/(4*np.pi)-1),
        inner_volume=inner_volume,analytic_sphere_volume=4*np.pi/3,
        wall_distance='radial distance to analytic unit sphere; wall fluxes use the faceted surface',
        geometry='full six-block cubed sphere with closed triangulated polyhedral control volumes; no axisymmetric reduction')
    assert metadata['closure_linf']<1e-10 and metadata['volume_balance_error']<1e-8
    (destination/'mesh.json').write_text(json.dumps(metadata,indent=2)+'\n')
    np.savez_compressed(
        destination/'geometry.npz', points=points, centers=centers,
        volumes=volumes, cell_quads=np.asarray(quads, dtype=np.int32),
        wall_triangles=np.array([
            points[f['vertices']] for f in faces
            if f['right'] < 0 and f['bc'] == 1
        ]),
    )
    print(json.dumps(metadata,indent=2));return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=12);p.add_argument('--nr',type=int,default=24)
    a=p.parse_args();build(a.out,a.n,a.nr)
