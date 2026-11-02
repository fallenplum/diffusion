"""Actual level-0 lattice edges, never sequence or sparse-GRID neighbors."""
from __future__ import annotations
import hashlib
import numpy as np


def canonical(coords):
    xy=np.asarray(coords)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.issubdtype(xy.dtype,np.integer) or len(xy)==0:raise ValueError('INVALID_INTEGER_XY')
    order=np.lexsort((xy[:,1],xy[:,0]));x=xy[order]
    if len(x)>1 and np.any(np.all(x[1:]==x[:-1],axis=1)):raise ValueError('DUPLICATE_COORDS')
    return order


def metadata_hop(attrs):
    required=('patch_size','patch_size_level0','level0_magnification','target_magnification','overlap')
    if any(k not in attrs for k in required):raise ValueError('GEOMETRY_METADATA_INCOMPLETE')
    ps,p0,mag,target,overlap=[float(attrs[k]) for k in required]
    if min(ps,p0,mag,target)<=0 or not 0<=overlap<ps:raise ValueError('INVALID_GEOMETRY_METADATA')
    if abs(p0-ps*mag/target)>1e-6:raise ValueError('PATCH_SCALE_INCONSISTENT')
    hop=(ps-overlap)*mag/target
    if abs(hop-round(hop))>1e-6:raise ValueError('NONINTEGER_PHYSICAL_HOP')
    return int(round(hop))


def native_edges(coords,hop):
    order=canonical(coords);xy=np.asarray(coords,dtype=np.int64)[order]
    if hop<=0 or not np.all((xy-xy[0])%hop==0):raise ValueError('NOT_ON_METADATA_LATTICE')
    lookup={tuple(p):i for i,p in enumerate(xy)};out=[]
    for i,(x,y) in enumerate(xy):
        for dx,dy in ((hop,0),(0,hop),(hop,hop),(hop,-hop)):
            j=lookup.get((x+dx,y+dy))
            if j is not None:out.append((min(i,j),max(i,j)))
    edges=np.asarray(sorted(out),dtype=np.int64).reshape(-1,2)
    return xy,edges,order


def spatial_sample(xy,edges,cap=4096,seed=62001):
    if len(edges)==0:return edges.copy(),0.
    mid=(xy[edges[:,0]]+xy[edges[:,1]])/2
    bins=np.minimum(15,((mid-mid.min(0))/np.maximum(np.ptp(mid,axis=0),1)*16).astype(int))
    cell=bins[:,0]*16+bins[:,1]
    if len(edges)<=cap:return edges.copy(),1.
    hs=np.array([int.from_bytes(hashlib.blake2b((str(seed)+':'+str(tuple(xy[u]))+':'+str(tuple(xy[v]))).encode(),digest_size=8).digest(),'little') for u,v in edges],dtype=np.uint64)
    rank=np.zeros(len(edges),int)
    for c in np.unique(cell):
        ix=np.flatnonzero(cell==c);rank[ix[np.argsort(hs[ix],kind='stable')]]=np.arange(len(ix))
    take=np.lexsort((hs,cell,rank))[:cap];take=np.sort(take)
    return edges[take],len(np.unique(cell[take]))/len(np.unique(cell))


def diagnostics(n,edges):
    deg=np.bincount(edges.ravel(),minlength=n)
    parent=np.arange(n)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for u,v in edges:
        a,b=find(u),find(v)
        if a!=b:parent[b]=a
    return {'components':len({find(i) for i in range(n)}),'isolated_ratio':float(np.mean(deg==0)),
            'degree_mean':float(deg.mean()),'degree_max':int(deg.max())}
