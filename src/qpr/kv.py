"""Frozen, partition-local node and symmetric edge evidence projections."""
import numpy as np


def relation_features(h,xy,edges,hop):
    a,b=h[edges[:,0]],h[edges[:,1]]
    d=np.linalg.norm(xy[edges[:,0]]-xy[edges[:,1]],axis=1)/(hop*np.sqrt(2))
    psi=np.stack([d,d*d,np.exp(-d),np.ones(len(d))],1)
    return np.c_[(a+b)/2,np.abs(a-b),a*b,psi].astype(np.float32)


def bounded(x):return x/np.maximum(1.,np.linalg.norm(x,axis=1,keepdims=True))


def project_edges(x,pca):
    z=pca.transform(x).astype(np.float32)
    return bounded(z[:,:32]).astype(np.float32),bounded(z[:,:64]).astype(np.float32)
