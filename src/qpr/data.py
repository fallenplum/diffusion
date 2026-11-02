"""Outcome-free, fit-local evidence for the independent G2 mechanism test."""
import numpy as np
from sklearn.decomposition import PCA
from src.bgb_en.data import read_h5_slide
from src.bgb_en.protocol import clinical_fit,scale_fit,scale_apply
from src.semantic_b0b7.data import ObservedScale,read_membership
from src.qpr.graph import metadata_hop,native_edges,spatial_sample
from src.qpr.kv import relation_features,project_edges


def make_spatial_partition(raw,catalog,fit,evaluation,gmt):
    if any(k in raw for k in ('t','e','DSS','time','event')):raise ValueError('ENDPOINTS_FORBIDDEN_IN_SPATIAL_PREPROCESSING')
    fit_rows=np.asarray(fit);evaluation_rows=np.asarray(evaluation)
    if (fit_rows.ndim!=1 or evaluation_rows.ndim!=1 or len(np.unique(fit_rows))!=len(fit_rows)
            or len(np.unique(evaluation_rows))!=len(evaluation_rows)
            or np.intersect1d(fit_rows,evaluation_rows).size):raise ValueError('DISJOINT_UNIQUE_PARTITIONS_REQUIRED')
    fit=set(map(int,fit));scope=fit|set(map(int,evaluation));samples=[]
    def read(r):
        d=read_h5_slide(r['path'],feature_dim=768);hop=metadata_hop(d['coordinate_attrs']);xy,edges,order=native_edges(d['coords'],hop)
        selected,_=spatial_sample(xy,edges)
        return d['features'][order],xy,selected,hop
    for r in catalog:
        if r['patient'] in fit:
            x,xy,e,hop=read(r);samples.append(x[np.linspace(0,len(x)-1,min(64,len(x))).astype(int)])
    pn=PCA(32,svd_solver='full').fit(np.concatenate(samples));graphs=[];sample_edges=[]
    for i,r in enumerate(catalog):
        if r['patient'] not in scope:continue
        x,xy,e,hop=read(r);h=pn.transform(x).astype(np.float32);rel=relation_features(h,xy,e,hop)
        if r['patient'] in fit:sample_edges.append(rel[np.linspace(0,len(rel)-1,min(64,len(rel))).astype(int)])
        perm=np.random.default_rng(62001+i).permutation(len(h))
        wrong=relation_features(h[perm],xy,e,hop)
        # Content-only self evidence; every source node remains available, but
        # no pair/edge relation or adjacency is consumed by this control.
        # The same feature definition evaluated at a single node: no neighbor
        # is used and distance is exactly zero, with the same constant channels.
        node_ids=np.arange(len(h));content=relation_features(h,xy,np.c_[node_ids,node_ids],hop)
        graphs.append({'owner':r['patient'],'slide_index':i,'relation':rel,'wrong_relation':wrong,'content_relation':content})
    pe=PCA(64,svd_solver='full').fit(np.concatenate(sample_edges))
    for g in graphs:
        for name,source in [('true','relation'),('wrong','wrong_relation'),('content','content_relation')]:g[name]=project_edges(g.pop(source),pe)
    names,member=read_membership(gmt,raw['rna_symbols']);idx=np.array(sorted(fit),int)
    rna,mask=ObservedScale.fit(raw['rna'],idx).transform(raw['rna']);count=mask.astype(np.float32)@member.T
    coverage=count/np.maximum(member.sum(1),1)[None];measured=(coverage>=.5)&(count>=5)
    value=np.where(measured,rna@member.T/np.maximum(count,1),0)
    program=np.stack([value,measured.astype(float),coverage],-1).astype(np.float64)
    c,cs=clinical_fit(raw,idx);ss=scale_fit(raw['S32'],idx);s32=scale_apply(raw['S32'],ss)
    return {'graphs':graphs,'clinical':c,'s32':s32,'program':program,'program_mask':measured,
            'preprocessing':{'node_pca':pn,'edge_pca':pe,'clinical':cs,'S32':ss,'pathways':names,'fit':idx},'DSS_in_preprocessing':False}
