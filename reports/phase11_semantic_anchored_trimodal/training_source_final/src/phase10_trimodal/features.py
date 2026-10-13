"""Fit-local features with explicit source provenance and separate endpoints."""
import hashlib, warnings
import numpy as np
from sklearn.decomposition import PCA

CLINICAL=['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']

def ahash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def validate_provenance(columns,provenance):
    if set(columns)!=set(provenance):raise ValueError('unregistered column')
    for col in columns:
        p=provenance[col]
        if p.get('source_field') not in CLINICAL or p.get('source_kind')!='diagnosis_clinical' or p.get('label_derived') is not False:
            raise ValueError('forbidden input provenance: '+col)

def clinical(d,fit,simplified=False,requested_fields=None,provenance=None):
    names=['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE'] if simplified else CLINICAL
    if requested_fields is not None:names=list(requested_fields)
    registry=provenance if provenance is not None else {name:{'source_field':name,'source_kind':'diagnosis_clinical','label_derived':False} for name in names}
    validate_provenance(names,registry)
    parts=[];audit={}
    for name in names:
        vals=np.asarray(d['clinical_'+name]).astype(str)
        if name=='AGE':
            a=np.array([float(v) if v.replace('.','',1).isdigit() else np.nan for v in vals]); med=np.nanmedian(a[fit]);a=np.where(np.isfinite(a),a,med);parts.append(a[:,None]);audit[name]={'median':float(med)}
        else:
            levels=np.unique(vals[fit]);parts.append(np.stack([vals==v for v in levels],axis=1));audit[name]={'levels':levels.tolist()}
    return np.concatenate(parts,axis=1).astype(float),audit

def gene_fit(x,fit):
    x=np.asarray(x,float)
    if np.isinf(x).any():raise ValueError('infinite gene evidence')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(x[fit],axis=0)
    med=np.where(np.isfinite(med),med,0.);v=np.where(np.isfinite(x),x,med);mu=v[fit].mean(0);sd=v[fit].std(0)
    usable=(np.isfinite(x[fit]).mean(0)>=.9)&(sd>1e-8)
    return (v-mu)/np.where(sd>1e-8,sd,1.),{'median':med,'mean':mu,'sd':sd,'usable':usable}

def scale(x,fit):
    mu=x[fit].mean(0);sd=x[fit].std(0);sd[sd<1e-8]=1.
    return (x-mu)/sd,{'mean':mu,'sd':sd}

def prepare(d,fit,groups,seed=7301):
    fit=np.asarray(fit,int);out={};audit={'fit_hash':ahash(fit),'fit_n':len(fit)}
    cv,ca=clinical(d,fit);out['C'],cs=scale(cv,fit);audit['clinical']={'fields':CLINICAL,'coding':ca,'state_hash':ahash(np.concatenate([cs['mean'],cs['sd']]))}
    out['C_min'],_=scale(clinical(d,fit,True)[0],fit)
    out['S'],ss=scale(np.asarray(d['static'],float),fit);audit['static_hash']=ahash(np.concatenate([ss['mean'],ss['sd']]))
    z,st=gene_fit(d['rna_genes'],fit) # already log2(x+1), duplicate symbols collapsed by audited source
    eligible=np.flatnonzero(st['usable']);top=eligible[np.argsort(-st['sd'][eligible],kind='stable')[:512]]
    if len(top)!=512:raise ValueError('fewer than 512 measured variable RNA genes')
    out['R']=z[:,top];audit['top512_hash']=ahash(top);audit['gene_fit_hash']=ahash(np.r_[st['median'],st['mean'],st['sd']])
    tv,ts=scale(np.asarray(d['titan'],float),fit);p=PCA(32,svd_solver='full').fit(tv[fit]);out['T']=p.transform(tv);audit['titan_pca_hash']=ahash(p.components_)
    pc=PCA(16,svd_solver='full').fit(out['R'][fit]);out['RPCA']=pc.transform(out['R']);audit['rna_pca_hash']=ahash(pc.components_)
    symbols=d['rna_gene_symbols'].astype(str); loc={s:j for j,s in enumerate(symbols)}
    c,cst=gene_fit(d['cna_genes'],fit);cloc={s:j for j,s in enumerate(d['cna_gene_symbols'].astype(str))}
    gene_sets=[];mol=[];gain=[];loss=[];rng=np.random.default_rng(seed)
    # One global outcome-blind permutation preserves group sizes and overlap.
    ranks=np.argsort(np.argsort(st['mean'][eligible],kind='stable'),kind='stable')*5//len(eligible)
    vari=np.argsort(np.argsort(st['sd'][eligible],kind='stable'),kind='stable')*5//len(eligible)
    perm=np.arange(z.shape[1]);bins=ranks*5+vari
    for b in np.unique(bins):
        members=rng.permutation(eligible[bins==b]);perm[members]=np.roll(members,1)
    random_sets=[]
    for values in groups.values():
        jj=np.array([loc[g] for g in values if g in loc and st['usable'][loc[g]]],int)
        if len(jj)<3:raise ValueError('group has insufficient measured genes')
        selected=jj[np.argsort(-st['sd'][jj],kind='stable')[:64]]
        gene_sets.append(selected);random_sets.append(perm[selected]);mol.append(z[:,jj].mean(1))
        cj=[cloc[g] for g in values if g in cloc and cst['usable'][cloc[g]]]
        if len(cj)<3:raise ValueError('CNA group insufficient')
        # CNA is signed log2 dosage: gain/loss are separate BEFORE centering.
        raw=np.asarray(d['cna_genes'],float)[:,cj];raw=np.where(np.isfinite(raw),raw,cst['median'][cj])
        gain.append(np.maximum(raw,0).mean(1));loss.append(np.maximum(-raw,0).mean(1))
    def pack(sets):
        a=np.zeros((len(z),len(sets),64))
        for k,jj in enumerate(sets):a[:,k,:len(jj)]=z[:,jj]
        return a
    out['G']=pack(gene_sets);out['G_random']=pack(random_sets);out['R_tokens']=out['R'].reshape(len(z),8,64)
    out['M'],_=scale(np.stack(mol,axis=1),fit);out['CNA'],_=scale(np.column_stack(gain+loss),fit)
    audit.update({'group_sizes':[len(jj) for jj in gene_sets],'group_hash':ahash(np.concatenate(gene_sets)),'random_group_hash':ahash(np.concatenate(random_sets)),'group_permutation_hash':ahash(perm),'cna_fit_hash':ahash(np.r_[cst['median'],cst['mean'],cst['sd']]),'rna_source':'log2 collapsed gene matrix, excludes symbol-free rows','cna_source':'log2 dosage, gain and loss separate'})
    return {k:v.astype(np.float32) for k,v in out.items()},audit

def derangement(d,indices,seed):
    ix=np.asarray(indices,int);rng=np.random.default_rng(seed);source=ix.copy()
    platform=np.isnan(d['raw_rna']).sum(1)>0
    blocks=np.char.add(d['cancer_type'].astype(str),np.where(platform,'_NA','_complete'))
    for b in np.unique(blocks[ix]):
        pos=np.flatnonzero(blocks[ix]==b)
        if len(pos)<2:raise ValueError('singleton matching block')
        order=rng.permutation(pos);source[order]=ix[np.roll(order,1)]
    if np.any(source==ix) or set(source)!=set(ix):raise ValueError('not a split-local derangement')
    return source
