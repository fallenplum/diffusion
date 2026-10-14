"""RNA selection/grouping observes only the current training partition."""
import numpy as np
from src.phase10_trimodal.features import gene_fit,ahash
from src.phase11_semantic.features import prepare as old_prepare


def bins(mean,sd,missing):
    n=len(mean)
    mu=np.argsort(np.argsort(mean,kind='stable'),kind='stable')*5//n
    sig=np.argsort(np.argsort(sd,kind='stable'),kind='stable')*5//n
    # Equal missing fractions stay together; no artificial rank split of ties.
    levels=np.unique(missing);na=np.searchsorted(levels,missing)
    return (mu*5+sig)*len(levels)+na


def shuffled(gm,mean,sd,missing,seed):
    b=bins(mean,sd,missing);perm=np.arange(len(mean));rng=np.random.default_rng(seed)
    for value in np.unique(b):
        ix=rng.permutation(np.flatnonzero(b==value))
        if len(ix)>1:perm[ix]=np.roll(ix,1)
    result=np.zeros_like(gm);result[:,perm]=gm
    assert np.array_equal(gm@gm.T,result@result.T)
    assert np.array_equal(b,b[perm])
    return result,{'permutation':perm.tolist(),'permutation_hash':ahash(perm),'bins':b.tolist(),'singleton_bins':int(sum(np.sum(b==k)==1 for k in np.unique(b)))}


def match_gene_set(eligible,selected,annotation,mean,sd,missing,seed):
    # Candidate genes are OUTSIDE the frozen annotation and disjoint from the
    # selected set. Match training mean/SD quintiles and exact missing fraction.
    bb=bins(mean[eligible],sd[eligible],missing[eligible]);loc={int(g):j for j,g in enumerate(eligible)}
    mu_rank=np.argsort(np.argsort(mean[eligible],kind='stable'),kind='stable')/len(eligible)
    sd_rank=np.argsort(np.argsort(sd[eligible],kind='stable'),kind='stable')/len(eligible)
    selected_set=set(selected);pool=set(int(g) for g in eligible if not annotation[g] and g not in selected_set)
    rng=np.random.default_rng(seed);matched=[];rows=[]
    # Randomized query order prevents systematically favoring the first genes.
    order=rng.permutation(len(selected));answer=np.empty(len(selected),int)
    for pos in order:
        g=int(selected[pos]);j=loc[g]
        exact=[h for h in sorted(pool) if bb[loc[h]]==bb[j] and missing[h]==missing[g]]
        if exact:
            h=int(rng.choice(exact));strict=True
        else:
            candidates=[h for h in sorted(pool) if missing[h]==missing[g]]
            if not candidates:raise ValueError('matched gene pool lacks exact missing-rate support')
            # Locked fallback: nearest rank distance in mean/SD; deterministic
            # randomized tie resolution; never outcomes or test statistics.
            distance=np.array([abs(mu_rank[loc[h]]-mu_rank[j])+abs(sd_rank[loc[h]]-sd_rank[j]) for h in candidates])
            best=np.flatnonzero(distance==distance.min());h=int(candidates[int(rng.choice(best))]);strict=False
        answer[pos]=h;pool.remove(h)
        rows.append({'selected_position':int(pos),'strict_bin_match':strict,'mean_rank_gap':float(abs(mu_rank[loc[h]]-mu_rank[j])),'SD_rank_gap':float(abs(sd_rank[loc[h]]-sd_rank[j]))})
    return answer,rows


def prepare(d,fit,groups,group_seeds,matched_seed):
    fit=np.asarray(fit,np.int64);f,m,a=old_prepare(d,fit,groups,group_seeds[0])
    x,state=gene_fit(d['rna_genes'],fit);eligible=np.flatnonzero(state['usable'])
    missing=np.isnan(d['rna_genes'][fit]).mean(0)
    symbols=d['rna_gene_symbols'].astype(str);annotated_union=set().union(*[set(v) for v in groups.values()]);annotated=np.array([s in annotated_union for s in symbols])
    top=np.asarray(a['top_indices']);top_missing=missing[top]
    if len(np.unique(top_missing))!=1:
        raise ValueError('legacy g0 does not match missing bins; must audit before claiming exact g0 reuse')
    group_masks={};group_meta={}
    for g,seed in enumerate(group_seeds):
        gm,meta=shuffled(m['true'],state['mean'][top],state['sd'][top],top_missing,seed)
        if g==0 and not np.array_equal(gm,m['random']):raise ValueError('g0 differs from original Phase11 grouping')
        group_masks['g'+str(g)]=gm;group_meta['g'+str(g)]={'seed':seed,**meta}
    pathway=eligible[annotated[eligible]]
    # Locked feasible branch: use all eligible pathway genes up to512, then
    # variance-ranked nonpathway fallback ONLY when fewer than512 are available.
    rank=lambda v:v[np.argsort(-state['sd'][v],kind='stable')]
    pathway=rank(pathway);enriched=pathway[:512]
    if len(enriched)<512:
        other=rank(eligible[~annotated[eligible]]);enriched=np.r_[enriched,other[:512-len(enriched)]]
    if len(enriched)!=512:raise ValueError('enriched selection cannot reach 512 measured genes')
    true=np.array([[s in values for s in symbols[enriched]] for values in groups.values()],np.float32)
    covered=int((true.sum(0)>0).sum())
    for j in np.flatnonzero(true.sum(0)==0):true[j%8,j]=1
    random,rm=shuffled(true,state['mean'][enriched],state['sd'][enriched],missing[enriched],group_seeds[0])
    control,matching=match_gene_set(eligible,enriched,annotated,state['mean'],state['sd'],missing,matched_seed)
    memberships={'top_true':m['true'],'enriched_true':true,'enriched_random':random,'matched_random':random.copy(),**group_masks}
    arrays={'top':f['R'],'enriched':x[:,enriched].astype(np.float32),'matched':x[:,control].astype(np.float32)}
    variants={k:{**f,'R':v} for k,v in arrays.items()}
    a['phase12']={'grouping':group_meta,'enriched_random':rm,'eligible_pathway_genes':len(pathway),'enriched_covered':covered,'enriched_fallback':512-covered,
       'enriched_indices':enriched.tolist(),'enriched_symbols':symbols[enriched].tolist(),'enriched_hash':ahash(enriched),
       'matched_indices':control.tolist(),'matched_symbols':symbols[control].tolist(),'matched_hash':ahash(control),'matching':matching,
       'original_top_overlap':int(len(set(top)&set(enriched))),'original_top_matched_overlap':int(len(set(top)&set(control))),
       'selected_missing_fractions':{'top':top_missing.tolist(),'enriched':missing[enriched].tolist(),'matched':missing[control].tolist()},
       'selection_source':'frozen annotation + TRAIN observation>=.9 and variance; NO survival labels',
       'matched_control_source':'outside annotated union; same512 budget; train mean/SD/missing bins; locked nearest-rank fallback'}
    return variants,m['C'],memberships,a
