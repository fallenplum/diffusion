"""Outcome-blind, role-fit-only molecular thresholds and partition-local pairing."""
import json,hashlib
import numpy as np
from src.phase16_semantic.core import ROOT,cfg,check_permutation
from src.phase10_trimodal.features import gene_fit

def molecular(d,fit):
    groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());symbols=d['rna_gene_symbols'].astype(str);lookup={s:j for j,s in enumerate(symbols)};raw=np.asarray(d['rna_genes'],float);z,st=gene_fit(raw,fit);states=[];scores=[];audits=[]
    for key,genes in groups.items():
        indices=np.array([lookup[g] for g in genes if g in lookup and st['usable'][lookup[g]]],int)
        if len(indices)<10:raise ValueError('insufficient mapped genes')
        group=z[:,indices].mean(1);coverage=np.isfinite(raw[:,indices]).mean(1);usable=(coverage>=.7)&np.isfinite(group);quantile=np.quantile(group[fit][usable[fit]],[1/3,2/3]);value=np.where(group<quantile[0],'low',np.where(group>quantile[1],'high','typical'));value[~usable]='unmeasured';scores.append(group);states.append(value);audits.append({'group':key,'indices':indices,'thresholds':quantile,'patient_coverage':coverage,'fit':fit})
    return np.stack(states,1),np.stack(scores,1),{'groups':audits,'gene_preprocessing':st,'fit':fit}
def derange(part,strata,seed):
    part=np.asarray(part,int);rng=np.random.default_rng(seed);answer=part.copy();fallback=np.zeros(len(part),int);remaining=[]
    for label in sorted(set(strata[part])):
        positions=np.flatnonzero(strata[part]==label)
        if len(positions)>1:
            shuffled=rng.permutation(positions);answer[shuffled]=part[np.roll(shuffled,1)]
        else:remaining.extend(positions.tolist())
    # Singleton strata cannot preserve both strict strata and no identity.
    # Pair singletons within same cancer if possible; all relaxations counted.
    cancer=np.array([s.split('|')[0] for s in strata]);left=[]
    for label in sorted(set(cancer[part[remaining]])):
        positions=np.array([p for p in remaining if cancer[part[p]]==label],int)
        if len(positions)>1:
            shuffled=rng.permutation(positions);answer[shuffled]=part[np.roll(shuffled,1)];fallback[positions]=1
        else:left.extend(positions.tolist())
    if len(left)>1:
        a=rng.permutation(left);answer[a]=part[np.roll(a,1)];fallback[a]=2
    elif len(left)==1:
        p=left[0];q=int(next(v for v in rng.permutation(len(part)) if v!=p));answer[p],answer[q]=answer[q],answer[p];fallback[[p,q]]=2
    check_permutation(part,answer);return answer,{'exact':int((fallback==0).sum()),'fallback_cancer_only':int((fallback==1).sum()),'fallback_partition_only':int((fallback==2).sum())}
def clinical_values(d,fit):
    fields=['AGE','SEX','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE','AJCC_PATHOLOGIC_TUMOR_STAGE'];x=np.stack([d['clinical_'+key].astype(str) for key in fields],1).astype(object);age=np.array([float(v) if v.replace('.','',1).isdigit() else np.nan for v in x[:,0]]);median=np.nanmedian(age[fit]);age=np.where(np.isfinite(age),age,median);x[:,0]=[str(int(v)) for v in age]
    for j in range(1,6):x[:,j]=[v if v not in ['nan','NaN','[Not Available]','','NA'] else 'unknown' for v in x[:,j]]
    return x,{'fit':fit,'age_median':float(median),'fields':fields}
def clinical_caption(values,no_stage=False,shuffle=False):
    a,s,t,n,m,stage=values
    if shuffle:a,s,t,n,m,stage=t,stage,a,s,n,m
    if no_stage:return f'Patient: age {a}, sex {s}.'
    return cfg()['clinical_template'].format(age=a,sex=s,T=t,N=n,M=m,Stage=stage)
def RNA_caption(values):
    pairs='; '.join(f'{description} {state}' for description,state in zip(cfg()['RNA_descriptions'],values));return cfg()['RNA_template'].format(states=pairs)
def captions(d,fit,test,seed,fold):
    values,ca=clinical_values(d,fit);states,scores,ra=molecular(d,fit);n=len(values);ci=np.arange(n);ri=np.arange(n);audits={};cancer=d['cancer_type'].astype(str);stage=d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str);missing=np.where(np.isnan(d['raw_rna']).any(1),'missing','complete');cs=np.char.add(np.char.add(cancer,'|'),stage);rs=np.char.add(np.char.add(cancer,'|'),missing)
    for k,part in [('fit',fit),('test',test)]:
        ci[part],ac=derange(part,cs,seed+fold*100+(k=='test')+159);ri[part],ar=derange(part,rs,seed+fold*100+(k=='test')+311);audits[k]={'clinical':ac,'RNA':ar}
    concepts=json.loads((ROOT/'configs/phase6a_titan_crc_morphology_concepts.json').read_text())['concepts'];base=[cfg()['concept_template'].format(morphology=k['text']) for k in concepts];out={}
    for selector in cfg()['G2_selectors']+cfg()['G3_selectors']+['NO_STAGE_CLIN','NO_STAGE_JOINT','NO_RNA_JOINT']:
        per=[]
        for i in range(n):
            c=ci[i] if selector in ['WRONG_CLIN','JOINT_WRONG_CLIN'] else i;r=ri[i] if selector in ['WRONG_RNA','JOINT_WRONG_RNA'] else i;clinical=clinical_caption(values[c],no_stage=selector.startswith('NO_STAGE'),shuffle=selector=='SHUFFLED_FIELDS');rna=RNA_caption(states[r]);context=''
            if selector in ['CLIN_TEXT','WRONG_CLIN','NO_STAGE_CLIN','NO_RNA_JOINT']:context=clinical
            elif selector in ['RNA_TEXT','WRONG_RNA']:context=rna
            elif selector in ['CLIN_RNA_TEXT','JOINT_WRONG_RNA','JOINT_WRONG_CLIN','SHUFFLED_FIELDS','NO_STAGE_JOINT']:context=clinical+' '+rna
            per.append([v+(' '+context if context else '') for v in base])
        out[selector]=per
    return out,{'fit':fit,'test':test,'clinical':ca,'RNA':ra,'clinical_permutation':ci,'RNA_permutation':ri,'fallback':audits,'RNA_scores':scores,'RNA_states':states}
