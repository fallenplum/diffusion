"""Frozen scientific gates; thresholds never depend on observed outcomes."""
import math


def synthetic_gate(rows):
    selected=[r for r in rows if r['n']==512]
    by={(r['scenario'],r['seed'],r['arm']):r for r in selected}
    seeds=sorted({r['seed'] for r in selected});comparisons={};passed=True
    if len(seeds)!=12:raise ValueError('TWELVE_REPLICATIONS_REQUIRED')
    for other in ('S1','S2'):
        delta=[by['CORRELATED_INTERACTION',s,'S3']['Harrell']-by['CORRELATED_INTERACTION',s,other]['Harrell'] for s in seeds]
        nll=[by['CORRELATED_INTERACTION',s,'S3']['Cox_NLL']-by['CORRELATED_INTERACTION',s,other]['Cox_NLL'] for s in seeds]
        ok=sum(delta)/12>=.003 and sum(d>0 for d in delta)>=8 and sum(nll)/12<=0
        comparisons['S3-'+other]={'Harrell_differences':delta,'mean_Harrell':sum(delta)/12,'positive_repeats':sum(d>0 for d in delta),
                                 'Cox_NLL_differences':nll,'mean_NLL_difference':sum(nll)/12,'pass':ok}
        passed=passed and ok
    for scenario in ('NULL','REDUNDANT'):
        delta=[by[scenario,s,'S3']['Harrell']-by[scenario,s,'S0']['Harrell'] for s in seeds]
        ok=sum(delta)/12<=.01;comparisons[scenario]={'differences':delta,'mean':sum(delta)/12,'pass':ok};passed=passed and ok
    return {'passed':passed,'status':'SYNTHETIC_GO' if passed else 'SYNTHETIC_NO_GO','comparisons':comparisons,
            'NLL_rule':'prelocked mean(S3 NLL - comparator NLL) <= 0 for BOTH S1 and S2',
            'large_sample_cannot_rescue':True}


def require_synthetic_go(decision):
    if decision.get('theory_pass') is not True or decision.get('synthetic_neural_pass') is not True:
        raise ValueError('TCGA_ACCESS_DENIED_BEFORE_THEORY_AND_NEURAL_GO')
