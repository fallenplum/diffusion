import copy
import pytest
from scripts.semantic_m123_verify_reports import grouped,verify_aggregates


def rows():
    return [{'arm':'M0','seed':0,'outer':i,'Harrell':.6,'Uno60':.7,
        'Harrell_num':.6*i,'Harrell_den':i,'Uno_num':.7*i*i,'Uno_den':i*i} for i in range(1,6)]


def test_component_ratio_not_unweighted_fold_mean():
    r=rows();r[0].update(Uno60=0.,Uno_num=0.)
    actual=grouped(r)['M0',0]['Uno60']
    assert actual==pytest.approx((.7*55-.7)/55)
    assert actual!=pytest.approx(sum(x['Uno60'] for x in r)/5)


@pytest.mark.parametrize('mutation',['duplicate','missing','ratio'])
def test_missing_duplicate_and_inconsistent_component_folds_rejected(mutation):
    r=rows()
    if mutation=='duplicate':r.append(copy.deepcopy(r[0]))
    if mutation=='missing':r.pop()
    if mutation=='ratio':r[0]['Harrell_num']=.1
    with pytest.raises(ValueError):grouped(r)


def test_repair_damage_is_a_mathematical_identity_and_coverage_required():
    r=rows();summary=[{'arm':'M0','Harrell':.6,'Uno60':.7,
        'seed_results':[{'seed':0,'Harrell':.6,'Uno60':.7}]}]
    repairs=[{'arm':'M0','seed':0,'outer':x['outer'],'Harrell_den':x['Harrell_den'],
        'Uno_den':x['Uno_den'],'Harrell_repair':0.,'Harrell_damage':0.,'Uno_repair':0.,'Uno_damage':0.} for x in r]
    assert verify_aggregates(r,summary,[],repairs)['repair_damage_rows']==5
    repairs[0]['Uno_damage']=.2
    with pytest.raises(ValueError,match='REPAIR_MINUS_DAMAGE'):verify_aggregates(r,summary,[],repairs)
