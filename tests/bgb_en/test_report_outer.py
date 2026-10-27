import pytest
from scripts.bgb_en_report_outer import aggregate,comparisons


def rows():
    result=[]
    for seed in (31,47,83):
        for fold in range(1,6):
            for name,num in [('CP_STRONG',fold),('CP_SCALAR',fold+.5)]:
                den=fold+2
                result.append(dict(model=name,endpoint='DSS',fold=fold,seed=seed,Harrell_num=num,
                    Harrell_den=den,Uno_num=2*num,Uno_den=3*den,Harrell=num/den,Uno60=2*num/(3*den),
                    repair=.5/den,damage=0.,Uno_repair=1/(3*den),Uno_damage=0.))
    return result


def test_counts_before_seed_mean_and_complete_coverage():
    result=aggregate(rows(),'CP_STRONG','DSS')
    assert result['Harrell']==pytest.approx(15/25)
    assert result['Uno60']==pytest.approx(30/75)
    assert aggregate(rows()[:-1],'CP_SCALAR','DSS')['status']=='NOT_ALL_FIVE_FOLDS_THREE_SEEDS'


def test_paired_differences_match_repair_and_denominator_guard():
    value=rows();paired=comparisons(value,'CP_SCALAR','CP_STRONG','DSS')
    for row in paired:
        assert row['paired_delta_H']==pytest.approx(row['repair']-row['damage'])
        assert row['paired_delta_U60']==pytest.approx(row['Uno_repair']-row['Uno_damage'])
    value[-1]['Harrell_den']+=1
    with pytest.raises(ValueError,match='Different populations'):comparisons(value,'CP_SCALAR','CP_STRONG','DSS')
