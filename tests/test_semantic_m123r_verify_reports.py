import pytest
from scripts.semantic_m123r_verify_reports import ratios


def fixture():
    return [{'inner':str(i),'Harrell_num':str(n),'Harrell_den':str(d),'Harrell':str(n/d),
        'Uno_num':str(n),'Uno_den':str(d),'Uno60':str(n/d)} for i,n,d in ((0,9,10),(1,9,10),(2,10,100))]


def test_exact_pair_weighting_avoids_equal_fold_average():
    assert ratios(fixture())['Harrell']==pytest.approx(28/120)


@pytest.mark.parametrize('mutation',['missing','duplicate','ratio'])
def test_public_arithmetic_rejects_incomplete_or_inconsistent_folds(mutation):
    r=fixture()
    if mutation=='missing':r.pop()
    if mutation=='duplicate':r[-1]['inner']='0'
    if mutation=='ratio':r[0]['Harrell']='0.5'
    with pytest.raises(ValueError):ratios(r)
