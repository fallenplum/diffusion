import pytest

from scripts.bgb_en_budget_saved_chain_audit import aggregate_cells


def metric(numerator,denominator,repair=0.):
    return dict(Harrell_num=numerator,Harrell_den=denominator,Uno_num=2*numerator,Uno_den=2*denominator,
        Harrell=numerator/denominator,Uno60=numerator/denominator,repair=repair,damage=0.,Uno_repair=repair,Uno_damage=0.)


def fixture_rows():
    rows=[]
    for arm in ('O3','O4'):
        for epochs in (160,320):
            for seed in (31,47):
                for part,(old,new,den) in enumerate([(1.,2.,2.),(1.,2.,10.),(3.,4.,4.)]):
                    before=metric(old,den);after=metric(new,den,(new-old)/den)
                    rows.append(dict(arm=arm,epochs=epochs,seed=seed,partition=part,
                        original_role='FAILED_PARENT80_NUMERICAL_GUARD' if (arm,epochs,seed,part)==('O4',160,47,0) else 'ORIGINAL_COMPLETE',
                        own80=before,final_vs_own80=after,final_vs_offset=after,train80=before,train_final=after,
                        final_decoder={'complete_tokens':{'R2':.2},'pooled32':{'R2':.1}},
                        training_Cox_final_minus_own80=-.1,final_delta_SD_over_offset_SD=.3))
    return rows


def test_full_saved_diagnostic_pools_pair_counts_before_seed_mean_and_retains_failed_role():
    seeds,means=aggregate_cells(fixture_rows())
    assert len(seeds)==8 and len(means)==4
    for row in means:
        assert row['own80']['Harrell']==pytest.approx(5/16)
        assert row['final_vs_own80']['Harrell']==pytest.approx(8/16)
        assert row['final_vs_own80']['delta_H_vs_own_offset']==pytest.approx(3/16)
    failed_group=next(row for row in means if row['arm']=='O4' and row['epochs']==160)
    assert (failed_group['original_COMPLETE_cells'],failed_group['original_FAILED_cells'])==(5,1)


def test_missing_failed_cell_cannot_be_silently_averaged_as_full24():
    rows=[row for row in fixture_rows() if row['original_role']=='ORIGINAL_COMPLETE']
    with pytest.raises(ValueError,match='MISSING_CELL'):
        aggregate_cells(rows)
