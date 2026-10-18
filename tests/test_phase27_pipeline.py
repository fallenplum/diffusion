"""Synthetic fit-only pipeline isolation, not a large-data experiment."""
import json
import numpy as np
import torch
from src.phase27_readout import pipeline,runtime


def test_matched_fit_only_projection_shared_head_and_immutable_main(tmp_path,monkeypatch):
    rng=np.random.default_rng(27);nfit,nvalid=36,7;fit=np.arange(nfit);valid=np.arange(nfit,nfit+nvalid)
    cp=rng.normal(size=nvalid)*.1;rna=rng.normal(size=nvalid)*.1;base=rng.normal(size=nfit)*.2
    x={arm:rng.normal(size=(nfit,256)) for arm in runtime.ARMS}
    z={arm:{name:rng.normal(size=(nvalid,256)) for name in runtime.CONTROLS} for arm in runtime.ARMS}
    features={'source':'test','stage':'prototype','seed':31,'fold':1,'fit':fit,'valid':valid,'base_fit':base,
        'M0_valid':cp+rna,'CP_valid':cp,'RNA_valid':rna,'fit_features':x,'valid_features':z,
        'old_risks':{a:cp+rna for a in runtime.ARMS},'old_fit_risks':{a:base for a in runtime.ARMS},
        'legacy_checkpoint_sha256':'old','prepared_sha256':'prepared'}
    t=np.arange(1,nfit+1,dtype=float);e=np.asarray([int(i%3==0) for i in fit],float)
    saved={'fit':fit,'valid':valid,'base':np.r_[base,cp+rna],'fit_time':t,'fit_event':e}
    private=tmp_path/'private';(private/'features').mkdir(parents=True)
    (private/'features/seed31_fold1.json').write_text(json.dumps({'synthetic':True}))
    monkeypatch.setattr(runtime,'PRIVATE',private);monkeypatch.setattr(runtime,'source',lambda:'test')
    monkeypatch.setattr(pipeline,'read_features',lambda seed,fold:(features,{'prepared_sha256':'prepared'}))
    monkeypatch.setattr(pipeline.old_data,'read_prepared',lambda fold:(saved,{'sha256':'prepared'}))
    original={k:np.array(v,copy=True) for k,v in saved.items()}
    result=pipeline.fit_cell(31,1)
    for k,v in original.items():np.testing.assert_array_equal(saved[k],v)
    for arm,head in result['heads'].items():
        assert head['parameter_count']==16 and head['optimizer']['gradient_max']<=1e-7
        q=head['projection'];coef=head['coef']
        manual=lambda a:((((a-q['pca_mean'])@q['components'].T)-q['pca_score_mean'])/q['pca_score_sd'])@coef
        np.testing.assert_allclose(head['fit_risk'],base+manual(x[arm]),rtol=0,atol=1e-14)
        for name in runtime.CONTROLS:
            np.testing.assert_allclose(head['valid_risks'][name],cp+rna+manual(z[arm][name]),rtol=0,atol=1e-14)
        assert head['fit_diagnostics']['fit_correction_mean_zero_error']<1e-12
    assert (runtime.cell_dir(31,1)/'DONE_PRIVATE.json').is_file()
    assert runtime.verify_done(runtime.cell_dir(31,1),'test',31,1)['new_neural_steps']==0
