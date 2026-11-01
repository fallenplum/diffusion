import importlib.util
from pathlib import Path
import torch
from src.rscil.geometry import EfronRiskSets


def test_supplied_reference_ties_and_extreme_scores():
    path=Path(__file__).parent/'reference/riskset_core.py'
    spec=importlib.util.spec_from_file_location('supplied_riskset_core',path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    torch.manual_seed(392);x=torch.randn(20,5,dtype=torch.float64)
    t=torch.arange(20,dtype=torch.float64)//3;e=torch.arange(20)%2==0;r=x[:,0]*.6
    rs=EfronRiskSets(t,e)
    assert (rs.loss(r)-ref.efron_nll(r,t,e)).abs()<1e-12
    assert (rs.information(r,x)-ref.riskset_information(r,t,e,x)).abs().max()<1e-10
    extreme=torch.linspace(-1200,1200,20,dtype=torch.float64)
    assert torch.isfinite(rs.loss(extreme))
    assert (rs.loss(extreme)-ref.efron_nll(extreme,t,e)).abs()<1e-10
