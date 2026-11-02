import importlib.util,json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.qpr.graph import native_edges,metadata_hop,spatial_sample
from src.qpr.attention import unit_ball,retrieve,compress
from src.qpr.protocol import require_stage,run_stage
from src.rscil.geometry import EfronRiskSets


def reference():
    p=Path(__file__).parent/'reference/math_reference.py';s=importlib.util.spec_from_file_location('qpr_supplied_reference',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_reference_five_groups_rerun():reference().main()


@pytest.mark.parametrize('tau',[.1,1.,25.,300.])
def test_mass_chunking_and_numpy_equivalence(tau):
    torch.manual_seed(500);k=unit_ball(torch.randn(35,8,dtype=torch.float64));v=unit_ball(torch.randn(35,10,dtype=torch.float64));q=unit_ball(torch.randn(6,8,dtype=torch.float64));w=torch.softmax(torch.randn(35,dtype=torch.float64),0)
    full,lz=retrieve(q,k,v,w,tau=tau,chunk=7);dense,ld=retrieve(q,k,v,w,tau=tau,chunk=1000)
    assert torch.allclose(full,dense,atol=1e-12,rtol=0) and torch.allclose(lz,ld,atol=1e-12,rtol=0)
    for i in range(len(q)):
        n,nz=reference().full_answer(q[i].numpy(),k.numpy(),v.numpy(),w.numpy(),tau=tau)
        assert np.max(abs(n-full[i].numpy()))<1e-12 and abs(nz-float(lz[i]))<1e-12


def test_mass_exactness_zero_slots_and_gradients():
    k=torch.tensor([[.2,0],[.2,0],[-.3,.1],[-.3,.1]],dtype=torch.float64);v=torch.tensor([[1.,0],[1,0],[0,1],[0,1]],dtype=torch.float64)
    w=torch.tensor([.7,.2,.07,.03],dtype=torch.float64);s=torch.tensor([[1.,0,0],[1,0,0],[0,1,0],[0,1,0]],dtype=torch.float64)
    q=torch.tensor([[.4,.5]],dtype=torch.float64,requires_grad=True);pi,kb,vb=compress(k,v,w,s)
    a,_=retrieve(q,k,v,w);b,_=retrieve(q,kb,vb,pi)
    assert (a-b).abs().max()<1e-12
    b.sum().backward();assert torch.isfinite(q.grad).all()
    variable=torch.randn(5,3,dtype=torch.float64,requires_grad=True);weights=torch.softmax(variable,1)
    kk=torch.randn(5,2,dtype=torch.float64);vv=torch.randn(5,2,dtype=torch.float64)
    p,km,vm=compress(kk,vv,torch.ones(5,dtype=torch.float64)/5,weights)
    retrieve(q,km,vm,p)[0].square().sum().backward();assert variable.grad.norm()>0


def test_random_answer_risk_and_efron_bounds():
    ref=reference();ref.test_random_bounds()
    torch.manual_seed(214);k=unit_ball(torch.randn(40,8,dtype=torch.float64));v=unit_ball(torch.randn(40,10,dtype=torch.float64));q=unit_ball(torch.randn(12,8,dtype=torch.float64));w=torch.ones(40,dtype=torch.float64)/40;s=torch.softmax(torch.randn(40,7,dtype=torch.float64),1)
    pi,kb,vb=compress(k,v,w,s);a,_=retrieve(q,k,v,w);b,_=retrieve(q,kb,vb,pi)
    eps=float((a-b).norm(dim=1).max());head=unit_ball(torch.randn(1,10,dtype=torch.float64)).squeeze(0)
    r=a@head;rp=b@head;assert float((r-rp).abs().max())<=eps+1e-12
    t=torch.tensor([1,1,2,2,3,4,5,5,6,7,9,10],dtype=torch.float64);e=torch.tensor([1,1,0,1,1,0,1,0,1,0,1,0])
    rs=EfronRiskSets(t,e);assert abs(float(rs.loss(r)-rs.loss(rp)))<=2*eps+1e-12
    margin=(r[:,None]-r[None,:]).abs()>2*eps
    assert torch.equal(torch.sign(r[:,None]-r[None,:])[margin],torch.sign(rp[:,None]-rp[None,:])[margin])
    assert abs(float(rs.loss(r))-ref.efron_event_mean_nll(t.numpy(),e.numpy(),r.numpy()))<1e-12


def test_unit_ball_autograd_inside_and_outside():
    for value in ([.2,.3],[1.5,2.]):
        x=torch.tensor(value,dtype=torch.float64,requires_grad=True)
        assert torch.autograd.gradcheck(unit_ball,(x,))
        assert unit_ball(x).norm()<=1+1e-12


def test_metadata_lattice_sparse_points_cannot_invent_hop():
    attrs={'patch_size':512,'patch_size_level0':1024,'level0_magnification':40,'target_magnification':20,'overlap':0}
    hop=metadata_hop(attrs);xy=np.array([[0,0],[2048,0],[0,2048]],dtype=np.int64)
    assert len(native_edges(xy,hop)[1])==0
    with pytest.raises(ValueError):metadata_hop({})
    with pytest.raises(ValueError):native_edges(np.array([[0.,0.]]),1024)


def test_permutation_no_cross_slide_translation_and_true_topology():
    xy=np.array([(i*1024,j*1024) for i in range(8) for j in range(8)])
    c,e,_=native_edges(xy,1024);perm=np.random.default_rng(55).permutation(len(xy));cp,ep,_=native_edges(xy[perm],1024)
    assert np.array_equal(c,cp) and np.array_equal(e,ep)
    assert np.array_equal(spatial_sample(c,e,32)[0],spatial_sample(cp,ep,32)[0])
    assert np.array_equal(e,native_edges(xy+2048,1024)[1])
    assert len(e)==210
    ref=reference();assert ref.test_chain_spatial_counterexample()==(1,3)


def test_stage_guards_no_unsafe_direct_entry(tmp_path):
    with pytest.raises(ValueError,match='MISSING_PRIOR'):require_stage('gate1',tmp_path)
    (tmp_path/'gate0').mkdir();(tmp_path/'gate0/decision.json').write_text(json.dumps({'pass':True}))
    require_stage('gate1',tmp_path);require_stage('gate2',tmp_path)
    with pytest.raises(ValueError):require_stage('gate3',tmp_path)
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'status':'DRAFT'}))
    with pytest.raises(ValueError,match='PRECOMMIT'):run_stage('gate1',report_root=tmp_path,lock=lock)
