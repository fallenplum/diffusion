import torch
from src.qpr.compressor import SparseCompressor,summary,answer,losses
from src.qpr.query import MolecularQuery


def test_sparse_assignment_equivariance_and_teacher_fixed():
    torch.manual_seed(433);m=SparseCompressor();k=torch.randn(8,32);v=torch.randn(8,64)
    edges=torch.tensor([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,2],[1,4]])
    a=m(k,v,edges,6);order=torch.randperm(8);b=m(k[order],v[order],edges[order],6)
    assert torch.allclose(a[order],b,atol=1e-6)
    q=torch.randn(256,32)*.05;teacher=answer(q,k,v);before=[x.clone() for x in (k,v)]
    la,lc=losses(k,v,a,q,teacher);(la+.1*lc).backward()
    assert all(x.grad is None for x in (k,v)) and all(torch.equal(a,b) for a,b in zip(before,(k,v)))
    assert sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)>0


def test_query_bound_and_masked_programs_do_not_contribute_to_loss():
    torch.manual_seed(33);model=MolecularQuery(7)
    q=model(torch.randn(2,50,3)*100,torch.randn(2,7)*100,torch.randn(2,32)*100)
    assert q.norm(dim=-1).max()<1+1e-6
    # The masked query result is excluded rather than treated as a measured 0.
    k=torch.randn(8,32);v=torch.randn(8,64);s=torch.softmax(torch.randn(8,32),1);q=torch.randn(256,32)*.01
    teacher=answer(q,k,v);mask=torch.ones(256,dtype=torch.bool);mask[:3]=False
    a,_=losses(k,v,s,q,teacher,mask)
    altered=(teacher[0].clone(),teacher[1].clone());altered[0][:3]=999.;altered[1][:3]=-999.
    b,_=losses(k,v,s,q,altered,mask)
    assert torch.equal(a,b)


def test_exactly_empty_mass_has_finite_zero_gradient():
    p=torch.tensor([1.,0.],requires_grad=True);k=torch.zeros(2,32);v=torch.tensor([[1.,0.],[0.,1.]])
    result,_=answer(torch.zeros(1,32),k,v,p);result.square().sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad[1]==0


def test_positive_mass_forward_unchanged_by_zero_mask_fix():
    torch.manual_seed(454);q=torch.randn(7,32);k=torch.randn(8,32);v=torch.randn(8,64);p=torch.softmax(torch.randn(8),0)
    expected=torch.softmax(q@k.T+torch.log(p)[None],1)@v
    assert torch.equal(answer(q,k,v,p)[0],expected)
