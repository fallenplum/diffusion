import ast
import inspect
import numpy as np
import torch
from src.phase21_relative.operators import direct_update
from src.phase20_selective.operators import conservative_update


def example():
    base=torch.tensor([[.15,.25,.35,.25]],dtype=torch.double)
    gate=torch.tensor([[0.,.2,.8,0.]],dtype=torch.double)
    mask=torch.ones_like(base,dtype=torch.bool)
    return base,gate,mask

def test_probability_identity_and_mask():
    b,g,m=example();d=torch.tensor([[.4,-.4,.3,.2]],dtype=torch.double)
    a=direct_update(b,g,d,m)
    assert a.min()>=0 and abs(a.sum()-1)<1e-14
    assert torch.equal(direct_update(b,g,d*0,m),b)
    assert torch.equal(direct_update(b,g*0,d,m),b)
    assert torch.equal(direct_update(b,g,d,m*False),b)

def test_common_row_shift_invariant():
    b,g,m=example();d=torch.tensor([[.4,-.4,.3,.2]],dtype=torch.double)
    torch.testing.assert_close(direct_update(b,g,d,m),direct_update(b,g,d+.123,m),atol=1e-14,rtol=0)
    torch.testing.assert_close(direct_update(b,g,torch.ones_like(d)*.2,m),b,atol=1e-14,rtol=0)

def test_null_derivative_not_suppressed():
    b,g,m=example();d=torch.zeros_like(b,requires_grad=True)
    grad=torch.autograd.grad((direct_update(b,g,d,m)*torch.arange(4)).sum(),d)[0]
    assert grad.abs().max()>0
    assert torch.autograd.gradcheck(lambda x:direct_update(b,g,x,m),(d,),eps=1e-6,atol=1e-6)

def test_same_first_order_as_conservative_and_different_high_order():
    b,g,m=example();z=torch.tensor([[.3,-.4,.5,.2]],dtype=torch.double)
    eps=1e-6
    torch.testing.assert_close((direct_update(b,g,z*eps,m)-b)/eps,
                               (conservative_update(b,g,z*eps,m)-b)/eps,atol=1e-8,rtol=1e-6)
    assert (direct_update(b,g,z,m)-conservative_update(b,g,z,m)).abs().max()>1e-5

def test_only_forward_operator_is_changed():
    from src.phase20_selective.model import SelectiveModel
    from src.phase21_relative.model import DirectModel
    import textwrap
    first=ast.parse(textwrap.dedent(inspect.getsource(SelectiveModel.forward)))
    second=ast.parse(textwrap.dedent(inspect.getsource(DirectModel.forward)))
    for node in ast.walk(second):
        if isinstance(node,ast.Name) and node.id=='direct_update': node.id='conservative_update'
    assert ast.dump(first)==ast.dump(second)
    assert DirectModel.__init__ is SelectiveModel.__init__

def test_parameter_budget_and_same_initialization():
    from src.phase20_selective.model import SelectiveModel
    from src.phase21_relative.model import DirectModel
    config={'hidden':32,'tokenizer':{'filters':16},'selective':{},'training':{'dropout':.1}}
    member=np.ones((8,20),bool)
    models=[]
    for cls,arm in [(SelectiveModel,'P16'),(SelectiveModel,'S16'),(DirectModel,'S16')]:
        torch.manual_seed(131)
        models.append(cls(4,member,np.ones(10),np.zeros(4),.7,arm,config))
    for other in models[1:]:
        assert list(models[0].state_dict())==list(other.state_dict())
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in models[0].state_dict().items())
        assert sum(p.numel() for p in other.parameters())==sum(p.numel() for p in models[0].parameters())
