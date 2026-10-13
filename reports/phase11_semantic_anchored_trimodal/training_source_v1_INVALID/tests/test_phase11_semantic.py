"""Synthetic hard gates; no patient/outcome selection from real outer tests."""
import unittest
import numpy as np, torch
from src.phase11_semantic.models import SemanticModel, CrossAttention
from src.phase11_semantic.data import nested_order
from src.phase10_trimodal.math import log_weights, torch_cox
from src.phase10_trimodal.reference import reference_efron
from src.phase10_trimodal.features import clinical, derangement
from src.phase11_semantic.features import prepare


def example(arm='B0', seed=31, bs=4, n=17):
    torch.manual_seed(seed)
    cm = np.eye(6, dtype=np.float32); gm = np.zeros((8,512), np.float32)
    for k in range(8): gm[k,k*64:(k+1)*64] = 1
    m = SemanticModel(arm, cm, gm, 16).eval()
    values = [torch.randn(bs,6),torch.randn(bs,32),torch.randn(bs,512),torch.randn(bs,2,n,768),torch.ones(bs,2,n,dtype=torch.bool)]
    values[4][:,1,-3:] = False
    return m, values


class MechanismTests(unittest.TestCase):
    def test_s32_and_rna_live(self):
        for seed in [31,47]:
            for bs in [3,5]:
                m, x = example(seed=seed,bs=bs); r = m(*x); r.square().sum().backward()
                for part in [m.concept,m.rna,m.cp,m.rp,m.patch_projection,m.head]:
                    self.assertGreater(sum(float(p.grad.norm()) for p in part.parameters() if p.grad is not None),1e-8)
                for k in [1,2]:
                    changed = list(x); changed[k] = changed[k]*0
                    self.assertGreater(float((r-m(*changed)).abs().max()),1e-8)
                # No anchor escape: learned semantic-query output must respond.
                with torch.no_grad():
                    st=m.concept(x[1]); st0=m.concept(x[1]*0)
                    pp=m.patch_projection(x[3][:,0])
                    self.assertGreater(float((m.cp(st,pp)-m.cp(st0,pp)).abs().max()),1e-8)

    def test_no_rna_controls(self):
        for arm in ['A5','A6','B6','D3']:
            m,x=example(arm); expected=m(*x); changed=list(x);changed[2]=x[2]*100+999
            torch.testing.assert_close(expected,m(*changed),rtol=0,atol=0)
            expected.sum().backward()
            self.assertTrue(all(p.grad is None or not torch.count_nonzero(p.grad) for p in m.rna.parameters()))

    def test_auxiliary_separate(self):
        m,x=example('D3'); r,_,aux,_,_=m(*x,details=True);aux.square().sum().backward()
        self.assertGreater(float(m.rna.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in m.patch_projection.parameters()))

    def test_order_and_padding(self):
        for arm in ['A5','A6','B0','B1','B2','B3','B4','B5','C2']:
            m,x=example(arm);r=m(*x);p=torch.randperm(x[3].shape[2]);v=list(x);v[3]=x[3][:,:,p];v[4]=x[4][:,:,p]
            torch.testing.assert_close(r,m(*v),rtol=1e-5,atol=2e-6)
            v=list(x);v[3]=x[3].clone();v[3][~x[4]]=1e15
            torch.testing.assert_close(r,m(*v),rtol=0,atol=0)
            v=list(x);v[3]=x[3].flip(1);v[4]=x[4].flip(1)
            torch.testing.assert_close(r,m(*v),rtol=1e-5,atol=2e-6)

    def test_attention_oracle(self):
        torch.manual_seed(7); a=CrossAttention(4).double().eval();q=torch.randn(2,3,4,dtype=torch.float64);kv=torch.randn(2,5,4,dtype=torch.float64);mask=torch.tensor([[1,1,0,1,0],[1,0,1,1,0]],dtype=torch.bool)
        actual,w=a(q,kv,mask,True)
        qq=a.qkv(a.norm(q))[...,:4].detach().numpy();_,kk,vv=a.qkv(a.norm(kv)).detach().numpy().reshape(2,5,3,4).transpose(2,0,1,3)
        oracle=np.zeros((2,3,5))
        for b in range(2):
            for i in range(3):
                values=np.array([np.dot(qq[b,i],kk[b,j])/2 if mask[b,j] else -np.inf for j in range(5)])
                values=np.exp(values-np.max(values));oracle[b,i]=values/values.sum()
        np.testing.assert_allclose(w.detach().numpy(),oracle,atol=1e-14)
        pre=q+torch.tensor(oracle@vv);torch.testing.assert_close(actual,pre+a.ff(pre),rtol=1e-12,atol=1e-12)

    def test_full_risk_set(self):
        t=np.array([1,1,2,2,4,5.]);e=np.array([1,1,0,1,1,0]);r=torch.tensor([1000,999,-1000,900,700,600.],requires_grad=True,dtype=torch.float64)
        actual=torch_cox(r,log_weights(t,e,'cpu'),torch.tensor(e))
        self.assertAlmostEqual(float(actual),reference_efron(t,e,r.detach().numpy()),places=10)
        actual.backward();self.assertTrue(torch.isfinite(r.grad).all())
        small=torch_cox(r[:3],log_weights(t[:3],e[:3],'cpu'),torch.tensor(e[:3]))
        self.assertGreater(abs(float(actual-small)),1e-3)

    def test_nested_prefixes(self):
        rng=np.random.default_rng(44)
        for count in [150,760,2201]:
            xy=np.c_[rng.permutation(count),rng.permutation(count)]*512
            order=nested_order(xy);self.assertEqual(len(set(order)),min(count,1024))
            self.assertTrue(set(order[:128])<=set(order[:512])<=set(order[:1024]))
            perm=rng.permutation(count); mapped=perm[nested_order(xy[perm])]
            np.testing.assert_array_equal(order,mapped)

    def test_wrong_split_local(self):
        d={'raw_rna':np.ones((16,4)),'cancer_type':np.array(['COAD']*8+['READ']*8)}
        for ix in [np.arange(8),np.arange(8,16)]:
            perm=derangement(d,ix,31);self.assertEqual(set(ix),set(perm));self.assertTrue((ix!=perm).all())

    def test_source_alias_rejected(self):
        d={'clinical_AGE':np.array(['1','2'])}
        with self.assertRaises(ValueError):
            clinical(d,np.arange(2),requested_fields=['benign_age'],provenance={'benign_age':{'source_field':'DSS_time','source_kind':'diagnosis_clinical','label_derived':False}})

    def test_all_arm_shapes(self):
        for a in ['A5','A6','A7']+['B'+str(i) for i in range(8)]+['C'+str(i) for i in range(1,6)]+['D'+str(i) for i in range(4)]:
            m,x=example(a);outputs=m(*x,details=True);self.assertEqual(outputs[0].shape,(4,));self.assertTrue(torch.isfinite(outputs[0]).all())

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
