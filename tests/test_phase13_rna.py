import unittest
import numpy as np
import torch
from src.phase13_rna.math import block_fit,choose_block
from src.phase10_trimodal.math import fit_ridge
from src.phase12_c4.models import C4Model
from src.phase13_rna.models import RNAOnly
from scripts.replay_phase11 import ridge_objective


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(701);self.x=rng.normal(size=(18,7));self.t=np.array([1,1,1,2,2,3,4,5,6,7,8,9,10,11,12,13,14,15.]);self.e=np.array([1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0])
    def test_equal_penalty_reduces_to_existing_ridge(self):
        a,_=block_fit(self.x,self.t,self.e,np.ones(7)*.3);b,_=fit_ridge(self.x,self.t,self.e,.3)
        np.testing.assert_allclose(a,b,atol=2e-7,rtol=0)
    def test_independent_efron_stationarity(self):
        penalty=np.array([.2,.2,.2,1,1,1,1]);w,_=block_fit(self.x,self.t,self.e,penalty)
        def objective(z):return ridge_objective(self.x,self.t,self.e,z,0)[0]+.5*np.sum(penalty*z*z)
        step=1e-6;directions=np.eye(len(w))*step
        gradient=np.array([(objective(w+v)-objective(w-v))/(2*step) for v in directions])
        self.assertLess(np.max(abs(gradient)),1e-7)
    def test_off_is_exact_baseline_and_wins_tie(self):
        r=np.zeros((18,3));lam,rows=choose_block(self.x,r,self.x,r,self.t,self.e,self.t,self.e,.3,[.1,1,'OFF']);self.assertEqual(lam,'OFF')
        self.assertAlmostEqual(rows[0]['inner_Harrell'],rows[-1]['inner_Harrell'])
    def test_shared_initial_encoder_and_full_mean_identity(self):
        cm=np.zeros((6,12),np.float32);gm=np.zeros((8,512),np.float32)
        for j in range(8):gm[j,j*64:(j+1)*64]=1
        torch.manual_seed(31);reference=C4Model(cm,gm);only=RNAOnly(cm,gm,31)
        for key,value in reference.rna.encoder.state_dict().items():self.assertTrue(torch.equal(value,only.encoder.state_dict()[key]))
        only.eval();x=torch.randn(5,512);risk,taps=only(x)
        self.assertTrue(torch.allclose(taps['processed_full'].reshape(5,8,64).mean(1),taps['processed_mean']))
        self.assertEqual(taps['processed_full'].shape,(5,512))
    def test_encoder_gradient_path_live(self):
        cm=np.zeros((6,12),np.float32);gm=np.zeros((8,512),np.float32)
        for j in range(8):gm[j,j*64:(j+1)*64]=1
        model=RNAOnly(cm,gm,47);x=torch.randn(12,512,requires_grad=True);r,_=model(x);r.square().mean().backward()
        self.assertGreater(float(x.grad.norm()),0);self.assertGreater(float(model.encoder.weight.grad.norm()),0)


if __name__=='__main__':unittest.main()
