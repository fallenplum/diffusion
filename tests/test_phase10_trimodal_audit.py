import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np,torch
from src.phase10_trimodal.math import log_weights,torch_cox,comparable_pairs
from src.phase10_trimodal.reference import reference_efron,components,uno_weights,metrics,censor_probability
from src.phase10_trimodal.features import prepare,validate_provenance,derangement
from src.phase10_trimodal.models import MechanismModel,IFA,PairAttention

torch.set_num_threads(2)

class MathematicalAudit(unittest.TestCase):
    def test_efron_independent_extremes_and_gradient(self):
        t=np.array([1,1,1,2,3,4.]);e=np.array([1,1,0,1,0,1]);v=np.array([1000,-1000,700,-900,100,0.])
        r=torch.tensor(v,requires_grad=True,dtype=torch.float64);la=log_weights(t,e,'cpu');loss=torch_cox(r,la,torch.tensor(e))
        self.assertAlmostEqual(loss.item(),reference_efron(t,e,v),places=9);loss.backward();self.assertTrue(torch.isfinite(r.grad).all())
        for j in range(len(v)):
            vp=v.copy();vm=v.copy();vp[j]+=1e-4;vm[j]-=1e-4
            self.assertAlmostEqual(r.grad[j].item(),(reference_efron(t,e,vp)-reference_efron(t,e,vm))/2e-4,places=5)
    def test_zero_events_rejected(self):
        with self.assertRaises(ValueError):log_weights([1,2],[0,0],'cpu')
    def test_ridge_ill_conditioned_convergence(self):
        from src.phase10_trimodal.math import fit_ridge
        rng=np.random.default_rng(88);x=rng.normal(size=(50,8))*np.array([1,2,3,4,50,100,200,500]);t=np.arange(50)+1.;e=np.tile([1,0],25)
        w,info=fit_ridge(x,t,e,.01);self.assertLess(info['gradient_max'],2e-5);self.assertTrue(np.isfinite(w).all())
    def test_hand_harrell_and_comparable_ties(self):
        t=np.array([1,1,2]);e=np.array([1,0,1]);r=np.array([2,2,1.]);self.assertEqual(components(t,e,r),(1.5,2.))
        i,j=comparable_pairs([1,1,1],[1,1,0]);self.assertEqual(set(zip(i,j)),{(0,2),(1,2)})
    def test_metrics_against_sksurv(self):
        from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
        tt=np.array([1,1,2,3,4,5,6,7,8,9,10.]);te=np.array([1,0,1,0,1,1,0,1,0,1,0],bool)
        t=np.array([1,2,2,4,6,7.]);e=np.array([1,0,1,1,0,1],bool);r=np.array([2.,1,1,3,0,2])
        y=lambda ev,tm:np.array(list(zip(ev,tm)),dtype=[('event','?'),('time','<f8')])
        n,d=components(t,e,r);self.assertAlmostEqual(n/d,concordance_index_censored(e,t,r)[0])
        w=uno_weights(tt,te,t,e,8.);n,d=components(t,e,r,w)
        self.assertAlmostEqual(n/d,concordance_index_ipcw(y(te,tt),y(e,t),r,tau=8)[0],places=10)
    def test_micro_and_bootstrap_duplicates(self):
        t=np.array([1,2,1,2.]);e=np.array([1,0,1,0]);r=np.array([1,0,0,1]);f=np.array([1,1,2,2]);w=np.array([1,0,4,0.])
        m=metrics(t,e,r,f,w);self.assertEqual(m['Harrell'],.5);self.assertEqual(m['Uno60'],.2)
        counts=np.array([2,1,1,2]);explicit=np.repeat(np.arange(4),counts)
        a=metrics(t,e,r,f,w,counts);b=metrics(t[explicit],e[explicit],r[explicit],f[explicit],w[explicit])
        self.assertEqual(a,b)


def fake_data():
    rng=np.random.default_rng(124);n=48;genes=1000
    d={'rna_genes':rng.normal(size=(n,genes)),'cna_genes':rng.normal(size=(n,genes)),'rna_gene_symbols':np.array([f'G{j}' for j in range(genes)]),'cna_gene_symbols':np.array([f'G{j}' for j in range(genes)]),'static':rng.normal(size=(n,32)),'titan':rng.normal(size=(n,40)),'raw_rna':np.zeros((n,genes)),'cancer_type':np.array(['COAD']*n)}
    for k in ['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']:d['clinical_'+k]=np.array(['60']*n if k=='AGE' else ['I','II']*(n//2))
    groups={f's{k}':[f'G{j}' for j in range(k*60,(k+1)*60)] for k in range(8)}
    return d,groups

class BoundaryAudit(unittest.TestCase):
    def test_constructor_rejects_disguised_label_source(self):
        from src.phase10_trimodal.features import clinical
        d,g=fake_data();d['clinical_normal_score']=np.array(['1','0']*24)
        with self.assertRaises(ValueError):clinical(d,np.arange(36),requested_fields=['normal_score'],provenance={'normal_score':{'source_field':'DSS_event','source_kind':'diagnosis_clinical','label_derived':True}})
    def test_reused_metadata_container_cannot_override_identity(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('phase10_runner',Path(__file__).resolve().parents[1]/'scripts/run_phase10_trimodal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);m.save_cell(out,'D0',31,1,np.array([0,1]),np.array([1.,2.]),{'prediction_sha256':'WRONG_B0','private_manifest_sha256':'WRONG_B0'},{'split':'test'})
            record=json.loads((out/'cells/D0_seed31_fold1.json').read_text());self.assertNotEqual(record['prediction_sha256'],'WRONG_B0');self.assertNotEqual(record['private_manifest_sha256'],'WRONG_B0')
    def test_test_perturbation_fitted_state_invariant(self):
        d,g=fake_data();train=np.arange(36);a,ha=prepare(d,train,g)
        d2=copy.deepcopy(d)
        for key in ['rna_genes','cna_genes','static','titan']:d2[key][36:]=1e12
        for key in ['clinical_SEX','clinical_PATH_T_STAGE']:d2[key]=d2[key].astype('U100');d2[key][36:]='unseen_test_category'
        b,hb=prepare(d2,train,g);self.assertEqual(ha,hb)
        for k in a:np.testing.assert_array_equal(a[k][train],b[k][train])
    def test_future_source_even_benign_name_rejected(self):
        with self.assertRaises(ValueError):validate_provenance(['normal_score'],{'normal_score':{'source_field':'DSS_event','source_kind':'diagnosis_clinical','label_derived':False}})
        with self.assertRaises(ValueError):validate_provenance(['AGE'],{'AGE':{'source_field':'AGE','source_kind':'diagnosis_clinical','label_derived':True}})
    def test_wrong_pair_split_boundary(self):
        d,g=fake_data();ix=np.arange(24);source=derangement(d,ix,7);self.assertFalse(np.any(source==ix));self.assertEqual(set(source),set(ix))
    def test_cna_separate_polarities(self):
        dosage=np.array([1,1,-1,-1.]);self.assertEqual(dosage.mean(),0);self.assertEqual(np.maximum(dosage,0).mean(),.5);self.assertEqual(np.maximum(-dosage,0).mean(),.5)
    def test_teacher_heldout_times_cannot_change_own_prediction(self):
        from src.phase10_trimodal.train import teacher_crossfit
        d,g=fake_data();d={k:np.tile(v,(3,1)) if v.ndim==2 else np.tile(v,3) for k,v in d.items() if k not in ['rna_gene_symbols','cna_gene_symbols']}
        d['rna_gene_symbols']=np.array([f'G{j}' for j in range(1000)]);d['cna_gene_symbols']=d['rna_gene_symbols'].copy()
        d['DSS_time']=np.arange(144,dtype=float)+1;d['DSS_event']=np.tile([1,0,0,0],36)
        cfg={'split_seed':73,'ridge_grid':[.1,1.]};fit=np.arange(120)
        a,audit=teacher_crossfit(d,fit,g,cfg);hold=np.array(audit[0]['heldout_indices']);d2=copy.deepcopy(d);d2['DSS_time'][hold]+=10000
        b,_=teacher_crossfit(d2,fit,g,cfg);np.testing.assert_array_equal(a[hold],b[hold])

class MechanismAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123);cls.c=torch.randn(4,12);cls.r=torch.randn(4,8,64);cls.p=torch.randn(4,2,9,768);cls.mask=torch.ones(4,2,9,dtype=torch.bool)
    def test_all_mechanisms_forward_gradient(self):
        for arm in ['A2','A5','A6','B0','B1','B2','B3','B4','B5','C0','C1','C2','C3','C4','C5','D1','D2','D3','E0','E1']:
            with self.subTest(arm=arm):
                torch.manual_seed(3);model=MechanismModel(arm,12);model.eval();risk=model(self.c,self.r,self.p,self.mask);risk.square().sum().backward()
                grad=model.patch_projection[1].weight.grad
                self.assertIsNotNone(grad);self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.norm().item(),0)
                if arm not in ['A2','A5','B1','C2']:
                    self.assertGreater(model.rna_projection[0][0].weight.grad.norm().item(),0)
    def test_patch_joint_permutation_and_slide_order(self):
        m=MechanismModel('B0',12).eval();a=m(self.c,self.r,self.p,self.mask)
        perm=torch.randperm(9);b=m(self.c,self.r,self.p[:,:,perm],self.mask[:,:,perm]);torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
        b=m(self.c,self.r,self.p.flip(1),self.mask.flip(1));torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    def test_padding_values_cannot_change_any_arm_output(self):
        mask=self.mask.clone();mask[:,:,-2:]=False;p=self.p.clone();q=p.clone();q[:,:,-2:]=1e5
        for arm in ['A2','A5','A6','B0','C0','E1']:
            m=MechanismModel(arm,12).eval();a=m(self.c,self.r,p,mask);b=m(self.c,self.r,q,mask);torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
    def test_gate_perturbation_changes_full_risk(self):
        m=MechanismModel('E1',12).eval();a=m(self.c,self.r,self.p,self.mask)
        with torch.no_grad():m.ifa.conf_p.bias.add_(4.);m.ifa.conf_r.bias.sub_(4.)
        b=m(self.c,self.r,self.p,self.mask);self.assertGreater((a-b).abs().max().item(),1e-7)
    def test_gate_active_vs_source_dead(self):
        m=IFA(16).eval();r=torch.randn(3,8,16);p=torch.randn(3,10,16);mask=torch.ones(3,10,dtype=torch.bool)
        a=m(r,p,mask,gate=True);sum(v.square().sum() for v in a[:2]).backward();self.assertGreater(m.conf_r.weight.grad.norm().item(),0)
        m.zero_grad();b=m(r,p,mask,gate=True,source_behavior=True);sum(v.square().sum() for v in b[:2]).backward();self.assertIsNone(m.conf_r.weight.grad)
    def test_matched_budget(self):
        count=lambda a:sum(p.numel() for p in MechanismModel(a,12).parameters())
        self.assertEqual(len(set(count(a) for a in ['A2','A5','A6','B0','B1','B2','B3','B4','B5','C0','C1','C2','C3','C4','C5','D1','D2','D3'])),1)
        self.assertEqual(count('E0'),count('E1'))
    def test_constant_and_removed_rna_do_not_use_rna(self):
        for arm in ['A2','A5','B1','C2']:
            m=MechanismModel(arm,12).eval();a=m(self.c,self.r,self.p,self.mask);b=m(self.c,self.r*100+99,self.p,self.mask);torch.testing.assert_close(a,b,atol=0,rtol=0)
        m=MechanismModel('B0',12).eval();a=m(self.c,self.r,self.p,self.mask);b=m(self.c,self.r.flip(0),self.p,self.mask);self.assertGreater((a-b).abs().max().item(),1e-7)
    def test_attention_order_and_qkv_against_manual(self):
        m=PairAttention(8).eval();m.ff=torch.nn.Identity();x=torch.randn(2,3,8);y=torch.randn(2,7,8);mask=torch.ones(2,7,dtype=torch.bool)
        a,b=m(x,y,mask);q,k,v=m.qkv(m.norm(torch.cat([x,y],1))).chunk(3,-1);q=q/(8**.5)
        ea=((q[:,:3]@k.transpose(-1,-2)).softmax(-1)@v).mean(1)
        eb=((q[:,3:]@k[:,:3].transpose(-1,-2)).softmax(-1)@v[:,:3]).mean(1)
        torch.testing.assert_close(a,ea);torch.testing.assert_close(b,eb)
    def test_determinism_dropout_checkpoint(self):
        torch.manual_seed(11);a=MechanismModel('B0',12);torch.manual_seed(11);b=MechanismModel('B0',12)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        a.eval();x=a(self.c,self.r,self.p,self.mask);y=a(self.c,self.r,self.p,self.mask);torch.testing.assert_close(x,y,atol=0,rtol=0)
        a.train();x=a(self.c,self.r,self.p,self.mask);y=a(self.c,self.r,self.p,self.mask);self.assertFalse(torch.equal(x,y))

if __name__=='__main__':unittest.main(verbosity=2)
