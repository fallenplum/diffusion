"""Control semantics, parent equivalence, RNG, and grouping provenance gates."""
import unittest,numpy as np,torch
from src.phase11_semantic.models import SemanticModel
from src.phase12_c4.models import C4Model,convert_parent_state
from src.phase12_c4.features import shuffled,match_gene_set
from src.phase12_c4.train import derange,prepare_inputs


def example(kind='real',no_anchor=False,seed=31,batch=3):
    torch.manual_seed(seed);cm=np.eye(6,dtype=np.float32);gm=np.zeros((8,512),np.float32)
    for k in range(8):gm[k,k*64:(k+1)*64]=1
    model=C4Model(cm,gm,8,kind,no_anchor)
    x=[torch.randn(batch,6),torch.randn(batch,32),torch.randn(batch,512),torch.randn(batch,2,7,768),torch.ones(batch,2,7,dtype=torch.bool)]
    x[4][:,-1,-2:]=False
    return model,x,cm,gm


class Controls(unittest.TestCase):
    def test_parent_state_eval_exact(self):
        for seed in [31,47]:
            for batch in [2,5]:
                new,x,cm,gm=example(seed=seed,batch=batch);torch.manual_seed(seed);old=SemanticModel('C4',cm,gm,8)
                new.load_state_dict(convert_parent_state(old.state_dict()));old.eval();new.eval()
                torch.testing.assert_close(new(*x),old(*x),rtol=0,atol=0)
    def test_initialization_and_dropout_rng(self):
        model,x,cm,gm=example();state=torch.get_rng_state();torch.manual_seed(31);old=SemanticModel('C4',cm,gm,8)
        # Inputs already consumed RNG; compare constructors by separate reseeds.
        torch.manual_seed(31);real=C4Model(cm,gm,8);r1=torch.get_rng_state()
        torch.manual_seed(31);null=C4Model(cm,gm,8,'null');r2=torch.get_rng_state();self.assertTrue(torch.equal(r1,r2))
        real.train();null.train();torch.manual_seed(981);real(*x);after_real=torch.get_rng_state()
        torch.manual_seed(981);null(*x);self.assertTrue(torch.equal(after_real,torch.get_rng_state()))
    def test_parent_training_forward_and_gradients(self):
        new,x,cm,gm=example();torch.manual_seed(31);old=SemanticModel('C4',cm,gm,8);new.load_state_dict(convert_parent_state(old.state_dict()))
        torch.manual_seed(19);r1=old(*x);r1.square().mean().backward()
        torch.manual_seed(19);r2=new(*x);r2.square().mean().backward();torch.testing.assert_close(r1,r2,rtol=0,atol=0)
        newp=dict(new.named_parameters())
        for name,p in old.named_parameters():
            target=newp['rna.encoder.'+name[4:] if name.startswith('rna.') else name]
            if p.grad is None:self.assertIsNone(target.grad)
            else:torch.testing.assert_close(p.grad,target.grad,rtol=0,atol=0)
    def test_null_input_independence_and_live_constant(self):
        for seed in [31,47]:
            model,x,_,_=example('null',seed=seed);model.eval();r=model(*x);other=list(x);other[2]=x[2]*100+1000
            torch.testing.assert_close(r,model(*other),rtol=0,atol=0);r.square().mean().backward()
            self.assertGreater(model.rna.constant.grad.norm().item(),0);self.assertIsNone(model.rna.encoder.weight.grad)
    def test_real_RNA_S32_live(self):
        for seed in [31,47]:
            for batch in [2,4]:
                model,x,_,_=example(seed=seed,batch=batch);model.eval();r=model(*x)
                for index in [1,2]:
                    other=list(x);other[index]=x[index]*0;self.assertGreater((r-model(*other)).abs().max().item(),0)
                r.square().mean().backward();self.assertGreater(model.rna.encoder.weight.grad.norm().item(),0);self.assertGreater(model.concept.scalar[0].weight.grad.norm().item(),0)
    def test_no_anchor_no_linear_prediction(self):
        for kind in ['real','null']:
            model,x,_,_=example(kind,True);model.eval();r=model(*x)
            with torch.no_grad():model.anchor.weight.fill_(1e10)
            torch.testing.assert_close(r,model(*x),rtol=0,atol=0);self.assertEqual(len(list(model.anchor.parameters())),0)
            r.square().mean().backward();self.assertGreater(model.head[0].weight.grad.norm().item(),0)
    def test_padding_and_slide_order(self):
        for kind in ['real','null']:
            model,x,_,_=example(kind);model.eval();r=model(*x);changed=[v.clone() for v in x];changed[3][~x[4]]=1e12
            torch.testing.assert_close(r,model(*changed),rtol=0,atol=0)
            changed=list(x);changed[3]=x[3].flip(1);changed[4]=x[4].flip(1);torch.testing.assert_close(r,model(*changed),rtol=1e-6,atol=1e-6)
    def test_group_global_overlap_and_missing_bins(self):
        rng=np.random.default_rng(31);gm=(rng.random((8,512))<.2).astype(np.float32);gm[0,gm.sum(0)==0]=1
        mean=rng.normal(size=512);sd=np.exp(rng.normal(size=512));missing=np.r_[np.zeros(400),np.full(112,.05)]
        a,meta=shuffled(gm,mean,sd,missing,117301);b,_=shuffled(gm,mean,sd,missing,918703)
        np.testing.assert_array_equal(gm@gm.T,a@a.T);np.testing.assert_array_equal(gm.sum(1),a.sum(1));self.assertFalse(np.array_equal(a,b))
        perm=np.array(meta['permutation']);np.testing.assert_array_equal(missing,missing[perm])
    def test_wrong_split_and_singleton_fallback(self):
        d={'raw_rna':np.zeros((12,4)),'cancer_type':np.array(['C']*6+['R']*6)};d['raw_rna'][0,0]=np.nan
        source,meta=derange(d,np.arange(6),42);self.assertTrue(np.all(source!=np.arange(6)));self.assertEqual(set(source),set(range(6)));self.assertTrue(meta['fallback'])
        f={'R':np.arange(24).reshape(12,2)};result,maps=prepare_inputs(d,f,[('train',np.arange(6),101),('test',np.arange(6,12),303)],'wrong',7301)
        self.assertEqual(set(maps['test']['RNA']),set(range(6,12)));np.testing.assert_array_equal(f['R'],np.arange(24).reshape(12,2))
    def test_gene_set_match_disjoint_and_train_only_stats(self):
        rng=np.random.default_rng(15);eligible=np.arange(1800);selected=np.arange(512);annotated=eligible<600
        mean=rng.normal(size=1800);sd=np.exp(rng.normal(size=1800));missing=np.zeros(1800)
        control,meta=match_gene_set(eligible,selected,annotated,mean,sd,missing,428019)
        self.assertEqual(len(set(control)),512);self.assertFalse(annotated[control].any());self.assertFalse(set(control)&set(selected))
        again,_=match_gene_set(eligible,selected,annotated,mean,sd,missing,428019);np.testing.assert_array_equal(control,again)


if __name__=='__main__':unittest.main()
