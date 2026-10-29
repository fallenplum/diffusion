"""Selection leakage, input separation, and real synthetic state replay tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.spgel_s5_hest import data,execution as run,protocol
from src.spgel_s5_hest.models import build_model,parameter_counts,paired_candidates
from src.spgel_hest.assets import read_json

TMP=protocol.ROOT/"tmp"

def tiny_cache(n=24,m=9):
    rng=np.random.default_rng(123)
    return dict(train_x=rng.normal(size=(n,256)).astype(np.float32),train_y=np.log1p(np.abs(rng.normal(size=(n,50)))).astype(np.float64),
        train_xy=np.column_stack((np.arange(n),np.arange(n)%4)),train_slide=np.zeros(n,np.int64),
        validation_x=rng.normal(size=(m,256)).astype(np.float32),validation_y=np.log1p(np.abs(rng.normal(size=(m,50)))).astype(np.float64),
        validation_xy=np.column_stack((np.arange(m),np.arange(m)%3)),validation_slide=np.zeros(m,np.int64))

def image_bank(model,c):
    ctx,_=run._context(c,"train");x=torch.from_numpy(c["train_x"])
    h=torch.relu(model.input_projection(x));local=h[ctx["indices"]]
    key=model.image_key(local);value=model.image_value(local)
    scores=torch.einsum("qd,nkd->nqk",model.queries,key)/(32**0.5)
    attn=torch.softmax(scores.masked_fill(~ctx["valid"][:,None,:],-torch.inf),-1)
    return dict(keys=key,values=value,slots=torch.einsum("nqk,nkd->nqd",attn,value),valid=ctx["valid"]),ctx,x

class S5Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_query_prediction_exactly_independent_of_auxiliary_expression(self):
        c=tiny_cache();ctx,_=run._context(c,"train");x=torch.from_numpy(c["train_x"]);y=torch.from_numpy(c["train_y"].astype(np.float32))
        model=build_model("SPGEL50_ADAPT")
        a=model(x,ctx,None);b=model(x,ctx,y);d=model(x,ctx,y.flip(0))
        self.assertTrue(torch.equal(a["prediction"],b["prediction"]))
        self.assertTrue(torch.equal(a["prediction"],d["prediction"]))
        loss=torch.nn.functional.mse_loss(b["prediction"],y)+0.1*b["auxiliary_loss"]
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.gene_value.weight.grad.norm()),0)

    def test_graph_identity_exact_initial_parameters_and_budget(self):
        a,b=build_model("GRAPHSAGE"),build_model("IDENTITY")
        self.assertEqual(parameter_counts(a),parameter_counts(b))
        self.assertTrue(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()))
        self.assertEqual(parameter_counts(a)["total"],72242)
        for arm in protocol.ARMS:
            m=build_model(arm)
            self.assertFalse(any(isinstance(x,(torch.nn.Dropout,torch.nn.BatchNorm1d)) for x in m.modules()))

    def test_multiple_slide_inner_split_and_spatial_blocks(self):
        xy=np.column_stack((np.arange(600),np.arange(600)%23))
        spec=np.repeat(np.arange(2),300)
        left,right,a=data.inner_indices(xy,spec)
        self.assertTrue(np.all(spec[left]==1));self.assertTrue(np.all(spec[right]==0))
        grid=np.indices((32,32)).reshape(2,-1).T
        l,r,b=data.inner_indices(grid,np.zeros(len(grid),np.int64))
        self.assertFalse(np.intersect1d(l,r).size)
        graph,idx,valid=data.graph_payload(grid[l],np.zeros(len(l),np.int64))
        self.assertEqual(graph.audit["cross_specimen_edges"],0)
        self.assertLess(idx[valid].max(),len(l))
        self.assertFalse(b["uses_expression_or_outcomes"])
        self.assertTrue(np.array_equal(data.inner_indices(grid,np.zeros(len(grid),np.int64))[0],l))

    def test_insufficient_or_degenerate_single_slide_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"NONDEGENERATE"):
            data.inner_indices(np.ones((600,2)),np.zeros(600,np.int64))
        with self.assertRaisesRegex(ValueError,"INFEASIBLE"):
            data.inner_indices(np.column_stack((np.arange(32),np.arange(32)%2)),np.zeros(32,np.int64))

    def test_inner_PCA_never_fits_validation_extremes(self):
        rng=np.random.default_rng(7);a=rng.normal(size=(600,600)).astype(np.float32);b=rng.normal(size=(19,600)).astype(np.float32)
        pipe,x,y=data.fit_inputs(a,b);other,x2,y2=data.fit_inputs(a,b+1000)
        self.assertEqual(pipe["PCA"]._fit_svd_solver,"randomized")
        for attr in ("mean_","components_","singular_values_"):
            self.assertTrue(np.array_equal(getattr(pipe["PCA"],attr),getattr(other["PCA"],attr)))
        self.assertTrue(np.array_equal(x,x2));self.assertFalse(np.array_equal(y,y2))
        self.assertTrue(np.allclose(pipe["scaler"].mean_,a.mean(0,dtype=np.float64)))

    def test_zero_optimizer_real_forward_backward_preserves_state(self):
        result=run._profile(tiny_cache())
        self.assertEqual(result["new_fits"],0);self.assertEqual(result["optimizer_steps"],0)
        self.assertEqual(len(result["units"]),5)
        self.assertTrue(all(x["model_state_unchanged"] for x in result["units"]))

    def test_each_fold_selects_its_own_inner_schedule_not_crossfold_pool(self):
        with tempfile.TemporaryDirectory(dir=TMP) as t:
            private=Path(t);cfg=dict(tasks=["COAD","READ"],folds=[0,1]);lock=dict(internal_contract_sha256="fixture",
                partitions=[dict(dataset=d,fold=f,cache_metadata_sha256="synthetic_cache") for d in ("COAD","READ") for f in (0,1)])
            expected={}
            for d in cfg["tasks"]:
                for f in cfg["folds"]:
                    winner=80 if f==0 else 320;expected[(d,f)]=winner
                    for arm in protocol.ARMS:
                        folder=private/"dev_cells"/protocol.name(d,f,arm);folder.mkdir(parents=True)
                        row=dict(epochs=1280,snapshots=[dict(epoch=e,validation_MSE=0.1 if e==winner else 9.0) for e in protocol.CHECKPOINTS])
                        protocol.write_json(folder/"COMPLETE.private.json",row)
            with patch.object(run,"_check",return_value=(cfg,lock,{})),patch.object(run,"_check_cell"):
                result=run.select_schedule("unused","unused",private)
            self.assertEqual({(r["dataset"],r["fold"]):r["selected_epoch"] for r in result["selections"]},expected)
            self.assertFalse(result["official_test_expression_loaded"])

    def test_official_test_open_rejected_until_all_final_cells_exist(self):
        with tempfile.TemporaryDirectory(dir=TMP) as t:
            private=Path(t);cfg=dict(tasks=["COAD","READ"],folds=[0,1]);lock=dict(internal_contract_sha256="fixture")
            with patch.object(run,"_check",return_value=(cfg,lock,{})),patch.object(run,"_final_lock",return_value=lock),patch.object(run,"load_partition") as loader:
                with self.assertRaises(FileNotFoundError):run.evaluate_final("unused","unused",private)
                loader.assert_not_called()
            self.assertFalse((private/"OFFICIAL_TEST_OPEN_GATE.private.json").exists())

    def test_cache_tamper_rejected_before_training(self):
        with tempfile.TemporaryDirectory(dir=TMP) as t:
            folder=Path(t)/"cache";c=tiny_cache();meta=data.save_cache(folder,c,{"synthetic_pipeline":True},{"synthetic_only":True})
            expected=dict(cache_metadata_sha256=protocol.sha256(folder/"CACHE.private.json"))
            loaded,_=data.load_cache(folder,expected);self.assertTrue(np.array_equal(loaded["train_x"],c["train_x"]))
            with (folder/"inputs.private.npz").open("ab") as stream:stream.write(b"changed")
            with self.assertRaisesRegex(ValueError,"CACHE_FILE_CHANGED"):data.load_cache(folder,expected)

    def test_aggregate_evaluation_cannot_change_after_cell_predictions(self):
        with tempfile.TemporaryDirectory(dir=TMP) as t:
            private=Path(t);rows=[];binding={}
            for d in ("COAD","READ"):
                for f in (0,1):
                    testfolder=private/"test_inputs"/protocol.name(d,f);testfolder.mkdir(parents=True)
                    (testfolder/"inputs.private.npz").write_bytes(b"synthetic_cache_hash_only")
                    (testfolder/"pipeline.private.joblib").write_bytes(b"synthetic_pipeline_hash_only")
                    protocol.write_json(testfolder/"CACHE.private.json",dict(synthetic=True,
                        input_file_sha256=protocol.sha256(testfolder/"inputs.private.npz"),pipeline_sha256=protocol.sha256(testfolder/"pipeline.private.joblib")))
                    for arm in protocol.ARMS:
                        cell=protocol.name(d,f,arm);folder=private/"final_cells"/cell;folder.mkdir(parents=True)
                        protocol.write_json(folder/"COMPLETE.private.json",dict(synthetic=True));binding[cell]=protocol.sha256(folder/"COMPLETE.private.json")
                        (folder/"official_test_prediction.private.npz").write_bytes(b"synthetic_hash_only_fixture_not_an_array")
                        row=dict(dataset=d,fold=f,arm=arm,COMPLETE_sha256=binding[cell],
                            test_prediction_sha256=protocol.sha256(folder/"official_test_prediction.private.npz"),
                            test_cache_metadata_sha256=protocol.sha256(testfolder/"CACHE.private.json"),
                            first_test_evaluation_after_all_twenty_final_fits=True,new_model_fits=0,synthetic_metric=1.0)
                        protocol.write_json(folder/"EVALUATION.private.json",row);rows.append(row)
            protocol.write_json(private/"OFFICIAL_TEST_OPEN_GATE.private.json",dict(status="OPENED_AFTER_ALL_20_FINAL_STATES_FROZEN",final_completion_sha256=binding))
            aggregate=dict(status="COMPLETE_TWENTY_FROZEN_FINAL_TEST_EVALUATIONS",units=rows,final_input_contract="fixture",
                test_gate_sha256=protocol.sha256(private/"OFFICIAL_TEST_OPEN_GATE.private.json"))
            path=private/"S5B_OFFICIAL_EVALUATION.private.json";protocol.write_json(path,aggregate)
            self.assertEqual(len(run._validated_evaluation(private,dict(internal_contract_sha256="fixture"))["units"]),20)
            aggregate["units"][0]["synthetic_metric"]=99.0
            import json
            path.write_text(json.dumps(aggregate),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"EVALUATION_CHANGED_FROM_ACTUAL_CELL"):
                run._validated_evaluation(private,dict(internal_contract_sha256="fixture"))

    def test_actual_synthetic_320step_three_state_replay(self):
        with tempfile.TemporaryDirectory(dir=TMP) as t:
            folder=Path(t);c=tiny_cache(n=16,m=7)
            r=run._train(c,"SPGEL50_ADAPT",320,folder,"synthetic_contract",True,
                dict(dataset="COAD",fold=0,input_cache_metadata_sha256="synthetic_cache"))
            self.assertEqual(r["optimizer_updates"],320)
            self.assertEqual([s["epoch"] for s in r["snapshots"]],[80,160,320])
            a,_=run._context(c,"train");b,_=run._context(c,"validation")
            for s in r["snapshots"]:
                model=build_model("SPGEL50_ADAPT")
                state=torch.load(folder/f"epoch{s['epoch']:03d}.private.pt",weights_only=True,map_location="cpu")
                model.load_state_dict(state["state_dict"],strict=True)
                with threadpool_limits(limits=2):left=run._predict(model,c["train_x"],a);right=run._predict(model,c["validation_x"],b)
                with np.load(folder/f"epoch{s['epoch']:03d}.private.npz",allow_pickle=False) as stored:
                    self.assertTrue(np.array_equal(left,stored["train_prediction"]))
                    self.assertTrue(np.array_equal(right,stored["validation_prediction"]))

    def test_new_guard_exact_dev_and_final_epoch_sets(self):
        import copy
        with tempfile.TemporaryDirectory(dir=TMP) as t:
            private=Path(t);folder=private/"dev_cells"/protocol.name("COAD",0,"QUERY4");folder.mkdir(parents=True)
            cache=private/"dev_inputs"/protocol.name("COAD",0);cache.mkdir(parents=True)
            protocol.write_json(cache/"CACHE.private.json",dict(synthetic=True));digest=protocol.sha256(cache/"CACHE.private.json")
            row=dict(dataset="COAD",fold=0,arm="QUERY4",phase="DEV",status="COMPLETE_ONE_DEVELOPMENT_FIT",input_cache_metadata_sha256=digest,
                internal_contract_sha256="fixture",new_model_fits=1,optimizer_updates=1280,epochs=1280,
                learning_curve=[dict(epoch=i) for i in range(1,1281)],official_test_expression_loaded=False,snapshots=[])
            for e in protocol.CHECKPOINTS:
                state=folder/f"epoch{e:03d}.private.pt";pred=folder/f"epoch{e:03d}.private.npz"
                state.write_bytes(f"synthetic_hash_only_state_{e}".encode());pred.write_bytes(f"synthetic_hash_only_prediction_{e}".encode())
                row["snapshots"].append(dict(epoch=e,state_sha256=protocol.sha256(state),predictions_sha256=protocol.sha256(pred)))
            expected=dict(dataset="COAD",fold=0,arm="QUERY4",phase="DEV",epochs=1280,input_cache_metadata_sha256=digest)
            run._check_cell(folder,row,"fixture",expected)
            for repeats in (3,5):
                bad=copy.deepcopy(row);bad["snapshots"]=[copy.deepcopy(row["snapshots"][0])]*repeats
                with self.assertRaisesRegex(ValueError,"EXACT_EPOCH_SET_CHANGED"):run._check_cell(folder,bad,"fixture",expected)
            bad=copy.deepcopy(row);bad.update(phase="FINAL",status="COMPLETE_ONE_FINAL_FIT",epochs=80,optimizer_updates=80,learning_curve=row["learning_curve"][:80])
            final_expected=dict(expected,phase="FINAL",epochs=80)
            for states in ([row["snapshots"][0],row["snapshots"][0]],[row["snapshots"][1]]):
                bad["snapshots"]=states
                with self.assertRaisesRegex(ValueError,"EXACT_EPOCH_SET_CHANGED"):run._check_cell(folder,bad,"fixture",final_expected)

    def test_new_guard_state_dataset_fold_phase_and_input_binding(self):
        row=dict(dataset="COAD",fold=0,arm="QUERY4",phase="DEV",input_cache_metadata_sha256="cache_a",internal_contract_sha256="contract_a")
        state=dict(schema="s5_hest_state_v1",dataset="COAD",fold=0,arm="QUERY4",phase="DEV",input_cache_metadata_sha256="cache_a",contract="contract_a",epoch=80)
        run._validate_state(state,row,80)
        for field,value in (("dataset","READ"),("fold",1),("phase","FINAL"),("input_cache_metadata_sha256","wrong_cache"),("epoch",160)):
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"CHECKPOINT_PHASE_DATASET_FOLD_OR_INPUT_CACHE_CHANGED"):
                run._validate_state(dict(state,**{field:value}),row,80)

    def test_new_guard_120_replays_must_be_unique_exact_states(self):
        rows=[]
        for stage,epochs in (("DEV",protocol.CHECKPOINTS),("FINAL",(80,))):
            for d in ("COAD","READ"):
                for f in (0,1):
                    for a in protocol.ARMS:
                        for e in epochs:rows.append(dict(stage=stage,dataset=d,fold=f,arm=a,epoch=e,state_sha256=f"synthetic_state_{len(rows)}",COMPLETE_sha256=f"synthetic_cell_{stage}_{d}_{f}_{a}"))
        expected={(r["stage"],r["dataset"],r["fold"],r["arm"],r["epoch"],r["state_sha256"],r["COMPLETE_sha256"]) for r in rows}
        run._validate_replay_set(dict(units=rows),expected)
        bad=list(rows);bad[1]=dict(rows[0])
        with self.assertRaisesRegex(ValueError,"EXACTLY_120_UNIQUE_STATES"):run._validate_replay_set(dict(units=bad),expected)
        bad=[dict(r) for r in rows];bad[-1]["state_sha256"]="wrong_state"
        with self.assertRaisesRegex(ValueError,"EXACTLY_120_UNIQUE_STATES"):run._validate_replay_set(dict(units=bad),expected)

    def test_new_graphsage_matches_exclude_self_mean_reference(self):
        c=tiny_cache(n=4);c["train_xy"]=np.asarray([[0,0],[1,0],[2,0],[0,0]])
        c["train_slide"]=np.asarray([0,0,0,1]);ctx,audit=run._context(c,"train")
        adjacency=ctx["neighbor_adjacency"].to_dense()
        self.assertTrue(torch.equal(adjacency.diag(),torch.zeros(4)))
        self.assertTrue(torch.equal(adjacency[3],torch.zeros(4)))
        self.assertTrue(torch.allclose(adjacency[:3].sum(1),torch.ones(3)))
        model=build_model("GRAPHSAGE");x=torch.from_numpy(c["train_x"]);b=model.baseline
        h=torch.relu(b.input_projection(x));reference=b.output_projection(torch.relu(b.self_projection(h)+b.neighbour_projection(adjacency@h)+b.hidden_bias))
        self.assertTrue(torch.allclose(model(x,ctx)["prediction"],reference,atol=1e-7,rtol=1e-6))
        self.assertEqual(audit["graphsage_messages"]["self_edges"],0)

    def test_new_actual_saved_state_contains_fit_cache_identity(self):
        with tempfile.TemporaryDirectory(dir=TMP) as t:
            folder=Path(t);c=tiny_cache(n=12,m=5);binding=dict(dataset="READ",fold=1,input_cache_metadata_sha256="synthetic_cache_binding")
            row=run._train(c,"SPGEL50_ADAPT",80,folder,"synthetic_contract",True,binding)
            state=torch.load(folder/"epoch080.private.pt",weights_only=True,map_location="cpu")
            run._validate_state(state,row,80)
            self.assertEqual(state["dataset"],"READ");self.assertEqual(state["fold"],1)
            self.assertEqual(state["input_cache_metadata_sha256"],"synthetic_cache_binding")

    def test_v3_factorized_gene_read_matches_explicit_raw_evidence(self):
        model=build_model("SPGEL50_ADAPT");g=torch.Generator().manual_seed(88)
        y=torch.rand((3,50),generator=g);observed=torch.rand((3,50),generator=g)>.25;observed[-1]=False
        q=torch.randn((3,4,32),generator=g)
        actual=model.gene_read(y,observed,q)
        safe=torch.where(observed,y,torch.zeros_like(y))
        evidence=model.gene_value(torch.stack((safe,observed.float()),dim=-1))+model.gene_id.weight[None,:,:]
        keys=model.gene_key(evidence);score=torch.einsum("nqd,ngd->nqg",q,keys)/(32**0.5)
        score=score.masked_fill(~observed[:,None,:],-torch.inf)
        score=torch.where(observed.any(-1)[:,None,None],score,torch.zeros_like(score))
        weights=torch.softmax(score,-1)*observed[:,None,:]
        reference=weights@evidence
        self.assertTrue(torch.allclose(actual["tokens"],reference,atol=1e-6,rtol=1e-5))
        self.assertTrue(torch.equal(actual["tokens"][-1],torch.zeros_like(actual["tokens"][-1])))

    def test_v3_every_candidate_is_reconditioned_matches_raw_loop(self):
        model=build_model("SPGEL50_ADAPT");c=tiny_cache(n=28);bank,ctx,x=image_bank(model,c)
        y=torch.from_numpy(c["train_y"].astype(np.float32))
        with torch.no_grad():
            model.gene_to_image.weight.copy_(torch.eye(32)*.7)
            model.image_to_gene.weight.copy_(torch.eye(32)*.4)
        left=torch.tensor([[0,0,0],[2,2,2]],dtype=torch.long);right=torch.tensor([[0,4,7],[2,11,20]],dtype=torch.long)
        for irows,grows in ((left,right),(right,left)):
            actual=model.paired_scores(bank,y,irows,grows,return_details=True)
            ref=torch.empty_like(actual["scores"])
            for a in range(irows.shape[0]):
                for b in range(irows.shape[1]):
                    i,j=int(irows[a,b]),int(grows[a,b]);observed=torch.ones(50)
                    raw=model.gene_value(torch.stack((y[j],observed),-1))+model.gene_id.weight
                    gkey=model.gene_key(raw)
                    gbase=torch.softmax(model.queries@gkey.T/(32**0.5),-1)@raw
                    qi=model.queries+model.gene_to_image(gbase)
                    iw=torch.softmax((qi@bank["keys"][i].T/(32**0.5)).masked_fill(~bank["valid"][i][None,:],-torch.inf),-1)
                    ci=iw@bank["values"][i]
                    qg=model.queries+model.image_to_gene(bank["slots"][i])
                    cg=torch.softmax(qg@gkey.T/(32**0.5),-1)@raw
                    ref[a,b]=torch.nn.functional.cosine_similarity(ci,cg,dim=-1,eps=1e-12).mean()
                    one=model.paired_scores(bank,y,irows[a:a+1,b:b+1],grows[a:a+1,b:b+1])["scores"][0,0]
                    self.assertTrue(torch.allclose(one,actual["scores"][a,b],atol=1e-6,rtol=1e-5))
            self.assertTrue(torch.allclose(actual["scores"],ref,atol=1e-6,rtol=1e-5))
        own=torch.arange(len(y))[:,None]
        positives=model.paired_scores(bank,y,own,own,return_details=True)
        wrong=torch.nn.functional.cosine_similarity(positives["conditioned_image_tokens"][:,0][left],positives["conditioned_gene_tokens"][:,0][right],dim=-1).mean(-1)
        correct=model.paired_scores(bank,y,left,right)["scores"]
        self.assertGreater(float((correct-wrong).abs().max()),1e-6)

    def test_v3_dual_anchor_ITC_matches_CE_and_both_condition_gradients(self):
        model=build_model("SPGEL50_ADAPT");c=tiny_cache(n=32);bank,ctx,x=image_bank(model,c)
        y=torch.from_numpy(c["train_y"].astype(np.float32));anchor,candidates=paired_candidates(len(y),3,seed=1)
        repeated=anchor[:,None].expand_as(candidates)
        f=model.paired_scores(bank,y,repeated,candidates)["scores"]/.1
        r=model.paired_scores(bank,y,candidates,repeated)["scores"]/.1
        truth=torch.zeros(len(anchor),dtype=torch.long)
        a=torch.nn.functional.cross_entropy(f,truth);b=torch.nn.functional.cross_entropy(r,truth)
        actual,audit=model.paired_itc(bank,y,epoch=3)
        self.assertTrue(torch.allclose(actual,(a+b)/2,atol=1e-7,rtol=1e-6))
        for loss in (a,b):
            grads=torch.autograd.grad(loss,(model.gene_to_image.weight,model.image_to_gene.weight),retain_graph=True)
            self.assertTrue(all(torch.isfinite(v).all() and float(v.norm())>0 for v in grads))
        self.assertFalse(hasattr(model,"gene_decoder"));self.assertEqual(audit["temperature"],.1)
        self.assertTrue(audit["candidate_condition_recomputed_both_directions"])

    def test_v3_train_only_sampler_cap_rotation_seed_and_no_self(self):
        a,c=paired_candidates(12000,1,seed=3);b,d=paired_candidates(12000,2,seed=3)
        self.assertEqual(tuple(c.shape),(512,9));self.assertEqual(c.numel(),4608)
        self.assertTrue(torch.equal(a,c[:,0]));self.assertTrue(torch.equal(b,(a+1)%12000))
        self.assertTrue(torch.all(c[:,1:]!=a[:,None]))
        self.assertTrue(all(len(torch.unique(row))==9 for row in c))
        again,repeat=paired_candidates(12000,12001,seed=3)
        self.assertTrue(torch.equal(a,again));self.assertTrue(torch.equal(c,repeat))
        small,csmall=paired_candidates(3,2,seed=3);self.assertEqual(tuple(csmall.shape),(3,3))

    def test_v3_prediction_no_RNA_value_or_gradient_dependency(self):
        model=build_model("SPGEL50_ADAPT");c=tiny_cache(n=24);bank,ctx,x=image_bank(model,c)
        y=torch.tensor(c["train_y"],dtype=torch.float32,requires_grad=True)
        a=model(x,ctx,None,epoch=1)["prediction"]
        b=model(x,ctx,y,epoch=1)["prediction"];d=model(x,ctx,y.flip(0),epoch=7)["prediction"]
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a,d))
        self.assertIsNone(torch.autograd.grad(b.sum(),y,allow_unused=True)[0])

    def test_v3_nonpaired_arms_keep_original_prediction_operators(self):
        from src.spgel_hest_neural.models import build_model as old
        c=tiny_cache(n=24);ctx,_=run._context(c,"train");x=torch.from_numpy(c["train_x"])
        for arm,old_arm in (("MLP","MLP"),("GRAPHSAGE","SPATIAL_GNN"),("IDENTITY","IDENTITY_GNN")):
            m=build_model(arm);reference=old(old_arm)
            adj=ctx["identity_adjacency"] if arm=="IDENTITY" else ctx["neighbor_adjacency"]
            self.assertTrue(torch.equal(m(x,ctx)["prediction"],reference(x,adj)))
        m=build_model("QUERY4");bank,_,_=image_bank(m,c)
        expected=m.output_projection(torch.relu(m.hidden(bank["slots"].flatten(1))))
        self.assertTrue(torch.equal(m(x,ctx)["prediction"],expected))

    def test_v3_parameter_split_matches_actual_prediction_and_ITC_gradients(self):
        model=build_model("SPGEL50_ADAPT");c=tiny_cache(n=32);bank,ctx,x=image_bank(model,c)
        y=torch.from_numpy(c["train_y"].astype(np.float32))
        out=model(x,ctx,y,epoch=2)
        torch.nn.functional.mse_loss(out["prediction"],y).backward()
        prefixes=("gene_id","gene_value","gene_key","gene_to_image","image_to_gene")
        for name,p in model.named_parameters():
            if name.startswith(prefixes):self.assertIsNone(p.grad)
        self.assertIsNotNone(model.queries.grad);self.assertGreater(float(model.input_projection.weight.grad.norm()),0)
        model.zero_grad(set_to_none=True);out=model(x,ctx,y,epoch=2)
        (torch.nn.functional.mse_loss(out["prediction"],y)+.1*out["auxiliary_loss"]).backward()
        for module in (model.gene_to_image,model.image_to_gene):
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all());self.assertGreater(float(module.weight.grad.norm()),0)
        self.assertEqual(parameter_counts(model),dict(total=69042,image_prediction=64242,training_only_gene_auxiliary=4800))

if __name__=="__main__":unittest.main()
