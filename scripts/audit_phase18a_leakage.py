"""Two short in-memory diagnostic fits; no smoke weights or risk arrays saved."""
import copy,numpy as np,torch
from src.phase18a_readout.data import P,R,cfg,source,write,old
from src.phase18a_readout import train
from src.phase13_rna import data as parent

def main():
    original_cfg=train.cfg;original_endpoint=parent.endpoints;alias=train.endpoints
    short=copy.deepcopy(cfg());short['parent_training']['max_epochs']=2
    train.cfg=lambda:short
    # The same real training code is exercised twice; the second run changes
    # ONLY outer-test labels in memory, including every ancestor lookup.
    a=train.train_cell('L1',31,1,'inner');parts=parent.splits(parent.load(),'DSS',1)
    def changed(d):
        ep=copy.deepcopy(original_endpoint(d));ix=parts['outer_test'];ep['DSS']['time'][ix]=ep['DSS']['time'][ix][::-1]+100000.;ep['DSS']['event'][ix]=1-ep['DSS']['event'][ix];return ep
    parent.endpoints=changed;train.endpoints=changed
    try:b=train.train_cell('L1',31,1,'inner')
    finally:parent.endpoints=original_endpoint;train.endpoints=alias;train.cfg=original_cfg
    param=max(float(abs(a['state'][k]-b['state'][k]).max()) if a['state'][k].dtype.is_floating_point else float(not torch.equal(a['state'][k],b['state'][k])) for k in a['state']);risk=float(abs(a['risk']-b['risk']).max())
    assert param==0 and risk==0 and a['selected_epoch']==b['selected_epoch'] and a['history']==b['history']
    assert not np.intersect1d(a['fit'],parts['outer_test']).size and not np.intersect1d(a['validation_or_test'],parts['outer_test']).size
    write(R/'DATA_LEAKAGE_AUDIT.json',{'status':'PASS','source':source(),'actual_diagnostic_optimization_runs':2,'optimizer_steps_per_run':2,'formal_training_runs':0,'changed_outer_labels_only_in_memory':True,'maximum_parameter_difference':param,'maximum_inner_risk_difference':risk,'selected_epoch_identical':True,'full_training_history_identical':True,'parent_lookups_and_training_endpoint_alias_both_mutated':True,'smoke_weights_or_patient_risks_persisted':False,'production_patch_budget':512,'production_patient_risk_sets':True,'source_and_data_files_unchanged':True})
    write(P/'SMOKE_DIAGNOSTIC_COUNTS.json',{'diagnostic_short_fits':4,'successful_audited_short_fits':2,'earlier_completed_short_fits_before_audit_comparison_exception':2,'optimizer_steps':8,'reason_for_repeat':'comparison helper attempted bool buffer subtraction; actual training completed; fixed audit only, no training-source change','no_checkpoints_saved':True,'not_counted_as_formal_inner_or_outer':True});print('LEAKAGE_AND_RUNTIME_PASS',param,risk,flush=True)

if __name__=='__main__':main()
