import csv
import importlib.util
from pathlib import Path

PATH=Path(__file__).resolve().parents[2]/'scripts/bgb_en_plot.py'
SPEC=importlib.util.spec_from_file_location('bgb_plot_tested',PATH)
plot=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(plot)


def test_missing_data_produces_no_placeholder_performance_figures(tmp_path):
    result=plot.Renderer(tmp_path,tmp_path/'figures').run()
    assert result['plots']==[] and len(result['skipped'])==8
    assert not list((tmp_path/'figures').glob('*.png'))


def test_partial_arm_values_are_not_drawn_as_completed(tmp_path):
    path=tmp_path/'R1_ARCH_ABLATION.csv'
    fields=['arm','status','Harrell','Uno60','parameters_mean','mean_unit_elapsed_seconds','peak_GPU_bytes_max']
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerow(dict(arm='P0',status='COMPLETE',Harrell=.75,Uno60=.7,parameters_mean=10,mean_unit_elapsed_seconds=1,peak_GPU_bytes_max=0))
        writer.writerow(dict(arm='P1',status='RUNNING',Harrell=.99,Uno60=.99,parameters_mean=1000,mean_unit_elapsed_seconds=2,peak_GPU_bytes_max=100))
    renderer=plot.Renderer(tmp_path,tmp_path/'figures');renderer.performance('R1',path.name,'P0')
    assert len(renderer.plots)==1 and renderer.plots[0]['rows_drawn']==1
    assert {item['file'] for item in renderer.plots[0]['files']}=={'R1_PERFORMANCE_COST.png','R1_PERFORMANCE_COST.svg'}


def test_throughput_uses_only_positive_completed_primary_and_separates_ready_budgets(tmp_path):
    path=tmp_path/'PATCH_THROUGHPUT_AUDIT.csv'
    row=dict(root='core',stage='R1',arm='P1',role='PRIMARY',category='NEURAL',
        throughput_status='SUPPORTED_CONTRACT_DERIVED_LOGICAL_VALID_NODES',
        logical_valid_node_admissions_per_second=1000,logical_valid_node_admissions=10000,
        budget_per_patient=512,visual_feature_dim=768,cache_selector='REGIONS')
    rows=[row,{**row,'budget_per_patient':128},
        {**row,'role':'HISTORICAL'},{**row,'throughput_status':'NA_FAILED_OR_UNFINISHED_EPOCH_UNKNOWN'},
        {**row,'category':'REFERENCE'},{**row,'logical_valid_node_admissions':0},
        {**row,'budget_per_patient':''}]
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
    renderer=plot.Renderer(tmp_path,tmp_path/'figures');renderer.throughput()
    assert len(renderer.plots)==1 and renderer.plots[0]['rows_drawn']==2
    assert renderer.plots[0]['source_files'][0]['file']==path.name
    svg=(tmp_path/'figures/PATCH_THROUGHPUT.svg').read_text()
    assert 'B128' in svg and 'B512' in svg and 'Descriptive median' in svg
    assert 'not unique tissue' in svg
