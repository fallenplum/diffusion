"""Publication checks exercise real failure boundaries with synthetic strings."""
import importlib.util
import json
import csv
from pathlib import Path
import re

import pytest

PATH = Path(__file__).resolve().parents[2]/'scripts/bgb_en_publication.py'
SPEC = importlib.util.spec_from_file_location('bgb_en_publication_tested', PATH)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def write_report(root, text):
    path = root/'reports/bgb_en/R1_RESULTS.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return 'reports/bgb_en/R1_RESULTS.md'


def test_actual_deny_value_is_blocked_without_printing_value(tmp_path):
    token = 'synthetic'+'private'+'barcode'
    relative = write_report(tmp_path, 'Summary '+token)
    with pytest.raises(publication.PublicationError) as error:
        publication.scan_file(tmp_path, relative, re.compile(token))
    assert token not in str(error.value)


def test_unknown_file_and_binary_payload_never_enter_manifest(tmp_path):
    relative = write_report(tmp_path, 'anonymous aggregate')
    with pytest.raises(publication.PublicationError, match='explicit'):
        publication.scan_file(tmp_path, 'reports/bgb_en/unknown_patient_output.md', re.compile('no-match'))
    (tmp_path/relative).write_bytes(b'aggregate\x00binary')
    with pytest.raises(publication.PublicationError, match='binary'):
        publication.scan_file(tmp_path, relative, re.compile('no-match'))


def test_credential_patterns_and_private_paths_are_rejected(tmp_path):
    payloads = ['hf_' + 'x'*30, '/root/'+'synthetic_private/run', 'connect.'+'westc.seetacloud.com']
    for payload in payloads:
        relative = write_report(tmp_path, payload)
        with pytest.raises(publication.PublicationError, match='privacy rule') as error:
            publication.scan_file(tmp_path, relative, re.compile('no-match'))
        assert payload not in str(error.value)


def test_public_json_allows_risk_summaries_and_rejects_row_arrays():
    publication.validate_public_json({'risk': {'SD': .1, 'mean': 0.}, 'rows': [{'arm': 'F0', 'H': .8}]})
    for payload in ({'risk': [.1, .2]}, {'patient_indices': [1]}, {'unspecified_matrix': list(range(512))}):
        with pytest.raises(publication.PublicationError):
            publication.validate_public_json(payload)


def test_safe_file_receipt_hashes_actual_bytes(tmp_path):
    relative = write_report(tmp_path, 'RUNNING; no completed scientific claim')
    receipt = publication.scan_file(tmp_path, relative, re.compile('no-match'))
    assert receipt['file'] == relative and receipt['sha256'] == publication.sha256(tmp_path/relative)
    assert receipt['bytes'] == (tmp_path/relative).stat().st_size


def test_cache_hash_exception_is_scoped_and_cannot_publish_coordinates():
    hashes = {key: 'a'*64 for key in ('features', 'coords', 'mask', 'owner')}
    value = {'new_feature_only_cache': {'array_sha256': hashes}}
    publication.validate_public_json(value, allow_cache_hashes=True)
    with pytest.raises(publication.PublicationError):
        publication.validate_public_json(value)
    hashes['coords'] = [[1, 2]]
    with pytest.raises(publication.PublicationError):
        publication.validate_public_json(value, allow_cache_hashes=True)


def test_r0_qualification_fails_when_loader_changes_after_audit(tmp_path):
    code = tmp_path/'src/bgb_en/data.py'
    code.parent.mkdir(parents=True)
    code.write_text('frozen outcome-free input contract\n')
    report = tmp_path/'reports/bgb_en'
    report.mkdir(parents=True)
    asset = {'schema': 'bgb_en_R0_asset_audit_v1', 'status': 'PASS_COHORT_AND_AVAILABLE_PATCH_ASSETS',
             'model_fits': 0, 'patient_identifiers_public': False, 'private_paths_public': False,
             'cohort': {'patients': 512, 'DSS_events': 65, 'outer_folds': 5,
                        'hashes': {'cohort_sha256': publication.CANONICAL_COHORT_SHA256,
                                   'patient_order_sha256': publication.CANONICAL_PATIENT_SHA256}},
             'CONCH': {'slides': 519, 'patches': 1278642, 'full_original_H5_hashes_verified': True},
             'UNI2': {'slides': 520, 'patches': 3295129}}
    cache = lambda uni: {'status': 'COMPLETE', 'outcome_reads': 0,
                         'coordinate_step_counts': {'512': 520} if uni else {'1024': 519}}
    topology = {'schema': 'bgb_en_R0_topology_qualification_v1',
                'status': 'PASS_INPUTS_FROZEN_BEFORE_FITS', 'outcome_reads': 0, 'model_fits': 0,
                'risk_labels_used_to_choose_patch_selector': False, 'main_selector': 'REGIONS512',
                'source_data_module_sha256': publication.sha256(code),
                'caches': {key: cache(key == 'patch_uni2') for key in (
                    'patch_regions', 'patch_grid', 'patch_rand', 'patch_diverse', 'patch_uni2')}}
    (report/'R0_ASSET_AUDIT.json').write_text(json.dumps(asset))
    (report/'R0_TOPOLOGY_QUALIFICATION.json').write_text(json.dumps(topology))
    assert publication.validate_r0_qualification(tmp_path)['status'] == 'PASS_FROZEN_R0_SOURCE_QUALIFICATION'
    code.write_text('modified after audit\n')
    with pytest.raises(publication.PublicationError, match='loader differs'):
        publication.validate_r0_qualification(tmp_path)


def test_diagnostic_png_requires_bound_anonymous_csv_and_searchable_svg(tmp_path):
    plot_path=PATH.parent/'bgb_en_plot.py'
    spec=importlib.util.spec_from_file_location('publication_plot_fixture',plot_path)
    plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/bgb_en_plot.py').write_bytes(plot_path.read_bytes())
    report=tmp_path/'reports/bgb_en';report.mkdir(parents=True)
    source=report/'R1_ARCH_ABLATION.csv'
    fields=['arm','status','Harrell','Uno60','parameters_mean','mean_unit_elapsed_seconds','peak_GPU_bytes_max']
    with source.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerow(dict(arm='P0',status='COMPLETE',Harrell=.75,Uno60=.7,parameters_mean=10,mean_unit_elapsed_seconds=1,peak_GPU_bytes_max=0))
    plot.Renderer(report,report/'figures').run()
    relative='reports/bgb_en/figures/R1_PERFORMANCE_COST.png'
    assert publication.scan_file(tmp_path,relative,re.compile('no-match'))['bytes']>100
    source.write_text(source.read_text()+'\n')
    with pytest.raises(publication.PublicationError,match='source CSV hash changed'):
        publication.scan_file(tmp_path,relative,re.compile('no-match'))
