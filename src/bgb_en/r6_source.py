"""Outcome-free, provenance-locked spatial source evidence for R6.

Raw counts and private composite identities never enter the public audit.
Feature normalization is stateless; source target scaling belongs to the later
training-donor-only trainer, not this loader.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from src.phase22_spatial_bridge.data_contract import read_h5ad_raw


PROGRAM_ORDER=("iCMS2","iCMS3","FAP_fibroblast_associated_top25","MARCO_macrophage_associated_top25")
SOURCE_VALUE_UNIT="float64_log1p_10000_times_source_count_over_full_measured_gene_library"
FEATURE_UNIT="float32_per_row_L2_norm_division_reject_norm_le_1e-12"


def require(value: Any,code: str) -> None:
    if not bool(value):raise ValueError(code)


def sha256(path: str|Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True,repr=False)
class SourcePaths:
    prepared_data: Path
    prepared_manifest: Path
    programs: Path
    fresh_alignment_certificate: Path
    raw_development_root: Path


@dataclass(frozen=True)
class SourceContract:
    fresh_certificate_sha256: str
    prepared_data_sha256: str="b5a2a44862123f629314cb28ad66abae437d7d08e923fb6128f82f7e9059c025"
    prepared_manifest_sha256: str="9cc2f840b8010eb1881831e971682780461607437bb220b1a6d7a0a39efafeb0"
    program_definition_sha256: str="f8031c03cbb9ec2409d72c799129a6de74f080860afbc81cb30fb187d2560143"
    qualified_stage_a_manifest_sha256: str="6f40062efacecc8a43a17ae6fa34ffa22f3b7b98ab2a5db874e1ab5a691da27f"
    primary_spots: int=13011
    sections: int=12
    donors: int=6
    full_source_genes: int=36601
    radius_um: float=300.
    maximum_neighbors: int=16
    program_replay_tolerance: float=2e-12

    def __post_init__(self):
        require(all(isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) for v in
            (self.fresh_certificate_sha256,self.prepared_data_sha256,self.prepared_manifest_sha256,
             self.program_definition_sha256,self.qualified_stage_a_manifest_sha256)),
            'SOURCE_CONTRACT_REQUIRES_EXACT_SHA256')
        require(0<self.program_replay_tolerance<=2e-12,'SOURCE_REPLAY_TOLERANCE_CANNOT_BE_RELAXED')
        require(np.isfinite(self.radius_um) and self.radius_um>0 and 1<=self.maximum_neighbors<=16,
            'SOURCE_SPATIAL_CONTRACT_INVALID')


def composite_identity(section,barcode,donor):
    section=np.asarray(section).astype(str);barcode=np.asarray(barcode).astype(str);donor=np.asarray(donor).astype(str)
    require(section.ndim==1 and barcode.shape==donor.shape==section.shape and len(section)>0,'BAD_COMPOSITE_IDENTITY_SHAPES')
    pairs=list(zip(section.tolist(),barcode.tolist()))
    require(len(set(pairs))==len(pairs),'DUPLICATE_SECTION_BARCODE_IDENTITY')
    section_names=np.unique(section);donor_names=np.unique(donor)
    for name in section_names:
        require(len(np.unique(donor[section==name]))==1,'ONE_SECTION_CROSSES_DONORS')
    sc=np.searchsorted(section_names,section).astype(np.int64)
    dc=np.searchsorted(donor_names,donor).astype(np.int64)
    return sc,dc,dict(composite_identity_sha256=canonical_hash(pairs),
        rows=len(pairs),unique_composite_keys=len(set(pairs)),
        globally_unique_barcode_strings=len(set(barcode.tolist())),
        sections=len(section_names),donors=len(donor_names),
        identity_key='section_plus_exact_barcode_never_global_barcode_only')


def normalize_features(raw):
    x=np.asarray(raw)
    require(x.ndim==2 and x.dtype==np.float32 and np.isfinite(x).all(),'FEATURES_MUST_BE_FINITE_FLOAT32')
    norm=np.linalg.norm(x,axis=1)
    require(np.isfinite(norm).all() and np.all(norm>1e-12),'ZERO_OR_NONFINITE_FEATURE_NORM')
    unit=x/norm[:,None]
    error=float(np.max(np.abs(np.linalg.norm(unit,axis=1)-1)))
    require(error<=2e-6,'FLOAT32_FEATURE_UNIT_REPLAY_FAILED')
    return unit,dict(unit=FEATURE_UNIT,raw_norm_median=float(np.median(norm)),
        normalized_max_abs_unit_error=error,normalization_is_stateless=True)


def program_index(programs):
    require(bool(programs) and len({p['id'] for p in programs})==len(programs),'DUPLICATE_OR_EMPTY_PROGRAMS')
    for p in programs:
        u,d=p['up_genes'],p['down_genes']
        require(bool(u) and len(u+d)==len(set(u+d)) and all(isinstance(g,str) and g and g==g.strip() for g in u+d),
            'BAD_SIGNED_PROGRAM_MEMBERSHIP')
    genes=np.asarray(sorted({g for p in programs for g in p['up_genes']+p['down_genes']}),dtype=str)
    lookup={g:i for i,g in enumerate(genes)}
    up=np.zeros((len(programs),len(genes)),bool);down=np.zeros_like(up)
    for k,p in enumerate(programs):
        up[k,[lookup[g] for g in p['up_genes']]]=True
        down[k,[lookup[g] for g in p['down_genes']]]=True
    return genes,up,down


def measured_gene_bank(raw_counts,raw_gene_names,requested_genes):
    names=np.asarray(raw_gene_names).astype(str);requested=np.asarray(requested_genes).astype(str)
    require(names.ndim==requested.ndim==1 and len(set(names))==len(names) and len(set(requested))==len(requested),
        'DUPLICATE_GENE_VOCABULARY')
    require(raw_counts.ndim==2 and raw_counts.shape[1]==len(names),'RAW_GENE_MATRIX_SHAPE_MISMATCH')
    values=raw_counts.data if sparse.issparse(raw_counts) else np.asarray(raw_counts)
    require(np.isfinite(values).all() and np.all(values>=0) and np.all(values==np.floor(values)),
        'RAW_SOURCE_REQUIRES_NONNEGATIVE_INTEGER_COUNTS')
    matrix=sparse.csr_matrix(raw_counts,dtype=np.float64)
    library=np.asarray(matrix.sum(1)).ravel()
    require(np.all(library>0) and np.all(library<2**53) and np.isfinite(library).all(),'BAD_FULL_GENE_LIBRARY')
    lookup={g:i for i,g in enumerate(names)}
    present=np.asarray([g in lookup for g in requested],bool)
    target_cols=np.flatnonzero(present)
    selected=matrix[:,[lookup[g] for g in requested[present]]].tocoo()
    bank=sparse.csr_matrix((selected.data,(selected.row,target_cols[selected.col])),shape=(matrix.shape[0],len(requested)))
    log_values=np.log1p(bank.toarray()*(10000./library[:,None]))
    measured=np.broadcast_to(present,(matrix.shape[0],len(requested))).copy()
    detected=np.asarray((matrix>0).sum(1)).ravel().astype(np.int64)
    return bank,measured,log_values,library,detected


def scores_from_gene_bank(log_values,measured,up,down,*,minimum_genes=5,minimum_coverage=.95):
    x=np.asarray(log_values,np.float64);mask=np.asarray(measured,bool)
    require(x.ndim==2 and mask.shape==x.shape and up.shape==down.shape and up.shape[1]==x.shape[1],
        'PROGRAM_BANK_SHAPES_DIFFER')
    require(np.isfinite(x).all(),'PROGRAM_BANK_NONFINITE_VALUES')
    scores=np.zeros((len(x),len(up)),np.float64);valid=np.zeros_like(scores,dtype=bool)
    for k in range(len(up)):
        mu=mask&up[k];md=mask&down[k]
        nu=mu.sum(1);nd=md.sum(1);fixed=int(up[k].sum()+down[k].sum())
        require(fixed>0,'EMPTY_PROGRAM_MEMBERSHIP')
        available=(nu+nd)/fixed
        ok=(nu>=minimum_genes)&((nd>=minimum_genes) if down[k].any() else True)&(available>=minimum_coverage)
        u=np.sum(np.where(mu,x,0.),axis=1)/np.maximum(nu,1)
        d=np.sum(np.where(md,x,0.),axis=1)/np.maximum(nd,1)
        scores[:,k]=np.where(ok,u-d,0.);valid[:,k]=ok
    return scores,valid


def spatial_graph(features_unit,coords_um,section_codes,*,radius_um=300.,maximum_neighbors=16):
    x=np.asarray(features_unit);coords=np.asarray(coords_um,np.float64);section=np.asarray(section_codes)
    require(x.ndim==2 and coords.shape==(len(x),2) and section.shape==(len(x),),'BAD_SPATIAL_SOURCE_SHAPES')
    require(x.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(coords).all(),'NONFINITE_SPATIAL_SOURCE')
    require(np.max(np.abs(np.linalg.norm(x,axis=1)-1))<=2e-6,'SPATIAL_INPUT_NOT_UNIT_FEATURES')
    require(0<radius_um and 1<=maximum_neighbors<=16,'BAD_LOCKED_NEIGHBOR_LIMIT')
    indices=np.full((len(x),maximum_neighbors),-1,np.int64);counts=np.zeros(len(x),np.int64)
    mean=np.zeros_like(x)
    for identity in np.unique(section):
        rows=np.flatnonzero(section==identity);xy=coords[rows]
        require(len(np.unique(xy,axis=0))==len(xy),'DUPLICATE_WITHIN_SECTION_COORDINATES')
        tree=cKDTree(xy)
        for local,row in enumerate(rows):
            candidates=np.asarray(tree.query_ball_point(xy[local],radius_um),np.int64)
            distance=np.sum((xy[candidates]-xy[local])**2,axis=1)
            keep=distance<=radius_um**2;candidates,distance=candidates[keep],distance[keep]
            selected=rows[candidates[np.lexsort((rows[candidates],distance))[:maximum_neighbors]]]
            require(len(selected)>0 and selected[0]==row and np.all(section[selected]==identity),'CROSS_SECTION_OR_MISSING_SELF_EDGE')
            indices[row,:len(selected)]=selected;counts[row]=len(selected)
            mean[row]=np.mean(x[selected],axis=0,dtype=np.float64).astype(np.float32)
    norm=np.linalg.norm(mean,axis=1)
    require(np.all(norm>1e-12),'NEIGHBOR_UNIT_MEAN_HAS_ZERO_NORM')
    mean_unit=mean/norm[:,None]
    return indices,counts,mean,mean_unit


class SourceSpotDataset:
    def __init__(self,*,features_unit,counts,measured_mask,log1p_cp10k,program_targets,tissue_y,program_up,
                 program_down,gene_symbols,coords_um,section_codes,donor_codes,library_size,
                 n_detected_genes,neighbors,neighbor_counts,neighbor_mean_of_unit,neighbor_mean_unit,audit):
        self.features_unit=features_unit;self.counts=counts;self.measured_mask=measured_mask
        self.log1p_cp10k=log1p_cp10k;self.program_targets=program_targets;self.tissue_y=tissue_y
        self.program_up=program_up;self.program_down=program_down;self.gene_symbols=gene_symbols
        self.coords_um=coords_um;self.section_codes=section_codes;self.donor_codes=donor_codes
        self.library_size=library_size;self.n_detected_genes=n_detected_genes
        self.neighbors=neighbors;self.neighbor_counts=neighbor_counts
        self.neighbor_mean_of_unit=neighbor_mean_of_unit;self.neighbor_mean_unit=neighbor_mean_unit
        self._audit=dict(audit)

    def __len__(self):return len(self.features_unit)

    def __getitem__(self,index):return self.private_item(index)

    def __repr__(self):return f'SourceSpotDataset(spots={len(self)}, feature_dim={self.features_unit.shape[1]}, gene_union={len(self.gene_symbols)})'

    def public_audit(self):return json.loads(json.dumps(self._audit,allow_nan=False))

    def model_gene_values(self,indices):
        return self.log1p_cp10k[np.asarray(indices,np.int64)].astype(np.float32)

    def private_item(self,index):
        require(isinstance(index,(int,np.integer)) and 0<=index<len(self),'SOURCE_ROW_OUT_OF_BOUNDS')
        return dict(feature_unit=self.features_unit[index],raw_gene_counts=self.counts[index].toarray()[0],
            gene_log1p_cp10k=self.log1p_cp10k[index],measured_mask=self.measured_mask[index],
            program_targets=self.program_targets[index],tissue_y=self.tissue_y[index],coords_um=self.coords_um[index],
            section_code=self.section_codes[index],donor_code=self.donor_codes[index],
            neighbor_indices=self.neighbors[index,:self.neighbor_counts[index]])


def load_source_spots(paths: SourcePaths,contract: SourceContract) -> SourceSpotDataset:
    require(len(contract.fresh_certificate_sha256)==64,'PRELOCKED_FRESH_CERTIFICATE_SHA_REQUIRED')
    expected=((paths.prepared_data,contract.prepared_data_sha256),
        (paths.prepared_manifest,contract.prepared_manifest_sha256),
        (paths.programs,contract.program_definition_sha256),
        (paths.fresh_alignment_certificate,contract.fresh_certificate_sha256))
    for path,digest in expected:require(sha256(path)==digest,'SOURCE_INPUT_SHA_MISMATCH')
    manifest=json.loads(Path(paths.prepared_manifest).read_text(encoding='utf-8-sig'))
    certificate=json.loads(Path(paths.fresh_alignment_certificate).read_text(encoding='utf-8-sig'))
    definitions=json.loads(Path(paths.programs).read_text(encoding='utf-8-sig'))
    require(manifest['status']=='PASS_PREPARED_SIGNED_PROGRAMS' and
        manifest['stage_a_manifest_sha256']==contract.qualified_stage_a_manifest_sha256,'UNQUALIFIED_PRIMARY_MANIFEST')
    require(certificate['status']=='PASS_FRESH_RAW_GENE_VISUAL_PRIMARY_ALIGNMENT' and
        certificate['source_array_data_sha256']==contract.prepared_data_sha256 and
        certificate['stage_b_manifest_sha256']==contract.prepared_manifest_sha256 and
        certificate['fixed_program_definition_sha256']==contract.program_definition_sha256 and
        certificate['stage_a_qualified_manifest_sha256']==contract.qualified_stage_a_manifest_sha256 and
        certificate['TCGA_DSS_read'] is False and certificate['original_ST_test_expression_read'] is False,
        'FRESH_ALIGNMENT_CERTIFICATE_MISMATCH')
    programs=definitions['programs'];require(tuple(p['id'] for p in programs)==PROGRAM_ORDER,'FOUR_PROGRAM_ORDER_CHANGED')
    genes,up,down=program_index(programs)
    with np.load(paths.prepared_data,allow_pickle=False) as archive:
        required=('X','program_y','tissue_y','donor','section','barcodes','coords_um','library_size','n_detected_genes')
        require(set(required)<=set(archive.files),'PRIMARY_ARRAYS_MISSING')
        data={k:np.array(archive[k],copy=True) for k in required}
    require(data['X'].shape==(contract.primary_spots,1536),'PRIMARY_FEATURE_SHAPE_CHANGED')
    require(data['program_y'].shape==(contract.primary_spots,4) and np.isfinite(data['program_y']).all(),
        'PRIMARY_PROGRAM_TARGETS_INVALID')
    require(data['tissue_y'].shape==(contract.primary_spots,) and data['tissue_y'].dtype.kind in 'iu' and
        np.isin(data['tissue_y'],[0,1,2]).all(),'PRIMARY_TISSUE_LABELS_INVALID')
    sc,dc,identity=composite_identity(data['section'],data['barcodes'],data['donor'])
    require(identity['sections']==contract.sections and identity['donors']==contract.donors,'SOURCE_GROUP_COUNTS_CHANGED')
    features,norm_audit=normalize_features(data['X'])
    measured=np.zeros((len(features),len(genes)),bool);log_values=np.zeros(measured.shape,np.float64)
    all_rows=[];all_cols=[];all_values=[];gene_order_sha=None;raw_hashes=[]
    sources=manifest['source_files']
    require(len(sources)==contract.sections and {r['section_id_private'] for r in sources}==set(data['section']),
        'RAW_DEVELOPMENT_SET_CHANGED')
    for source in sources:
        section=source['section_id_private'];raw_path=Path(paths.raw_development_root)/(section+'.h5ad')
        require(raw_path.resolve().is_relative_to(Path(paths.raw_development_root).resolve()) and
            sha256(raw_path)==source['raw_count_sha256'],'RAW_DEVELOPMENT_SHA_MISMATCH')
        raw=read_h5ad_raw(raw_path);raw_hashes.append(source['raw_count_sha256'])
        require(len(raw['gene_names'])==contract.full_source_genes,'FULL_SOURCE_GENE_COUNT_CHANGED')
        order_sha=canonical_hash(raw['gene_names'].tolist())
        require(gene_order_sha is None or gene_order_sha==order_sha,'SOURCE_GENE_ORDER_CHANGED');gene_order_sha=order_sha
        rows=np.flatnonzero(data['section']==section);wanted=data['barcodes'][rows].astype(str)
        lookup={value:i for i,value in enumerate(raw['barcodes'])}
        require(all(value in lookup for value in wanted),'RAW_PRIMARY_COMPOSITE_JOIN_FAILED')
        positions=np.asarray([lookup[value] for value in wanted],np.int64)
        require(np.array_equal(raw['barcodes'][positions],wanted),'RAW_PRIMARY_ROW_ORDER_FAILED')
        bank,mask,values,library,detected=measured_gene_bank(raw['matrix'][positions],raw['gene_names'],genes)
        require(np.array_equal(library,data['library_size'][rows]) and
            np.array_equal(detected,data['n_detected_genes'][rows]),'FULL_LIBRARY_OR_DEPTH_REPLAY_FAILED')
        measured[rows]=mask;log_values[rows]=values
        coordinate=bank.tocoo();all_rows.append(rows[coordinate.row]);all_cols.append(coordinate.col);all_values.append(coordinate.data)
    counts=sparse.csr_matrix((np.concatenate(all_values),(np.concatenate(all_rows),np.concatenate(all_cols))),shape=measured.shape)
    replay,valid=scores_from_gene_bank(log_values,measured,up,down)
    require(valid.all(),'SOURCE_PROGRAM_MEASUREMENT_QUALIFICATION_FAILED')
    error=float(np.max(np.abs(replay-data['program_y'])))
    require(error<=contract.program_replay_tolerance,'SOURCE_PROGRAM_REPLAY_TOLERANCE_FAILED')
    ni,nc,mean,mean_unit=spatial_graph(features,data['coords_um'],sc,radius_um=contract.radius_um,
        maximum_neighbors=contract.maximum_neighbors)
    audit=dict(schema='bgb_en_r6_source_dataset_v1',status='PASS_SOURCE_SPOT_DATASET_NO_MODEL_FIT',
        primary_spots=len(features),source_donors=contract.donors,source_sections=contract.sections,
        tissue_class_counts=np.bincount(data['tissue_y'],minlength=3).tolist(),
        full_source_genes=contract.full_source_genes,fixed_program_gene_union=len(genes),
        signed_membership_edges=int(up.sum()+down.sum()),programs=list(PROGRAM_ORDER),
        program_definition_sha256=contract.program_definition_sha256,prepared_data_sha256=contract.prepared_data_sha256,
        prepared_manifest_sha256=contract.prepared_manifest_sha256,fresh_certificate_sha256=contract.fresh_certificate_sha256,
        gene_union_order_sha256=canonical_hash(genes.tolist()),raw_gene_order_sha256=gene_order_sha,
        raw_source_manifest_digest=canonical_hash(sorted(raw_hashes)),identity=identity,
        feature_normalization=norm_audit,gene_value_unit=SOURCE_VALUE_UNIT,
        count_values_are_source_nonnegative_integer=True,natural_zero_is_measured=True,
        missing_values_require_explicit_mask=True,program_replay_max_abs_error=error,
        program_replay_tolerance=contract.program_replay_tolerance,
        measured_gene_count_min=int(measured.sum(1).min()),measured_gene_count_max=int(measured.sum(1).max()),
        radius_um=contract.radius_um,maximum_neighbors_including_self=contract.maximum_neighbors,
        neighbor_count_min=int(nc.min()),neighbor_count_median=float(np.median(nc)),neighbor_count_max=int(nc.max()),
        cross_section_edges=0,neighbor_mean_definition='float64_mean_of_unit_rows_then_float32_not_necessarily_unit',
        neighbor_unit_definition='explicit_float32_row_L2_of_mean_of_unit_rows',
        target_scaler_fit=False,TCGA_scaler_reused=False,TCGA_DSS_read=False,original_ST_test_read=False,
        original_sources_modified=False,new_model_fits=0,new_optimizer_updates=0,GPU_used=False,
        private_identifiers_coordinates_gene_values_or_paths_exported=False)
    return SourceSpotDataset(features_unit=features,counts=counts,measured_mask=measured,log1p_cp10k=log_values,
        program_targets=data['program_y'].astype(np.float64),tissue_y=data['tissue_y'].astype(np.int64),
        program_up=up,program_down=down,gene_symbols=genes,
        coords_um=data['coords_um'],section_codes=sc,donor_codes=dc,library_size=data['library_size'],
        n_detected_genes=data['n_detected_genes'],neighbors=ni,neighbor_counts=nc,
        neighbor_mean_of_unit=mean,neighbor_mean_unit=mean_unit,audit=audit)


def write_private_cache(dataset: SourceSpotDataset,directory: Path,*,repository_root: Path,raw_root: Path):
    target=Path(directory).resolve()
    require(not target.is_relative_to(Path(repository_root).resolve()) and
        not target.is_relative_to(Path(raw_root).resolve()) and
        not any((parent/'.git').exists() for parent in (target,*target.parents)),
        'CACHE_MUST_BE_OUTSIDE_REPOSITORY_AND_RAW_SOURCE')
    require(not target.exists(),'EXISTING_CACHE_PRESERVED_NO_OVERWRITE')
    target.mkdir(parents=True,mode=0o700)
    count_file=target/'gene_counts.private.npz';array_file=target/'source_arrays.private.npz'
    sparse.save_npz(count_file,dataset.counts,compressed=True)
    np.savez_compressed(array_file,features_unit=dataset.features_unit,measured_mask=dataset.measured_mask,
        log1p_cp10k=dataset.log1p_cp10k,program_targets=dataset.program_targets,tissue_y=dataset.tissue_y,
        program_up=dataset.program_up,
        program_down=dataset.program_down,gene_symbols=dataset.gene_symbols,coords_um=dataset.coords_um,
        section_codes=dataset.section_codes,donor_codes=dataset.donor_codes,library_size=dataset.library_size,
        n_detected_genes=dataset.n_detected_genes,neighbors=dataset.neighbors,neighbor_counts=dataset.neighbor_counts,
        neighbor_mean_of_unit=dataset.neighbor_mean_of_unit,neighbor_mean_unit=dataset.neighbor_mean_unit)
    manifest=dict(schema='bgb_en_r6_private_source_cache_v1',status='COMPLETE_PRIVATE_DERIVED_CACHE',
        source_audit=dataset.public_audit(),files={p.name:sha256(p) for p in (count_file,array_file)},
        true_barcodes_or_source_names_stored=False,contains_private_coordinates_expression_features=True,
        new_model_fits=0,source_module_sha256=sha256(Path(__file__)))
    with (target/'manifest.private.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    return dict(status=manifest['status'],cache_files=len(manifest['files']),
        total_bytes=sum(p.stat().st_size for p in target.iterdir()),manifest_sha256=sha256(target/'manifest.private.json'))


def load_private_cache(directory: Path,contract: SourceContract,*,expected_manifest_sha256: str) -> SourceSpotDataset:
    root=Path(directory)
    require(sha256(root/'manifest.private.json')==expected_manifest_sha256,'PRIVATE_CACHE_MANIFEST_SHA_CHANGED')
    manifest=json.loads((root/'manifest.private.json').read_text(encoding='utf-8'))
    require(manifest.get('schema')=='bgb_en_r6_private_source_cache_v1' and
        manifest.get('status')=='COMPLETE_PRIVATE_DERIVED_CACHE','INCOMPLETE_OR_FOREIGN_PRIVATE_CACHE')
    audit=manifest['source_audit']
    for key,value in (('prepared_data_sha256',contract.prepared_data_sha256),
        ('prepared_manifest_sha256',contract.prepared_manifest_sha256),
        ('program_definition_sha256',contract.program_definition_sha256),
        ('fresh_certificate_sha256',contract.fresh_certificate_sha256)):
        require(audit.get(key)==value,'PRIVATE_CACHE_SOURCE_CONTRACT_CHANGED')
    for key,value in (('primary_spots',contract.primary_spots),('source_donors',contract.donors),
        ('source_sections',contract.sections),('full_source_genes',contract.full_source_genes),
        ('radius_um',contract.radius_um),('maximum_neighbors_including_self',contract.maximum_neighbors),
        ('program_replay_tolerance',contract.program_replay_tolerance)):
        require(audit.get(key)==value,'PRIVATE_CACHE_GEOMETRY_OR_MEASUREMENT_CONTRACT_CHANGED')
    require(set(manifest['files'])=={'gene_counts.private.npz','source_arrays.private.npz'},'PRIVATE_CACHE_FILE_SET_CHANGED')
    for name,digest in manifest['files'].items():require(sha256(root/name)==digest,'PRIVATE_CACHE_PAYLOAD_CHANGED')
    require(manifest['source_module_sha256']==sha256(Path(__file__)),'PRIVATE_CACHE_LOADER_SOURCE_CHANGED')
    with np.load(root/'source_arrays.private.npz',allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
    counts=sparse.load_npz(root/'gene_counts.private.npz')
    return SourceSpotDataset(counts=counts,audit=audit,**arrays)
