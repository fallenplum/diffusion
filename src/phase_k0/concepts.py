from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


CONCEPTS = [
    {
        "concept_id": "K1",
        "short_name": "immune_inflamed_interferon",
        "fixed_medical_description": "Interferon and cytotoxic immune activation within an inflamed colorectal tumour microenvironment.",
        "pathology_morphology_description": "Lymphocyte-rich tumour interface, tumour-infiltrating immune cells, and necro-inflammatory foci.",
        "omics_biological_description": "Interferon response, antigen presentation, allograft-rejection-like and IL2-STAT5 transcriptional activity.",
        "clinical_interpretation": "A stronger signal is compatible with an immune-inflamed phenotype; it is not a treatment-response label.",
        "mapped_hallmark_gene_sets": [
            "HALLMARK_INTERFERON_ALPHA_RESPONSE",
            "HALLMARK_INTERFERON_GAMMA_RESPONSE",
            "HALLMARK_ALLOGRAFT_REJECTION",
            "HALLMARK_IL2_STAT5_SIGNALING",
        ],
        "mcat_repository_categories": ["Cytokines and Growth Factors", "Transcription Factors"],
    },
    {
        "concept_id": "K2",
        "short_name": "canonical_epithelial_wnt_myc",
        "fixed_medical_description": "Canonical epithelial identity with WNT-beta-catenin and MYC-driven transcriptional activity.",
        "pathology_morphology_description": "Cohesive gland-forming epithelial architecture, preserved polarity, and epithelial tumour nests.",
        "omics_biological_description": "WNT-beta-catenin, MYC target, and KRAS-upstream epithelial programme activity.",
        "clinical_interpretation": "A stronger signal indicates epithelial tumour biology and does not itself encode DSS outcome.",
        "mapped_hallmark_gene_sets": [
            "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
            "HALLMARK_MYC_TARGETS_V1",
            "HALLMARK_MYC_TARGETS_V2",
            "HALLMARK_KRAS_SIGNALING_UP",
        ],
        "mcat_repository_categories": ["Oncogenes", "Cell Differentiation Markers", "Transcription Factors"],
    },
    {
        "concept_id": "K3",
        "short_name": "metabolic_phenotype",
        "fixed_medical_description": "Tumour metabolic state spanning glycolysis, oxidative phosphorylation, lipid handling, and peroxisomal metabolism.",
        "pathology_morphology_description": "Morphologic heterogeneity compatible with metabolic stress, mucinous change, or altered tumour-cell cytoplasm.",
        "omics_biological_description": "Glycolytic, oxidative-phosphorylation, fatty-acid, bile-acid, and peroxisome pathway activity.",
        "clinical_interpretation": "A metabolic signal is a molecular phenotype summary, not a direct clinical risk score.",
        "mapped_hallmark_gene_sets": [
            "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
            "HALLMARK_GLYCOLYSIS",
            "HALLMARK_FATTY_ACID_METABOLISM",
            "HALLMARK_BILE_ACID_METABOLISM",
            "HALLMARK_PEROXISOME",
        ],
        "mcat_repository_categories": ["Protein Kinases", "Transcription Factors"],
    },
    {
        "concept_id": "K4",
        "short_name": "mesenchymal_tgf_beta_stromal",
        "fixed_medical_description": "Mesenchymal and stromal remodelling driven by TGF-beta and tumour-stroma interaction.",
        "pathology_morphology_description": "Desmoplastic stroma, spindle-cell or fibroblast-rich areas, and stromal expansion around tumour glands.",
        "omics_biological_description": "TGF-beta, epithelial-mesenchymal transition, angiogenic, and apical-junction remodelling activity.",
        "clinical_interpretation": "A stronger signal indicates stromal or mesenchymal context and does not establish causality for outcome.",
        "mapped_hallmark_gene_sets": [
            "HALLMARK_TGF_BETA_SIGNALING",
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
            "HALLMARK_ANGIOGENESIS",
            "HALLMARK_APICAL_JUNCTION",
        ],
        "mcat_repository_categories": ["Cell Differentiation Markers", "Cytokines and Growth Factors"],
    },
    {
        "concept_id": "K5",
        "short_name": "proliferation_cell_cycle",
        "fixed_medical_description": "Tumour proliferation and cell-cycle activity spanning E2F, mitotic spindle, G2M, DNA repair, and MYC programmes.",
        "pathology_morphology_description": "High mitotic activity, increased nuclear-to-cytoplasmic ratio, and proliferative tumour regions.",
        "omics_biological_description": "E2F target, G2M checkpoint, mitotic-spindle, DNA-repair, and MYC target activity.",
        "clinical_interpretation": "A proliferative signal is a biological state descriptor and is not selected using DSS labels.",
        "mapped_hallmark_gene_sets": [
            "HALLMARK_E2F_TARGETS",
            "HALLMARK_G2M_CHECKPOINT",
            "HALLMARK_MITOTIC_SPINDLE",
            "HALLMARK_DNA_REPAIR",
            "HALLMARK_MYC_TARGETS_V1",
        ],
        "mcat_repository_categories": ["Oncogenes", "Protein Kinases", "Transcription Factors"],
    },
    {
        "concept_id": "K6",
        "short_name": "hypoxia_angiogenesis",
        "fixed_medical_description": "Hypoxic tumour stress and vascular adaptation, including angiogenic and reactive-oxygen responses.",
        "pathology_morphology_description": "Hypoxia-compatible necrosis, irregular vascularity, hemorrhagic change, and poorly perfused tumour areas.",
        "omics_biological_description": "Hypoxia, angiogenesis, glycolysis, and reactive-oxygen pathway activity.",
        "clinical_interpretation": "This summarizes tumour microenvironment stress; it is not a patient-level survival prediction.",
        "mapped_hallmark_gene_sets": [
            "HALLMARK_HYPOXIA",
            "HALLMARK_ANGIOGENESIS",
            "HALLMARK_GLYCOLYSIS",
            "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
        ],
        "mcat_repository_categories": ["Cytokines and Growth Factors", "Transcription Factors"],
    },
    {
        "concept_id": "K7",
        "short_name": "invasion_emt_ecm",
        "fixed_medical_description": "Invasion and extracellular-matrix remodelling associated with EMT, cell adhesion, coagulation, and tissue-interface change.",
        "pathology_morphology_description": "Irregular invasive fronts, tumour budding, loss of cohesive architecture, and matrix-associated tumour spread.",
        "omics_biological_description": "EMT, apical-junction, coagulation, complement, and tissue-remodelling activity.",
        "clinical_interpretation": "A stronger signal indicates an invasion-related biological context without using outcome labels to define the concept.",
        "mapped_hallmark_gene_sets": [
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
            "HALLMARK_APICAL_JUNCTION",
            "HALLMARK_COAGULATION",
            "HALLMARK_COMPLEMENT",
        ],
        "mcat_repository_categories": ["Cell Differentiation Markers", "Cytokines and Growth Factors"],
    },
    {
        "concept_id": "K8",
        "short_name": "inflammatory_immune_suppressive",
        "fixed_medical_description": "Inflammatory and potentially immune-suppressive tumour-microenvironment signalling through NF-kB, IL6-STAT3, and inflammatory response programmes.",
        "pathology_morphology_description": "Mixed inflammatory infiltrates, necro-inflammatory debris, and tumour-associated stromal or myeloid-rich regions.",
        "omics_biological_description": "TNF-NF-kB, IL6-JAK-STAT3, inflammatory-response, and complement pathway activity.",
        "clinical_interpretation": "This is a microenvironmental concept summary and does not label individual patient risk.",
        "mapped_hallmark_gene_sets": [
            "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
            "HALLMARK_IL6_JAK_STAT3_SIGNALING",
            "HALLMARK_INFLAMMATORY_RESPONSE",
            "HALLMARK_COMPLEMENT",
        ],
        "mcat_repository_categories": ["Cytokines and Growth Factors", "Cell Differentiation Markers"],
    },
]


def _sha_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_gene_level_shuffle(concepts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Create a frozen gene-membership shuffle with unchanged concept sizes.

    The negative control must change which genes are assigned to each concept,
    rather than merely reordering already-computed concept channels.  The pool
    and deterministic seed are fixed before any fold or outcome is observed.
    """
    pool = sorted({gene for concept in concepts for gene in concept["mapped_gene_symbols"]})
    if not pool:
        raise ValueError("cannot build an empty concept gene shuffle")
    rng = random.Random(20261003)
    rng.shuffle(pool)
    shuffled = []
    cursor = 0
    for concept in concepts:
        length = len(concept["mapped_gene_symbols"])
        if length > len(pool):
            raise ValueError(f"shuffle pool is smaller than {concept['concept_id']} gene set")
        genes = sorted(pool[(cursor + offset) % len(pool)] for offset in range(length))
        if len(genes) != length:
            raise ValueError(f"gene shuffle size mismatch for {concept['concept_id']}")
        item = dict(concept)
        item["shuffled_mapped_gene_symbols"] = genes
        item["shuffled_mapped_gene_count"] = len(genes)
        shuffled.append(item)
        cursor = (cursor + length + 173) % len(pool)
    mapping_payload = [
        {"concept_id": item["concept_id"], "genes": item["shuffled_mapped_gene_symbols"]}
        for item in shuffled
    ]
    return shuffled, _sha_payload(mapping_payload)


def build_concept_bank(metadata_path: Path, mcat_path: Path | None = None) -> dict[str, Any]:
    import pandas as pd

    table = pd.read_csv(metadata_path, dtype=str).fillna("")
    available = {str(c).strip().upper(): [str(x).strip().upper() for x in table[c].tolist() if str(x).strip()] for c in table.columns}
    bank = []
    for concept in CONCEPTS:
        mapped = []
        missing = []
        for group in concept["mapped_hallmark_gene_sets"]:
            key = group.upper()
            if key not in available:
                missing.append(group)
            else:
                mapped.extend(available[key])
        if missing:
            raise ValueError(f"concept {concept['concept_id']} missing fixed repository sets: {missing}")
        item = dict(concept)
        item["mapped_gene_symbols"] = sorted(set(mapped))
        item["mapped_gene_count"] = len(item["mapped_gene_symbols"])
        item["text_source"] = "fixed author-written description; no LLM or patient outcome input"
        bank.append(item)
    bank, shuffle_sha = _build_gene_level_shuffle(bank)
    payload = {
        "version": "concept_bank_v1",
        "disease": "colorectal cancer",
        "endpoint": "DSS",
        "selection_basis": "prespecified CRC biology and repository gene sets only; no DSS labels",
        "repository_sources": [metadata_path.name] + ([mcat_path.name] if mcat_path else []),
        "text_encoder": "none",
        "text_descriptions_used_in_model": False,
        "concepts": bank,
        "shuffle_mapping_method": "deterministic_gene_membership_shuffle",
        "shuffle_mapping_seed": 20261003,
        "shuffle_mapping_sha256": shuffle_sha,
    }
    payload["content_sha256"] = _sha_payload(payload)
    return payload


def write_concept_bank(metadata_path: Path, output_path: Path, mcat_path: Path | None = None) -> dict[str, Any]:
    payload = build_concept_bank(metadata_path, mcat_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def load_concept_bank(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("content_sha256", None)
    actual = _sha_payload(payload)
    payload["content_sha256"] = expected
    if expected != actual:
        raise ValueError(f"concept bank hash mismatch: expected={expected}, actual={actual}")
    if payload.get("version") != "concept_bank_v1" or len(payload.get("concepts", [])) != 8:
        raise ValueError("invalid concept bank version or concept count")
    if payload.get("shuffle_mapping_method") != "deterministic_gene_membership_shuffle":
        raise ValueError("concept bank is missing the frozen gene-level shuffle mapping")
    shuffle_payload = [
        {"concept_id": concept["concept_id"], "genes": concept["shuffled_mapped_gene_symbols"]}
        for concept in payload["concepts"]
    ]
    if payload.get("shuffle_mapping_sha256") != _sha_payload(shuffle_payload):
        raise ValueError("concept bank shuffle mapping hash mismatch")
    return payload

