"""Publish aggregate Phase6A evidence; no patient-level predictions or labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[1]/"reports/phase6a_titan_semantic_preflight")
    args = parser.parse_args()

    def read(name):
        return json.loads((args.report/name).read_text(encoding="utf-8"))

    done = read("DONE.json")
    runtime = read("runtime.json")
    audit, geom, data = read("embedding_audit.json"), read("concept_geometry.json"), read("data_preflight.json")
    lines = ["# Phase6A — TITAN CRC morphology semantic preflight", "", f"Run status: **{done['status']}**.", "",
             "## 1. Frozen official slide/text representation", "",
             f"- Successful CONCH slides: **{audit['successful_slides']}/519**; patients: **{audit['patients']}/512**.",
             "- Original UNI2 inventory has 520 slides. One unavailable CONCH DX1 is retained as missing; the same patient's original matched DX2 is available. No patch or patient substitution.",
             f"- Native and projected semantic dimensions: {audit['native_dim']} / {audit['semantic_dim']}; finite embeddings: {audit['finite']}.",
             f"- Native slide norm range: {audit['native_slide_norm_range']}; peak CUDA allocation: {audit['peak_cuda_gib']:.3f} GiB.",
             f"- Maximum official zero_shot probability agreement error: {audit['official_zero_shot_max_error']:.3g}.",
             "- Native output of encode_slide_from_patch_features is NOT text-aligned by itself. Apply the official vision_encoder.proj and L2 normalization, then compute raw cosine scores. Official zero_shot softmax is used solely as an equivalence check.",
             "- All patches and original coordinates are used; coordinate scale is read from each H5. Native patient representation: normalized arithmetic mean of slide embeddings. Semantic patient representation: normalized equal-weight mean of normalized projected slides.",
             f"- Exact weights and official source revision are pinned. Runtime uses CUDA torch {runtime['versions']['torch']} instead of documented torch 2.0.1; Transformers/timm/einops match the official listed versions. Tokenizer and recursive-import I/O are routed to identical local pinned source files. Allocation mode is documented in allocation_equivalence.json.", "",
             "## 2. Outcome-blind concept variation", "",
             "- Fixed 32 CRC morphologic/clinicopathologic concepts and two text templates; no molecular pathway vocabulary, DSS concept selection, prompt tuning, or cohort-fitted semantic projector.",
             f"- Geometry gate: **{geom['status']}**.",
             f"- Covariance-spectrum entropy effective rank: **{geom['raw_centered_cosine_spectrum']['effective_rank']:.4f}**; first component fraction: {geom['raw_centered_cosine_spectrum']['first_component_fraction']:.4f}.",
             f"- Correlation-spectrum effective rank: {geom['correlation_spectrum']['effective_rank']:.4f}; concept standard deviation range: {geom['std_range']}.",
             f"- Median absolute interconcept correlation: {geom['median_abs_concept_correlation']:.4f}; constant concepts: {geom['constant_concepts']}.",
             "- Preregistered hard stop: raw covariance effective rank <=1.5 or no patient variation. Text geometry is also reported; correlated descriptors alone do not validate individual morphologic annotations.", "",
             "## 3. Patient-level TNM/Stage information audit", ""]
    lines.append("| Endpoint | Valid patients | Excluded | Class counts |")
    lines.append("|---|---:|---:|---|")
    for endpoint, counts in data["clinical_targets"].items():
        lines.append(f"| {endpoint} | {counts['valid']} | {counts['excluded']} | {counts['classes']} |")
    lines.extend(["", "- Only patient ID and T/N/M/Stage columns are loaded into experiment computations; historical 512-patient cohort membership is preserved.",
                  "- Three seeds x five patient folds; identical folds across representations; three-fold inner selection over four fixed regularization strengths. Scaling and PCA are fitted only on each training partition.",
                  "- Seed metrics are computed after concatenating the five test folds. Patient OOF probabilities and fold assignments are persisted outside Git and mirrored to the local E-drive cache.", ""])
    if done["clinical_evaluation_performed"]:
        frame = pd.read_csv(args.report/"seed_oof_metrics.csv")
        lines.extend(["### Three-seed OOF balanced accuracy (mean +/- seed SD)", "", "| Endpoint | TITAN native | CRC concepts | TITAN PCA32 | Prior only |", "|---|---:|---:|---:|---:|"])
        for endpoint in ["T", "N", "M", "Stage"]:
            cells = []
            for name in ["TITAN_embedding", "CRC_concept_cosine", "TITAN_embedding_PCA32", "prior_only"]:
                values = frame.loc[(frame.endpoint == endpoint)&(frame.model == name), "balanced_accuracy"]
                if len(values) != 3:
                    raise RuntimeError("Missing seed OOF metric")
                cells.append(f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}")
            lines.append("| "+endpoint+" | "+" | ".join(cells)+" |")
        lines.extend(["", "### Three-seed OOF macro one-vs-rest AUROC (mean)", "", "| Endpoint | TITAN native | CRC concepts | TITAN PCA32 |", "|---|---:|---:|---:|"])
        for endpoint in ["T", "N", "M", "Stage"]:
            cells = [f"{frame.loc[(frame.endpoint==endpoint)&(frame.model==name),'macro_ovr_auroc'].mean():.4f}"
                     for name in ["TITAN_embedding", "CRC_concept_cosine", "TITAN_embedding_PCA32"]]
            lines.append("| "+endpoint+" | "+" | ".join(cells)+" |")
        comparisons, gate = read("paired_comparisons.json"), read("clinical_information_gate.json")
        lines.extend(["", "### Paired concept-minus-comparator OOF differences", "", "| Endpoint | Comparator | Seed deltas | Paired mean | 95% interval |", "|---|---|---|---:|---|"])
        for row in comparisons:
            ds = ", ".join(f"{d:+.4f}" for d in row["seed_deltas"])
            ci = row["paired_delta_ci95"]
            lines.append(f"| {row['endpoint']} | {row['comparison'].split('-minus-')[1]} | {ds} | {sum(row['seed_deltas'])/3:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        lines.extend(["", f"Clinical information gate: **{gate['status']}**; passing targets: {gate['passing_targets']}.",
                      "Gate requires concept-minus-prior balanced accuracy >0 for all three seeds and positive lower bound of a familywise 98.75% paired patient-bootstrap interval. Paired bootstrap resamples the same patients across all seeds (2,000 draws); it does not refit models or provide external-cohort uncertainty.",
                      "Repeated seeds reuse the same cohort and are not three independent validation datasets."])
        native_pairs = [r for r in comparisons if r['comparison']=="CRC_concept_cosine-minus-TITAN_embedding"]
        superiority = [r['endpoint'] for r in native_pairs if r['paired_delta_ci95'][0]>0]
        lines.extend(["", f"Endpoints with a positive 95% paired lower bound versus native TITAN: **{superiority}**.",
                      "Concept balanced accuracy averages are higher for T/N/Stage and lower for M. All four native-comparator intervals include zero in this run; superiority over generic TITAN is not established. The PCA32 control also prevents attributing all benefits to medical semantics, and AUROC does not show a consistent concept advantage."])
    else:
        lines.append("**Stopped before clinical prediction because the preregistered concept-collapse gate failed. No clinical predictive conclusion is available.**")
    lines.extend(["", "## 4. Interpretation and limits", "",
                  "- Advantage: a pretrained and verified pathology-language space is available without raw SVS or fitting a small-cohort UNI2-to-text projector; native TITAN signal is preserved as a comparator.",
                  "- Risk: frozen cosine descriptors compress native representation; any predictive advantage can reflect regularization/sample efficiency rather than additional information. The training-only PCA32 control helps distinguish semantic prior from dimensional compression.",
                  "- Noncollapse does not prove concept correctness. TNM/Stage prediction does not prove DSS benefit, causality, or that every named feature is visible in every diagnostic slide. Independent pathology annotation and cohort validation remain necessary.",
                  "- TNM/Stage are cross-sectional disease-state labels here; this audit does not establish longitudinal progression prediction.",
                  "- No prompt/config revision from these targets; no survival model, Omics branch, or CTF tuning is executed in this phase.",
                  "- Execution disclosure: an administrative preview displayed one raw clinical row during setup. The experimental readers load only the five permitted patient/TNM/Stage columns; no survival value enters concept definition, model fitting, or evaluation.",
                  "- Confidence: high for verified technical/geometry evidence; clinical estimates are limited to repeated OOF evaluation within this cohort.", "",
                  "## Sources", "",
                  "- [TITAN official model and usage](https://huggingface.co/MahmoodLab/TITAN)",
                  "- [Official zero-shot example](https://github.com/mahmoodlab/TITAN/blob/9e34c66ff66445c6c590da0dbf7acc103d39a40b/notebooks/zeroshot_demo.ipynb)",
                  "- [ICCR colorectal cancer pathology dataset](https://www.iccr-cancer.org/datasets/published-datasets/digestive-tract/colorectal/)", ""])
    (args.report/"RESULTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("REPORT_WRITTEN", args.report/"RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
