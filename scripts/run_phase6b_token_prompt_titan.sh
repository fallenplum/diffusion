#!/usr/bin/env bash
# Fixed protocol; persistent weights and private OOF are retained on the server.
set -u
out=/root/phase6b_token_prompt_titan_repair
mkdir -p "$out"
if [ -f "$out/DONE.json" ]; then
  echo "Already completed; refusing to rerun."
  exit 0
fi
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/phase6a_titan_runtime/env/bin/python -u \
  /root/diffusion_phase5_20261007/scripts/phase6b_token_prompt_titan.py \
  --cache /root/phase6a_titan_runtime/cache_v1 \
  --rna /root/autodl-tmp/diffusion_data/data/data_8/coadread_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt \
  --folds /root/diffusion_phase5_20261007/outputs/phase_m1r/fixed_patient_folds.csv \
  --model /root/phase6a_titan_runtime/model \
  --concepts /root/diffusion_phase5_20261007/configs/phase6a_titan_crc_morphology_concepts.json \
  --clinical-targets /root/phase6a_titan_runtime/cache_v1/clinical_targets.csv \
  --out "$out" \
  --reuse-baselines /root/phase6b_token_prompt_titan \
  --resume
run_status=$?
printf '%s\n' "$run_status" > "$out/EXIT_CODE"
if [ "$run_status" -eq 0 ]; then
  /root/phase6a_titan_runtime/env/bin/python \
    /root/diffusion_phase5_20261007/scripts/summarize_phase6b_token_prompt.py --out "$out"
fi
exit "$run_status"
