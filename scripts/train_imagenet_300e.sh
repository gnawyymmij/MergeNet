#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT_DIR}"

CONFIG=${1:-configs/mergenet_l2_spatial_r3.yaml}
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}" >&2; exit 2; }
[[ -n "${DATA_DIR:-}" ]] || { echo "DATA_DIR is required" >&2; exit 2; }
[[ -d "${DATA_DIR}/train" && -d "${DATA_DIR}/val" ]] || {
  echo "DATA_DIR must contain train/ and val/ ImageFolder splits" >&2
  exit 2
}

if [[ -n "${GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=${GPUS}
fi

PYTHON_BIN=${PYTHON_BIN:-python}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.device_count())')}
BATCH_SIZE=${BATCH_SIZE:-64}
GLOBAL_BATCH=${GLOBAL_BATCH:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
RUN_NAME=${RUN_NAME:-mergenet_l2_r3}

for value in "${NPROC_PER_NODE}" "${BATCH_SIZE}" "${GLOBAL_BATCH}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "NPROC_PER_NODE, BATCH_SIZE, and GLOBAL_BATCH must be positive integers" >&2
    exit 2
  }
done

MICRO_GLOBAL=$((NPROC_PER_NODE * BATCH_SIZE))
(( GLOBAL_BATCH % MICRO_GLOBAL == 0 )) || {
  echo "GLOBAL_BATCH must be divisible by NPROC_PER_NODE * BATCH_SIZE" >&2
  exit 2
}
UPDATE_FREQ=$((GLOBAL_BATCH / MICRO_GLOBAL))

mkdir -p "${OUTPUT_DIR}"
export OPENTOME_SKIP_OPTIONAL_NLP=1
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE}"
  trainer/classification/in1k_trainer.py
  --config "${CONFIG}"
  --data_dir "${DATA_DIR}"
  --batch_size "${BATCH_SIZE}"
  --update_freq "${UPDATE_FREQ}"
  --output "${OUTPUT_DIR}"
  --experiment "${RUN_NAME}"
)

if [[ -n "${RESUME:-}" ]]; then
  [[ -f "${RESUME}" ]] || { echo "Resume checkpoint not found: ${RESUME}" >&2; exit 2; }
  CMD+=(--resume "${RESUME}")
fi

printf 'Launching:'
printf ' %q' "${CMD[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: command validated; training was not started."
  exit 0
fi
exec "${CMD[@]}"
