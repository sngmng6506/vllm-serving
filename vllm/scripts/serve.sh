#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/configs/vllm.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

: "${MODEL_NAME:=gemma3-12b-it}"
: "${MODEL_BASE_DIR:=/home/h202403659/LLM-Server/models/llm}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=8192}"
: "${DTYPE:=auto}"
: "${TRUST_REMOTE_CODE:=false}"
: "${ENABLE_CHUNKED_PREFILL:=false}"
: "${MAX_NUM_BATCHED_TOKENS:=}"
: "${SWAP_SPACE:=}"
: "${KV_CACHE_DTYPE:=}"
: "${NCCL_P2P_DISABLE:=}"
: "${NCCL_DEBUG:=}"
: "${NCCL_SHM_DISABLE:=}"
: "${NCCL_CUMEM_ENABLE:=}"
: "${NCCL_SOCKET_IFNAME:=}"
: "${NCCL_MAX_NCHANNELS:=}"
: "${NCCL_BUFFSIZE:=}"
: "${CUDA_VISIBLE_DEVICES:=}"
: "${TORCHINDUCTOR_DISABLE:=}"
: "${TORCHDYNAMO_DISABLE:=}"
: "${EXTRA_ARGS:=}"

MODEL_PATH="${MODEL_BASE_DIR}/${MODEL_NAME}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "모델 디렉토리를 찾을 수 없습니다: ${MODEL_PATH}" >&2
  echo "사용 가능한 모델:" >&2
  ls -1 "${MODEL_BASE_DIR}" >&2 || true
  exit 1
fi

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if [[ -n "${NCCL_P2P_DISABLE}" ]]; then
  export NCCL_P2P_DISABLE
fi
if [[ -n "${NCCL_DEBUG}" ]]; then
  export NCCL_DEBUG
fi
if [[ -n "${NCCL_SHM_DISABLE}" ]]; then
  export NCCL_SHM_DISABLE
fi
if [[ -n "${NCCL_CUMEM_ENABLE}" ]]; then
  export NCCL_CUMEM_ENABLE
fi
if [[ -n "${NCCL_SOCKET_IFNAME}" ]]; then
  export NCCL_SOCKET_IFNAME
fi
if [[ -n "${NCCL_MAX_NCHANNELS}" ]]; then
  export NCCL_MAX_NCHANNELS
fi
if [[ -n "${NCCL_BUFFSIZE}" ]]; then
  export NCCL_BUFFSIZE
fi
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
if [[ -n "${TORCHINDUCTOR_DISABLE}" ]]; then
  export TORCHINDUCTOR_DISABLE
fi
if [[ -n "${TORCHDYNAMO_DISABLE}" ]]; then
  export TORCHDYNAMO_DISABLE
fi

echo "vLLM 서빙 시작: ${MODEL_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "HOST=${HOST} PORT=${PORT}"

ARGS=(serve "${MODEL_PATH}" --host "${HOST}" --port "${PORT}")
if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
  ARGS+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  ARGS+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${DTYPE}" ]]; then
  ARGS+=(--dtype "${DTYPE}")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi
if [[ -n "${SWAP_SPACE}" ]]; then
  ARGS+=(--swap-space "${SWAP_SPACE}")
fi
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
  ARGS+=(--trust-remote-code)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
  ARGS+=(--enable-chunked-prefill)
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(${EXTRA_ARGS})
fi
vllm "${ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/vllm.log"
