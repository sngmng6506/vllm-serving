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

# 환경변수가 설정되지 않았으면 오류 발생
: "${MODEL_NAME:?MODEL_NAME must be set in vllm.env}"
: "${MODEL_BASE_DIR:?MODEL_BASE_DIR must be set in vllm.env}"

MODEL_PATH="${MODEL_BASE_DIR}/${MODEL_NAME}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "모델 디렉토리를 찾을 수 없습니다: ${MODEL_PATH}" >&2
  echo "사용 가능한 모델:" >&2
  ls -1 "${MODEL_BASE_DIR}" >&2 || true
  exit 1
fi

LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 환경변수 export (WSL2 멀티GPU 필수)
for var in NCCL_P2P_DISABLE NCCL_IB_DISABLE NCCL_SHM_DISABLE NCCL_CUMEM_ENABLE NCCL_SOCKET_IFNAME CUDA_VISIBLE_DEVICES TORCHINDUCTOR_DISABLE TORCHDYNAMO_DISABLE; do
  if [[ -n "${!var:-}" ]]; then
    export "$var"
  fi
done

echo "============================================"
echo "vLLM 서빙 시작 (api_server): ${MODEL_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL_NAME}}"
echo "HOST=${HOST} PORT=${PORT}"
echo "TP=${TENSOR_PARALLEL_SIZE} GPU_MEM=${GPU_MEMORY_UTILIZATION}"
echo "============================================"

# 이전 검증된 방식: python -m vllm.entrypoints.openai.api_server (V0 엔진)
ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --tokenizer "${MODEL_PATH}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --swap-space "${SWAP_SPACE}"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ -n "${SERVED_MODEL_NAME}" ]]; then
  ARGS+=(--served-model-name "${SERVED_MODEL_NAME}")
fi
if [[ -n "${CHAT_TEMPLATE_CONTENT_FORMAT}" ]]; then
  ARGS+=(--chat-template-content-format "${CHAT_TEMPLATE_CONTENT_FORMAT}")
fi
if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  ARGS+=(--enforce-eager)
fi

echo "실행 명령: python ${ARGS[*]}"
echo "============================================"
python "${ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/vllm.log"
