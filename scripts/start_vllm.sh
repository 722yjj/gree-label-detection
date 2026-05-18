#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VLLM_VENV="${VLLM_VENV:-/home/jnu/venvs/vllm}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_VENV}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_VENV}/bin/vllm}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/home/jnu/models/Qwen3.6-27B-int4-AutoRound}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen3.6-27b-int4}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:${VLLM_PORT}/v1}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-49152}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-12288}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
VLLM_SPECULATIVE_CONFIG="${VLLM_SPECULATIVE_CONFIG:-{\"method\": \"mtp\", \"num_speculative_tokens\": 1}}"
VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-300}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MANAGED_MODEL_PATH="${VLLM_MANAGED_MODEL_PATH:-${VLLM_MODEL_PATH}}"
VLLM_MANAGED_MODEL_NAME="${VLLM_MANAGED_MODEL_NAME:-${VLLM_SERVED_MODEL_NAME}}"

LOG_DIR="${PROJECT_ROOT}/results/vllm"
VLLM_STATE_DIR="${VLLM_STATE_DIR:-${LOG_DIR}}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-${LOG_DIR}/server.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-${LOG_DIR}/server.pid}"
VLLM_DESKTOP_SESSION_FILE="${VLLM_DESKTOP_SESSION_FILE:-${VLLM_STATE_DIR}/desktop-session.token}"
VLLM_DESKTOP_WATCHDOG_PID_FILE="${VLLM_DESKTOP_WATCHDOG_PID_FILE:-${VLLM_STATE_DIR}/desktop-watchdog.pid}"
VLLM_DESKTOP_SHUTDOWN_DELAY="${VLLM_DESKTOP_SHUTDOWN_DELAY:-1800}"

MODE="start"
FOREGROUND=0
WAIT_FOR_READY=1
SERVE_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/start_vllm.sh [command] [options]

Commands:
  start       Start vLLM in the background and wait until /v1/models is ready. Default.
  foreground  Start vLLM in the foreground.
  status      Show vLLM endpoint and PID status.
  check       Check the vLLM environment, model files, CUDA, and port status.
  stop        Stop the vLLM process recorded in the PID file.
  restart     Stop the recorded vLLM process, then start it again.
  delayed-stop Wait for the desktop process to exit, then stop vLLM after a delay.

Options:
  --no-wait     Return immediately after background start.
  --print-env   Print environment exports for running this project against vLLM.
  -h, --help    Show this help.

Important environment overrides:
  VLLM_VENV=/home/jnu/venvs/vllm
  VLLM_MODEL_PATH=/home/jnu/models/Qwen3.6-27B-int4-AutoRound
  VLLM_SERVED_MODEL_NAME=qwen3.6-27b-int4
  VLLM_PORT=8000
  VLLM_MAX_MODEL_LEN=49152
  VLLM_GPU_MEMORY_UTILIZATION=0.30
  VLLM_KV_CACHE_DTYPE=fp8
  VLLM_DESKTOP_SHUTDOWN_DELAY=1800
EOF
}

enabled() {
  case "${1,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

models_url() {
  printf '%s/models' "${VLLM_API_BASE%/}"
}

fetch_models() {
  local url
  url="$(models_url)"
  "${VLLM_PYTHON}" - "${url}" "${VLLM_API_KEY}" <<'PY'
import json
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

url = sys.argv[1]
api_key = sys.argv[2]
headers = {"Accept": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
request = Request(url, headers=headers)
try:
    with urlopen(request, timeout=2) as response:
        if not (200 <= response.status < 300):
            raise SystemExit(1)
        payload = json.load(response)
except (OSError, URLError, json.JSONDecodeError):
    raise SystemExit(1)

for item in payload.get("data", []):
    model_id = str(item.get("id") or "").strip()
    if model_id:
        print(model_id)
PY
}

model_is_expected() {
  local models="$1"
  local basename
  basename="$(basename "${VLLM_MODEL_PATH}")"

  while IFS= read -r model_id; do
    case "${model_id}" in
      "${VLLM_SERVED_MODEL_NAME}"|"${VLLM_MODEL_PATH}"|"${basename}")
        return 0
        ;;
    esac
  done <<<"${models}"

  return 1
}

port_listeners() {
  ss -ltnp 2>/dev/null | awk -v port=":${VLLM_PORT}" '$4 ~ port "$" {print}'
}

validate_paths() {
  local ok=0

  if [[ ! -d "${VLLM_VENV}" ]]; then
    echo "Missing vLLM venv: ${VLLM_VENV}" >&2
    ok=1
  fi
  if [[ ! -f "${VLLM_VENV}/bin/activate" ]]; then
    echo "Missing vLLM activate script: ${VLLM_VENV}/bin/activate" >&2
    ok=1
  fi
  if [[ ! -x "${VLLM_PYTHON}" ]]; then
    echo "Missing vLLM Python: ${VLLM_PYTHON}" >&2
    ok=1
  fi
  if [[ ! -x "${VLLM_BIN}" ]]; then
    echo "Missing vLLM executable: ${VLLM_BIN}" >&2
    ok=1
  fi
  if [[ ! -d "${VLLM_MODEL_PATH}" ]]; then
    echo "Missing model directory: ${VLLM_MODEL_PATH}" >&2
    ok=1
  elif [[ ! -f "${VLLM_MODEL_PATH}/config.json" ]]; then
    echo "Missing model config: ${VLLM_MODEL_PATH}/config.json" >&2
    ok=1
  elif ! compgen -G "${VLLM_MODEL_PATH}/model*.safetensors" >/dev/null; then
    echo "No safetensors shards found under: ${VLLM_MODEL_PATH}" >&2
    ok=1
  fi

  return "${ok}"
}

read_trimmed_file() {
  local path="$1"

  [[ -f "${path}" ]] || return 1
  tr -d '[:space:]' <"${path}"
}

pid_cmdline() {
  local pid="$1"

  if [[ -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' <"/proc/${pid}/cmdline"
    return 0
  fi

  ps -p "${pid}" -o args= 2>/dev/null || true
}

is_managed_vllm_pid() {
  local pid="$1" cmdline

  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1

  cmdline="$(pid_cmdline "${pid}")"
  [[ -n "${cmdline}" ]] || return 1
  printf '%s' "${cmdline}" | grep -Fq -- "serve" || return 1
  printf '%s' "${cmdline}" | grep -Fq -- "${VLLM_MANAGED_MODEL_PATH}" || return 1
  printf '%s' "${cmdline}" | grep -Fq -- "${VLLM_MANAGED_MODEL_NAME}" || return 1
  return 0
}

build_serve_args() {
  SERVE_ARGS=(
    serve "${VLLM_MODEL_PATH}"
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --served-model-name "${VLLM_SERVED_MODEL_NAME}"
    --dtype "${VLLM_DTYPE}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}"
  )

  if enabled "${VLLM_TRUST_REMOTE_CODE}"; then
    SERVE_ARGS+=(--trust-remote-code)
  fi
  if [[ -n "${VLLM_SPECULATIVE_CONFIG}" ]]; then
    SERVE_ARGS+=(--speculative-config "${VLLM_SPECULATIVE_CONFIG}")
  fi
}

print_command() {
  printf '%q ' "${VLLM_BIN}" "${SERVE_ARGS[@]}"
  printf '\n'
}

print_app_env() {
  cat <<EOF
export LLM_PROVIDER=vllm
export VLLM_API_BASE=${VLLM_API_BASE}
export OPENAI_COMPATIBLE_API_BASE=${VLLM_API_BASE}
export VLLM_API_KEY=${VLLM_API_KEY}
export OPENAI_COMPATIBLE_API_KEY=${VLLM_API_KEY}
export VLLM_VENV=${VLLM_VENV}
export VLLM_BIN=${VLLM_BIN}
export VLLM_MODEL_PATH=${VLLM_MODEL_PATH}
export VLLM_SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME}
export VLLM_MODEL=${VLLM_SERVED_MODEL_NAME}
export OPENAI_COMPATIBLE_MODEL=${VLLM_SERVED_MODEL_NAME}
export TEXT_LLM_MODEL=${VLLM_SERVED_MODEL_NAME}
export GRAPHIC_VLM_MODEL=${VLLM_SERVED_MODEL_NAME}
EOF
}

launch_vllm_detached() {
  nohup bash -c '
    set -Eeuo pipefail
    # shellcheck disable=SC1091
    source "$1/bin/activate"
    shift
    exec "$@"
  ' _ "${VLLM_VENV}" "${VLLM_BIN}" "${SERVE_ARGS[@]}" >>"${VLLM_LOG_FILE}" 2>&1 </dev/null &
  printf '%s\n' "$!"
}

show_endpoint_status() {
  local models listener
  if models="$(fetch_models 2>/dev/null)"; then
    echo "OpenAI-compatible endpoint is reachable: $(models_url)"
    echo "Advertised models:"
    sed 's/^/  - /' <<<"${models}"
    if model_is_expected "${models}"; then
      echo "Expected model is ready: ${VLLM_SERVED_MODEL_NAME}"
      return 0
    fi
    echo "Endpoint is reachable, but expected model is not advertised: ${VLLM_SERVED_MODEL_NAME}" >&2
    return 1
  fi

  listener="$(port_listeners || true)"
  if [[ -n "${listener}" ]]; then
    echo "Port ${VLLM_PORT} is occupied, but ${VLLM_API_BASE%/}/models is not ready:" >&2
    echo "${listener}" >&2
    return 1
  fi

  echo "No vLLM/OpenAI-compatible service is listening on port ${VLLM_PORT}."
  return 1
}

show_desktop_shutdown_status() {
  local watchdog_pid

  if [[ ! -f "${VLLM_DESKTOP_WATCHDOG_PID_FILE}" ]]; then
    return 0
  fi

  watchdog_pid="$(tr -d '[:space:]' <"${VLLM_DESKTOP_WATCHDOG_PID_FILE}")"
  if [[ "${watchdog_pid}" =~ ^[0-9]+$ ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
    echo "Desktop delayed-stop watchdog is running: PID ${watchdog_pid}"
    echo "Desktop session token file: ${VLLM_DESKTOP_SESSION_FILE}"
  else
    echo "Desktop delayed-stop watchdog is not running: ${watchdog_pid:-empty}"
  fi
}

show_pid_status() {
  local pid

  if [[ -f "${VLLM_PID_FILE}" ]]; then
    pid="$(tr -d '[:space:]' <"${VLLM_PID_FILE}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      if is_managed_vllm_pid "${pid}"; then
        echo "Recorded PID is running and looks managed: ${pid}"
      else
        echo "Recorded PID is running but does not look like managed vLLM: ${pid}" >&2
      fi
    else
      echo "Recorded PID is not running: ${pid:-empty}"
    fi
  else
    echo "No PID file: ${VLLM_PID_FILE}"
  fi
}

check_cuda() {
  "${VLLM_PYTHON}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"cuda_total_memory_gib={total:.2f}")
PY
}

run_check() {
  echo "Project root: ${PROJECT_ROOT}"
  echo "vLLM venv: ${VLLM_VENV}"
  echo "Model path: ${VLLM_MODEL_PATH}"
  echo "Served model name: ${VLLM_SERVED_MODEL_NAME}"
  echo "API base: ${VLLM_API_BASE}"
  echo

  validate_paths

  echo "vLLM version:"
  "${VLLM_BIN}" --version
  echo

  echo "CUDA check:"
  check_cuda
  echo

  show_endpoint_status || true
}

wait_until_ready() {
  local pid="$1"
  local elapsed models

  for ((elapsed = 0; elapsed <= VLLM_START_TIMEOUT; elapsed += 2)); do
    if models="$(fetch_models 2>/dev/null)" && model_is_expected "${models}"; then
      echo "vLLM is ready: $(models_url)"
      echo "Advertised models:"
      sed 's/^/  - /' <<<"${models}"
      return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "vLLM process exited before becoming ready. Recent log:" >&2
      tail -n 80 "${VLLM_LOG_FILE}" >&2 || true
      return 1
    fi

    sleep 2
  done

  echo "Timed out after ${VLLM_START_TIMEOUT}s waiting for vLLM. Recent log:" >&2
  tail -n 80 "${VLLM_LOG_FILE}" >&2 || true
  return 1
}

start_vllm() {
  local models listener pid

  mkdir -p "${LOG_DIR}"
  validate_paths

  if models="$(fetch_models 2>/dev/null)"; then
    if model_is_expected "${models}"; then
      echo "vLLM is already ready: $(models_url)"
      echo "Advertised models:"
      sed 's/^/  - /' <<<"${models}"
      return 0
    fi

    echo "Port ${VLLM_PORT} has an OpenAI-compatible service, but it does not advertise ${VLLM_SERVED_MODEL_NAME}:" >&2
    sed 's/^/  - /' <<<"${models}" >&2
    return 1
  fi

  listener="$(port_listeners || true)"
  if [[ -n "${listener}" ]]; then
    echo "Port ${VLLM_PORT} is already occupied:" >&2
    echo "${listener}" >&2
    return 1
  fi

  build_serve_args
  echo "Starting vLLM. Log: ${VLLM_LOG_FILE}"
  echo "Command: $(print_command)"
  {
    echo
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') starting vLLM ====="
    echo "Command: $(print_command)"
  } >>"${VLLM_LOG_FILE}"

  if [[ "${FOREGROUND}" == "1" ]]; then
    # shellcheck disable=SC1091
    source "${VLLM_VENV}/bin/activate"
    exec "${VLLM_BIN}" "${SERVE_ARGS[@]}"
  fi

  pid="$(launch_vllm_detached)"
  printf '%s\n' "${pid}" >"${VLLM_PID_FILE}"
  echo "vLLM PID: ${pid}"

  if [[ "${WAIT_FOR_READY}" == "1" ]]; then
    wait_until_ready "${pid}"
  else
    echo "Started without waiting. Check status with: scripts/start_vllm.sh status"
  fi
}

stop_vllm() {
  local pid elapsed

  if [[ ! -f "${VLLM_PID_FILE}" ]]; then
    echo "No PID file found: ${VLLM_PID_FILE}"
    return 0
  fi

  pid="$(tr -d '[:space:]' <"${VLLM_PID_FILE}")"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    echo "Invalid PID file, removing: ${VLLM_PID_FILE}" >&2
    rm -f "${VLLM_PID_FILE}"
    return 1
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Recorded vLLM process is not running: ${pid}"
    rm -f "${VLLM_PID_FILE}"
    return 0
  fi

  if ! is_managed_vllm_pid "${pid}"; then
    echo "Recorded PID does not look like managed vLLM: ${pid}; removing PID file without stopping it." >&2
    rm -f "${VLLM_PID_FILE}"
    return 0
  fi

  echo "Stopping vLLM PID ${pid}..."
  kill "${pid}"
  for ((elapsed = 0; elapsed < 30; elapsed += 1)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      rm -f "${VLLM_PID_FILE}"
      echo "vLLM stopped."
      return 0
    fi
    sleep 1
  done

  echo "vLLM did not exit within 30s. PID remains: ${pid}" >&2
  return 1
}

delayed_stop_vllm() {
  local desktop_pid session_token current_token

  mkdir -p "${LOG_DIR}"

  desktop_pid="${VLLM_DESKTOP_PID:-}"
  session_token="${VLLM_DESKTOP_SESSION_TOKEN:-}"
  if [[ ! "${desktop_pid}" =~ ^[0-9]+$ ]]; then
    echo "Invalid or missing VLLM_DESKTOP_PID: ${desktop_pid:-empty}" >&2
    return 1
  fi
  if [[ -z "${session_token}" ]]; then
    echo "Missing VLLM_DESKTOP_SESSION_TOKEN." >&2
    return 1
  fi
  if [[ ! "${VLLM_DESKTOP_SHUTDOWN_DELAY}" =~ ^[0-9]+$ ]]; then
    echo "Invalid VLLM_DESKTOP_SHUTDOWN_DELAY: ${VLLM_DESKTOP_SHUTDOWN_DELAY}" >&2
    return 1
  fi

  printf '%s\n' "$$" >"${VLLM_DESKTOP_WATCHDOG_PID_FILE}"
  trap 'rm -f "${VLLM_DESKTOP_WATCHDOG_PID_FILE}"' EXIT

  echo "Desktop watchdog armed for PID ${desktop_pid}; waiting for exit..."
  while kill -0 "${desktop_pid}" 2>/dev/null; do
    sleep 2
  done

  echo "Desktop PID ${desktop_pid} exited; waiting ${VLLM_DESKTOP_SHUTDOWN_DELAY}s before stopping vLLM."
  sleep "${VLLM_DESKTOP_SHUTDOWN_DELAY}"

  current_token="$(read_trimmed_file "${VLLM_DESKTOP_SESSION_FILE}" 2>/dev/null || true)"
  if [[ -z "${current_token}" ]]; then
    echo "Desktop session token file missing; skipping delayed stop."
    return 0
  fi
  if [[ "${current_token}" != "${session_token}" ]]; then
    echo "Desktop session token changed; skipping delayed stop."
    return 0
  fi

  echo "Desktop session token unchanged; stopping vLLM."
  stop_vllm
}

show_status() {
  local status=0

  show_pid_status
  show_endpoint_status || status=$?
  show_desktop_shutdown_status
  return "${status}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|foreground|status|check|stop|restart|delayed-stop)
      MODE="$1"
      ;;
    --foreground)
      MODE="foreground"
      ;;
    --no-wait)
      WAIT_FOR_READY=0
      ;;
    --print-env)
      print_app_env
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "${MODE}" in
  start)
    start_vllm
    ;;
  foreground)
    FOREGROUND=1
    start_vllm
    ;;
  status)
    show_status
    ;;
  check)
    run_check
    ;;
  stop)
    stop_vllm
    ;;
  restart)
    stop_vllm
    start_vllm
    ;;
  delayed-stop)
    delayed_stop_vllm
    ;;
esac
