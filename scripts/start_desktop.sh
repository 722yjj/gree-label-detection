#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_ROOT="${DEPLOY_ROOT:-${PROJECT_ROOT}}"
DEPLOY_ENV="${DEPLOY_ENV:-${DEPLOY_ROOT}/config/deploy.env}"
if [[ -f "${DEPLOY_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${DEPLOY_ENV}"
  set +a
fi

APP_ROOT="${APP_ROOT:-${PROJECT_ROOT}}"
PYTHON="${APP_PYTHON:-${PYTHON:-${APP_ROOT}/.venv/bin/python}}"
RESULTS_ROOT="${RESULTS_ROOT:-${DEPLOY_ROOT}/results}"
LOG_DIR="${RESULTS_ROOT}/desktop_app"
LOG_FILE="${LOG_DIR}/startup.log"
OLLAMA_LOG_FILE="${LOG_DIR}/ollama.log"
LLM_PROVIDER_VALUE="${LLM_PROVIDER:-ollama}"
OLLAMA_API_BASE_VALUE="${OLLAMA_API_BASE:-http://localhost:11434}"
OLLAMA_AUTOSTART_VALUE="${OLLAMA_AUTOSTART:-1}"
OLLAMA_START_TIMEOUT_VALUE="${OLLAMA_START_TIMEOUT:-20}"
VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-${APP_ROOT}/scripts/start_vllm.sh}"
VLLM_AUTOSTART_VALUE="${VLLM_AUTOSTART:-1}"
VLLM_STATE_DIR="${VLLM_STATE_DIR:-${RESULTS_ROOT}/vllm}"
VLLM_PID_FILE="${VLLM_PID_FILE:-${VLLM_STATE_DIR}/server.pid}"
VLLM_DESKTOP_SESSION_FILE="${VLLM_DESKTOP_SESSION_FILE:-${VLLM_STATE_DIR}/desktop-session.token}"
VLLM_DESKTOP_WATCHDOG_PID_FILE="${VLLM_DESKTOP_WATCHDOG_PID_FILE:-${VLLM_STATE_DIR}/desktop-watchdog.pid}"
VLLM_DESKTOP_SHUTDOWN_DELAY_VALUE="${VLLM_DESKTOP_SHUTDOWN_DELAY:-1800}"
VLLM_WAS_MANAGED=0
VLLM_MANAGED_MODEL_PATH="${VLLM_MANAGED_MODEL_PATH:-}"
VLLM_MANAGED_MODEL_NAME="${VLLM_MANAGED_MODEL_NAME:-}"
LICENSE_VALID=1

usage() {
  cat <<'EOF'
Usage: scripts/start_desktop.sh [--check] [--strict-services]

Options:
  --check             Only run desktop environment checks.
  --strict-services   Treat LLM/service connectivity warnings as failures.

Environment:
  LLM_PROVIDER=ollama        Use Ollama by default for direct script runs.
  LLM_PROVIDER=vllm          Use project vLLM; desktop launcher sets this by default.
  VLLM_AUTOSTART=0           Disable automatic vLLM startup.
  VLLM_START_SCRIPT=path     Override the vLLM startup script.
  LABEL_DETECTION_LICENSE=path Override the offline license path.
  VLLM_DESKTOP_SHUTDOWN_DELAY=1800  Seconds to wait after desktop exit before stopping vLLM.
  OLLAMA_AUTOSTART=0         Disable automatic local Ollama startup.
  OLLAMA_BIN=/path/ollama   Override the Ollama executable path.
  OLLAMA_START_TIMEOUT=20   Seconds to wait for Ollama startup.
EOF
}

generate_session_token() {
  local token_source
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen | tr '[:upper:]' '[:lower:]'
    return 0
  fi
  token_source="$("${PYTHON}" - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
)"
  printf '%s\n' "${token_source}"
}

clear_previous_vllm_watchdog() {
  local watchdog_pid

  if [[ ! -f "${VLLM_DESKTOP_WATCHDOG_PID_FILE}" ]]; then
    return 0
  fi

  watchdog_pid="$(tr -d '[:space:]' <"${VLLM_DESKTOP_WATCHDOG_PID_FILE}")"
  if is_vllm_watchdog_pid "${watchdog_pid}" && kill -0 "${watchdog_pid}" 2>/dev/null; then
    echo "停止旧的 vLLM delayed-stop watchdog: PID ${watchdog_pid}" | tee -a "${LOG_FILE}"
    kill "${watchdog_pid}" 2>/dev/null || true
    for ((i = 0; i < 10; i++)); do
      if ! kill -0 "${watchdog_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  else
    echo "移除过期的 vLLM delayed-stop watchdog 记录: ${watchdog_pid:-empty}" | tee -a "${LOG_FILE}"
  fi

  rm -f "${VLLM_DESKTOP_WATCHDOG_PID_FILE}"
}

watchdog_cmdline() {
  local pid="$1"

  if [[ -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' <"/proc/${pid}/cmdline"
    return 0
  fi

  ps -p "${pid}" -o args= 2>/dev/null || true
}

is_vllm_watchdog_pid() {
  local pid="$1" cmdline

  if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  cmdline="$(watchdog_cmdline "${pid}")"
  [[ "${cmdline}" == *"${VLLM_START_SCRIPT}"*"delayed-stop"* ]]
}

is_project_managed_vllm_pid() {
  local pid cmdline

  if [[ ! -f "${VLLM_PID_FILE}" ]]; then
    return 1
  fi

  pid="$(tr -d '[:space:]' <"${VLLM_PID_FILE}")"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${pid}" 2>/dev/null; then
    return 1
  fi

  cmdline="$(watchdog_cmdline "${pid}")"
  [[ -n "${cmdline}" ]] || return 1
  [[ -n "${VLLM_BIN}" ]] || return 1
  [[ "${cmdline}" == *"${VLLM_BIN}"* ]] || return 1
  [[ "${cmdline}" == *"serve"* ]] || return 1
  [[ -n "${VLLM_MANAGED_MODEL_PATH}" ]] || return 1
  [[ -n "${VLLM_MANAGED_MODEL_NAME}" ]] || return 1
  [[ "${cmdline}" == *"${VLLM_MANAGED_MODEL_PATH}"* ]] || return 1
  [[ "${cmdline}" == *"${VLLM_MANAGED_MODEL_NAME}"* ]] || return 1
  return 0
}

is_local_ollama_endpoint() {
  case "${OLLAMA_API_BASE_VALUE}" in
    http://localhost|http://localhost:*|http://localhost/*|http://localhost:*/?*|\
    http://127.0.0.1|http://127.0.0.1:*|http://127.0.0.1/*|http://127.0.0.1:*/?*|\
    http://[::1]|http://[::1]:*|http://[::1]/*|http://[::1]:*/?*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

ollama_tags_url() {
  printf '%s/api/tags' "${OLLAMA_API_BASE_VALUE%/}"
}

ollama_ready() {
  local tags_url
  tags_url="$(ollama_tags_url)"
  "${PYTHON}" - "${tags_url}" >/dev/null 2>&1 <<'PY'
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

request = Request(sys.argv[1], headers={"Accept": "application/json"})
try:
    with urlopen(request, timeout=2) as response:
        raise SystemExit(0 if 200 <= response.status < 300 else 1)
except (OSError, URLError):
    raise SystemExit(1)
PY
}

find_ollama_bin() {
  if [[ -n "${OLLAMA_BIN:-}" && -x "${OLLAMA_BIN}" ]]; then
    printf '%s\n' "${OLLAMA_BIN}"
    return 0
  fi

  local candidate
  for candidate in \
    "${HOME}/.local/bin/ollama" \
    "/usr/local/bin/ollama" \
    "/usr/bin/ollama" \
    "/snap/bin/ollama"
  do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  command -v ollama 2>/dev/null || return 1
}

start_ollama_if_needed() {
  case "${LLM_PROVIDER_VALUE}" in
    openai-compatible|openai_compatible|vllm)
      echo "跳过 Ollama 自动启动: LLM_PROVIDER=${LLM_PROVIDER_VALUE}" | tee -a "${LOG_FILE}"
      return 0
      ;;
  esac

  if [[ "${OLLAMA_AUTOSTART_VALUE}" =~ ^(0|false|False|FALSE|no|No|NO|off|Off|OFF)$ ]]; then
    echo "跳过 Ollama 自动启动: OLLAMA_AUTOSTART=${OLLAMA_AUTOSTART_VALUE}" | tee -a "${LOG_FILE}"
    return 0
  fi

  if ! is_local_ollama_endpoint; then
    echo "跳过 Ollama 自动启动: OLLAMA_API_BASE=${OLLAMA_API_BASE_VALUE} 不是本机地址" | tee -a "${LOG_FILE}"
    return 0
  fi

  if ollama_ready; then
    echo "Ollama 服务已运行: ${OLLAMA_API_BASE_VALUE%/}" | tee -a "${LOG_FILE}"
    return 0
  fi

  local ollama_bin
  if ! ollama_bin="$(find_ollama_bin)"; then
    echo "未找到 ollama 可执行文件，无法自动启动 Ollama。" | tee -a "${LOG_FILE}"
    return 0
  fi

  echo "正在自动启动 Ollama: ${ollama_bin} serve" | tee -a "${LOG_FILE}"
  if command -v setsid >/dev/null 2>&1; then
    setsid -f "${ollama_bin}" serve >>"${OLLAMA_LOG_FILE}" 2>&1 </dev/null
  else
    nohup "${ollama_bin}" serve >>"${OLLAMA_LOG_FILE}" 2>&1 </dev/null &
    disown || true
  fi

  local waited
  for ((waited = 1; waited <= OLLAMA_START_TIMEOUT_VALUE; waited++)); do
    if ollama_ready; then
      echo "Ollama 服务已就绪: ${OLLAMA_API_BASE_VALUE%/}" | tee -a "${LOG_FILE}"
      return 0
    fi
    sleep 1
  done

  echo "Ollama 自动启动后仍未连通，详情见: ${OLLAMA_LOG_FILE}" | tee -a "${LOG_FILE}"
}

vllm_autostart_enabled() {
  [[ ! "${VLLM_AUTOSTART_VALUE}" =~ ^(0|false|False|FALSE|no|No|NO|off|Off|OFF)$ ]]
}

load_vllm_env() {
  local vllm_env
  if ! vllm_env="$("${VLLM_START_SCRIPT}" --print-env)"; then
    echo "读取 vLLM 环境变量失败: ${VLLM_START_SCRIPT} --print-env" | tee -a "${LOG_FILE}"
    return 1
  fi
  eval "${vllm_env}"
  LLM_PROVIDER_VALUE="${LLM_PROVIDER:-${LLM_PROVIDER_VALUE}}"
}

start_vllm_if_needed() {
  case "${LLM_PROVIDER_VALUE}" in
    vllm)
      ;;
    *)
      return 0
      ;;
  esac

  if [[ ! -x "${VLLM_START_SCRIPT}" ]]; then
    echo "缺少 vLLM 启动脚本: ${VLLM_START_SCRIPT}" | tee -a "${LOG_FILE}"
    return 1
  fi

  load_vllm_env
  VLLM_MANAGED_MODEL_PATH="${VLLM_MODEL_PATH:-${VLLM_MANAGED_MODEL_PATH}}"
  VLLM_MANAGED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${VLLM_MANAGED_MODEL_NAME}}"

  if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "检查 vLLM 环境: ${VLLM_START_SCRIPT} check" | tee -a "${LOG_FILE}"
    "${VLLM_START_SCRIPT}" check 2>&1 | tee -a "${LOG_FILE}"
    return 0
  fi

  if ! vllm_autostart_enabled; then
    echo "跳过 vLLM 自动启动: VLLM_AUTOSTART=${VLLM_AUTOSTART_VALUE}" | tee -a "${LOG_FILE}"
    return 0
  fi

  echo "确保 vLLM 服务已启动: ${VLLM_START_SCRIPT}" | tee -a "${LOG_FILE}"
  "${VLLM_START_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
  if is_project_managed_vllm_pid; then
    VLLM_WAS_MANAGED=1
  else
    VLLM_WAS_MANAGED=0
  fi
}

schedule_vllm_delayed_stop() {
  local desktop_token watchdog_pid

  if [[ "${LLM_PROVIDER_VALUE}" != "vllm" || "${CHECK_ONLY}" == "1" || "${VLLM_WAS_MANAGED}" != "1" ]]; then
    return 0
  fi

  if [[ ! "${VLLM_DESKTOP_SHUTDOWN_DELAY_VALUE}" =~ ^[0-9]+$ ]]; then
    echo "Invalid VLLM_DESKTOP_SHUTDOWN_DELAY: ${VLLM_DESKTOP_SHUTDOWN_DELAY_VALUE}" | tee -a "${LOG_FILE}"
    return 1
  fi

  mkdir -p "${VLLM_STATE_DIR}"
  desktop_token="$(generate_session_token)"
  printf '%s\n' "${desktop_token}" >"${VLLM_DESKTOP_SESSION_FILE}"

  clear_previous_vllm_watchdog

  echo "启动 vLLM 延迟关闭 watchdog..." | tee -a "${LOG_FILE}"
  nohup env \
    VLLM_STATE_DIR="${VLLM_STATE_DIR}" \
    VLLM_DESKTOP_SESSION_FILE="${VLLM_DESKTOP_SESSION_FILE}" \
    VLLM_DESKTOP_WATCHDOG_PID_FILE="${VLLM_DESKTOP_WATCHDOG_PID_FILE}" \
    VLLM_MANAGED_MODEL_PATH="${VLLM_MANAGED_MODEL_PATH}" \
    VLLM_MANAGED_MODEL_NAME="${VLLM_MANAGED_MODEL_NAME}" \
    VLLM_DESKTOP_PID="$$" \
    VLLM_DESKTOP_SESSION_TOKEN="${desktop_token}" \
    VLLM_DESKTOP_SHUTDOWN_DELAY="${VLLM_DESKTOP_SHUTDOWN_DELAY_VALUE}" \
    "${VLLM_START_SCRIPT}" delayed-stop >>"${LOG_FILE}" 2>&1 </dev/null &
  watchdog_pid="$!"
  disown || true
  echo "vLLM delayed-stop watchdog PID: ${watchdog_pid}" | tee -a "${LOG_FILE}"
}

CHECK_ONLY=0
STRICT_SERVICES=0
for arg in "$@"; do
  case "${arg}" in
    --check)
      CHECK_ONLY=1
      ;;
    --strict-services)
      STRICT_SERVICES=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: ${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${APP_ROOT}"
mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "缺少项目虚拟环境: ${PYTHON}" | tee "${LOG_FILE}"
  if command -v uv >/dev/null 2>&1; then
    echo "请先在项目根目录执行: uv sync --extra desktop" | tee -a "${LOG_FILE}"
  else
    echo "未找到 uv。请先安装依赖并创建 .venv。" | tee -a "${LOG_FILE}"
  fi
  exit 1
fi

echo "准备启动标签检测桌面端..." | tee "${LOG_FILE}"
echo "部署目录: ${DEPLOY_ROOT}" | tee -a "${LOG_FILE}"
echo "应用目录: ${APP_ROOT}" | tee -a "${LOG_FILE}"

echo "检查离线授权..." | tee -a "${LOG_FILE}"
if ! "${PYTHON}" -m label_detection.license verify \
  --license "${LABEL_DETECTION_LICENSE:-${DEPLOY_ROOT}/licenses/license.json}" \
  2>&1 | tee -a "${LOG_FILE}"; then
  LICENSE_VALID=0
  if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "授权检查失败。" | tee -a "${LOG_FILE}"
    exit 1
  fi
  echo "授权无效，桌面端将只显示机器码和授权状态。" | tee -a "${LOG_FILE}"
fi
if [[ "${LICENSE_VALID}" == "1" ]]; then
  start_vllm_if_needed
  if [[ "${LLM_PROVIDER_VALUE}" == "vllm" ]]; then
    if is_project_managed_vllm_pid; then
      VLLM_WAS_MANAGED=1
    fi
  fi
  start_ollama_if_needed
else
  echo "跳过模型服务自动启动: 授权无效" | tee -a "${LOG_FILE}"
fi

CHECK_ARGS=()
if [[ "${STRICT_SERVICES}" == "1" ]]; then
  CHECK_ARGS+=(--strict-services)
fi

echo "运行桌面端启动检查..." | tee -a "${LOG_FILE}"
if ! "${PYTHON}" "${APP_ROOT}/scripts/check_desktop_env.py" "${CHECK_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
  echo "启动检查失败，详情见: ${LOG_FILE}" | tee -a "${LOG_FILE}"
  exit 1
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "启动检查通过。" | tee -a "${LOG_FILE}"
  exit 0
fi

echo "启动标签检测桌面端..." | tee -a "${LOG_FILE}"
schedule_vllm_delayed_stop
exec "${PYTHON}" -m desktop_app.main
