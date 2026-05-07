#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/desktop_app"
LOG_FILE="${LOG_DIR}/startup.log"
OLLAMA_LOG_FILE="${LOG_DIR}/ollama.log"
LLM_PROVIDER_VALUE="${LLM_PROVIDER:-ollama}"
OLLAMA_API_BASE_VALUE="${OLLAMA_API_BASE:-http://localhost:11434}"
OLLAMA_AUTOSTART_VALUE="${OLLAMA_AUTOSTART:-1}"
OLLAMA_START_TIMEOUT_VALUE="${OLLAMA_START_TIMEOUT:-20}"

usage() {
  cat <<'EOF'
Usage: scripts/start_desktop.sh [--check] [--strict-services]

Options:
  --check             Only run desktop environment checks.
  --strict-services   Treat LLM/service connectivity warnings as failures.

Environment:
  LLM_PROVIDER=ollama       Use Ollama by default; set vllm/openai-compatible to skip Ollama autostart.
  OLLAMA_AUTOSTART=0       Disable automatic local Ollama startup.
  OLLAMA_BIN=/path/ollama  Override the Ollama executable path.
  OLLAMA_START_TIMEOUT=20  Seconds to wait for Ollama startup.
EOF
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

cd "${PROJECT_ROOT}"
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
start_ollama_if_needed

CHECK_ARGS=()
if [[ "${STRICT_SERVICES}" == "1" ]]; then
  CHECK_ARGS+=(--strict-services)
fi

echo "运行桌面端启动检查..." | tee -a "${LOG_FILE}"
if ! "${PYTHON}" "${PROJECT_ROOT}/scripts/check_desktop_env.py" "${CHECK_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
  echo "启动检查失败，详情见: ${LOG_FILE}" | tee -a "${LOG_FILE}"
  exit 1
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "启动检查通过。" | tee -a "${LOG_FILE}"
  exit 0
fi

echo "启动标签检测桌面端..." | tee -a "${LOG_FILE}"
exec "${PYTHON}" -m desktop_app.main
