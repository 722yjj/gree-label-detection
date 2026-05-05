#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
LOG_DIR="${PROJECT_ROOT}/results/desktop_app"
LOG_FILE="${LOG_DIR}/startup.log"

usage() {
  cat <<'EOF'
Usage: scripts/start_desktop.sh [--check] [--strict-services]

Options:
  --check             Only run desktop environment checks.
  --strict-services   Treat Ollama/service connectivity warnings as failures.
EOF
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

CHECK_ARGS=()
if [[ "${STRICT_SERVICES}" == "1" ]]; then
  CHECK_ARGS+=(--strict-services)
fi

echo "运行桌面端启动检查..." | tee "${LOG_FILE}"
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
