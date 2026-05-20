#!/usr/bin/env bash
set -Eeuo pipefail

BIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${BIN_DIR}/.." && pwd)"
DEPLOY_ENV="${DEPLOY_ENV:-${DEPLOY_ROOT}/config/deploy.env}"
if [[ -f "${DEPLOY_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${DEPLOY_ENV}"
  set +a
fi

APP_ROOT="${APP_ROOT:-${DEPLOY_ROOT}/app}"
if [[ ! -d "${APP_ROOT}" && -f "${DEPLOY_ROOT}/main.py" ]]; then
  APP_ROOT="${DEPLOY_ROOT}"
fi
APP_PYTHON="${APP_PYTHON:-${PYTHON:-${DEPLOY_ROOT}/venvs/app/bin/python}}"
if [[ ! -x "${APP_PYTHON}" && -x "${APP_ROOT}/.venv/bin/python" ]]; then
  APP_PYTHON="${APP_ROOT}/.venv/bin/python"
fi

export DEPLOY_ROOT APP_ROOT LABEL_DETECTION_DEPLOY_ROOT="${DEPLOY_ROOT}"
export LABEL_DETECTION_LICENSE="${LABEL_DETECTION_LICENSE:-${DEPLOY_ROOT}/licenses/license.json}"

echo "Deploy root: ${DEPLOY_ROOT}"
echo "App root: ${APP_ROOT}"
echo "Python: ${APP_PYTHON}"

if [[ ! -x "${APP_PYTHON}" ]]; then
  echo "Missing app Python: ${APP_PYTHON}" >&2
  exit 1
fi

"${APP_PYTHON}" -m label_detection.license fingerprint --json
"${APP_PYTHON}" -m label_detection.license verify --license "${LABEL_DETECTION_LICENSE}"

if [[ -x "${APP_ROOT}/scripts/start_vllm.sh" ]]; then
  DEPLOY_ROOT="${DEPLOY_ROOT}" APP_ROOT="${APP_ROOT}" "${APP_ROOT}/scripts/start_vllm.sh" check
else
  echo "Skip vLLM check: missing ${APP_ROOT}/scripts/start_vllm.sh"
fi

"${APP_PYTHON}" "${APP_ROOT}/scripts/check_desktop_env.py"
