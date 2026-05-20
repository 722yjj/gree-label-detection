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

APP_DIR="${HOME}/.local/share/applications"
DESKTOP_DIR="${HOME}/Desktop"
APP_FILE="${APP_DIR}/label-detection.desktop"
DESKTOP_FILE="${DESKTOP_DIR}/label-detection.desktop"
START_SCRIPT="${DEPLOY_ROOT}/bin/start_desktop.sh"

mkdir -p "${APP_DIR}" "${DEPLOY_ROOT}/licenses" "${DEPLOY_ROOT}/results"
cat >"${APP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=标签检测
Comment=启动交付版标签检测桌面端
Exec=${START_SCRIPT}
Path=${DEPLOY_ROOT}
Terminal=true
Categories=Utility;
StartupNotify=true
EOF
chmod +x "${APP_FILE}" "${START_SCRIPT}" "${DEPLOY_ROOT}/bin/check.sh"

echo "Installed application entry: ${APP_FILE}"

if [[ -d "${DESKTOP_DIR}" ]]; then
  cp "${APP_FILE}" "${DESKTOP_FILE}"
  chmod +x "${DESKTOP_FILE}"
  echo "Installed desktop icon: ${DESKTOP_FILE}"
else
  echo "Desktop directory not found, skipped icon: ${DESKTOP_DIR}"
fi

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${APP_DIR}" >/dev/null 2>&1 || true
fi

echo "Run deployment check: ${DEPLOY_ROOT}/bin/check.sh"
