#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE="${PROJECT_ROOT}/packaging/linux/gree-label-detection.desktop.template"
APP_DIR="${HOME}/.local/share/applications"
APP_FILE="${APP_DIR}/gree-label-detection.desktop"
DESKTOP_DIR="${HOME}/Desktop"
DESKTOP_FILE="${DESKTOP_DIR}/gree-label-detection.desktop"

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "缺少模板文件: ${TEMPLATE}" >&2
  exit 1
fi

mkdir -p "${APP_DIR}"
sed "s|@PROJECT_ROOT@|${PROJECT_ROOT}|g" "${TEMPLATE}" > "${APP_FILE}"
chmod +x "${APP_FILE}"

echo "已安装应用菜单入口: ${APP_FILE}"

if [[ -d "${DESKTOP_DIR}" ]]; then
  cp "${APP_FILE}" "${DESKTOP_FILE}"
  chmod +x "${DESKTOP_FILE}"
  echo "已安装桌面图标: ${DESKTOP_FILE}"
else
  echo "未发现桌面目录，跳过桌面图标: ${DESKTOP_DIR}"
fi

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${APP_DIR}" >/dev/null 2>&1 || true
fi

echo "可先运行检查: ${PROJECT_ROOT}/scripts/start_desktop.sh --check"
