# 海康 MVS 工业相机集成修改方案

更新时间：2026-04-24

## 目标

将当前已经可用的海康 MVS 单张取图能力，整理成桌面端可稳定长期使用的相机集成功能。优先保证现场操作不误导、不假死、可诊断、可回退。

本方案聚焦 MV-CU060-10GC 这台 GigE Vision 相机，但实现应保持对同类 HIKROBOT MVS 相机可复用。

## 当前状态

实机验证已经通过：

```bash
.venv/bin/python scripts/camera_capture_mvs.py --model MV-CU060-10GC
.venv/bin/python scripts/camera_probe_mvs.py --json
```

已确认相机可枚举、可抓图：

```json
{
  "transport_layer": "GigE",
  "model_name": "MV-CU060-10GC",
  "serial_number": "DA3592226",
  "vendor_name": "Hikrobot",
  "ip_address": "169.254.1.200"
}
```

已落地的代码能力：

- `desktop_app/devices/camera/hikrobot_mvs.py`：MVS SDK 枚举、选型、抓图、资源释放。
- `scripts/camera_probe_mvs.py`：现场枚举诊断。
- `scripts/camera_capture_mvs.py`：现场单张抓图诊断。
- `desktop_app/devices/camera/factory.py`：通过 `DESKTOP_CAMERA_BACKEND` 切换 mock / hikrobot-mvs。
- 桌面端按钮文案已从 Mock 相机调整为“相机取图”。

当前测试情况：

- `.venv` 原本没有 `pytest`。
- 当前执行环境无法访问 PyPI/镜像源，网络安装暂不可用。
- 使用系统 Python 的 pytest 跑过相机相关测试：`10 passed, 1 skipped`。
- `.venv` 已补 `pip`，但 `pytest` 依赖未完整安装，不把它作为本方案的阻塞项。

## 主要问题

### P0：桌面端相机取图会阻塞主线程

位置：

- `desktop_app/controllers/app_controller.py`
- `desktop_app/devices/camera/hikrobot_mvs.py`

当前 `capture_mock_image()` 直接调用 `camera_adapter.capture()`。真实相机断线、网络切换、SDK 超时或首次初始化时，Qt 主线程会被阻塞，表现为界面卡住。

这必须优先修。工业相机现场使用时，网络和供电状态变化是常态，UI 不能因为一次取图失败而假死。

### P0：真实相机后端启用方式不够显式

位置：

- `desktop_app/devices/camera/factory.py`
- `desktop_app/app.py`
- `desktop_app/ui/main_window.py`

默认后端仍是 `mock`，但按钮已经显示“相机取图”。如果未设置：

```bash
DESKTOP_CAMERA_BACKEND=hikrobot-mvs
```

按钮实际会从样本图片取图。这对现场操作有误导风险。

### P1：输出文件名只精确到秒，连续取图可能覆盖

位置：

- `desktop_app/devices/camera/hikrobot_mvs.py`

当前输出格式类似：

```text
20260424-144925_uncoded.jpg
```

同一秒内连续点击可能生成同名文件。需要加入毫秒或微秒。

### P1：命名仍残留 mock 语义

位置：

- `MainWindow.capture_mock_requested`
- `AppController.capture_mock_image`

功能已经扩展为真实相机，内部命名仍是 mock，后续维护容易误判。

### P1：现场网络切换和相机启动缺少项目内文档

目前相机接入依赖单网口切换：

```bash
sudo nmcli con up hik-camera
sudo nmcli con down hik-camera
sudo nmcli dev connect enp4s0
```

这些命令需要写入项目文档，避免后续现场操作靠聊天记录。

### P2：相机状态缺少 UI 可见反馈

桌面端目前只有“相机取图”按钮。用户无法直接看到当前后端、目标型号、相机 IP、最近错误、是否正在取图。

这不是首要阻塞，但会影响现场排障效率。

## 修改方案

### 阶段 1：相机取图后台化

新增文件：

- `desktop_app/workers/camera_capture_worker.py`

建议实现：

```python
class CameraCaptureWorker(QObject):
    started_status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, camera_adapter: CameraAdapter, preferred_code: str | None) -> None:
        ...

    @Slot()
    def run(self) -> None:
        ...
```

控制器修改：

- 将 `capture_mock_image()` 重命名为 `capture_camera_image()`。
- 新增 `_camera_thread`、`_camera_worker`，与检测 worker 分开管理。
- 点击“相机取图”后禁用取图按钮、选择图片按钮和开始检测按钮。
- 取图成功后调用 `view.set_target_image_path(path)`。
- 取图失败后显示明确错误，例如：

```text
相机取图失败：未枚举到任何海康相机，请检查网线、供电、IP 网段和 MVS 环境。
```

验收标准：

- 相机断开时点击“相机取图”，UI 不假死。
- 超时后能恢复按钮状态。
- 成功取图后目标图预览更新。
- 检测任务运行期间不允许再次取图。

### 阶段 2：明确真实相机后端

保守做法：默认仍保持 `mock`，避免没有相机的开发环境启动失败；但 UI 必须显示当前后端。

建议修改：

- `CameraAdapter` 增加可选显示字段，或直接使用 `camera_adapter.name`。
- MainWindow 增加一个简短状态标签，例如：

```text
相机：mock-camera
相机：hikrobot-mvs
```

启动真实相机桌面端的命令固定为：

```bash
DESKTOP_CAMERA_BACKEND=hikrobot-mvs \
HIKROBOT_CAMERA_MODEL=MV-CU060-10GC \
.venv/bin/python -m desktop_app.main
```

更稳的生产启动命令建议指定序列号：

```bash
DESKTOP_CAMERA_BACKEND=hikrobot-mvs \
HIKROBOT_CAMERA_SERIAL=DA3592226 \
HIKROBOT_CAMERA_MODEL=MV-CU060-10GC \
.venv/bin/python -m desktop_app.main
```

验收标准：

- 未设置环境变量时，UI 明确显示 `mock-camera`。
- 设置真实相机后端时，UI 明确显示 `hikrobot-mvs`。
- 后端配置错误时，启动不崩溃，点击取图时给出可读错误。

### 阶段 3：修正输出文件命名

修改：

- `build_capture_output_path()`

建议从：

```python
datetime.now().strftime("%Y%m%d-%H%M%S")
```

改为：

```python
datetime.now().strftime("%Y%m%d-%H%M%S-%f")
```

验收标准：

- 同一秒内连续两次取图不会覆盖。
- 文件仍按编码分目录保存。

### 阶段 4：清理 mock 命名

修改：

- `capture_mock_requested` -> `capture_camera_requested`
- `capture_mock_image()` -> `capture_camera_image()`
- 更新对应测试名和信号连接。

兼容策略：

- 如果担心一次性改动影响测试，可以先新增新名称，保留旧名称作为别名一版。
- 新代码和文档统一使用 camera 命名。

验收标准：

- `rg "capture_mock|Mock 相机"` 只在历史计划文档或 mock adapter 测试中出现。
- 桌面端行为不变。

### 阶段 5：补充现场运行文档

建议新增或更新：

- `docs/hikrobot_mvs_camera_runbook.md`
- 或在 `README.md` 增加“海康相机接入”章节。

内容至少包含：

1. 单网口切换到相机模式：

```bash
sudo nmcli con up hik-camera
ip -4 addr show dev enp4s0
.venv/bin/python scripts/camera_probe_mvs.py --json
```

2. 单张抓图验证：

```bash
.venv/bin/python scripts/camera_capture_mvs.py --model MV-CU060-10GC
```

3. 启动真实相机桌面端：

```bash
DESKTOP_CAMERA_BACKEND=hikrobot-mvs \
HIKROBOT_CAMERA_MODEL=MV-CU060-10GC \
.venv/bin/python -m desktop_app.main
```

4. 恢复上网：

```bash
sudo nmcli con down hik-camera
sudo nmcli dev connect enp4s0
```

5. 常见排障：

- `find no device`：检查供电、网线、IP 网段、`hik-camera` 配置。
- `carrier=0`：物理链路未连接或相机未供电。
- MVS 枚举不到网卡：先确认 `ip -4 addr show dev enp4s0` 有 IPv4。
- CLI 走代理失败：临时取消 `HTTP_PROXY/HTTPS_PROXY` 后再测。

## 建议的代码改动清单

第一批必须改：

- `desktop_app/workers/camera_capture_worker.py`
- `desktop_app/controllers/app_controller.py`
- `desktop_app/ui/main_window.py`
- `tests/test_desktop_scanner_controller.py`
- `tests/test_hikrobot_mvs_camera.py`

第二批建议改：

- `desktop_app/devices/camera/base.py`
- `desktop_app/devices/camera/factory.py`
- `desktop_app/devices/camera/hikrobot_mvs.py`
- `README.md` 或 `docs/hikrobot_mvs_camera_runbook.md`

## 验证命令

基础语法检查：

```bash
.venv/bin/python -m py_compile \
  desktop_app/devices/camera/hikrobot_mvs.py \
  desktop_app/devices/camera/factory.py \
  scripts/camera_capture_mvs.py \
  scripts/camera_probe_mvs.py \
  desktop_app/controllers/app_controller.py \
  desktop_app/app.py
```

相机枚举：

```bash
.venv/bin/python scripts/camera_probe_mvs.py --json
```

相机单张抓图：

```bash
.venv/bin/python scripts/camera_capture_mvs.py --model MV-CU060-10GC
```

桌面端真实相机模式：

```bash
DESKTOP_CAMERA_BACKEND=hikrobot-mvs \
HIKROBOT_CAMERA_MODEL=MV-CU060-10GC \
.venv/bin/python -m desktop_app.main
```

单元测试，等 `.venv` pytest 可用后执行：

```bash
.venv/bin/python -m pytest \
  tests/test_hikrobot_mvs_camera.py \
  tests/test_desktop_camera_factory.py \
  tests/test_desktop_scanner_controller.py \
  -q
```

当前如果 `.venv` 里 pytest 不可用，可临时用系统 Python 做参考验证：

```bash
python3 -m pytest \
  tests/test_hikrobot_mvs_camera.py \
  tests/test_desktop_camera_factory.py \
  tests/test_desktop_scanner_controller.py \
  -q
```

## 风险和注意事项

- MVS SDK 初始化/反初始化是进程级资源，当前 `_MVS_LOCK` 需要保留，避免并发取图或枚举。
- 真实相机单网口使用会影响上网，桌面端不应自动改网卡配置。
- 不建议在 UI 线程直接调用 MVS SDK。
- 默认后端保持 mock 更适合开发；生产现场应通过启动脚本显式启用 `hikrobot-mvs`。
- 相机参数如曝光、增益、UserSet 不应硬编码，保留环境变量或后续 UI 配置入口。

## 推荐实施顺序

1. 新增 `CameraCaptureWorker`，让相机取图不阻塞 UI。
2. 控制器和 UI 信号从 mock 命名改为 camera 命名。
3. 修复输出文件名到微秒级。
4. UI 显示当前相机后端。
5. 补充现场 runbook。
6. pytest 可用后补齐/更新测试并全量跑相机相关测试。

