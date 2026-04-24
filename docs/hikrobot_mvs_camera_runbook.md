# 海康 MVS 相机现场运行手册

更新时间：2026-04-24

## 适用范围

本手册用于在当前服务器 checkout 中验证和启动海康 HIKROBOT MVS 工业相机。当前已验证型号为 `MV-CU060-10GC`，同类 GigE Vision MVS 相机可复用相同流程。

## 切换到相机网络

现场单网口连接相机时，先切到相机网络配置：

```bash
sudo nmcli con up hik-camera
ip -4 addr show dev enp4s0
.venv/bin/python scripts/camera_probe_mvs.py --json
```

`camera_probe_mvs.py --json` 应能看到类似字段：

```json
[
  {
    "transport_layer": "GigE",
    "model_name": "MV-CU060-10GC",
    "serial_number": "DA3592226",
    "vendor_name": "Hikrobot",
    "ip_address": "169.254.1.200"
  }
]
```

## 单张抓图验证

优先按型号验证：

```bash
.venv/bin/python scripts/camera_capture_mvs.py --model MV-CU060-10GC
```

现场多相机或型号重复时，建议按序列号验证：

```bash
.venv/bin/python scripts/camera_capture_mvs.py \
  --model MV-CU060-10GC \
  --serial DA3592226
```

## 启动桌面端

桌面端默认使用真实相机后端 `hikrobot-mvs`，不会从样本图片模拟取图。工具栏会显示当前相机后端，启动后会自动检查相机并进入实时预览。
工具栏同时显示 `画面` 亮度状态：`正常` 可以直接拍照；`略暗` 建议确认补光；`偏暗` 时软件会提示先补光或调曝光，否则检测结果容易不稳定。桌面端默认会对偏暗相机帧做软件增亮，实时预览和拍照保存使用同一张增亮后的图，避免“预览看着可以、检测用图偏暗”的不一致。

现场操作流程：

1. 有相机时，目标图区域显示实时预览。
2. 点击 `拍照保存`，当前画面会冻结并保存为本次检测目标图。
3. 点击 `开始检测`，检测这张已锁定的目标图。
4. 查看结果后点击 `下一张`，目标图区域恢复实时预览，准备下一次拍照。
5. 如果画面不合适，点击 `重拍` 回到实时预览。

真实相机模式：

```bash
DESKTOP_CAMERA_BACKEND=hikrobot-mvs \
HIKROBOT_CAMERA_MODEL=MV-CU060-10GC \
.venv/bin/python -m desktop_app.main
```

桌面端默认会尝试启用连续自动曝光。如果现场光源稳定，推荐先用默认自动曝光观察 `画面` 状态；如果仍然偏暗，再使用手动曝光/增益固定参数：

```bash
DESKTOP_CAMERA_BACKEND=hikrobot-mvs \
HIKROBOT_CAMERA_MODEL=MV-CU060-10GC \
HIKROBOT_CAMERA_EXPOSURE_US=15000 \
HIKROBOT_CAMERA_GAIN=8 \
.venv/bin/python -m desktop_app.main
```

如果画面仍偏暗，可以把 `HIKROBOT_CAMERA_EXPOSURE_US` 调到 `25000` 左右再试；如果画面拖影，优先增加补光，避免继续拉高曝光。手动曝光或手动增益存在时，会覆盖自动曝光/自动增益。

如果需要完全保存相机原始亮度，不做软件增亮，可以显式关闭：

```bash
DESKTOP_CAMERA_AUTO_ENHANCE=0
```

自动曝光/增益也可以显式配置：

```bash
HIKROBOT_CAMERA_EXPOSURE_AUTO=continuous
HIKROBOT_CAMERA_GAIN_AUTO=continuous
```

可用值为 `off`、`once`、`continuous`。

生产现场建议指定序列号，避免插入多台相机时选错设备：

```bash
DESKTOP_CAMERA_BACKEND=hikrobot-mvs \
HIKROBOT_CAMERA_SERIAL=DA3592226 \
HIKROBOT_CAMERA_MODEL=MV-CU060-10GC \
.venv/bin/python -m desktop_app.main
```

## 恢复上网

相机验证或桌面端使用结束后恢复普通网络：

```bash
sudo nmcli con down hik-camera
sudo nmcli dev connect enp4s0
```

## 常见排障

- `find no device` 或未枚举到相机：检查供电、网线、IP 网段和 `hik-camera` 配置。
- `carrier=0`：物理链路未连接，或相机未供电。
- MVS 枚举不到网卡：先确认 `ip -4 addr show dev enp4s0` 有 IPv4 地址。
- CLI 走代理失败：临时取消 `HTTP_PROXY`、`HTTPS_PROXY` 后再测。
- 桌面端提示 `未检测到相机`：检查供电、网线、IP 网段、MVS 环境和 `HIKROBOT_CAMERA_MODEL/HIKROBOT_CAMERA_SERIAL` 筛选条件。
- 桌面端显示 `hikrobot-mvs (unavailable)`：后端配置初始化失败，点击 `重连相机` 会显示具体错误。
