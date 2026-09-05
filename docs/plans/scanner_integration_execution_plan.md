# 扫码器接入执行方案

## 1. 目标与边界

本方案基于当前桌面端实现，完成“扫码器进入现有桌面检测流程”的落地设计，目标工作流如下：

```text
扫码器输出编码
  -> 桌面端接收并规范化编码
  -> 自动查询模板
  -> 操作员确认模板
  -> 相机拍照 / 选择图片
  -> 点击开始检测
  -> 展示检测结果
```

本方案的边界如下：

- 不改动检测核心入口，检测仍由 `label_detection.workflows.unified.run_unified_detection(...)` 执行。
- 不改动模板发现规则，模板查询仍由 `desktop_app.repositories.template_repository.TemplateRepository` 负责。
- 不改动当前图像输入主线，目标图仍来自“本地图片”或 `MockCameraAdapter`，后续再扩展真实相机。
- 本轮重点是“扫码输入如何稳定驱动已有 UI 流程”，不是一次性做完真实硬件生态。

## 2. 当前项目现状

结合现有代码，桌面端与扫码器相关的现状如下。

### 2.1 已有能力

- `desktop_app/ui/main_window.py`
  - 已有编码输入框 `code_input`
  - 已有“查询模板”“模拟扫码”“选择图片”“Mock 相机取图”“开始检测”按钮
  - `code_input.returnPressed` 已连接模板查询入口
  - `set_templates(...)` 已内置“若存在模板则默认选中第一项”
- `desktop_app/controllers/app_controller.py`
  - `query_templates()` 已完成“读取编码 -> normalize -> 查模板 -> 刷新模板列表”
  - `capture_mock_image()` 会读取当前编码并优先选择同编码样本图
  - `run_detection()` 已通过 `DetectionWorker` + `DetectionService` 在后台线程执行检测
- `desktop_app/repositories/template_repository.py`
  - 当前从 `samples/pdfs` 和 `samples/images/original` 扫描模板
  - `find_by_code(code)` 采用精确匹配
- `desktop_app/devices/scanner_input.py`
  - 当前仅提供 `KeyboardWedgeScannerInput.normalize()`

### 2.2 当前缺口

- `desktop_app/app.py` 构造 `AppController` 时，还没有扫码器 adapter 注入点。
- `MainWindow.mock_scan_button` 当前直接连接 `query_code_requested`，并没有独立的“模拟扫码完成”行为。
- `AppController` 还没有统一的扫码入口，例如 `handle_scanned_code(code: str)`。
- `MainWindow.set_busy()` 只禁用了按钮，未禁用 `code_input`，检测运行期间扫码枪输入仍可能打到编码框。
- 当前没有统一的焦点恢复策略，键盘模式扫码枪的稳定性还依赖人工保持焦点。
- 当前没有真实扫码器抽象层，无法平滑接入 `MV-ORH` 这类非键盘模式设备。

## 3. 设计约束

本方案需要满足以下约束。

### 3.1 不破坏现有主线

扫码器接入只应扩展编码来源，不应改变以下既有责任边界：

- 模板查询仍由 `TemplateRepository.find_by_code(...)` 完成。
- 模板默认选中第一项仍由 `MainWindow.set_templates(...)` 负责。
- 模板预览刷新仍由 `template_selection_changed -> refresh_template_preview()` 负责。
- 检测仍由 `run_detection()` 发起，不因扫码行为自动触发。

### 3.2 编码必须先规范化再查询

当前模板查询是精确匹配：

```python
TemplateRepository.find_by_code(code)
```

因此扫码输入如果残留以下内容，就会直接导致查不到模板：

- 回车 `\r` / `\n`
- `Tab`
- 前缀、后缀
- 设备拼接的说明文字

扫码规范化逻辑必须先于模板查询执行。

### 3.3 检测线程不能被扫码事件打断

当前检测在 `QThread` 中执行，扫码器接入后必须满足：

- 检测运行时不允许新的扫码输入破坏 UI 状态
- 设备线程或回调不能直接操作 Qt UI
- 非主线程来的扫码结果必须通过 Qt signal 进入主线程

## 4. 接入策略

### 4.1 推荐优先级

按投入产出比，建议接入顺序如下：

1. USB HID 键盘模式
2. 厂商 SDK / TCP / 串口模式
3. 更复杂的 HID-POS / CDC / 工业协议模式

### 4.2 为什么优先 USB HID 键盘模式

如果 `MV-ORH` 支持 keyboard wedge / HID keyboard 模式，则当前项目几乎不需要写专用驱动，设备行为会退化为：

```text
输入编码字符串 + 回车
```

这与当前 UI 已有的 `code_input.returnPressed` 机制天然兼容，投入最小、验证最快、风险最低。

### 4.3 什么时候需要专用 adapter

只有在设备无法工作于键盘模式，或者需要以下能力时，才进入专用 adapter 路线：

- 后台持续监听设备结果
- 读取串口、TCP 或厂商 SDK 回调
- 获取设备在线状态
- 接收主动推送的扫码结果
- 对扫码失败、断线、重复报码做专门处理

## 5. 目标架构

建议把扫码接入拆成“输入层”和“控制层”两部分。

### 5.1 输入层职责

输入层只负责产生原始编码，不负责模板查询或 UI 编排。输入来源包括：

- 手工键盘输入
- “模拟扫码”按钮
- USB HID 键盘模式扫码枪
- 后续 `MV-ORH` 专用 adapter

### 5.2 控制层职责

控制层统一处理扫码后的动作：

```text
收到编码
  -> normalize
  -> 写回 code_input
  -> 查模板
  -> 更新模板列表
  -> 保持 / 恢复编码框焦点
```

推荐落在 `AppController` 中，避免把扫码逻辑散落到 UI 或设备层。

### 5.3 统一调用链

建议形成如下调用链：

```text
手输回车 / 模拟扫码 / 真扫码
  -> AppController.handle_scanned_code(code, source=...)
  -> KeyboardWedgeScannerInput.normalize(...)
  -> TemplateRepository.refresh()
  -> TemplateRepository.find_by_code(code)
  -> MainWindow.set_templates(...)
  -> MainWindow.focus_code_input()
```

其中：

- `MainWindow.set_templates(...)` 已会默认选中第一项，无需在控制器中重复实现。
- 模板预览会沿用当前 `template_selection_changed` 机制自动刷新。

## 6. 详细设计

### 6.1 控制器改造

`desktop_app/controllers/app_controller.py` 建议新增以下入口。

```python
def handle_scanned_code(self, code: str, source: str = "scanner") -> None: ...
def handle_manual_query(self) -> None: ...
```

推荐行为：

- `handle_scanned_code(...)`
  - 统一处理真扫码、模拟扫码、回车扫码
  - 调用 `KeyboardWedgeScannerInput.normalize(...)`
  - 更新输入框文本
  - 复用模板查询逻辑
  - 查询结束后恢复编码框焦点
- `handle_manual_query()`
  - 从 `view.code_text()` 读取当前值
  - 再委托给 `handle_scanned_code(...)`

同时补充以下控制逻辑：

- 若检测正在运行，新的扫码输入直接忽略，并提示“检测进行中，已忽略扫码输入”
- 查询完成后统一恢复编码框焦点
- 检测完成或失败后统一恢复编码框焦点

### 6.2 UI 信号与焦点管理

`desktop_app/ui/main_window.py` 建议从“复用一个查询信号”调整为“明确区分来源”。

建议信号：

```python
manual_query_requested = Signal()
simulate_scan_requested = Signal()
```

建议连接关系：

- `query_button.clicked -> manual_query_requested`
- `mock_scan_button.clicked -> simulate_scan_requested`
- `code_input.returnPressed -> simulate_scan_requested`

这样可以明确表达：

- “查询模板”按钮是人工确认查询
- 回车和“模拟扫码”按钮代表“扫码完成”

建议新增以下 UI 方法：

```python
def focus_code_input(self, select_all: bool = False) -> None: ...
def set_code_input_enabled(self, enabled: bool) -> None: ...
```

建议行为：

- 窗口首次显示后自动聚焦编码框
- 模板查询结束后恢复焦点
- 检测结束后恢复焦点
- 检测运行时禁用 `code_input`，避免扫码枪数据在检测中落到输入框

### 6.3 编码规范化策略

当前 `KeyboardWedgeScannerInput.normalize()` 只做 `strip()`，后续应增强为至少支持：

- 去除 `\r` / `\n` / `\t`
- 去除首尾空白
- 可选剥离已知固定前缀和后缀

如果真实设备输出内容不是纯编码，可复用现有批处理中的主编码提取规则：

- 参考 `label_detection.batch_samples.extract_label_code(name)`
- 在必要时从扫码结果中提取主数字串

推荐规则：

1. 默认仅做轻量清洗，不主动修改合法编码
2. 只有真实设备确认带固定前后缀时，才追加定制剥离规则
3. 如果设备输出存在包装文本，再启用“提取主数字串”策略

### 6.4 扫码器 adapter 抽象

为后续非键盘模式设备预留统一接口，建议将扫码设备抽象独立到新目录：

```text
desktop_app/devices/
  scanner/
    __init__.py
    base.py
    keyboard_wedge.py
    mock.py
    mv_orh.py
```

建议基础接口如下：

```python
from PySide6.QtCore import QObject, Signal


class ScannerAdapter(QObject):
    code_scanned = Signal(str)
    availability_changed = Signal(bool, str)

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_available(self) -> bool: ...
```

说明：

- `KeyboardWedge` 模式可以不依赖后台线程，但仍可保留一个轻量 adapter 占位，统一依赖注入方式。
- `availability_changed` 用于向 UI 反馈“设备可用 / 不可用 / 断线”。
- 真设备线程不得直接操作 `MainWindow`，只能发 signal 给 `AppController`。

### 6.5 依赖注入点

`desktop_app/app.py` 建议扩展为可注入扫码器 adapter：

```python
controller = AppController(
    view=window,
    template_repository=TemplateRepository(),
    detection_service=DetectionService(),
    camera_adapter=MockCameraAdapter(),
    scanner_adapter=scanner_adapter,
)
```

对应要求：

- 无真实设备时可传 `None`
- 键盘模式下可暂不传 adapter，仅通过 UI 回车链路工作
- 非键盘模式设备接入时，由 `app.py` 统一创建并注入

## 7. 文件级改动清单

建议按以下文件粒度实施。

### 7.1 `desktop_app/ui/main_window.py`

改动目标：

- 拆分查询与模拟扫码信号
- 增加编码框焦点控制方法
- 在检测运行时禁用编码框
- 窗口初始化后默认聚焦编码框

建议新增或调整：

- `manual_query_requested`
- `simulate_scan_requested`
- `focus_code_input(...)`
- `set_code_input_enabled(...)`
- `set_busy(...)` 内同步处理 `code_input`

### 7.2 `desktop_app/controllers/app_controller.py`

改动目标：

- 建立统一扫码入口
- 将手输回车、模拟扫码、真设备扫码收敛到同一控制路径
- 增加忙碌态保护和焦点恢复

建议新增或调整：

- `handle_scanned_code(...)`
- `handle_manual_query()`
- `_query_templates_for_code(...)`
- `_restore_code_focus()`
- `_connect_scanner_signals()`，用于接入可选 adapter

### 7.3 `desktop_app/devices/scanner_input.py` 或 `desktop_app/devices/scanner/keyboard_wedge.py`

改动目标：

- 继续承载键盘模式输入规范化
- 后续可迁移到新目录

建议新增能力：

- 清理控制字符
- 前后缀剥离
- 必要时提取主数字串

### 7.4 `desktop_app/devices/scanner/base.py`

改动目标：

- 定义统一扫码器接口
- 明确 signal、生命周期和可用性表达

### 7.5 `desktop_app/devices/scanner/mock.py`

改动目标：

- 用于无硬件环境下模拟专用 adapter 回调
- 支持开发阶段验证 controller 链路

### 7.6 `desktop_app/devices/scanner/mv_orh.py`

改动目标：

- 在确认设备通信方式后承接真实设备实现
- 隔离厂商协议、线程和异常处理

### 7.7 `desktop_app/app.py`

改动目标：

- 增加扫码器 adapter 的装配点
- 保持无设备时仍可正常启动桌面端

## 8. 分阶段实施

### Phase 1：统一扫码入口

目标：

- 控制器具备统一扫码处理主线

任务：

- 在 `AppController` 中新增 `handle_scanned_code(...)`
- 将“回车”“模拟扫码”“查询模板”统一收敛到控制器
- 清理当前 `mock_scan_button` 仅复用查询信号的临时写法

交付标准：

- 手工输入编码后按回车，行为与未来真扫码一致
- “模拟扫码”按钮走统一扫码入口
- 查询完成后模板列表照常刷新，第一项自动选中

### Phase 2：补齐键盘模式体验

目标：

- 让 USB HID 键盘模式扫码枪可以直接投入使用

任务：

- 增加窗口初始聚焦
- 查询后恢复编码框焦点
- 检测完成后恢复编码框焦点
- 检测中禁用编码框，避免扫码输入污染 UI

交付标准：

- 连续多次扫码时，编码框都能稳定收到数据
- 每次扫码后自动查模板，无需额外点击
- 检测运行期间扫码不会打断当前任务

### Phase 3：抽象真实扫码器接口

目标：

- 为非键盘模式设备提供标准接入层

任务：

- 新增 `ScannerAdapter` 基础接口
- 将控制器扩展为可选接收 `scanner_adapter`
- 通过 signal 将扫码结果送入 `handle_scanned_code(...)`

交付标准：

- 控制层不再关心编码来自手输、mock 还是真设备
- 真设备接入时不需要重写模板查询逻辑

### Phase 4：接入 `MV-ORH`

目标：

- 在明确设备通信方式后新增专用实现

路径 A：设备支持键盘模式

- 不做复杂 adapter
- 只保留焦点管理和输入规范化
- 现场按 HID 键盘设备使用

路径 B：设备只支持 SDK / TCP / 串口

- 新增 `MVORHScannerAdapter`
- 在后台线程或异步监听中接收扫码结果
- 通过 Qt signal 派发到主线程
- 处理断线、超时、格式错误和重复报码

交付标准：

- 真设备报码后，UI 自动更新编码并自动查模板
- 设备断开时能给出明确状态提示

## 9. `MV-ORH` 接入前必须确认的信息

在编写专用 adapter 前，必须先向设备方确认以下信息：

1. 设备是否支持 USB HID 键盘模式
2. 设备输出是否自带回车、Tab 或固定前后缀
3. 设备是否会重复推送同一结果
4. 设备通信方式是串口、TCP、SDK 还是其他协议
5. 厂商 SDK 是否支持当前服务器 / 桌面运行环境
6. 异常状态如何体现：断线、超时、扫码失败、权限错误

如果第 1 项可行，应优先走键盘模式，不建议先写 SDK 版本。

## 10. UI 行为要求

扫码器接入后，界面行为建议固定为：

1. 窗口打开时，编码输入框默认聚焦
2. 扫码成功后自动查模板
3. 若模板列表为空，状态区提示“未找到对应模板”
4. 若模板列表非空，沿用当前 `set_templates(...)` 自动选中第一条
5. 操作员可手动切换模板
6. 操作员确认模板后，再进行拍照或选图
7. 点击“开始检测”后，新的扫码输入不应打断正在运行的检测

可选增强项：

- 若同编码只匹配 1 个模板，提示“已自动选中模板”
- 若同编码匹配多个模板，提示“请确认模板版本”
- 在界面上展示当前扫码器状态，例如“键盘模式 / 已连接 / 未连接”

## 11. 风险点与应对

### 11.1 焦点丢失

这是键盘模式最常见的问题。

表现：

- 扫码结果进入了错误控件
- 扫码后没有自动触发查询

应对：

- 窗口启动后聚焦编码框
- 查询完成后恢复焦点
- 检测完成后恢复焦点
- 检测中禁用编码框，避免扫码输入落入 UI

### 11.2 扫码器带前后缀

表现：

- UI 中能看到字符串，但查不到模板

应对：

- 先增强 `normalize()`
- 若确认设备输出带包装文本，再增加主数字串提取策略
- 不在没有设备样本前过度写死规则

### 11.3 真设备协议不明确

表现：

- 过早进入 SDK 开发，造成返工

应对：

- 先验证是否支持 HID 键盘模式
- 只有 HID 不可行时，再进入协议适配

### 11.4 检测期间被新扫码覆盖

表现：

- 任务尚未结束，编码框被新扫码内容覆盖

应对：

- 忙碌态忽略扫码
- 忙碌态禁用编码框
- 状态区提示扫码已被忽略

## 12. 测试与验收

### 12.1 单元测试建议

建议新增以下测试用例：

- `KeyboardWedgeScannerInput.normalize()` 能正确清理 `\r`、`\n`、`\t`
- `handle_scanned_code(...)` 会写回编码框并触发模板查询
- 查询后模板列表默认选中第一项
- 检测忙碌时扫码输入被忽略
- 专用 adapter 的 `code_scanned` 信号能够驱动控制器主线

### 12.2 手工验收场景

至少覆盖以下场景：

1. 输入 `600004075219` 后按回车，模板列表自动出现候选
2. 点击“模拟扫码”，行为与回车一致
3. 连续扫描多个编码时，编码框始终能收到数据
4. 模板选中后，`MockCameraAdapter` 仍能按当前编码选取样本图
5. 点击“开始检测”后，扫码输入不会打断当前检测
6. 检测完成后，焦点重新回到编码框

### 12.3 完成标准

本轮完成后，应至少满足：

1. 扫码结果可以稳定进入当前桌面端流程
2. 扫码完成后自动查询模板
3. 模板确认、图片选择、开始检测的主线不变
4. 无真实设备时仍可通过“回车 + 模拟扫码”验证链路
5. 后续接入 `MV-ORH` 时，只需要补设备适配层，不需要重写控制器主线

## 13. 建议实施顺序

建议按以下顺序开发：

1. 在 `AppController` 中增加统一扫码入口
2. 调整 `MainWindow` 信号，将“查询模板”和“模拟扫码”区分开
3. 增加编码框焦点恢复和忙碌态保护
4. 完成 USB HID 键盘模式实测
5. 只有在 HID 不可用时，再新增 `ScannerAdapter` 与 `MVORHScannerAdapter`

## 14. 本轮不建议同步做的内容

为控制风险，本轮不建议同时做：

- 自动触发检测
- 真实相机接入
- 模板库持久化改造
- 扫码器复杂配置界面
- 在未确认设备模式前直接写厂商 SDK 集成

## 15. 实施后的推荐验证命令

无真实扫码设备时，可直接在当前仓库执行：

```bash
uv sync --extra desktop
uv run python -m desktop_app.main
```

验证步骤：

1. 在编码框输入 `600004075219`
2. 按回车，确认模板列表自动刷新
3. 点击“模拟扫码”，确认走同一套模板查询链路
4. 点击“Mock 相机取图”，确认能优先取到同编码样本图
5. 点击“开始检测”，确认检测完成后结果区域正常更新

如现场已有真实扫码枪，再补做以下验证：

1. 将设备切到 HID 键盘模式
2. 打开桌面端并确认编码框聚焦
3. 连续扫码多个编码
4. 确认每次都自动查模板，且检测运行中不会被新扫码打断
