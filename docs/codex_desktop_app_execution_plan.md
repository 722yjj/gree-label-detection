# PySide6 桌面端重构执行方案（以参考图为目标）

> 适用范围：当前分支已经存在 `desktop_app/` 实现，本方案用于指导**增量重构**，不是从零搭建第一版。
>
> 目标：在不重写检测主流程的前提下，把当前桌面端重构为接近参考图的信息架构、交互节奏和结果呈现方式。
>
> 参考目标：本次界面以用户提供的参考图为主要目标，优先对齐**布局层次、信息密度、操作顺序**，不要求逐像素复刻。

---

## 0. 已确认的当前基线

以下内容已经在仓库中存在，应作为重构基础保留：

1. 桌面端入口已存在：`uv run python -m desktop_app.main`
2. 当前窗口和控制器已存在：
   - `desktop_app/ui/main_window.py`
   - `desktop_app/controllers/app_controller.py`
3. 模板查询链路已存在：
   - `desktop_app/repositories/template_repository.py`
4. 检测调用链路已存在：
   - `desktop_app/services/detection_service.py`
   - `desktop_app/workers/detection_worker.py`
5. 模拟相机和扫码输入已存在：
   - `desktop_app/devices/camera/mock_camera.py`
   - `desktop_app/devices/scanner_input.py`
   - `desktop_app/devices/scanner/*`
6. 核心检测入口已确认：

```python
label_detection.workflows.unified.run_unified_detection(
    template_input_path,
    target_image_path,
    output_dir,
)
```

### 本次重构必须保留的事实

1. 不重写 `run_unified_detection(...)` 的业务逻辑。
2. 检测仍然只能通过 service 层调用主流程。
3. 检测必须继续走后台线程，不能回退到 UI 主线程执行。
4. 模板输入仍然支持 PDF 和图片两种来源。
5. 当前输出目录结构 `results/desktop_app/{code}/{timestamp}_{variant}` 可以保留。

---

## 1. 当前实现与目标界面的主要差距

当前代码已经能运行，但距离参考图仍有几个关键差距：

1. 当前顶部区域还是“编码 + 目标图”的普通表单，不是参考图里的**全局操作条**。
2. 当前右侧预览区是三图并排：模板图、目标图、结果图；而参考图的重点是：
   - 左侧准备区放模板参考图
   - 右侧主工作区只保留目标图和检测结果图两张主图
3. 当前结果区还是普通 `QFormLayout + QPlainTextEdit`，缺少参考图中的**结果卡片/状态色块**。
4. 当前历史记录只是内存中的字符串列表，没有本地持久化，也没有“打开目录/回看结果”动作。
5. 当前“开始检测”按钮没有按“模板已选 + 目标图已就绪”做严格启用控制。
6. 当前模板预览解析逻辑放在 controller 中，后续继续扩展会让控制器变重。
7. 当前模板候选区是简单 `QListWidget`，与参考图中更结构化的记录区还有差距。

---

## 2. 本次重构的目标界面定义

### 2.1 设计原则

1. 真正的业务输入仍然只有两个：
   - 编码
   - 目标图
2. 模板不是第三输入，而是“查询结果 + 当前选择结果”。
3. 参考图顶部出现的长路径框应被视为**只读信息条**，不是可编辑输入框。
4. 右侧主工作区只放最关键的结果内容：
   - 目标图
   - 检测结果图
   - 综合判定卡片
   - 历史记录
5. 模板参考图留在左侧准备区，不再挤占右侧主结果区域。

### 2.2 目标布局文字图

```text
顶部全局操作条
  - 当前目标图路径（只读展示条）
  - 选择图片
  - 模拟扫码
  - 开始检测（主按钮）

主区域 = 左右分栏

左侧固定宽度（准备区）
  Step 1 编码输入
    - 编码输入框
    - 查询模板按钮

  Step 2 模板确认
    - 当前模板路径（只读）
    - 打开模板按钮
    - 复制路径按钮

  模板参考预览
    - 当前模板大图预览

  模板候选 / 当前匹配记录区
    - 结构化列表或表格

右侧自适应（主工作区）
  上半区：双图预览
    - 左：目标图预览
    - 右：检测结果图预览

  中间：检测结果卡片
    - verdict 色块
    - 摘要说明
    - 文字匹配统计
    - 图形匹配统计
    - 输出目录
    - 打开目录 / 复制路径

  下半区：历史记录
    - 最近记录列表
    - 每条支持至少“打开目录”
    - 如实现成本可控，再支持“回看结果”
```

### 2.3 与参考图对齐时的取舍

1. 允许文字、按钮顺序、图标与参考图略有差异。
2. 优先保证操作链路和信息层次接近参考图，而不是像素级复刻。
3. 若某些动作在第一轮实现里过重，可先保留按钮占位，但按钮文案和区域位置要先定下来。

---

## 3. 明确的重构边界

### 允许做的事

1. 重排 `MainWindow` 布局。
2. 新增 `HistoryRepository`。
3. 新增 `PreviewService`，把模板预览解析从 controller 中抽离出去。
4. 扩展现有 dataclass 模型，让结果卡片可直接消费统计字段。
5. 为历史记录和输出目录增加按钮动作。
6. 为 `MainWindow` 增加更清晰的 UI 状态切换方法。

### 不允许做的事

1. 不要重写 `label_detection/workflows/unified.py` 的主流程判定逻辑。
2. 不要为了对齐参考图而改成多窗口架构。
3. 不要引入 Web 服务或浏览器壳。
4. 不要新增重型 UI 框架。
5. 不要为了“看起来像设计稿”而牺牲当前可运行链路。

---

## 4. 推荐的目录和文件策略

### 4.1 已存在文件，优先增量修改

```text
desktop_app/
  app.py
  main.py
  models.py
  controllers/app_controller.py
  repositories/template_repository.py
  services/detection_service.py
  workers/detection_worker.py
  ui/main_window.py
```

### 4.2 本轮建议新增文件

```text
desktop_app/
  repositories/history_repository.py
  services/preview_service.py
```

### 4.3 可选新增文件

如果 `main_window.py` 在重排后过大，再考虑拆出：

```text
desktop_app/ui/widgets/
  image_panel.py
  result_card.py
  history_item_widget.py
```

注意：这三个 widget 不是本轮强制项。若直接在 `MainWindow` 中完成第一轮重构更稳，可以先不拆。

---

## 5. 数据模型调整建议

当前项目里已经有：

- `TemplateRecord`
- `DetectionJobRequest`
- `DetectionJobResult`

本轮不要强行推翻命名，建议**在现有模型上扩展字段**。

### 5.1 `TemplateRecord`

保留现有结构即可，只需要继续支持：

1. `code`
2. `variant`
3. `display_name`
4. `source_type`
5. `source_path`
6. `preview_path`
7. `is_default`

### 5.2 `DetectionJobResult`

建议在当前 dataclass 上补充以下字段，方便结果卡片直接显示：

```python
text_match_count: int = 0
text_total_count: int = 0
graphic_match_count: int = 0
graphic_mismatch_count: int = 0
graphic_review_count: int = 0
target_image_path: Optional[Path] = None
template_path: Optional[Path] = None
```

关键约束：

1. `verdict` 继续直接复用 workflow 返回值。
2. 统计字段可以由 `DetectionService` 从结果 dict 中补齐。
3. 不要在桌面端重新发明一套与 workflow 相冲突的 verdict 规则。

### 5.3 `HistoryRecord`

新增 dataclass，建议字段如下：

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class HistoryRecord:
    created_at: str
    code: str
    template_name: str
    verdict: str
    output_dir: Path
    target_image_path: Path
    visualization_path: Path | None = None
    summary_text: str = ""
```

说明：

1. 第一版直接存 ISO 字符串时间即可，不必先上 `datetime` 反序列化复杂度。
2. `output_dir` 和图片路径在仓库当前语境下都适合保存为绝对路径。

---

## 6. Repository 层重构要求

### 6.1 `TemplateRepository`

当前实现方向基本正确，应继续保留：

1. 默认扫描 `samples/pdfs`
2. 同时支持扫描 `samples/images/original`
3. 通过文件名抽取主编码
4. 识别 `-01`、`_02` 等 variant

本轮只需要补强以下行为：

1. 给 UI 提供更适合展示的记录排序和展示文本。
2. 如有需要，增加 `to_display_rows()` 一类的轻量辅助方法，但不要把 UI 拼装逻辑塞进 repository。

### 6.2 `HistoryRepository`

新增文件：`desktop_app/repositories/history_repository.py`

第一版使用 JSON 文件即可，推荐路径：

```text
results/desktop_app/history.json
```

推荐接口：

```python
class HistoryRepository:
    def __init__(self, history_file: Path):
        ...

    def list_recent(self, limit: int = 20) -> list[HistoryRecord]:
        ...

    def append(self, record: HistoryRecord) -> None:
        ...
```

容错要求：

1. 文件不存在时返回空列表。
2. JSON 损坏时不崩溃，返回空列表并保留错误日志。
3. 新增记录时应自动创建父目录。

---

## 7. Service 层重构要求

### 7.1 `PreviewService`

新增文件：`desktop_app/services/preview_service.py`

职责：

1. 接收 `TemplateRecord`
2. 返回可供 UI 展示的预览图路径
3. 统一管理模板预览缓存目录

推荐策略：

1. 模板本身是图片：直接返回原图。
2. 模板是 PDF：
   - 优先复用 `label_detection.extraction.template_source.resolve_template_input(...)`
   - 缓存到 `results/desktop_app/template_preview_cache/{template_stem}/`
3. 失败时返回 `None`，由 UI 展示“暂无预览”

注意：

1. 当前 controller 里已有预览解析逻辑，本轮应迁移到 `PreviewService`。
2. controller 只负责调用 service，不再直接管理缓存目录细节。

### 7.2 `DetectionService`

当前 service 已经可用，本轮重点是**补结构化摘要字段**，而不是推翻重写。

保留：

1. 组织输出目录
2. 调用 `run_unified_detection(...)`
3. 读取 `visualization_diff.jpg`
4. 返回 `DetectionJobResult`

补强：

1. 从 workflow 返回结果中提取：
   - 文字匹配数
   - 文字总字段数
   - 图形匹配数
   - 图形不一致数
   - 图形待复核数
2. 保持 `summary_text` 面向操作员可读。
3. 把 `template_path`、`target_image_path` 一并写回结果对象，供历史记录使用。

关键原则：

1. `verdict` 直接信任 workflow 结果。
2. 结果摘要是补充展示，不是重新判案。

---

## 8. Controller 层重构要求

文件：`desktop_app/controllers/app_controller.py`

当前 controller 已负责主流程编排，本轮重构重点是“减重 + 状态显式化”。

### 8.1 建议职责

1. 响应编码查询
2. 响应模板选择
3. 响应目标图选择 / mock 相机取图
4. 维护当前 UI 状态
5. 启动 worker
6. 处理检测完成 / 失败
7. 调用 `HistoryRepository` 持久化历史记录
8. 调用 `PreviewService` 获取模板预览

### 8.2 必须新增的控制逻辑

1. 程序启动时加载最近历史记录。
2. 每次检测成功后写入历史记录。
3. 模板和目标图都就绪前，禁止点击“开始检测”。
4. 当目标图切换时，清空旧结果图和旧结果卡片。
5. 当模板切换时，刷新模板路径和模板预览。

### 8.3 当前 controller 里适合迁出的逻辑

以下逻辑建议从 controller 抽出：

1. 模板预览缓存路径解析 -> `PreviewService`
2. 历史记录读写 -> `HistoryRepository`

---

## 9. UI 层重构要求

文件：`desktop_app/ui/main_window.py`

本轮重构的主战场是 `MainWindow`。

### 9.1 目标区域划分

#### 顶部全局操作条

包含：

1. 当前目标图路径展示条
2. 选择图片按钮
3. 模拟扫码按钮
4. 开始检测按钮

说明：

1. 顶部路径条是**只读展示控件**。
2. “开始检测”必须是视觉主按钮。

#### 左侧准备区

包含：

1. Step 1 编码输入区
2. Step 2 当前模板信息区
3. 模板参考预览区
4. 模板候选 / 当前匹配记录区

#### 右侧主工作区

包含：

1. 目标图预览
2. 检测结果图预览
3. 结果卡片
4. 历史记录区

### 9.2 结果卡片显示规则

结果卡片至少显示：

1. 综合判定
2. 摘要说明
3. 文字匹配统计
4. 图形匹配统计
5. 输出目录
6. 打开目录按钮
7. 复制路径按钮

### 9.3 历史记录显示规则

每条历史记录至少显示：

1. 时间
2. 编码
3. 模板名
4. verdict
5. 输出目录

每条记录的动作最低要求：

1. 打开目录

可选动作：

1. 回看结果
2. 复制路径

### 9.4 目标状态机

#### 状态 A：初始态

1. 无模板
2. 无目标图
3. 开始检测按钮禁用
4. 结果卡片显示“待检测”

#### 状态 B：已输入编码并查到模板

1. 模板候选列表已刷新
2. 当前模板路径已显示
3. 若还没目标图，开始检测仍禁用

#### 状态 C：模板和目标图已就绪

1. 开始检测按钮启用
2. 结果图仍为空
3. 结果卡片保持待检测或显示上次结果已清空

#### 状态 D：检测中

1. 开始检测按钮禁用
2. 顶部状态/结果卡片显示“检测中”
3. 避免重复触发

#### 状态 E：检测完成

1. 刷新结果图
2. 刷新结果卡片
3. 追加历史记录
4. 恢复按钮状态

#### 状态 F：检测失败

1. 弹出错误提示
2. 结果卡片显示失败
3. 恢复按钮状态

---

## 10. 分阶段执行顺序

### Phase 1：把现有窗口改成参考图的信息架构

目标：

1. 不动主流程调用
2. 先把布局层次改对

任务：

1. 重构 `MainWindow` 顶部为全局操作条
2. 把模板参考预览移到左侧
3. 右侧只保留目标图和结果图两张主图
4. 预留结果卡片和历史记录区域

验收：

1. 窗口结构接近参考图
2. 现有信号和基本动作仍然可触发

### Phase 2：重构模板准备区

目标：

1. 让编码查询、当前模板、模板预览的关系更清晰

任务：

1. 编码输入区只保留编码和查询动作
2. 当前模板路径做成只读信息区
3. 增加“打开模板 / 复制路径”按钮
4. 让模板候选或当前匹配记录区更结构化

验收：

1. 输入编码后模板列表正确刷新
2. 选中模板后路径与预览同步更新

### Phase 3：补 PreviewService 和历史持久化

目标：

1. 降低 controller 复杂度
2. 打通历史记录

任务：

1. 新增 `PreviewService`
2. 新增 `HistoryRepository`
3. controller 接入两者
4. 启动时加载历史记录

验收：

1. 模板预览逻辑已从 controller 主体移出
2. 重启程序后能看到最近历史

### Phase 4：补结果卡片和严格状态机

目标：

1. 让“开始检测”行为和结果区更接近参考图

任务：

1. 为 `DetectionJobResult` 增加统计字段
2. 结果卡片展示 verdict、摘要、统计、输出目录
3. 只有模板和目标图都就绪时才启用“开始检测”
4. 检测中、成功、失败三种状态可视化

验收：

1. 结果展示从“文本块”升级为“卡片”
2. 按钮启用条件符合预期

### Phase 5：补历史动作和细节收尾

目标：

1. 让桌面端接近可长期使用状态

任务：

1. 历史记录支持“打开目录”
2. 如成本可控，再支持“回看结果”
3. 增加复制路径动作
4. 清理明显重复 UI 更新代码

验收：

1. 操作员可快速回到输出目录
2. 常用路径不需要手动复制

---

## 11. 文件级任务清单

### `desktop_app/ui/main_window.py`

本轮重点文件。

职责：

1. 完成参考图导向的布局重排
2. 暴露清晰的 UI 更新方法
3. 管理按钮启用/禁用和只读展示控件

### `desktop_app/controllers/app_controller.py`

职责：

1. 串起 UI、template repository、preview service、history repository、detection service
2. 统一管理 UI 状态切换

### `desktop_app/models.py`

职责：

1. 扩展 `DetectionJobResult`
2. 新增 `HistoryRecord`

### `desktop_app/repositories/history_repository.py`

职责：

1. 历史记录 JSON 读写
2. 容错处理

### `desktop_app/services/preview_service.py`

职责：

1. 模板预览解析
2. 缓存路径管理

### `desktop_app/services/detection_service.py`

职责：

1. 保留主流程调用
2. 补结构化摘要字段

### `desktop_app/workers/detection_worker.py`

职责：

1. 保持后台线程执行
2. 保持 started / finished / failed 信号链路清晰

---

## 12. 测试与验证要求

### 自动化验证

现有测试：

```text
tests/test_desktop_scanner_controller.py
```

本轮建议补的测试：

1. `tests/test_history_repository.py`
2. `tests/test_detection_service_summary.py`
3. 如 PySide6 测试稳定，再补一条按钮状态机测试

### 手工验证路径

按以下顺序人工验证：

1. 启动桌面端
2. 输入编码并查询模板
3. 选择模板并确认模板预览
4. 选择目标图片
5. 检查“开始检测”按钮是否启用
6. 执行一次检测
7. 检查结果卡片、结果图、历史记录
8. 重启程序，确认历史记录仍在

---

## 13. 最终验收清单

### 布局

- [ ] 顶部已改为全局操作条
- [ ] 左侧是准备区，右侧是结果区
- [ ] 模板参考图已从右侧移回左侧
- [ ] 右侧只保留目标图和结果图两张主图

### 功能

- [ ] 输入编码后能查到模板
- [ ] 模板路径、模板预览可刷新
- [ ] 可选择目标图
- [ ] mock 相机仍可用
- [ ] 模板与目标图就绪前，开始检测按钮禁用
- [ ] 检测仍在后台线程执行
- [ ] 检测完成后结果卡片正确更新

### 历史记录

- [ ] 检测完成后会写入历史
- [ ] 重启后能加载最近历史
- [ ] 历史记录可打开输出目录

### 容错

- [ ] 编码为空时有提示
- [ ] 模板不存在时有提示
- [ ] 目标图不存在时有提示
- [ ] 检测失败时 UI 不崩溃
- [ ] 历史 JSON 损坏时程序不崩溃

---

## 14. 给 Codex 的直接执行指令

下面这段可以直接作为下一轮实现任务说明：

```text
请基于当前仓库中已经存在的 desktop_app 代码做增量重构，不要从零重建桌面端。

目标界面以参考图为准，重点对齐布局层次和操作顺序：
- 顶部全局操作条：当前目标图路径、选择图片、模拟扫码、开始检测
- 左侧准备区：编码输入、当前模板、模板参考预览、模板候选/记录区
- 右侧主工作区：目标图、结果图、结果卡片、历史记录

实现约束：
1. 不修改 label_detection/workflows/unified.py 的主流程逻辑
2. 继续通过 service 层调用 run_unified_detection(...)
3. 继续使用后台线程执行检测
4. 保留当前入口 python -m desktop_app.main
5. 在现有 models.py 上扩展 DetectionJobResult，不要强行换一套命名
6. 新增 HistoryRepository 和 PreviewService
7. 历史记录使用 JSON 持久化到 results/desktop_app/history.json
8. 开始检测按钮必须按“模板已选 + 目标图已就绪 + 当前不在运行中”控制启用状态
9. 结果区改成卡片式展示 verdict、摘要、统计和输出目录
10. 每次检测完成后写入历史，启动时加载最近历史

推荐修改顺序：
1. 先重构 MainWindow 布局
2. 再抽 PreviewService、接入 HistoryRepository
3. 再扩展 DetectionService 返回的结构化结果
4. 最后补历史动作、按钮状态机和测试
```

---

## 15. 本文件用途说明

本文件是“基于当前代码的桌面端重构执行方案”。

它不是“从零搭第一版”的脚手架说明，而是下一轮实现时应遵循的开发顺序、边界和验收标准。
