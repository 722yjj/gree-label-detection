# Agent Change Log

用于记录由 code agent 对项目做出的关键修改，降低黑盒改动带来的追溯成本。

建议后续每次涉及以下类型修改时，都追加一条记录：
- 预处理、匹配、OCR、VLM、工作流等核心逻辑变更
- 接口行为变化
- 删除旧方法、替换算法、调整回退链路
- 新增重要约束、假设、调试输出、验证结论

## 记录模板

### YYYY-MM-DD - 标题

- 背景：
- 修改文件：
- 修改内容：
- 修改原因：
- 影响范围：
- 验证情况：
- 风险 / 待验证项：

---

## 2026-03-10 - 新增独立 VLM 目标检测测试脚本

- 背景：
  当前项目中的 `label_detection/services/vlm_service.py` 主要面向“区域对比”场景，适合比较模板局部图和实拍局部图是否一致，但不适合直接拿单张图测试 VLM 的目标检测能力。为便于单独验证 `qwen3-vl:8b` 在标签图上的框选能力，需要补一个独立的测试入口。

- 修改文件：
  - `label_detection/services/vlm_detection.py`
  - `scripts/run_vlm_object_detection.py`
  - `tests/test_vlm_detection.py`
  - `AGENT_CHANGELOG.md`

- 修改内容：
  - 新增 `label_detection/services/vlm_detection.py`：
    - 封装单图 VLM 目标检测调用。
    - 复用项目现有 Ollama 配置。
    - 约束模型输出固定 JSON，并统一解析为像素坐标和 `bbox_1000`。
    - 兼容 `bbox_1000`、像素坐标、比例坐标等常见输出格式。
  - 新增 CLI 脚本 `scripts/run_vlm_object_detection.py`：
    - 支持传入图片路径和目标描述进行单图检测。
    - 自动保存标注图、JSON 结果，以及可选的原始模型响应。
    - 支持自定义模型、API 地址、超时、最大目标数和可视化置信度阈值。
  - 新增解析单测 `tests/test_vlm_detection.py`，覆盖：
    - `bbox_1000` 输出解析
    - Markdown 代码块 JSON 解析
    - 比例坐标转换
    - 非 JSON 输出失败兜底

- 修改原因：
  - 给项目补一个“单独测 VLM 检测能力”的最小可用入口，避免每次都借整条 unified 流程间接验证。
  - 把检测脚本和解析逻辑拆开，既方便手动调试，也方便后续继续演进成正式能力。

- 影响范围：
  - 不影响现有 unified 检测主流程。
  - 新增一个独立脚本入口，可用于人工验证 VLM 的框选能力和输出稳定性。

- 验证情况：
  - 计划通过 `tests/test_vlm_detection.py` 做解析单测。
  - 计划通过 `python scripts/run_vlm_object_detection.py --help` 验证 CLI 可运行。
  - 尚未完成真实 Ollama 推理验证，仍需在本地模型服务已启动的环境下实测。

- 风险 / 待验证项：
  - VLM 的目标检测本质仍是提示词驱动输出，框位置和格式稳定性不如专用检测模型。
  - 当前脚本主要用于能力探索和人工验证，不应直接替代正式检测算法。
  - 真实效果仍取决于模型版本、提示词和图像内容复杂度。

---

## 2026-03-10 - 新增直跑式 VLM 布局区域对比脚本

- 背景：
  用户的真实目标不是做通用单图目标检测 CLI，而是验证“VLM 能否完成当前项目里布局检测模型负责的图形区域检测工作”。因此需要一个把参数、图片路径、提示词都写在代码顶部的直跑脚本，并且最好直接对比 `PP-DocLayoutV3` 的 `image` 区域结果。

- 修改文件：
  - `scripts/test_vlm_layout_regions.py`
  - `label_detection/services/vlm_detection.py`
  - `label_detection/preprocessing/pipeline.py`
  - `tests/test_vlm_detection.py`
  - `AGENT_CHANGELOG.md`

- 修改内容：
  - 新增直跑脚本 `scripts/test_vlm_layout_regions.py`：
    - 在文件顶部直接配置图片路径、PDF 路径、提示词、IoU 阈值、输出目录等参数。
    - 复用项目现有模板提取、预处理、布局检测能力。
    - 默认对模板图和实拍图分别运行 `PP-DocLayoutV3` 与 VLM 区域检测。
    - 输出模板图/实拍图各自的布局模型框、VLM 框、叠加图、原始响应和汇总 JSON。
  - `label_detection/services/vlm_detection.py` 新增对 `regions` 键的兼容，便于 VLM 用更贴近“区域检测”语义的 JSON 输出。
  - 修复 `label_detection/preprocessing/pipeline.py` 中缺失 `numpy` 导入的问题，避免目标图预处理 fallback 触发时报 `NameError`。
  - 在 `tests/test_vlm_detection.py` 中补充 `regions` 键解析测试。

- 修改原因：
  - 让测试脚本直接对齐当前项目的核心问题：VLM 是否有机会替代布局模型做图形区域检测。
  - 降低运行门槛，避免每次手动传命令行参数。
  - 让测试输出既能肉眼看，也能通过 IoU 和计数做粗粒度比较。

- 影响范围：
  - 不改变 unified 主流程行为。
  - 新增一个偏实验性质的评估脚本，主要用于能力验证和人工分析。
  - 预处理模块因补充 `numpy` 导入而更稳定。

- 验证情况：
  - 已完成帮助信息/源码编译级检查。
  - 已完成 VLM 区域 JSON 解析的内联断言验证。
  - 尚未在当前命令环境完成真实图像 + 真实 Ollama 模型的端到端运行。

- 风险 / 待验证项：
  - VLM 对“版面图形区域”的理解可能和 `PP-DocLayoutV3` 的 `image` 标签定义不完全一致，结果存在语义偏差。
  - IoU 对比只适合做粗比较，不代表真正可替代性结论。
  - 如果默认 Python 环境仍缺 OpenCV / NumPy / PaddleX，对端到端运行仍有依赖门槛。

---

## 2026-03-08 - 标签透视预处理链路清理与候选框筛选增强

- 背景：
  当前 `label_detection/preprocessing/perspective.py` 中原有“方法2”基于浅色区域整体分割，容易把整张图、外层底纸、背景反光或胶带区域一起纳入候选框。对于带外层黄底纸、复杂背景或反光的实拍图，算法容易错误地按整图最大轮廓做透视矫正，导致取不到真正的标签主体内容。

- 修改文件：
  - `label_detection/preprocessing/perspective.py`
  - `label_detection/preprocessing/pipeline.py`

- 修改内容：
  - 在 `label_detection/preprocessing/perspective.py` 中删除原“方法2: 浅色区域检测”。
  - 保留方法1和方法3，但不再使用“直接取最大轮廓”的方式。
  - 新增候选四边形筛选逻辑：
    - 将轮廓优先近似为四边形，失败时退化为最小外接旋转矩形。
    - 使用模板图片的长宽比作为参考约束。
    - 综合候选区域面积占比、长宽比、轮廓填充度、是否贴边等因素打分。
    - 在多个候选框中选择最像标签主体的区域，而不是直接取整图上的最大轮廓。
  - 方法1调整为“深色区域聚合”：
    - 通过阈值化和形态学操作将文字、黑框、条码等深色前景聚合成标签主体。
  - 方法3保留为边缘检测：
    - 继续从边缘轮廓中提取候选四边形参与评分。
  - 透视矫正输出尺寸会参考模板长宽比做约束，减少矫正结果比例失真。
  - 在 `label_detection/preprocessing/pipeline.py` 中同步删除 fallback 里的“方法2: 颜色分割（浅色标签检测）”。
  - 将目标图预处理链路统一为：
    1. 先尝试 `detect_and_correct_perspective`
    2. 若失败，则回退到黑框检测
    3. 若仍失败，则回退到边缘检测
  - 新增调试输出：
    - `target_method1_mask.jpg`
    - `target_method3_edges.jpg`

- 修改原因：
  - 让预处理链路与当前样例问题保持一致：重点识别标签主体，而不是识别整张图中的浅色大块区域。
  - 避免“浅色区域检测”把黄底纸、白底背景、反光区等误识别为标签。
  - 提高不同拍摄背景下提取真实标签内容的稳定性。
  - 让 `perspective.py` 和 `pipeline.py` 的方法定义、回退逻辑、日志口径保持一致，减少后续维护歧义。

- 影响范围：
  - 影响目标图预处理阶段，主要是透视矫正与裁剪区域的确定方式。
  - 对模板预处理无直接影响。
  - 下游 OCR、结构化提取、图形区域比对会间接受益，因为输入图更接近真实标签主体。

- 验证情况：
  - 已完成语法级校验：
    - `label_detection/preprocessing/perspective.py`
    - `label_detection/preprocessing/pipeline.py`
  - 已确认项目内不再残留“方法2 / 颜色分割 / 浅色标签检测”相关链路文案。
  - 未完成基于 OpenCV 的样图运行验证；当前命令环境中的默认 `python` 缺少 `cv2`，因此本次只完成静态检查，没有完成实图输出验证。

- 风险 / 待验证项：
  - `pipeline.py` 的 fallback 目前仍是“黑框检测 / 边缘检测 + 最大轮廓裁剪”，虽然已去掉方法2，但回退策略本身仍弱于 `perspective.py` 中的候选框评分逻辑。
  - 对极端场景仍需样图验证：
    - 标签外还有大面积黑色背景
    - 标签黑框断裂严重
    - 条码或大字区域比边框更显著
    - 反光导致边缘缺失
  - 建议后续在可用运行环境中至少补一轮样图验证，并保留成功/失败样例。

- 建议后续记录要求：
  - 每次 agent 修改核心流程后，至少记录“修改原因、修改文件、行为变化、验证结论、未验证项”。
  - 如果是替换算法，不要只写“优化了逻辑”，要明确写出“旧方法是什么，新方法是什么，为什么替换”。
  - 如果没有运行验证，要明确写清原因，不要默认视为已验证。

---

## 2026-03-08 - 根入口改名与项目文档整理

- 背景：
  项目根目录原入口文件名为 `unified_detection.py`，名称偏向历史实现细节，不利于作为长期稳定入口使用。与此同时，`TODO.md` 中包含阶段性说明，已不再适合作为正式项目文档保留。

- 修改文件：
  - `main.py`
  - `README.md`
  - `AGENT_CHANGELOG.md`
  - `label_detection/matching/layout.py`
  - 删除 `unified_detection.py`
  - 删除 `TODO.md`

- 修改内容：
  - 新增根入口 `main.py`，保持原有 CLI 行为不变，继续调用 `label_detection.workflows.unified.main`。
  - 删除旧入口 `unified_detection.py`。
  - 删除 `TODO.md`。
  - 更新 `README.md`，同步当前真实项目状态：
    - 根入口改为 `main.py`
    - 目录结构加入 `AGENT_CHANGELOG.md`
    - 示例运行命令改为新的样例路径
    - 补充当前预处理策略说明
    - 补充变更追溯说明
    - 更新输出结果和当前风险描述
  - 将 `label_detection/matching/layout.py` 中提到旧入口文件名的说明改为 `main.py`。

- 修改原因：
  - 让项目对外入口名称更稳定、直观。
  - 移除已经失效的阶段性文档，避免误导后续维护者。
  - 让 README 成为和当前仓库状态一致的正式说明文档。
  - 为多 agent 协作保留更清晰的入口、说明和追溯链路。

- 影响范围：
  - 根目录执行命令从 `python unified_detection.py ...` 变为 `python main.py ...`。
  - 所有依赖 README 或源码注释理解项目入口的人员，需要以 `main.py` 为准。

- 验证情况：
  - 已检查并更新仓库内对旧入口名的 README / 注释引用。
  - 未执行端到端运行验证。

- 风险 / 待验证项：
  - 外部脚本、快捷方式或人工操作习惯如果仍引用 `unified_detection.py`，需要同步切换到 `main.py`。
  - 本次未验证运行环境是否能直接执行完整流程。
