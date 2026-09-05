# 桌面端结果输出收敛方案

## 1. 背景

当前桌面端每次检测都会在 `results/desktop_app/{code}/{timestamp}_{variant}` 下输出大量中间文件。  
这些文件在排查“哪一步出问题”时有价值，但在实际运行阶段会带来三个问题：

1. 结果目录噪声很大，最终结果不够突出。
2. 历史记录打开目录后，很难第一眼找到真正要看的文件。
3. 长期运行会累积大量调试图、Excel、候选裁剪图，目录体积增长很快。

## 2. 当前落盘点

桌面端本身只负责创建任务目录，真正的大量输出来自统一检测工作流和下游模块：

- `desktop_app/services/detection_service.py`
  - 为每次检测创建独立输出目录，并调用 `run_unified_detection(...)`
- `label_detection/workflows/unified.py`
  - 当前默认把模板解析、预处理、文字对比、区域对比、可视化过程中的中间产物都直接写到结果目录
- `label_detection/preprocessing/border.py`
  - 写 `template_thresh.jpg`、`template_content_mask.jpg`、`template_crop_candidates.txt`
- `label_detection/preprocessing/perspective.py`
  - 写 `target_method1_mask.jpg`、`target_method3_edges.jpg`、`target_corners_detected.jpg`、`target_perspective_corrected.jpg`
- `label_detection/preprocessing/pipeline.py`
  - 写 `target_thresh.jpg`、`target_edges.jpg`、`target_cropped.jpg`
- `label_detection/matching/layout.py` 和 `label_detection/services/vlm_service.py`
  - 写区域裁剪图、轮廓图、VLM 对比图、`graphic_comparison/`

## 3. 目标

桌面端默认只保留最终交付结果：

- `final_result.json`
- `visualization_diff.jpg`

当需要排查后端流程、实验算法或比对中间步骤时，再通过开关启用详细输出。

## 4. 推荐方案

### 4.1 增加统一输出模式

新增统一输出模式，建议名称：

- `final`
  - 默认模式
  - 只保留最终结果
- `debug`
  - 调试模式
  - 保留当前详细输出

建议不要只做一个零散布尔值，而是增加一个统一配置对象，例如：

```python
@dataclass(frozen=True)
class WorkflowOutputOptions:
    mode: Literal["final", "debug"] = "final"
    save_template_assets: bool = False
    save_preprocess_images: bool = False
    save_text_excel: bool = False
    save_graphic_debug: bool = False
    save_region_crops: bool = False
    save_vlm_debug: bool = False
```

这样后面即使再细分“只开预处理调试”也不需要重构接口。

### 4.2 桌面端增加开关，默认关闭

修改建议：

- `desktop_app/models.py`
  - `DetectionJobRequest` 增加 `output_mode: str = "final"`
- `desktop_app/ui/main_window.py`
  - 增加一个复选框，例如“保存详细调试结果”
  - 默认不勾选
- `desktop_app/controllers/app_controller.py`
  - 发起检测时把 UI 选择传给 `DetectionJobRequest`
- `desktop_app/services/detection_service.py`
  - 调用 `run_unified_detection(..., output_mode=request.output_mode)`  

这样桌面端实际运行默认就是最终结果模式，只有做实验时才打开详细输出。

### 4.3 工作流层统一接收 `output_mode`

建议改造 `label_detection/workflows/unified.py`：

- `run_unified_detection(...)` 增加 `output_mode: str = "debug"` 或 `output_options`
- 保持底层脚本默认兼容现状
  - 低层工作流默认仍可保持 `debug`
  - 桌面端显式传 `final`

这样不会一下子破坏现有测试脚本和人工排查习惯。

## 5. 输出目录结构建议

### 5.1 `final` 模式

结果目录只保留：

```text
results/desktop_app/{code}/{timestamp}_{variant}/
  final_result.json
  visualization_diff.jpg
```

### 5.2 `debug` 模式

最终结果仍放根目录，中间产物统一收进 `debug/`，避免和最终结果混在一起：

```text
results/desktop_app/{code}/{timestamp}_{variant}/
  final_result.json
  visualization_diff.jpg
  debug/
    template_assets/
    preprocess/
    text_comparison.xlsx
    graphic_comparison/
    ...
```

这个结构比“所有文件平铺在根目录”更清晰，也方便用户只同步最终文件。

## 6. 各类文件保留策略

建议按下面的规则收敛：

| 文件/目录 | `final` | `debug` |
| --- | --- | --- |
| `final_result.json` | 保留 | 保留 |
| `visualization_diff.jpg` | 保留 | 保留 |
| `template_assets/` | 不保留 | 保留 |
| `text_comparison.xlsx` | 不保留 | 保留 |
| `target_preprocessed.jpg` | 不保留 | 保留 |
| `target_corners_detected.jpg` | 不保留 | 保留 |
| `target_method1_mask.jpg` / `target_method3_edges.jpg` | 不保留 | 保留 |
| `template_thresh.jpg` / `template_content_mask.jpg` / `template_crop_candidates.txt` | 不保留 | 保留 |
| `graphic_comparison/` | 不保留 | 保留 |
| VLM 区域裁剪图/轮廓图 | 不保留 | 保留 |

## 7. 实现建议

关键点不是在最后“删文件”，而是在写文件之前就受控：

1. 新增一个统一的 `should_save_*` 判断层。
2. 底层函数不要再用“只要传了 `output_dir` 就默认写调试图”的方式。
3. `output_dir` 只表示结果根目录，是否写中间产物由 `output_mode` 决定。
4. 调试文件统一写到 `debug/` 子目录，最终文件留在根目录。

原因：

- 后删文件会多做无用 I/O，速度和磁盘占用都没有改善。
- 某些调试目录很深，后删逻辑容易漏。
- “写前控制”更容易测试，也更容易维护。

## 8. 建议改动顺序

### 第一阶段

先把桌面端跑通最终结果模式：

1. `DetectionJobRequest` 增加 `output_mode`
2. UI 增加“详细输出”开关
3. `DetectionService` 把 `output_mode` 传入工作流
4. `run_unified_detection()` 先把根目录输出控制住

### 第二阶段

把底层模块调试输出全部归拢到 `debug/`：

1. `border.py`
2. `perspective.py`
3. `pipeline.py`
4. `layout.py`
5. `vlm_service.py`

### 第三阶段

顺手把 batch 场景也统一：

- `label_detection/batch_samples.py` 也显式走“最终结果模式”
- 与 `AGENTS.md` 中“batch 只保留最终产物”的要求保持一致

## 9. 测试建议

至少补三类测试：

1. `final` 模式下目录只包含 `final_result.json` 和 `visualization_diff.jpg`
2. `debug` 模式下仍保留现有详细产物
3. 桌面端 UI 开关状态能正确透传到 `DetectionService`

建议新增或补强的测试位置：

- `tests/test_detection_service_summary.py`
- 新增 `tests/test_detection_service_output_mode.py`
- 如需 UI 透传校验，可补 `tests/test_desktop_scanner_controller.py`

## 10. 结论

推荐采用“默认 final，按需 debug”的双模式方案。  
对桌面端用户来说，结果目录会立即变干净；对开发调试来说，现有中间产物能力不丢，只是改成显式开启。  
这比单纯在桌面端跑完后删文件更稳，因为它从工作流入口就把输出职责分清了。
