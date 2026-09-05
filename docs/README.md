# 文档索引

`docs/` 按用途分成八个子目录。根目录只保留 `project_research_summary.md`，因为它是
`CLAUDE.md` / `AGENTS.md` / `README.md` 共同引用的当前研究基线。

| 目录 | 放什么 | 读它的时机 |
|------|--------|-----------|
| [`research/`](research/) | 跨实验的进展报告、研究计划、模型方案 | 想知道研究主线走到哪一步 |
| [`experiments/`](experiments/) | 单轮实验的原始记录 | 要复现或追某一轮的数据、命令、指标 |
| [`datasets/`](datasets/) | 数据集说明与人工标注流程 | 要用或重建某个数据集 / GT |
| [`pipeline/`](pipeline/) | 主流程说明与关键开关 | 要读懂或改动生产检测流程 |
| [`plans/`](plans/) | 历史执行方案与开发计划 | 想知道某个模块当初为什么这么设计 |
| [`operations/`](operations/) | 部署、启动、现场运行手册、许可 | 要把系统跑起来或交付 |
| [`poc/`](poc/) | 技术选型 POC 追溯 | 要知道某条技术路线为什么被选中/放弃 |
| [`literature/`](literature/) | 论文 PDF 与综述 | 找相关工作 |

另有两个非文档目录：`report_assets/` 是各文档共享的图片（多个文档引用同一批图，不要按主题
拆分），`exports/` 是由 `scripts/export_docs_to_docx.py` 生成的 docx，不要手改。

## 当前基线

- [`project_research_summary.md`](project_research_summary.md) —— 标签细粒度差异检测项目阶段总结。
  跨实验的综述和研究主线。与旧笔记冲突时以它和实际代码为准。

## research/ —— 研究主线

- [`experiment_progress_report_20260901.md`](research/experiment_progress_report_20260901.md) ——
  实验进展报告（以 D06 为分界）。按研究脉络重组的进展报告，每节末尾指向对应的原始记录。
- [`clip_pairwise_label_verification_research_plan.md`](research/clip_pairwise_label_verification_research_plan.md) ——
  基于 CLIP 的成对标签差异确认：模型设计、训练与论文实验方案。兼作执行日志。
- [`pair_interaction_adapter_model_proposal.md`](research/pair_interaction_adapter_model_proposal.md) ——
  M11 后续 Pair Interaction Adapter 模型建议。
- [`clip_replace_vlm_stage0_exploration.md`](research/clip_replace_vlm_stage0_exploration.md) ——
  CLIP 替代 VLM keep/discard 的端到端实验方案。

## experiments/ —— 单轮实验记录

按时间顺序：

- [`clip_pair_experiment_recovery_20260713.md`](experiments/clip_pair_experiment_recovery_20260713.md) ——
  CLIP Pair 实验恢复快照（2026-07-13）。
- [`clip_pair_experiment_report_20260721.md`](experiments/clip_pair_experiment_report_20260721.md) ——
  实验汇报（2026-07-21）。导出件为 `exports/实验汇报_20260721.docx`。
- [`d06_pair_interaction_a0_a1_experiment.md`](experiments/d06_pair_interaction_a0_a1_experiment.md) ——
  D06 Pair Interaction A0/A1 首轮实验。
- [`d06_padding_ablation_p0_p4_experiment.md`](experiments/d06_padding_ablation_p0_p4_experiment.md) ——
  D06 输入白边与 Pair Adapter P0-P4 实验。
- [`d06_padding_ablation_real_development_evaluation.md`](experiments/d06_padding_ablation_real_development_evaluation.md) ——
  D06 P0/P4 真实开发集评估。
- [`d07_d08_symmetric_capture_augmentation_experiment.md`](experiments/d07_d08_symmetric_capture_augmentation_experiment.md) ——
  D07/D08 真实风格增强实验。
- [`c0_c3_shape_input_ablation_experiment.md`](experiments/c0_c3_shape_input_ablation_experiment.md) ——
  C0-C3 灰度与轮廓输入消融实验。
- [`superscript_synthetic_real_matched_audit.md`](experiments/superscript_synthetic_real_matched_audit.md) ——
  上标变化合成-实拍对应审计。
- [`d09_micro_preservation_experiment.md`](experiments/d09_micro_preservation_experiment.md) ——
  D09 微小字符保真数据与 P4 复现实验。
- [`d06_local_metric_a2_experiment.md`](experiments/d06_local_metric_a2_experiment.md) ——
  D06 局部度量损失 A2 实验。
- [`real50_photo_reference_p4_experiment.md`](experiments/real50_photo_reference_p4_experiment.md) ——
  Real50 正常实拍参考实验。
- [`qwen38_nvfp4_real_photo_evaluation_20260830.md`](experiments/qwen38_nvfp4_real_photo_evaluation_20260830.md) ——
  Qwen3.8-27B-NVFP4 真实图片评测。当前配置以文末章节为准。

## datasets/ —— 数据集与标注

- [`d06_single_mutation_dataset.md`](datasets/d06_single_mutation_dataset.md) ——
  D06 单变更标签数据集。导出件为 `exports/d06_single_mutation_dataset.docx`。
- [`dataset_closed_loop_validation.md`](datasets/dataset_closed_loop_validation.md) ——
  `dataset` 项目与主项目之间的自动闭环验证。
- [`real_photo_manual_annotation.md`](datasets/real_photo_manual_annotation.md) ——
  桌面端实拍检测后的人工标注流程。
- [`latest15_gt_audit.md`](datasets/latest15_gt_audit.md) —— latest15 GT 与候选审计。
- [`latest15_manual_gt_method_comparison.md`](datasets/latest15_manual_gt_method_comparison.md) ——
  latest15 上 traditional 与 hybrid 的人工 GT 对比。

`real50` 和 `latest15` 都是开发集，不是未触碰的最终测试集。

## pipeline/ —— 主流程

- [`current_text_graphic_check_flow.md`](pipeline/current_text_graphic_check_flow.md) ——
  当前文字检查和图形检查流程，覆盖 CLI、桌面端和批量入口的调用链。
- [`sensitive_text_promotion_toggle.md`](pipeline/sensitive_text_promotion_toggle.md) ——
  兜底提权开关的配置指南。

## plans/ —— 历史执行方案

- [`pyside6_desktop_app_plan.md`](plans/pyside6_desktop_app_plan.md) —— PySide6 桌面端开发方案。
- [`codex_desktop_app_execution_plan.md`](plans/codex_desktop_app_execution_plan.md) ——
  桌面端重构执行方案（以参考图为目标）。
- [`desktop_app_output_optimization_plan.md`](plans/desktop_app_output_optimization_plan.md) ——
  桌面端结果输出收敛方案。
- [`scanner_integration_execution_plan.md`](plans/scanner_integration_execution_plan.md) ——
  扫码器接入执行方案。
- [`hikrobot_mvs_camera_integration_improvement_plan.md`](plans/hikrobot_mvs_camera_integration_improvement_plan.md) ——
  海康 MVS 工业相机集成修改方案。

## operations/ —— 部署与运行

- [`desktop_one_click_launch.md`](operations/desktop_one_click_launch.md) —— 桌面端本机一键启动。
- [`vllm_startup.md`](operations/vllm_startup.md) —— vLLM 启动说明。默认值不一定和最近一轮评测
  用的模型一致，以评测脚本里实际传的参数为准。
- [`hikrobot_mvs_camera_runbook.md`](operations/hikrobot_mvs_camera_runbook.md) ——
  海康 MVS 相机现场运行手册。
- [`linux_deployment_and_license.md`](operations/linux_deployment_and_license.md) ——
  Linux 部署包与离线授权。
- [`third_party_license_checklist.md`](operations/third_party_license_checklist.md) ——
  第三方许可证交付清单。

## poc/ —— 技术选型追溯

- [`ocr_resource_and_accuracy_notes.md`](poc/ocr_resource_and_accuracy_notes.md) ——
  OCR 准确率、显存与延迟记录。
- [`rapidocr_tensorrt_poc_trace.md`](poc/rapidocr_tensorrt_poc_trace.md) ——
  RapidOCR + PP-OCRv5 + TensorRT POC 追溯记录。
- [`pp_doclayout_hf_gpu_poc_trace.md`](poc/pp_doclayout_hf_gpu_poc_trace.md) ——
  PP-DocLayoutV3 HuggingFace/PyTorch GPU POC 追溯记录。

## 新增文档放哪

- 跑完一轮实验 → `experiments/<主题>_<日期>.md`，图片放 `report_assets/<主题>/`，正文用
  `../report_assets/...` 引用。
- 综述、进展报告、模型方案 → `research/`。
- 新数据集或标注流程 → `datasets/`。
- 改了生产流程的行为或开关 → 更新 `pipeline/` 下对应文档，并在根目录 `AGENT_CHANGELOG.md`
  追加一条。
