# 参考模板条件下的细粒度标签差异检测：2024–2026 文献调研

> 调研时间：2026-07-15  
> 对应项目：`gree-label-detection`、`aa_clip_exp`、`dataset`  
> 当前实验基线：M18，冻结 DINOv2 + 局部 token 匹配 + 学习式定位 + 双路 OCR

## 1. 调研问题与范围

项目的实际任务不是标准的单图工业异常检测，而是：给定 PDF 模板和实拍标签，传统
差分先产生候选区域，模型再判断对应位置是否存在真实文字/图形变化，并定位变化区域。
更准确的定义是 **reference-conditioned pairwise fine-grained difference verification**。

本次调研检索 2024–2026 年 CVPR、ICCV、ECCV 等正式论文，重点回答四个问题：

1. 如何在轻微错位下建立可靠的模板—实拍局部对应；
2. 如何区分真实字符变化与光照、模糊、打印、配准残差；
3. 如何生成接近真实决策边界的合成困难样本；
4. 如何利用少量正常实拍参考缓解 PDF—相机域差异。

共下载并阅读 19 篇论文。下文重点综述与 M18 最相关的工作，其余论文作为辅助对照。
PDF 位于 [`papers/`](papers/)，均来自会议官方开放页面或作者公开版本。

## 2. 主要结论

1. **M18 的主要问题不是 backbone 不够大，而是对应关系过粗且没有匹配置信度。**
   当前在 `24×24` token 网格上硬选 `±2 patch` 最相似位置；单个 patch 约覆盖 14×14
   输入像素，无法稳定定位上标等细小变化。RoMa、DFM 和 COG 都表明，应将匹配构造成
   概率分布或可学习 cost volume，并显式输出 matchability/uncertainty。

2. **PDF 是语义真值，但不是实拍外观的充分正常参考。** FastRef、RAID 和
   Odd-One-Out 均说明，多参考或正常原型可以覆盖正常外观的多模态变化。项目应保留 PDF
   作为内容锚点，同时增加每模板少量正常实拍 reference bank，建模打印、相机和光照噪声。

3. **合成数据必须同时拥有精确语义 mask 和接近边界的困难程度。** 只修复 mutation
   mask 可以解决定位监督污染，但不能自动解决 synthetic/real 分数漂移。RealNet、GLASS
   和 PGBL 共同指向“可控强度 + 接近正常流形/原型边界”的合成，而不是无限增加普通
   blur、shadow 或随机噪声。

4. **局部变化信号会在深层和大 crop 中被稀释。** DCP-SFR 与 Triad 分别通过早期多层
   defect cue 保留和高分辨率 ROI token 解决这一问题。项目更适合采用“整行上下文低分辨率
   + 候选局部高分辨率”的双尺度输入，而不是把整个 review crop 统一缩放为 336×336。

5. **真实域校准必须被当成独立问题。** ARF 和 PIAD 都不把大残差直接视为异常，而是
   分离正常域变化与持续异常信号。M18 三 seed 的阈值从 0.024 到 0.577 波动，说明下一阶段
   除 F1 外必须报告分数分布、ECE/Brier、阈值方差和 seed 方差。

## 3. 核心论文与项目映射

| 年份/会议 | 工作 | 核心思想 | 对本项目的直接价值 | 主要边界 |
| --- | --- | --- | --- | --- |
| 2025 CVPR | [DFM](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_DFM_Differentiable_Feature_Matching_for_Anomaly_Detection_CVPR_2025_paper.html) ([PDF](papers/2025_CVPR_DFM.pdf)) | 把近邻匹配改造成可微 similarity-matrix pooling，匹配头与特征适配联合优化 | 将 M18 的硬局部最近邻替换为可学习局部 cost volume；两阶段交替训练可降低联合训练不稳定性 | 标准异常检测使用正常 memory bank，不是固定位置的 PDF—照片对 |
| 2024 CVPR | [RoMa](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html) ([PDF](papers/2024_CVPR_RoMa.pdf)) | DINOv2 做鲁棒粗匹配，专用卷积特征做精细定位，同时预测 matchability | 支持“DINO 粗对应 + 高分辨率浅层特征细化 + 置信度”的结构 | 完整 RoMa 较重，不能直接作为桌面端最终模型 |
| 2026 CVPR | [COG](https://openaccess.thecvf.com/content/CVPR2026/html/Che_COG_Confidence-aware_Optimal_Geometric_Correspondence_for_Unsupervised_Single-reference_Novel_Object_CVPR_2026_paper.html) ([PDF](papers/2026_CVPR_COG.pdf)) | 单参考条件下建立置信度感知的几何对应，并抑制不可靠匹配 | 匹配置信度可用于衰减配准不确定区域，而不是把其 residual 当作字符变化 | 面向 6D 姿态估计，需抽取 correspondence 思想而非整套模型 |
| 2025 CVPR | [PIAD](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_PIAD_Pose_and_Illumination_agnostic_Anomaly_Detection_CVPR_2025_paper.html) ([PDF](papers/2025_CVPR_PIAD.pdf)) | 在 RGB 与 illumination-invariant reflectance 域联合比较，并先恢复对应姿态 | 提示加入反射率/梯度等光照不变分支，分离光照变化和内容变化 | 使用多视角 3DGS，项目不需要其 3D 部分 |
| 2026 CVPR | [ARF](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Anomaly-Related_Residual_Fields_for_Cross-domain_Anomaly_Detection_CVPR_2026_paper.html) ([PDF](papers/2026_CVPR_ARF.pdf)) | 从 residual 演化中提取跨步骤持续的非平稳异常信号，并做跨域 field alignment | 说明 residual 幅值本身不可靠；可用多增强/多层一致性提取稳定变化，抑制随机配准噪声 | 基于 diffusion 时间序列，直接实现成本较高 |
| 2026 CVPR | [FastRef](https://openaccess.thecvf.com/content/CVPR2026/html/Li_FastRef_Fast_Prototype_Refinement_for_Few-shot_Industrial_Anomaly_Detection_CVPR_2026_paper.html) ([PDF](papers/2026_CVPR_FastRef.pdf)) | 用 query 适配少量正常 prototype，同时通过最优传输避免 anomaly 泄漏进 prototype | 可用于每模板少量正常实拍参考的 query-conditioned 外观适配 | 若异常抑制失败，query adaptation 可能反而吸收真实错误 |
| 2026 CVPR | [RAID](https://openaccess.thecvf.com/content/CVPR2026/html/Cai_RAID_Retrieval-Augmented_Anomaly_Detection_CVPR_2026_paper.html) ([PDF](papers/2026_CVPR_RAID.pdf)) | 按类别原型、语义原型、实例 token 分层检索正常模板，再过滤匹配 cost volume 的噪声 | 支持“PDF + 多张正常照片”的分层参考库与 top-k 稳健聚合 | 通用多类系统较复杂，本项目模板数少，可做简化版 |
| 2025 CVPR | [Odd-One-Out](https://openaccess.thecvf.com/content/CVPR2025/html/Bhunia_Odd-One-Out_Anomaly_Detection_by_Comparing_with_Neighbors_CVPR_2025_paper.html) ([PDF](papers/2025_CVPR_OddOneOut.pdf)) | 正常性由同场景多个对象的相对比较定义，采用跨实例局部匹配 | 证明参考条件异常判断比单图全局正常性更贴近质检任务 | 工作依赖多视角 3D 对象，本项目只借鉴相对正常性定义 |
| 2026 CVPR | [PGBL](https://openaccess.thecvf.com/content/CVPR2026/html/Liao_Multi-Prototype_Compactness_and_Boundary-Aware_Synthesis_for_Unsupervised_Anomaly_Detection_CVPR_2026_paper.html) ([PDF](papers/2026_CVPR_MPCBS.pdf)) | 多原型刻画正常域，在原型簇边界合成困难伪异常并学习紧致边界 | 可把不同光照/打印风格建成多个正常 prototype，并在边界附近做 feature-space hard mining | 生成的是特征伪异常，不能代替精确字符级 mutation mask |
| 2024 ECCV | [GLASS](https://doi.org/10.1007/978-3-031-72855-6_3) ([PDF](papers/2024_ECCV_GLASS.pdf)) | 全局特征与局部图像联合合成，利用梯度上升生成 near-distribution 弱异常 | 适合构造“肉眼微小但应判错”的上标、字母大小写和型号后缀困难正例 | 通用纹理异常策略可能破坏文字语义 |
| 2024 CVPR | [RealNet](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.html) ([PDF](papers/2024_CVPR_RealNet.pdf)) | 可控强度 diffusion anomaly synthesis，并选择有效多层特征与 residual | 支持按难度做 curriculum，并说明多层特征需要选择而非全部拼接 | diffusion 生成文字时可能产生不可控拼写变化，不宜直接生成标签 GT |
| 2026 CVPR | [DCP-SFR](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Defect_Cue-Preserved_Structural_Feature_Refinement_for_Few-Shot_Anomaly_Detection_CVPR_2026_paper.html) ([PDF](papers/2026_CVPR_DCSFR.pdf)) | 保留早期多层 defect cue，条件放大后用正常参考重建，并做结构/边缘约束 | 直接对应 M18 上标信号在大 crop、深层特征中消失的问题 | 使用 CLIP few-shot 框架；本项目应迁移 cue-preservation 而非 CLIP prompt |
| 2025 ICCV | [Triad](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Triad_Empowering_LMM-based_Anomaly_Detection_with_Expert-guided_Region-of-Interest_Tokenizer_and_ICCV_2025_paper.html) ([PDF](papers/2025_ICCV_Triad.pdf)) | 视觉专家提供高分辨率可疑 ROI token，再结合制造知识判断 | 支持整行上下文与候选高分辨率 ROI 双流输入 | LMM 推理成本高，不适合作为当前快速判别器 |
| 2025 ICCV | [OmniDiff](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_OmniDiff_A_Comprehensive_Benchmark_for_Fine-grained_Image_Difference_Captioning_ICCV_2025_paper.html) ([PDF](papers/2025_ICCV_OmniDiff.pdf)) | 多尺度差分感知，数据覆盖 OCR、光照、视角、数量与颜色等 12 类变化 | 可用于建立标签差异/干扰因素 taxonomy 和分层评估 | 目标是差异描述，不直接输出工业检测决策 |

## 4. 对现有 M18 的诊断

### 4.1 对应关系过于离散

M18 对每个 target token 在模板 `5×5` token 邻域内硬选一个最相似 token。该过程没有：

- 亚 token 位移；
- 多峰匹配分布；
- 匹配熵或置信度；
- 前后向循环一致性；
- 细粒度浅层特征复核。

因此，轻微字形错位会被当作 residual，而真实上标也可能被错误匹配到相邻字符。RoMa 的
“foundation feature 粗匹配 + 专用特征细化”和 DFM 的可微 cost-volume pooling 比继续
扩大搜索半径更有依据。

### 4.2 正常域只由 PDF 表示

PDF 能确定应印刷的字符，但与实拍在字体栅格化、打印网点、曝光、模糊和反光方面存在
系统差异。单一 PDF prototype 无法区分这些多模态正常变化。PGBL、FastRef 和 RAID 均
支持把正常状态表示为多个 prototype，而非单一中心。

### 4.3 定位监督与决策边界是两个问题

精确 mutation mask 能修复 M18 attention 学错位置的问题，但当前 synthetic validation
与 real50 的概率范围仍可能不同。应同时解决：

1. **定位标签质量**：生成器在任何退化前记录真实 glyph mutation mask，并同步执行几何变换；
2. **困难程度质量**：对 mutation 强度、字符相似度、模糊度和对比度做 curriculum；
3. **正常域覆盖**：使用多种正常打印/相机风格形成 prototype；
4. **校准**：评估概率分布和固定阈值迁移，而不只看 synthetic F1。

### 4.4 CLIP 文本方向优先级较低

[AA-CLIP](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.html)
([PDF](papers/2025_CVPR_AA-CLIP.pdf))、
[PromptAD](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html)
([PDF](papers/2024_CVPR_PromptAD.pdf)) 等证明了 prompt 对通用异常语义有效，但项目已有实验
表明 generic/PDF CLIP text 不能稳定提升细字符差异，真正有效的是目标侧 OCR 数值特征。
因此不建议下一轮重新投入 generic prompt engineering。

## 5. 建议的下一阶段实验

### E19-0：先修正数据与评估契约

这是后续模型实验的前置条件。

1. `dataset` 在字符替换/插入/删除渲染时直接输出精确 mutation alpha mask；
2. mask 与目标图共享透视、缩放、模糊和打印退化变换，不再从 traditional diff 反推；
3. 保留当前 PDF 整行 semantic mask，只作为搜索支持区；
4. real50/latest15 只作历史开发集，不再用于选择阈值、seed 或结构；
5. 新采独立 real validation/final set，重点覆盖上标、型号后缀和行末插入。

### E19-A：置信度感知的双尺度对应定位器（最高优先级）

建议结构：

```text
整行 crop 336×336
  -> frozen DINOv2 多层 token，提供粗语义对应

候选 ROI 高分辨率 crop（保持字符尺度，短边 192/256）
  -> stride 4/8 的浅层 CNN 或高分辨率 ViT 特征，提供笔画级定位

粗特征局部 correlation volume
  -> softmax match distribution
  -> expected warp + entropy/matchability + forward/backward consistency

细特征在粗 warp 周围 refinement
  -> confidence-gated aligned residual
  -> spatial adapter + heatmap
  -> 双路 OCR + classifier
```

与 M18 相比，关键变化不是换 backbone，而是：

- 将 hard argmin 改为 soft correspondence distribution；
- 输出匹配 entropy/confidence；
- 对低置信区域降权，而非直接把其 residual 判异常；
- 引入候选高分辨率支路，避免上标小于一个 DINO patch；
- 使用 exact mutation mask 监督 heatmap，并加入 edge/boundary loss。

最小消融顺序：

1. M18 hard local match；
2. `+` soft local correlation；
3. `+` confidence/cycle gating；
4. `+` high-resolution fine branch；
5. `+` exact-mask boundary loss。

每项固定三 seed、相同 split、相同 OCR 和 synthetic Recall≥0.95 阈值规则。

### E19-B：每模板正常实拍 reference bank

为每个固定模板采集 3–5 张确认无错误的正常实拍，覆盖合理的曝光、轻微透视和清晰度。
推理时同时比较：

```text
PDF reference：提供字符内容和几何锚点
normal-photo references：提供打印/相机正常外观分布
```

先做简单、可解释的版本：按 PDF 行定位相同 ROI，从正常照片中检索 top-k 对应 patch，
对 residual 使用 trimmed mean 或低分位聚合。只有简单版本稳定后，再尝试 FastRef 的
query-conditioned prototype refinement；必须加入 anomaly suppression，避免把真实错误吸收进正常原型。

### E19-C：光照/成像不变 residual

借鉴 PIAD，不直接依赖 RGB/DINO residual 幅值。为模板—目标增加并行的结构域，例如：

- 局部对比度归一化灰度；
- Sobel/Scharr 梯度；
- 可选轻量 Retinex reflectance；
- 同一输入的两种 photometric augmentation 一致性约束。

分类器使用“跨域都持续存在的 residual”，把只出现在 RGB、且在结构域消失的响应视为
光照/打印候选。先用固定算子做消融，确认有效后再训练小型融合模块。

### E19-D：边界附近的标签专用困难样本

不建议直接使用通用 diffusion 改写标签，因为它可能改变未标注字符。更安全的实现是：

1. 继续使用程序化字符级 mutation 保证语义和 mask 精确；
2. 按 OCR/字形相似度选择困难 mutation，例如 `3/8`、`I/1/l`、`m³/m²`；
3. 连续控制 mutation 对比度、笔画粗细、打印模糊和成像强度；
4. 在冻结视觉特征空间做 boundary-aware hard mining，优先保留离正常 prototype 最近的正例；
5. 负例使用相同文字、独立打印/相机退化，形成与困难正例配对的 hard negatives。

这相当于把 GLASS/PGBL 的 near-boundary 思想约束在标签语义生成器内，避免失去 GT 可信度。

## 6. 评估协议补充

后续除现有最终框 TP/FP/FN 外，至少增加：

- candidate PR-AUC，观察可分性而不只看一个阈值；
- ECE、Brier score 和各 seed 的 synthetic threshold；
- 阈值、F1、Recall 的三 seed 均值与标准差；
- exact mutation mask 上的 pointing accuracy、IoU 和中心距离；
- 按变化类型分层：上标、型号后缀、大小写、插入/删除、数值/单位；
- 按干扰类型分层：光照、模糊、透视、反光、打印、OCR 失败；
- 完整 replay 后的 PDF 预设 display box 指标和单图耗时。

任何模型选择都只使用 synthetic validation 或新增 real validation；全新 real final set 在
配置冻结后只运行一次。

## 7. 推荐顺序与停止条件

推荐顺序：`E19-0 exact mask -> E19-A confidence matching -> E19-B normal reference bank -> E19-C invariance`。

暂不建议：

- 继续扩大 DINO/CLIP backbone；
- 继续训练 generic CLIP prompts；
- 只增加普通 blur/shadow negatives；
- 在 real50/latest15 上继续选 seed 或调阈值；
- 在没有精确 mutation mask 前继续解释 attention heatmap。

E19-A 只有同时满足以下条件才进入下一阶段：

1. synthetic 三 seed 定位指标和分类 F1 均不低于 M18；
2. 历史 real50/latest15 三 seed均值提高，而不是只有单 seed 提高；
3. threshold 跨 seed 波动明显收窄；
4. `850m³/h -> 850m²/h` 等关键样本的 attention 峰值确实落在变化字符；
5. 全新 real final set 完整 replay 优于当前部署候选。

## 8. 辅助阅读

- [Dinomaly, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.html)
  ([PDF](papers/2025_CVPR_Dinomaly.pdf))：说明 DINO foundation feature 与松约束重建的价值；
- [SeaS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Dai_SeaS_Few-shot_Industrial_Anomaly_Image_Generation_with_Separation_and_Sharing_ICCV_2025_paper.html)
  ([PDF](papers/2025_ICCV_SeaS.pdf))：few-shot anomaly generation 中分离正常外观和 defect 表征；
- [Multi-View Pose-Agnostic Change Localization, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Galappaththige_Multi-View_Pose-Agnostic_Change_Localization_with_Zero_Labels_CVPR_2025_paper.html)
  ([PDF](papers/2025_CVPR_MultiViewChange.pdf))：多视角一致性抑制阴影、反射和错位伪变化。

## 9. 总结

近期文献与当前实验共同指向同一个结论：标签差异检测的下一步应从“更强的全局分类器”
转向“更可靠的对应、更细的局部特征、更完整的正常参考分布，以及更可信的边界监督”。

最有价值的单项改进是 **置信度感知的粗到细 correspondence localizer**；最有价值的
数据改进是 **退化前生成的精确 glyph mutation mask**；最有价值的真实数据改进是
**每模板少量正常实拍 reference bank + 独立 real final set**。
