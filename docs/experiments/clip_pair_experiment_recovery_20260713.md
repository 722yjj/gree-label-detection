# CLIP Pair 实验恢复快照（2026-07-13）

## 实验目的与边界

- 企业目前只提供5个固定PDF模板，目标是训练closed-template specialist，降低这5个模板
  的实拍漏检与误报；不再以未知模板zero-shot泛化为主要交付目标。
- 主流程保持不变：traditional diff产生trigger candidate，候选分配到PDF映射预设框，
  exact review crop送入pair确认器，keep后由原流程执行预设框refine/merge和最终显示。
- 模型输入可以使用trigger candidate mask帮助pooling，但最终框评估必须使用完整主流程
  display box，不能直接拿`candidate_box_final`作为最终输出框。
- 研究主日志：`docs/research/clip_pairwise_label_verification_research_plan.md`。
- 中英术语的首次统一定义见主日志`0.4 术语说明`；后续日志不再重复展开。

## 当前推荐模型 M07

- Frozen OpenCLIP `ViT-L-14-336-quickgelu`，官方权重：
  `/home/jnu/models/clip/ViT-L-14-336px.pt`。
- 第6/12/18/24层对应patch absolute difference；candidate masked mean/max、global mean、
  candidate top-10% pooling；每分量L2 normalize。
- Patch feature 16,384维 + OCRv6数值特征24维；head hidden=128。
- OCR引擎必须使用PP-OCRv6 medium det/rec Transformers环境：
  `/home/jnu/venvs/gree-layout-hf-gpu`，不能回退旧Paddle推理。
- 阈值规则：只在D05 synthetic validation选择`Recall >= 0.95`时Precision最高、阈值最高
  的点。该规则是在观察latest15后提出，因此latest15是开发集，不是严格final test。
- M07训练目录：
  `aa_clip_exp/results/label_pair/synthetic_d05_specialist_linegt_v2_patch_ocr_h128_seed*`。

## 数据进度

### D05 五模板specialist

- 300个独立mutation、390 GT；600 defect case + 150 clean case。
- 3948 review pair：740正/3208负。
- train/val/test=`2829/589/530`，均包含5模板；source分组零泄漏。
- Pair数据：`results/clip_pair_datasets/synthetic_d05_specialist`。
- 视觉cache：
  `aa_clip_exp/results/label_pair/feature_cache/synthetic_d05_specialist_linegt_v2_l14_336_multilayer_norm.pt`。
- 旧GT直接使用mutation span，会产生单字符/极小上标框；已修正为PDF文本行与精确mutation
  框的并集。严格修正版固定旧3948 pair和全部视觉/OCR特征，只重标24个旧负例：
  764正/3184负，唯一有效路径为
  `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v2_relabel`。
- 重新跑当前traditional得到的3295-pair目录
  `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v2`混入了候选流程漂移，只用于
  审计，不能训练或与旧M07作GT单变量对照。

### latest15开发集

- 15张、15 GT；已用于策略观察，不再作为独立final test。
- corrected M07加默认高风险PDF行候选恢复后，完整主流程三seed全部为15/0/0，F1=1.0。
- 产物：`results/evaluation/latest15_linegt_m07_vs_candidate_recovery`。

### 新增独立50张 single10_20260712

- 实际为5模板各10张，共50张、50 GT。
- 用户修正过case
  `600004078454_20260712-164724-555646_600004078454`：GT由错误的`590m/h`改为底部地址行。
- 正确pair版本必须使用：
  `results/clip_pair_datasets/real_single10_20260712_v2_gtcorrected`；v1标签已过期。
- 修正后265 pair：52正/213负；47/50 GT有trigger candidate，3个候选阶段FN。
- 正确总评估目录：
  `results/evaluation/single10_20260712_unfiltered_vs_m07_gtcorrected_175005`。
- 未过滤traditional：47/202/3，F1=0.3144。
- M07完整主流程三seed：
- corrected M07加默认高风险PDF行候选恢复后的完整主流程三seed：
  - seed10：42/7/8，F1=0.8485；
  - seed11：43/12/7，F1=0.8190；
  - seed12：42/5/8，F1=0.8660。
- 推荐real pair路径更新为：
  `results/clip_pair_datasets/real_single10_20260712_v4_gtcorrected_sensitive_line_recovery`；
  291 pair（57正/234负），50/50 GT均有trigger candidate。
- Seed12按模板：
  - 600004085656：10/0/0；
  - 600004083205：10/2/0；
  - 600004075219：8/2/2；
  - 600001076226：6/1/4；
  - 600004078454：6/0/4。

## 错误审计 M08-error-audit

- 产物：`results/evaluation/single10_20260712_m07_error_audit`。
- 50 case分类：31三seed全正确；3 candidate-stage FN；6稳定model FN；1 seed-variable FN；
  5 stable-FP case；4 seed-variable-FP case。
- Candidate FN：两个`600001076226`型号行I/1/括号微变化；一个`600004078454`风量上标。
- 稳定model FN：4个集中于`m³/h -> m²/h`/极小上标，2个型号后缀/括号/WIFI。
- Stable FP多伴随OCR边界伪字符：`ZAirFlowVolume`、地址末尾多`0`、编码前多字符、
  局部品牌残片。
- Raw patch对照三seed：32/1/18、30/1/20、34/1/16，F1=0.7711/0.7407/0.8000。
  OCR版本Recall/F1更高，因此不能简单删除OCR。

## M08-A 定向上标实验（负结果，DONE）

- Dataset builder新增`mutation_profile=superscript`；默认diverse行为不变。
- `600001076226`无可变独立上标span，强制生成失败；应单独处理型号后缀。
- 其余4模板共60独立上标mutation、120 case、492 pair（181正/311负）。
- 只把M08 train 317 pair（119正/198负）加入D05 train；D05 val/test冻结。
- 合并时发现D05/M08 pair ID冲突；不完整v1合并产物不要使用。正确合并版本：
  `results/clip_pair_datasets/synthetic_m08_d05_plus_superscript_train_v2`，M08 pair加
  `m08sup__`前缀。
- M08三seedD05 test全部84/1/21、F1=0.8842，主要降方差。
- 独立50张完整主流程：
  - M07：40/8/10、41/13/9、40/5/10；F1均值0.8156；
  - M08-A：40/7/10、39/5/11、40/8/10；F1均值0.8236。
- TP没有提升；原稳定FN全部仍在，平均F1仅+0.008。简单上标合成过于容易，M08-A为
  负结果，不替换M07。
- 对照：`results/evaluation/single10_20260712_m07_vs_m08_superscript_gtcorrected`。
- line-GT修正后M08的492 pair标签不变；基于corrected D05重训后完整主流程三seed为
  40/5/10、40/5/10、44/23/6，F1=0.8421/0.8421/0.7521。seed12找回3个上标/单位
  case和1个WIFI case，但FP升至23；三seedF1均值0.8121，仍低于corrected M07的
  0.8210，不替换M07。
- corrected对照：
  `results/evaluation/single10_20260712_linegt_v2_m07_vs_m08_superscript_gtcorrected`。

## 已完成的关键代码

- `scripts/build_clip_pair_dataset.py`：synthetic pair导出、within-template source split。
- `scripts/build_real_clip_pair_dataset.py`：真实GT pair导出。
- `scripts/build_ocrv6_pair_features.py`：OCRv6缓存。
- `scripts/replay_clip_pair_predictions.py`：缓存决策重放完整预设框refine/merge流程。
- `scripts/analyze_single10_m07_errors.py`：分层错误审计。
- `aa_clip_exp/label_pair/{dataset.py,model.py,train_patch.py,ocr_features.py,evaluate_checkpoint.py}`。
- `aa_clip_exp/label_pair/prepare_m08_training.py`：D05+M08 cache/manifest合并；只用v2产物。
- `aa_clip_exp/label_pair/train_real_grouped_cv.py`：按template/case分组的5折真实微调、
  epoch 0对照、强制微调和OOF预测。

## M10 真实困难样本适配（负结果，DONE）

- real50已经参与M08/M09错误分析和规则开发，因此本轮只按开发集5折交叉验证报告，不再
  当作独立final test。每折每模板6 train/2 validation/2 test，同一case候选不拆分；
  291/291 pair各作为OOF test一次。
- corrected M07三seed初始化，冻结CLIP/OCR特征，只用`lr=1e-5`微调h128 head 20 epoch。
  三seed共15折全部由validation选择epoch 0；强制微调版本也只选择epoch 1。
- 完整主流程：
  - M09 baseline：42/7/8、43/12/7、42/5/8，F1均值0.8445；
  - real-validation校准：40/12/10、41/13/9、42/8/8，F1均值0.8043；
  - 强制微调：40/12/10、41/13/9、40/8/10，F1均值0.7964。
- 10-case validation阈值跨折严重漂移；稳定FN由baseline 7个变为校准7个、强制微调9个。
  M10不替换M07+M09，5折checkpoint和real-validation阈值均不可部署。
- 有效产物只使用带`v2`的训练目录：
  `aa_clip_exp/results/label_pair/m10_real50_grouped5fold_hard_adapt_lr1e5_v2_seed*`；
  最终框对照：
  `results/evaluation/single10_20260712_m10_grouped5fold_hard_adaptation`。

## PDF语义候选重训（2026-07-13）

- 已按正确结构完成D05v3：差分只触发，pair候选是PDF文本框子集；数值、单位和上标合并。
  750 case导出3730 pair（788正/2942负），3730/3730精确属于PDF框，未映射=0，source
  泄漏=0；OCRv6 3730/3730、0 error。
- 纯D05v3三seed在real50完整流程为44/6/6、43/6/7、45/13/5，F1均值0.8607；旧M07
  在相同语义候选上的均值为0.8522。latest15为15/0/0、15/0/0、15/1/0。
- 纯D05v3仍未稳定命中关键`850m³->850m²`：seed10/11为FN，seed12为TP。
- M08上标源也按语义框重建，只追加train的279 pair；三seed均命中关键case，但real50为
  44/2/6、45/12/5、50/36/0，F1=0.9167/0.8411/0.7353，方差和seed12误报不可接受。
  2/3多数投票为45/12/5，未解决共同误报。不得只挑seed10作为最终结论。
- 数据与评估产物：
  `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v3_pdf_semantic`、
  `results/clip_pair_datasets/synthetic_m08_superscript_linegt_v3_pdf_semantic`、
  `results/evaluation/single10_20260712_v6_pdf_semantic_d05v3_m08semantic_recall95`。

## DINOv2与letterbox 2×2实验（2026-07-13）

- 已实现统一OpenCLIP/DINOv2视觉接口和center-crop/letterbox同几何图片+mask变换。旧CLIP
  center接口与旧缓存最大误差2.1e-6；letterbox图片/mask几何测试通过。
- 固定D05v3+M08、OCR、pooling、h128、三seed和Recall95规则。real50完整流程F1：
  - CLIP center：0.9167/0.8411/0.7353，均值0.8310；
  - CLIP letterbox：0.8491/0.8654/0.7328，均值0.8158；
  - DINO center：0.8824/0.8679/0.7460，均值0.8321；
  - DINO letterbox：0.9412/0.8764/0.8936，均值0.9037。
- DINO-letterbox latest15为1.0000/0.8889/1.0000，三seed均0 FP；real50平均FP=2.0，
  远低于CLIP-center的16.67，但平均Recall由0.9267降至0.8600。
- 当前开发集最佳checkpoint：
  `aa_clip_exp/results/label_pair/synthetic_d05v3_m08_dinov2_letterbox_patch_ocr_h128_seed20260710/best_checkpoint.pt`；
  real50=48/4/2、F1=0.9412，latest15=15/0/0。关键`850m³->850m²`命中但该case另有1FP；
  DINO-letterbox另外两个seed未命中该上标case。
- 可视化：`results/interpretability/850m3_to_850m2_dinov2_letterbox_seed10`。letterbox完整
  保留候选，梯度×差异证据集中于`850m²/h`字符内部。
- DINO官方权重：`/home/jnu/models/dinov2/dinov2_vitl14_pretrain.pth`，SHA256=
  `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`。
- 这是开发集最佳而非独立final结论；正式替换需要新增真实final set。

## Candidate-mask OCR消融（2026-07-14）

- OCR改为mask外接框+8%水平/15%垂直padding+8像素白边+最小96像素高；其他视觉cache、
  split、模型和Recall95规则冻结。关键case OCR成功变为`850m3/h -> 850m2/h`唯一替换。
- candidate OCR覆盖：synthetic 4009/4009、real50 268/268、latest15 75/75，均0 error。
- real50完整流程由review OCR的48/4/2、39/0/11、42/2/8变为candidate OCR的
  42/3/8、40/3/10、41/3/9；F1均值0.9037降至0.8722。
- latest15 F1均值由0.9630降至0.8583。candidate OCR三seed均0 FP，但Recall明显下降。
- 关键`850m³->850m²`由review OCR的1/3 seed命中，改善为candidate OCR 3/3命中且case
  0 FP。这里旧版`A/I -> A/II`边界丢失分析只适用于拆分的`/I`小mask；型号整行合并后，
  candidate OCR已正确读为`GWH18AUDXE-K6DNA1A/I -> GWH18AUDXE-K6DNA1A/II`。但固定PDF
  整行框仍不能覆盖行末之外的新字符，例如`...(WIFI) -> ...(WIFI)I`；整行合并只修复拆分
  后缀，不等于消除所有候选边界问题。
- 边界丢失可视化：
  `results/interpretability/candidate_ocr_boundary_context_A_I_to_A_II.png`。
- 结论：不替换当前DINO-letterbox review-OCR seed10。若继续OCR方向，只做review+
  candidate双路特征的严格消融，不继续围绕同一开发集调padding。

## PDF型号整行候选修复（2026-07-14）

- 发现`600004078454`、`600004083205`、`600004085656`的PDF文本层会把型号主体与
  `/I`拆成两个region。已在PDF语义region阶段合并相邻型号主体和后缀，例如
  `GWH18AUDXE-K6DNA1A + /I -> GWH18AUDXE-K6DNA1A/I`；无关同排文本不会合并。
- 真实样本`600004083205_20260712-165038...`端到端确认候选框为完整型号行
  `[472,107,878,144]`，review crop为460x63，实际mask非零bbox为
  `[32,13,438,50]`，不再是单独`/I`。
- D05按新几何重建为3714 pair（795正/2919负），3714/3714精确等于PDF框，独立
  `/I`由22降为0，22个旧样本全部有完整型号替代，source split泄漏为0。real50为
  264 pair、latest15为73 pair，独立`/I`均为0，candidate recall仍为50/50和15/15。
- review-OCR重建覆盖3714/3714、264/264、73/73，全部0 error。D05v4加冻结M08 train
  后为train/val/test=`2918/572/503`；val/test仍只来自D05v4。
- 新几何重训DINO-letterbox三seed完整real50为`38/0/12`、`41/2/9`、`37/0/13`，
  F1=`0.8636/0.8817/0.8506`；latest15为`10/0/5`、`13/0/2`、`10/0/5`，
  F1=`0.8000/0.9286/0.8000`。Recall明显退化，不替换旧checkpoint。
- 关键对照：旧DINO-letterbox seed10 checkpoint直接使用新整行crop/mask和新OCR，阈值仍
  由旧synthetic validation Recall95冻结为`0.2733518183`。修复跨PDF行display-box误合并
  后，完整real50为`48/6/2, F1=0.9231`，latest15为`15/1/0, F1=0.9677`。因此部署候选为
  “新整行候选几何 + 旧review-OCR DINO-letterbox seed10 checkpoint”，不是本轮重训模型。
- 最终框合并新增PDF行约束：两个候选都有`text_line_key`且key不同时禁止合并；无key候选
  也不能桥接两个不同PDF行。修复前real50的`48/4/2`和latest15的`15/0/0`分别隐藏了2个
  和1个跨行FP，只保留为历史指标，不再作为当前基准。
- 关键产物：
  `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v4_pdf_semantic_modelrow`、
  `results/clip_pair_datasets/real_single10_20260712_v7_gtcorrected_pdf_semantic_modelrow`、
  `results/clip_pair_datasets/real_latest15_v4_pdf_semantic_modelrow`、
  `results/evaluation/single10_20260712_v7_modelrow_old_vs_retrained_best`、
  `results/evaluation/latest15_v4_modelrow_old_vs_retrained_best`、
  `results/evaluation/single10_20260712_v7_modelrow_cross_line_merge_fix`、
  `results/evaluation/latest15_v4_modelrow_cross_line_merge_fix`。

## Review + candidate双路OCR消融（2026-07-14）

- 在D05v4整行几何上重新生成candidate-mask OCR：synthetic D05v4/M08为3714/407条，
  real50/latest15为264/73条，全部0 error。真实负例strict equal由review的169/212、44/58
  提升为candidate的191/212、58/58；`A/I -> A/II`也能保留完整型号行变化。
- 冻结DINOv2-L/14视觉cache、split、h128、三seed和synthetic validation Recall>=0.95规则，
  将24维review OCR与24维candidate OCR拼接为48维。完整line-safe最终框结果：
  - real50：`45/1/5`、`38/0/12`、`41/0/9`，F1=`0.9375/0.8636/0.9011`；
  - latest15：`14/0/1`、`10/0/5`、`13/0/2`，F1=`0.9655/0.8000/0.9286`。
- seed10在real50优于当前review基准`48/6/2, F1=0.9231`：FP从6降至1，但TP也从48降至45。
  用户指出的`600004075219...164003`从`1 TP + 2 FP`改善为`1 TP + 0 FP`；另外三个FP也被
  消除，但新增漏检`600001076226...163822`、`600004078454...164604`和`...164630`。
- 错误pair审计进一步确认：`590m³/h -> 590m⁸/h`两例被两路OCR都读成`590m3/h`；型号
  行另有行末新增字符落在PDF行mask之外、行内插入字符OCR漏读、大小写变化被folded特征
  弱化。唯一FP来自candidate OCR将真实`/I`误读成`/1`。latest15的`5.60kW`在模板、目标
  和两路OCR中均一致，应先复核该GT。
- 三seed均值和Recall稳定性不支持替换默认模型；seed10的提升只能记作开发集候选，不能继续
  在real50/latest15上选seed。部署候选仍是旧M11 review-OCR seed10 checkpoint配合新整行
  几何和line-safe最终框合并，待全新真实final set再比较双路OCR。
- 关键产物：
  `aa_clip_exp/results/label_pair/synthetic_d05v4_m08_modelrow_dinov2_letterbox_dualocr_patch_ocr_h128_seed*`、
  `results/evaluation/single10_20260712_v7_modelrow_dualocr_3seed_line_safe`、
  `results/evaluation/latest15_v4_modelrow_dualocr_3seed_line_safe`、
  `results/interpretability/dualocr_seed10_error_audit`。

## Synthetic配准污染修复（2026-07-14）

- synthetic错误审计发现模板文字行经常对应target空白或条码。根因是已经正视、尺寸近似
  模板的合成图再次做透视检测：内部条码仅占约5.98%面积却被当成整张标签；之后SIFT用
  17/23个条码局部匹配接受约`0.267x0.232`的错误单应缩放，ECC相关性仅0.196仍标记成功。
- 修复包括：同画布target跳过重复透视；synthetic exporter显式设置
  `target_prealigned=True, skip_feature_align=True, skip_ecc=True`；通用SIFT增加投影面积
  `0.25..4.0`和双轴尺度`0.5..2.0`门禁；ECC相关性低于0.3时回退到输入。
- D05v3/v4各750个case中均有77个灾难性单应尺度，说明当前旧M11也曾在污染数据上训练。
  M08的120个case没有灾难性尺度。真实real50/latest15没有同类`0.2x`坍缩；真实指标仍
  有效，但旧模型只能保留为经验基准。
- 新D05v5使用同一源图、GT、split seed和候选逻辑重建：750 case、3122 pair，相比v4
  3714 pair减少592个错配伪候选；600个有GT的case仍`600/600`有正候选。train/val/test=
  `2205/490/427`，正例=`536/120/116`。加入冻结M08 train后为`2484/490/427`。
- OCR污染同步消失：review OCR test负例双侧非空由旧`293/382`变为`311/311`，平均编辑
  距离由0.338降至0.033；candidate OCR为`310/311`，严格相同`282/311`。两路均0 error。
- 修复后synthetic test最大F1：review三seed=`0.9735/0.9780/0.9735`，dual=
  `0.9782/0.9784/0.9782`；dual seed10在冻结Recall95阈值下只剩`4FN+1FP`，旧版为
  `5FN+40FP`，且新错误图已无跨行错位。
- 但synthetic validation几乎可分，使Recall95阈值升至review约0.998-0.999、dual约
  0.46-0.98，暴露真实域概率未校准。完整line-safe结果：
  - fixed review real50=`32/0/18`,`30/0/20`,`30/0/20`；latest15均`8/0/7`；
  - fixed dual real50=`36/0/14`,`35/0/15`,`35/0/15`；latest15=`11/0/4`,`9/0/6`,`10/0/5`。
  不得用real50/latest15下调阈值；fixed模型暂不替换旧部署候选。
- 关键产物：
  `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v5_prealigned_pdf_semantic_modelrow`、
  `aa_clip_exp/results/label_pair/synthetic_d05v5_m08_prealigned_modelrow_dinov2_letterbox_*`、
  `results/evaluation/single10_20260712_v8_prealigned_training_3seed_line_safe`、
  `results/evaluation/latest15_v5_prealigned_training_3seed_line_safe`、
  `results/interpretability/dualocr_seed10_synthetic_test_error_audit_prealigned`。

## Semantic/evidence双mask消融（2026-07-15）

- 按“现有crop/OCR/标签/split不变，只新增原始差分mask”实现：workflow保存
  `diff_mask | small_text_diff_mask`，每个pair再用其原始trigger components隔离对应像素，
  生成crop-local `evidence_mask`；`candidate_mask`继续用于OCR和候选语义。所有3122个D05v6
  pair均可追溯，evidence面积中位数仅占review crop约1.0%。
- hard-negative审计：D05v6有2350个负pair，其中clean/same约1034个，覆盖geometry、mixed、
  print-scan、shadow、soft-capture；其余1316个来自有真实变更图中的未变候选。约90%的负例
  candidate OCR严格相同。synthetic/real负例evidence面积分布接近，当前错误并非明显缺少
  hard negatives；真实完整结果已接近0 FP，继续只加负例会进一步压低Recall。
- 严格保持旧M08v3的407 pair和279条train不变，只使用M08v4原始diff像素为旧pair补
  evidence mask。最终train/val/test仍为`2484/490/427`，双路OCR和三seed不变。
- pooling消融：semantic/evidence单mask均每层4组特征；evidence在patch grid仅膨胀一圈；
  dual每层为semantic mean/max/top-k + evidence mean/max/top-k + global mean，patch维度由
  16384增至28672，head仍h128。
- synthetic test最大F1：semantic=`0.9782/0.9784/0.9782`；evidence三seed均
  `113/1/3, F1=0.9826`；dual=`0.9697/0.9739/0.9386`。evidence在合成集小幅且稳定改善，
  直接高维拼接反而更不稳定。
- frozen synthetic Recall95完整line-safe真实结果：
  - semantic real50=`36/0/14`,`35/0/15`,`35/0/15`；latest15=`11/0/4`,`9/0/6`,`10/0/5`；
  - evidence real50=`34/0/16`,`17/0/33`,`17/0/33`；latest15=`10/0/5`,`5/0/10`,`5/0/10`；
  - dual real50=`33/1/17`,`29/0/21`,`31/1/19`；latest15=`10/0/5`,`10/0/5`,`9/0/6`。
- 根因：synthetic mutation的raw diff通常等于真实变更；real raw diff会混入局部配准残差。
  `A/I->A/II` evidence准确落在新增I并找回case，但关键`850m³->850m²` evidence主要落在
  `m`的错位笔画而非上标。seed10 evidence相对semantic找回2个real50 TP却丢4个，净退化。
- 决策：保留evidence mask作为审计资产和后续软引导输入，但不替换semantic pooling，
  不部署evidence/dual checkpoint。下一步若继续，应改进real evidence质量或使用低权重软引导，
  并增加real-like hard positives；单纯增加hard negatives不是当前瓶颈。
- 关键产物：
  `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v6_prealigned_dualmask_modelrow`、
  `results/clip_pair_datasets/real_single10_20260712_v9_gtcorrected_dualmask_modelrow`、
  `aa_clip_exp/results/label_pair/synthetic_d05v6_m08v3_*mask*`、
  `results/evaluation/single10_20260712_v9_mask_pooling_ablation_line_safe`、
  `results/evaluation/latest15_v6_mask_pooling_ablation_line_safe`、
  `results/interpretability/mask_pooling_ablation`。

## 当前生产VLM复测（2026-07-17）

- 使用当前修复后的real50 v9和latest15 v6候选/review crop，冻结主流程实际
  `qwen3.6-27b-int4`、temperature=0、256 tokens、普通/micro-batch原prompt、template+
  target+diff三图和`keep_unknown=False`。逐条响应可断点恢复，337次请求0 exception、
  0 unknown。
- real50候选判断为`46 TP / 3 FP / 6 FN / 209 TN`，accuracy=`0.9659`、P/R/F1=
  `0.9388/0.8846/0.9109`；缓存决策重放当前display refine/merge后的完整框为
  `44 TP / 5 FP / 6 FN`，P/R/F1=`0.8980/0.8800/0.8889`。50/50 GT在候选阶段均有
  覆盖，6 FN全部由VLM过滤造成。
- latest15候选和完整框均为`14 TP / 1 FP / 1 FN`，P/R/F1均=`0.9333`。同一27B模型
  的旧候选/旧流程结果为`13/5/2, F1=0.7879`；当前候选、crop、prompt修复后FP 5->1、
  FN 2->1，但这不是单独prompt消融，提升应归因于整套前处理与候选结构修复。
- real50的6个FN包括两个`590m³/h`上标`3->8`、长型号字符变化、WIFI型号变化和两个数值
  行；VLM均以约0.95置信度声称内容相同。3个候选FP来自错位/裁切后的幻读。micro来源
  precision=1.0但recall偏低：single micro=`0.75`，micro batch=`0.60`；普通候选产生全部FP。
- VLM自报confidence大量固定为0.95，正确和错误没有可用校准性。单流实测约5.15秒/候选，
  real50平均每case约5.28次请求，因此仅VLM判断约27秒/case；不可把缓存重放的0.42秒/case
  当作线上耗时。
- 关键产物：`scripts/evaluate_vlm_pair_judge.py`、
  `results/evaluation/real50_v9_production_vlm_qwen36_27b_20260717`、
  `results/evaluation/latest15_v6_production_vlm_qwen36_27b_20260717`，以及对应带
  `_fullflow_20260717`的完整框评测目录。错误三联图位于各pair评测目录的`visualizations/`。

### 生产VLM V2.2候选精细区域+OCR复测（2026-07-17）

- 输入改为两张未标注的candidate-focus crop：模板候选区域和实拍对应区域。focus box由
  原始candidate box增加受图像边界约束的小幅字形上下文得到，不再发送整块review crop、
  坐标、模板/目标前景比例或灰度绝对差分图。
- 文本先验只保留PDF模板文字，并增加对同一实拍focus crop单独执行PP-OCR得到的文字和
  置信度。prompt明确说明实拍OCR可能出错，最终判断必须以两张图的可见字形为准；响应额外
  记录template/target reading以及是否同OCR提示一致，便于审计。
- real50候选级=`45 TP / 0 FP / 7 FN / 212 TN`，P/R/F1=
  `1.0000/0.8654/0.9278`；完整框=`44 TP / 1 FP / 6 FN`，P/R/F1=
  `0.9778/0.8800/0.9263`。唯一框级FP来自一个GT样本保留了两个输出框，并非负候选被VLM
  错判为正。
- latest15候选级和完整框均=`13 TP / 0 FP / 2 FN`，P/R/F1=
  `1.0000/0.8667/0.9286`。两套数据共计完整框=`57 TP / 1 FP / 8 FN`，F1=`0.9268`；
  旧三图方案共计=`58 TP / 6 FP / 7 FN`，F1约`0.8992`。新方案以少1个TP换来少5个FP，
  当前更适合误报成本较高的生产场景，但不是无条件支配旧方案。
- 剩余FN仍以细微字形变化为主。多个数值行和长型号行被模型高置信度解释为“相同文字+模糊”，
  且实拍OCR经常也输出模板值；说明缩小视觉范围解决了错位幻读和背景干扰，却不能解决模型对
  小字符差异的保守偏置。VLM自报0.95/1.0仍不可作为可靠置信度。
- 独立缓存重放后的全流程评测与pair评测的box metrics完全一致，real50/latest15分别50/15
  个样本、0失败，排除了评测脚本或display refine/merge造成指标虚高的可能。
- 关键产物：
  `results/evaluation/real50_v9_production_vlm_v22_candidatefocus_ocr_20260717`、
  `results/evaluation/latest15_v6_production_vlm_v22_candidatefocus_ocr_20260717`、
  对应的`*_fullflow_20260717`目录，以及
  `results/batch_desktop_captures/*_production_vlm_v22_replay_20260717`。

### 生产VLM V2.3 review crop消融（2026-07-17）

- 严格保持V2.2的PDF模板文字、实拍crop OCR、prompt、模型、temperature和unknown策略不变，
  只将两张视觉输入从candidate-focus crop换回完整review crop。每条记录均为`image_count=2`、
  `difference_image_sent=false`；错误图最右侧Diff是在模型响应后生成的审计可视化，从未发送给VLM。
- real50候选级=`45 TP / 1 FP / 7 FN / 211 TN`，P/R/F1=
  `0.9783/0.8654/0.9184`；完整框=`44 TP / 2 FP / 6 FN`，P/R/F1=
  `0.9565/0.8800/0.9167`。相对V2.2没有增加净TP或框Recall，却增加1个候选FP和1个框FP。
- latest15候选级和完整框均=`13 TP / 0 FP / 2 FN`，与V2.2完全相同。因此单纯回退
  review crop是负结果，不替换V2.2。
- 两种crop仅有5个real50候选决策变化：review找回`5.60kW->5.60kWW`和独立右括号候选，
  但丢掉`590m³/h`、`Air Flow Volume`两个原本TP，并在`Weight`邻近区域新增1个FP。
  说明review crop可恢复模板文字框外的新增字符，却也会稀释候选内微差异并引入候选外幻读。
- PDF小片段不是未匹配文字行：PyMuPDF原始文本层将`GWH... WIFI`、`(`、`)`提取为三个
  独立line；现有语义合并只处理单位上标和型号`/I...`后缀，不会把相邻括号合并回型号，
  因而差分框会合法匹配到文本为`")"`的独立区域。
- 关键产物：
  `results/evaluation/real50_v9_production_vlm_v23_reviewcrop_ocr_20260717`、
  `results/evaluation/latest15_v6_production_vlm_v23_reviewcrop_ocr_20260717`、
  对应的`*_fullflow_20260717`和`results/batch_desktop_captures/*_v23_*_replay_20260717`。

## DINO局部定位与real-like正样本平衡（2026-07-17）

- 在冻结DINOv2-L/14、336 letterbox、第`5/11/17/23`层、radius=2 hard local
  matching上训练token localizer。candidate整行mask只限制有效区域，synthetic
  evidence mask只作为定位监督，不作为模型输入；分类仍融合review+candidate两路48维OCR。
- E21新增candidate区域内逐通道残差标准化，消除synthetic/real捕获强度尺度差。相对M18：
  real50三seed框F1由`0.7072`升至`0.7945`，latest15由`0.7751`升至`0.8154`；real50
  pointing/IoU由`0.4615/0.0098`升至`0.6474/0.1835`。E21曾是本阶段最佳。
- 数据审计发现augmentation-label捷径：train的`print_scan`为`230 negative / 0 positive`，
  `geometry/mixed`也只有negative；最接近real的capture profile不能出现在positive。
- E22给全部308个original train positive增加无几何位移的print-scan target副本，mask不变，
  两路OCR均0 error。real50 PR-AUC和IoU改善，但阈值均升至约0.99，框Recall降至0.6267，
  三seed框F1=`0.7312`；全量增强为负结果，不替换E21。
- E23用稳定hash只选77/308个positive做同样增强。real50三seed：
  `36/3/14, F1=0.8090`、`37/3/13, F1=0.8222`、`40/7/10, F1=0.8247`；均值
  precision/recall/F1=`0.8997/0.7533/0.8187`。latest15为`10/0/5`、`10/0/5`、
  `11/0/4`，平均F1=`0.8154`，与E21相同且仍0 FP。E23是当前最佳开发集折中，
  但real50/latest15已反复用于研究比较，仍需全新盲测集才能决定部署。
- E23 seed11的3个real50 FP都来自同一张`600004075219...164019`，分布在标题、风量、
  型号三行；可视化显示全局残余配准误差而非OCR错误。E24只将local matching radius从2
  增到4，seed10 real50为`34/4/16, F1=0.7727`，低于E23 seed10的`36/3/14,
  F1=0.8090`，且synthetic定位IoU下降；扩大搜索会吞掉真实小差异，门控后停止其余seed。
- 关键产物：
  `results/clip_pair_datasets/synthetic_d05v6_plus_m08v3_e23_printscan_positive77_train`、
  `aa_clip_exp/results/label_pair/synthetic_d05v6_m08v3_e23_printscanpositive77_*`、
  `aa_clip_exp/results/label_pair/e23_printscanpositive77_maskedchannel_*`。代表性错误/热力图在
  `aa_clip_exp/results/label_pair/e23_printscanpositive77_maskedchannel_real50_seed20260711_visualized/visualizations`。

## 下一执行点（2026-07-17）

1. 当前研究候选为E23 radius=2；不要选E22全量增强或E24 radius=4。
2. 下一轮特征改进应针对非刚性/尺度残差，但必须保留小字符变化；优先尝试可学习的
   offset penalty或coarse-to-fine correspondence，不再简单扩大hard-match窗口。
3. 合成数据需要在source-level生成带正确变换mask的geometry/mixed正例；crop级实验只能
   安全使用不改变坐标的print_scan。
4. 阈值继续只由synthetic validation Recall>=0.95规则确定，不得用real50/latest15调阈值。

### 历史约束

1. 不继续堆简单PDF上标样本，也不升级backbone/adapter。
2. 3个candidate-stage FN已由M09高风险PDF文本行恢复全部覆盖；其中2个成为TP，另1个转为
   稳定model FN。默认启用，`ENABLE_SENSITIVE_PDF_LINE_RECOVERY=0`可关闭。
3. M10已完成按case分组的真实微调并得到负结果；不要继续在同一real50上搜索学习率、
   epoch或阈值。严格R02必须新增独立真实train/validation/final-test。
4. PDF语义候选、DINOv2+letterbox均已完成。下一步优先采集全新真实final set，重点覆盖
   `600001076226`型号行和`m³/h`上标；如继续合成，应加入相邻PDF行/crop边界hard
   negatives，而非普通全图blur/shadow。synthetic validation阈值规则继续冻结。
5. 任何新指标都必须使用修正GT、PDF预设最终框和完整主流程；旧trigger-box离线框指标
   已被superseded。
6. 型号主体与`/I`必须保持整行候选。当前部署模型仍是旧DINO-letterbox seed10；不得误用
   D05v4 candidate-only或dual-OCR重训checkpoint。双路seed10只能在全新final set盲测。
7. D05v3/v4 synthetic pair已被配准污染结果取代；后续只能使用D05v5 prealigned数据。
   fixed模型的真实Recall下降来自synthetic/real分数域差异，下一步需要独立真实validation
   校准或更贴近真实的合成validation，禁止继续用real50/latest15调阈值。

## Git与产物注意

- 三个项目工作树仍有大量未提交改动；不要清理、reset或覆盖旧结果。
- `aa_clip_exp/label_pair/`和`results/label_pair/`整体仍未跟踪。
- `gree-label-detection`研究文档、实验脚本和多数结果未跟踪。
- `dataset`包含既有用户修改与本实验生成器修改，必须增量工作。
- 旧错误/失败产物保留用于追踪：real v1旧GT、M08合并v1冲突、superscript首次包含
  600001模板的失败目录；后续只使用上文明确的v2路径。
