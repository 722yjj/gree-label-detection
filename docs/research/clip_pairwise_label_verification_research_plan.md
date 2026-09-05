# 基于 CLIP 的成对标签差异确认：模型设计、训练与论文实验方案

> 记录时间：2026-07-10  
> 主项目：`gree-label-detection`  
> 参考项目：`aa_clip_exp`（AA-CLIP，CVPR 2025）

## 0. 执行总览与进度追踪

本文档同时作为实验设计和执行日志。状态只使用以下四种：

- `DONE`：代码、产物和最小验证均已完成；
- `RUNNING`：正在实现或运行；
- `READY`：前置条件已满足，可以开始；
- `TODO`：尚未开始或仍缺前置条件。

| ID | 阶段 | 状态 | 验收条件 | 当前结果/阻塞 |
| --- | --- | --- | --- | --- |
| E00 | 相同主流程的裸 CLIP cosine 基线 | DONE | latest15 端到端输出；候选/review box 与 VLM 一致 | 69/69 review box 一致；P=0.4286，R=0.2000，F1=0.2727，0.793s/图 |
| D01 | 合成数据转主流程 review pair 数据集 | DONE | 导出 template/target crop、candidate/review box、0/1 label、split manifest | 10 case、43 pair；14 正/29 负；按 5 个 template 分组切分 |
| M01 | Frozen B/32 global pair head | DONE | 可训练、保存 checkpoint；输出 val/test candidate 指标 | D02-v3 test：2/0/1、F1=0.8000；cosine 3/86/0、F1=0.0652；仍是小样本 pilot |
| M02 | Frozen L/14@336 cosine 与 global pair head | DONE | 与 B/32 使用相同 split 完成对照 | 同 GPU 环境：B/32 head F1=0.8000；L/14@336 head F1=1.0000；test 仅 3 正例 |
| D02 | clean/same 与 camera-domain augmentation | DONE | dataset 生成 clean pair 和成像扰动；产生 hard negatives | 可复现 v3：30/30 clean 触发候选；271 个 clean hard-negative pair |
| D03 | 正负同域安全增强 | DONE | defect 仅光度增强且 GT 不变；提高 val/test 正例数 | 504 pair；71正/433负；val/test各15正，但仅3个独立 test mutation |
| D04 | Diverse 独立 mutation | DONE | 5模板覆盖多类编辑并做安全同域增强 | 30独立样本/40 GT；候选覆盖36/40；534 pair；test含8个独立 mutation |
| D05 | 五模板专用大样本与同模板source split | DONE | 五模板均进入train/val/test且source零泄漏 | 300独立mutation/387行级GT；750 case；3948 pair；修正后764正/3184负 |
| M03 | Multi-layer patch pair branch | DONE | 对应 patch 差异、candidate-aware pooling 可训练 | D04独立mutation：patch三seed平均F1=0.7009，global=0.5191；现实难度高于D03 |
| M04 | 合成 mutation mask 辅助监督 | TODO | patch map + BCE/Dice；不改变最终框评估 | 当前只有粗 bbox mask；等待精确 changed-pixel mask，不能伪装分割 GT |
| M05 | Residual/feature adapter | DONE | frozen backbone 下完成 adapter 消融 | 64/256维 projection 与 vector/scalar gate 均未稳定超过 raw patch-only；当前否定复杂化 |
| M06 | Generic/template/OCR text branch | DONE | 三类文字输入逐项消融 | generic/PDF CLIP text均为负增益；OCR数值特征将D04三seed均值F1从同维zero control的0.7345提升到0.8333 |
| M07 | 五模板specialist patch/OCR | DONE | 固定五模板内source split训练并在latest15冻结评估 | corrected line-GT三seedlatest15完整重放均为14/0/1，F1=0.9655 |
| R01 | Synthetic-to-real 零样本迁移 | DONE | 合成训练 checkpoint 在独立实拍集评估 | 修正GT独立50张：unfiltered F1=0.3144；corrected M07三seedF1=0.8247/0.7961/0.8421 |
| M09 | 高风险PDF文本行候选恢复 | DONE | 修复3个candidate-stage FN且不增加最终FP | 独立50张三seed均+2TP/0FP；latest15三seed均15/0/0 |
| M10 | 按case分组的真实困难样本适配 | DONE | 5折无case泄漏、三seed折外预测、完整replay | 负结果：15折均选择epoch 0；强制微调F1均值0.7964，低于M09的0.8445 |
| R02 | Few-shot real adaptation | TODO | synthetic pretrain + 少量 real 微调对照 | 需要独立 real train/val/test |

### 0.1 代码与数据放置约定

```text
/home/jnu/projects/dataset
  负责 clean/different、文字/图形变异和程序化 GT

/home/jnu/projects/gree-label-detection
  scripts/build_clip_pair_dataset.py
  results/clip_pair_datasets/<version>/
  负责运行真实 traditional 候选流程、导出 review pair、端到端框评估

/home/jnu/projects/aa_clip_exp
  label_pair/
  results/label_pair/<experiment>/
  负责 CLIP pair 模型、训练、候选级评估和 checkpoint
```

### 0.2 每阶段必须记录的内容

每个实验完成后必须在本文档末尾的“执行日志”追加：

1. 日期、实验 ID 和 Git/代码状态；
2. 输入 dataset version 和 split；
3. 完整命令和配置；
4. 样本数、正负分布、模板分布；
5. candidate-level 与 end-to-end 指标；
6. checkpoint、manifest、summary 和可视化路径；
7. 是否达到验收条件；
8. 失败原因和下一步，不用成功结果覆盖失败记录。

### 0.3 固定执行顺序

以下顺序用于防止同时改数据、模型和评估后无法归因；只有上一阶段验收通过才进入下一
阶段：

1. `D01 + M01`：打通真实主流程 pair 导出和冻结 B/32 global head（已完成 pilot）；
2. `D02`：扩充 clean/same、困难文字变化和成像增强，使 cosine 不再轻易满分；
3. 重跑 `M01`，确认 global pair head 是否在独立 template test 上稳定优于 cosine；
4. `M02`：在同一 split 对比 B/32、L/14、L/14@336，模型规模只改一个变量；
5. `M03`：加入多层对应 patch、candidate-aware/top-k pooling；
6. `M04`：只在合成数据上加入粗到细 mutation mask 辅助损失；
7. `M05/M06`：依次做 adapter 和文字分支消融；
8. `R01/R02`：冻结选择规则后做真实实拍 zero-shot 和 few-shot，最终仍以框 TP/FP/FN
   评价。

### 0.4 术语说明（首次统一定义）

- **closed-template specialist（固定模板专用模型）**：只服务企业已经确定的少量PDF模板，
  允许这些模板参与训练；目标不是泛化到从未见过的新模板。
- **trigger candidate（触发候选框）**：traditional diff发现局部像素差异后产生的小框，
  表示“这里可能有错”，尚未经过pair模型确认，也不是最终显示框。
- **candidate-stage FN（候选阶段漏检）**：人工GT中确实有错误，但traditional diff没有生成
  与该GT匹配的trigger candidate。由于pair模型只能筛选已有候选，这类漏检无法靠调pair
  模型阈值恢复，必须修改候选生成或召回逻辑。
- **review crop（复核裁剪区域）**：围绕trigger candidate构造、送给VLM或CLIP pair模型的
  模板/目标成对局部图。对于文字候选，通常扩展到PDF预设文本行以提供上下文。
- **hard example（困难样本）**：当前模型容易判断错、但对实际业务重要的样本，例如极小
  上标变化、`I/1`混淆、括号后缀变化，以及OCR边界产生的伪字符。
- **hard-example adaptation（困难样本适配）**：从synthetic checkpoint初始化，用少量真实
  困难正负样本做低学习率微调或校准。必须按真实case分组切分，不能把同一张照片的不同
  candidate随机分到train/val/test，否则会造成数据泄漏。
- **replay（决策重放）**：把缓存的pair keep/discard预测按原请求顺序送回traditional完整
  流程，让原代码执行PDF预设框映射、display-box refine和merge；最终指标必须来自replay
  后的显示框，不能直接用trigger candidate框。
- **line-level GT（行级GT）**：文字错误以所属PDF预设文本行作为检测GT，并与精确mutation
  框取并集以包含独立上标；精确字符框只作为变异证据保留。
- **seed（随机种子）**：控制head初始化、batch顺序等训练随机性的整数。同一配置报告多个
  seed，用于观察结果方差，不能只挑最好的一次。
- **out-of-fold / OOF（折外预测）**：每个case只由没有用该case训练或选阈值的fold模型
  预测一次，再合并全部fold的test预测。它可估计开发集交叉验证表现，但5个fold的模型不
  是一个可直接部署的单一checkpoint。

## 1. 研究背景与任务定义

桌面端默认流程为：

```text
模板 PDF/图片 + 实拍图
  -> 预处理和透视校正
  -> 特征匹配和 ECC 对齐
  -> tolerant diff
  -> standard / small-text 候选生成
  -> review region 构造与合并
  -> VLM 判 keep/discard
  -> display box refine/merge
  -> 最终通过/不一致 verdict
```

本研究不替换整条检测流程，只研究最后的候选确认器：给定同一位置、已经对齐的模板
和目标 review crop，判断二者是真实内容差异（`keep`）还是成像、打印、对齐等造成的
误报（`discard`）。

该任务与 AA-CLIP 的单图异常检测不同：

| 维度 | AA-CLIP | 本项目 |
| --- | --- | --- |
| 输入 | 单张待测图片 | 模板与目标成对 crop |
| 参考 | normal/anomaly 概念 | 明确的模板图像 |
| 目标 | 判断单图是否异常并定位 | 判断同一位置是否发生内容变化 |
| 监督 | 图像标签与异常 mask | same/different 标签、mutation mask、最终框 GT |
| 评估 | image/pixel AUC、AP | 候选级 PR、最终 TP/FP/FN、标签级 verdict |

本项目更准确的任务名称是：

> **Reference-conditioned pairwise label verification**  
> 基于参考模板的成对细粒度标签差异确认

研究重点不是照搬 AA-CLIP，而是吸收其文本锚点、多层 patch 特征、残差 adapter 和
局部 mask 监督思想，重新设计适合模板—目标成对比较的模型。

## 2. 当前裸 CLIP 基线及其局限

当前端到端实验使用冻结的 `ViT-B-32-quickgelu/openai`：

```text
template_crop -> CLIP image encoder -> f_template
target_crop   -> CLIP image encoder -> f_target

diff_score = 1 - cosine(f_template, f_target)
diff_score >= 0.25 -> keep
diff_score <  0.25 -> discard
```

在 latest15、人工 GT `overlap >= 0.3` 下：

| 方法 | TP | FP | FN | Precision | Recall | F1 | 平均耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VLM | 13 | 5 | 2 | 0.7222 | 0.8667 | 0.7879 | 23.140s |
| 裸 CLIP cosine | 3 | 4 | 12 | 0.4286 | 0.2000 | 0.2727 | 0.793s |

候选和输入已验证一致：15/15 个 case 的候选框一致，69/69 个 `review_box` 一致。
因此首轮结果说明的是判别器能力差异，而不是主流程或 crop 不一致。

裸 CLIP 存在三个信息瓶颈：

1. 整个 crop 被压缩为一个全局向量，小字符变化容易被大量相同内容淹没。
2. cosine 将数百维特征直接压成一个标量，无法学习哪些维度变化代表真实错误。
3. 没有显式利用已对齐 patch 和 candidate 位置，不知道局部哪个字符笔画发生了变化。

当前结果只能证明“冻结 CLIP + 全局 embedding + 单一 cosine 阈值”不能直接替代
VLM，不能证明监督训练后的 CLIP pair 模型无效。

## 3. 研究问题

### RQ1：Pair learning 是否优于裸 CLIP cosine？

CLIP 特征中可能已经包含细粒度差异信息，只是单一 cosine 没有充分利用。监督式
pair classifier 可以验证 CLIP 表征是否具有可学习的区分信息。

### RQ2：Patch correspondence 能否解决局部字符变化？

模板与目标 crop 已对齐，可以直接比较对应位置的 patch token。需要验证局部 patch
差异是否能提升字符、数字、单位、上标和型号后缀变化的 Recall。

### RQ3：文字侧信息是否有效？

需要分别验证 generic same/different prompt、PDF 模板真实文字和 OCR/字符级特征的
增益，而不是笼统地认为“加入文本一定有效”。

### RQ4：Adapter 和模型规模的作用是什么？

需要区分性能提升来自更高分辨率、更小 patch、监督 pair head，还是 residual adapter
和 backbone 微调。

## 4. CLIP 特征的利用方式

### 4.1 全局 pair classifier

冻结共享 CLIP encoder，分别提取全局特征：

```text
f_t = CLIP(template_crop)
f_g = CLIP(target_crop)
```

构造成对特征，而不是只计算 cosine：

```text
z_global = [
    f_t,
    f_g,
    |f_t - f_g|,
    f_t * f_g,
    cosine(f_t, f_g)
]
```

然后训练小型 MLP：

```text
z_global -> Linear -> GELU -> Dropout -> Linear -> keep probability
```

- `|f_t-f_g|` 表示每个特征维度的变化强度；
- `f_t*f_g` 表示每个维度的一致性；
- MLP 可以学习区分对齐残差、模糊、曝光变化与真实内容变化。

这是成本最低、应最先验证的训练方案。如果 global pair head 显著优于 cosine，可以
证明 CLIP 特征中存在差异信息，只是原来的聚合规则过弱。

### 4.2 Patch-level 成对特征

从 CLIP ViT 的多个中间层提取 patch token：

```text
template patches: T_l ∈ R^(N×D)
target patches:   G_l ∈ R^(N×D)
```

由于两张 crop 已对齐，可以比较相同空间位置的 token：

```text
D_l,i = [
    |T_l,i - G_l,i|,
    T_l,i * G_l,i,
    cosine(T_l,i, G_l,i)
]
```

这样得到 patch difference map。局部字符变化即使不足以改变全局 embedding，也可能
在对应 patch 上形成明显响应。

建议从多层特征开始，例如 ViT-L/14 的第 6、12、18、24 层。浅层包含字形、边缘和
纹理信息，深层包含结构和语义信息。各层通过 projection adapter 投影到统一维度后
融合。

Patch 特征可采用：

- candidate mask 内平均池化；
- candidate mask 内 top-k 异常 patch；
- candidate 周围上下文池化；
- attention pooling；
- 小型 Transformer 或 cross-attention；
- 多层 anomaly map 加权融合。

首版优先使用平均池化 + top-k pooling，之后再尝试 cross-attention。

### 4.3 Candidate-aware pooling

传统 CV 已经提供触发 review 的候选框。应将候选框转换到 review crop 局部坐标，再
映射到 CLIP patch 网格：

```text
review crop
┌─────────────────────────────┐
│       candidate box         │
│          ┌────┐             │
│          │    │             │
│          └────┘             │
└─────────────────────────────┘
```

模型分别提取 candidate 内部、候选周围上下文和整个 review crop 的特征。这样模型
同时知道传统 diff 在哪里触发、局部变了什么、整行或整个对象是什么。

## 5. 文字特征的三种方案

### 5.1 T1：Generic same/different 文本锚点

参考 AA-CLIP 的 normal/anomaly anchor，构建：

```text
same:
- two label regions with identical printed content
- a correctly matched template and printed label

different:
- two label regions with different printed content
- a label region containing a character substitution
- a label region with missing or extra text
```

使用 CLIP text encoder 得到 `t_same` 和 `t_different`，让视觉 pair 特征与两个文本
锚点对齐。该方案保留 CLIP 图文空间，但提示词抽象，不能指出具体字符。预期可能有
小幅帮助，难以独立解决 `I/1`、`O/0`。

### 5.2 T2：PDF 模板真实文字

若 PDF 文字层给出 `GWH18AUDXE-K6DNA1A`，可以构造：

```text
a product label line correctly displaying GWH18AUDXE-K6DNA1A
```

比较模板图和目标图分别与模板文字 embedding 的相似度，并将相似度差加入 pair
classifier。该方案可能帮助缺字、多字或明显文字变化，但 CLIP 不是 OCR 模型，对长
型号、纯数字、单位字符串的精确图文对齐能力有限，只适合作为辅助分支。

### 5.3 T3：OCR/字符级特征

分别识别模板和目标：

```text
template OCR: GWH18AUDXE-K6DNA1A
target OCR:   GWH18AUDXE-K6DNA1A/I
```

提取 edit distance、字符增加/删除/替换、混淆字符变化、OCR confidence 和字符级
encoder embedding。该方案最直接，但严格来说不再是纯 CLIP。若论文强调工业实用性，
可加入完整模型；若强调 CLIP adaptation，可作为增强版或对照。

预期能力排序为：

```text
字符级 OCR 特征
  > 模板真实文字的 CLIP embedding
  > generic same/different prompt
```

但 OCR 本身存在识别错误，最终仍应与视觉 patch 特征融合。

## 6. 推荐模型结构

```text
template crop ─┐
               ├─ Shared CLIP ViT ─┬─ Global pair branch ─────┐
target crop ───┘                   │                          │
                                   ├─ Patch difference branch ├─ Fusion MLP
candidate mask ────────────────────┘                          │
                                                              ├─ keep/discard
template text / prompts ─ Text encoder ─ Text branch ─────────┘
```

- Global branch 使用 `[CLS_t, CLS_g, |CLS_t-CLS_g|, CLS_t*CLS_g, cosine]`。
- Patch branch 使用多层对应 patch 差异、projection adapter、candidate-aware pooling
  和 top-k pooling，并可输出 anomaly map。
- Text branch 按“无文字 → generic anchor → PDF 真实文字 → OCR 字符特征”依次消融。
- Fusion 首版使用 `concat -> 2-layer MLP -> keep probability`，验证有效后再尝试
  cross-attention。

## 7. Adapter 设计

AA-CLIP 使用残差 adapter：

```text
x_adapted = w * Adapter(x) + (1-w) * x
```

它冻结 CLIP backbone，只训练少量参数，可降低小数据集上的过拟合和灾难性遗忘风险。

### 7.1 Feature adapter

不修改 CLIP block，直接适配输出 token：

```text
patch_token -> Linear(D, 256) -> GELU -> Linear(256, D)
```

这是第一版推荐方案，简单且稳定。

### 7.2 In-block residual adapter

在 ViT 最后若干 Transformer block 中插入：

```text
x_new = x + α Adapter(x)
```

冻结原 backbone，只更新 adapter。该方案第二阶段再做。AA-CLIP 当前 `AdaptedCLIP` 的
部分通道维度和 24 层结构针对 ViT-L/14 写死，不能直接无修改用于 ViT-B 系列。

## 8. 训练数据构造

训练单位必须是主流程实际产生的 review candidate，而不是任意随机裁剪。每个样本应
保存：

```text
template_crop
target_crop
pair_label: same/different
mutation_mask: 真实修改像素区域
candidate_mask: 传统 CV 候选位置
review_box / candidate_box
kind: text/graphic
template_text: 可选 PDF 文字
```

### 8.1 正样本 different

覆盖单字符替换、字符增加/删除、数字和大小写变化、单位和上下标变化、型号前后缀
变化，以及图标、logo、条码、符号替换或缺失。

重点建立 hard text mutation：

```text
I/l/1, O/0, S/5, Z/2, B/8, G/6
m³/h <-> m3/h <-> m/h
```

### 8.2 负样本 same

不能只使用完全相同的两张图。应覆盖轻微平移、旋转、缩放、ECC 残差、模糊、曝光、
阴影、JPEG 压缩、打印粗细和墨色变化、相机噪声、局部污点和边缘残留。

更重要的是：无真实变化样本也必须完整经过传统候选生成器，只训练那些确实触发 diff
candidate 的 hard negatives，保证训练分布与部署时第 6 步一致。

### 8.3 数据切分

不能随机拆 crop，应按 template code、产品型号/产品族、原始实拍图/session 和
mutation family 分组切分。同一模板或目标图的相似 crop 不得同时进入 train/test。

latest15 不应用于选模型、挑阈值和调超参数，只能作为最终真实测试集之一。15 张只够
做先导实验，正式论文需要扩展真实样本并报告 bootstrap confidence interval。

## 9. 损失函数

### 9.1 Pair classification loss

```text
L_cls = BCE/FocalLoss(pred_keep, pair_label)
```

实际 discard 候选通常更多，可使用 class-balanced BCE 或 focal loss。

### 9.2 Patch localization loss

用合成 mutation mask 监督 patch anomaly map：

```text
L_patch = BCE/Focal + Dice
```

该部分可迁移 AA-CLIP 的 focal/dice 思想，让模型不仅判断是否不同，还学习差异位置。

### 9.3 对比或排序损失

约束 same pair 分数较低、different pair 分数较高：

```text
L_rank = max(0, margin + score_same - score_different)
```

也可以尝试 supervised contrastive loss 或 triplet loss。初始总损失可设为：

```text
L = L_cls + 0.5 * L_patch + 0.1 * L_rank
```

这些权重只是初始搜索点，必须通过 validation ablation 决定。

## 10. 更大 CLIP 模型

更大的模型可能有帮助，尤其是 `ViT-L/14@336`，但主要收益不仅来自参数量，还来自
更高输入分辨率和更小 patch：

```text
ViT-B/32:
  input ≈ 224×224, patch size = 32, patch grid ≈ 7×7（约 49 个）

ViT-L/14@336:
  input = 336×336, patch size = 14, patch grid = 24×24（576 个）
```

一个小字符在 B/32 中可能不到一个完整 patch，在 L/14@336 中可能覆盖多个 patch。
但 L/14 全局 cosine 仍可能被大量相同内容淹没，因此模型规模必须与 pair learning、
patch branch 分开消融。

| 实验 | Backbone | 判别方式 | 目的 |
| --- | --- | --- | --- |
| B0 | B/32 | cosine | 当前基线 |
| B1 | L/14@336 | cosine | 只看规模和分辨率收益 |
| M1 | B/32 | global pair head | 验证监督 pair learning |
| M2 | L/14@336 | global pair head | 规模 + pair learning |
| M3 | L/14@336 | patch pair branch | 验证局部对应关系 |
| M4 | L/14@336 | patch + adapter | 验证适配收益 |
| M5 | L/14@336 | patch + adapter + text | 验证文字分支收益 |

## 11. AA-CLIP 的可迁移部分

AA-CLIP 原始流程是：学习 normal/anomaly 文本锚点，提取多层 patch 特征，生成
anomaly map，使用 mask 的 focal/dice loss，再通过 residual adapter 适配文本和视觉
空间。

适合迁移：

- 多层 patch token；
- residual adapter；
- same/different 文本锚点；
- focal + dice patch supervision；
- 冻结 backbone、分阶段训练；
- 防止灾难性遗忘的思想。

不适合直接照搬：

- 单图异常检测设定；
- MVTec/VisA 的正常/异常类别定义；
- zero-shot 跨类别评估协议；
- 单张图 anomaly map 的建模方式；
- 随机水平/垂直翻转等不适合文字标签的增强；
- 泛化“正常产品/异常产品”文本 prompt。

核心改造是：

```text
single image vs normal/anomaly text
```

变为：

```text
template-target pair
  + corresponding patch difference
  + candidate-conditioned pooling
  + optional text guidance
```

## 12. 推荐炼丹顺序

### Phase 1：冻结 backbone，训练 global pair head

目标是回答 CLIP 特征中是否存在可学习信息。初始配置：

```text
optimizer: AdamW
head learning rate: 1e-3
weight decay: 1e-4
batch size: 32~64
epochs: 20
loss: balanced BCE 或 focal loss
early stopping: validation PR-AUC / Recall
```

分别训练 B/32 和 L/14@336。如果 pair MLP 显著优于 cosine，则证明 CLIP 特征有效，
只是单一 cosine 没有利用好。

### Phase 2：加入 patch pair branch

冻结 CLIP，训练 patch projection、candidate-aware pooling、top-k pooling 和
global/patch fusion head：

```text
adapter/head learning rate: 1e-4 ~ 3e-4
loss: L_cls + 0.5 * L_patch
```

重点观察 text subset Recall，以及 anomaly map 是否聚焦真实字符变化。

### Phase 3：加入文字分支

保持其他结构不变，依次比较：

```text
无文字
generic prompt
PDF 模板真实文字
OCR/字符级特征
```

### Phase 4：训练 adapter

先冻结 backbone，只训练 adapter/head：

```text
adapter learning rate: 1e-4
head learning rate: 3e-4
```

若仍不足，再只解冻 ViT 最后 2~4 个 block：

```text
backbone learning rate: 1e-5
adapter/head learning rate: 1e-4
```

不建议一开始 full fine-tune ViT-L，容易在合成数据上过拟合并破坏预训练能力。

### Phase 5：Hard-negative mining

用模型重新扫描训练集，收集高置信度 FP、低置信度真实差异，以及易被当成字符变化的
污点、边缘和对齐残差，加入下一轮训练。每个关键配置至少运行 3 个随机种子，报告
均值和标准差。

## 13. 论文实验组织

### 13.1 Baselines

- 传统 pixel/diff 统计和 SSIM；
- frozen CLIP cosine；
- 更大 CLIP cosine；
- VLM；
- OCR hybrid；
- AA-CLIP 风格单图 anomaly baseline；
- 本文 pairwise 模型。

### 13.2 消融实验

RQ1，Pair learning：

```text
pixel/SSIM -> CLIP cosine -> CLIP global pair MLP
```

RQ2，Patch correspondence：

```text
global only
patch only
global + patch
global + candidate-aware patch
```

RQ3，文字特征：

```text
no text
generic text anchors
template text prompt
OCR/character features
```

RQ4，规模与适配：

```text
B/32, B/16, L/14, L/14@336
frozen, feature adapter, in-block adapter, last-block fine-tune
```

### 13.3 数据和域迁移

- 合成训练、合成测试；
- 合成训练、真实测试；
- 加入少量真实训练数据；
- 跨 template code/product family 泛化；
- 不同成像噪声和对齐误差强度的鲁棒性。

### 13.4 指标

候选级：PR-AUC、ROC-AUC、固定高 Recall 下的 Precision、confusion matrix，以及
text/graphic 分组指标。

端到端：固定 `overlap >= 0.3` 的最终 TP/FP/FN、Precision、Recall、F1、标签级
通过/不一致、候选数、保留框数、单图耗时、显存和参数量。

阈值必须在 validation set 选择，test set 只评估一次。最终结果应报告 bootstrap
confidence interval。

## 14. 第一条推荐实现路线

如果只选择一条最值得优先实现的路线：

```text
ViT-L/14@336 frozen backbone
  + global pair head
  + multi-layer corresponding patch difference
  + candidate-aware top-k pooling
  + BCE/Focal pair classification loss
  + synthetic mutation mask Dice loss
```

第一版暂不加入 text encoder，也不插入复杂 in-block adapter。原因是：

1. 先确认视觉 pair learning 和 patch correspondence 能解决多少问题；
2. 文字分支可作为清晰的后续消融；
3. 同时加入 text、OCR、adapter 和大模型将无法判断提升来自哪个模块；
4. 该路线与 AA-CLIP 有理论联系，又对 paired reference task 做了实质改造。

实际开发先用 B/32 完成数据管线和 global pair head 冒烟，再切换 L/14@336 完成正式
patch 模型，避免训练代码尚未稳定时浪费大模型算力。

## 15. 潜在论文贡献

1. 将工业标签检查定义为 reference-conditioned pairwise anomaly verification，而非
   传统单图异常检测。
2. 提出对应位置的多层 CLIP patch difference 和 candidate-conditioned pooling，增强
   微小字符及图形变化检测。
3. 系统研究 generic prompt、真实模板文字和字符级特征在标签验证中的作用。
4. 构建贯穿合成数据、真实拍照、传统候选生成和人工 GT 的端到端评估协议。
5. 分析模型规模、adapter、文本信息和 synthetic-to-real domain gap 的影响。

## 16. 当前结论与下一步

当前端到端实验已经保证 CLIP 与 VLM 使用相同候选、相同 review region 和相同最终
评估。裸 CLIP 获得明显速度收益，但 Recall 很低，说明直接阈值替换不可行。

下一步不应继续只调 cosine 阈值，而应：

1. 从主流程导出标准化 pair training manifest；
2. 建立按 template/product/session 分组的数据切分；
3. 先训练冻结 backbone 的 global pair MLP；
4. 再加入 patch difference 和 candidate-aware pooling；
5. 最后按消融顺序加入 text branch、adapter 和更深层微调。

这条路线既保留工程任务的真实输入与评价方式，也形成了区别于 AA-CLIP 单图异常检测
的独立研究问题。

## 17. 执行日志

### 2026-07-10 / E00 / DONE

- 目标：验证在完全相同的 traditional 主流程输入下，用冻结 CLIP cosine 替换 VLM。
- 模型：`ViT-B-32-quickgelu/openai`，`diff_score >= 0.25` 判 keep。
- 数据：latest15 人工 GT。
- 一致性：15/15 case 的候选一致，69/69 review box 一致。
- 结果：VLM `P/R/F1=0.7222/0.8667/0.7879`；CLIP
  `P/R/F1=0.4286/0.2000/0.2727`。
- 耗时：VLM 23.140s/图；CLIP 0.793s/图。
- 产物：
  - `results/batch_desktop_captures/latest15_clip_vitb32_t025_20260710`
  - `results/evaluation/latest15_vlm_vs_clip_t025_20260710`
- 结论：工程替换方式正确；裸全局 cosine Recall 不足，进入监督 pair learning。

### 2026-07-10 / D01 / DONE

- 目标：把 dataset 项目带 GT 的渲染样本完整送入 traditional 候选流程，导出训练用
  review pair manifest。
- 新增代码：`scripts/build_clip_pair_dataset.py`；测试：
  `tests/test_build_clip_pair_dataset.py`。
- 输入：
  `results/dataset_closed_loop/actual_20260626-200730/evaluation_dataset`，10 个 text-only
  defect case、5 个 template，每个 template 2 个 case。
- 标签：候选框映射回原图后，使用唯一规则
  `intersection / min(candidate_area, gt_area) >= 0.3`。review box 只决定模型输入，不参与
  0/1 标签，避免大 crop 碰到 GT 造成假阳性。
- 切分：按 `template_id` 分组，seed=`20260710`；train/val/test 的 template 数为 3/1/1，
  无 template 泄漏。
- 运行命令：

```bash
cd /home/jnu/projects/gree-label-detection
.venv/bin/python scripts/build_clip_pair_dataset.py \
  --output-dir results/clip_pair_datasets/synthetic_pilot_v1 \
  --overwrite
```

- 结果：共 43 pair，14 正、29 负；train=`25 (8+/17-)`，val=`10 (3+/7-)`，
  test=`8 (3+/5-)`。
- 每条记录包含 exact template/target VLM review crop、candidate/review 的 panel/final 坐标、
  candidate-GT overlap、mutation metadata、candidate mask 和粗 GT bbox mask。
- 产物：
  `results/clip_pair_datasets/synthetic_pilot_v1/{manifest.jsonl,summary.json,splits/,pairs/,masks/}`。
- 验证：3 个坐标/overlap/split 单元测试通过；43/43 pair 的图片与 mask 路径存在；每个
  template 只属于一个 split。
- 尚未完成：clean/same 源样本、精确 changed-pixel mutation mask、camera-domain
  augmentation。这三项归入 D02/M04，不能把当前粗 bbox mask 写成分割 GT。

### 2026-07-10 / M01 / DONE（仅 pilot 验收）

- 目标：验证冻结 CLIP 图像特征可以接监督 pair classifier，并形成可复现 checkpoint。
- 新增代码：
  `/home/jnu/projects/aa_clip_exp/label_pair/{dataset.py,model.py,train_global.py}`。
- 模型：冻结 `ViT-B-32-quickgelu/openai`；pair feature 为
  `[f_template, f_target, abs(diff), product, cosine]`，共 2049 维；两层 MLP、balanced
  BCE，仅训练 head。
- 配置：CPU，seed=`20260710`，AdamW，lr=`1e-3`，weight decay=`1e-4`，batch=16，
  epochs=30；阈值只在 val 上按最高 F1 选择。
- 运行命令：

```bash
cd /home/jnu/projects/aa_clip_exp
uv run --no-sync python -m label_pair.train_global \
  --train-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_pilot_v1/splits/train.jsonl \
  --val-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_pilot_v1/splits/val.jsonl \
  --test-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_pilot_v1/splits/test.jsonl \
  --output-dir results/label_pair/synthetic_pilot_v1_b32_global \
  --epochs 30 --batch-size 16
```

- pair head：val `TP/FP/FN=3/0/0`，test `3/0/0`，val/test F1 均为 1.0；test
  ROC-AUC/PR-AUC 均为 1.0。
- 相同特征、相同 split 的裸 cosine baseline：val/test 也均为 `3/0/0`、F1=1.0。
- 产物：
  `/home/jnu/projects/aa_clip_exp/results/label_pair/synthetic_pilot_v1_b32_global/`，包括
  `best_checkpoint.pt`、`summary.json`、`history.json`、`predictions.json`。
- 验证：AA-CLIP 环境中模块编译、2049 维 feature/head 冒烟、checkpoint/summary
  加载检查均通过。
- 结论：M01 的工程验收完成，但研究假设尚未得到验证。8 个 test pair 太少，而且简单
  渲染变化使 cosine 已满分；不能据此声称 pair head 优于 cosine，也不能进入论文表格。
- 下一步：先做 D02，把 clean/same、成像扰动和困难字符变化完整送入主流程形成 hard
  negatives，再用冻结 split 重跑 M01；之后才值得跑 L/14@336。

### 2026-07-10 / D02-v1、v2 / SUPERSEDED

- 新增 dataset 项目代码：
  `/home/jnu/projects/dataset/pdf_svg_pipeline/visual_pair_augmentation.py` 和
  `/home/jnu/projects/dataset/scripts/augment_visual_pair_dataset.py`。
- 语义保持增强包括 `soft_capture`、`print_scan`、`shadow`、`geometry`、`mixed`：轻微
  旋转/平移、笔画粗细、模糊、亮度/对比度、阴影、传感器噪声和 JPEG 退化。
- v1 同时增强 defect 和 clean，70 case 导出 549 pair（54 正/495 负）；test 上 pair head
  `TP/FP/FN=6/17/3，F1=0.3750`，cosine `9/139/0，F1=0.1146`。
- v1 只作为 provisional：defect 几何/笔画增强会改变预处理的黑框和透视分支，虽然
  alignment 最终回到模板坐标，但增强 GT 尚未逐例抽样审核，不能并入正式训练结果。
- v2 改为原始 defect + 增强 clean，避免增强正例 GT 风险；但复现性检查发现 Pillow
  `effect_noise` 不接受外部 seed，因此 v2 数据像素不能严格复现。
- 处理：v1/v2 产物均保留作问题追踪，不作为当前正式 pilot；噪声实现改为显式 seed
  驱动的低分辨率高斯噪声，再进入 v3。

### 2026-07-10 / D02-v3 / DONE

- 数据生成命令：

```bash
cd /home/jnu/projects/dataset
uv run --no-sync python scripts/augment_visual_pair_dataset.py \
  --input-dir /home/jnu/projects/gree-label-detection/results/dataset_closed_loop/actual_20260626-200730/evaluation_dataset \
  --output-dir outputs/clip_pair_d02_v3_clean_safe \
  --defect-variants 0 --clean-variants 6 --severity 1.0 --overwrite
```

- 输入仍为 10 个原始 defect case、5 个 template；新增 30 个 clean/same case，每个
  template 覆盖 6 个增强变体。总计 40 case。
- clean/same 使用模板真实内容作为 target，仅改变成像，不含语义错误，
  `change_regions=[]`；之后完整运行默认 traditional 候选流程，不直接随机裁 crop。
- pair 导出命令：

```bash
cd /home/jnu/projects/gree-label-detection
.venv/bin/python scripts/build_clip_pair_dataset.py \
  --dataset-root /home/jnu/projects/dataset/outputs/clip_pair_d02_v3_clean_safe \
  --output-dir results/clip_pair_datasets/synthetic_d02_v3_clean_safe \
  --overwrite
```

- 30/30 clean case 都真实触发至少一个候选，共产生 271 个 clean hard-negative pair。
- 总计 314 pair，14 正/300 负；train=`167 (8+/159-)`，val=`50 (3+/47-)`，
  test=`97 (3+/94-)`，仍按 template 3/1/1 分组且无泄漏。
- profile 数量均衡：`soft_capture/print_scan/shadow/geometry/mixed` 各 6 case；每条 pair
  manifest 保存具体 profile、seed 和参数。
- 复现验证：相同输入、profile、seed 的两次增强像素完全一致；模块编译通过；314 条
  pair 的图片/mask/split 结构由 exporter 完整生成。
- 产物：
  - `/home/jnu/projects/dataset/outputs/clip_pair_d02_v3_clean_safe`
  - `results/clip_pair_datasets/synthetic_d02_v3_clean_safe`
- 限制：clean 是程序化成像近似，不等于打印实拍；正例仍只有原始的 14 个候选，存在
  正负域线索不完全对称的问题。下一版必须增加经过 GT 审核的增强正例和更多 mutation
  family。

### 2026-07-10 / M01-D02-v3 / DONE（pilot）

- 训练命令：

```bash
cd /home/jnu/projects/aa_clip_exp
uv run --no-sync python -m label_pair.train_global \
  --train-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/train.jsonl \
  --val-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/val.jsonl \
  --test-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/test.jsonl \
  --output-dir results/label_pair/synthetic_d02_v3_clean_safe_b32_global \
  --epochs 30 --batch-size 32
```

- 模型仍为 frozen `ViT-B-32-quickgelu/openai` + 2049 维 global pair feature + MLP；
  train `pos_weight=19.875`；阈值只在 val 按 F1 选择，为 `0.517668`。
- test 对照：

| 方法 | TP | FP | FN | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 裸 cosine | 3 | 86 | 0 | 0.0337 | 1.0000 | 0.0652 | 0.1028 | 0.0230 |
| global pair head | 2 | 0 | 1 | 1.0000 | 0.6667 | 0.8000 | 0.9965 | 0.9167 |

- clean/same test 子集为 89 个候选，pair head 全部判负；原始 text-only 子集为 8 个候选，
  其中 3 正，结果 `TP/FP/FN=2/0/1`。
- 解释：成像扰动使裸 cosine 的 diff score 普遍高于真实小字符错误，因此 cosine 排序
  方向在本数据上几乎反转；监督 pair head 能利用 embedding 各维和交互信息，而不只看
  距离大小。这是支持 pair learning 的首个有效证据。
- 仍不能写成最终论文结论：test 只有 3 个正例，正例未做同域增强，增强分布可能泄露
  “clean”线索。必须扩充并对称化正例后报告多 seed 和置信区间。
- 产物：
  `/home/jnu/projects/aa_clip_exp/results/label_pair/synthetic_d02_v3_clean_safe_b32_global/`。
- M02 状态：数据和训练接口 READY，但本机仅缓存 B/32 权重，未发现 L/14@336 权重；
  不在未记录的情况下自动下载约 GB 级模型。M03 可先用 B/32 patch token 验证代码。

### 2026-07-10 / M02 / DONE（B/32 vs L/14@336）

#### 权重与环境

- 用户已明确允许下载 L/14@336。
- OpenCLIP 的 Hugging Face 自动下载连接被重置，改用配置中 OpenAI 官方 Azure URL：
  `ViT-L-14-336px.pt`。
- 第一次断点续传文件 SHA256 不匹配，未用于实验；重新从零下载后校验为：
  `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`，与官方 URL
  中的完整 hash 一致。
- 有效权重：`/home/jnu/models/clip/ViT-L-14-336px.pt`，934,088,680 bytes；模型为
  `ViT-L-14-336-quickgelu`，427,944,193 参数，输入 336×336、全局特征 768 维。
- 原 AA-CLIP `.venv` 是 Python 3.10 + `torch 2.3.1` ARM64 CPU wheel：
  `torch.version.cuda=None`，因此之前 `--device auto` 正确回退到 CPU，并非主动禁用 GPU。
- 机器实际为 NVIDIA GB10、driver 580.142、CUDA 13.0、compute capability 12.1；另一个
  环境的 `torch 2.12.0+cu130` 已能识别 GPU。
- 为不破坏论文仓库锁定环境，新增本地隔离 `.venv-gpu`：Python 3.12、
  `torch 2.12.0+cu130`、`torchvision 0.27.0+cu130`、OpenCLIP 3.3.0。原 `.venv` 保留。
- OpenAI `.pt` 是已校验的 TorchScript archive；PyTorch 2.12 默认
  `weights_only=True` 会拒绝加载。`train_global.py` 现在只对存在的本地权重路径传
  `weights_only=False`，网络 tag 仍保持安全默认。
- GPU 上已有约 24GB 的 vLLM 进程，但 L/14@336 两图编码实际分配约 1.63GiB，完整
  实验没有 OOM，无需停止 VLM 服务。

#### 控制变量实验

两个模型使用完全相同的：

- D02-v3 train/val/test manifest；
- seed=`20260710`；
- balanced BCE、hidden=512、lr=`1e-3`、weight decay=`1e-4`；
- epochs=30、batch=32、CUDA/PyTorch 2.12；
- validation 最大 F1 阈值规则。

B/32 GPU 命令的核心参数：

```bash
cd /home/jnu/projects/aa_clip_exp
.venv-gpu/bin/python -m label_pair.train_global \
  --model ViT-B-32-quickgelu --device cuda --batch-size 32 --epochs 30 \
  --train-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/train.jsonl \
  --val-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/val.jsonl \
  --test-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/test.jsonl \
  --output-dir results/label_pair/synthetic_d02_v3_clean_safe_b32_global_gpu
```

L/14@336 仅替换：

```text
--model ViT-L-14-336-quickgelu
--pretrained /home/jnu/models/clip/ViT-L-14-336px.pt
--output-dir results/label_pair/synthetic_d02_v3_clean_safe_l14_336_global_gpu
```

test candidate-level 结果：

| Backbone | 判别器 | TP | FP | FN | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B/32 | cosine | 3 | 86 | 0 | 0.0337 | 1.0000 | 0.0652 | 0.1028 | 0.0230 |
| B/32 | global pair head | 2 | 0 | 1 | 1.0000 | 0.6667 | 0.8000 | 0.9965 | 0.9167 |
| L/14@336 | cosine | 2 | 89 | 1 | 0.0220 | 0.6667 | 0.0426 | 0.1064 | 0.0224 |
| L/14@336 | global pair head | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

- 总 wall time（含加载、全部 CLIP 编码、30 epoch head 训练和评估）：B/32 GPU
  11.47s；L/14@336 GPU 37.35s，L 约为 3.26 倍。
- CPU L/14@336 preliminary（batch=8、PyTorch 2.3）为 `TP/FP/FN=3/2/0，F1=0.75`；
  因环境和 batch 不同，不用于 backbone 主对照，只证明 CPU 路径也可运行。
- 结论一：更大模型并不会挽救单一 cosine；L/14@336 cosine 仍然失败，证明主要问题
  是判别方式，而不只是 backbone 尺寸。
- 结论二：同一 global pair head 下 L/14@336 在当前 test 恢复了 B/32 漏掉的 1 个正例，
  同时保持 0 FP，方向上支持更高分辨率和更细 patch。
- 结论边界：test 只有 3 个正例，`F1=1.0` 方差极大，不能声称 L 显著优于 B；需要
  扩大正例、运行多 seed，并在 M03 patch branch 和真实实拍集上复核。
- 产物：
  - `/home/jnu/projects/aa_clip_exp/results/label_pair/synthetic_d02_v3_clean_safe_b32_global_gpu`
  - `/home/jnu/projects/aa_clip_exp/results/label_pair/synthetic_d02_v3_clean_safe_l14_336_global_gpu`

### 2026-07-10 / M03 / DONE（Frozen multi-layer patch prototype）

#### 实现

- 新增 `label_pair/train_patch.py`，并扩展 `label_pair/dataset.py`、
  `label_pair/model.py`。
- backbone：冻结 `ViT-L-14-336-quickgelu`，从 Transformer block
  `[5, 11, 17, 23]` 提取四层对应 patch token；每层形状为 `1024×24×24`。
- template/target 使用共享 encoder，对应位置特征取绝对差：
  `abs(patch_template - patch_target)`。
- candidate bbox mask 经过与 OpenCLIP 相同的 shortest-side resize + center crop，再以
  adaptive max-pool 映射到 24×24。它只用于告诉 pooling 关注哪个候选区域，不是分割
  输出，也没有改变最终框评估。
- 每层提取四种池化：candidate masked mean、candidate masked max、global mean、
  candidate 内 top-10% change tokens mean；每个分量先独立 L2 normalize。
- patch feature=`4 layers × 4 pooling × 1024 = 16,384` 维；global feature=3,073 维；
  fusion feature=19,457 维。
- 特征缓存：
  `results/label_pair/feature_cache/synthetic_d02_v3_l14_336_multilayer_norm.pt`，约 24MB；
  三种 head/seed 共享相同冻结特征，避免重复编码和输入漂移。

#### 数值稳定性修正

- M03-v0 未对 pooled patch feature 归一化，初期 loss 达到 40–50，patch-only threshold
  饱和到 0.994。该版本即使 test 指标较高也不可信，产物保留但不作为主结果。
- M03-v1 对每层每种 pooled vector 分别 L2 normalize，初期 loss 降至约 2，阈值回到
  0.5 附近；cache schema 标记为 `component_l2_normalized_v2`，防止误加载旧缓存。

#### 运行配置

共同配置为 D02-v3 split、L/14@336、CUDA、balanced BCE、epochs=30、batch=16、
hidden=512、top-k ratio=0.1。首次运行使用 `--rebuild-cache`：

```bash
cd /home/jnu/projects/aa_clip_exp
.venv-gpu/bin/python -m label_pair.train_patch \
  --train-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/train.jsonl \
  --val-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/val.jsonl \
  --test-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d02_v3_clean_safe/splits/test.jsonl \
  --feature-cache results/label_pair/feature_cache/synthetic_d02_v3_l14_336_multilayer_norm.pt \
  --model ViT-L-14-336-quickgelu \
  --pretrained /home/jnu/models/clip/ViT-L-14-336px.pt \
  --layers 5,11,17,23 --topk-ratio 0.1 \
  --feature-mode global_patch --device cuda --epochs 30 --batch-size 16
```

之后使用同一 cache 分别令 `--feature-mode global/patch/global_patch`，运行 seeds
`20260710/20260711/20260712`。

#### 三 seed 结果

| 特征 | Val F1（三 seed） | Test F1（三 seed） | Test TP/FP/FN |
| --- | --- | --- | --- |
| global-only | 0.800 / 0.857 / 0.857 | 0.857 / 0.857 / 0.857 | 每次 3/1/0 |
| patch-only | 0.800 / 0.800 / 0.800 | 1.000 / 1.000 / 1.000 | 每次 3/0/0 |
| global+patch | 1.000 / 1.000 / 1.000 | 0.857 / 1.000 / 0.857 | 两次 3/1/0，一次 3/0/0 |

- patch-only 在三个 seed 上稳定消除了 global-only 的同一个 FP：
  `600004075219_clean_006__pair_015`。该样本内容仍为 `4.60kW`，只是 print-scan
  增强使笔画明显变细并带噪声/JPEG 纹理；global/fusion 容易把整体外观变化当错误，
  candidate-aware patch 对应比较正确判为 same。
- global+patch 在 validation 始终最好，但 test 有 1 个 FP 的 seed 方差，说明直接把
  19,457 维特征 concat 给 MLP 并不稳定；下一步应做低维 patch projection、门控融合或
  residual adapter，而不是继续增加裸特征维数。
- 当前证据支持“patch 特征含有能处理局部成像差异的信息”，但仍不能宣称最终提升：
  test 只有 3 个正例，且 D02-v3 正负域不完全对称。
- 主产物：
  - `results/label_pair/synthetic_d02_v3_l14_336_patch_only_norm`
  - `results/label_pair/synthetic_d02_v3_l14_336_global_patch_norm`
  - 同目录下带 `seed20260711/seed20260712` 的重复实验。
- 下一步：优先扩充/审核增强正例；模型侧进入 M05 的 projection/gated adapter 小消融。
  M04 mutation mask 辅助损失应在精确 changed-pixel mask 可用后再做，不能把 bbox mask
  冒充像素级真值。

### 2026-07-10 / M05 / DONE（负结果：adapter 未稳定增益）

#### 目标与实现

- 新增 `label_pair/train_adapter.py`，扩展 `label_pair/model.py`。
- 只训练 feature adapter，CLIP 和 M03 多层 patch cache 完全冻结；与 in-block adapter
  区分开，避免一次修改 backbone 和融合结构。
- `ResidualFeatureProjector`：`LayerNorm -> Linear projection`，再接 residual MLP；
  residual scale 由可学习 sigmoid 参数控制，初始化 `sigmoid(-2)≈0.12`。
- `patch_adapter`：16,384 维 patch 压缩后分类。
- `gated_fusion`：global 和 patch 分别投影到相同维数，学习
  `gate * global + (1-gate) * patch`。测试 vector gate 和 sample-level scalar gate。
- 所有实验复用 M03 的 normalized feature cache、相同 split、三个 seed、balanced BCE、
  batch=16、epochs=30。

#### M05-v1：256 维 projection

| Head | 参数量 | Val F1（三 seed） | Test F1（三 seed） |
| --- | ---: | --- | --- |
| raw patch-only（M03） | 约 839万 | 0.800 / 0.800 / 0.800 | **1.000 / 1.000 / 1.000** |
| patch adapter 256 | 436万 | 0.800 / 0.857 / 0.857 | 1.000 / 0.600 / 0.750 |
| raw global+patch（M03） | 约 996万 | 1.000 / 1.000 / 1.000 | 0.857 / 1.000 / 0.857 |
| vector-gated fusion 256 | 542万 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 0.750 |

- vector gate 的 test mean global weight 为 0.56–0.59，但维度标准差约 0.49，且 min/max
  接近 0/1；说明许多 gate 维度饱和成硬选择，在小数据上容易过拟合。
- residual adapter scale 最终仍约 0.118–0.135，接近初始化，未显示 adapter residual
  分支获得稳定作用。

#### M05-v2：64 维低容量与 scalar gate

使用 projection=64、lr=`3e-4`；scalar gate 首先使用正则
`0.01 * mean((gate-0.5)^2)`：

| Head | 参数量 | Val F1（三 seed） | Test F1（三 seed） |
| --- | ---: | --- | --- |
| patch adapter 64 | 109万 | 0.800 / 0.800 / 0.800 | 1.000 / 0.857 / 1.000 |
| scalar-gated fusion 64, reg=0.01 | 130万 | 0.857 / 1.000 / 1.000 | 1.000 / 0.857 / 1.000 |
| scalar-gated fusion 64, reg=1.0 | 130万 | 0.857 / 1.000 / 1.000 | 0.750 / 0.600 / 1.000 |

- 降维显著减少参数并改善 v1 方差，但仍没有超过 raw patch-only。
- `reg=0.01` 的 scalar gate 在两个 seed 塌缩到接近 100% global；`reg=1.0` 仍偏向
  global（test mean global weight 0.85–0.98）且指标下降，说明简单二次正则不能解决融合
  可辨识性问题。
- 所有配置 test ROC-AUC 仍接近或等于 1，但 validation 阈值落到 test 后产生不同数量
  FP；主要瓶颈仍是只有 3 个 test positive 和极少 validation positive，阈值/校准方差
  大于模型排序差异。

#### 结论

- M05 验收完成，但结果是否定性的：当前数据下 residual feature adapter、低维
  projection、vector/scalar gate 都没有稳定优于 M03 raw normalized patch-only。
- 当前推荐 checkpoint 仍是 M03 patch-only，而不是挑选某个恰好 F1=1 的 gated seed。
- 不继续做更复杂 in-block adapter；先扩充正例、做同域正例增强并提高 validation/test
  正例数量。数据充足后再重新判断 adapter。
- 产物位于 `results/label_pair/` 下：
  - `synthetic_d02_v3_l14_336_patch_adapter_seed*`
  - `synthetic_d02_v3_l14_336_gated_fusion_seed*`
  - `synthetic_d02_v3_l14_336_patch_adapter64_seed*`
  - `synthetic_d02_v3_l14_336_scalar_gate64_seed*`
  - `synthetic_d02_v3_l14_336_scalar_gate64_reg1_seed*`

### 2026-07-10 / D03 + M03 rerun / DONE（正负同域安全增强）

#### 动机与数据

- D02-v3 的14个正例只来自原始渲染，而大量负例来自成像增强，可能让模型利用域线索。
- 扩展 dataset 项目的 `visual_pair_augmentation.py`，支持 defect/clean 独立 profile
  白名单。
- defect 只允许 `soft_capture` 和 `shadow`：只改变亮度、对比度、模糊、传感器噪声、
  阴影和 JPEG，不做旋转、平移、笔画 morphology，保持原 GT 坐标不变。
- clean 继续使用五种 profile，覆盖完整 hard-negative 分布。
- 数据生成命令：

```bash
cd /home/jnu/projects/dataset
uv run --no-sync python scripts/augment_visual_pair_dataset.py \
  --input-dir /home/jnu/projects/gree-label-detection/results/dataset_closed_loop/actual_20260626-200730/evaluation_dataset \
  --output-dir outputs/clip_pair_d03_symmetric_safe \
  --defect-variants 4 --clean-variants 6 \
  --defect-profiles soft_capture,shadow \
  --clean-profiles soft_capture,print_scan,shadow,geometry,mixed \
  --severity 1.0 --overwrite
```

- 生成80 case：10原始 defect、40增强 defect、30 clean/same。
- 验证：80/80 图像尺寸与对应模板一致；增强 defect 的 `change_regions` 与源 mutation
  GT 逐字段完全相同；defect profile 只出现 original/soft_capture/shadow。
- 经默认 traditional 主流程导出504 pair：71正、433负；train=`41+/230-`，
  val=`15+/86-`，test=`15+/117-`。50/50 defect case 和30/30 clean case 都触发候选。
- 产物：
  - `/home/jnu/projects/dataset/outputs/clip_pair_d03_symmetric_safe`
  - `results/clip_pair_datasets/synthetic_d03_symmetric_safe`

#### L/14@336 global vs patch 重跑

- 新 feature cache：
  `/home/jnu/projects/aa_clip_exp/results/label_pair/feature_cache/synthetic_d03_l14_336_multilayer_norm.pt`
  （约38MB）。
- 使用 M03 normalized patch 定义、layers `[5,11,17,23]`、batch=16、epochs=30，
  seeds=`20260710/20260711/20260712`。
- 首次 `--rebuild-cache` 会创建 CLIP 模型并消耗 RNG；为严格控制 head seed，最终表中
  三个 seed 都采用“加载同一 cache 后再初始化 head”的路径，首次边提取边训练的结果
  不纳入对照。

| 方法 | Val F1（三 seed） | Test Precision | Test Recall | Test F1 | Test TP/FP/FN |
| --- | --- | --- | --- | --- | --- |
| L/14@336 cosine | 0.3117 | 0.0886 | 0.4667 | 0.1489 | 7/72/8 |
| global head | 0.645/0.625/0.667 | 0.484/0.441/0.600 | 1.000/1.000/1.000 | 0.652/0.612/0.750 | 15/16/0、15/19/0、15/10/0 |
| patch-only | 0.696/0.696/0.696 | 1.000/0.929/1.000 | 0.933/0.867/0.867 | **0.966/0.897/0.929** | 14/0/1、13/1/2、13/0/2 |

- 三 seed 平均 Test F1：global=`0.6715`，patch=`0.9302`。
- global 的主要问题是成像误报：保持全部15个正例，但产生10–19个 FP。
- patch 的主要取舍是高 Precision、略低 Recall：FP降为0–1，FN为1–2。这比 D02
  只有3个正例时更清楚地支持 candidate-aware local correspondence。

#### Mutation/增强分组

- `decimal_shift: 5.20kW -> 52.0kW`：patch 三 seed 都为5/5。
- 第一组 confusable mutation：三 seed 都为5/5。
- `Sound Pressure Level(H) -> Sound Pres5ure Level(H)`：三 seed 为4/5、3/5、3/5；
  漏检集中在更强的 soft-capture/shadow 版本，而不是随机分散。
- 因此下一模型问题已具体化为：长文本中的单字符变化叠加成像噪声。文字分支或更细
  candidate-conditioned patch aggregation应优先针对该类，而不是泛化地增加模型容量。

#### 结论边界

- test 有15个 positive pair，但只是3个独立 mutation 各自的成像重复，不能把它们当
  15个独立错误计算统计显著性；有效独立样本量仍很小。
- D03 证明 patch 对同一语义错误的成像鲁棒性优于 global，但论文正式实验仍需生成更多
  独立 mutation family/template，并用实拍人工 GT 复核。
- 当前推荐仍为 frozen L/14@336 normalized patch-only；M05 adapter 负结论不变。

### 2026-07-10 / D04 + M03 rerun / DONE（Diverse 独立 mutation）

#### 数据生成

- 扩展 `/home/jnu/projects/dataset/pdf_svg_pipeline/dataset_builder.py`，新增向后兼容的
  `--mutation-profile diverse`；原默认仍为 `doubtful`，不改变已有闭环行为。
- `diverse` 使用 `strategy=auto`，允许 confusable、decimal_shift、superscript、
  substitute、delete、duplicate、swap、insert、case。
- 生成命令：

```bash
cd /home/jnu/projects/dataset
uv run --no-sync python scripts/build_dataset.py \
  --source-dir inputs/pdfs \
  --output-dir outputs/datasets/clip_pair_d04_diverse \
  --text-only-count 6 --mutation-profile diverse --seed 20260710
```

- 5个模板各6个样本，共30个独立 mutation sample、40处程序化 GT。
- 实际策略分布：confusable=18、insert=5、substitute=5、delete=4、case=4、
  duplicate=3、superscript=1。本 seed 未抽到 decimal_shift/swap，不能把“允许”误写成
  “已经覆盖”。
- 转换为 visual manifest 后，每个 defect 增加1个 soft-capture/shadow 安全变体；每模板
  生成6个 clean，得到90 case（60 defect、30 clean）。增强 defect 的 GT 和尺寸不变量
  验证通过。
- 产物：
  - `/home/jnu/projects/dataset/outputs/datasets/clip_pair_d04_diverse`
  - `/home/jnu/projects/dataset/outputs/clip_pair_d04_diverse_symmetric`
  - `results/dataset_project_eval_dataset/clip_pair_d04_diverse`

#### 默认主流程候选覆盖

- 导出534 pair：75正、459负；train=`43+/258-`，val=`15+/75-`，
  test=`17+/126-`。
- 在 pair 标签之前单独核查40个独立 GT 是否至少被传统候选覆盖：36/40，Recall=90%。
- 分策略候选覆盖：case 4/4、confusable 16/18、delete 4/4、duplicate 3/3、insert 5/5、
  substitute 3/5、superscript 1/1。
- 候选阶段漏检4个：`I->O`、`I->l`、`I->L`、`3->8`。这些错误未进入 review pair，
  因此不能归咎于 CLIP，也无法由后端确认器补救。
- pair 产物：`results/clip_pair_datasets/synthetic_d04_diverse_symmetric`。

#### L/14@336 global vs patch 三 seed

- cache：
  `/home/jnu/projects/aa_clip_exp/results/label_pair/feature_cache/synthetic_d04_l14_336_multilayer_norm.pt`。
- 仍使用 `[5,11,17,23]` 四层、normalized pooling、batch=16、epochs=30；所有 seed
  从同一 cache 初始化 head。

| 方法 | Val F1（三 seed） | Test Precision | Test Recall | Test F1 | Test TP/FP/FN |
| --- | --- | --- | --- | --- | --- |
| cosine | 0.3125 | 0.1136 | 0.8824 | 0.2013 | 15/117/2 |
| global head | 0.875/0.828/0.875 | 0.433/0.500/0.333 | 0.765/0.647/0.647 | 0.553/0.564/0.440 | 13/17/4、11/11/6、11/22/6 |
| patch-only | 0.882/0.857/0.857 | 0.667/0.583/0.583 | 0.824/0.824/0.824 | **0.737/0.683/0.683** | 14/7/3、14/10/3、14/10/3 |

- 三 seed 平均 Test F1：global=`0.5191`，patch=`0.7009`。patch 仍稳定优于 global，
  但明显低于 D03 的0.9302，证明 D03 的高分受少量底层错误重复增强影响。
- patch 三 seed TP/FN 完全一致，差异来自 validation 阈值导致 FP=7/10/10；模型排序
  ROC-AUC约0.91、PR-AUC约0.83，尚有明显提升空间。

#### Test mutation family 分解

- test 有17个 positive pair，对应8个独立 source mutation。
- patch 三 seed：confusable 4/4、delete 2/2、duplicate 2/2、insert 4/5、
  substitute 2/4。
- 稳定完全漏掉 `46dB(A) -> 46dB(C)` 的原始和 shadow 两个 pair；另有一个 seed/增强
  相关漏检来自 `850m/h -> 850m/hc`。
- FP 主要来自 shadow、geometry、mixed，少量 print-scan；说明 clean domain 的困难
  成像误报仍未完全解决。

#### 结论与下一步

- D04 是当前比 D03 更可信的 synthetic pilot：独立 test mutation 从3增至8，且策略更加
  多样。当前不能再引用 D03 的近满分作为模型能力代表，应以 D04 三 seed 为主要结果。
- 工作被明确拆成两个问题：
  1. traditional 候选召回：36/40，优先修复极小单字符 `I/l/1/O/L`、`3/8`；
  2. pair 确认：patch 优于 global，但对长 review line 内 `A/C` 和尾部插入字符仍弱。
- 下一模型实验应针对第二点加入真实模板文字/OCR字符信息；与此同时数据侧继续补齐
  decimal_shift、swap 和更多跨模板独立 mutation。

### 2026-07-10 / M06-A / DONE（CLIP generic/PDF text 为负增益）

#### 文字覆盖与设计

- D04 pair 的 PDF 文字覆盖：train 259/301、val 72/90、test 104/143；所有正例
  train 43/43、val 15/15、test 17/17 均有匹配模板行和片段。
- 新增 `label_pair/build_text_cache.py`，使用同一冻结 L/14@336 text encoder 编码68个
  唯一 PDF 片段/行；cache：
  `results/label_pair/feature_cache/synthetic_d04_l14_336_text.pt`。
- generic direction 使用两个固定 prompt 的归一化差向量：
  - `two printed product label regions have identical text content`
  - `two printed product label regions have different text content`
- 视觉—文字交互包含 text、signed/absolute visual difference×text、template/target×text、
  两侧 cosine 和 has-text，共3,844维。
- 为排除参数量影响，三组输入维度完全相同，均为16,384维 patch + 3,844维 text
  interaction =20,228维：
  - `patch_text_none`：interaction 全零；
  - `patch_text_generic`：所有样本使用 generic direction；
  - `patch_text_pdf`：使用实际 PDF segment/line，缺失候选为零。
- 新增代码位于 `label_pair/model.py` 和 `label_pair/train_patch.py`；三个模式使用同一
  visual/text cache、batch=16、epochs=30、三个 seed。

#### 三 seed 结果

| 特征 | Val F1（三 seed） | Test F1（三 seed） | Test F1 均值 |
| --- | --- | --- | ---: |
| raw patch-only（16,384维） | 0.882/0.857/0.857 | 0.737/0.683/0.683 | 0.7009 |
| patch + zero text control | 0.857/0.882/0.889 | 0.750/0.683/0.741 | 0.7246 |
| patch + generic CLIP text | 0.929/0.929/0.897 | 0.643/0.640/0.690 | 0.6575 |
| patch + PDF CLIP text | 0.750/0.690/0.692 | 0.480/0.563/0.533 | 0.5253 |

- zero-text 同维度控制与 raw patch 的差异属于 head 初始化/额外零权重参数方差，不能
  解释成文字收益。
- generic prompt validation 很高但 test 下降，显示跨模板校准/捷径问题。
- PDF text 明显降低 Recall，三个 seed 均劣于 zero control；真实模板字符串输入本身不
  等于字符比较能力。

#### Mutation family

- raw/zero control 对 substitute 都为2/4，包含稳定漏检的 `46dB(A)->46dB(C)`；generic
  仍为2/4，PDF 为1/4或2/4，没有修复 A/C。
- generic 对 duplicate 为0/2、0/2、1/2；PDF 三 seed均为0/2，反而破坏原 patch 对
  重复字符的识别。
- PDF 对 insert 仍只有2/5，未改善尾部插入。

#### 结论

- M06-A 是明确负结果：CLIP text encoder 更偏语义概念，不是精确 OCR/字符序列编码；
  将模板文字 embedding 与视觉差特征直接相乘/拼接，无法告诉模型目标图究竟是 A 还是
  C，并可能学习模板语义捷径。
- 不继续调 generic prompt 或堆 CLIP text MLP。下一步 M06-B 必须引入目标侧字符证据：
  OCR template/target string edit features，或字符级视觉 encoder/cross-attention。
- 为保持归因清楚，先做 OCR 字符差异作为独立分支，并单独报告 OCR coverage/error；
  不把 OCR 失败默认为 same。
- 产物：`results/label_pair/synthetic_d04_l14_336_patch_text_{none,generic,pdf}_seed*`。

### 2026-07-12 / M06-B / DONE（OCRv6 字符差异特征为正增益）

#### OCR 引擎与缓存

- 严格复用最新 OCRv6 Transformers 路径，不使用旧 Paddle 推理：
  `/home/jnu/venvs/gree-layout-hf-gpu`，PP-OCRv6 medium det/rec，
  `engine="transformers"`。
- 新增 `gree-label-detection/scripts/build_ocrv6_pair_features.py`，分别识别 exact
  template/target review crop，保存原始文字、置信度、空结果状态、Levenshtein
  增删替换、大小写折叠距离及 PDF 模板文字一致性。
- 全量缓存：
  `results/clip_pair_datasets/synthetic_d04_diverse_symmetric/ocrv6_features.jsonl`；
  534/534 pair，0 OCR exception，耗时34.30s（0.0642s/pair）。
- 双侧 OCR 非空覆盖：train=`193/301`、val=`72/90`、test=`129/143`。test positive
  为17/17，test negative为112/126。
- OCR 不能作为硬规则：test positive 平均 normalized edit distance=`0.1606`，negative
  反而为`0.4189`；直接按文字不等判 keep 会产生大量成像噪声 FP。

#### 特征与控制实验

- 新增 `aa_clip_exp/label_pair/ocr_features.py`，把缓存转换为24维有界数值特征：
  双侧可用性、严格/折叠相等、序列相似度、编辑距离及操作比例、长度差、OCR
  confidence、PDF-template 一致性。缺失状态显式编码，不把 OCR 失败默认为 same。
- 扩展 `label_pair/train_patch.py`，新增三个 feature mode：
  - `ocr_only`：只使用24维 OCR 数值特征；
  - `patch_ocr_none`：16,384维 patch + 24维全零，作为同参数量控制；
  - `patch_ocr`：16,384维 patch + 24维 OCR。
- 三组均复用 D04 split、同一 frozen L/14@336 patch cache、balanced BCE、batch=16、
  epochs=30、validation F1选阈值和 seeds=`20260710/20260711/20260712`。OCR-only
  hidden=64，其余 hidden=512。
- 代表命令：

```bash
cd /home/jnu/projects/aa_clip_exp
.venv-gpu/bin/python -m label_pair.train_patch \
  --train-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d04_diverse_symmetric/splits/train.jsonl \
  --val-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d04_diverse_symmetric/splits/val.jsonl \
  --test-manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d04_diverse_symmetric/splits/test.jsonl \
  --feature-cache results/label_pair/feature_cache/synthetic_d04_l14_336_multilayer_norm.pt \
  --ocr-feature-jsonl /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/synthetic_d04_diverse_symmetric/ocrv6_features.jsonl \
  --model ViT-L-14-336-quickgelu \
  --pretrained /home/jnu/models/clip/ViT-L-14-336px.pt \
  --layers 5,11,17,23 --topk-ratio 0.1 \
  --feature-mode patch_ocr --device cuda --epochs 30 --batch-size 16 \
  --hidden-dim 512 --seed 20260710 \
  --output-dir results/label_pair/synthetic_d04_l14_336_patch_ocr_seed20260710
```

#### 三 seed 结果

| 特征 | Test F1（三 seed） | F1均值 | Precision均值 | Recall均值 | ROC-AUC均值 | PR-AUC均值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| patch + zero OCR | 0.737 / 0.667 / 0.800 | 0.7345 | 0.7379 | 0.7451 | 0.9132 | 0.7920 |
| OCR-only | 0.520 / 0.520 / 0.531 | 0.5235 | 0.3980 | 0.7647 | 0.9194 | 0.6067 |
| patch + OCR | **0.875 / 0.750 / 0.875** | **0.8333** | **0.8396** | **0.8431** | **0.9687** | **0.9151** |

- patch+OCR 在三个配对 seed 上均超过同维 zero control，平均F1绝对提升0.0988；这不是
  额外输入维度本身造成的单次偶然结果。
- OCR-only 的FP为20/20/19，说明字符差异和置信度不足以独立处理成像噪声；正增益来自
  OCR质量/编辑证据与视觉 patch correspondence 互补。
- patch+OCR 的FP为1/8/1，FN为3/2/3。seed `20260711` 的validation阈值迁移仍产生
  8个FP，校准方差尚未解决，不能只报告两个最好 seed。
- mutation recall：confusable、delete、duplicate 三 seed均全检；insert为4/5、5/5、
  4/5；substitute仍为2/4。
- `Sound Pressure Level(H) 46dB(A) -> 46dB(C)` 的原始和shadow pair，OCR均准确输出
  A/C且normalized distance=1/28，但三个 seed仍稳定漏检。M06-B主要降低hard-negative
  FP并改善部分insert，而不是解决全部微小替换。

#### 结论与产物

- M06验收完成：generic/PDF CLIP text是负结果，OCRv6字符差异数值分支在D04上产生
  稳定正增益。当前推荐配置更新为 frozen L/14@336 normalized patch + OCR features。
- 结论仍限于8个独立test mutation和程序化成像增强；进入真实集前不能声称统计显著，
  也不能把增强siblings当作独立错误。
- 产物：
  - `results/label_pair/synthetic_d04_l14_336_patch_ocr_none_seed*`
  - `results/label_pair/synthetic_d04_l14_336_ocr_only_seed*`
  - `results/label_pair/synthetic_d04_l14_336_patch_ocr_seed*`
- 下一步：冻结 M06-B 方案进入 R01 synthetic-to-real zero-shot；同时单独处理 traditional
  候选的36/40召回问题，不把候选漏检归因于pair确认器。

### 2026-07-12 / R01 / DONE（latest15 synthetic-to-real zero-shot pilot）

#### 真实 pair 数据与协议

- 新增 `scripts/build_real_clip_pair_dataset.py`，通过 capture-only review client 重跑默认
  traditional 主流程并导出 exact template/target review crop 与 candidate mask；不调用
  VLM，不使用模型判断生成标签。
- 输入为冻结的 latest15 manifest 和既有人工 GT；标签规则仍为
  `intersection / min(candidate_area, gt_area) >= 0.3`，与 E00/D04 一致。
- 输出 `results/clip_pair_datasets/real_latest15_v1`：15 case、69 review pair，与 E00
  已验证数量完全一致；15 positive candidate、54 negative candidate、5个真实模板。
- 15个 GT 中14个至少有匹配候选；`600004085656_20260701-150603-071461_600004085656`
  的 GT 没有进入 review candidate，因此 pair模型最多恢复14/15个真实框。
- OCRv6 Transformers 对69 pair重新提取并保存到
  `real_latest15_v1/ocrv6_features.jsonl`：0 exception，正例15/15、负例53/54双侧非空；
  正例平均normalized distance=`0.1196`，负例=`0.0560`。
- latest15只用于一次冻结测试：不重新训练head，不在真实label上选择阈值。每个checkpoint
  使用其D04 synthetic validation阈值。

数据导出命令：

```bash
cd /home/jnu/projects/gree-label-detection
.venv/bin/python scripts/build_real_clip_pair_dataset.py \
  --output-dir results/clip_pair_datasets/real_latest15_v1

/home/jnu/venvs/gree-layout-hf-gpu/bin/python scripts/build_ocrv6_pair_features.py \
  --manifest results/clip_pair_datasets/real_latest15_v1/manifest.jsonl \
  --output results/clip_pair_datasets/real_latest15_v1/ocrv6_features.jsonl \
  --device gpu:0
```

#### 冻结 checkpoint 推理

- 新增 `aa_clip_exp/label_pair/evaluate_checkpoint.py`，加载已有head checkpoint、固定阈值
  和同一套L/14@336 multi-layer candidate-aware patch定义。69 pair视觉特征只编码一次：
  `results/label_pair/feature_cache/real_latest15_l14_336_multilayer_norm.pt`。
- 同时报告candidate confusion和按人工GT贪心匹配的框指标。当前框指标使用kept candidate
  box，尚未进入desktop display-box refine/merge，因此作为R01离线pilot，不冒充完整桌面
  端运行耗时或最终可视框结果。
- 代表命令：

```bash
cd /home/jnu/projects/aa_clip_exp
.venv-gpu/bin/python -m label_pair.evaluate_checkpoint \
  --manifest /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/real_latest15_v1/manifest.jsonl \
  --checkpoint results/label_pair/synthetic_d04_l14_336_patch_ocr_seed20260710/best_checkpoint.pt \
  --visual-feature-cache results/label_pair/feature_cache/real_latest15_l14_336_multilayer_norm.pt \
  --ocr-feature-jsonl /home/jnu/projects/gree-label-detection/results/clip_pair_datasets/real_latest15_v1/ocrv6_features.jsonl \
  --output-dir results/label_pair/real_latest15_r01_patch_ocr_seed20260710 \
  --device cuda --batch-size 16
```

#### 三 seed 冻结阈值结果

| 特征 | 框TP/FP/FN（三seed） | 框F1（三seed） | F1均值 | ROC-AUC均值 | PR-AUC均值 |
| --- | --- | --- | ---: | ---: | ---: |
| raw patch | 10/0/5、10/0/5、10/0/5 | **0.800/0.800/0.800** | **0.8000** | 0.9066 | 0.8718 |
| patch + zero OCR | 9/0/6、6/0/9、10/0/5 | 0.750/0.571/0.800 | 0.7071 | 0.8979 | 0.8274 |
| patch + OCR | 9/0/6、11/5/4、9/0/6 | 0.750/0.710/0.750 | 0.7366 | **0.9745** | **0.9285** |
| OCR-only | 13/11/2、13/11/2、12/10/3 | 0.667/0.667/0.649 | 0.6607 | 0.9329 | 0.8415 |

- raw patch三seed使用各自D04 validation阈值，真实框结果仍完全一致，Precision=1.0、
  Recall=0.6667；它是当前冻结阈值zero-shot的最佳配置。
- patch+OCR在真实集的ranking明显更好，但固定synthetic阈值产生两种失配：两个seed
  保守到9TP/0FP，一个seed变为11TP/5FP。平均F1没有超过raw patch，故M06-B的synthetic
  推荐不能直接升级为部署推荐。
- OCR-only Recall高但产生10–11个框FP，并有同一GT的重复候选；仍不能独立使用。
- raw patch漏掉的5个positive candidate分数集中在0.4646–0.4749附近，接近但低于对应
  synthetic阈值；另有1个总体FN来自候选阶段。下一问题是跨域校准/少量real adaptation，
  不是继续在latest15上选阈值。

#### 结论与下一步

- R01 latest15 pilot验收完成，但样本仍只有15个真实GT，不能报告显著性或替代扩充实拍
  test。当前zero-shot checkpoint选择回到raw patch，而不是patch+OCR。
- M06-B的OCR特征保留：其跨域排序提升表明信息有效，R02应在独立real validation上做
  calibration或少量真实微调，再用隔离real test评估；不得直接用latest15调阈值后回报。
- 产物：`results/label_pair/real_latest15_r01_{patch,patch_ocr_none,patch_ocr,ocr_only}_seed*`。

### 2026-07-12 / D05 + M07 / DONE（五模板专用模型）

#### 任务边界调整

- 企业当前只提供5个固定模板，因此本阶段目标改为closed-template specialist：五个模板
  都允许参与训练，但同一底层mutation及其增强兄弟不能跨split。
- 不再用D04的按模板隔离协议评价未知模板泛化；论文/汇报必须明确这是固定模板部署，
  不能把结果表述为对新产品模板的zero-shot能力。

#### D05数据

- dataset项目对每模板生成60个独立diverse mutation，共300 sample、390处程序化GT；
  实际策略为confusable=130、duplicate=53、substitute=51、insert=44、delete=34、
  case=32、swap=32、superscript=14，本seed仍未生成decimal_shift。
- 每个defect保留original并增加1个soft-capture/shadow安全增强；每模板生成30个clean，
  最终750 case：600 defect、150 clean。
- 扩展`build_clip_pair_dataset.py --split-mode within_template_source`：每个模板内部按
  source做70/15/15近似切分，增强兄弟随source固定到同一split。
- 主流程导出3948 pair：740正/3208负。split：
  - train=`2829 (519+/2310-)`，315个独立source；
  - val=`589 (116+/473-)`，70个独立source；
  - test=`530 (105+/425-)`，65个独立source。
- train/val/test均包含全部5个模板，三组source交集为0。
- 产物：
  - `/home/jnu/projects/dataset/outputs/datasets/clip_pair_d05_specialist`
  - `/home/jnu/projects/dataset/outputs/clip_pair_d05_specialist_symmetric`
  - `results/dataset_project_eval_dataset/clip_pair_d05_specialist`
  - `results/clip_pair_datasets/synthetic_d05_specialist`

#### M07 raw patch与容量

- 复用M03 frozen L/14@336四层normalized candidate-aware patch定义；新cache：
  `results/label_pair/feature_cache/synthetic_d05_specialist_l14_336_multilayer_norm.pt`。
- head hidden从512降到128，可训练参数约从839万降到210万；epochs=30、batch=16、
  balanced BCE，validation选阈值，三个seed。

| Head | Synthetic test F1 | TP/FP/FN |
| --- | --- | --- |
| raw patch h128 seed 20260710 | 0.9016 | 87/1/18 |
| raw patch h128 seed 20260711 | 0.8901 | 85/1/20 |
| raw patch h128 seed 20260712 | 0.9016 | 87/1/18 |
| raw patch h512 seed 20260710 | 0.9016 | 87/1/18 |

- h128三seed平均F1=`0.8977`，明显高于D04跨模板pilot的`0.7009`；h512没有增益，
  后续保留h128。
- 冻结最大F1阈值直接迁移latest15时，h128三seed框结果为11/0/4、8/0/7、12/0/3，
  F1=`0.846/0.696/0.889`；平均略优于旧D04 raw patch的0.800，但阈值方差仍明显。

#### OCR与高Recall阈值

- D05 OCRv6 cache覆盖3948/3948 pair、0 exception，耗时235.95s。
- 相同h128和seed下，patch+OCR synthetic test F1=`0.884/0.860/0.878`，低于同维
  zero control的`0.902/0.890/0.907`；OCR并未改善最大F1模型。
- 但patch+OCR在latest15三个seed的ROC-AUC/PR-AUC均为1.0，15个正候选和54个负候选
  完全可排序，失败来自最大F1阈值过度保守。
- 为匹配工业质检的漏检优先目标，扩展`evaluate_checkpoint.py`：只在synthetic validation
  上选择`Recall >= 0.95`时Precision最高、阈值最高的点；不读取real label选阈值。
- 但高Recall“规则选择”是在观察latest15最大F1迁移和ranking结果后提出，因此latest15
  从这一刻起属于开发/展示集，不再是严格未触碰final test。具体阈值没有用real label
  搜索，但策略选择具有post-hoc成分；最终结论仍需企业新增照片或未来独立采集复核。

| Seed | Synthetic val阈值 | Candidate TP/FP/FN | 完整主流程TP/FP/FN | P/R/F1 |
| --- | ---: | --- | --- | --- |
| 20260710 | 0.192473 | 15/0/0 | **14/0/1** | 1.000/0.933/0.966 |
| 20260711 | 0.656149 | 15/1/0 | 14/1/1 | 0.933/0.933/0.933 |
| 20260712 | 0.195552 | 15/0/0 | **14/0/1** | 1.000/0.933/0.966 |

- latest15有15个positive trigger candidate但只覆盖14个独立GT：一个GT由完整型号行和
  `/I`两个触发候选命中，另一个GT完全没有traditional candidate。
- 初版离线评估错误地直接使用`candidate_box_final`作为输出框，多算了重复FP。修正后新增
  `replay_clip_pair_predictions.py`，把缓存keep/discard按原顺序重放进完整traditional流程，
  由原代码执行PDF预设文字框映射、`refine_display_box`和display merge；两个trigger最终
  合并为一个型号行框。69/69 review请求硬校验一致。
- 与旧D04 raw patch的`10/0/5`相比，M07的synthetic-val高Recall策略在latest15开发集
  恢复到14个GT；剩余1个FN属于候选生成阶段，pair确认器无法补救。

#### 当前选择

- 展示用主结果：M07 patch+OCR h128 + synthetic-validation Recall95阈值，报告三seed
  完整主流程重放指标，不只挑最好seed。
- 稳健低误报备选：M07 raw patch h128最大F1阈值。
- 完整主流程重放产物：
  `results/batch_desktop_captures/latest15_m07_recall95_seed*_replay`及对应
  `results/evaluation/latest15_m07_recall95_seed*_replay`。下一步是将实时M07 worker接入
  desktop，而不是继续用latest15标签调阈值。

### 2026-07-12 / R01-independent50 / DONE（新增每模板10张独立实拍）

- 用户在M07和Recall95规则冻结后新增`single10_20260712`，实际为5模板各10张，共50张
  新实拍、50个人工GT；与latest15不同，可作为更可信的独立验证集。
- manifest：`results/batch_desktop_captures/single10_20260712_manifest/manifest.json`；
  GT：`results/manual_gt/single10_20260712/results/desktop_app`。50/50 annotation均有且仅有
  1个GT。
- 首版GT中`600004078454_20260712-164724-555646_600004078454`误标为`590m/h`；用户
  于17:50修正为底部地址行。修正后新建`real_single10_20260712_v2_gtcorrected`，不覆盖v1。
- 修正GT后exact主流程仍为265个review pair：52正/213负；47/50个GT至少有匹配
  trigger candidate。候选阶段漏掉3个GT：模板`600001076226`两个、`600004078454`一个。
- OCRv6覆盖265/265 pair、0 exception；正例50/51、负例208/214双侧非空。
- M07三seed沿用D05 synthetic validation Recall95规则和固定阈值，不使用这50张选择阈值；
  每个seed的keep/discard均重放回完整traditional预设框refine/merge流程。

| 方法 | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unfiltered traditional candidates | 47 | 202 | 3 | 0.1888 | 0.9400 | 0.3144 |
| M07 seed 20260710 | 40 | 8 | 10 | 0.8333 | 0.8000 | 0.8163 |
| M07 seed 20260711 | 41 | 13 | 9 | 0.7593 | 0.8200 | 0.7885 |
| M07 seed 20260712 | 40 | 5 | 10 | 0.8889 | 0.8000 | **0.8421** |

- M07将FP从202降到5–13，F1绝对提升0.4741–0.5277，证明pair确认器对固定五模板有
  明确价值；但模型在3个候选漏检之外又漏掉6–7个已进入review的GT，Recall仍需改进。
- seed20260712按模板：`600004085656`为10/0/0，`600004083205`为10/2/0，
  `600004075219`为8/2/2，`600001076226`为6/1/4，`600004078454`为6/0/4。
  后两模板是下一轮定向补数据和错误分析重点。
- 产物：
  - `results/clip_pair_datasets/real_single10_20260712_v2_gtcorrected`
  - `results/label_pair/real_single10_20260712_m07_patch_ocr_recall95_seed*`
  - `results/batch_desktop_captures/single10_20260712_m07_recall95_seed*_replay`
  - `results/evaluation/single10_20260712_unfiltered_vs_m07_gtcorrected_175005`
- 50张四方法并排可视化位于
  `results/evaluation/single10_20260712_unfiltered_vs_m07_gtcorrected_175005/visualizations/compare/`。

### 2026-07-12 / M08-error-audit / DONE（独立50张分层错误归因）

- 新增`scripts/analyze_single10_m07_errors.py`，以修正GT后的v2 manifest、三seed概率、
  OCR文本和完整主流程框指标生成结构化错误审计：
  `results/evaluation/single10_20260712_m07_error_audit`。
- 50 case分类：31个三seed全正确；3个candidate-stage FN；6个三seed稳定model FN；
  1个seed-variable FN；5个stable-FP case；4个seed-variable-FP case。
- 模板分布：`600004085656`为10/10三seed全正确；`600004083205`只有FP、无FN；FN集中
  在`600001076226`、`600004075219`、`600004078454`。
- 3个candidate FN中，两个是`600001076226`型号行的`I/1/括号`细变化，一个是
  `600004078454`风量上标区域；需要改traditional候选召回。
- 6个稳定model FN中，4个集中在`m³/h -> m²/h`或极小上标，2个集中在型号后缀、
  括号和WIFI。D05只有14处superscript mutation，该类覆盖不足。
- stable FP主要伴随OCR边界伪字符：`AirFlowVolume -> ZAirFlowVolume`、地址末尾多`0`、
  产品编码前多字符、局部品牌字母残片。它们应作为真实hard negatives，而不是继续降低
  全局阈值。
- seed11在多个边界噪声case上单独产生FP，进一步确认单head校准方差；最终部署可考虑
  三seed一致性或ensemble，但必须同时保护上标Recall。
- 补做同一D05 specialist的raw patch对照并完整重放50张：三seed为32/1/18、30/1/20、
  34/1/16，F1=`0.7711/0.7407/0.8000`；OCR版本为40/8/10、41/13/9、40/5/10，
  F1=`0.8163/0.7885/0.8421`。OCR确实提高6–11个TP且三个seed F1均更高，不能简单
  删除；目标应是保留OCR Recall并通过hard negatives降低其边界伪字符FP。
- raw/OCR对照产物：`results/evaluation/single10_20260712_raw_vs_ocr_gtcorrected`。
- 下一实验M08应只改变数据：定向增加上标、单位、型号后缀/括号hard positives，并加入
  OCR首尾伪字符/crop边界hard negatives；保持L/14 patch和h128 head不变后重跑三seed。

### 2026-07-13 / M08-A / DONE（负结果：简单上标合成未修复真实FN）

- dataset builder新增`mutation_profile=superscript`，强制单mutation样本；不改变diverse
  默认行为。
- `600001076226`没有可变独立上标span，强制生成在10次尝试后失败；保留失败记录，后续
  用型号后缀/括号策略处理，不伪造上标数据。
- 其余4模板各生成15个独立superscript mutation，共60 source；每个增加1个安全
  soft-capture/shadow变体，得到120 case。
- 主流程导出492 pair：181正/311负；train=`317 (119+/198-)`、val=`70 (27+/43-)`、
  test=`105 (35+/70-)`，四模板均进入各split且source分组。
- 产物：
  - `/home/jnu/projects/dataset/outputs/datasets/clip_pair_m08_superscript_v2`
  - `/home/jnu/projects/dataset/outputs/clip_pair_m08_superscript_symmetric`
  - `results/dataset_project_eval_dataset/clip_pair_m08_superscript`
  - `results/clip_pair_datasets/synthetic_m08_superscript`
- 只将M08-A train的317 pair（119正/198负）并入D05 train；D05 val/test完全冻结。为避免
  D05/M08重复pair ID，M08记录增加`m08sup__`命名空间；合并train=3146，val/test仍为
  589/530。复用旧D05 cache，只编码新增492 pair。
- D05 test三seed全部为84/1/21、F1=`0.8842`；相比M07的84/1/21、80/1/25、83/1/22，
  主要降低seed方差，没有提高最佳Recall。
- 修正后独立50张完整主流程：

| 方法 | TP/FP/FN（三seed） | F1（三seed） | F1均值 |
| --- | --- | --- | ---: |
| M07 | 40/8/10、41/13/9、40/5/10 | 0.8163/0.7885/0.8421 | 0.8156 |
| M08-A | 40/7/10、39/5/11、40/8/10 | 0.8247/0.8298/0.8163 | 0.8236 |

- M08-A平均F1仅+0.0080，TP没有提升；6个稳定model FN和3个candidate FN全部仍在，
  seed11还新增漏掉一个原本命中的case。简单渲染上标集单独训练1 epoch即满分，说明其
  外观远比真实拍照上标简单，域差距未被解决。
- 结论：M08-A作为负结果结束，不替换M07推荐模型。下一步不再堆同类简单PDF上标样本；
  应使用真实hard-positive crop做few-shot/蒸馏，或生成更接近实拍的局部上标退化，同时将
  3个candidate-stage FN单独交给传统候选模块。
- 对照产物：`results/evaluation/single10_20260712_m07_vs_m08_superscript_gtcorrected`。

### 2026-07-13 / D05-M08-line-GT-v2 / DONE（修正合成文字GT并重训）

- 发现旧合成文字GT直接使用PDF mutation span：独立字符只框一个字符，上标只框极小
  glyph；这与真实人工GT和主流程最终PDF预设行框不一致。
- `prepare_dataset_project_eval_dataset.py`改为将精确mutation映射到主流程PDF文本行，最终
  GT取文本行与精确mutation框的并集；精确框另存为`mutation_pixel_bbox`和
  `mutation_evidence`。同一PDF行的多个mutation合成一个GT；无法映射行时转换直接失败。
- D05基础集300 sample、390 mutation evidence合成387个唯一行级GT，全部成功映射；M08
  60/60 mutation全部成功映射。修正GT图片也从无框目标图重新绘制，不再复制旧小框预览。
- 重新运行当前traditional workflow会得到3295 pair，而旧D05为3948 pair。逐文件SHA256
  验证新旧300基础图和750增强图完全相同，差异来自工作树内候选/分组流程已继续演进。
  因而`synthetic_d05_specialist_linegt_v2`只保留作流程漂移审计，不能用于本轮GT单变量
  对照。
- 严格实验固定旧3948个pair、crop、candidate mask、split和OCR/CLIP特征，只用新GT重新
  标注：24 pair由负转正，无正例转负；总计764正/3184负。split为train
  `2829 (534+/2295-)`、val `589 (118+/471-)`、test `530 (112+/418-)`。
- 唯一有效D05 line-GT pair路径：
  `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v2_relabel`。视觉cache的global/
  patch tensor逐元素与旧cache相同，只替换train/val/test的15/2/7个label。
- corrected M07 synthetic test三seed为`84/1/28`、`81/1/31`、`83/1/29`，F1=
  `0.8528/0.8351/0.8469`。正例总数已由105变112，不能直接用旧F1作同分母比较。
- corrected M07在修正GT独立50张的完整主流程为：
  - seed10：40/7/10，F1=0.8247；
  - seed11：41/12/9，F1=0.7961；
  - seed12：40/5/10，F1=0.8421；
  - F1均值0.8210，比旧M07均值0.8156高0.0053；Recall不变，仅seed10/11各减少1 FP。
- corrected M07在latest15开发集完整重放三seed全部为`14/0/1`、F1=0.9655；旧M07
  seed11的1个FP被消除。latest15仍是开发集，不改变其post-hoc边界。
- M08 line-GT relabel保持492 pair的181正/311负完全不变；将其317个train pair加入corrected
  D05 train，corrected D05 val/test继续冻结。corrected M08 synthetic test三seed均为
  `84/1/28`、F1=0.8528。
- corrected M08在独立50张完整主流程为：
  - seed10：40/5/10，F1=0.8421；
  - seed11：40/5/10，F1=0.8421；
  - seed12：44/23/6，F1=0.7521；
  - seed12找回3个真实上标/单位case和1个`WIFI -> WIFi`，但新增大量FP，单
    `600004075219`模板即14 FP。三seedF1均值0.8121，低于corrected M07的0.8210。
- 结论：行级GT修复是必要的数据正确性修复，但仅修24个D05错标不能解决稳定真实FN；
  简单M08上标数据能提高某个seed的Recall，却带来不可接受的校准方差。推荐模型更新为
  corrected M07，M08仍不替换M07。下一步仍是traditional candidate召回专项和按case分组
  的真实hard-example adaptation。
- 关键产物：
  - `results/dataset_project_eval_dataset/clip_pair_d05_specialist_linegt_v2`；
  - `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v2_relabel`；
  - `results/label_pair/synthetic_d05_specialist_linegt_v2_patch_ocr_h128_seed*`；
  - `results/evaluation/single10_20260712_m07_vs_linegt_v2_gtcorrected`；
  - `results/clip_pair_datasets/synthetic_m08_superscript_linegt_v2_relabel`；
  - `results/label_pair/synthetic_m08_d05_linegt_v2_superscript_patch_ocr_h128_seed*`；
  - `results/evaluation/single10_20260712_linegt_v2_m07_vs_m08_superscript_gtcorrected`。

### 2026-07-13 / M09-sensitive-PDF-line-recovery / DONE（候选召回修复）

- 对3个candidate-stage FN逐案保存debug mask后确认两类根因：
  - 两个`600001076226`型号行在small-text diff中有强差异，但连通块面积超过
    `normal_min_area=40`而被微候选排除；普通diff的`tolerance=3`又将它们抹掉；
  - 一个`600004078454`风量上标有8–23像素合格小连通块，但被全图micro candidate
    top-8限制挤掉。
- 新增高风险PDF文本行恢复：只处理包含长型号编码的行和`m /h`紧凑单位行，在行内聚合
  small-text diff；若该行已有候选则跳过。证据少于8像素视为不足，超过1000像素视为
  整行错位噪声。恢复框继续走原`micro_text_candidate` review和pair流程，并记录
  `recovery_source=sensitive_pdf_text_line`。
- 默认启用；设置`ENABLE_SENSITIVE_PDF_LINE_RECOVERY=0`可关闭做消融。
- 修正GT独立50张pair从265增至291（+9.8%）：57正/234负；50/50 GT现在都有trigger
  candidate。冻结corrected M07和原synthetic Recall95阈值，完整主流程为：
  - seed10：42/7/8，P=0.8571，R=0.84，F1=0.8485；
  - seed11：43/12/7，P=0.7818，R=0.86，F1=0.8190；
  - seed12：42/5/8，P=0.8936，R=0.84，F1=0.8660。
- 相比无恢复的corrected M07，三个seed均增加2 TP、增加0 FP；平均Recall从0.8067升到
  0.8467，平均F1从0.8210升到0.8445。两个恢复TP是一个型号行和一个风量上标。
  另一个型号行已有候选但三个seed均discard，现从candidate-stage FN转为model FN。
- latest15 pair从69增至81；三个seed完整主流程均从14/0/1提升到15/0/0，未增加FP。
- v1恢复未限制最大行内diff像素，在`600004075219...164019`未变化风量行产生2020像素
  错位候选并造成稳定FP；v4增加1000像素上限后该FP消失。后续只使用v4产物。
- 结论：候选召回专项达到验收条件，corrected M07 + 默认高风险PDF行恢复成为当前推荐
  系统。下一数据实验进入按case分组的真实hard-example adaptation，重点处理剩余7个稳定
  model FN及OCR边界hard negatives。
- 关键产物：
  - `results/clip_pair_datasets/real_single10_20260712_v4_gtcorrected_sensitive_line_recovery`；
  - `results/label_pair/real_single10_20260712_candidate_recovery_v4_m07_recall95_seed*`；
  - `results/evaluation/single10_20260712_linegt_m07_vs_candidate_recovery_v4_gtcorrected`；
  - `results/clip_pair_datasets/real_latest15_sensitive_line_recovery_v2`；
  - `results/evaluation/latest15_linegt_m07_vs_candidate_recovery`。

### 2026-07-13 / M10-grouped-real-hard-adaptation / DONE（负结果：真实微调未改善）

- 使用M09的real50 v4：50 case、291 pair（57正/234负）。该50张此前已用于错误分析和
  候选规则开发，因此本实验明确是开发集交叉验证，不再称为独立final test。
- 固定`split_seed=20260713`，按template内case哈希排序做5折：每折每模板6个train、2个
  validation、2个test，共30/10/10 case；同一照片的全部candidate始终在同一split。
  每个case恰好作为test一次，OOF覆盖291/291 pair，无case泄漏。
- 三个corrected M07 seed分别初始化h128 patch+OCR head；只微调head，CLIP/OCR特征冻结；
  AdamW、lr=`1e-5`、weight decay=`1e-4`、batch=32、20 epoch。每折阈值和epoch只由该折
  validation选择。
- 为拆分阈值与训练影响，同时报告三条路径：原synthetic Recall95阈值；epoch 0但用real
  validation重新校准；只在epoch 1–20中选择最佳微调checkpoint。三seed共15折的部署选择
  全部停在epoch 0；强制微调最佳也全部为epoch 1，说明首个更新后validation已不再改善。
- OOF candidate F1：
  - synthetic阈值：`0.8148 / 0.8000 / 0.8302`；
  - real validation校准：`0.7568 / 0.7611 / 0.7963`；
  - 强制微调：`0.7568 / 0.7611 / 0.7736`。
- 完整PDF预设display-box replay：
  - M09 baseline：`42/7/8`、`43/12/7`、`42/5/8`，F1均值`0.8445`；
  - validation选择（均epoch 0）：`40/12/10`、`41/13/9`、`42/8/8`，F1均值`0.8043`；
  - 强制微调：`40/12/10`、`41/13/9`、`40/8/10`，F1均值`0.7964`。
- 各折real-validation阈值波动很大：seed10为`0.1285–0.9567`，seed11为
  `0.3020–0.9807`，seed12为`0.0741–0.5000`。10-case validation不足以稳定重校准。
- baseline有7个三seed稳定FN；validation选择后仍为7个，但只是偶然找回一个case同时让
  另一个原TP变为三seedFN；强制微调的稳定FN增至9。没有形成可部署的困难样本改善。
- 结论：M10为负结果，不替换M07+M09，不使用real-validation阈值或5折checkpoint部署。
  继续R02需要新增独立真实train/validation/final-test，而不是在这50张上调更多超参数。
- 关键代码与产物：
  - `aa_clip_exp/label_pair/train_real_grouped_cv.py`；
  - `aa_clip_exp/results/label_pair/m10_real50_grouped5fold_hard_adapt_lr1e5_v2_seed*`；
  - `results/evaluation/single10_20260712_m10_grouped5fold_hard_adaptation`。

### 2026-07-13 / D05v3-PDF-semantic-candidates / DONE（按正确候选几何重训）

- 修正主流程候选定义：traditional diff只负责触发，送入pair判断的`box`必须吸附到PDF
  文本区域；有PDF区域时丢弃无法映射的局部框，不再把相邻PDF区域按扩张review crop合并。
  `850m³/h`等被PDF拆分的数值、单位和上标先合并为一个语义区域。
- 额外修复一次二次匹配漂移：候选构建阶段已映射到型号行，但送审阶段重新按原始差分
  质心评分时可能被相邻`)`抢走元数据。后续阶段现在以`text_line_key`为权威映射。
- 用line-GT-v2源数据重建750 case，得到3730 pair：788正/2942负；split为train
  `2652 (545+/2107-)`、val `574 (122+/452-)`、test `504 (121+/383-)`。
  3730/3730候选均为精确PDF区域，框与文本元数据一致，未映射和重复候选均为0；450个
  source组零泄漏。600个defect case中588个有正候选，candidate recall=98.0%。
- 完整单位语义框共141 pair（83正）：`590m³/h`49、`1000m³/h`32、`850m³/h`31、
  `2000m³/h`29。OCRv6覆盖3730/3730、0 error；视觉cache约277 MB。
- 纯D05v3 patch+OCR h128合成test三seed最大F1为`0.8195/0.8252/0.8195`；冻结
  synthetic-validation Recall95阈值后，real50完整主流程为：
  - seed10：44/6/6，F1=0.8800；
  - seed11：43/6/7，F1=0.8687；
  - seed12：45/13/5，F1=0.8333；
  - F1均值0.8607，高于旧M07在相同PDF语义候选上的0.8522；平均Recall由0.8733升至
    0.8800，平均FP由9.0降至8.33。
- latest15完整主流程为`15/0/0`、`15/0/0`、`15/1/0`。但关键case
  `600004075219...164003`的`850m³->850m²`仅seed12命中；seed10正例概率0.379367，
  比阈值0.380317低0.00095，seed11则明显低置信度。纯D05v3不能称为稳定解决该case。

#### M08语义框补充实验

- 将原M08上标源按同一PDF语义候选逻辑重建：120 case、407 pair（120正/287负），
  407/407均为PDF框；只把train的279 pair（80正/199负）加入D05v3 train。D05v3 val/test
  保持574/504不变，模型结构和Recall95规则不变。
- 合成test最大F1为`0.8789/0.8269/0.8365`。real50完整主流程为：
  - seed10：44/2/6，P=0.9565，R=0.88，F1=0.9167；
  - seed11：45/12/5，P=0.7895，R=0.90，F1=0.8411；
  - seed12：50/36/0，P=0.5814，R=1.00，F1=0.7353。
- 关键`850m³->850m²` case三seed均恢复为TP，概率0.4405/0.8291/0.9625；但seed11/12
  在该case各有3个额外FP。latest15为`15/0/0`、`15/1/0`、`15/10/0`。
- 2/3多数投票未消除共同误报：real50为45/12/5、F1=0.8411；latest15为15/1/0。
- 结论：正确PDF候选几何必须保留，纯D05v3获得小幅平均增益；M08语义补充证明完整单位框
  能稳定提高目标上标Recall，但跨seed校准方差仍不可接受。real50已经用于本轮分析，不能
  据此只挑F1=0.9167的seed10并宣称最终模型。M08版本暂不替换推荐模型，需新增独立真实
  final test，或先加入PDF相邻行/crop边界hard negatives后按冻结规则复验。
- 关键产物：
  - `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v3_pdf_semantic`；
  - `results/clip_pair_datasets/synthetic_m08_superscript_linegt_v3_pdf_semantic`；
  - `aa_clip_exp/results/label_pair/feature_cache/synthetic_d05_specialist_linegt_v3_pdf_semantic_l14_336_multilayer_norm.pt`；
  - `results/evaluation/single10_20260712_v6_pdf_semantic_old_m07_vs_d05v3_retrain`；
  - `results/evaluation/single10_20260712_v6_pdf_semantic_d05v3_m08semantic_recall95`；
  - `results/evaluation/latest15_v3_pdf_semantic_d05v3_m08semantic_recall95`。

### 2026-07-13 / M11-DINOv2-letterbox-2x2 / DONE（backbone与预处理消融）

- 动机：当前patch+OCR模型没有使用CLIP text encoder，且可视化发现336中心裁剪只看到
  review crop中央正方形。目标`850m³/h`候选final x范围为867–990，CLIP实际可见范围仅
  880.6–967.4；mask在模型输入中变成横向铺满条带，占58.3%像素。
- 做严格2×2对照：OpenCLIP/DINOv2-L/14 × center-crop/letterbox。固定D05v3+M08 train、
  val/test、OCRv6、block 6/12/18/24、24×24 token grid、四种pooling、h128、三个seed和
  synthetic Recall95阈值规则。letterbox后目标mask占比17.9%，完整候选未被水平裁断。
- DINOv2使用Meta官方`dinov2_vitl14_pretrain.pth`，SHA256=
  `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`；timm模型
  `vit_large_patch14_dinov2.lvd142m`。官方checkpoint仅多出推理不用的`mask_token`，其余
  343个参数键完整加载。
- 合成test最大F1三seed均值/总体标准差：
  - CLIP center：0.8475 / 0.0226；
  - CLIP letterbox：0.8232 / 0.0109；
  - DINOv2 center：0.8519 / 0.0146；
  - DINOv2 letterbox：0.8439 / 0.0026。
  合成最大F1不能体现最终迁移，DINO-letterbox方差最低。
- real50完整PDF display-box replay：
  - CLIP center：`44/2/6`、`45/12/5`、`50/36/0`，F1=
    `0.9167/0.8411/0.7353`，均值0.8310；
  - CLIP letterbox：`45/11/5`、`45/9/5`、`48/33/2`，F1=
    `0.8491/0.8654/0.7328`，均值0.8158；
  - DINOv2 center：`45/7/5`、`46/10/4`、`47/29/3`，F1=
    `0.8824/0.8679/0.7460`，均值0.8321；
  - DINOv2 letterbox：`48/4/2`、`39/0/11`、`42/2/8`，F1=
    `0.9412/0.8764/0.8936`，均值**0.9037**、标准差0.0274。
- DINO-letterbox三seed平均P=0.9592、R=0.8600、FP=2.0；CLIP-center平均P=0.7758、
  R=0.9267、FP=16.67。主要收益是显著压低边界/邻行误报，代价是平均Recall下降。
- latest15完整replay：CLIP center=`1.0000/0.9677/0.7500`，CLIP letterbox=
  `0.9091/0.9091/0.7895`，DINO center=`0.9677/0.9375/0.7895`，DINO letterbox=
  `1.0000/0.8889/1.0000`；DINO-letterbox均值0.9630且三seed均0 FP。
- 关键`850m³->850m²`：CLIP center和CLIP letterbox均3/3命中；DINO center 0/3；
  DINO letterbox仅seed10命中（p=0.28094，阈值0.27335）。DINO整体更好不代表该上标case
  稳定改善。DINO-letterbox仍有两个三seed稳定FN，均为`600001076226`型号行case。
- 当前开发集最佳单checkpoint为DINO-letterbox seed20260710：real50=48/4/2、
  F1=0.9412，latest15=15/0/0，关键上标case为1TP+1FP。相较原CLIP-center seed10，
  real50增加4TP、2FP、减少4FN，F1绝对+0.0245。
- batch16纯backbone双图前向：CLIP center 77.15ms/pair、CLIP letterbox 77.76、
  DINO center 72.46、DINO letterbox 72.70；DINO没有速度代价。该计时不含OCR和传统流程。
- 结论：DINOv2+letterbox成为新的**开发集最佳配置**，证明backbone和保持完整候选必须
  联合考虑；只给CLIP换letterbox或只换DINO均没有稳定增益。但real50/latest15均已用于
  选择配置和seed，不能再作为独立final test。部署候选可暂用DINO-letterbox seed10，正式
  替换仍需全新真实final set，尤其复核上标和`600001076226`型号行Recall。
- 关键代码与产物：
  - `aa_clip_exp/label_pair/visual_backbone.py`；
  - `aa_clip_exp/label_pair/visualize_pair_explanation.py`；
  - `aa_clip_exp/results/label_pair/synthetic_d05v3_m08_dinov2_letterbox_patch_ocr_h128_seed*`；
  - `results/evaluation/single10_v6_backbone_preprocess_2x2`；
  - `results/evaluation/latest15_v3_backbone_preprocess_2x2`；
  - `results/interpretability/850m3_to_850m2_dinov2_letterbox_seed10`。

### 2026-07-14 / M12-candidate-mask-OCR / DONE（负结果：目标稳定但总体Recall下降）

- 动机：review crop OCR在关键case输出`5.20KW850m3/h -> 850m2/h12.5kg`，相邻行污染
  编辑距离和增删替换比例。新增`build_ocrv6_pair_features.py --crop-mode candidate_mask`：
  在原pair crop坐标取mask外接框，水平padding 8%、垂直15%，四周8像素白边，等比例放大
  到至少96像素高；模板/目标使用同一坐标。默认仍为review，历史实验不变。
- 关键case mask bbox为109×45，处理后OCR输入183×96；OCR从旧的7替换+3插入+3删除
  改善为`850m3/h -> 850m2/h`唯一一次替换，normalized distance从1.0降到1/7，模板PDF
  文字相似度从0.70升至1.00。
- 合成4009 pair candidate OCR覆盖4009/4009、0 error。双侧非空由review的3154降至3011；
  负例strict equal由1454增至1787，但正例误识别相同也由22增至33。固定DINO-letterbox
  视觉cache，只替换24维OCR并重训h128三seed。
- 合成test最大F1：review OCR=`0.8476/0.8421/0.8421`，均值0.8439；candidate OCR=
  `0.8365/0.8421/0.8365`，均值0.8384。candidate OCR test ROC-AUC均值约0.9901，略高于
  review约0.9883，但最大F1无增益。
- real50完整流程：
  - review OCR：`48/4/2`、`39/0/11`、`42/2/8`，F1=
    `0.9412/0.8764/0.8936`，均值0.9037；
  - candidate OCR：`42/3/8`、`40/3/10`、`41/3/9`，F1=
    `0.8842/0.8602/0.8723`，均值0.8722。
- latest15：review OCR F1=`1.0000/0.8889/1.0000`，均值0.9630；candidate OCR=
  `0.9286/0.8462/0.8000`，均值0.8583，三seed虽均0 FP但FN增至2/4/5。
- 关键`850m³->850m²`取得预期收益：candidate OCR概率为0.965/0.848/0.922，三seed
  全部命中且该case 0 FP；review OCR仅seed10命中且有1 FP。
- seed10逐case比较：candidate OCR相对review失去7个TP、找回1个TP。部分损失来自mask
  过紧导致真实边界变化消失（`A/I -> A/II`在candidate crop两侧均识别成`/I`）；部分case
  OCR内容相同但新head/Recall95阈值更保守。候选OCR不是简单全面优于review OCR。
- `A/I -> A/II`边界示例可视化：绿色为PDF `/I` mask，黄色为candidate OCR crop；新增
  第二个`I`落在固定PDF框右侧并在candidate输入中被截断：
  `results/interpretability/candidate_ocr_boundary_context_A_I_to_A_II.png`。
- 结论：candidate-only OCR不替换当前DINO-letterbox review-OCR seed10。它证明局部OCR能
  稳定上标case并消除邻行污染，但丢失上下文导致总体Recall明显下降。若继续，应做双路
  `review OCR + candidate OCR`特征并设置同维zero control，而不是继续调candidate padding。
- 关键产物：
  - `results/clip_pair_datasets/synthetic_d05v3_plus_m08_pdf_semantic_train/ocrv6_candidate_features.jsonl`；
  - `aa_clip_exp/results/label_pair/synthetic_d05v3_m08_dinov2_letterbox_candidateocr_patch_ocr_h128_seed*`；
  - `results/evaluation/single10_v6_dinov2_letterbox_review_vs_candidate_ocr`；
  - `results/evaluation/latest15_v3_dinov2_letterbox_review_vs_candidate_ocr`；
  - `results/interpretability/850m3_to_850m2_dinov2_letterbox_candidateocr_seed10`。

### 2026-07-14 / M13-PDF-model-row-merge / DONE（几何修复保留，重训负结果）

- 根因：三个PDF模板将型号主体和`/I`后缀拆成独立文本region，导致traditional diff在后缀
  触发时送入模型的是单独`/I` mask。新增严格相邻合并：左侧必须是带连字符的长型号，
  右侧必须匹配`/[A-Za-z0-9]+`，垂直重叠和水平间距同时满足才合并。
- 真实`600004083205...165038`验证得到完整语义文本`GWH18AUDXE-K6DNA1A/I`和框
  `[472,107,878,144]`；实际pair crop为460x63，mask非零bbox为`[32,13,438,50]`。
- 重建D05v4为3714 pair（795正/2919负），独立`/I`从22降为0，全部候选精确属于PDF
  region，source split零泄漏；real50/latest15重建为264/73 pair，仍分别覆盖50/50和
  15/15 GT。三套review-OCR共4051条，0 error。
- 固定DINOv2-L/14、letterbox、block 6/12/18/24、candidate-aware pooling、review-OCR
  24维、h128、三seed和synthetic Recall95规则重训。完整最终框结果：
  - real50：`38/0/12`、`41/2/9`、`37/0/13`，F1=`0.8636/0.8817/0.8506`；
  - latest15：`10/0/5`、`13/0/2`、`10/0/5`，F1=`0.8000/0.9286/0.8000`。
  新训练头Recall明显下降，本轮重训是负结果。
- 再做checkpoint/geometry解耦：旧M11 DINO-letterbox seed10 checkpoint直接读取新整行
  crop/mask和新OCR，仍使用旧synthetic val Recall95阈值`0.2733518183`。发现旧最终框合并
  会把相邻PDF行的FP吞入TP大框；增加不同`text_line_key`禁止直接或间接合并后，完整
  real50为`48/6/2, F1=0.9231`，latest15为`15/1/0, F1=0.9677`。修复前的`48/4/2`和
  `15/0/0`分别隐藏2个和1个FP。这仍证明整行几何没有造成主要退化，重训head的Recall
  下降才是M13负结果的主因。
- 决策：保留PDF型号整行合并代码；当前开发集部署候选继续使用旧M11 review-OCR
  DINO-letterbox seed10 checkpoint，不使用M13重训checkpoint。正式结论仍需全新真实final
  set，real50/latest15不得继续用于选择新超参数或seed。
- 关键产物：
  - `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v4_pdf_semantic_modelrow`；
  - `results/clip_pair_datasets/real_single10_20260712_v7_gtcorrected_pdf_semantic_modelrow`；
  - `results/clip_pair_datasets/real_latest15_v4_pdf_semantic_modelrow`；
  - `aa_clip_exp/results/label_pair/synthetic_d05v4_m08_modelrow_dinov2_letterbox_patch_ocr_h128_seed*`；
  - `results/evaluation/single10_20260712_v7_modelrow_old_vs_retrained_best`；
  - `results/evaluation/latest15_v4_modelrow_old_vs_retrained_best`；
  - `results/evaluation/single10_20260712_v7_modelrow_cross_line_merge_fix`；
  - `results/evaluation/latest15_v4_modelrow_cross_line_merge_fix`。

### 2026-07-14 / M14-model-row-dual-OCR / DONE（seed10改善但跨seed不稳定）

- M13整行mask后重新生成candidate OCR，覆盖D05v4/M08/real50/latest15的
  `3714/407/264/73`条pair，全部0 error。旧M12的`A/I -> A/II`截断原因已消失：OCR现在
  正确得到完整型号`GWH18AUDXE-K6DNA1A/I -> GWH18AUDXE-K6DNA1A/II`。但固定整行PDF框
  仍可能漏掉行末外新增字符，如`...(WIFI) -> ...(WIFI)I`，不能把“型号region已合并”解释
  为“所有mask边界问题已消失”。
- 固定DINOv2-L/14、letterbox、block 6/12/18/24、candidate-aware pooling、D05v4+冻结
  M08 train、h128、三seed和synthetic Recall95阈值规则，仅把24维review OCR和24维
  candidate OCR拼接成48维。训练/评估接口新增`--ocr-feature-jsonl-secondary`，按pair顺序
  独立校验后拼接。
- 完整line-safe最终框：
  - real50：seed10/11/12=`45/1/5`、`38/0/12`、`41/0/9`，F1=
    `0.9375/0.8636/0.9011`；
  - latest15：`14/0/1`、`10/0/5`、`13/0/2`，F1=
    `0.9655/0.8000/0.9286`。
- seed10将`600004075219...164003`由`1TP+2FP`修为`1TP+0FP`，real50总FP由review基准
  6降到1；但相对review基准新增3个FN。seed11/12召回进一步下降，说明双路OCR倾向于
  提升精度但未形成稳定Recall收益。
- pair审计图显示：两个`590m³/h -> 590m⁸/h`漏报的候选overlap均为1.0，但两路OCR都输出
  `590m3/h -> 590m3/h`；型号行漏报还包括行末mask外插入、OCR漏读行内插入和大小写fold
  弱化。唯一FP是candidate OCR把`/I`误读成`/1`。latest15的`5.60kW`模板/目标视觉和OCR
  均一致，GT需单独复核。审计图位于`results/interpretability/dualocr_seed10_error_audit`。
- 决策：M14不替换当前M11 review-OCR seed10部署候选，也不根据real50挑选M14 seed10。
  保留双路实现和checkpoint，下一步只能在全新真实final set盲测；当前开发集停止调阈值、
  loss和seed。
- 关键产物：
  - `aa_clip_exp/results/label_pair/synthetic_d05v4_m08_modelrow_dinov2_letterbox_dualocr_patch_ocr_h128_seed*`；
  - `results/evaluation/single10_20260712_v7_modelrow_dualocr_3seed_line_safe`；
  - `results/evaluation/latest15_v4_modelrow_dualocr_3seed_line_safe`。

### 2026-07-14 / M15-synthetic-prealignment / DONE（数据修复，模型暂不替换）

- synthetic FP审计确认pair位置错配。典型`600001076226_clean_009`源target为完整标签，
  预处理却把面积5.98%的内部条码裁成`520x153`目标，再由SIFT接受`0.267x0.232`单应缩放；
  target aligned仅剩右下条码。D05v3/v4均有77/750个灾难性尺度case，M08为0/120。
- 实现三层防护：同尺寸画布直接跳过透视；synthetic构建显式prealigned并关闭SIFT/ECC；
  通用feature homography检查投影面积/双轴尺度，ECC要求相关性>=0.3，否则回退。已知坏样本
  默认路径恢复为693/797 SIFT内点、投影面积0.996、尺度约1.0、ECC相关性0.900。
- D05v5重建为3122 pair（772正/2350负），比v4减少592个错配伪候选；600/600 GT case
  candidate covered。split仍按原source seed冻结：train/val/test=`2205/490/427`；追加冻结
  M08 train 279条后=`2484/490/427`。
- review OCR test负例双侧非空由293/382提升为311/311、平均编辑距离0.338降至0.033；
  candidate OCR为310/311、严格相同282/311。修复后dual seed10 Recall95 synthetic test为
  `112/1/4, F1=0.9782`，错误审计由旧`5FN+40FP`降为`4FN+1FP`且位置正确。
- 三seed synthetic最大F1稳定：review=`0.9735/0.9780/0.9735`，dual=
  `0.9782/0.9784/0.9782`。但近乎可分的synthetic validation将Recall95阈值推高，真实完整
  回放0 FP但Recall明显不足：fixed dual real50=`36/0/14`,`35/0/15`,`35/0/15`，latest15=
  `11/0/4`,`9/0/6`,`10/0/5`；fixed review更低。
- 决策：保留配准和数据修复，D05v3/v4指标全部标记superseded；fixed checkpoint暂不部署，
  也不使用real50/latest15调阈值。旧M11虽由污染D05v3训练，但当前真实基准仍较高，暂作经验
  部署候选；正式选择需要独立真实validation/final set或预先定义的域校准方案。
- 关键产物：
  - `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v5_prealigned_pdf_semantic_modelrow`；
  - `aa_clip_exp/results/label_pair/synthetic_d05v5_m08_prealigned_modelrow_dinov2_letterbox_*`；
  - `results/evaluation/single10_20260712_v8_prealigned_training_3seed_line_safe`；
  - `results/evaluation/latest15_v5_prealigned_training_3seed_line_safe`；
  - `results/interpretability/dualocr_seed10_synthetic_test_error_audit_prealigned`。

### 2026-07-15 / M16-raw-diff-evidence-mask / DONE（合成改善，真实负结果）

- 新增双mask契约：PDF整行`candidate_mask`继续服务OCR/语义；workflow额外导出原始
  tolerant diff与small-text diff并集，再按pair trigger components截取像素级
  `evidence_mask`。D05v6的3122 pair全部非空可追溯，evidence/review面积中位数约1.0%。
- hard negative并非明显短缺：2350个D05负pair中约1034个clean/same覆盖5类成像增强，
  1316个是变更图中的其他未变候选；约90% candidate OCR严格相同。real负例evidence面积
  统计与synthetic相近，而且各新模型真实FP已为0或1，瓶颈是正例域差异和阈值校准。
- 为严格控制变量，M08保持旧v3的407 pair/279 train和全部旧crop/OCR/标签，只从重跑
  M08v4中提取raw evidence补回旧记录。最终划分仍`2484/490/427`。
- 三组均使用DINOv2-L/14 letterbox、block 6/12/18/24、双OCR48维、h128、三seed与
  synthetic Recall95：semantic旧4池化；evidence替换mask且patch grid膨胀一圈；dual为
  semantic/evidence各mean/max/top-k加global，patch维度28672。
- synthetic最大F1：semantic约0.9782，evidence三seed均0.9826，dual均值约0.9607；
  evidence-only在合成test稳定小幅改善。
- 完整真实最终框：evidence real50=`34/0/16`,`17/0/33`,`17/0/33`，latest15=
  `10/0/5`,`5/0/10`,`5/0/10`；dual real50=`33/1/17`,`29/0/21`,`31/1/19`，latest15=
  `10/0/5`,`10/0/5`,`9/0/6`。均未超过semantic seed10的`36/0/14`与`11/0/4`。
- 可视化确认raw diff在real上不等于语义变化：`A/I->A/II`定位正确并被找回；
  `850m³->850m²`却主要定位到`m`的残余错位笔画。evidence seed10相对semantic在real50
  找回2 TP、丢4 TP。保留mask资产，不部署新checkpoint。
- 下一步：若继续该方向，只做更稳的real evidence（字符匹配/局部残差配准/soft heatmap）
  或低维软引导，并补real-like hard positives；不继续单独堆hard negatives或扩大dual head。
- 关键产物：
  - `results/clip_pair_datasets/synthetic_d05_specialist_linegt_v6_prealigned_dualmask_modelrow`；
  - `results/evaluation/single10_20260712_v9_mask_pooling_ablation_line_safe`；
  - `results/evaluation/latest15_v6_mask_pooling_ablation_line_safe`；
  - `results/interpretability/mask_pooling_ablation`。
