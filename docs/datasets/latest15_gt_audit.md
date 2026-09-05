# latest15 GT 与候选审计

## 结论

latest15 可以作为五模板项目的开发/展示集，但不适合作为唯一正式测试集。GT整行框是
按PDF预设文字区域标注的，符合当前主流程设计；主要限制是样本构成和候选覆盖。

## 已确认问题

1. 15张图片每张恰好1个人工GT，共15个GT；没有整图正常样本。候选级有54个负候选，
   但它们都来自包含真实异常的图片，不能替代独立正常照片。
2. `manual_annotation.json`中的GT均为`miss_0`，`label`和box级`notes`为空；没有记录错误
   字符、正确文字、错误类型或审核人复核状态。
3. GT通常是PDF映射得到的整行预设区域，宽度范围124–1928像素，平均528.2像素。这不是
   当前任务定义下的标注错误；评估对象应是主流程refine/merge后的预设框，不能拿原始
   trigger candidate小框直接作为最终输出框。
4. `600004085656_20260701-150603-071461_600004085656`的GT是`2000m²/h`区域，但7个
   traditional候选均未覆盖它。该FN属于候选生成阶段，pair确认器无法恢复。
5. `600004083205_20260701-150424-135536_600004083205`的型号行由完整型号和`/I`两个
   trigger触发。完整主流程重放后，它们按PDF text-line映射并merge为一个型号行TP，
   不应计为重复FP。
6. Recall95策略是在观察latest15初步结果后提出，因此latest15已经是开发/展示集，不再
   是严格未触碰final test。

## 可视化

- trigger诊断总览（不是最终输出框）：
  `results/evaluation/latest15_m07_gt_audit/latest15_gt_candidate_contact_sheet.jpg`
- 单case：`results/evaluation/latest15_m07_gt_audit/cases/`
- 审计结构化结果：`results/evaluation/latest15_m07_gt_audit/summary.json`

颜色：

- 蓝色：人工GT；
- 灰色：traditional产生的全部候选；
- 绿色：M07保留且命中GT的候选；
- 红色：M07保留但未命中GT的候选。

## 汇报建议

对外应报告：M07在latest15开发集覆盖14/15个GT，剩余1个GT没有传统候选。完整主流程
三seed结果为14/0/1、14/1/1、14/0/1。不要把15个positive trigger candidate写成15个
独立GT，也不要用trigger小框代替PDF预设最终框。

若后续能补标，应优先增加独立正常照片，并在每个GT中记录`expected_text`、
`observed_text`、错误类型和复核状态。现有大框可保留用于区域级评估，同时增加精确字符
框用于定位评估。
