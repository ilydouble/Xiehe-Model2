# 第一批侧面脊柱补标注流程

## 结论

新回传的 `yolo11l_lateral_23cls_best/weights/best.pt` 已确认是23类、每个椎体4个角点的YOLO Pose模型。它在第二批验证集上表现很好，但第一批存在明显域差异：30张分层样本中，使用已有C7-L5 polygon作代理真值时，三种输入视图的召回都只有约59%。因此本流程生成的是**人工复核候选标注**，不是可直接训练的最终真值。

第一批有404张PNG，但只有384张有严格同stem LabelMe JSON。脚本只处理这384对；20张无配对JSON的图片写入 `unpaired_images.csv`，不会猜测或套用同目录中的其他JSON。

## 安全边界

- `/Volumes/E/spine_data/LAT202511` 全程只读。
- 输出到workspace下独立目录；已存在时脚本拒绝覆盖。
- 原JSON中的C7-L5、S1、CFH、T13、L6以及其他shape全部保留。
- 只追加当前缺失的C2-C7/T1-T12/L1-L5 polygon。
- 老数据的18个 `C2 circle` 是股骨头辅助标记，不算颈椎C2；可与新增的 `C2 polygon` 共存。
- 每个新增shape都带 `pseudo_label=true`、`needs_review=true` 和质量分级。
- 候选图片是指向E盘原图的符号链接，不复制或改写原图。

## 三视图推理

每张图最多推理三次，并把坐标统一映射回原图：

1. 原图；
2. 保留2个缩略采样像素安全余量的连续黑边裁剪图；
3. 全高、宽高比约0.353并以现有椎体polygon居中的训练比例视图。

同一类别优先选择至少两个视图位置一致的候选。质量等级只决定复核顺序，即使是 `high` 也必须人工确认。

## 使用方法

先做检查，不运行推理、不写文件：

```bash
./5-inference/run_pseudo_label_first_lateral.sh
```

小样本试标：

```bash
./5-inference/run_pseudo_label_first_lateral.sh \
  --selection-file /path/to/sample_list.txt \
  --output datasets/lateral_first_batch_pseudolabel_sample \
  --preview-limit 30 \
  --apply
```

全量生成候选：

```bash
./5-inference/run_pseudo_label_first_lateral.sh --apply
```

若本机Python位置不同，可显式指定：

```bash
PYTHON_BIN=/path/to/python3 ./5-inference/run_pseudo_label_first_lateral.sh --apply
```

## 人工复核

用LabelMe打开输出目录中的 `candidates/`。建议按 `review_queue.csv` 的顺序先处理 `unresolved` 和 `low`，再处理 `medium`、`high`：

1. 检查C2-C6是否各有且仅有一个颈椎四边形；
2. 检查从C2到C7的上下顺序；
3. 拖动四角点贴合椎体上下终板和前后缘；
4. 删除错误候选，补画未检出的类别；
5. 确认后清除该shape的 `pseudo_label`、`needs_review` 旗标。

在全部复核完成前，不要把这些JSON并入正式训练集。
