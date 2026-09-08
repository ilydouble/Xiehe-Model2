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

生成后执行独立完整性审计：

```bash
./5-inference/run_pseudo_label_first_lateral.sh --audit-only
```

全量命令会同时创建扁平的 `labelme_review/`。如果已有候选输出但没有该目录，可单独补建：

```bash
./5-inference/run_pseudo_label_first_lateral.sh --prepare-review-only
```

若本机Python位置不同，可显式指定：

```bash
PYTHON_BIN=/path/to/python3 ./5-inference/run_pseudo_label_first_lateral.sh --apply
```

## 人工复核

用LabelMe打开输出目录中的 `labelme_review/`，即可连续翻阅全部384张。建议在表格软件中按 `review_queue.csv` 的 `status` 和 `quality` 筛选，先处理 `unresolved` 和 `low`，再处理 `medium`、`high`：

1. 检查C2-C6是否各有且仅有一个颈椎四边形；
2. 检查从C2到C7的上下顺序；
3. 拖动四角点贴合椎体上下终板和前后缘；
4. 删除错误候选，补画未检出的类别；
5. 确认后清除该shape的 `pseudo_label`、`needs_review` 旗标。

在全部复核完成前，不要把这些JSON并入正式训练集。

## 当前全量输出（2026-09-08）

输出目录为 `datasets/lateral_first_batch_pseudolabel_23cls`：

- 处理384张严格配对图像；20张无同stem JSON的图片隔离；
- 新增1,942个缺失polygon候选；100个类别没有可靠候选；
- high 1,547、medium 231、low 164，三种等级都需要人工复核；
- 354张具有完整C2-C6候选，30张至少缺一类；
- `--audit-only` 全量检查通过，错误数0。

以下6张完全没有C2-C6候选，应直接人工绘制：WANG_YANG两张、XU_LONG_YUE、YANG_HUI_JUN、YU_CHU_YI、ZHAO_WEN_QI。精确文件名可在 `review_queue.csv` 中筛选 `status=no_candidate` 查看。

## E盘完整复核可视化（2026-09-08）

全量复核包位于 `/Volumes/E/spine_data/LAT202511_侧面补标人工复核可视化_384份_20260908`。其中每张JPG左侧为带全部C2-L5层级名的完整侧位片，右侧为C2-C7放大图；原始人工polygon为青色，模型补标high/medium/low分别为绿、橙、红。

双击 `打开此文件逐张人工复核.html` 可按缺候选、含low、含medium、仅high筛选，保存人工判断并导出CSV。包内另有 `人工复核索引.csv`、`manifest.json`、`audit_report.json` 和 `未纳入补标_无同名JSON_20张.csv`。终验确认384张JPG全部可解码且SHA-256匹配；20张无同名JSON原图没有候选标注，单独列清单而没有伪造空JSON。

## 当前严格C2-C6接受/拒绝复核包（2026-09-08）

用户最终确认不因原始C7-L5缺级自动排除病例。新版复核包位于 `/Volumes/E/spine_data/LAT202511_仅补C2-C6_人工接受拒绝复核包_20260908`，384张配对样本全部进入人工判断：

- 仅显示模型新增的C2、C3、C4、C5、C6；C7-L5只显示原始人工polygon，绝不显示模型对这些层级的补齐；
- 354张具有完整C2-C6候选，30张至少缺一个候选；新增候选共1,849个，其中C2/C3/C4/C5/C6分别为366/377/366/370/370个；
- C2状态为安全318张、触边33张、近边15张、无候选18张；原始C7-L5缺级27张，这些都只用于筛选和提示，不自动接受或排除；
- HTML支持“接受、拒绝、修改后接受、不确定”，结论保存在当前浏览器并可导出 `仅补C2-C6人工接受拒绝结果.csv`；
- 独立审计确认384张预览全部可解码、SHA-256全部匹配、伪标签类别越界为0，AppleDouble旁车已清理。

旧候选目录包含对其他缺失级别的模型输出，不应直接用作严格C2-C6训练标签。待人工CSV返回后，再按结论生成最终LabelMe数据；在此之前原始数据和候选JSON均保持不变。

## yolo_corner口径纠正版（最终使用）

用户进一步确认第一批训练基线是 `datasets/yolo_corner`，不是LAT中全部同stem LabelMe样本。因此上面的384张LAT复核包保留作历史检查，但不得用于构建yolo_corner补标数据；最终应使用：

`/Volumes/E/spine_data/yolo_corner_仅补C2-C6_人工接受拒绝复核包_368张_20260908`

纠正版严格覆盖yolo_corner的368张图：294 train、37 val、37 test。青色为现有18类YOLO Pose四角点，绿/橙/红仅为C2-C6候选。358张复用既有模型候选，10张单独补推理；新增候选共1,770个，C2/C3/C4/C5/C6分别为352/361/350/354/353，另有70个无候选项分布在29张图中。

HTML还可筛选10张“无同stem JSON”启发式旧标签错配风险、4个跨split患者所涉8张图、1组精确重复所涉2张图、2张基线缺级以及C2触边/近边。后续全量反向来源审计证明该启发式低估：实际有15张标签来自其他JSON，其中1张是逐字节相同的重复图，其余14张必须补全23类。当前复核包应暂停作为最终依据，完整名单见 `docs/yolo_corner_label_source_audit.md`。最终仍需另建23类数据集并按原始患者目录ID重新划分，不能覆盖或直接沿用当前yolo_corner split。

重新生成或审计命令：

```bash
/opt/miniconda3/bin/python3 5-inference/render_yolo_corner_c2c6_review.py
/opt/miniconda3/bin/python3 5-inference/render_yolo_corner_c2c6_review.py --audit-only
```
