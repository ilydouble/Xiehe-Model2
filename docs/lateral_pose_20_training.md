# 人工筛选联合侧面20类 Pose 训练说明

## 数据集定义

- 路径：`datasets/yolo_lateral_reviewed_combined_20cls`
- 类别顺序：`C2、C7、T1-T13、L1-L5`，共20类
- 明确不含：`C3-C6`
- 每个椎体4个关键点：左上、右上、右下、左下
- 重点指标：C2和C7

数据集共995张：第一批297张人工接受且来源正确的图，第二批698张清洗后的图。拆分为train 796、val 99、test 100，按患者分组且没有跨集合患者。14张已确认旧标签错配图和1张等价重复图均不在训练清单中。

默认训练还会读取`datasets/yolo_lateral_reviewed_combined_20cls_black_roi`：它只从train中的185张明显连续黑边图生成去黑边派生视图（第一批148、第二批37），并同步变换全部椎体bbox和四关键点。原始796张train图仍保留，因此实际训练视图为981张；val 99张和test 100张始终只使用原始全图。ROI与原图沿用同一患者split，不会把train患者引入val/test。

T13只有第一批的2个对象，两个患者都固定在train。因此T13可以参与辅助训练，但val/test不能给出可靠的T13泛化指标；本轮最终指标只看C2和C7。

## 上传到AutoDL

`datasets/`被Git忽略，因此不能只拉取代码。需要上传下面两项：

1. 整个`datasets/yolo_lateral_reviewed_combined_20cls`目录（约3.6 GB）。
2. 整个`datasets/yolo_lateral_reviewed_combined_20cls_black_roi`目录（约1.0 GB）。

服务器上保持相同相对路径，训练脚本就能直接找到混合数据。

## 训练

先检查数据而不启动GPU：

```bash
cd /root/autodl-tmp/Model1
./3-model_training/train_lateral_pose_20.sh --best --dry-run
```

标准训练：

```bash
./3-model_training/train_lateral_pose_20.sh --best
```

`--best`只有一个明确含义：`yolo11l-pose.yaml`建立Large Pose网络并随机初始化，300轮、1280输入、batch 4，不加载任何旧`.pt`或迁移权重。训练数据默认读取`lateral_pose_20_black_roi_mixed.yaml`，即796张原始train加185张去黑边ROI，共981个训练视图；模型结构YAML本身不决定是否混合数据。

Mosaic、MixUp、Copy-Paste均关闭，保留轻微亮度/旋转/平移/缩放和水平翻转；`flip_idx`会同步交换左右角点。

如需做不含ROI的对照实验：

```bash
./3-model_training/train_lateral_pose_20.sh --best --full-frame-only
```

显存不足时：

```bash
./3-model_training/train_lateral_pose_20.sh --best --batch 2
```

训练结束后，脚本加载最佳权重并在test集上使用`classes=[0, 1]`只评估C2和C7。输出位置：

- 最佳权重：`3-model_training/runs/pose/yolo11l_lateral_20cls_scratch_best/weights/best.pt`
- C2/C7测试：`3-model_training/runs/pose/yolo11l_lateral_20cls_scratch_best_test_C2_C7`

如需续训：

```bash
./3-model_training/train_lateral_pose_20.sh --resume
```

## 可追溯文件

- `build_report.json`：类别计数、拆分统计、角点算法与质量规则
- `manifest.csv`：每张图的批次、患者、split、现有/缺失类别
- `excluded_samples.csv`：第一批人工拒绝/错配和第二批上游排除项
- `metadata/first_batch_human_review.csv`：原始人工接受/拒绝结果
- `metadata/first_batch_label_source_audit.csv`：旧标签来源错配审计

ROI目录另外包含`manifest.csv`、`build_report.json`和`leakage_audit.json`，记录每张派生图的来源患者、原split、裁剪框和图像哈希。需要在本机重新检查或重建时：

```bash
# 只预演，不写数据
python3 scripts/build_lateral_black_roi_views.py

# 原子方式正式重建
python3 scripts/build_lateral_black_roi_views.py --apply

# 对已经生成的ROI做独立审计
python3 scripts/build_lateral_black_roi_views.py --audit-only
```
