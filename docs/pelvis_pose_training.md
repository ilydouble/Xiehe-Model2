# 联合骨盆三关键点 YOLO Pose 训练说明

## 任务定义

这是一个关键点定位任务，不是三个检测类别。每张侧位片有一个 `pelvis` 对象，关键点固定为：

1. `CFH`：股骨头综合中心；
2. `S1_left`：图像坐标中较左的 S1 终板端点；
3. `S1_right`：图像坐标中较右的 S1 终板端点。

新批次只有 `FH-1`、`FH-2` 时取两点中点作为 CFH。水平翻转使用
`flip_idx=[0,2,1]`，CFH 不换索引，两个 S1 端点互换。

## 数据集

本地生成目录是 `datasets/yolo_pelvis_3kpt_all`，包含：

- 1,028 张图像、1,023 个患者组；
- 824 train、102 val、102 test；
- 旧批 334 张、新批 694 张；
- 569 个直接 CFH、459 个 FH 双点中点；
- 每张图一个对象，每行 14 个 YOLO Pose 字段。

`datasets/` 被 Git 忽略，因此拉取仓库代码不会自动得到图像。连接 AutoDL 后，需把整个
`datasets/yolo_pelvis_3kpt_all` 目录单独上传到服务器项目的 `datasets/` 下。

若需在另一台本地机器重新生成：

```bash
python3 scripts/convert_combined_pelvis_pose.py \
  --old-source /Volumes/E/spine_data/LAT202511 \
  --new-source '/Volumes/E/spine_data/20260903-侧面数据第二批/labelme-export/P202607076308'
```

转换器不会改写原始数据，且若输出目录已存在会直接停止，避免覆盖。

## AutoDL 训练

在项目根目录安装与检查：

```bash
pip install -r 4-model_training_CFH/requirements.txt

python3 4-model_training_CFH/train_pelvis_pose_3kpt.py --dry-run
```

如果先完成了 23 类侧面脊柱模型训练，推荐用其 `best.pt` 初始化。这会迁移已经学到的侧位片
特征；由于本任务是 1 类、3 关键点，Ultralytics 会按新数据配置重建并训练新的 Pose head：

```bash
python3 4-model_training_CFH/train_pelvis_pose_3kpt.py \
  --model runs/pose/yolo11m_lateral_23cls/weights/best.pt \
  --device 0 \
  --imgsz 1280 \
  --batch 4 \
  --epochs 200
```

如果不使用上一阶段权重，省略 `--model` 即默认从官方 `yolo11m-pose.pt` 初始化。显存不足时，
优先把 `--batch 4` 改为 `--batch 2`；图像较高且关键点小，不建议先降低 `imgsz`。

断点续训：

```bash
python3 4-model_training_CFH/train_pelvis_pose_3kpt.py --resume
```

默认输出目录为 `runs/pose/yolo11m_pelvis_3kpt_all/`，最终权重位于
`weights/best.pt`。训练前脚本会自动全量检查数据格式、图片标签配对和患者 split。
