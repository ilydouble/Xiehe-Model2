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

训练结构与原CFH模块保持一致：`.sh` 负责常用预设和环境检查，Python脚本负责数据全量校验与训练。
在项目根目录安装与检查：

```bash
pip install -r 4-model_training_CFH/requirements.txt

./4-model_training_CFH/train_pelvis_pose_3kpt.sh --standard --dry-run
```

常用预设为 `--quick`、`--standard` 和 `--best`。如果先完成了 23 类侧面脊柱模型训练，推荐使用
`--spine-transfer`。它会迁移已经学到的侧位片特征；由于本任务是1类、3关键点，Ultralytics
会按新数据配置重建并训练新的 Pose head：

```bash
./4-model_training_CFH/train_pelvis_pose_3kpt.sh \
  --spine-transfer \
  --device 0 \
  --batch 4
```

如果不使用上一阶段权重，执行 `--standard` 即从官方 `yolo11m-pose.pt` 初始化。显存不足时，
增加 `--batch 2`；图像较高且关键点小，不建议先降低 `imgsz`。所有参数仍可直接传给Python入口做
更细的控制。

断点续训：

```bash
./4-model_training_CFH/train_pelvis_pose_3kpt.sh --resume
```

默认输出目录为 `runs/pose/yolo11m_pelvis_3kpt_all/`，最终权重位于
`weights/best.pt`。训练前脚本会自动全量检查数据格式、图片标签配对和患者 split。
