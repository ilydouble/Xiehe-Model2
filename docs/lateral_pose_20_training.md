# 人工筛选联合侧面20类 Pose 训练说明

## 数据集定义

- 路径：`datasets/yolo_lateral_reviewed_combined_20cls`
- 类别顺序：`C2、C7、T1-T13、L1-L5`，共20类
- 明确不含：`C3-C6`
- 每个椎体4个关键点：左上、右上、右下、左下
- 重点指标：C2和C7

数据集共995张：第一批297张人工接受且来源正确的图，第二批698张清洗后的图。拆分为train 796、val 99、test 100，按患者分组且没有跨集合患者。14张已确认旧标签错配图和1张等价重复图均不在训练清单中。

T13只有第一批的2个对象，两个患者都固定在train。因此T13可以参与辅助训练，但val/test不能给出可靠的T13泛化指标；本轮最终指标只看C2和C7。

## 上传到AutoDL

`datasets/`被Git忽略，因此不能只拉取代码。需要上传下面两项：

1. 整个`datasets/yolo_lateral_reviewed_combined_20cls`目录（约3.6 GB）。
2. 第二批训练得到的`3-model_training/runs/pose/yolo11l_lateral_23cls_best/weights/best.pt`（约51 MB）。

服务器上保持相同相对路径，训练脚本就能直接找到数据和迁移权重。

## 训练

先检查数据而不启动GPU：

```bash
cd /root/autodl-tmp/Model1
./3-model_training/train_lateral_pose_20.sh --standard --dry-run
```

标准训练：

```bash
./3-model_training/train_lateral_pose_20.sh --standard
```

标准配置为YOLO11l-Pose、200轮、1280输入、batch 2，从第二批23类`best.pt`迁移。Mosaic、MixUp、Copy-Paste均关闭，保留轻微亮度/旋转/平移/缩放和水平翻转；`flip_idx`会同步交换左右角点。

显存不足时：

```bash
./3-model_training/train_lateral_pose_20.sh --standard --batch 1
```

训练结束后，脚本加载最佳权重并在test集上使用`classes=[0, 1]`只评估C2和C7。输出位置：

- 最佳权重：`3-model_training/runs/pose/yolo11l_lateral_reviewed_20cls/weights/best.pt`
- C2/C7测试：`3-model_training/runs/pose/yolo11l_lateral_reviewed_20cls_test_C2_C7`

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
