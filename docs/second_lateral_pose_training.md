# 第二批侧位脊柱 23 类 YOLO Pose 训练说明

## 数据集

本地已生成：

```text
datasets/yolo_lateral_20260903_23cls/
├── data.yaml
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── manifest.csv
├── excluded_samples.csv
└── conversion_report.json
```

模型类别为 `C2-C7、T1-T12、L1-L5`，共23类。每个椎体统一为4个关键点，顺序是左上、右上、右下、左下。

共纳入698张图像、697名患者，按患者拆分为558张训练、70张验证、70张测试；8例明确漏标、内部缺层或疑似错标样本保存在 `excluded_samples.csv`，没有加入训练集。

T13在本批没有样本，因此不属于当前23类模型。S1是两点线段，CFH/FH是单点或双点，也没有混入本模型。

## 上传到 AutoDL

`datasets/` 被Git忽略，因此只执行 `git clone` 或 `git pull` 不会得到训练图像。需要把整个目录单独上传到AutoDL项目下，保证服务器结构为：

```text
<项目根目录>/datasets/yolo_lateral_20260903_23cls/data.yaml
<项目根目录>/3-model_training/train_lateral_pose_23.py
<项目根目录>/3-model_training/train_lateral_pose_23.sh
```

推荐把项目和数据放在 `/root/autodl-tmp/`，避免系统盘空间不足。

## 安装环境

在项目根目录运行：

```bash
python3 -m pip install -r 3-model_training/requirements.txt
```

先做不启动GPU的完整数据检查：

```bash
./3-model_training/train_lateral_pose_23.sh --standard --dry-run
```

检查结果应显示698张图像和16,032个椎体对象。

## 推荐训练命令

从官方YOLO11m Pose权重开始：

```bash
./3-model_training/train_lateral_pose_23.sh --standard --device 0
```

如果已把旧18类模型上传到服务器，可以利用其脊柱特征做迁移学习：

```bash
./3-model_training/train_lateral_pose_23.sh --old-transfer --device 0
```

旧权重只能用于初始化；由于类别从18变为23，新的检测/Pose输出头必须重新训练。

如果显存不足，先把 `--batch` 改为2；仍不足时再把 `--imgsz` 改为1024。全脊柱图像纵向很长，建议优先保留1280分辨率。

脚本默认不缓存解码后的高分辨率图像，以避免额外占用十几GB磁盘。AutoDL数据盘空间充足且希望减少重复解码时，可以添加 `--cache disk`。

## 断点续训

默认实验的断点续训：

```bash
./3-model_training/train_lateral_pose_23.sh --resume
```

指定其他检查点：

```bash
./3-model_training/train_lateral_pose_23.sh \
  --resume runs/pose/yolo11m_lateral_23cls/weights/last.pt
```

最佳权重默认输出到：

```text
runs/pose/yolo11m_lateral_23cls/weights/best.pt
```

部署这个模型前，还需要把后端的类别映射从旧18类更新为新23类；否则推理结果的类别名称会错位。

## 重新生成数据集

如果以后修订源LabelMe或复核表，先将旧输出目录移走，再运行：

```bash
python3 scripts/convert_second_lateral_pose.py
```

转换器不会覆盖已有输出，也不会改动外接盘上的原始LabelMe文件。
