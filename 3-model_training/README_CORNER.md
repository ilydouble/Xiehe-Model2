# YOLO11 Pose训练 - Corner数据集

用于训练脊柱椎体角点检测模型（18个椎体类别，每个4个关键点）

## 📊 数据集信息

- **数据集**: `yolo_corner`
- **类别数量**: 18个椎体类别
- **类别列表**: C7, T1-T12, L1-L5
- **关键点**: 每个椎体4个角点
- **图像数量**: 368张
- **标注数量**: 6610个椎体

## 🚀 快速开始

### 1. 训练模型

使用默认配置（standard）训练：

```bash
./train_corner.sh
```

使用其他配置：

```bash
# 快速测试（nano模型，50轮）
./train_corner.sh --config quick_test

# 高精度（medium模型，150轮）
./train_corner.sh --config high_accuracy

# 最佳性能（large模型，200轮）
./train_corner.sh --config best_performance
```

指定GPU设备：

```bash
./train_corner.sh --config standard --device 0
```

从上次中断处继续训练：

```bash
./train_corner.sh --config standard --resume
```

### 2. 推理

对单张图像推理：

```bash
python3 predict_corner.py \
    --model runs/pose/yolo11s_corner_standard/weights/best.pt \
    --source path/to/image.png \
    --output runs/predict_corner
```

对整个目录推理：

```bash
python3 predict_corner.py \
    --model runs/pose/yolo11s_corner_standard/weights/best.pt \
    --source path/to/images/ \
    --output runs/predict_corner
```

保存文本结果：

```bash
python3 predict_corner.py \
    --model runs/pose/yolo11s_corner_standard/weights/best.pt \
    --source path/to/image.png \
    --save-txt
```

## 📁 文件结构

```
3-model_training/
├── train_configs_corner.yaml  # 训练配置文件
├── train_corner.py            # 训练脚本
├── train_corner.sh            # 训练启动脚本
├── predict_corner.py          # 推理脚本
└── README_CORNER.md           # 本文档

datasets/
└── yolo_corner/
    ├── data.yaml              # 数据集配置
    ├── images/                # 图像目录
    └── labels/                # 标注目录

runs/
└── pose/
    └── yolo11*_corner_*/      # 训练结果
        ├── weights/
        │   ├── best.pt        # 最佳模型
        │   └── last.pt        # 最后一轮模型
        └── results.png        # 训练曲线
```

## ⚙️ 配置说明

### 可用配置

| 配置名称 | 模型大小 | 训练轮数 | 批次大小 | 图像大小 | 说明 |
|---------|---------|---------|---------|---------|------|
| `quick_test` | nano | 50 | 16 | 640 | 快速测试 |
| `standard` | small | 100 | 16 | 640 | 标准训练（推荐） |
| `high_accuracy` | medium | 150 | 8 | 800 | 高精度 |
| `best_performance` | large | 200 | 4 | 1024 | 最佳性能 |
| `from_scratch` | small | 300 | 16 | 640 | 从头训练 |

### 自定义配置

编辑 `train_configs_corner.yaml` 文件来自定义训练参数：

```yaml
# 数据增强
augmentation:
  hsv_h: 0.015      # 色调增强
  hsv_s: 0.7        # 饱和度增强
  hsv_v: 0.4        # 亮度增强
  translate: 0.1    # 平移增强
  scale: 0.5        # 缩放增强
  fliplr: 0.5       # 左右翻转

# 优化器
optimizer:
  optimizer: 'AdamW'
  lr0: 0.001        # 初始学习率
  lrf: 0.01         # 最终学习率因子
  weight_decay: 0.0005

# 损失函数权重
loss_weights:
  box: 7.5          # 边界框损失
  cls: 0.5          # 分类损失
  pose: 12.0        # 关键点损失
```

## 📈 训练监控

训练过程中会自动保存：

1. **训练曲线**: `runs/pose/yolo11*_corner_*/results.png`
2. **验证结果**: `runs/pose/yolo11*_corner_*/val_batch*.jpg`
3. **模型权重**: `runs/pose/yolo11*_corner_*/weights/`

## 🎯 推理结果

推理脚本会在图像上绘制：

1. **边界框**: 每个椎体的检测框（不同颜色）
2. **类别标签**: 椎体名称和置信度
3. **关键点**: 4个角点（带编号）
4. **连线**: 连接4个角点形成矩形

## 💡 提示

1. **GPU内存不足**: 减小 `batch` 或 `imgsz`
2. **训练速度慢**: 使用更小的模型（nano或small）
3. **精度不够**: 使用更大的模型（medium或large）或增加训练轮数
4. **过拟合**: 增加数据增强或减少训练轮数

## 🔧 故障排除

### 问题1: CUDA out of memory

```bash
# 减小批次大小
./train_corner.sh --config standard  # 然后修改配置文件中的batch
```

### 问题2: 训练中断

```bash
# 从上次中断处继续
./train_corner.sh --config standard --resume
```

### 问题3: 数据集路径错误

检查 `datasets/yolo_corner/data.yaml` 中的路径配置。

## 📞 支持

如有问题，请检查：
1. 数据集是否正确准备
2. Python环境是否安装ultralytics
3. GPU驱动是否正确安装

