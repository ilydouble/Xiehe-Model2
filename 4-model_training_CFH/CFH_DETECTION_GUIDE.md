# CFH检测系统使用指南

## 📋 目录

1. [系统概述](#系统概述)
2. [快速开始](#快速开始)
3. [数据集准备](#数据集准备)
4. [模型训练](#模型训练)
5. [模型推理](#模型推理)
6. [常见问题](#常见问题)

## 系统概述

### 功能

这是一个基于YOLO11的股骨头（CFH）检测系统，可以：

- ✅ 将CFH关键点转换为检测框
- ✅ 训练YOLO检测模型
- ✅ 进行单张/批量推理
- ✅ 可视化检测结果
- ✅ 导出CSV格式结果

### 文件结构

```
4-inference/
├── convert_cfh_to_detection.py    # 数据集转换脚本
├── train_cfh_detection.py         # 训练脚本
├── train_cfh_detection.sh         # 训练启动脚本
├── predict_cfh_detection.py       # 推理脚本
└── CFH_DETECTION_GUIDE.md         # 本文档

datasets/
└── yolo_cfh_detection/            # CFH检测数据集
    ├── data.yaml                  # 数据集配置
    ├── images/                    # 图像（370张）
    ├── labels/                    # 标注（370个）
    ├── visualizations/            # 可视化样本
    └── README.md                  # 数据集文档
```

## 快速开始

### 1️⃣ 生成数据集

```bash
cd 4-inference

# 使用默认参数（1.8倍椎体大小）
python3 convert_cfh_to_detection.py

# 自定义包围盒大小
python3 convert_cfh_to_detection.py --bbox-multiplier 2.0

# 可视化样本
python3 -c "from convert_cfh_to_detection import visualize_samples; visualize_samples('../datasets/yolo_cfh_detection', 5)"
```

### 2️⃣ 快速测试训练

```bash
# 快速测试（10轮，小模型）
./train_cfh_detection.sh --quick

# 查看结果
open runs/cfh_detection/quick_test
```

### 3️⃣ 标准训练

```bash
# 标准配置（150轮，中等模型）
./train_cfh_detection.sh --standard

# 或使用Python脚本
python3 train_cfh_detection.py --model m --epochs 150 --imgsz 800 --batch 8
```

### 4️⃣ 推理测试

```bash
# 单张图像
python3 predict_cfh_detection.py \
    --model runs/cfh_detection/train/weights/best.pt \
    --source ../datasets/yolo_cfh_detection/images/sample.png

# 批量推理
python3 predict_cfh_detection.py \
    --model runs/cfh_detection/train/weights/best.pt \
    --source ../datasets/yolo_cfh_detection/images/ \
    --batch-csv cfh_results.csv
```

## 数据集准备

### 转换参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入目录 | `../datasets/LAT202511` |
| `--output` | 输出目录 | `../datasets/yolo_cfh_detection` |
| `--bbox-multiplier` | 包围盒倍数 | `1.8` |

### 包围盒大小选择

- **1.5x**: 紧凑的检测框，适合精确定位
- **1.8x**: 标准大小（推荐）
- **2.0x**: 较大的检测框，容错率高
- **2.5x**: 很大的检测框，适合初步筛查

### 数据集统计

```
图像数量: 370张
CFH数量: 370个
平均检测框: 198 x 277 像素
归一化大小: 0.1143 x 0.0593
```

## 模型训练

### 预设配置

#### 快速测试
```bash
./train_cfh_detection.sh --quick
```
- 模型: YOLO11n
- 轮数: 10
- 图像: 640
- 批次: 16
- 用途: 快速验证流程

#### 标准训练
```bash
./train_cfh_detection.sh --standard
```
- 模型: YOLO11m
- 轮数: 150
- 图像: 800
- 批次: 8
- 用途: 日常训练（推荐）

#### 最佳性能
```bash
./train_cfh_detection.sh --best
```
- 模型: YOLO11l
- 轮数: 200
- 图像: 1024
- 批次: 4
- 用途: 追求最高精度

### 自定义训练

```bash
python3 train_cfh_detection.py \
    --model m \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --name my_experiment
```

### 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model` | 模型大小 (n/s/m/l/x) | m |
| `--epochs` | 训练轮数 | 150 |
| `--imgsz` | 图像大小 | 800 |
| `--batch` | 批次大小 | 8 |
| `--device` | GPU设备 | 0 |

### 预期性能

| 模型 | mAP50 | mAP50-95 | 速度 | 显存 |
|------|-------|----------|------|------|
| YOLO11n | 0.80-0.85 | 0.55-0.60 | 快 | 2GB |
| YOLO11s | 0.85-0.90 | 0.60-0.65 | 较快 | 4GB |
| YOLO11m | 0.90-0.95 | 0.65-0.70 | 中等 | 6GB |
| YOLO11l | 0.92-0.97 | 0.70-0.75 | 较慢 | 8GB |

## 模型推理

### 单张图像推理

```bash
python3 predict_cfh_detection.py \
    --model runs/cfh_detection/train/weights/best.pt \
    --source image.png \
    --conf 0.25
```

### 批量推理

```bash
python3 predict_cfh_detection.py \
    --model runs/cfh_detection/train/weights/best.pt \
    --source images_dir/ \
    --conf 0.25
```

### 导出CSV结果

```bash
python3 predict_cfh_detection.py \
    --model runs/cfh_detection/train/weights/best.pt \
    --source images_dir/ \
    --batch-csv results.csv
```

CSV格式：
```csv
image,x1,y1,x2,y2,confidence,center_x,center_y
sample.png,100,200,300,400,0.95,200,300
```

## 常见问题

### Q1: 检测框太小/太大？

**解决方案**: 调整包围盒倍数重新生成数据集

```bash
# 更大的框
python3 convert_cfh_to_detection.py --bbox-multiplier 2.5

# 更小的框
python3 convert_cfh_to_detection.py --bbox-multiplier 1.5
```

### Q2: 训练精度不够？

**解决方案**:
1. 使用更大的模型: `--model l`
2. 增加训练轮数: `--epochs 200`
3. 增加图像大小: `--imgsz 1024`
4. 检查数据集质量

### Q3: 训练速度慢？

**解决方案**:
1. 使用更小的模型: `--model n`
2. 减小图像大小: `--imgsz 640`
3. 减小批次大小: `--batch 4`
4. 使用更好的GPU

### Q4: 显存不足？

**解决方案**:
1. 减小批次大小: `--batch 4` 或 `--batch 2`
2. 减小图像大小: `--imgsz 640`
3. 使用更小的模型: `--model s`

### Q5: 如何查看训练进度？

```bash
# 查看训练日志
cat runs/cfh_detection/train/results.csv

# 查看训练曲线
open runs/cfh_detection/train/results.png

# 实时监控（如果使用TensorBoard）
tensorboard --logdir runs/cfh_detection
```

## 最佳实践

### 1. 数据集准备
- ✅ 先用默认参数（1.8x）生成数据集
- ✅ 可视化检查几个样本
- ✅ 如果检测框不合适，调整倍数重新生成

### 2. 模型训练
- ✅ 先用 `--quick` 快速测试流程
- ✅ 确认无误后用 `--standard` 正式训练
- ✅ 如需更高精度，使用 `--best`

### 3. 模型评估
- ✅ 查看验证集mAP指标
- ✅ 在测试图像上可视化检测结果
- ✅ 检查误检和漏检情况

### 4. 模型部署
- ✅ 使用best.pt模型
- ✅ 根据应用场景调整置信度阈值
- ✅ 批量推理时使用CSV导出功能

---

**创建时间**: 2026-01-07  
**版本**: 1.0  
**状态**: ✅ 已完成

