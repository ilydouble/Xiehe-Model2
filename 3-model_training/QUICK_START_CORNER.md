# Corner数据集训练快速指南

## 🚀 一键开始

### 本地测试（快速验证）

```bash
# 1. 测试环境
python3 test_corner_pipeline.py

# 2. 快速训练（nano模型，50轮，约10-15分钟）
./train_corner.sh --config quick_test

# 3. 推理测试
python3 predict_corner.py \
    --model runs/pose/yolo11n_corner_quick_test/weights/best.pt \
    --source ../datasets/yolo_corner/images/ \
    --output runs/predict_corner_test
```

### 服务器训练（推荐）

```bash
# 1. 上传代码和数据到服务器
scp -r Model1/ user@server:/path/to/

# 2. SSH登录服务器
ssh user@server

# 3. 进入训练目录
cd /path/to/Model1/3-model_training

# 4. 标准训练（small模型，100轮）
./train_corner.sh --config standard --device 0

# 或者高精度训练（medium模型，150轮）
./train_corner.sh --config high_accuracy --device 0
```

## 📊 数据集信息

- **名称**: yolo_corner
- **类别**: 18个椎体（C7, T1-T12, L1-L5）
- **关键点**: 每个椎体4个角点
- **图像**: 368张
- **标注**: 6610个椎体

## ⚙️ 配置选择

| 场景 | 配置 | 模型 | 轮数 | 时间估计 | 推荐设备 |
|------|------|------|------|---------|---------|
| 快速测试 | `quick_test` | nano | 50 | 10-15分钟 | 本地GPU |
| 标准训练 | `standard` | small | 100 | 30-45分钟 | 服务器GPU |
| 高精度 | `high_accuracy` | medium | 150 | 1-2小时 | 服务器GPU |
| 最佳性能 | `best_performance` | large | 200 | 2-4小时 | 服务器GPU |

## 📁 重要文件

```
训练相关:
├── train_corner.py              # 训练脚本
├── train_corner.sh              # 训练启动脚本
├── train_configs_corner.yaml    # 配置文件
└── test_corner_pipeline.py      # 测试脚本

推理相关:
└── predict_corner.py            # 推理脚本

数据集:
└── ../datasets/yolo_corner/
    ├── data.yaml                # 数据集配置
    ├── images/                  # 图像
    └── labels/                  # 标注

结果:
└── runs/pose/yolo11*_corner_*/
    ├── weights/best.pt          # 最佳模型
    ├── weights/last.pt          # 最后模型
    └── results.png              # 训练曲线
```

## 🎯 常用命令

### 训练

```bash
# 标准训练
./train_corner.sh --config standard

# 指定GPU
./train_corner.sh --config standard --device 0

# 继续训练
./train_corner.sh --config standard --resume

# 从头训练（不用预训练权重）
./train_corner.sh --config from_scratch
```

### 推理

```bash
# 单张图像
python3 predict_corner.py \
    --model runs/pose/yolo11s_corner_standard/weights/best.pt \
    --source image.png

# 整个目录
python3 predict_corner.py \
    --model runs/pose/yolo11s_corner_standard/weights/best.pt \
    --source ../datasets/yolo_corner/images/

# 保存文本结果
python3 predict_corner.py \
    --model runs/pose/yolo11s_corner_standard/weights/best.pt \
    --source image.png \
    --save-txt

# 调整置信度阈值
python3 predict_corner.py \
    --model runs/pose/yolo11s_corner_standard/weights/best.pt \
    --source image.png \
    --conf 0.3
```

## 🔧 故障排除

### GPU内存不足

```bash
# 方法1: 减小批次大小（编辑train_configs_corner.yaml）
batch: 8  # 改为4或2

# 方法2: 减小图像大小
imgsz: 640  # 改为512或480

# 方法3: 使用更小的模型
./train_corner.sh --config quick_test  # 使用nano模型
```

### 训练中断

```bash
# 从上次中断处继续
./train_corner.sh --config standard --resume
```

### 找不到CUDA

```bash
# 使用CPU训练（慢）
./train_corner.sh --config quick_test --device cpu
```

## 📈 监控训练

### 查看训练曲线

```bash
# 训练过程中会自动保存
open runs/pose/yolo11s_corner_standard/results.png
```

### 查看验证结果

```bash
# 查看验证批次的可视化
open runs/pose/yolo11s_corner_standard/val_batch0_pred.jpg
```

### 实时监控（TensorBoard）

```bash
# 如果安装了tensorboard
tensorboard --logdir runs/pose
```

## 💡 最佳实践

1. **先快速测试**: 使用 `quick_test` 配置验证流程
2. **再标准训练**: 使用 `standard` 配置获得基准性能
3. **最后高精度**: 如果需要更高精度，使用 `high_accuracy` 或 `best_performance`
4. **保存检查点**: 训练会自动保存 `best.pt` 和 `last.pt`
5. **验证结果**: 训练后用推理脚本验证效果

## 🎓 下一步

训练完成后：

1. **评估模型**: 查看 `results.png` 中的指标
2. **推理测试**: 在测试集上运行推理
3. **部署模型**: 将 `best.pt` 用于生产环境
4. **迭代优化**: 根据结果调整配置

## 📞 需要帮助？

- 检查 `README_CORNER.md` 获取详细文档
- 运行 `python3 test_corner_pipeline.py` 诊断问题
- 查看训练日志了解错误信息

