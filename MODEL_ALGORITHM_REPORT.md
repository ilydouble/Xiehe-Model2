# 脊柱模型算法与实验结果说明

本文档整理当前仓库中脊柱侧位片分析模型的算法方案、数据预处理、训练参数、推理参数和实验结果。当前系统由两个 YOLO11 子模型组成：椎体角点检测模型和股骨头中心点检测模型。两个模型的输出在后端服务中汇总，用于后续脊柱-骨盆参数计算。

## 1. 总体方案

| 模块 | 任务类型 | 输入 | 输出 | 当前部署权重 |
|------|----------|------|------|--------------|
| Corner 模型 | YOLO11 Pose | 脊柱侧位 X 光图像 | 18 类椎体检测框及每个椎体 4 个角点 | `6-app_backend/models/corner_model.pt` |
| CFH 模型 | YOLO11 Detection | 脊柱侧位 X 光图像 | 股骨头检测框及中心点 | `6-app_backend/models/cfh_model.pt` |

推理流程为：

1. 读取输入图像，保持原图尺寸。
2. Corner 模型检测 C7、T1-T12、L1-L5 共 18 类椎体，并输出每个椎体的 4 个角点。
3. CFH 模型检测股骨头区域，取置信度最高的检测框作为股骨头结果。
4. 后端将检测框、关键点和中心点归一化到 `[0, 1]`，返回给前端或指标计算模块。

## 2. Backbone 与网络结构

两个模型均使用 YOLO11 Large 规格作为主干，区别在于检测头不同：

| 模型 | 任务头 | 类别数 | 关键点配置 | 参数量 |
|------|--------|--------|------------|--------|
| Corner | Pose Head | 18 | `[4, 3]`，每个椎体 4 个点，每点包含 x/y/visibility | 26,175,338 |
| CFH | Detect Head | 1 | 无 | 25,311,251 |

YOLO11 Large 的主体结构包括：

- Backbone：多级 `Conv + C3k2` 特征提取，使用 SiLU 激活和 BatchNorm。
- 高层语义增强：`SPPF` 扩大感受野，`C2PSA` 引入注意力模块增强全局上下文。
- Neck：上采样、Concat 和多尺度 `C3k2` 融合，形成 PAN/FPN 式多尺度特征。
- Head：
  - Corner 使用 Pose Head，同时预测检测框、类别和 4 个椎体角点。
  - CFH 使用 Detect Head，只预测单类股骨头检测框。

## 3. 数据与标注预处理

### 3.1 Corner 数据集

数据集路径：`datasets/yolo_corner`

| 项目 | 数值 |
|------|------|
| 图像数 | 368 |
| 标注文件数 | 368 |
| 数据来源 | `LAT202511` |
| 类别 | C7、T1-T12、L1-L5，共 18 类 |
| 排除标签 | `CFH`、`T13`、`S1` |

预处理脚本：`2-build_dataset/convert_to_yolo_pose.py`

转换逻辑：

1. 读取原始 JSON polygon 标注。
2. 只保留椎体 polygon，排除 `CFH`、`T13`、`S1`。
3. 对每个椎体 polygon 计算轮廓中心。
4. 从轮廓点中选取相对中心呈十字对角分布的 4 个远端点作为关键点。
5. 由 polygon 的最小外接矩形生成 YOLO 检测框。
6. 输出 YOLO Pose 格式：

```text
class x_center y_center width height kp1_x kp1_y kp1_v ... kp4_x kp4_y kp4_v
```

所有坐标均按图像宽高归一化，关键点可见性 `v=2`。

### 3.2 CFH 数据集

数据集路径：`datasets/yolo_cfh_detection`

| 项目 | 数值 |
|------|------|
| 图像数 | 370 |
| 标注文件数 | 370 |
| CFH 对象数 | 370 |
| 类别 | `CFH`，单类别 |
| 默认检测框倍数 | `1.8` |

预处理脚本：`4-model_training_CFH/convert_cfh_to_detection.py`

转换逻辑：

1. 读取原始 JSON 中的 `CFH` 点标注。
2. 将单个 CFH 中心点扩展为检测框。
3. 如图像中存在椎体标注，则用平均椎体高度作为参考，检测框宽高设为 `avg_vertebra_height * bbox_multiplier`。
4. 默认 `bbox_multiplier=1.8`。
5. 输出 YOLO Detection 格式：

```text
0 center_x center_y width height
```

### 3.3 训练时图像处理与增强

训练由 Ultralytics YOLO 负责图像 resize/letterbox、多尺度批处理和归一化。当前主要增强参数如下：

| 参数 | Corner | CFH | 说明 |
|------|--------|-----|------|
| `imgsz` | 800 | 800 | 当前最佳模型训练尺寸 |
| `hsv_h` | 0.015 | 0.015 | 色调增强 |
| `hsv_s` | 0.7 | 0.7 | 饱和度增强 |
| `hsv_v` | 0.4 | 0.4 | 亮度增强 |
| `degrees` | 0.0 | 0.0 | 医学侧位片不做旋转 |
| `translate` | 0.1 | 0.1 | 平移增强 |
| `scale` | 0.5 | 0.5 | 缩放增强 |
| `flipud` | 0.0 | 0.0 | 不做上下翻转 |
| `fliplr` | 0.5 | 0.5 | 左右翻转 |
| `mosaic` | 1.0 | 1.0 | Mosaic 增强 |
| `mixup` | 0.0 | 0.0 | 不使用 MixUp |
| `multi_scale` | 0.5 | 0.5 | 多尺度训练 |

注意：当前 `data.yaml` 中 train 和 val 均指向同一图像目录，因此训练日志中的验证指标更适合作为模型收敛和相对对比依据，不应等同于严格独立测试集指标。

## 4. 训练参数

### 4.1 Corner 最佳模型

结果目录：`3-model_training/runs/pose/yolo11l_corner_best`

| 参数 | 值 |
|------|----|
| 模型 | `yolo11l-pose.pt` |
| 任务 | Pose |
| epochs | 200 |
| batch | 4 |
| imgsz | 800 |
| optimizer | AdamW |
| 初始学习率 `lr0` | 0.001 |
| 最终学习率因子 `lrf` | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 3.0 |
| box loss weight | 7.5 |
| cls loss weight | 0.5 |
| dfl loss weight | 1.5 |
| pose loss weight | 12.0 |
| kobj loss weight | 1.0 |

### 4.2 CFH 最佳模型

结果目录：`4-model_training_CFH/runs/cfh_detection/best-2`

| 参数 | 值 |
|------|----|
| 模型 | `yolo11l.pt` |
| 任务 | Detection |
| epochs | 400 |
| batch | 4 |
| imgsz | 800 |
| optimizer | AdamW |
| 初始学习率 `lr0` | 0.001 |
| 最终学习率因子 `lrf` | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| patience | 50 |
| save_period | 10 |
| workers | 8 |

## 5. 推理参数与后处理

后端配置文件：`6-app_backend/config.py`

| 参数 | 值 |
|------|----|
| Corner 置信度阈值 | 0.2 |
| CFH 置信度阈值 | 0.1 |
| 服务地址 | `0.0.0.0:8000` |

后处理逻辑：

- Corner：逐个读取检测框、类别、置信度和关键点；检测框转换为 `[cx, cy, w, h]` 归一化格式；关键点坐标除以原图宽高归一化。
- CFH：读取检测结果后取第一个检测框，即置信度最高框；中心点由检测框中心计算；检测框和中心点均归一化。
- 输出类别映射：`C7, L1-L5, T1-T12`，类别 ID 与 `datasets/yolo_corner/data.yaml` 保持一致。

## 6. 实验结果

### 6.1 Corner 模型

最佳实验：`yolo11l_corner_best`

| 指标 | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|------|-------|-----------|--------|-------|----------|
| Box | 199 | 0.98940 | 0.99032 | 0.99441 | 0.78818 |
| Pose | 199 | 0.98726 | 0.98823 | 0.99189 | 0.97495 |

与 standard 配置最终 epoch 对比：

| 模型 | Epoch | Box mAP50 | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 |
|------|-------|-----------|--------------|------------|---------------|
| `yolo11s_corner_standard` | 100 | 0.98011 | 0.70610 | 0.97008 | 0.92173 |
| `yolo11l_corner_best` | 200 | 0.99437 | 0.78799 | 0.99182 | 0.97414 |

### 6.2 CFH 模型

最佳实验：`best-2`

| 实验 | Epoch | Precision | Recall | mAP50 | mAP50-95 |
|------|-------|-----------|--------|-------|----------|
| `standard` | 150 | 0.80665 | 0.80541 | 0.87594 | 0.62204 |
| `best` | 199 | 0.88817 | 0.85676 | 0.93335 | 0.71071 |
| `best-2` | 388 | 0.95571 | 0.93316 | 0.97865 | 0.80958 |

### 6.3 联合推理测试

批量推理统计文件：`datasets/prediction_results/prediction_stats.json`

| 项目 | 数值 |
|------|------|
| 测试图像数 | 10 |
| Corner 检测数 | 204 |
| Corner 关键点数 | 816 |
| CFH 检测数 | 7 |

后端模型测试报告：`6-app_backend/MODEL_TEST_REPORT.md`

| 测试项 | 结果 |
|--------|------|
| Corner 模型加载 | 通过 |
| CFH 模型加载 | 通过 |
| Corner 推理 | 通过 |
| CFH 推理 | 通过 |
| 空白图像测试 | 通过，返回 0 个检测 |
| 真实图像测试 | Corner 检出 18 个椎体，CFH 检出 1 个股骨头 |
| API 输出格式 | 通过 |
| 坐标归一化 | 通过 |
| 指标计算 | 通过 |

## 7. 当前结论

1. 当前部署版本采用 YOLO11 Large 双模型方案，Corner 负责椎体框和 4 点定位，CFH 负责股骨头区域定位。
2. Corner 模型在当前验证设置下达到 Box mAP50 0.99441、Pose mAP50 0.99189，关键点 mAP50-95 达到 0.97495。
3. CFH 的 `best-2` 实验明显优于 `standard` 和 `best`，mAP50-95 从 0.62204 提升到 0.80958。
4. 后端测试显示当前模型可以正常加载、推理、归一化输出，并在真实样例上输出完整椎体和 CFH 结果。
5. 由于当前 train/val 使用同一数据目录，若用于论文、验收或正式性能声明，建议补充独立测试集或交叉验证结果。

## 8. 复现实验入口

Corner 训练：

```bash
cd 3-model_training
python3 train_corner.py --config best_performance --device 0
```

CFH 训练：

```bash
cd 4-model_training_CFH
python3 train_cfh_detection.py --model l --epochs 400 --imgsz 800 --batch 4 --device 0 --name best-2
```

联合推理：

```bash
cd 5-inference
python3 batch_predict.py \
  --corner-model ../3-model_training/runs/pose/yolo11l_corner_best/weights/best.pt \
  --cfh-model ../4-model_training_CFH/runs/cfh_detection/best-2/weights/best.pt \
  --image-dir ../datasets/yolo_corner/images \
  --output-dir ../datasets/prediction_results \
  --num-samples 10 \
  --conf 0.25 \
  --create-grid
```

