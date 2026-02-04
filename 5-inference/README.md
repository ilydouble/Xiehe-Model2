# 批量预测与可视化

## 📋 概述

这个目录包含批量预测脚本，可以同时使用Corner模型和CFH检测模型对图像进行预测，并生成可视化结果和HTML分析报告。

## 📁 文件说明

- **batch_predict.py**: 批量预测Python脚本
- **generate_html_report.py**: 生成HTML报告脚本
- **run_batch_predict.sh**: 批量预测启动脚本
- **run_full_analysis.sh**: 完整分析流程（预测+报告）
- **README.md**: 本文档

## 🚀 快速开始

### 方式1: 一键完整分析（推荐）

```bash
cd 5-inference
./run_full_analysis.sh
```

这会自动完成：
1. ✅ 批量预测（默认10个样本）
2. ✅ 生成HTML报告
3. ✅ 在浏览器中打开报告

### 方式2: 仅批量预测

```bash
cd 5-inference
./run_batch_predict.sh
```

默认配置：
- 测试样本：10张
- 置信度阈值：0.25
- 自动创建对比网格图

### 方式3: 自定义参数

```bash
# 测试20个样本，置信度0.5
./run_full_analysis.sh --num-samples 20 --conf 0.5

# 测试5个样本，HTML显示6张图
./run_full_analysis.sh --num-samples 5 --max-images 6
```

## 📊 输出结果

### 1. HTML分析报告

**文件**: `model_analysis_report.html`

包含内容：
- ✅ 执行摘要和项目目标
- ✅ 预测结果统计（图表展示）
- ✅ 两个模型的详细对比
- ✅ 技术对比表格
- ✅ 可视化预测结果（嵌入式图像）
- ✅ 精美的响应式设计

**特点**：
- 📱 响应式设计，支持移动端
- 🎨 渐变色彩，视觉效果好
- 📊 统计卡片，数据清晰
- 🖼️ 嵌入式图像，无需外部文件
- 📤 可直接分享给他人

### 2. 预测结果图像

运行后会在 `datasets/prediction_results/` 目录下生成以下内容：

```
prediction_results/
├── corner_only/           # Corner模型单独结果
│   ├── corner_image1.png
│   ├── corner_image2.png
│   └── ...
├── cfh_only/             # CFH模型单独结果
│   ├── cfh_image1.png
│   ├── cfh_image2.png
│   └── ...
├── combined/             # 组合结果（两个模型）
│   ├── combined_image1.png
│   ├── combined_image2.png
│   └── ...
├── grid_image1.png       # 对比网格图（3列对比）
├── grid_image2.png
├── ...
└── prediction_stats.json # 统计信息
```

## 🎨 可视化说明

### Corner模型结果
- ✅ 彩色边界框：不同椎体用不同颜色
- ✅ 关键点：每个椎体的4个角点
- ✅ 连线：连接4个角点形成四边形
- ✅ 标签：椎体名称（C7, T1-T12, L1-L5）和置信度

### CFH模型结果
- ✅ 绿色边界框：股骨头检测框
- ✅ 红色中心点：股骨头中心位置
- ✅ 标签：CFH和置信度

### 组合结果
- ✅ 同时显示Corner和CFH的检测结果
- ✅ CFH用紫色边界框区分
- ✅ 包含图例说明

### 对比网格图
- ✅ 3列对比：Corner | CFH | Combined
- ✅ 每列有标题
- ✅ 方便直观对比两个模型的效果

## 📈 统计信息

`prediction_stats.json` 包含：

```json
{
  "total_images": 5,
  "corner_detections": 101,
  "corner_keypoints": 404,
  "cfh_detections": 5
}
```

- **total_images**: 处理的图像数量
- **corner_detections**: 检测到的椎体总数
- **corner_keypoints**: 检测到的关键点总数
- **cfh_detections**: 检测到的股骨头总数

## 🔧 高级用法

### 使用Python脚本直接调用

```bash
python3 batch_predict.py \
    --corner-model ../3-model_training/runs/pose/yolo11s_corner_standard/weights/best.pt \
    --cfh-model ../4-model_training_CFH/runs/cfh_detection/standard/weights/best.pt \
    --image-dir ../datasets/yolo_corner/images \
    --output-dir ../datasets/prediction_results \
    --num-samples 10 \
    --conf 0.25 \
    --create-grid
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--corner-model` | Corner模型路径 | `../3-model_training/runs/pose/.../best.pt` |
| `--cfh-model` | CFH模型路径 | `../4-model_training_CFH/runs/.../best.pt` |
| `--image-dir` | 图像目录 | `../datasets/yolo_corner/images` |
| `--output-dir` | 输出目录 | `../datasets/prediction_results` |
| `--num-samples` | 测试样本数量 | 10 |
| `--conf` | 置信度阈值 | 0.25 |
| `--create-grid` | 创建对比网格图 | 启用 |
| `--no-grid` | 不创建对比网格图 | - |

## 📊 查看结果

### 在macOS上

```bash
# 查看所有结果
open ../datasets/prediction_results/

# 查看Corner结果
open ../datasets/prediction_results/corner_only/

# 查看CFH结果
open ../datasets/prediction_results/cfh_only/

# 查看组合结果
open ../datasets/prediction_results/combined/

# 查看对比网格图
open ../datasets/prediction_results/grid_*.png
```

### 在Linux上

```bash
# 使用默认图像查看器
xdg-open ../datasets/prediction_results/

# 或使用特定查看器
eog ../datasets/prediction_results/grid_*.png
```

## 💡 使用建议

### 推荐工作流程

1. **首次测试**: 使用完整分析流程，快速查看效果
   ```bash
   ./run_full_analysis.sh --num-samples 5
   ```

2. **调整参数**: 根据结果调整置信度和样本数
   ```bash
   ./run_full_analysis.sh --num-samples 20 --conf 0.5
   ```

3. **查看HTML报告**: 在浏览器中查看详细分析
   ```bash
   open model_analysis_report.html
   ```

4. **分享结果**: HTML报告可以直接分享给他人
   - 文件包含所有图像（base64编码）
   - 无需额外文件，单文件即可查看
   - 支持移动端浏览

### 高级用法

**仅批量预测（不生成HTML）**:
```bash
./run_batch_predict.sh --num-samples 10 --conf 0.25
```

**仅生成HTML报告（使用已有预测结果）**:
```bash
python3 generate_html_report.py --max-images 6
```

**自定义模型路径**:
```bash
./run_batch_predict.sh \
    --corner-model path/to/corner/model.pt \
    --cfh-model path/to/cfh/model.pt \
    --num-samples 10
```

## 🎯 实际应用示例

### 示例1: 快速验证模型效果

```bash
# 测试5个样本，快速查看效果
./run_batch_predict.sh --num-samples 5
open ../datasets/prediction_results/combined/
```

### 示例2: 评估模型精度

```bash
# 测试30个样本，评估整体性能
./run_batch_predict.sh --num-samples 30 --conf 0.3
cat ../datasets/prediction_results/prediction_stats.json
```

### 示例3: 对比不同置信度

```bash
# 低置信度
./run_batch_predict.sh --num-samples 10 --conf 0.1 --output-dir ../datasets/results_conf01

# 高置信度
./run_batch_predict.sh --num-samples 10 --conf 0.5 --output-dir ../datasets/results_conf05
```

## 📝 注意事项

1. **模型路径**: 确保Corner和CFH模型已训练完成
2. **图像格式**: 支持PNG和JPG格式
3. **内存占用**: 大批量预测可能占用较多内存
4. **处理时间**: 每张图像约需1-2秒

## 🔍 故障排查

### 问题1: 模型文件不存在

```
⚠️  Corner模型不存在
```

**解决方案**: 先训练模型或指定正确路径
```bash
cd ../3-model_training
./train_corner.sh --config standard
```

### 问题2: 没有检测结果

**解决方案**: 降低置信度阈值
```bash
./run_batch_predict.sh --conf 0.1
```

### 问题3: 内存不足

**解决方案**: 减少样本数量
```bash
./run_batch_predict.sh --num-samples 5
```

---

**创建时间**: 2026-01-07  
**版本**: 1.0  
**状态**: ✅ 已完成

