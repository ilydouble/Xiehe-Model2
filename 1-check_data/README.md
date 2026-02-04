# 标注完整性分析工具

这个工具用于分析LAT202511数据集的标注完整性，生成详细的HTML可视化报告。

## 功能特性

- ✅ 统计数据集中的图像总数
- ✅ 统计有标注和无标注的样本数量
- ✅ 分析分割标注（polygon）的完整性
- ✅ 分析关键点标注（point）的完整性
- ✅ 列出所有缺少标注的样本
- ✅ 列出分割标注不完整的样本（显示前50个）
- ✅ 列出关键点标注不完整的样本（显示前50个）
- ✅ 生成带有可视化图表的HTML报告

## 使用方法

### 1. 运行完整分析

```bash
cd 1-check_data
python3 analyze_dataset.py --dataset ../datasets/LAT202511 --labels ../datasets/LAT202511/label.txt --output annotation_report.html
```

### 2. 查看分析摘要（终端输出）

```bash
cd 1-check_data
python3 print_summary.py
```

这会在终端显示一个美观的摘要报告，包括：
- 基本统计信息
- 标注状态分布
- 各标签缺失情况的可视化条形图
- 最需要关注的样本列表

### 3. 参数说明

- `--dataset`: 数据集根目录路径（默认: `datasets/LAT202511`）
- `--labels`: 标签文件路径（默认: `datasets/LAT202511/label.txt`）
- `--output`: 输出HTML报告路径（默认: `1-check_data/annotation_report.html`）

### 3. 输出文件

运行后会生成两个文件：

1. **annotation_report.json** - 详细的分析数据（JSON格式）
2. **annotation_report.html** - 可视化HTML报告

## 报告内容

### 数据概览
- 总样本数
- 总图像数
- 有标注样本数
- 无标注样本数
- 分割标注不完整样本数
- 关键点标注不完整样本数

### 可视化图表
1. **标注状态分布** - 饼图显示有标注vs无标注的比例
2. **完整性统计** - 柱状图显示不完整标注的数量
3. **分割标注缺失统计** - 按标签统计缺失情况
4. **关键点标注缺失统计** - 按标签统计缺失情况

### 详细列表
1. **无标注样本列表** - 列出所有没有JSON标注文件的样本
2. **分割标注不完整样本** - 按缺失数量排序，显示前50个
3. **关键点标注不完整样本** - 按缺失数量排序，显示前50个

## 标注类型说明

### 分割标注（Polygon）
用于标注椎体的轮廓，shape_type为"polygon"

### 关键点标注（Point）
用于标注关键点位置，shape_type为"point"

### 期望的标签
根据`label.txt`文件，期望的标签包括：
- C7
- CFH
- L1, L2, L3, L4, L5
- S1
- T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13

## 文件说明

- `analyze_dataset.py` - 主分析脚本
- `generate_html_report.py` - HTML报告生成器
- `check_annotations.py` - 旧版检查脚本（针对特定样本列表）
- `generate_report_html.py` - 旧版HTML生成器

## 示例输出

```
期望的标签: ['C7', 'CFH', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'T1', 'T10', 'T11', 'T12', 'T13', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']
开始分析数据集: ../datasets/LAT202511
进度: 50/391
进度: 100/391
...
分析完成，共 391 个样本
分析结果已保存到: annotation_report.json
HTML报告已生成: annotation_report.html
```

## 注意事项

1. 确保数据集路径正确
2. 确保label.txt文件存在且格式正确
3. 生成的HTML报告需要网络连接来加载Chart.js库
4. 如果样本数量很多，分析可能需要一些时间

## 技术栈

- Python 3
- Chart.js 4.4.0（用于图表可视化）
- 纯HTML/CSS/JavaScript（无需额外依赖）

