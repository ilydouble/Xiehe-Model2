#!/bin/bash
# 一键运行标注完整性分析

echo "=========================================="
echo "  标注完整性分析工具"
echo "=========================================="
echo ""

# 设置默认参数
DATASET="../datasets/LAT202511"
LABELS="../datasets/LAT202511/label.txt"
OUTPUT="annotation_report.html"

# 检查数据集是否存在
if [ ! -d "$DATASET" ]; then
    echo "❌ 错误: 数据集目录不存在: $DATASET"
    exit 1
fi

# 检查标签文件是否存在
if [ ! -f "$LABELS" ]; then
    echo "❌ 错误: 标签文件不存在: $LABELS"
    exit 1
fi

echo "📁 数据集路径: $DATASET"
echo "🏷️  标签文件: $LABELS"
echo "📄 输出文件: $OUTPUT"
echo ""

# 运行分析
echo "🔍 开始分析..."
python3 analyze_dataset.py --dataset "$DATASET" --labels "$LABELS" --output "$OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 分析完成！"
    echo ""
    echo "=========================================="
    echo "  分析摘要"
    echo "=========================================="
    echo ""
    
    # 显示摘要
    python3 print_summary.py
    
    echo ""
    echo "=========================================="
    echo "  下一步操作"
    echo "=========================================="
    echo ""
    echo "📊 查看详细HTML报告:"
    echo "   open $OUTPUT"
    echo ""
    echo "📋 查看JSON数据:"
    echo "   cat annotation_report.json | python3 -m json.tool | less"
    echo ""
else
    echo ""
    echo "❌ 分析失败，请检查错误信息"
    exit 1
fi

