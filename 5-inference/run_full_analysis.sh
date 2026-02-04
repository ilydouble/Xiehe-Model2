#!/bin/bash
# 完整分析流程：批量预测 + 生成HTML报告

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo "================================================================================"
echo "🚀 完整模型分析流程"
echo "================================================================================"
echo ""

# 默认参数
NUM_SAMPLES=10
CONF=0.25
MAX_IMAGES=4

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --max-images)
            MAX_IMAGES="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --num-samples <num>     测试样本数量 (默认: 10)"
            echo "  --conf <threshold>      置信度阈值 (默认: 0.25)"
            echo "  --max-images <num>      HTML中最多显示的图像数量 (默认: 4)"
            echo "  --help                  显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                      # 使用默认参数"
            echo "  $0 --num-samples 20     # 测试20个样本"
            echo "  $0 --conf 0.5 --max-images 6  # 自定义参数"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}配置信息:${NC}"
echo "  测试样本数量: $NUM_SAMPLES"
echo "  置信度阈值: $CONF"
echo "  HTML显示图像: $MAX_IMAGES"
echo ""

# 步骤1: 批量预测
echo "================================================================================"
echo -e "${PURPLE}步骤 1/2: 批量预测${NC}"
echo "================================================================================"
echo ""

./run_batch_predict.sh --num-samples "$NUM_SAMPLES" --conf "$CONF"

echo ""
echo -e "${GREEN}✅ 批量预测完成${NC}"
echo ""

# 步骤2: 生成HTML报告
echo "================================================================================"
echo -e "${PURPLE}步骤 2/2: 生成HTML报告${NC}"
echo "================================================================================"
echo ""

python3 generate_html_report.py --max-images "$MAX_IMAGES"

echo ""
echo -e "${GREEN}✅ HTML报告生成完成${NC}"
echo ""

# 完成
echo "================================================================================"
echo -e "${GREEN}🎉 完整分析流程完成！${NC}"
echo "================================================================================"
echo ""
echo "📊 查看结果:"
echo "  1. HTML报告: open model_analysis_report.html"
echo "  2. Corner结果: open ../datasets/prediction_results/corner_only/"
echo "  3. CFH结果: open ../datasets/prediction_results/cfh_only/"
echo "  4. 组合结果: open ../datasets/prediction_results/combined/"
echo ""
echo "💡 提示:"
echo "  - HTML报告包含完整的统计信息和可视化结果"
echo "  - 可以直接在浏览器中查看，无需额外软件"
echo "  - 报告文件可以分享给其他人查看"
echo ""

# 自动打开HTML报告
read -p "是否现在打开HTML报告? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open model_analysis_report.html
    echo -e "${GREEN}✅ 已在浏览器中打开报告${NC}"
fi

echo ""
echo "完成！"

