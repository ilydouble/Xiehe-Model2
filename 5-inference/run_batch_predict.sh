#!/bin/bash
# 批量预测启动脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "🚀 批量预测测试"
echo "================================================================================"

# 默认参数
CORNER_MODEL="../3-model_training/runs/pose/yolo11s_corner_standard/weights/best.pt"
CFH_MODEL="../4-model_training_CFH/runs/cfh_detection/standard/weights/best.pt"
IMAGE_DIR="../datasets/yolo_corner/images"
OUTPUT_DIR="../datasets/prediction_results"
NUM_SAMPLES=10
CONF=0.25
CREATE_GRID="--create-grid"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --corner-model)
            CORNER_MODEL="$2"
            shift 2
            ;;
        --cfh-model)
            CFH_MODEL="$2"
            shift 2
            ;;
        --image-dir)
            IMAGE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --no-grid)
            CREATE_GRID=""
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --corner-model <path>   Corner模型路径"
            echo "  --cfh-model <path>      CFH模型路径"
            echo "  --image-dir <path>      图像目录"
            echo "  --output-dir <path>     输出目录"
            echo "  --num-samples <num>     测试样本数量 (默认: 10)"
            echo "  --conf <threshold>      置信度阈值 (默认: 0.25)"
            echo "  --no-grid               不创建对比网格图"
            echo "  --help                  显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                      # 使用默认参数"
            echo "  $0 --num-samples 20     # 测试20个样本"
            echo "  $0 --conf 0.5           # 使用0.5置信度阈值"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 显示配置
echo -e "${BLUE}配置信息:${NC}"
echo "  Corner模型: $CORNER_MODEL"
echo "  CFH模型: $CFH_MODEL"
echo "  图像目录: $IMAGE_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  样本数量: $NUM_SAMPLES"
echo "  置信度阈值: $CONF"
echo ""

# 检查模型文件
if [ ! -f "$CORNER_MODEL" ]; then
    echo -e "${YELLOW}⚠️  Corner模型不存在: $CORNER_MODEL${NC}"
    echo "请先训练Corner模型或指定正确的模型路径"
    exit 1
fi

if [ ! -f "$CFH_MODEL" ]; then
    echo -e "${YELLOW}⚠️  CFH模型不存在: $CFH_MODEL${NC}"
    echo "请先训练CFH模型或指定正确的模型路径"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo -e "${YELLOW}⚠️  图像目录不存在: $IMAGE_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 所有检查通过${NC}"
echo ""

# 运行预测
echo -e "${BLUE}开始预测...${NC}"
echo ""

python3 batch_predict.py \
    --corner-model "$CORNER_MODEL" \
    --cfh-model "$CFH_MODEL" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --conf "$CONF" \
    $CREATE_GRID

# 完成
echo ""
echo "================================================================================"
echo -e "${GREEN}✅ 批量预测完成！${NC}"
echo "================================================================================"
echo ""
echo "查看结果:"
echo "  Corner结果: open $OUTPUT_DIR/corner_only/"
echo "  CFH结果: open $OUTPUT_DIR/cfh_only/"
echo "  组合结果: open $OUTPUT_DIR/combined/"
echo "  对比网格: open $OUTPUT_DIR/grid_*.png"
echo ""

