#!/bin/bash
# CFH检测模型训练启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
CFH检测模型训练脚本

用法:
    $0 [选项]

选项:
    --model <size>      模型大小 (n/s/m/l/x, 默认: m)
    --epochs <num>      训练轮数 (默认: 150)
    --imgsz <size>      图像大小 (默认: 800)
    --batch <size>      批次大小 (默认: 8)
    --device <id>       GPU设备ID (默认: 0)
    --name <name>       实验名称 (默认: train)
    --quick             快速测试 (10轮，小图像)
    --help              显示此帮助信息

预设配置:
    --quick             快速测试: n模型, 10轮, 640图像, 16批次
    --standard          标准训练: m模型, 150轮, 800图像, 8批次
    --best              最佳性能: l模型, 200轮, 1024图像, 4批次

示例:
    # 快速测试
    $0 --quick

    # 标准训练
    $0 --standard

    # 自定义配置
    $0 --model m --epochs 100 --imgsz 640 --batch 16 --device 0

    # 最佳性能
    $0 --best --device 0
EOF
}

# 默认参数
MODEL="m"
EPOCHS=150
IMGSZ=800
BATCH=8
DEVICE="0"
NAME="train"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --imgsz)
            IMGSZ="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --quick)
            MODEL="n"
            EPOCHS=10
            IMGSZ=640
            BATCH=16
            NAME="quick_test"
            shift
            ;;
        --standard)
            MODEL="m"
            EPOCHS=150
            IMGSZ=800
            BATCH=8
            NAME="standard"
            shift
            ;;
        --best)
            MODEL="l"
            EPOCHS=200
            IMGSZ=800
            BATCH=4
            NAME="best"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 打印配置
echo "================================================================================"
echo "🚀 CFH检测模型训练"
echo "================================================================================"
print_info "模型大小: YOLO11${MODEL}"
print_info "训练轮数: ${EPOCHS}"
print_info "图像大小: ${IMGSZ}"
print_info "批次大小: ${BATCH}"
print_info "GPU设备: ${DEVICE}"
print_info "实验名称: ${NAME}"
echo ""

# 检查数据集
DATASET_DIR="../datasets/yolo_cfh_detection"
if [ ! -d "$DATASET_DIR" ]; then
    print_error "数据集不存在: $DATASET_DIR"
    print_info "请先运行: python3 convert_cfh_to_detection.py"
    exit 1
fi

print_success "数据集检查通过"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    print_error "Python3未安装"
    exit 1
fi

print_success "Python环境检查通过"

# 开始训练
echo ""
print_info "开始训练..."
echo ""

python3 train_cfh_detection.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --name "$NAME"

# 训练完成
echo ""
echo "================================================================================"
print_success "训练完成！"
echo "================================================================================"
print_info "查看结果: open runs/cfh_detection/${NAME}"
print_info "最佳模型: runs/cfh_detection/${NAME}/weights/best.pt"
echo ""

