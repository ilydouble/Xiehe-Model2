#!/bin/bash
# YOLO11 Pose训练脚本 - Corner数据集
# 用于训练脊柱椎体角点检测模型

set -e

echo "=========================================="
echo "YOLO11 Pose训练 - Corner数据集"
echo "=========================================="
echo ""

# 默认配置
CONFIG="standard"
DEVICE="0"
RESUME=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--config CONFIG] [--device DEVICE] [--resume]"
            echo ""
            echo "可用配置:"
            echo "  quick_test       - 快速测试 (nano, 50 epochs)"
            echo "  standard         - 标准训练 (small, 100 epochs) [默认]"
            echo "  high_accuracy    - 高精度 (medium, 150 epochs)"
            echo "  best_performance - 最佳性能 (large, 200 epochs)"
            echo "  from_scratch     - 从头训练 (small, 300 epochs)"
            exit 1
            ;;
    esac
done

echo "配置: $CONFIG"
echo "设备: $DEVICE"
echo "继续训练: ${RESUME:-否}"
echo ""

# 检查数据集
if [ ! -f "../datasets/yolo_corner/data.yaml" ]; then
    echo "❌ 错误: 数据集配置文件不存在"
    echo "   请确保 ../datasets/yolo_corner/data.yaml 存在"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

# 检查ultralytics
if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo "❌ 错误: 未安装ultralytics"
    echo "   请运行: pip install ultralytics"
    exit 1
fi

# 开始训练
echo "开始训练..."
echo ""

python3 train_corner.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    $RESUME

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""
echo "模型保存在: runs/pose/yolo11*_corner_*/weights/"
echo ""
echo "下一步:"
echo "  1. 查看训练结果: runs/pose/yolo11*_corner_*/results.png"
echo "  2. 使用最佳模型推理:"
echo "     python3 predict_corner.py --model runs/pose/yolo11*_corner_*/weights/best.pt --source <图像路径>"
echo ""

