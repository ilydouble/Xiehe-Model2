#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SPINE_MODEL="$PROJECT_ROOT/3-model_training/runs/pose/yolo11l_lateral_20cls_scratch_best/weights/best.pt"
PELVIS_MODEL="$PROJECT_ROOT/4-model_training_CFH/runs/yolo11l_pelvis_3kpt_roi_mixed_best/weights/best.pt"
IMAGE_DIR="$PROJECT_ROOT/datasets/yolo_lateral_reviewed_combined_20cls/images/test"
OUTPUT_DIR="$PROJECT_ROOT/datasets/lateral_combined_predictions"
LIMIT=0
CONFIDENCE=0.10
KEYPOINT_CONFIDENCE=0.25
IOU=0.70
IMGSZ=1280
DEVICE=cpu
INFERENCE_PYTHON="${INFERENCE_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --spine-model|--corner-model) SPINE_MODEL="$2"; shift 2 ;;
        --pelvis-model|--cfh-model) PELVIS_MODEL="$2"; shift 2 ;;
        --image-dir) IMAGE_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --limit|--num-samples) LIMIT="$2"; shift 2 ;;
        --confidence|--conf) CONFIDENCE="$2"; shift 2 ;;
        --keypoint-confidence) KEYPOINT_CONFIDENCE="$2"; shift 2 ;;
        --iou) IOU="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --help)
            cat <<'EOF'
用法: ./run_batch_predict.sh [选项]

默认同时运行最终20类整脊柱Pose模型和pelvis三关键点Pose模型。

  --spine-model PATH          20类脊柱模型权重
  --pelvis-model PATH         pelvis三关键点模型权重
  --image-dir PATH            输入图像目录
  --output-dir PATH           输出目录
  --limit N                   只处理排序后的前N张；0表示全部
  --confidence FLOAT          检测置信度阈值，默认0.10
  --keypoint-confidence FLOAT 关键点显示阈值，默认0.25
  --iou FLOAT                 NMS IoU阈值，默认0.70
  --imgsz N                   推理尺寸，默认1280
  --device DEVICE             cpu、mps或CUDA设备号

兼容旧参数名：--corner-model、--cfh-model、--num-samples、--conf。
EOF
            exit 0
            ;;
        *) echo "未知参数: $1" >&2; exit 2 ;;
    esac
done

for path in "$SPINE_MODEL" "$PELVIS_MODEL"; do
    if [[ ! -f "$path" ]]; then
        echo "模型不存在: $path" >&2
        exit 1
    fi
done
if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "图像目录不存在: $IMAGE_DIR" >&2
    exit 1
fi

if ! "$INFERENCE_PYTHON" -c 'import ultralytics' >/dev/null 2>&1; then
    if [[ -x /opt/miniconda3/bin/python3 ]] && /opt/miniconda3/bin/python3 -c 'import ultralytics' >/dev/null 2>&1; then
        INFERENCE_PYTHON=/opt/miniconda3/bin/python3
    else
        echo "找不到安装了ultralytics的Python；请激活模型环境或设置INFERENCE_PYTHON。" >&2
        exit 1
    fi
fi

"$INFERENCE_PYTHON" "$SCRIPT_DIR/batch_predict.py" \
    --spine-model "$SPINE_MODEL" \
    --pelvis-model "$PELVIS_MODEL" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --limit "$LIMIT" \
    --confidence "$CONFIDENCE" \
    --keypoint-confidence "$KEYPOINT_CONFIDENCE" \
    --iou "$IOU" \
    --imgsz "$IMGSZ" \
    --device "$DEVICE"

echo "联合可视化: $OUTPUT_DIR/index.html"
echo "结构化结果: $OUTPUT_DIR/predictions.jsonl"
