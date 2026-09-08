#!/usr/bin/env bash
# 人工筛选合并20类侧面脊柱 YOLO Pose 训练入口

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRANSFER_MODEL="${SCRIPT_DIR}/runs/pose/yolo11l_lateral_23cls_best/weights/best.pt"

show_help() {
    cat <<'EOF'
人工筛选合并20类侧面脊柱 YOLO Pose

类别: C2、C7、T1-T13、L1-L5（不含C3-C6）
训练结束后默认只用test集的C2/C7计算最终重点指标。

用法:
    ./3-model_training/train_lateral_pose_20.sh [预设] [选项]

预设:
    --quick       YOLO11n-Pose，10轮，640，batch 8（流程检查）
    --standard    第二批23类best.pt迁移，200轮，1280，batch 2（默认）
    --best        第二批23类best.pt迁移，300轮，1280，batch 2
    --base        从官方yolo11l-pose.pt初始化，不使用第二批best.pt

选项:
    --model <path/name>  指定初始化模型
    --epochs <num>       训练轮数
    --imgsz <size>       输入尺寸
    --batch <size>       batch
    --device <id>        GPU，默认0
    --workers <num>      DataLoader进程，默认8
    --name <name>        实验名
    --data <path>        data.yaml路径
    --resume [path]      续训
    --skip-final-test    不执行训练后的C2/C7 test评估
    --dry-run            只做全量数据校验
EOF
}

MODEL="${TRANSFER_MODEL}"
EPOCHS=200
IMGSZ=1280
BATCH=2
DEVICE="0"
WORKERS=8
NAME="yolo11l_lateral_reviewed_20cls"
DATA="${PROJECT_ROOT}/datasets/yolo_lateral_reviewed_combined_20cls/data.yaml"
RESUME=""
DRY_RUN=false
SKIP_FINAL_TEST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) MODEL="yolo11n-pose.pt"; EPOCHS=10; IMGSZ=640; BATCH=8; NAME="yolo11n_lateral_reviewed_20cls_quick"; shift ;;
        --standard) MODEL="${TRANSFER_MODEL}"; EPOCHS=200; IMGSZ=1280; BATCH=2; NAME="yolo11l_lateral_reviewed_20cls"; shift ;;
        --best) MODEL="${TRANSFER_MODEL}"; EPOCHS=300; IMGSZ=1280; BATCH=2; NAME="yolo11l_lateral_reviewed_20cls_best"; shift ;;
        --base) MODEL="${SCRIPT_DIR}/yolo11l-pose.pt"; NAME="yolo11l_lateral_reviewed_20cls_base"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --resume) RESUME="auto"; if [[ $# -gt 1 && "$2" != --* ]]; then RESUME="$2"; shift 2; else shift; fi ;;
        --skip-final-test) SKIP_FINAL_TEST=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "未知参数: $1" >&2; show_help; exit 1 ;;
    esac
done

echo "数据集: ${DATA}"
echo "初始化: ${MODEL}"
echo "配置: epochs=${EPOCHS}, imgsz=${IMGSZ}, batch=${BATCH}, device=${DEVICE}"
echo "实验: ${NAME}"

[[ -f "${DATA}" ]] || { echo "数据集配置不存在: ${DATA}" >&2; exit 1; }
if [[ "${MODEL}" == */* && ! -f "${MODEL}" && -z "${RESUME}" ]]; then
    echo "初始化权重不存在: ${MODEL}" >&2
    echo "请把第二批best.pt保留在对应路径，或用 --base / --model 指定权重。" >&2
    exit 1
fi

COMMAND=(
    python3 "${SCRIPT_DIR}/train_lateral_pose_20.py"
    --data "${DATA}" --model "${MODEL}" --epochs "${EPOCHS}"
    --imgsz "${IMGSZ}" --batch "${BATCH}" --device "${DEVICE}"
    --workers "${WORKERS}" --name "${NAME}"
)
[[ -n "${RESUME}" ]] && COMMAND+=(--resume "${RESUME}")
${SKIP_FINAL_TEST} && COMMAND+=(--skip-final-test)
${DRY_RUN} && COMMAND+=(--dry-run)
"${COMMAND[@]}"

if ! ${DRY_RUN}; then
    echo "最佳权重: ${SCRIPT_DIR}/runs/pose/${NAME}/weights/best.pt"
    ${SKIP_FINAL_TEST} || echo "C2/C7测试结果: ${SCRIPT_DIR}/runs/pose/${NAME}_test_C2_C7"
fi
