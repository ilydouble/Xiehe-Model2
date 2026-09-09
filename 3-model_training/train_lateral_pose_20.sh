#!/usr/bin/env bash
# 人工筛选合并20类侧面脊柱 YOLO Pose 训练入口

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

show_help() {
    cat <<'EOF'
人工筛选合并20类侧面脊柱 YOLO Pose

类别: C2、C7、T1-T13、L1-L5（不含C3-C6）
训练结束后评估test集全部椎体。

用法:
    ./3-model_training/train_lateral_pose_20.sh --best [选项]

预设:
    --best        YOLO11l-Pose随机初始化，混合数据，300轮，1280，batch 4（默认）

选项:
    --full-frame-only    只使用796张原图，不加入185张黑边ROI
    --model <path/name>  高级选项；覆盖默认随机初始化结构
    --epochs <num>       训练轮数
    --imgsz <size>       输入尺寸
    --batch <size>       batch
    --device <id>        GPU，默认0
    --workers <num>      DataLoader进程，默认8
    --name <name>        实验名
    --data <path>        data.yaml路径
    --resume [path]      续训
    --skip-final-test    不执行训练后的全脊柱test评估
    --dry-run            只做全量数据校验
EOF
}

MODEL="yolo11l-pose.yaml"
EPOCHS=300
IMGSZ=1280
BATCH=4
DEVICE="0"
WORKERS=8
NAME="yolo11l_lateral_20cls_scratch_best"
DATA="${SCRIPT_DIR}/lateral_pose_20_black_roi_mixed.yaml"
RESUME=""
DRY_RUN=false
SKIP_FINAL_TEST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --best) MODEL="yolo11l-pose.yaml"; EPOCHS=300; IMGSZ=1280; BATCH=4; NAME="yolo11l_lateral_20cls_scratch_best"; shift ;;
        --full-frame-only) DATA="${PROJECT_ROOT}/datasets/yolo_lateral_reviewed_combined_20cls/data.yaml"; NAME="yolo11l_lateral_20cls_scratch_best_full_frame"; shift ;;
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
echo "初始化: ${MODEL}（随机初始化，不加载迁移权重）"
echo "配置: epochs=${EPOCHS}, imgsz=${IMGSZ}, batch=${BATCH}, device=${DEVICE}"
echo "实验: ${NAME}"

[[ -f "${DATA}" ]] || { echo "数据集配置不存在: ${DATA}" >&2; exit 1; }
if [[ "${MODEL}" == */* && ! -f "${MODEL}" && -z "${RESUME}" ]]; then
    echo "模型文件不存在: ${MODEL}" >&2
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
    if ! ${SKIP_FINAL_TEST}; then
        echo "全脊柱测试结果: ${SCRIPT_DIR}/runs/pose/${NAME}_test_all_spine"
    fi
fi
