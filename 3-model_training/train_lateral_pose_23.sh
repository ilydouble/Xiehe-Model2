#!/usr/bin/env bash
# 第二批23类侧面脊柱 YOLO Pose 训练启动脚本

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

show_help() {
    cat <<'EOF'
第二批23类侧面脊柱 YOLO Pose 训练脚本

用法:
    ./train_lateral_pose_23.sh [预设] [选项]

预设配置:
    --quick             快速检查: YOLO11n-Pose, 10轮, 640图像, batch 8
    --standard          标准训练: YOLO11m-Pose, 200轮, 1280图像, batch 4（默认）
    --best              高精度: YOLO11l-Pose, 300轮, 1280图像, batch 2
    --old-transfer      使用原18类corner_model.pt初始化

自定义选项:
    --model <path/name> 模型文件或Ultralytics模型名
    --epochs <num>      训练轮数
    --imgsz <size>      输入尺寸
    --batch <size>      批次大小
    --device <id>       GPU设备ID，默认0
    --name <name>       实验名称
    --data <path>       data.yaml路径
    --workers <num>     数据加载进程数，默认8
    --resume [path]     续训；不提供路径时读取默认last.pt
    --dry-run           只做全量数据和参数检查，不启动训练
    --help              显示帮助

示例:
    ./train_lateral_pose_23.sh --quick
    ./train_lateral_pose_23.sh --standard --device 0
    ./train_lateral_pose_23.sh --old-transfer --batch 2
    ./train_lateral_pose_23.sh --resume
EOF
}

MODEL="yolo11m-pose.pt"
EPOCHS=200
IMGSZ=1280
BATCH=4
DEVICE="0"
NAME="yolo11m_lateral_23cls"
DATA="${PROJECT_ROOT}/datasets/yolo_lateral_20260903_23cls/data.yaml"
WORKERS=8
RESUME=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            MODEL="yolo11n-pose.pt"; EPOCHS=10; IMGSZ=640; BATCH=8
            NAME="yolo11n_lateral_23cls_quick"; shift ;;
        --standard)
            MODEL="yolo11m-pose.pt"; EPOCHS=200; IMGSZ=1280; BATCH=4
            NAME="yolo11m_lateral_23cls"; shift ;;
        --best)
            MODEL="yolo11l-pose.pt"; EPOCHS=300; IMGSZ=1280; BATCH=2
            NAME="yolo11l_lateral_23cls_best"; shift ;;
        --old-transfer)
            MODEL="${PROJECT_ROOT}/6-app_backend/models/corner_model.pt"
            NAME="yolo11m_lateral_23cls_old_transfer"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --resume)
            RESUME="auto"
            if [[ $# -gt 1 && "$2" != --* ]]; then RESUME="$2"; shift 2; else shift; fi ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) print_error "未知参数: $1"; show_help; exit 1 ;;
    esac
done

echo "================================================================================"
echo "🚀 第二批23类侧面脊柱 YOLO Pose 训练"
echo "================================================================================"
print_info "模型: ${MODEL}"
print_info "训练轮数: ${EPOCHS}"
print_info "图像大小: ${IMGSZ}"
print_info "批次大小: ${BATCH}"
print_info "GPU设备: ${DEVICE}"
print_info "实验名称: ${NAME}"
print_info "数据集: ${DATA}"
[[ -n "${RESUME}" ]] && print_info "续训: ${RESUME}"
${DRY_RUN} && print_warning "dry-run：不会启动GPU训练"
echo ""

if [[ ! -f "${DATA}" ]]; then
    print_error "数据集配置不存在: ${DATA}"
    print_info "请先上传或生成 datasets/yolo_lateral_20260903_23cls"
    exit 1
fi
print_success "数据集路径检查通过"

if ! command -v python3 >/dev/null 2>&1; then print_error "未找到python3"; exit 1; fi
if ! ${DRY_RUN} && ! python3 -c "import ultralytics" 2>/dev/null; then
    print_error "未安装ultralytics"
    print_info "请运行: pip install -r 3-model_training/requirements.txt"
    exit 1
fi
print_success "Python环境检查通过"

COMMAND=(
    python3 "${SCRIPT_DIR}/train_lateral_pose_23.py"
    --data "${DATA}" --model "${MODEL}" --epochs "${EPOCHS}"
    --imgsz "${IMGSZ}" --batch "${BATCH}" --device "${DEVICE}"
    --workers "${WORKERS}" --name "${NAME}"
)
[[ -n "${RESUME}" ]] && COMMAND+=(--resume "${RESUME}")
${DRY_RUN} && COMMAND+=(--dry-run)

echo ""
print_info "开始执行..."
"${COMMAND[@]}"

echo ""
echo "================================================================================"
print_success "流程完成"
echo "================================================================================"
if ! ${DRY_RUN}; then
    print_info "最佳模型: ${PROJECT_ROOT}/runs/pose/${NAME}/weights/best.pt"
fi
