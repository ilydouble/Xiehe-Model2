#!/bin/bash
# 启动后端服务

set -e

echo "================================================================================"
echo "🚀 启动脊柱分析后端服务"
echo "================================================================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
python3 -c "import fastapi, uvicorn, ultralytics" 2>/dev/null || {
    echo "❌ 缺少依赖，请先安装:"
    echo "   pip install -r requirements.txt"
    exit 1
}

# 检查模型文件
echo "🔍 检查模型文件..."
CORNER_MODEL="models/corner_model.pt"
CFH_MODEL="models/cfh_model.pt"

if [ ! -f "$CORNER_MODEL" ]; then
    echo "❌ 错误: Corner模型不存在: $CORNER_MODEL"
    echo "   请确保模型文件存在于 models/ 文件夹中"
    exit 1
fi

if [ ! -f "$CFH_MODEL" ]; then
    echo "❌ 错误: CFH模型不存在: $CFH_MODEL"
    echo "   请确保模型文件存在于 models/ 文件夹中"
    exit 1
fi

echo "   ✅ Corner模型: $CORNER_MODEL ($(du -h $CORNER_MODEL | cut -f1))"
echo "   ✅ CFH模型: $CFH_MODEL ($(du -h $CFH_MODEL | cut -f1))"

echo ""
echo "✅ 环境检查完成"
echo ""

# 启动服务
echo "🌐 启动服务..."
echo "   访问地址: http://localhost:8000"
echo "   API文档: http://localhost:8000/docs"
echo ""

python3 app.py

