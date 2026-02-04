# 脊柱分析后端服务（侧面）

基于FastAPI的侧面脊柱X光片分析服务。

## 🎯 三步实现方案

1. **步骤1**: 调用模型得到检测结果（椎体角点 + 股骨头位置）
2. **步骤2**: 根据检测结果生成各指标的测量点（关键点JSON）
3. **步骤3**: 根据关键点JSON计算所有指标（指标JSON）

## 📋 输出格式

参考正面JSON格式，按指标类型组织：

```json
{
  "imageId": "lateral_spine",
  "measurements": [
    {
      "type": "T1 Slope",
      "points": [
        {"x": 0.5, "y": 0.3},
        {"x": 0.6, "y": 0.3}
      ]
    },
    {
      "type": "Lumbar Lordosis",
      "points": [
        {"x": 0.5, "y": 0.4},
        {"x": 0.6, "y": 0.4},
        {"x": 0.5, "y": 0.7},
        {"x": 0.6, "y": 0.7}
      ]
    }
  ]
}
```

## 🔢 支持的指标

1. **T1 Slope** - T1倾斜角（2个点：T1上终板）
2. **Cervical Lordosis** - 颈椎前凸角（4个点：C2上终板 + C7下终板）
3. **Thoracic Kyphosis T2-T5** - 上胸椎后凸角（4个点）
4. **Thoracic Kyphosis T5-T12** - 主胸椎后凸角（4个点）
5. **Lumbar Lordosis** - 腰椎前凸角（4个点：L1上终板 + L5下终板）
6. **SVA** - 矢状面垂直轴（2个点：C7后上角 + S1估算点）
7. **TPA** - T1骨盆角（3个点：T1中心 + CFH中心 + S1估算点）
8. **PI** - 骨盆入射角（2个点：CFH中心 + S1估算点）
9. **PT** - 骨盆倾斜角（2个点：CFH中心 + S1估算点）
10. **SS** - 骶骨倾斜角（2个点：L5下终板）

## 📡 API接口

### 步骤1: 检测接口
- **端点**: `POST /api/detect`
- **输入**: 图像文件
- **输出**: 椎体角点 + 股骨头位置（归一化坐标）

### 步骤2: 关键点计算接口（推荐）
- **端点**: `POST /api/detect_and_keypoints`
- **输入**: 图像文件
- **输出**: 关键点JSON（按指标组织的测量点）

### 步骤3: 指标计算接口
- **端点**: `POST /api/calculate_metrics`
- **输入**: 关键点JSON（步骤2的输出）
- **输出**: 指标JSON（包含所有计算出的指标值）

## 🚀 快速开始

### 1. 安装依赖

```bash
cd 6-app_backend
pip install -r requirements.txt
```

### 2. 模型文件

模型文件已包含在 `models/` 文件夹中：

```
6-app_backend/models/
├── corner_model.pt  # Corner检测模型（YOLO11 Pose）
└── cfh_model.pt     # CFH检测模型（YOLO11 Detection）
```

如需更新模型，替换这两个文件即可。

### 3. 启动服务

```bash
# 方式1: 使用启动脚本（推荐）
./start_server.sh

# 方式2: 直接运行
python3 app.py
```

服务启动后：
- **访问地址**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

### 4. 测试服务

先启动服务，然后运行测试脚本：

```bash
# 终端1: 启动服务
./start_server.sh

# 终端2: 运行测试
cd example
python test_metrics.py \
  --image ../../datasets/yolo_corner/images/sample.png \
  --api-url http://localhost:8000 \
  --output-dir ./output
```

这将生成两个JSON文件：
- `sample_keypoints.json` - 关键点JSON（步骤2输出）
- `sample_metrics.json` - 指标JSON（步骤3输出）

## 📁 文件结构

```
6-app_backend/
├── app.py                    # FastAPI应用
├── config.py                 # 配置
├── models.py                 # 数据模型
├── inference_service.py      # 步骤1: 模型推理
├── keypoints_service.py      # 步骤2: 生成测量点JSON
├── start_server.sh           # 启动脚本
├── requirements.txt          # 依赖
├── models/                   # 模型文件
│   ├── corner_model.pt
│   └── cfh_model.pt
└── example/
    └── test_metrics.py       # 测试脚本
```

## 📞 故障排查

### 问题1: 模型文件不存在

```bash
# 检查模型文件
ls -lh models/

# 应该看到：
# corner_model.pt  (~19MB)
# cfh_model.pt     (~39MB)

# 如果文件不存在，需要从训练结果复制：
cp ../3-model_training/runs/pose/yolo11s_corner_standard/weights/best.pt models/corner_model.pt
cp ../4-model_training_CFH/runs/cfh_detection/standard/weights/best.pt models/cfh_model.pt
```

### 问题2: 端口被占用

修改 `config.py` 中的端口：
```python
PORT = 8001  # 改为其他端口
```

### 问题3: 依赖缺失

```bash
pip install -r requirements.txt
```

---

**创建时间**: 2026-01-07  
**版本**: 1.0.0  
**状态**: ✅ 已完成

