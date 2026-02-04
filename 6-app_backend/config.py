"""
配置文件
"""
from pathlib import Path

# 当前目录（6-app_backend）
CURRENT_DIR = Path(__file__).parent

# 模型路径（本地models文件夹）
CORNER_MODEL_PATH = CURRENT_DIR / "models/corner_model.pt"
CFH_MODEL_PATH = CURRENT_DIR / "models/cfh_model.pt"

# 推理参数
CORNER_CONF_THRESHOLD = 0.2  # Corner模型置信度阈值
CFH_CONF_THRESHOLD = 0.1    # CFH模型置信度阈值

# 服务器配置
HOST = "0.0.0.0"
PORT = 8000

# 椎体类别名称映射
VERTEBRA_NAMES = {
    0: 'C7',
    1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5',
    6: 'T1', 7: 'T2', 8: 'T3', 9: 'T4', 10: 'T5', 11: 'T6',
    12: 'T7', 13: 'T8', 14: 'T9', 15: 'T10', 16: 'T11', 17: 'T12'
}

# 反向映射
VERTEBRA_IDS = {v: k for k, v in VERTEBRA_NAMES.items()}

