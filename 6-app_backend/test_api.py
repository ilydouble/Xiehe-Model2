#!/usr/bin/env python3
"""
测试API输出格式
"""
import sys
sys.path.insert(0, '.')

from models import Point, VertebraDetection, CFHDetection
from keypoints_service import compute_keypoints

# 辅助函数：创建椎体
def make_vertebra(label, y_start):
    corners = [
        Point(x=0.1, y=y_start),
        Point(x=0.2, y=y_start),
        Point(x=0.1, y=y_start + 0.05),
        Point(x=0.2, y=y_start + 0.05)
    ]
    return VertebraDetection(
        label=label,
        corners=corners,
        confidence=0.9,
        bbox=[0.1, y_start, 0.2, y_start + 0.05],
        keypoints=corners  # 使用相同的corners
    )

# 创建测试数据
vertebrae = [
    make_vertebra("T1", 0.1),
    make_vertebra("C2", 0.2),
    make_vertebra("C7", 0.3),
    make_vertebra("L1", 0.6),
    make_vertebra("L4", 0.75),
    make_vertebra("L5", 0.85),
]

cfh = CFHDetection(
    center=Point(x=0.5, y=0.95),
    confidence=0.9,
    bbox=[0.5, 0.95, 0.05, 0.05]
)

# 测试计算
result = compute_keypoints(
    vertebrae=vertebrae,
    cfh=cfh,
    image_width=1920,
    image_height=1080,
    image_id="test_image"
)

# 输出JSON
import json
output = result.model_dump()
print(json.dumps(output, indent=2, ensure_ascii=False))

# 验证格式
print("\n" + "="*80)
print("✅ 格式验证:")
print(f"  - imageId: {output['imageId']}")
print(f"  - imageWidth: {output['imageWidth']}")
print(f"  - imageHeight: {output['imageHeight']}")
print(f"  - standardDistance: {output['standardDistance']}")
print(f"  - 指标数量: {len(output['measurements'])}")
print("\n指标列表:")
for m in output['measurements']:
    print(f"  - {m['type']}: {len(m['points'])} 个点")

