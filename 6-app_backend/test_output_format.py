#!/usr/bin/env python3
"""
测试输出格式
"""
import json
from models import Point, VertebraDetection, CFHDetection
from keypoints_service import compute_keypoints

# 创建简单的测试数据
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
        keypoints=corners
    )

vertebrae = [
    make_vertebra("T1", 0.1),
    make_vertebra("C2", 0.2),
    make_vertebra("C7", 0.3),
    make_vertebra("T2", 0.35),
    make_vertebra("T5", 0.45),
    make_vertebra("T12", 0.6),
    make_vertebra("L1", 0.65),
    make_vertebra("L4", 0.8),
    make_vertebra("L5", 0.85),
]

cfh = CFHDetection(
    center=Point(x=0.5, y=0.95),
    confidence=0.9,
    bbox=[0.5, 0.95, 0.05, 0.05]
)

# 测试：图像尺寸 1920x1080
result = compute_keypoints(
    vertebrae=vertebrae,
    cfh=cfh,
    image_width=1920,
    image_height=1080,
    image_id="test"
)

# 输出JSON
output = result.model_dump()
print(json.dumps(output, indent=2, ensure_ascii=False))

# 验证
print("\n" + "="*80)
print("✅ 格式验证:")
print(f"  imageId: {output['imageId']}")
print(f"  imageWidth: {output['imageWidth']} (应该是 1920)")
print(f"  imageHeight: {output['imageHeight']} (应该是 1080)")
print(f"  standardDistance: {output['standardDistance']}")
print(f"\n指标数量: {len(output['measurements'])}")
print("\n各指标点数:")
for m in output['measurements']:
    print(f"  {m['type']:20s}: {len(m['points'])} 个点")
    # 检查坐标是否是像素坐标（应该 > 1）
    if m['points']:
        first_point = m['points'][0]
        is_pixel = first_point['x'] > 1 or first_point['y'] > 1
        coord_type = "像素坐标 ✅" if is_pixel else "归一化坐标 ❌"
        print(f"      第一个点: ({first_point['x']:.2f}, {first_point['y']:.2f}) - {coord_type}")

print("\n预期点数:")
print("  T1 Slope: 2 个点")
print("  C2-C7 CL: 4 个点")
print("  TK T2-T5: 4 个点")
print("  TK T5-T12: 4 个点")
print("  LL L1-S1: 4 个点")
print("  LL L1-L4: 4 个点")
print("  LL L4-S1: 4 个点")
print("  SVA: 2 个点")
print("  TPA: 7 个点 (T1四个角点 + CFH + S1左 + S1右)")
print("  PI: 3 个点 (CFH + S1左 + S1右)")
print("  PT: 3 个点 (CFH + S1左 + S1右)")
print("  SS: 2 个点")

