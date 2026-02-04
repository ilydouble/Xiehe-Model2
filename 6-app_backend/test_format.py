#!/usr/bin/env python3
"""
测试新格式输出
"""
import sys
sys.path.insert(0, '.')

from models import Point, Measurement, KeypointsResponse

# 创建一个测试输出
measurements = [
    Measurement(
        type="T1 Slope",
        points=[Point(x=100.5, y=200.3), Point(x=150.2, y=210.1)]
    ),
    Measurement(
        type="C2-C7 CL",
        points=[
            Point(x=100.0, y=200.0),
            Point(x=150.0, y=200.0),
            Point(x=100.0, y=300.0),
            Point(x=150.0, y=300.0)
        ]
    ),
    Measurement(
        type="TPA",
        points=[
            Point(x=100.0, y=200.0),  # T1中心
            Point(x=150.0, y=300.0),  # CFH中心
            Point(x=200.0, y=400.0),  # S1左
            Point(x=250.0, y=400.0)   # S1右
        ]
    ),
    Measurement(
        type="PI",
        points=[
            Point(x=150.0, y=300.0),  # CFH中心
            Point(x=200.0, y=400.0),  # S1左
            Point(x=250.0, y=400.0)   # S1右
        ]
    ),
]

response = KeypointsResponse(
    imageId="test",
    imageWidth=1920,
    imageHeight=1080,
    measurements=measurements,
    standardDistance=100.0,
    standardDistancePoints=[Point(x=0, y=0), Point(x=200, y=0)]
)

# 输出JSON
import json
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))

