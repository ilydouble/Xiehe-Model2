"""
关键点计算服务

根据检测到的椎体角点，计算用于各指标计算的关键点
输出格式参考正面JSON，按指标类型组织
"""
import numpy as np
from typing import List, Optional, Dict

from models import (
    Point,
    VertebraDetection,
    CFHDetection,
    Measurement,
    KeypointsResponse
)


def compute_keypoints(
    vertebrae: List[VertebraDetection],
    cfh: Optional[CFHDetection],
    image_width: int,
    image_height: int,
    image_id: str = "lateral_spine"
) -> KeypointsResponse:
    """
    计算关键点，按指标类型组织输出（像素坐标）

    Args:
        vertebrae: 椎体检测结果列表
        cfh: 股骨头检测结果
        image_width: 图像宽度（像素）
        image_height: 图像高度（像素）
        image_id: 图像ID

    Returns:
        KeypointsResponse: 按指标组织的测量点（像素坐标）
    """
    # 先计算每个椎体的终板点（归一化坐标）
    vertebrae_dict = {}
    for v in vertebrae:
        vertebrae_dict[v.label] = _extract_endplate_points(v)

    # 转换为像素坐标的辅助函数
    def to_pixel(point: Point) -> Point:
        return Point(
            x=point.x * image_width,
            y=point.y * image_height
        )

    # 生成各指标的测量点（像素坐标）
    measurements = []

    # 1. T1 Slope - T1上终板
    if 'T1' in vertebrae_dict:
        measurements.append(Measurement(
            type="T1 Slope",
            points=[to_pixel(p) for p in vertebrae_dict['T1']['upper']]
        ))

    # 2. C2-C7 CL (Cervical Lordosis) - C2下终板和C7下终板
    if 'C2' in vertebrae_dict and 'C7' in vertebrae_dict:
        measurements.append(Measurement(
            type="C2-C7 CL",
            points=[to_pixel(p) for p in (vertebrae_dict['C2']['lower'] + vertebrae_dict['C7']['lower'])]
        ))

    # 3. TK T2-T5 (Thoracic Kyphosis T2-T5)
    if 'T2' in vertebrae_dict and 'T5' in vertebrae_dict:
        measurements.append(Measurement(
            type="TK T2-T5",
            points=[to_pixel(p) for p in (vertebrae_dict['T2']['upper'] + vertebrae_dict['T5']['lower'])]
        ))

    # 4. TK T5-T12 (Thoracic Kyphosis T5-T12)
    if 'T5' in vertebrae_dict and 'T12' in vertebrae_dict:
        measurements.append(Measurement(
            type="TK T5-T12",
            points=[to_pixel(p) for p in (vertebrae_dict['T5']['upper'] + vertebrae_dict['T12']['lower'])]
        ))

    # 5. LL L1-S1 (Lumbar Lordosis L1-S1)
    if 'L1' in vertebrae_dict and 'L5' in vertebrae_dict:
        measurements.append(Measurement(
            type="LL L1-S1",
            points=[to_pixel(p) for p in (vertebrae_dict['L1']['upper'] + vertebrae_dict['L5']['lower'])]
        ))

    # 6. LL L1-L4 (Lumbar Lordosis L1-L4) - 新增
    if 'L1' in vertebrae_dict and 'L4' in vertebrae_dict:
        measurements.append(Measurement(
            type="LL L1-L4",
            points=[to_pixel(p) for p in (vertebrae_dict['L1']['upper'] + vertebrae_dict['L4']['lower'])]
        ))

    # 7. LL L4-S1 (Lumbar Lordosis L4-S1) - 新增
    if 'L4' in vertebrae_dict and 'L5' in vertebrae_dict:
        measurements.append(Measurement(
            type="LL L4-S1",
            points=[to_pixel(p) for p in (vertebrae_dict['L4']['upper'] + vertebrae_dict['L5']['lower'])]
        ))

    # 8. SVA - C7后上角和S1估算点
    if 'C7' in vertebrae_dict:
        c7_posterior_upper = vertebrae_dict['C7']['upper'][1]  # 后角
        s1_point = _estimate_s1_point(vertebrae_dict)
        if s1_point:
            measurements.append(Measurement(
                type="SVA",
                points=[to_pixel(c7_posterior_upper), to_pixel(s1_point)]
            ))

    # 9. TPA - T1四个角点、CFH中心、S1上终板左右角（7个点）
    if 'T1' in vertebrae_dict and cfh and 'L5' in vertebrae_dict:
        # T1的4个角点
        t1_upper = vertebrae_dict['T1']['upper']  # [前角, 后角]
        t1_lower = vertebrae_dict['T1']['lower']  # [前角, 后角]

        cfh_center = cfh.center

        # S1用L5下终板近似
        s1_left = vertebrae_dict['L5']['lower'][0]   # 前角（左侧）
        s1_right = vertebrae_dict['L5']['lower'][1]  # 后角（右侧）

        # TPA: [T1上前, T1上后, T1下前, T1下后, CFH中心, S1左, S1右]
        measurements.append(Measurement(
            type="TPA",
            points=[
                to_pixel(t1_upper[0]),   # T1上终板前角
                to_pixel(t1_upper[1]),   # T1上终板后角
                to_pixel(t1_lower[0]),   # T1下终板前角
                to_pixel(t1_lower[1]),   # T1下终板后角
                to_pixel(cfh_center),    # CFH中心
                to_pixel(s1_left),       # S1左
                to_pixel(s1_right)       # S1右
            ]
        ))

    # 10. PI - 骨盆入射角（CFH中心、S1上终板左右角，3个点）
    if cfh and 'L5' in vertebrae_dict:
        cfh_center = cfh.center
        # S1用L5下终板近似
        s1_left = vertebrae_dict['L5']['lower'][0]   # 前角（左侧）
        s1_right = vertebrae_dict['L5']['lower'][1]  # 后角（右侧）

        # PI: [CFH中心, S1左, S1右]
        measurements.append(Measurement(
            type="PI",
            points=[
                to_pixel(cfh_center),
                to_pixel(s1_left),
                to_pixel(s1_right)
            ]
        ))

    # 11. PT - 骨盆倾斜角（CFH中心、S1上终板左右角，3个点）
    if cfh and 'L5' in vertebrae_dict:
        cfh_center = cfh.center
        # S1用L5下终板近似
        s1_left = vertebrae_dict['L5']['lower'][0]   # 前角（左侧）
        s1_right = vertebrae_dict['L5']['lower'][1]  # 后角（右侧）

        # PT: [CFH中心, S1左, S1右]
        measurements.append(Measurement(
            type="PT",
            points=[
                to_pixel(cfh_center),
                to_pixel(s1_left),
                to_pixel(s1_right)
            ]
        ))

    # 12. SS - 骶骨倾斜角（用L5下终板近似）
    if 'L5' in vertebrae_dict:
        measurements.append(Measurement(
            type="SS",
            points=[to_pixel(p) for p in vertebrae_dict['L5']['lower']]
        ))

    return KeypointsResponse(
        imageId=image_id,
        imageWidth=image_width,
        imageHeight=image_height,
        measurements=measurements,
        standardDistance=100.0,
        standardDistancePoints=[Point(x=0, y=0), Point(x=200, y=0)]
    )


def _extract_endplate_points(vertebra: VertebraDetection) -> Dict:
    """
    提取椎体的终板点

    返回格式：
    {
        'upper': [anterior_point, posterior_point],  # 上终板：前角、后角
        'lower': [anterior_point, posterior_point],  # 下终板：前角、后角
        'center': center_point  # 中心点
    }
    """
    kpts = vertebra.keypoints

    if len(kpts) != 4:
        raise ValueError(f"椎体 {vertebra.label} 的关键点数量不是4个")

    # 转换为numpy数组
    points = np.array([[p.x, p.y] for p in kpts])

    # 按Y坐标排序，分为上下两组
    sorted_by_y = sorted(enumerate(points), key=lambda x: x[1][1])

    # 上终板的两个点（Y坐标较小）
    upper_indices = [sorted_by_y[0][0], sorted_by_y[1][0]]
    upper_points = points[upper_indices]

    # 下终板的两个点（Y坐标较大）
    lower_indices = [sorted_by_y[2][0], sorted_by_y[3][0]]
    lower_points = points[lower_indices]

    # 在上终板中，X小的是后角，X大的是前角
    if upper_points[0][0] < upper_points[1][0]:
        upper_posterior = Point(x=float(upper_points[0][0]), y=float(upper_points[0][1]))
        upper_anterior = Point(x=float(upper_points[1][0]), y=float(upper_points[1][1]))
    else:
        upper_posterior = Point(x=float(upper_points[1][0]), y=float(upper_points[1][1]))
        upper_anterior = Point(x=float(upper_points[0][0]), y=float(upper_points[0][1]))

    # 在下终板中，X小的是后角，X大的是前角
    if lower_points[0][0] < lower_points[1][0]:
        lower_posterior = Point(x=float(lower_points[0][0]), y=float(lower_points[0][1]))
        lower_anterior = Point(x=float(lower_points[1][0]), y=float(lower_points[1][1]))
    else:
        lower_posterior = Point(x=float(lower_points[1][0]), y=float(lower_points[1][1]))
        lower_anterior = Point(x=float(lower_points[0][0]), y=float(lower_points[0][1]))

    # 计算椎体中心
    center_x = float(np.mean(points[:, 0]))
    center_y = float(np.mean(points[:, 1]))
    center = Point(x=center_x, y=center_y)

    return {
        'upper': [upper_anterior, upper_posterior],
        'lower': [lower_anterior, lower_posterior],
        'center': center
    }


def _estimate_s1_point(vertebrae_dict: Dict) -> Optional[Point]:
    """
    估算S1上终板中点

    基于L5下终板向下延伸
    """
    if 'L5' not in vertebrae_dict:
        return None

    l5_lower = vertebrae_dict['L5']['lower']

    # L5下终板中点
    l5_lower_center_x = (l5_lower[0].x + l5_lower[1].x) / 2
    l5_lower_center_y = (l5_lower[0].y + l5_lower[1].y) / 2

    # L5上终板中点
    l5_upper = vertebrae_dict['L5']['upper']
    l5_upper_center_y = (l5_upper[0].y + l5_upper[1].y) / 2

    # L5椎体高度
    l5_height = abs(l5_lower_center_y - l5_upper_center_y)

    # S1估算：在L5下方约1.2倍椎体高度
    s1_y = l5_lower_center_y + l5_height * 1.2
    s1_x = l5_lower_center_x

    return Point(x=s1_x, y=s1_y)

