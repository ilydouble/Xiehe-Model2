"""
数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Point(BaseModel):
    """点坐标（归一化）"""
    x: float = Field(..., description="X坐标（归一化，0-1）")
    y: float = Field(..., description="Y坐标（归一化，0-1）")


class VertebraDetection(BaseModel):
    """椎体检测结果"""
    label: str = Field(..., description="椎体标签，如C7, T1, L1等")
    confidence: float = Field(..., description="置信度")
    bbox: List[float] = Field(..., description="边界框 [x_center, y_center, width, height]，归一化坐标")
    keypoints: List[Point] = Field(..., description="4个角点坐标（归一化）")


class CFHDetection(BaseModel):
    """股骨头检测结果"""
    confidence: float = Field(..., description="置信度")
    bbox: List[float] = Field(..., description="边界框 [x_center, y_center, width, height]，归一化坐标")
    center: Point = Field(..., description="股骨头中心点（归一化）")


class DetectionResponse(BaseModel):
    """检测结果响应"""
    vertebrae: List[VertebraDetection] = Field(..., description="椎体检测结果列表")
    cfh: Optional[CFHDetection] = Field(None, description="股骨头检测结果")
    image_width: int = Field(..., description="图像宽度（像素）")
    image_height: int = Field(..., description="图像高度（像素）")


class KeypointsRequest(BaseModel):
    """关键点计算请求"""
    vertebrae: List[VertebraDetection] = Field(..., description="椎体检测结果")
    cfh: Optional[CFHDetection] = Field(None, description="股骨头检测结果")
    image_width: int = Field(..., description="图像宽度（像素）")
    image_height: int = Field(..., description="图像高度（像素）")


class VertebraKeypoints(BaseModel):
    """椎体关键点"""
    label: str = Field(..., description="椎体标签")
    upper_endplate_anterior: Point = Field(..., description="上终板前角")
    upper_endplate_posterior: Point = Field(..., description="上终板后角")
    lower_endplate_anterior: Point = Field(..., description="下终板前角")
    lower_endplate_posterior: Point = Field(..., description="下终板后角")
    center: Point = Field(..., description="椎体中心")


class Measurement(BaseModel):
    """单个指标的测量点"""
    type: str = Field(..., description="指标类型")
    points: List[Point] = Field(..., description="计算该指标所需的点")


class KeypointsResponse(BaseModel):
    """关键点计算响应 - 按指标组织"""
    imageId: str = Field(default="lateral_spine", description="图像ID")
    imageWidth: int = Field(..., description="图像宽度（像素）")
    imageHeight: int = Field(..., description="图像高度（像素）")
    measurements: List[Measurement] = Field(..., description="各指标的测量点")
    standardDistance: float = Field(default=100.0, description="标准距离（mm）")
    standardDistancePoints: List[Point] = Field(
        default_factory=lambda: [Point(x=0, y=0), Point(x=200, y=0)],
        description="标准距离参考点"
    )

