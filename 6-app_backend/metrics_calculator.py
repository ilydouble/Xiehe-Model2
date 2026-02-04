"""
脊柱指标计算模块

根据关键点计算各种脊柱参数：
- 矢状面角度参数（T1倾斜角、颈椎前凸角、胸椎后凸角、腰椎前凸角、T1骨盆角）
- 矢状面距离参数（SVA）
- 骨盆参数（PI、PT、SS）
"""
import numpy as np
from typing import Dict, Optional
from models import Point, VertebraKeypoints


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, vertebrae_keypoints: Dict[str, VertebraKeypoints], 
                 cfh_center: Optional[Point],
                 s1_upper_endplate_center: Optional[Point]):
        """
        初始化
        
        Args:
            vertebrae_keypoints: 椎体关键点字典
            cfh_center: 股骨头中心
            s1_upper_endplate_center: S1上终板中点
        """
        self.vertebrae = vertebrae_keypoints
        self.cfh_center = cfh_center
        self.s1_center = s1_upper_endplate_center
    
    def calculate_all_metrics(self) -> Dict:
        """计算所有指标"""
        metrics = {}
        
        # 矢状面角度参数
        metrics['T1_slope'] = self.calculate_t1_slope()
        metrics['cervical_lordosis'] = self.calculate_cervical_lordosis()
        metrics['thoracic_kyphosis_T2_T5'] = self.calculate_thoracic_kyphosis_T2_T5()
        metrics['thoracic_kyphosis_T5_T12'] = self.calculate_thoracic_kyphosis_T5_T12()
        metrics['lumbar_lordosis'] = self.calculate_lumbar_lordosis()
        metrics['T1_pelvic_angle'] = self.calculate_t1_pelvic_angle()
        
        # 矢状面距离参数
        metrics['SVA'] = self.calculate_sva()
        
        # 骨盆参数
        metrics['pelvic_incidence'] = self.calculate_pelvic_incidence()
        metrics['pelvic_tilt'] = self.calculate_pelvic_tilt()
        metrics['sacral_slope'] = self.calculate_sacral_slope()
        
        return metrics
    
    def _calculate_angle_between_lines(self, p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[float]:
        """
        计算两条线段的夹角
        
        线段1: p1 -> p2
        线段2: p3 -> p4
        
        Returns:
            角度（度），如果无法计算则返回None
        """
        # 向量1
        v1 = np.array([p2.x - p1.x, p2.y - p1.y])
        # 向量2
        v2 = np.array([p4.x - p3.x, p4.y - p3.y])
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return float(angle_deg)
    
    def _calculate_angle_with_horizontal(self, p1: Point, p2: Point) -> Optional[float]:
        """
        计算线段与水平线的夹角
        
        Args:
            p1, p2: 线段的两个端点
            
        Returns:
            角度（度），如果无法计算则返回None
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        return float(abs(angle_deg))
    
    def calculate_t1_slope(self) -> Optional[float]:
        """计算T1倾斜角"""
        if 'T1' not in self.vertebrae:
            return None
        
        t1 = self.vertebrae['T1']
        # T1上终板与水平线的夹角
        return self._calculate_angle_with_horizontal(
            t1.upper_endplate_posterior,
            t1.upper_endplate_anterior
        )
    
    def calculate_cervical_lordosis(self) -> Optional[float]:
        """计算C2-C7前凸角（需要C2和C7）"""
        # 由于我们的数据集中没有C2，这里用C7代替
        if 'C7' not in self.vertebrae:
            return None
        
        # 简化版本：只用C7下终板角度
        c7 = self.vertebrae['C7']
        return self._calculate_angle_with_horizontal(
            c7.lower_endplate_posterior,
            c7.lower_endplate_anterior
        )
    
    def calculate_thoracic_kyphosis_T2_T5(self) -> Optional[float]:
        """计算T2-T5胸椎后凸角"""
        if 'T2' not in self.vertebrae or 'T5' not in self.vertebrae:
            return None
        
        t2 = self.vertebrae['T2']
        t5 = self.vertebrae['T5']
        
        return self._calculate_angle_between_lines(
            t2.upper_endplate_posterior,
            t2.upper_endplate_anterior,
            t5.lower_endplate_posterior,
            t5.lower_endplate_anterior
        )
    
    def calculate_thoracic_kyphosis_T5_T12(self) -> Optional[float]:
        """计算T5-T12胸椎后凸角"""
        if 'T5' not in self.vertebrae or 'T12' not in self.vertebrae:
            return None
        
        t5 = self.vertebrae['T5']
        t12 = self.vertebrae['T12']
        
        return self._calculate_angle_between_lines(
            t5.upper_endplate_posterior,
            t5.upper_endplate_anterior,
            t12.lower_endplate_posterior,
            t12.lower_endplate_anterior
        )
    
    def calculate_lumbar_lordosis(self) -> Optional[float]:
        """计算腰椎前凸角（L1-S1）"""
        if 'L1' not in self.vertebrae or self.s1_center is None:
            return None

        l1 = self.vertebrae['L1']

        # L1上终板到S1上终板的角度
        # 这里简化处理，使用L1上终板的角度
        return self._calculate_angle_with_horizontal(
            l1.upper_endplate_posterior,
            l1.upper_endplate_anterior
        )

    def calculate_t1_pelvic_angle(self) -> Optional[float]:
        """计算T1骨盆角（TPA）"""
        if 'T1' not in self.vertebrae or self.cfh_center is None or self.s1_center is None:
            return None

        t1_center = self.vertebrae['T1'].center

        # 三个点：T1中心、CFH中心、S1中心
        # 计算角度
        v1 = np.array([t1_center.x - self.cfh_center.x, t1_center.y - self.cfh_center.y])
        v2 = np.array([self.s1_center.x - self.cfh_center.x, self.s1_center.y - self.cfh_center.y])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def calculate_sva(self) -> Optional[float]:
        """计算矢状面垂直轴（SVA）"""
        if 'C7' not in self.vertebrae or self.s1_center is None:
            return None

        c7 = self.vertebrae['C7']
        # C7后上角（posterior upper）
        c7_posterior_upper = c7.upper_endplate_posterior

        # 水平距离（归一化坐标）
        sva = abs(c7_posterior_upper.x - self.s1_center.x)

        return float(sva)

    def calculate_pelvic_incidence(self) -> Optional[float]:
        """计算骨盆入射角（PI）"""
        if self.cfh_center is None or self.s1_center is None:
            return None

        # S1上终板的垂线方向（假设终板是水平的，垂线是竖直的）
        # 简化处理：垂线方向为(0, 1)
        perpendicular = np.array([0, 1])

        # CFH中心到S1中心的向量
        cfh_to_s1 = np.array([
            self.s1_center.x - self.cfh_center.x,
            self.s1_center.y - self.cfh_center.y
        ])

        # 计算夹角
        cos_angle = np.dot(perpendicular, cfh_to_s1) / (np.linalg.norm(cfh_to_s1) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def calculate_pelvic_tilt(self) -> Optional[float]:
        """计算骨盆倾斜角（PT）"""
        if self.cfh_center is None or self.s1_center is None:
            return None

        # CFH中心到S1中心的连线与垂直线的夹角
        dx = self.s1_center.x - self.cfh_center.x
        dy = self.s1_center.y - self.cfh_center.y

        # 与垂直线的夹角
        angle_rad = np.arctan2(abs(dx), abs(dy))
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def calculate_sacral_slope(self) -> Optional[float]:
        """计算骶骨倾斜角（SS）"""
        # 由于我们没有S1的实际检测，这里用L5下终板近似
        if 'L5' not in self.vertebrae:
            return None

        l5 = self.vertebrae['L5']

        # L5下终板与水平线的夹角
        return self._calculate_angle_with_horizontal(
            l5.lower_endplate_posterior,
            l5.lower_endplate_anterior
        )

