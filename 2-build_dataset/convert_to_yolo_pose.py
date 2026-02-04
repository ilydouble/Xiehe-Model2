#!/usr/bin/env python3
"""
将分割标注转换为YOLO11 Pose格式
- 读取分割标注（除去CFH、T13、S1）
- 计算轮廓中心
- 找到距离中心最远的4个点作为关键点
- 生成YOLO11 Pose格式的训练数据
"""
import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def load_training_samples(json_file='training_samples.json'):
    """加载可用于训练的样本列表"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['valid_samples']


def calculate_polygon_center(points):
    """计算多边形的中心点"""
    points = np.array(points)
    return np.mean(points, axis=0)


def find_farthest_4_points(points, center):
    """
    找到十字对角分布的4个角点

    策略：
    1. 先找到距离中心最远的点（第1个角点）
    2. 在与第1个点成90度±45度范围内找最远的点（第2个角点）
    3. 在与第2个点成90度±45度范围内找最远的点（第3个角点）
    4. 在与第3个点成90度±45度范围内找最远的点（第4个角点）

    这样确保4个点呈十字对角分布
    """
    points = np.array(points)
    center = np.array(center)

    # 计算每个点到中心的距离和角度
    distances = np.linalg.norm(points - center, axis=1)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # 角度范围（90度 ± 45度 = 45度到135度）
    angle_range = np.pi / 4  # 45度

    selected_points = []
    selected_indices = []
    used_indices = set()

    # 第1个点：距离最远的点
    idx1 = np.argmax(distances)
    selected_points.append(points[idx1])
    selected_indices.append(idx1)
    used_indices.add(idx1)
    current_angle = angles[idx1]

    # 找剩余3个点，每次在90度±45度范围内找最远的点
    for i in range(3):
        # 目标角度：当前角度 + 90度
        target_angle = current_angle + np.pi / 2

        # 计算角度差（考虑周期性）
        angle_diffs = np.abs(angles - target_angle)
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)

        # 在角度范围内的点
        valid_mask = angle_diffs <= angle_range

        # 排除已选择的点
        for used_idx in used_indices:
            valid_mask[used_idx] = False

        # 在有效范围内找距离最远的点
        valid_distances = np.where(valid_mask, distances, -1)

        if np.max(valid_distances) > 0:
            idx = np.argmax(valid_distances)
            selected_points.append(points[idx])
            selected_indices.append(idx)
            used_indices.add(idx)
            current_angle = angles[idx]
        else:
            # 如果没找到，就在剩余点中找最远的
            remaining_distances = distances.copy()
            for used_idx in used_indices:
                remaining_distances[used_idx] = -1
            idx = np.argmax(remaining_distances)
            selected_points.append(points[idx])
            selected_indices.append(idx)
            used_indices.add(idx)
            current_angle = angles[idx]

    # 按照角度排序这4个点，确保顺序一致（从0度开始逆时针）
    selected_points = np.array(selected_points)
    point_angles = np.arctan2(selected_points[:, 1] - center[1],
                              selected_points[:, 0] - center[0])
    sorted_indices = np.argsort(point_angles)

    return selected_points[sorted_indices]


def parse_json_annotation(json_path, excluded_labels={'CFH', 'T13', 'S1'}):
    """解析JSON标注文件，提取分割标注"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = []
    
    for shape in data.get('shapes', []):
        label = shape.get('label')
        shape_type = shape.get('shape_type')
        points = shape.get('points', [])
        
        # 只处理分割标注，排除CFH、T13、S1
        if shape_type == 'polygon' and label not in excluded_labels and len(points) >= 4:
            annotations.append({
                'label': label,
                'points': points
            })
    
    return annotations


def convert_to_yolo_format(annotations, img_width, img_height, label_map):
    """
    转换为YOLO11 Pose格式
    格式: <class> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_v> ... <kp4_x> <kp4_y> <kp4_v>
    其中 kp_v 表示可见性，2表示可见
    """
    yolo_lines = []
    
    for ann in annotations:
        label = ann['label']
        points = ann['points']
        
        if label not in label_map:
            continue
        
        class_id = label_map[label]
        
        # 计算轮廓中心
        center = calculate_polygon_center(points)
        
        # 找到距离中心最远的4个点
        keypoints = find_farthest_4_points(points, center)
        
        # 计算边界框
        points_array = np.array(points)
        x_min, y_min = points_array.min(axis=0)
        x_max, y_max = points_array.max(axis=0)
        
        # 归一化坐标
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # 归一化关键点坐标
        kp_normalized = []
        for kp in keypoints:
            kp_x = kp[0] / img_width
            kp_y = kp[1] / img_height
            kp_v = 2  # 可见性标记，2表示可见
            kp_normalized.extend([kp_x, kp_y, kp_v])
        
        # 生成YOLO格式的行
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for i in range(0, len(kp_normalized), 3):
            line += f" {kp_normalized[i]:.6f} {kp_normalized[i+1]:.6f} {int(kp_normalized[i+2])}"
        
        yolo_lines.append(line)
    
    return yolo_lines


def get_image_size(image_path):
    """获取图像尺寸"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def process_sample(sample_name, dataset_root, output_root, label_map):
    """处理单个样本"""
    sample_path = Path(dataset_root) / sample_name
    
    # 查找JSON文件
    json_files = list(sample_path.glob('*.json'))
    if not json_files:
        return False, "No JSON file"
    
    json_path = json_files[0]
    
    # 查找图像文件
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(sample_path.glob(f'*{ext}')))
    
    if not image_files:
        return False, "No image file"
    
    # 解析标注
    annotations = parse_json_annotation(json_path)
    
    if not annotations:
        return False, "No valid annotations"
    
    # 处理每个图像
    success_count = 0
    for image_path in image_files:
        try:
            # 获取图像尺寸
            img_width, img_height = get_image_size(image_path)
            
            # 转换为YOLO格式
            yolo_lines = convert_to_yolo_format(annotations, img_width, img_height, label_map)
            
            if not yolo_lines:
                continue
            
            # 复制图像到输出目录
            output_image_path = output_root / 'images' / image_path.name
            shutil.copy(image_path, output_image_path)
            
            # 保存标注文件
            label_filename = image_path.stem + '.txt'
            output_label_path = output_root / 'labels' / label_filename
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            success_count += 1
        except Exception as e:
            print(f"  ⚠️  处理图像失败 {image_path.name}: {e}")
    
    return success_count > 0, f"Processed {success_count} images"


def main():
    # 配置
    dataset_root = Path('../datasets/LAT202511')
    output_root = Path('../datasets/yolo_corner')
    training_samples_json = 'training_samples.json'
    
    # 标签映射（除去CFH、T13、S1）
    labels = ['C7', 'L1', 'L2', 'L3', 'L4', 'L5', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    print('=' * 80)
    print('🔄 转换为YOLO11 Pose格式')
    print('=' * 80)
    print(f'数据集路径: {dataset_root}')
    print(f'输出路径: {output_root}')
    print(f'标签数量: {len(labels)}')
    print(f'标签列表: {labels}')
    print()
    
    # 创建输出目录
    (output_root / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 加载训练样本列表
    valid_samples = load_training_samples(training_samples_json)
    print(f'可用训练样本: {len(valid_samples)}')
    print()
    
    # 处理每个样本
    print('开始处理样本...')
    success_count = 0
    failed_samples = []
    
    for i, sample_name in enumerate(valid_samples, 1):
        if i % 50 == 0:
            print(f'进度: {i}/{len(valid_samples)}')
        
        success, message = process_sample(sample_name, dataset_root, output_root, label_map)
        if success:
            success_count += 1
        else:
            failed_samples.append((sample_name, message))
    
    print()
    print('=' * 80)
    print('✅ 处理完成')
    print('=' * 80)
    print(f'成功处理: {success_count}/{len(valid_samples)}')
    print(f'失败: {len(failed_samples)}')
    print()
    
    if failed_samples:
        print('失败的样本:')
        for name, msg in failed_samples[:10]:
            print(f'  - {name}: {msg}')
        if len(failed_samples) > 10:
            print(f'  ... 还有 {len(failed_samples) - 10} 个')


if __name__ == '__main__':
    main()

