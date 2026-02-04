#!/usr/bin/env python3
"""
提取CFH关键点标注，构建YOLO11 Pose数据集
- 只提取CFH的关键点标注（point类型）
- 生成YOLO11 Pose格式的训练数据
- 输出到datasets/yolo_keypoints
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


def parse_json_annotation(json_path):
    """解析JSON标注文件，提取CFH关键点"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cfh_keypoints = []
    
    for shape in data.get('shapes', []):
        label = shape.get('label')
        shape_type = shape.get('shape_type')
        points = shape.get('points', [])
        
        # 只处理CFH的关键点标注
        if label == 'CFH' and shape_type == 'point' and len(points) > 0:
            # point类型的points是[[x, y]]格式
            cfh_keypoints.append({
                'x': points[0][0],
                'y': points[0][1]
            })
    
    return cfh_keypoints


def convert_to_yolo_format(cfh_keypoints, img_width, img_height):
    """
    转换为YOLO11 Pose格式
    CFH有1个关键点
    格式: <class> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_v>
    """
    if len(cfh_keypoints) != 1:
        # 如果不是1个关键点，跳过
        return None
    
    # 计算边界框（对于单个关键点，创建一个固定大小的框）
    kp = cfh_keypoints[0]
    center_x = kp['x']
    center_y = kp['y']

    # 创建一个固定大小的边界框（例如64x64像素）
    bbox_size = 64
    half_size = bbox_size / 2

    x_min = max(0, center_x - half_size)
    y_min = max(0, center_y - half_size)
    x_max = min(img_width, center_x + half_size)
    y_max = min(img_height, center_y + half_size)
    
    # 归一化边界框坐标
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # 归一化关键点坐标
    kp_normalized = []
    for kp in cfh_keypoints:
        kp_x = kp['x'] / img_width
        kp_y = kp['y'] / img_height
        kp_v = 2  # 可见性标记，2表示可见
        kp_normalized.extend([kp_x, kp_y, kp_v])
    
    # 生成YOLO格式的行
    # class_id = 0 (只有一个类别：CFH)
    line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    for i in range(0, len(kp_normalized), 3):
        line += f" {kp_normalized[i]:.6f} {kp_normalized[i+1]:.6f} {int(kp_normalized[i+2])}"
    
    return line


def get_image_size(image_path):
    """获取图像尺寸"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def process_sample(sample_name, dataset_root, output_root):
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
    
    # 解析CFH关键点
    cfh_keypoints = parse_json_annotation(json_path)

    if len(cfh_keypoints) != 1:
        return False, f"CFH keypoints count: {len(cfh_keypoints)} (expected 1)"
    
    # 处理每个图像
    success_count = 0
    for image_path in image_files:
        try:
            # 获取图像尺寸
            img_width, img_height = get_image_size(image_path)
            
            # 转换为YOLO格式
            yolo_line = convert_to_yolo_format(cfh_keypoints, img_width, img_height)
            
            if not yolo_line:
                continue
            
            # 复制图像到输出目录
            output_image_path = output_root / 'images' / image_path.name
            shutil.copy(image_path, output_image_path)
            
            # 保存标注文件
            label_filename = image_path.stem + '.txt'
            output_label_path = output_root / 'labels' / label_filename
            with open(output_label_path, 'w') as f:
                f.write(yolo_line)
            
            success_count += 1
        except Exception as e:
            print(f"  ⚠️  处理图像失败 {image_path.name}: {e}")
    
    return success_count > 0, f"Processed {success_count} images"


def main():
    # 配置
    dataset_root = Path('../datasets/LAT202511')
    output_root = Path('../datasets/yolo_keypoints')
    training_samples_json = 'training_samples.json'
    
    print('=' * 80)
    print('🔄 提取CFH关键点，构建YOLO11 Pose数据集')
    print('=' * 80)
    print(f'数据集路径: {dataset_root}')
    print(f'输出路径: {output_root}')
    print(f'关键点类别: CFH (1个关键点)')
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
        
        success, message = process_sample(sample_name, dataset_root, output_root)
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
    
    if success_count > 0:
        print()
        print(f'✅ 成功提取 {success_count} 个样本的CFH关键点')
    
    if failed_samples:
        print()
        print(f'⚠️  失败的样本 (共 {len(failed_samples)} 个):')
        # 统计失败原因
        from collections import Counter
        reasons = Counter([msg for _, msg in failed_samples])
        for reason, count in reasons.most_common():
            print(f'   - {reason}: {count} 个样本')
        
        print()
        print('失败样本详情（前10个）:')
        for name, msg in failed_samples[:10]:
            print(f'  - {name}: {msg}')
        if len(failed_samples) > 10:
            print(f'  ... 还有 {len(failed_samples) - 10} 个')


if __name__ == '__main__':
    main()

