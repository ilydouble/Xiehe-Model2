#!/usr/bin/env python3
"""
验证生成的CFH YOLO11 Pose数据集
"""
import os
from pathlib import Path


def verify_dataset(dataset_path='../datasets/yolo_keypoints'):
    """验证数据集"""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    print('=' * 80)
    print('🔍 验证CFH YOLO11 Pose数据集')
    print('=' * 80)
    print()
    
    # 统计图像和标注文件
    image_files = list(images_dir.glob('*'))
    label_files = list(labels_dir.glob('*.txt'))
    
    print(f'📁 数据集路径: {dataset_path}')
    print(f'📸 图像数量: {len(image_files)}')
    print(f'🏷️  标注数量: {len(label_files)}')
    print()
    
    # 检查配对
    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}
    
    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems
    
    if missing_labels:
        print(f'⚠️  缺少标注的图像: {len(missing_labels)}')
    
    if missing_images:
        print(f'⚠️  缺少图像的标注: {len(missing_images)}')
    
    if not missing_labels and not missing_images:
        print('✅ 所有图像和标注文件配对正确')
        print()
    
    # 分析标注内容
    total_objects = 0
    total_keypoints = 0
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            total_objects += len(lines)
            total_keypoints += len(lines)  # 每个对象1个关键点
    
    print('📊 标注统计:')
    print(f'   总对象数: {total_objects}')
    print(f'   总关键点数: {total_keypoints}')
    print(f'   平均每张图像对象数: {total_objects/len(label_files):.2f}')
    print()
    
    # 检查标注格式
    print('🔍 检查标注格式:')
    sample_label = label_files[0]
    with open(sample_label, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        
        print(f'   示例文件: {sample_label.name}')
        print(f'   第一行: {first_line}')
        print(f'   字段数量: {len(parts)}')
        
        if len(parts) == 8:
            print(f'   ✅ 格式正确: class(1) + bbox(4) + keypoint(1*3) = 8 字段')
            print(f'   - 类别ID: {parts[0]}')
            print(f'   - 边界框: x_center={parts[1]}, y_center={parts[2]}, width={parts[3]}, height={parts[4]}')
            print(f'   - 关键点: x={parts[5]}, y={parts[6]}, v={parts[7]}')
        else:
            print(f'   ❌ 格式错误: 期望8个字段，实际{len(parts)}个')
    print()
    
    print('=' * 80)
    print('✅ 验证完成')
    print('=' * 80)


if __name__ == '__main__':
    verify_dataset()

