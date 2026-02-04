#!/usr/bin/env python3
"""
验证生成的YOLO11 Pose数据集
"""
import os
from pathlib import Path
from collections import defaultdict


def verify_dataset(dataset_path='../datasets/yolo_corner'):
    """验证数据集"""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    print('=' * 80)
    print('🔍 验证YOLO11 Pose数据集')
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
        for stem in list(missing_labels)[:5]:
            print(f'   - {stem}')
        if len(missing_labels) > 5:
            print(f'   ... 还有 {len(missing_labels) - 5} 个')
        print()
    
    if missing_images:
        print(f'⚠️  缺少图像的标注: {len(missing_images)}')
        for stem in list(missing_images)[:5]:
            print(f'   - {stem}')
        if len(missing_images) > 5:
            print(f'   ... 还有 {len(missing_images) - 5} 个')
        print()
    
    if not missing_labels and not missing_images:
        print('✅ 所有图像和标注文件配对正确')
        print()
    
    # 分析标注内容
    class_counts = defaultdict(int)
    total_objects = 0
    total_keypoints = 0
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            total_objects += len(lines)
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 17:  # class + bbox(4) + keypoints(4*3)
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_keypoints += 4
    
    print('📊 标注统计:')
    print(f'   总对象数: {total_objects}')
    print(f'   总关键点数: {total_keypoints}')
    print(f'   平均每张图像对象数: {total_objects/len(label_files):.2f}')
    print()
    
    # 类别分布
    print('📈 类别分布:')
    label_names = ['C7', 'L1', 'L2', 'L3', 'L4', 'L5', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        label_name = label_names[class_id] if class_id < len(label_names) else f'Unknown_{class_id}'
        percentage = count / total_objects * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f'   {label_name:4s} (ID {class_id:2d}): {bar} {count:4d} ({percentage:5.1f}%)')
    print()
    
    # 检查标注格式
    print('🔍 检查标注格式:')
    sample_label = label_files[0]
    with open(sample_label, 'r') as f:
        first_line = f.readline().strip()
        parts = first_line.split()
        
        print(f'   示例文件: {sample_label.name}')
        print(f'   第一行: {first_line[:100]}...')
        print(f'   字段数量: {len(parts)}')
        
        if len(parts) >= 17:
            print(f'   ✅ 格式正确: class(1) + bbox(4) + keypoints(4*3) = {len(parts)} 字段')
            print(f'   - 类别ID: {parts[0]}')
            print(f'   - 边界框: x_center={parts[1]}, y_center={parts[2]}, width={parts[3]}, height={parts[4]}')
            print(f'   - 关键点1: x={parts[5]}, y={parts[6]}, v={parts[7]}')
            print(f'   - 关键点2: x={parts[8]}, y={parts[9]}, v={parts[10]}')
            print(f'   - 关键点3: x={parts[11]}, y={parts[12]}, v={parts[13]}')
            print(f'   - 关键点4: x={parts[14]}, y={parts[15]}, v={parts[16]}')
        else:
            print(f'   ❌ 格式错误: 期望至少17个字段，实际{len(parts)}个')
    print()
    
    print('=' * 80)
    print('✅ 验证完成')
    print('=' * 80)


if __name__ == '__main__':
    verify_dataset()

