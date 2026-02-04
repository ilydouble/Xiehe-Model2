#!/usr/bin/env python3
"""
将CFH关键点转换为YOLO检测格式
从单个中心点扩展为检测框
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np


def calculate_bbox_size(image_shape, reference_vertebra_sizes=None):
    """
    计算CFH检测框的合理大小
    
    参数:
        image_shape: 图像尺寸 (height, width)
        reference_vertebra_sizes: 参考椎体大小列表 [(w, h), ...]
    
    返回:
        (width, height): 检测框大小（像素）
    """
    h, w = image_shape[:2]
    
    # 方法1: 基于图像大小的固定比例
    # 股骨头通常占图像宽度的5-8%，高度的3-5%
    default_w = int(w * 0.06)  # 6%的图像宽度
    default_h = int(h * 0.04)  # 4%的图像高度
    
    # 方法2: 如果有椎体参考，使用椎体大小的1.5-2倍
    if reference_vertebra_sizes and len(reference_vertebra_sizes) > 0:
        avg_v_w = np.mean([size[0] for size in reference_vertebra_sizes])
        avg_v_h = np.mean([size[1] for size in reference_vertebra_sizes])
        
        # 股骨头通常是椎体的1.5-2倍大
        ref_w = int(avg_v_h * 1.8)  # 使用椎体高度作为参考
        ref_h = int(avg_v_h * 1.8)
        
        # 取两种方法的平均值
        bbox_w = int((default_w + ref_w) / 2)
        bbox_h = int((default_h + ref_h) / 2)
    else:
        bbox_w = default_w
        bbox_h = default_h
    
    # 确保是正方形（股骨头是圆形的）
    bbox_size = max(bbox_w, bbox_h)
    
    return bbox_size, bbox_size


def convert_point_to_bbox(cx, cy, bbox_w, bbox_h, img_w, img_h):
    """
    将中心点转换为YOLO格式的边界框
    
    参数:
        cx, cy: 中心点坐标（像素）
        bbox_w, bbox_h: 边界框大小（像素）
        img_w, img_h: 图像大小（像素）
    
    返回:
        (cx_norm, cy_norm, w_norm, h_norm): YOLO格式的归一化坐标
    """
    # 归一化坐标
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm = bbox_w / img_w
    h_norm = bbox_h / img_h
    
    # 确保在[0, 1]范围内
    cx_norm = max(0, min(1, cx_norm))
    cy_norm = max(0, min(1, cy_norm))
    w_norm = max(0, min(1, w_norm))
    h_norm = max(0, min(1, h_norm))
    
    return cx_norm, cy_norm, w_norm, h_norm


def process_dataset(input_dir, output_dir, bbox_size_multiplier=1.8):
    """
    处理整个数据集
    
    参数:
        input_dir: 输入目录（LAT202511）
        output_dir: 输出目录
        bbox_size_multiplier: 相对于椎体的大小倍数
    """
    print('=' * 80)
    print('🔄 CFH关键点转检测框')
    print('=' * 80)
    print(f'输入目录: {input_dir}')
    print(f'输出目录: {output_dir}')
    print(f'包围盒倍数: {bbox_size_multiplier}x')
    print()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有患者目录
    patient_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    print(f'找到 {len(patient_dirs)} 个患者目录')
    print()
    
    # 统计信息
    total_images = 0
    total_cfh = 0
    skipped = 0
    
    # 处理每个患者
    for patient_dir in tqdm(patient_dirs, desc='处理患者'):
        # 查找JSON和图像文件
        json_files = list(patient_dir.glob('*.json'))
        
        if not json_files:
            continue
        
        for json_file in json_files:
            # 读取JSON标注
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取图像文件名
            image_filename = data.get('imagePath', '')
            if not image_filename:
                continue
            
            image_path = patient_dir / image_filename
            
            if not image_path.exists():
                skipped += 1
                continue
            
            # 读取图像获取尺寸
            img = cv2.imread(str(image_path))
            if img is None:
                skipped += 1
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 提取CFH点和椎体信息
            cfh_points = []
            vertebra_sizes = []
            
            for shape in data.get('shapes', []):
                label = shape.get('label', '')
                points = shape.get('points', [])
                
                if label == 'CFH' and len(points) > 0:
                    # CFH中心点
                    cfh_points.append(points[0])
                
                elif label != 'CFH' and len(points) >= 4:
                    # 椎体 - 计算大小作为参考
                    pts = np.array(points)
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    w = x_max - x_min
                    h = y_max - y_min
                    vertebra_sizes.append((w, h))
            
            # 如果没有CFH，跳过
            if not cfh_points:
                continue
            
            # 计算检测框大小
            bbox_w, bbox_h = calculate_bbox_size(
                (img_h, img_w), 
                vertebra_sizes if vertebra_sizes else None
            )
            
            # 如果有椎体参考，使用倍数调整
            if vertebra_sizes:
                avg_v_h = np.mean([size[1] for size in vertebra_sizes])
                bbox_w = int(avg_v_h * bbox_size_multiplier)
                bbox_h = int(avg_v_h * bbox_size_multiplier)
            
            # 生成YOLO标注
            yolo_labels = []
            
            for cfh_point in cfh_points:
                cx, cy = cfh_point
                
                # 转换为YOLO格式
                cx_norm, cy_norm, w_norm, h_norm = convert_point_to_bbox(
                    cx, cy, bbox_w, bbox_h, img_w, img_h
                )
                
                # YOLO检测格式: class_id cx cy w h
                # CFH的class_id设为0（单类别检测）
                yolo_labels.append(f'0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}')
                total_cfh += 1
            
            # 保存图像和标注
            output_image_path = images_dir / image_path.name
            output_label_path = labels_dir / f'{image_path.stem}.txt'
            
            # 复制图像
            shutil.copy2(image_path, output_image_path)
            
            # 保存标注
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            total_images += 1
    
    print()
    print('=' * 80)
    print('✅ 转换完成！')
    print('=' * 80)
    print(f'处理图像: {total_images}')
    print(f'CFH数量: {total_cfh}')
    print(f'跳过文件: {skipped}')
    print()
    
    return total_images, total_cfh


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将CFH关键点转换为YOLO检测格式')
    parser.add_argument('--input', type=str, default='../datasets/LAT202511',
                       help='输入目录（原始数据）')
    parser.add_argument('--output', type=str, default='../datasets/yolo_cfh_detection',
                       help='输出目录')
    parser.add_argument('--bbox-multiplier', type=float, default=1.8,
                       help='相对于椎体的包围盒大小倍数（默认1.8）')
    
    args = parser.parse_args()
    
    # 转换数据集
    total_images, total_cfh = process_dataset(
        args.input, 
        args.output,
        args.bbox_multiplier
    )
    
    # 创建data.yaml
    output_path = Path(args.output)
    data_yaml_path = output_path / 'data.yaml'
    
    with open(data_yaml_path, 'w') as f:
        f.write(f"""# YOLO Detection Dataset Configuration
# CFH (股骨头) 检测数据集

# 数据集路径
path: .  # 数据集根目录
train: images  # 训练图像目录
val: images    # 验证图像目录（暂时使用相同目录）

# 类别数量
nc: 1

# 类别名称
names:
  0: CFH

# 数据集信息
dataset_info:
  description: "股骨头(CFH)检测数据集"
  source: "LAT202511"
  total_images: {total_images}
  total_objects: {total_cfh}
  bbox_multiplier: {args.bbox_multiplier}
  note: "从单个中心点扩展为检测框"
""")
    
    print(f'✅ 数据集配置已保存: {data_yaml_path}')
    print()
    print('下一步:')
    print(f'  1. 查看数据集: ls -lh {args.output}/')
    print(f'  2. 训练模型: yolo detect train data={data_yaml_path} model=yolo11n.pt')
    print()


def visualize_samples(dataset_dir, num_samples=5):
    """可视化转换结果"""
    import random

    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    output_dir = dataset_path / 'visualizations'
    output_dir.mkdir(exist_ok=True)

    image_files = list(images_dir.glob('*.png'))

    if not image_files:
        print('❌ 没有找到图像文件')
        return

    samples = random.sample(image_files, min(num_samples, len(image_files)))

    print(f'\n📊 可视化 {len(samples)} 个样本...')

    for img_file in samples:
        label_file = labels_dir / f'{img_file.stem}.txt'

        if not label_file.exists():
            continue

        # 读取图像
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        h, w = img.shape[:2]

        # 读取标注
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, cx, cy, bw, bh = map(float, parts[:5])

                    # 转换为像素坐标
                    cx_px = int(cx * w)
                    cy_px = int(cy * h)
                    bw_px = int(bw * w)
                    bh_px = int(bh * h)

                    # 计算边界框
                    x1 = int(cx_px - bw_px / 2)
                    y1 = int(cy_px - bh_px / 2)
                    x2 = int(cx_px + bw_px / 2)
                    y2 = int(cy_px + bh_px / 2)

                    # 绘制边界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # 绘制中心点
                    cv2.circle(img, (cx_px, cy_px), 8, (0, 0, 255), -1)

                    # 添加标签
                    label_text = f'CFH ({bw_px}x{bh_px})'
                    cv2.putText(img, label_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 保存可视化结果
        output_path = output_dir / f'vis_{img_file.name}'
        cv2.imwrite(str(output_path), img)
        print(f'  ✅ {output_path.name}')

    print(f'\n✅ 可视化结果保存在: {output_dir}')


if __name__ == '__main__':
    main()

