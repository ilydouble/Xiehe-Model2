#!/usr/bin/env python3
"""
可视化YOLO Corner数据集标注
用于检查脊柱椎体角点检测数据集的标注质量
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import random
import yaml


# 18个椎体类别的颜色映射（使用不同的颜色便于区分）
COLORS = {
    0: (255, 0, 0),      # C7 - 红色
    1: (255, 128, 0),    # L1 - 橙色
    2: (255, 255, 0),    # L2 - 黄色
    3: (128, 255, 0),    # L3 - 黄绿色
    4: (0, 255, 0),      # L4 - 绿色
    5: (0, 255, 128),    # L5 - 青绿色
    6: (0, 255, 255),    # T1 - 青色
    7: (0, 128, 255),    # T2 - 浅蓝色
    8: (0, 0, 255),      # T3 - 蓝色
    9: (128, 0, 255),    # T4 - 紫色
    10: (255, 0, 255),   # T5 - 品红色
    11: (255, 0, 128),   # T6 - 粉红色
    12: (255, 64, 64),   # T7 - 浅红色
    13: (255, 128, 64),  # T8 - 浅橙色
    14: (255, 192, 64),  # T9 - 浅黄色
    15: (192, 255, 64),  # T10 - 浅黄绿色
    16: (64, 255, 128),  # T11 - 浅绿色
    17: (64, 192, 255),  # T12 - 浅蓝色
}


def load_class_names(data_yaml_path):
    """从data.yaml加载类别名称"""
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data['names']


def parse_yolo_corner_line(line):
    """
    解析YOLO Corner格式的标注行
    格式: class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v kp3_x kp3_y kp3_v kp4_x kp4_y kp4_v
    
    返回:
        class_id: 类别ID
        bbox: [cx, cy, w, h] 归一化边界框
        keypoints: [[x1, y1, v1], [x2, y2, v2], [x3, y3, v3], [x4, y4, v4]] 归一化关键点
    """
    parts = line.strip().split()
    class_id = int(parts[0])
    
    # 边界框 (归一化坐标)
    bbox = [float(x) for x in parts[1:5]]
    
    # 4个关键点，每个3个值 (x, y, visibility)
    keypoints = []
    for i in range(4):
        idx = 5 + i * 3
        kp = [float(parts[idx]), float(parts[idx + 1]), int(parts[idx + 2])]
        keypoints.append(kp)
    
    return class_id, bbox, keypoints


def visualize_annotation(image_path, label_path, class_names, output_path=None, show=True):
    """
    可视化单张图像的标注
    
    Args:
        image_path: 图像路径
        label_path: 标注文件路径
        class_names: 类别名称字典
        output_path: 输出路径（可选）
        show: 是否显示图像
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # 读取标注
    if not label_path.exists():
        print(f"⚠️  标注文件不存在: {label_path}")
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    print(f"\n📊 图像: {image_path.name}")
    print(f"   尺寸: {w} x {h}")
    print(f"   椎体数量: {len(lines)}")
    
    # 绘制每个椎体
    for line in lines:
        class_id, bbox, keypoints = parse_yolo_corner_line(line)
        
        # 获取类别名称和颜色
        class_name = class_names[class_id]
        color = COLORS.get(class_id, (255, 255, 255))
        
        # 转换归一化坐标到像素坐标
        cx, cy, bw, bh = bbox
        cx_px = int(cx * w)
        cy_px = int(cy * h)
        bw_px = int(bw * w)
        bh_px = int(bh * h)
        
        # 计算边界框左上角和右下角
        x1 = int(cx_px - bw_px / 2)
        y1 = int(cy_px - bh_px / 2)
        x2 = int(cx_px + bw_px / 2)
        y2 = int(cy_px + bh_px / 2)
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签
        label_text = f"{class_name}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制4个角点
        kp_pixels = []
        for i, (kp_x, kp_y, visibility) in enumerate(keypoints):
            if visibility > 0:  # 只绘制可见的关键点
                kp_x_px = int(kp_x * w)
                kp_y_px = int(kp_y * h)
                kp_pixels.append((kp_x_px, kp_y_px))
                
                # 绘制关键点（圆圈）
                cv2.circle(img, (kp_x_px, kp_y_px), 5, color, -1)
                cv2.circle(img, (kp_x_px, kp_y_px), 6, (255, 255, 255), 1)
                
                # 绘制关键点编号
                cv2.putText(img, str(i+1), (kp_x_px + 8, kp_y_px - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 连接4个角点形成矩形
        if len(kp_pixels) == 4:
            for i in range(4):
                pt1 = kp_pixels[i]
                pt2 = kp_pixels[(i + 1) % 4]
                cv2.line(img, pt1, pt2, color, 1)
    
    # 保存或显示
    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"✅ 已保存到: {output_path}")
    
    if show:
        # 调整显示大小（如果图像太大）
        max_height = 1200
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
        else:
            img_resized = img
        
        cv2.imshow('YOLO Corner Visualization', img_resized)
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='可视化YOLO Corner数据集标注')
    parser.add_argument('--dataset', type=str, default='datasets/yolo_corner',
                       help='数据集根目录')
    parser.add_argument('--image', type=str, default=None,
                       help='指定要可视化的图像名称（不含扩展名）')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='随机可视化的样本数量（当未指定--image时）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（可选）')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图像，只保存')
    
    args = parser.parse_args()
    
    # 路径设置
    dataset_root = Path(args.dataset)
    images_dir = dataset_root / 'images'
    labels_dir = dataset_root / 'labels'
    data_yaml = dataset_root / 'data.yaml'
    
    # 检查路径
    if not dataset_root.exists():
        print(f"❌ 数据集目录不存在: {dataset_root}")
        return
    
    # 加载类别名称
    class_names = load_class_names(data_yaml)
    print(f"📋 类别数量: {len(class_names)}")
    print(f"   类别: {', '.join([f'{k}:{v}' for k, v in class_names.items()])}")
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # 获取要可视化的图像列表
    if args.image:
        # 指定图像
        image_files = [images_dir / f"{args.image}.png"]
        if not image_files[0].exists():
            print(f"❌ 图像不存在: {image_files[0]}")
            return
    else:
        # 随机选择
        all_images = list(images_dir.glob('*.png'))
        if len(all_images) == 0:
            print(f"❌ 未找到图像文件: {images_dir}")
            return
        
        num_samples = min(args.num_samples, len(all_images))
        image_files = random.sample(all_images, num_samples)
        print(f"\n🎲 随机选择 {num_samples} 张图像进行可视化")
    
    # 可视化每张图像
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        if output_dir:
            output_path = output_dir / f"vis_{image_path.name}"
        else:
            output_path = None
        
        visualize_annotation(
            image_path, 
            label_path, 
            class_names,
            output_path=output_path,
            show=not args.no_show
        )
    
    print("\n✅ 可视化完成！")


if __name__ == '__main__':
    main()

