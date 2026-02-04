#!/usr/bin/env python3
"""
YOLO11 Pose推理脚本 - Corner数据集
用于脊柱椎体角点检测（18个椎体类别，每个4个关键点）
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse


# 椎体类别名称（18个类别）
VERTEBRA_NAMES = {
    0: 'C7', 1: 'L1', 2: 'L2', 3: 'L3', 4: 'L4', 5: 'L5',
    6: 'T1', 7: 'T2', 8: 'T3', 9: 'T4', 10: 'T5', 11: 'T6',
    12: 'T7', 13: 'T8', 14: 'T9', 15: 'T10', 16: 'T11', 17: 'T12'
}

# 颜色映射（为每个椎体分配不同颜色）
COLORS = {
    'C7': (255, 0, 0),      # 红色
    'T1': (255, 128, 0),    # 橙色
    'T2': (255, 255, 0),    # 黄色
    'T3': (128, 255, 0),    # 黄绿色
    'T4': (0, 255, 0),      # 绿色
    'T5': (0, 255, 128),    # 青绿色
    'T6': (0, 255, 255),    # 青色
    'T7': (0, 128, 255),    # 浅蓝色
    'T8': (0, 0, 255),      # 蓝色
    'T9': (128, 0, 255),    # 紫色
    'T10': (255, 0, 255),   # 品红色
    'T11': (255, 0, 128),   # 粉红色
    'T12': (255, 128, 128), # 浅红色
    'L1': (128, 255, 128),  # 浅绿色
    'L2': (128, 128, 255),  # 浅蓝色
    'L3': (255, 255, 128),  # 浅黄色
    'L4': (255, 128, 255),  # 浅紫色
    'L5': (128, 255, 255),  # 浅青色
}


def draw_keypoints(image, results, conf_threshold=0.5):
    """
    在图像上绘制检测结果
    
    参数:
        image: 输入图像
        results: YOLO检测结果
        conf_threshold: 置信度阈值
    """
    img = image.copy()
    
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        
        if boxes is None or keypoints is None:
            continue
        
        for i in range(len(boxes)):
            # 获取边界框
            box = boxes[i]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if conf < conf_threshold:
                continue
            
            # 获取类别名称和颜色
            class_name = VERTEBRA_NAMES.get(cls, f'Class_{cls}')
            color = COLORS.get(class_name, (255, 255, 255))
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f'{class_name} {conf:.2f}'
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制关键点（4个角点）
            kpts = keypoints[i].xy[0].cpu().numpy()  # shape: (4, 2)
            kpts_conf = keypoints[i].conf[0].cpu().numpy() if keypoints[i].conf is not None else np.ones(4)
            
            # 绘制关键点
            for j, (kpt, kpt_c) in enumerate(zip(kpts, kpts_conf)):
                if kpt_c > 0.5:  # 关键点置信度阈值
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(img, (x, y), 5, color, -1)
                    cv2.circle(img, (x, y), 7, (255, 255, 255), 2)
                    # 标注关键点编号
                    cv2.putText(img, str(j+1), (x+10, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 连接关键点形成矩形（按顺序连接4个角点）
            valid_kpts = [(int(kpt[0]), int(kpt[1])) for kpt, kpt_c in zip(kpts, kpts_conf) if kpt_c > 0.5]
            if len(valid_kpts) >= 2:
                for j in range(len(valid_kpts)):
                    pt1 = valid_kpts[j]
                    pt2 = valid_kpts[(j + 1) % len(valid_kpts)]
                    cv2.line(img, pt1, pt2, color, 2)
    
    return img


def predict_image(model, image_path, output_dir, conf_threshold=0.5, save_txt=False):
    """
    对单张图像进行推理
    
    参数:
        model: YOLO模型
        image_path: 图像路径
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        save_txt: 是否保存文本结果
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f'❌ 无法读取图像: {image_path}')
        return
    
    # 推理
    results = model(image, conf=conf_threshold, verbose=False)
    
    # 绘制结果
    output_image = draw_keypoints(image, results, conf_threshold)
    
    # 保存结果
    output_path = Path(output_dir) / f'{Path(image_path).stem}_pred.png'
    cv2.imwrite(str(output_path), output_image)
    
    # 保存文本结果
    if save_txt:
        txt_path = Path(output_dir) / f'{Path(image_path).stem}_pred.txt'
        with open(txt_path, 'w') as f:
            for result in results:
                boxes = result.boxes
                keypoints = result.keypoints
                
                if boxes is None or keypoints is None:
                    continue
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf < conf_threshold:
                        continue
                    
                    class_name = VERTEBRA_NAMES.get(cls, f'Class_{cls}')
                    kpts = keypoints[i].xy[0].cpu().numpy()
                    
                    f.write(f'{class_name} {conf:.4f}\n')
                    for j, kpt in enumerate(kpts):
                        f.write(f'  Point{j+1}: ({kpt[0]:.2f}, {kpt[1]:.2f})\n')
    
    print(f'✅ 保存结果: {output_path}')
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO11 Pose推理 - Corner数据集')
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径 (例如: runs/pose/yolo11s_corner_standard/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                       help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='runs/predict_corner',
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值')
    parser.add_argument('--save-txt', action='store_true',
                       help='保存文本结果')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f'加载模型: {args.model}')
    model = YOLO(args.model)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 推理
    source_path = Path(args.source)
    
    if source_path.is_file():
        # 单张图像
        predict_image(model, source_path, output_dir, args.conf, args.save_txt)
    elif source_path.is_dir():
        # 目录
        image_files = list(source_path.glob('*.png')) + list(source_path.glob('*.jpg'))
        print(f'找到 {len(image_files)} 张图像')
        
        for image_file in image_files:
            predict_image(model, image_file, output_dir, args.conf, args.save_txt)
    else:
        print(f'❌ 无效的输入路径: {args.source}')
    
    print(f'\n✅ 推理完成！结果保存在: {output_dir}')


if __name__ == '__main__':
    main()

