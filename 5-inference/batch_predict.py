#!/usr/bin/env python3
"""
批量预测脚本
同时使用Corner模型和CFH检测模型进行预测，并可视化结果
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import json
import random


def draw_corner_results(image, result, colors):
    """绘制Corner模型结果（椎体+关键点）"""
    img = image.copy()
    h, w = img.shape[:2]
    
    # 椎体类别名称
    vertebra_names = ['C7', 'L1', 'L2', 'L3', 'L4', 'L5', 
                      'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
                      'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    
    if result.keypoints is not None and len(result.keypoints) > 0:
        for i, (box, kpts) in enumerate(zip(result.boxes, result.keypoints)):
            # 获取类别和置信度
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id >= len(vertebra_names):
                continue
            
            vertebra_name = vertebra_names[cls_id]
            color = colors[cls_id % len(colors)]
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f'{vertebra_name} {conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 绘制关键点
            if kpts.data is not None and len(kpts.data) > 0:
                keypoints = kpts.data[0].cpu().numpy()  # shape: (4, 3)
                
                # 绘制4个角点
                for j, (kx, ky, kv) in enumerate(keypoints):
                    if kv > 0.5:  # 可见性阈值
                        kx_px = int(kx)
                        ky_px = int(ky)
                        
                        # 绘制关键点
                        cv2.circle(img, (kx_px, ky_px), 5, color, -1)
                        cv2.circle(img, (kx_px, ky_px), 7, (255, 255, 255), 2)
                        
                        # 标注角点编号
                        cv2.putText(img, str(j+1), (kx_px + 10, ky_px - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 连接关键点形成四边形
                if len(keypoints) == 4:
                    pts = []
                    for kx, ky, kv in keypoints:
                        if kv > 0.5:
                            pts.append([int(kx), int(ky)])
                    
                    if len(pts) == 4:
                        pts = np.array(pts, dtype=np.int32)
                        cv2.polylines(img, [pts], True, color, 2)
    
    return img


def draw_cfh_results(image, result, color=(0, 255, 0)):
    """绘制CFH检测结果"""
    img = image.copy()
    
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # 获取坐标和置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # 计算中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # 绘制中心点
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(img, (cx, cy), 12, (255, 255, 255), 2)
            
            # 绘制标签
            label = f'CFH {conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(img, label, (x1 + 5, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return img


def draw_combined_results(image, corner_result, cfh_result, colors):
    """绘制组合结果（Corner + CFH）"""
    img = image.copy()
    
    # 先绘制Corner结果
    img = draw_corner_results(img, corner_result, colors)
    
    # 再绘制CFH结果
    img = draw_cfh_results(img, cfh_result, color=(255, 0, 255))
    
    # 添加图例
    legend_y = 30
    cv2.putText(img, 'Corner Model: Vertebrae + Keypoints', (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, 'CFH Model: Femoral Head', (10, legend_y + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    return img


def batch_predict(
    corner_model_path,
    cfh_model_path,
    image_dir,
    output_dir,
    num_samples=10,
    conf_threshold=0.25
):
    """
    批量预测
    
    参数:
        corner_model_path: Corner模型路径
        cfh_model_path: CFH模型路径
        image_dir: 图像目录
        output_dir: 输出目录
        num_samples: 测试样本数量
        conf_threshold: 置信度阈值
    """
    print('=' * 80)
    print('🚀 批量预测测试')
    print('=' * 80)
    print(f'Corner模型: {corner_model_path}')
    print(f'CFH模型: {cfh_model_path}')
    print(f'图像目录: {image_dir}')
    print(f'输出目录: {output_dir}')
    print(f'样本数量: {num_samples}')
    print()
    
    # 创建输出目录
    output_path = Path(output_dir)
    corner_dir = output_path / 'corner_only'
    cfh_dir = output_path / 'cfh_only'
    combined_dir = output_path / 'combined'
    
    for d in [corner_dir, cfh_dir, combined_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print('📦 加载模型...')
    corner_model = YOLO(corner_model_path)
    cfh_model = YOLO(cfh_model_path)
    print('✅ 模型加载完成')
    print()
    
    # 获取图像文件
    image_path = Path(image_dir)
    image_files = list(image_path.glob('*.png')) + list(image_path.glob('*.jpg'))
    
    if len(image_files) == 0:
        print('❌ 未找到图像文件')
        return
    
    # 随机选择样本
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f'📊 从 {len(image_files)} 张图像中选择 {len(samples)} 个样本进行测试')
    print()
    
    # 生成颜色
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255)
    ]
    
    # 统计信息
    stats = {
        'total_images': len(samples),
        'corner_detections': 0,
        'cfh_detections': 0,
        'corner_keypoints': 0
    }
    
    # 批量预测
    print('🔍 开始预测...')
    for img_file in tqdm(samples, desc='处理图像'):
        # 读取图像
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Corner模型预测
        corner_results = corner_model.predict(
            source=str(img_file),
            conf=conf_threshold,
            verbose=False
        )

        # CFH模型预测
        cfh_results = cfh_model.predict(
            source=str(img_file),
            conf=conf_threshold,
            verbose=False
        )

        corner_result = corner_results[0]
        cfh_result = cfh_results[0]

        # 统计
        if corner_result.boxes is not None:
            stats['corner_detections'] += len(corner_result.boxes)

        if corner_result.keypoints is not None:
            for kpts in corner_result.keypoints:
                if kpts.data is not None:
                    stats['corner_keypoints'] += len(kpts.data[0])

        if cfh_result.boxes is not None:
            stats['cfh_detections'] += len(cfh_result.boxes)

        # 绘制结果
        corner_img = draw_corner_results(img, corner_result, colors)
        cfh_img = draw_cfh_results(img, cfh_result)
        combined_img = draw_combined_results(img, corner_result, cfh_result, colors)

        # 保存结果
        output_name = img_file.name
        cv2.imwrite(str(corner_dir / f'corner_{output_name}'), corner_img)
        cv2.imwrite(str(cfh_dir / f'cfh_{output_name}'), cfh_img)
        cv2.imwrite(str(combined_dir / f'combined_{output_name}'), combined_img)

    print()
    print('=' * 80)
    print('✅ 预测完成！')
    print('=' * 80)
    print(f'处理图像: {stats["total_images"]}')
    print(f'Corner检测: {stats["corner_detections"]} 个椎体')
    print(f'Corner关键点: {stats["corner_keypoints"]} 个点')
    print(f'CFH检测: {stats["cfh_detections"]} 个股骨头')
    print()
    print('结果保存在:')
    print(f'  Corner结果: {corner_dir}')
    print(f'  CFH结果: {cfh_dir}')
    print(f'  组合结果: {combined_dir}')
    print()

    # 保存统计信息
    stats_file = output_path / 'prediction_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'📊 统计信息已保存: {stats_file}')

    return stats


def create_comparison_grid(output_dir, num_display=4):
    """创建对比网格图"""
    print()
    print('📊 创建对比网格图...')

    output_path = Path(output_dir)
    corner_dir = output_path / 'corner_only'
    cfh_dir = output_path / 'cfh_only'
    combined_dir = output_path / 'combined'

    # 获取文件
    corner_files = sorted(list(corner_dir.glob('corner_*.png')))[:num_display]

    if len(corner_files) == 0:
        print('❌ 没有找到结果图像')
        return

    # 创建网格
    for corner_file in corner_files:
        base_name = corner_file.name.replace('corner_', '')
        cfh_file = cfh_dir / f'cfh_{base_name}'
        combined_file = combined_dir / f'combined_{base_name}'

        if not cfh_file.exists() or not combined_file.exists():
            continue

        # 读取图像
        corner_img = cv2.imread(str(corner_file))
        cfh_img = cv2.imread(str(cfh_file))
        combined_img = cv2.imread(str(combined_file))

        if corner_img is None or cfh_img is None or combined_img is None:
            continue

        # 调整大小
        h, w = corner_img.shape[:2]
        target_w = 800
        target_h = int(h * target_w / w)

        corner_img = cv2.resize(corner_img, (target_w, target_h))
        cfh_img = cv2.resize(cfh_img, (target_w, target_h))
        combined_img = cv2.resize(combined_img, (target_w, target_h))

        # 添加标题
        title_h = 50
        corner_with_title = np.ones((target_h + title_h, target_w, 3), dtype=np.uint8) * 255
        cfh_with_title = np.ones((target_h + title_h, target_w, 3), dtype=np.uint8) * 255
        combined_with_title = np.ones((target_h + title_h, target_w, 3), dtype=np.uint8) * 255

        corner_with_title[title_h:, :] = corner_img
        cfh_with_title[title_h:, :] = cfh_img
        combined_with_title[title_h:, :] = combined_img

        cv2.putText(corner_with_title, 'Corner Model', (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 128, 0), 2)
        cv2.putText(cfh_with_title, 'CFH Detection Model', (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 255), 2)
        cv2.putText(combined_with_title, 'Combined Results', (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)

        # 水平拼接
        grid = np.hstack([corner_with_title, cfh_with_title, combined_with_title])

        # 保存网格图
        grid_file = output_path / f'grid_{base_name}'
        cv2.imwrite(str(grid_file), grid)
        print(f'  ✅ {grid_file.name}')

    print(f'✅ 对比网格图已保存到: {output_path}')


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='批量预测测试')
    parser.add_argument('--corner-model', type=str,
                       default='3-model_training/runs/pose/yolo11s_corner_standard/weights/best.pt',
                       help='Corner模型路径')
    parser.add_argument('--cfh-model', type=str,
                       default='4-model_training_CFH/runs/cfh_detection/standard/weights/best.pt',
                       help='CFH模型路径')
    parser.add_argument('--image-dir', type=str,
                       default='datasets/yolo_corner/images',
                       help='图像目录')
    parser.add_argument('--output-dir', type=str,
                       default='datasets/prediction_results',
                       help='输出目录')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='测试样本数量')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--create-grid', action='store_true',
                       help='创建对比网格图')

    args = parser.parse_args()

    # 批量预测
    stats = batch_predict(
        corner_model_path=args.corner_model,
        cfh_model_path=args.cfh_model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        conf_threshold=args.conf
    )

    # 创建对比网格
    if args.create_grid:
        create_comparison_grid(args.output_dir, num_display=min(4, args.num_samples))

    print()
    print('🎉 全部完成！')
    print(f'查看结果: open {args.output_dir}')


if __name__ == '__main__':
    main()

