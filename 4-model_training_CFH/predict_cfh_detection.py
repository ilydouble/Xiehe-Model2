#!/usr/bin/env python3
"""
CFH检测推理脚本
使用训练好的YOLO模型进行股骨头检测
"""
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def predict_cfh(model_path, source, conf=0.25, save_dir='runs/cfh_predict', visualize=True):
    """
    CFH检测推理
    
    参数:
        model_path: 模型路径
        source: 输入源（图像路径或目录）
        conf: 置信度阈值
        save_dir: 保存目录
        visualize: 是否可视化
    """
    print('=' * 80)
    print('🔍 CFH检测推理')
    print('=' * 80)
    print(f'模型: {model_path}')
    print(f'输入: {source}')
    print(f'置信度: {conf}')
    print()
    
    # 检查模型
    if not Path(model_path).exists():
        print(f'❌ 模型不存在: {model_path}')
        return
    
    # 加载模型
    print('📦 加载模型...')
    model = YOLO(model_path)
    
    # 推理
    print('🔍 开始检测...')
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='predict',
        verbose=True
    )
    
    print()
    print('=' * 80)
    print('✅ 检测完成！')
    print('=' * 80)
    
    # 统计结果
    total_detections = 0
    for result in results:
        total_detections += len(result.boxes)
    
    print(f'检测到 {total_detections} 个CFH')
    print(f'结果保存在: {save_dir}/predict')
    print()
    
    # 可视化
    if visualize and results:
        visualize_results(results, save_dir)
    
    return results


def visualize_results(results, save_dir):
    """可视化检测结果"""
    print('📊 生成可视化...')
    
    vis_dir = Path(save_dir) / 'predict' / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        # 获取原始图像
        img = result.orig_img.copy()
        
        # 绘制检测框
        for box in result.boxes:
            # 获取坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # 计算中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 绘制中心点
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
            
            # 添加标签
            label = f'CFH {conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            
            # 绘制标签背景
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # 绘制标签文字
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 保存可视化结果
        if hasattr(result, 'path'):
            img_name = Path(result.path).name
        else:
            img_name = f'result_{i}.png'
        
        output_path = vis_dir / f'vis_{img_name}'
        cv2.imwrite(str(output_path), img)
        print(f'  ✅ {output_path.name}')
    
    print(f'\n✅ 可视化保存在: {vis_dir}')


def batch_predict(model_path, image_dir, output_csv='cfh_detections.csv', conf=0.25):
    """
    批量检测并保存结果到CSV
    
    参数:
        model_path: 模型路径
        image_dir: 图像目录
        output_csv: 输出CSV文件
        conf: 置信度阈值
    """
    print('=' * 80)
    print('📊 批量CFH检测')
    print('=' * 80)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取所有图像
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    
    print(f'找到 {len(image_files)} 张图像')
    print()
    
    # 批量推理
    results_list = []
    
    for img_file in image_files:
        results = model.predict(source=str(img_file), conf=conf, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf_score = float(box.conf[0])
                
                results_list.append({
                    'image': img_file.name,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': conf_score,
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2
                })
    
    # 保存到CSV
    import pandas as pd
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    
    print(f'✅ 检测到 {len(results_list)} 个CFH')
    print(f'✅ 结果保存到: {output_csv}')
    print()
    
    return df


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CFH检测推理')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--source', type=str, required=True,
                       help='输入源（图像或目录）')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (默认: 0.25)')
    parser.add_argument('--save-dir', type=str, default='runs/cfh_predict',
                       help='保存目录 (默认: runs/cfh_predict)')
    parser.add_argument('--batch-csv', type=str, default=None,
                       help='批量检测并保存到CSV')
    
    args = parser.parse_args()
    
    if args.batch_csv:
        # 批量检测模式
        batch_predict(args.model, args.source, args.batch_csv, args.conf)
    else:
        # 单次检测模式
        predict_cfh(args.model, args.source, args.conf, args.save_dir)


if __name__ == '__main__':
    main()

