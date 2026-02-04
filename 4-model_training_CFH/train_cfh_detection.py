#!/usr/bin/env python3
"""
CFH检测模型训练脚本
使用YOLO11进行股骨头检测
"""
import os
from pathlib import Path
from ultralytics import YOLO


def train_cfh_detection(
    model_size='m',
    epochs=150,
    imgsz=800,
    batch=8,
    device='0',
    project='runs/cfh_detection',
    name='train'
):
    """
    训练CFH检测模型
    
    参数:
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        epochs: 训练轮数
        imgsz: 图像大小
        batch: 批次大小
        device: 设备 ('0', '1', 'cpu')
        project: 项目目录
        name: 实验名称
    """
    print('=' * 80)
    print('🚀 CFH检测模型训练')
    print('=' * 80)
    print(f'模型: YOLO11{model_size}')
    print(f'训练轮数: {epochs}')
    print(f'图像大小: {imgsz}')
    print(f'批次大小: {batch}')
    print(f'设备: {device}')
    print()
    
    # 数据集路径
    data_yaml = '../datasets/yolo_cfh_detection/data.yaml'
    
    if not Path(data_yaml).exists():
        print(f'❌ 数据集配置文件不存在: {data_yaml}')
        return
    
    # 加载模型
    model_name = f'yolo11{model_size}.pt'
    print(f'📦 加载模型: {model_name}')
    model = YOLO(model_name)
    
    # 训练参数
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        
        # 优化器
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # 数据增强
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,      # 医学图像不旋转
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        
        # 其他
        'patience': 50,
        'save': True,
        'save_period': 10,
        'cache': False,
        'workers': 8,
        'verbose': True,
    }
    
    print('📋 训练参数:')
    for key, value in train_args.items():
        print(f'  {key}: {value}')
    print()
    
    # 开始训练
    print('🏋️ 开始训练...')
    print()
    
    results = model.train(**train_args)
    
    print()
    print('=' * 80)
    print('✅ 训练完成！')
    print('=' * 80)
    
    # 显示结果
    best_model = Path(project) / name / 'weights' / 'best.pt'
    last_model = Path(project) / name / 'weights' / 'last.pt'
    
    print(f'最佳模型: {best_model}')
    print(f'最后模型: {last_model}')
    print()
    
    # 验证
    print('📊 验证最佳模型...')
    model = YOLO(str(best_model))
    metrics = model.val(data=data_yaml)
    
    print()
    print('验证结果:')
    print(f'  mAP50: {metrics.box.map50:.4f}')
    print(f'  mAP50-95: {metrics.box.map:.4f}')
    print(f'  Precision: {metrics.box.mp:.4f}')
    print(f'  Recall: {metrics.box.mr:.4f}')
    print()
    
    print('下一步:')
    print(f'  1. 查看训练结果: open {project}/{name}')
    print(f'  2. 推理测试: yolo detect predict model={best_model} source=<图像路径>')
    print()
    
    return results, metrics


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练CFH检测模型')
    parser.add_argument('--model', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小 (默认: m)')
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数 (默认: 150)')
    parser.add_argument('--imgsz', type=int, default=800,
                       help='图像大小 (默认: 800)')
    parser.add_argument('--batch', type=int, default=8,
                       help='批次大小 (默认: 8)')
    parser.add_argument('--device', type=str, default='0',
                       help='设备 (默认: 0)')
    parser.add_argument('--project', type=str, default='runs/cfh_detection',
                       help='项目目录 (默认: runs/cfh_detection)')
    parser.add_argument('--name', type=str, default='train',
                       help='实验名称 (默认: train)')
    
    args = parser.parse_args()
    
    # 训练
    train_cfh_detection(
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == '__main__':
    main()

