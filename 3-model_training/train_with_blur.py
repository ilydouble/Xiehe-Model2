#!/usr/bin/env python3
"""
训练脚本 - 带模糊增强
使用 Albumentations 添加模糊增强
"""
import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️  警告: 未安装 albumentations")
    print("   安装命令: pip install albumentations")


def load_config(config_file='train_configs_corner.yaml', config_name='best_performance'):
    """加载训练配置"""
    if not Path(config_file).exists():
        raise FileNotFoundError(f'配置文件不存在: {config_file}')
    
    with open(config_file, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    
    if config_name not in configs:
        raise ValueError(f'配置不存在: {config_name}')
    
    return configs[config_name]


def get_blur_augmentations(strength='medium'):
    """
    获取模糊增强配置
    
    strength: 'light', 'medium', 'strong'
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    
    if strength == 'light':
        # 轻度模糊 - 推荐用于医学影像
        return [
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.2),  # 20% 概率应用模糊
        ]
    
    elif strength == 'medium':
        # 中度模糊
        return [
            A.OneOf([
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),  # 30% 概率应用模糊
        ]
    
    elif strength == 'strong':
        # 强度模糊 - 不推荐用于关键点检测
        return [
            A.OneOf([
                A.GaussianBlur(blur_limit=7, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.5),  # 50% 概率应用模糊
        ]
    
    return None


def train_with_blur(config, device='0', blur_strength='light'):
    """
    训练模型（带模糊增强）
    """
    print('=' * 80)
    print('🚀 训练 YOLO11 Pose 模型 - 带模糊增强')
    print('=' * 80)
    print(f'模型大小: {config["model_size"]}')
    print(f'训练轮数: {config["epochs"]}')
    print(f'批次大小: {config["batch"]}')
    print(f'图像大小: {config["imgsz"]}')
    print(f'模糊强度: {blur_strength}')
    print('=' * 80)
    print()
    
    # 检查 Albumentations
    if not ALBUMENTATIONS_AVAILABLE:
        print('❌ 未安装 albumentations，无法使用模糊增强')
        print('   请运行: pip install albumentations')
        return
    
    # 获取模糊增强
    blur_augs = get_blur_augmentations(blur_strength)
    
    # 加载模型
    model_name = f'yolo11{config["model_size"]}-pose.pt'
    print(f'📦 加载模型: {model_name}')
    model = YOLO(model_name)
    
    # 训练参数
    train_args = {
        'data': config['data'],
        'epochs': config['epochs'],
        'batch': config['batch'],
        'imgsz': config['imgsz'],
        'device': device,
        'project': 'runs/pose',
        'name': f'{config["name"]}_with_blur',
        'exist_ok': True,
        
        # 优化器
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'weight_decay': 0.0005,
        
        # 损失权重
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        
        # 数据增强（YOLO内置）
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'multi_scale': 0.5,
        
        # Albumentations 模糊增强
        'augment': True,
    }
    
    print('📋 训练参数:')
    for key, value in train_args.items():
        if key != 'augment':
            print(f'  {key}: {value}')
    
    print(f'\n📸 模糊增强配置:')
    print(f'  强度: {blur_strength}')
    print(f'  类型: 高斯模糊, 运动模糊, 中值模糊')
    print()
    
    # 开始训练（带自定义增强）
    print('🏋️ 开始训练...')
    results = model.train(**train_args, augmentations=blur_augs)
    
    print()
    print('=' * 80)
    print('✅ 训练完成！')
    print('=' * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练YOLO11 - 带模糊增强')
    parser.add_argument('--config', type=str, default='best_performance',
                       help='配置名称')
    parser.add_argument('--device', type=str, default='0',
                       help='训练设备')
    parser.add_argument('--blur', type=str, default='light',
                       choices=['light', 'medium', 'strong'],
                       help='模糊强度 (推荐: light)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config('train_configs_corner.yaml', args.config)
    
    # 训练
    train_with_blur(config, args.device, args.blur)
