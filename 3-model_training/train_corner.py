#!/usr/bin/env python3
"""
YOLO11 Pose训练脚本 - Corner数据集
用于训练脊柱椎体角点检测模型（18个椎体类别，每个4个关键点）
"""
import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO


def load_config(config_file='train_configs_corner.yaml', config_name='standard'):
    """加载训练配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    
    if config_name not in configs:
        raise ValueError(f"配置 '{config_name}' 不存在。可用配置: {list(configs.keys())}")
    
    # 获取基础配置
    config = configs[config_name].copy()
    
    # 合并其他配置
    if 'augmentation' in configs:
        config.update(configs['augmentation'])
    if 'optimizer' in configs:
        config.update(configs['optimizer'])
    if 'loss_weights' in configs:
        config.update(configs['loss_weights'])
    if 'validation' in configs:
        config.update(configs['validation'])
    
    return config


def train_model(config, device='0', resume=False):
    """
    训练YOLO11 Pose模型
    
    参数:
        config: 训练配置字典
        device: 训练设备
        resume: 是否从上次中断处继续训练
    """
    print('=' * 80)
    print('🚀 开始训练YOLO11 Pose模型 - Corner数据集')
    print('=' * 80)
    print(f'模型大小: {config["model_size"]}')
    print(f'训练轮数: {config["epochs"]}')
    print(f'批次大小: {config["batch"]}')
    print(f'图像大小: {config["imgsz"]}')
    print(f'训练设备: {device}')
    print(f'数据集: {config["data"]}')
    print(f'使用预训练权重: {config["pretrained"]}')
    print('=' * 80)
    print()
    
    # 检查数据集
    data_yaml = config['data']
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f'数据集配置文件不存在: {data_yaml}')
    
    # 加载模型
    model_name = f'yolo11{config["model_size"]}-pose.pt'
    
    if resume:
        # 从上次训练继续
        last_checkpoint = Path('runs/pose') / config['name'] / 'weights/last.pt'
        if last_checkpoint.exists():
            print(f'从检查点继续训练: {last_checkpoint}')
            model = YOLO(str(last_checkpoint))
        else:
            print(f'⚠️  未找到检查点，从头开始训练')
            model = YOLO(model_name)
    else:
        if config['pretrained']:
            print(f'加载预训练模型: {model_name}')
            model = YOLO(model_name)
        else:
            print(f'从头训练模型: {model_name}')
            model = YOLO(model_name.replace('.pt', '.yaml'))
    
    # 训练参数
    train_args = {
        'data': data_yaml,
        'epochs': config['epochs'],
        'batch': config['batch'],
        'imgsz': config['imgsz'],
        'device': device,
        'project': 'runs/pose',
        'name': config['name'],
        'exist_ok': True,
        'pretrained': config['pretrained'],
        'optimizer': config.get('optimizer', 'AdamW'),
        'lr0': config.get('lr0', 0.001),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3.0),
        'warmup_momentum': config.get('warmup_momentum', 0.8),
        'warmup_bias_lr': config.get('warmup_bias_lr', 0.1),
        'box': config.get('box', 7.5),
        'cls': config.get('cls', 0.5),
        'dfl': config.get('dfl', 1.5),
        'pose': config.get('pose', 12.0),
        'kobj': config.get('kobj', 1.0),
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': config.get('hsv_h', 0.015),
        'hsv_s': config.get('hsv_s', 0.7),
        'hsv_v': config.get('hsv_v', 0.4),
        'degrees': config.get('degrees', 0.0),
        'translate': config.get('translate', 0.1),
        'scale': config.get('scale', 0.5),
        'shear': config.get('shear', 0.0),
        'perspective': config.get('perspective', 0.0),
        'flipud': config.get('flipud', 0.0),
        'fliplr': config.get('fliplr', 0.5),
        'mosaic': config.get('mosaic', 1.0),
        'mixup': config.get('mixup', 0.0),
        'copy_paste': config.get('copy_paste', 0.0),
        'multi_scale': config.get('multi_scale', 0.5),  # 多尺度训练
        'save': True,
        'save_period': config.get('save_period', -1),
        'val': config.get('val', True),
        'plots': config.get('plots', True),
        'verbose': config.get('verbose', True),
    }
    
    # 开始训练
    print('开始训练...')
    print()
    results = model.train(**train_args)
    
    print()
    print('=' * 80)
    print('✅ 训练完成！')
    print('=' * 80)
    print(f'模型保存在: runs/pose/{config["name"]}/weights/')
    print(f'  - best.pt: 最佳模型（用于推理）')
    print(f'  - last.pt: 最后一轮模型（用于继续训练）')
    print()
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练YOLO11 Pose模型 - Corner数据集')
    parser.add_argument('--config', type=str, default='standard',
                       help='配置名称 (quick_test, standard, high_accuracy, best_performance, from_scratch)')
    parser.add_argument('--config-file', type=str, default='train_configs_corner.yaml',
                       help='配置文件路径')
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu',
                       help='训练设备 (0, 1, 2, ... 或 cpu)')
    parser.add_argument('--resume', action='store_true',
                       help='从上次中断处继续训练')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config_file, args.config)
    
    # 训练模型
    train_model(config, args.device, args.resume)


if __name__ == '__main__':
    main()

