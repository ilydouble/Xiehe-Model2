#!/usr/bin/env python3
"""
测试Corner数据集的训练和推理流程
快速验证所有脚本是否正常工作
"""
import os
import sys
from pathlib import Path
import yaml


def test_config_loading():
    """测试配置文件加载"""
    print('=' * 80)
    print('📋 测试1: 配置文件加载')
    print('=' * 80)
    
    config_file = 'train_configs_corner.yaml'
    
    if not Path(config_file).exists():
        print(f'❌ 配置文件不存在: {config_file}')
        return False
    
    with open(config_file, 'r', encoding='utf-8') as f:
        configs = yaml.safe_load(f)
    
    required_configs = ['quick_test', 'standard', 'high_accuracy', 'best_performance', 'from_scratch']
    
    for config_name in required_configs:
        if config_name not in configs:
            print(f'❌ 缺少配置: {config_name}')
            return False
        
        config = configs[config_name]
        required_keys = ['model_size', 'epochs', 'batch', 'imgsz', 'pretrained', 'name', 'data']
        
        for key in required_keys:
            if key not in config:
                print(f'❌ 配置 {config_name} 缺少键: {key}')
                return False
    
    print('✅ 配置文件格式正确')
    print(f'   可用配置: {", ".join(required_configs)}')
    print()
    return True


def test_dataset():
    """测试数据集"""
    print('=' * 80)
    print('📊 测试2: 数据集检查')
    print('=' * 80)
    
    data_yaml = Path('../datasets/yolo_corner/data.yaml')
    images_dir = Path('../datasets/yolo_corner/images')
    labels_dir = Path('../datasets/yolo_corner/labels')
    
    if not data_yaml.exists():
        print(f'❌ 数据集配置文件不存在: {data_yaml}')
        return False
    
    if not images_dir.exists():
        print(f'❌ 图像目录不存在: {images_dir}')
        return False
    
    if not labels_dir.exists():
        print(f'❌ 标注目录不存在: {labels_dir}')
        return False
    
    # 检查数据集配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    if 'nc' not in data_config or data_config['nc'] != 18:
        print(f'❌ 类别数量错误: 期望18，实际{data_config.get("nc", "未定义")}')
        return False
    
    if 'kpt_shape' not in data_config or data_config['kpt_shape'] != [4, 3]:
        print(f'❌ 关键点配置错误: 期望[4, 3]，实际{data_config.get("kpt_shape", "未定义")}')
        return False
    
    # 统计文件数量
    image_files = list(images_dir.glob('*.png'))
    label_files = list(labels_dir.glob('*.txt'))
    
    print(f'✅ 数据集配置正确')
    print(f'   类别数量: {data_config["nc"]}')
    print(f'   关键点配置: {data_config["kpt_shape"]}')
    print(f'   图像数量: {len(image_files)}')
    print(f'   标注数量: {len(label_files)}')
    print()
    return True


def test_scripts():
    """测试脚本文件"""
    print('=' * 80)
    print('📝 测试3: 脚本文件检查')
    print('=' * 80)
    
    required_files = {
        'train_corner.py': '训练脚本',
        'predict_corner.py': '推理脚本',
        'train_corner.sh': '训练启动脚本',
        'train_configs_corner.yaml': '配置文件',
        'README_CORNER.md': '文档'
    }
    
    all_exist = True
    for filename, description in required_files.items():
        filepath = Path(filename)
        if filepath.exists():
            print(f'✅ {description}: {filename}')
        else:
            print(f'❌ {description}不存在: {filename}')
            all_exist = False
    
    print()
    return all_exist


def test_imports():
    """测试Python依赖"""
    print('=' * 80)
    print('📦 测试4: Python依赖检查')
    print('=' * 80)
    
    required_packages = {
        'ultralytics': 'YOLO11',
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'yaml': 'PyYAML'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f'✅ {name}: 已安装')
        except ImportError:
            print(f'❌ {name}: 未安装 (pip install {package})')
            all_installed = False
    
    print()
    return all_installed


def test_sample_label():
    """测试标注格式"""
    print('=' * 80)
    print('🏷️  测试5: 标注格式检查')
    print('=' * 80)
    
    labels_dir = Path('../datasets/yolo_corner/labels')
    label_files = list(labels_dir.glob('*.txt'))
    
    if not label_files:
        print('❌ 没有找到标注文件')
        return False
    
    # 检查第一个标注文件
    sample_file = label_files[0]
    
    with open(sample_file, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        print(f'❌ 标注文件为空: {sample_file.name}')
        return False
    
    # 检查格式
    for i, line in enumerate(lines[:3]):  # 只检查前3行
        parts = line.strip().split()
        
        # YOLO Pose格式: class_id cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
        expected_length = 5 + 4 * 3  # 5 (bbox) + 4 keypoints * 3 (x, y, visibility)
        
        if len(parts) != expected_length:
            print(f'❌ 标注格式错误 (行{i+1}): 期望{expected_length}个值，实际{len(parts)}个')
            print(f'   内容: {line.strip()[:100]}...')
            return False
        
        # 检查类别ID
        class_id = int(parts[0])
        if class_id < 0 or class_id >= 18:
            print(f'❌ 类别ID超出范围 (行{i+1}): {class_id} (应该在0-17之间)')
            return False
    
    print(f'✅ 标注格式正确')
    print(f'   示例文件: {sample_file.name}')
    print(f'   标注数量: {len(lines)}')
    print()
    return True


def main():
    """主函数"""
    print()
    print('🧪 Corner数据集训练流程测试')
    print()
    
    tests = [
        ('配置文件加载', test_config_loading),
        ('数据集检查', test_dataset),
        ('脚本文件检查', test_scripts),
        ('Python依赖检查', test_imports),
        ('标注格式检查', test_sample_label),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f'❌ 测试失败: {test_name}')
            print(f'   错误: {e}')
            results.append((test_name, False))
    
    # 总结
    print('=' * 80)
    print('📊 测试总结')
    print('=' * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = '✅ 通过' if result else '❌ 失败'
        print(f'{status}: {test_name}')
    
    print()
    print(f'总计: {passed}/{total} 测试通过')
    print()
    
    if passed == total:
        print('🎉 所有测试通过！可以开始训练了。')
        print()
        print('下一步:')
        print('  1. 快速测试训练: ./train_corner.sh --config quick_test')
        print('  2. 标准训练: ./train_corner.sh --config standard')
        print()
        return 0
    else:
        print('⚠️  部分测试失败，请修复后再训练。')
        return 1


if __name__ == '__main__':
    sys.exit(main())

