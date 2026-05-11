#!/usr/bin/env python3
"""
测试更新后的模型
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from config import CORNER_MODEL_PATH, CFH_MODEL_PATH


def check_model_info(model_path, model_name):
    """检查模型信息"""
    print('=' * 80)
    print(f'📦 {model_name} 模型信息')
    print('=' * 80)
    
    if not model_path.exists():
        print(f'❌ 模型文件不存在: {model_path}')
        return False
    
    print(f'文件路径: {model_path}')
    print(f'文件大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB')
    
    try:
        # 加载模型
        model = YOLO(str(model_path))
        
        # 获取模型信息
        if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
            yaml_dict = model.model.yaml
            scale = yaml_dict.get('scale', 'unknown')
            nc = yaml_dict.get('nc', 'unknown')
            
            print(f'模型规模: {scale}')
            print(f'类别数量: {nc}')
            
            if 'kpt_shape' in yaml_dict:
                kpt_shape = yaml_dict['kpt_shape']
                print(f'关键点配置: {kpt_shape}')
                print(f'模型类型: Pose (姿态估计)')
            else:
                print(f'模型类型: Detection (目标检测)')
        
        print('✅ 模型加载成功')
        return True
        
    except Exception as e:
        print(f'❌ 模型加载失败: {e}')
        return False


def test_corner_model():
    """测试Corner模型"""
    print('\n')
    success = check_model_info(CORNER_MODEL_PATH, 'Corner')
    
    if success:
        try:
            model = YOLO(str(CORNER_MODEL_PATH))
            
            # 创建测试图像
            test_img = np.zeros((1024, 512, 3), dtype=np.uint8)
            test_img[:] = (128, 128, 128)
            
            print('\n测试推理...')
            results = model(test_img, conf=0.2, verbose=False)
            
            print(f'✅ 推理成功')
            print(f'检测到 {len(results[0].boxes)} 个对象')
            
            return True
        except Exception as e:
            print(f'❌ 推理失败: {e}')
            return False
    
    return False


def test_cfh_model():
    """测试CFH模型"""
    print('\n')
    success = check_model_info(CFH_MODEL_PATH, 'CFH')
    
    if success:
        try:
            model = YOLO(str(CFH_MODEL_PATH))
            
            # 创建测试图像
            test_img = np.zeros((1024, 512, 3), dtype=np.uint8)
            test_img[:] = (128, 128, 128)
            
            print('\n测试推理...')
            results = model(test_img, conf=0.1, verbose=False)
            
            print(f'✅ 推理成功')
            print(f'检测到 {len(results[0].boxes)} 个对象')
            
            return True
        except Exception as e:
            print(f'❌ 推理失败: {e}')
            return False
    
    return False


def test_with_real_image():
    """使用真实图像测试"""
    print('\n')
    print('=' * 80)
    print('🖼️  使用真实图像测试')
    print('=' * 80)
    
    # 查找示例图像
    example_img = Path('6-app_backend/example/1.png')
    if not example_img.exists():
        example_img = Path('example/1.png')
    
    if not example_img.exists():
        print('⚠️  未找到示例图像，跳过真实图像测试')
        return True
    
    print(f'示例图像: {example_img}')
    
    try:
        # 读取图像
        img = cv2.imread(str(example_img))
        if img is None:
            print('❌ 无法读取图像')
            return False
        
        print(f'图像尺寸: {img.shape[1]}x{img.shape[0]}')
        
        # 测试Corner模型
        print('\n测试 Corner 模型推理...')
        corner_model = YOLO(str(CORNER_MODEL_PATH))
        corner_results = corner_model(img, conf=0.2, verbose=False)
        corner_detections = len(corner_results[0].boxes)
        print(f'✅ Corner检测: {corner_detections} 个椎体')
        
        # 测试CFH模型
        print('\n测试 CFH 模型推理...')
        cfh_model = YOLO(str(CFH_MODEL_PATH))
        cfh_results = cfh_model(img, conf=0.1, verbose=False)
        cfh_detections = len(cfh_results[0].boxes)
        print(f'✅ CFH检测: {cfh_detections} 个股骨头')
        
        return True
        
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        return False


def main():
    """主函数"""
    print()
    print('🧪 模型测试')
    print()
    
    results = []
    
    # 测试Corner模型
    results.append(('Corner模型加载', test_corner_model()))
    
    # 测试CFH模型
    results.append(('CFH模型加载', test_cfh_model()))
    
    # 真实图像测试
    results.append(('真实图像测试', test_with_real_image()))
    
    # 总结
    print('\n')
    print('=' * 80)
    print('📊 测试总结')
    print('=' * 80)
    
    for test_name, result in results:
        status = '✅ 通过' if result else '❌ 失败'
        print(f'{status}: {test_name}')
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print('🎉 所有测试通过！模型已准备就绪。')
    else:
        print('⚠️  部分测试失败，请检查错误信息。')
    print()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
