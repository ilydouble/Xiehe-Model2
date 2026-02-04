#!/usr/bin/env python3
"""
合并椎体角点数据集和CFH关键点数据集
- 椎体角点：18个类别（C7, L1-L5, T1-T12），每个4个关键点
- CFH关键点：1个类别，1个关键点
- 合并后：19个类别，CFH的class_id为18
"""
import shutil
from pathlib import Path
from collections import defaultdict


def merge_datasets():
    """合并两个数据集"""
    corner_dataset = Path('../datasets/yolo_corner')
    cfh_dataset = Path('../datasets/yolo_keypoints')
    output_dataset = Path('../datasets/yolo_merged')
    
    print('=' * 80)
    print('🔄 合并椎体角点数据集和CFH关键点数据集')
    print('=' * 80)
    print(f'椎体角点数据集: {corner_dataset}')
    print(f'CFH关键点数据集: {cfh_dataset}')
    print(f'输出数据集: {output_dataset}')
    print()
    
    # 创建输出目录
    (output_dataset / 'images').mkdir(parents=True, exist_ok=True)
    (output_dataset / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'total_images': 0,
        'corner_only': 0,
        'cfh_only': 0,
        'both': 0,
        'vertebra_objects': 0,
        'cfh_objects': 0
    }
    
    # 获取所有图像文件
    corner_images = {f.stem: f for f in (corner_dataset / 'images').glob('*')}
    cfh_images = {f.stem: f for f in (cfh_dataset / 'images').glob('*')}
    
    all_image_stems = set(corner_images.keys()) | set(cfh_images.keys())
    
    print(f'椎体角点图像数量: {len(corner_images)}')
    print(f'CFH关键点图像数量: {len(cfh_images)}')
    print(f'总图像数量（去重）: {len(all_image_stems)}')
    print()
    
    print('开始合并...')
    
    for i, stem in enumerate(sorted(all_image_stems), 1):
        if i % 50 == 0:
            print(f'进度: {i}/{len(all_image_stems)}')
        
        # 确定使用哪个图像文件
        if stem in corner_images:
            image_file = corner_images[stem]
        else:
            image_file = cfh_images[stem]
        
        # 复制图像
        output_image = output_dataset / 'images' / image_file.name
        shutil.copy(image_file, output_image)
        stats['total_images'] += 1
        
        # 合并标注
        merged_labels = []
        
        # 读取椎体角点标注
        corner_label = corner_dataset / 'labels' / f'{stem}.txt'
        if corner_label.exists():
            with open(corner_label, 'r') as f:
                corner_lines = f.readlines()
                # 确保每行都有换行符
                for line in corner_lines:
                    if not line.endswith('\n'):
                        line = line + '\n'
                    merged_labels.append(line)
                stats['vertebra_objects'] += len(corner_lines)
        
        # 读取CFH关键点标注，并修改class_id和扩展关键点
        cfh_label = cfh_dataset / 'labels' / f'{stem}.txt'
        if cfh_label.exists():
            with open(cfh_label, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        # 将CFH的class_id从0改为18
                        parts[0] = '18'
                        # 保留bbox和第一个关键点
                        # parts[1-4]: bbox, parts[5-7]: kp1
                        # 添加3个不可见的关键点（visibility=0）
                        extended_parts = parts[:8]  # class + bbox + kp1
                        # 添加3个不可见关键点（坐标设为0，visibility设为0）
                        for _ in range(3):
                            extended_parts.extend(['0.0', '0.0', '0'])
                        merged_labels.append(' '.join(extended_parts) + '\n')
                        stats['cfh_objects'] += 1
        
        # 统计类型
        has_corner = corner_label.exists()
        has_cfh = cfh_label.exists()
        
        if has_corner and has_cfh:
            stats['both'] += 1
        elif has_corner:
            stats['corner_only'] += 1
        elif has_cfh:
            stats['cfh_only'] += 1
        
        # 保存合并后的标注
        output_label = output_dataset / 'labels' / f'{stem}.txt'
        with open(output_label, 'w') as f:
            f.writelines(merged_labels)
    
    print()
    print('=' * 80)
    print('✅ 合并完成')
    print('=' * 80)
    print(f'总图像数量: {stats["total_images"]}')
    print(f'  - 只有椎体角点: {stats["corner_only"]}')
    print(f'  - 只有CFH关键点: {stats["cfh_only"]}')
    print(f'  - 两者都有: {stats["both"]}')
    print()
    print(f'总对象数量: {stats["vertebra_objects"] + stats["cfh_objects"]}')
    print(f'  - 椎体对象: {stats["vertebra_objects"]}')
    print(f'  - CFH对象: {stats["cfh_objects"]}')
    print()
    
    # 创建data.yaml
    create_data_yaml(output_dataset, stats)
    
    print('✅ 数据集配置文件已创建: data.yaml')


def create_data_yaml(output_dataset, stats):
    """创建合并后的data.yaml配置文件"""
    yaml_content = f"""# YOLO11 Pose Dataset Configuration
# 脊柱椎体角点 + CFH关键点检测数据集

# 数据集路径
path: .  # 数据集根目录
train: images  # 训练图像目录
val: images    # 验证图像目录（暂时使用相同目录）

# 类别数量
nc: 19

# 类别名称
names:
  0: C7
  1: L1
  2: L2
  3: L3
  4: L4
  5: L5
  6: T1
  7: T2
  8: T3
  9: T4
  10: T5
  11: T6
  12: T7
  13: T8
  14: T9
  15: T10
  16: T11
  17: T12
  18: CFH

# 关键点配置
# 注意：椎体有4个关键点，CFH有1个关键点
# YOLO11需要统一的关键点数量，我们使用4个关键点
# 对于CFH，只有第1个关键点有效，其余3个设为不可见
kpt_shape: [4, 3]  # 每个对象有4个关键点，每个关键点3个值(x, y, visibility)

# 数据集信息
dataset_info:
  description: "脊柱椎体角点 + CFH关键点检测数据集"
  source: "LAT202511"
  total_images: {stats['total_images']}
  total_objects: {stats['vertebra_objects'] + stats['cfh_objects']}
  vertebra_objects: {stats['vertebra_objects']}
  cfh_objects: {stats['cfh_objects']}
  vertebra_keypoints: "每个椎体的4个关键点为距离轮廓中心最远的4个点"
  cfh_keypoints: "CFH的1个关键点（其余3个关键点标记为不可见）"
"""
    
    yaml_path = output_dataset / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


if __name__ == '__main__':
    merge_datasets()

