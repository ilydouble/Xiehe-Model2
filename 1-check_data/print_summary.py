#!/usr/bin/env python3
"""
打印标注完整性分析摘要
"""
import json
import sys


def print_summary(json_file='annotation_report.json'):
    """打印分析摘要"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print('=' * 80)
    print('📊 数据集标注完整性分析摘要')
    print('=' * 80)
    print()
    
    # 基本统计
    print('📁 基本统计:')
    print(f'  • 总样本数: {data["total_folders"]}')
    print(f'  • 总图像数: {data["total_images"]}')
    print(f'  • 平均每样本图像数: {data["total_images"] / data["total_folders"]:.2f}')
    print()
    
    # 标注状态
    print('✅ 标注状态:')
    print(f'  • 有标注样本: {data["total_with_annotation"]} ({data["total_with_annotation"]/data["total_folders"]*100:.1f}%)')
    print(f'  • 无标注样本: {data["total_without_annotation"]} ({data["total_without_annotation"]/data["total_folders"]*100:.1f}%)')
    print()
    
    # 完整性统计
    print('📋 完整性统计:')
    print(f'  • 分割标注不完整: {data["poly_incomplete_count"]} ({data["poly_incomplete_count"]/data["total_folders"]*100:.1f}%)')
    print(f'  • 关键点标注不完整: {data["point_incomplete_count"]} ({data["point_incomplete_count"]/data["total_folders"]*100:.1f}%)')
    print()
    
    # 无标注样本
    print(f'⚠️  无标注样本列表 (共 {len(data["no_annotation"])} 个):')
    if data['no_annotation']:
        for i, item in enumerate(data['no_annotation'][:20], 1):
            print(f'  {i:2d}. {item["folder_name"]} (图像数: {item["image_count"]})')
        if len(data['no_annotation']) > 20:
            print(f'  ... 还有 {len(data["no_annotation"]) - 20} 个样本')
    else:
        print('  ✓ 所有样本都有标注')
    print()
    
    # 分割标注缺失统计
    print('🔴 分割标注缺失统计 (按标签):')
    poly_missing = sorted(data['poly_missing_count'].items(), key=lambda x: x[1], reverse=True)
    for label, count in poly_missing:
        percentage = count / data['total_folders'] * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f'  {label:4s}: {bar} {count:3d} ({percentage:5.1f}%)')
    print()
    
    # 关键点标注缺失统计
    print('🔵 关键点标注缺失统计 (按标签):')
    point_missing = sorted(data['point_missing_count'].items(), key=lambda x: x[1], reverse=True)
    for label, count in point_missing:
        percentage = count / data['total_folders'] * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f'  {label:4s}: {bar} {count:3d} ({percentage:5.1f}%)')
    print()
    
    # 最需要关注的样本
    print('🎯 最需要关注的样本 (分割标注缺失最多的前10个):')
    for i, item in enumerate(data['poly_incomplete'][:10], 1):
        missing_count = len(item['missing_poly'])
        print(f'  {i:2d}. {item["folder_name"]:40s} 缺失 {missing_count:2d} 个分割标签')
    print()
    
    print('🎯 最需要关注的样本 (关键点标注缺失最多的前10个):')
    for i, item in enumerate(data['point_incomplete'][:10], 1):
        missing_count = len(item['missing_point'])
        print(f'  {i:2d}. {item["folder_name"]:40s} 缺失 {missing_count:2d} 个关键点标签')
    print()
    
    print('=' * 80)
    print(f'📄 详细报告请查看: annotation_report.html')
    print('=' * 80)


if __name__ == '__main__':
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'annotation_report.json'
    try:
        print_summary(json_file)
    except FileNotFoundError:
        print(f'错误: 找不到文件 {json_file}')
        print('请先运行 analyze_dataset.py 生成分析报告')
        sys.exit(1)
    except Exception as e:
        print(f'错误: {e}')
        sys.exit(1)

