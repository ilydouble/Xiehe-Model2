#!/usr/bin/env python3
"""
列出有S1和T13标注的样本
"""
import json
import sys


def list_s1_t13_samples(json_file='annotation_report.json'):
    """列出有S1和T13标注的样本"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从所有结果中筛选
    all_results = data.get('all_results', [])
    
    # 有S1标注的样本
    s1_samples = []
    # 有T13标注的样本
    t13_samples = []
    # 同时有S1和T13的样本
    both_samples = []
    
    for result in all_results:
        if not result['has_annotation']:
            continue
        
        poly_labels = set(result['poly_labels'])
        has_s1 = 'S1' in poly_labels
        has_t13 = 'T13' in poly_labels
        
        if has_s1:
            s1_samples.append(result['folder_name'])
        if has_t13:
            t13_samples.append(result['folder_name'])
        if has_s1 and has_t13:
            both_samples.append(result['folder_name'])
    
    print('=' * 80)
    print('📊 S1和T13标注统计')
    print('=' * 80)
    print()
    
    print(f'总样本数: {data["total_folders"]}')
    print(f'有标注样本: {data["total_with_annotation"]}')
    print()
    
    print(f'✅ 有S1标注的样本: {len(s1_samples)} ({len(s1_samples)/data["total_folders"]*100:.1f}%)')
    print(f'✅ 有T13标注的样本: {len(t13_samples)} ({len(t13_samples)/data["total_folders"]*100:.1f}%)')
    print(f'✅ 同时有S1和T13的样本: {len(both_samples)} ({len(both_samples)/data["total_folders"]*100:.1f}%)')
    print()
    
    print('=' * 80)
    print(f'📋 有S1标注的样本列表 (共 {len(s1_samples)} 个)')
    print('=' * 80)
    for i, name in enumerate(s1_samples, 1):
        print(f'{i:3d}. {name}')
    print()
    
    print('=' * 80)
    print(f'📋 有T13标注的样本列表 (共 {len(t13_samples)} 个)')
    print('=' * 80)
    for i, name in enumerate(t13_samples, 1):
        print(f'{i:3d}. {name}')
    print()
    
    if both_samples:
        print('=' * 80)
        print(f'📋 同时有S1和T13标注的样本列表 (共 {len(both_samples)} 个)')
        print('=' * 80)
        for i, name in enumerate(both_samples, 1):
            print(f'{i:3d}. {name}')
        print()
    
    # 保存到文件
    output = {
        's1_samples': s1_samples,
        't13_samples': t13_samples,
        'both_samples': both_samples,
        's1_count': len(s1_samples),
        't13_count': len(t13_samples),
        'both_count': len(both_samples),
    }
    
    with open('s1_t13_samples.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print('=' * 80)
    print('💾 结果已保存到: s1_t13_samples.json')
    print('=' * 80)


if __name__ == '__main__':
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'annotation_report.json'
    try:
        list_s1_t13_samples(json_file)
    except FileNotFoundError:
        print(f'错误: 找不到文件 {json_file}')
        print('请先运行 analyze_dataset.py 生成分析报告')
        sys.exit(1)
    except Exception as e:
        print(f'错误: {e}')
        sys.exit(1)

