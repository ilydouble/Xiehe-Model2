#!/usr/bin/env python3
"""
筛选可用于训练的样本
去除：
1. 没有标注的样本（12个）
2. 分割标注不完整的样本（26个）
"""
import json
import sys
import os


def select_training_samples(analysis_json='../1-check_data/annotation_report.json'):
    """筛选可用于训练的样本"""
    
    # 读取分析结果
    with open(analysis_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print('=' * 80)
    print('📊 筛选可用于训练的样本')
    print('=' * 80)
    print()
    
    print(f'总样本数: {data["total_folders"]}')
    print(f'无标注样本: {data["total_without_annotation"]}')
    print(f'分割标注不完整样本: {data["poly_incomplete_count"]}')
    print()
    
    # 获取所有结果
    all_results = data.get('all_results', [])
    
    # 筛选条件：有标注 且 分割标注完整
    valid_samples = []
    excluded_no_annotation = []
    excluded_incomplete = []
    
    for result in all_results:
        folder_name = result['folder_name']
        
        if not result['has_annotation']:
            # 没有标注
            excluded_no_annotation.append(folder_name)
        elif not result['poly_complete']:
            # 分割标注不完整
            excluded_incomplete.append({
                'folder_name': folder_name,
                'missing_count': len(result['missing_poly']),
                'missing_labels': result['missing_poly']
            })
        else:
            # 可用于训练
            valid_samples.append(folder_name)
    
    print(f'✅ 可用于训练的样本: {len(valid_samples)} ({len(valid_samples)/data["total_folders"]*100:.1f}%)')
    print(f'❌ 排除的样本: {len(excluded_no_annotation) + len(excluded_incomplete)} ({(len(excluded_no_annotation) + len(excluded_incomplete))/data["total_folders"]*100:.1f}%)')
    print(f'   - 无标注: {len(excluded_no_annotation)}')
    print(f'   - 分割标注不完整: {len(excluded_incomplete)}')
    print()
    
    # 保存结果
    output = {
        'total_samples': data['total_folders'],
        'valid_samples_count': len(valid_samples),
        'excluded_count': len(excluded_no_annotation) + len(excluded_incomplete),
        'valid_samples': sorted(valid_samples),
        'excluded_no_annotation': sorted(excluded_no_annotation),
        'excluded_incomplete': sorted(excluded_incomplete, key=lambda x: x['missing_count'], reverse=True),
    }
    
    output_file = 'training_samples.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print('=' * 80)
    print(f'💾 结果已保存到: {output_file}')
    print('=' * 80)
    print()
    
    # 显示排除的样本
    print('=' * 80)
    print(f'❌ 排除的样本 - 无标注 (共 {len(excluded_no_annotation)} 个)')
    print('=' * 80)
    for i, name in enumerate(sorted(excluded_no_annotation), 1):
        print(f'{i:2d}. {name}')
    print()
    
    print('=' * 80)
    print(f'❌ 排除的样本 - 分割标注不完整 (共 {len(excluded_incomplete)} 个)')
    print('=' * 80)
    for i, item in enumerate(sorted(excluded_incomplete, key=lambda x: x['missing_count'], reverse=True), 1):
        missing_str = ', '.join(item['missing_labels'])
        print(f'{i:2d}. {item["folder_name"]:40s} 缺失 {item["missing_count"]:2d} 个: {missing_str}')
    print()
    
    print('=' * 80)
    print(f'✅ 可用于训练的样本 (共 {len(valid_samples)} 个)')
    print('=' * 80)
    print('样本列表已保存到 training_samples.json 中的 "valid_samples" 字段')
    print()
    
    # 显示前20个
    print('前20个样本:')
    for i, name in enumerate(sorted(valid_samples)[:20], 1):
        print(f'{i:2d}. {name}')
    if len(valid_samples) > 20:
        print(f'... 还有 {len(valid_samples) - 20} 个样本')
    print()
    
    return output


if __name__ == '__main__':
    analysis_json = sys.argv[1] if len(sys.argv) > 1 else '../1-check_data/annotation_report.json'
    
    if not os.path.exists(analysis_json):
        print(f'错误: 找不到文件 {analysis_json}')
        print('请先运行 1-check_data/analyze_dataset.py 生成分析报告')
        sys.exit(1)
    
    try:
        select_training_samples(analysis_json)
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

