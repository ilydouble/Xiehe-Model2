#!/usr/bin/env python3
"""
全面分析LAT202511数据集的标注完整性
统计图像数量、标注情况、分割标注和关键点标注的完整性

用法:
    python analyze_dataset.py --dataset datasets/LAT202511 --labels datasets/LAT202511/label.txt --output annotation_report.html
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from collections import defaultdict


def load_expected_labels(path: str) -> List[str]:
    """加载期望的标签列表"""
    with open(path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def inspect_json(json_path: str) -> Tuple[Set[str], Set[str]]:
    """
    检查JSON文件，返回polygon标签和point标签
    Returns: (polygon_labels, point_labels)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法读取 JSON {json_path}: {e}")
        return set(), set()
    
    poly_labels = set()
    point_labels = set()
    
    for shape in data.get('shapes', []):
        label = shape.get('label')
        shape_type = shape.get('shape_type')
        if not label:
            continue
        if shape_type == 'polygon':
            poly_labels.add(label)
        elif shape_type == 'point':
            point_labels.add(label)
    
    return poly_labels, point_labels


def analyze_folder(folder_path: str, expected_poly_labels: List[str], expected_point_labels: List[str]) -> Dict:
    """分析单个样本文件夹"""
    folder_name = os.path.basename(folder_path)

    # 查找所有图像文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.dcm'}
    images = []
    for ext in image_extensions:
        images.extend(list(Path(folder_path).glob(f'*{ext}')))

    # 查找所有JSON文件
    json_files = list(Path(folder_path).glob('*.json'))

    # 汇总所有JSON中的标签
    all_poly_labels = set()
    all_point_labels = set()

    for json_file in json_files:
        poly, point = inspect_json(str(json_file))
        all_poly_labels |= poly
        all_point_labels |= point

    # 计算缺失的标签
    missing_poly = [lab for lab in expected_poly_labels if lab not in all_poly_labels]
    missing_point = [lab for lab in expected_point_labels if lab not in all_point_labels]

    return {
        'folder_name': folder_name,
        'image_count': len(images),
        'json_count': len(json_files),
        'has_annotation': len(json_files) > 0,
        'poly_labels': sorted(all_poly_labels),
        'point_labels': sorted(all_point_labels),
        'missing_poly': missing_poly,
        'missing_point': missing_point,
        'poly_complete': len(missing_poly) == 0,
        'point_complete': len(missing_point) == 0,
        'fully_complete': len(missing_poly) == 0 and len(missing_point) == 0,
    }


def analyze_dataset(dataset_root: str, expected_poly_labels: List[str], expected_point_labels: List[str]) -> Dict:
    """分析整个数据集"""
    print(f"开始分析数据集: {dataset_root}")
    print(f"期望的分割标签: {expected_poly_labels}")
    print(f"期望的关键点标签: {expected_point_labels}")

    folders = [f for f in Path(dataset_root).iterdir() if f.is_dir()]
    total_folders = len(folders)

    results = []
    for i, folder in enumerate(folders, 1):
        if i % 50 == 0:
            print(f"进度: {i}/{total_folders}")
        result = analyze_folder(str(folder), expected_poly_labels, expected_point_labels)
        results.append(result)
    
    print(f"分析完成，共 {total_folders} 个样本")
    
    # 统计汇总
    total_images = sum(r['image_count'] for r in results)
    total_with_annotation = sum(1 for r in results if r['has_annotation'])
    total_without_annotation = total_folders - total_with_annotation
    
    # 分割标注不完整的样本
    poly_incomplete = [r for r in results if r['has_annotation'] and not r['poly_complete']]
    # 关键点标注不完整的样本
    point_incomplete = [r for r in results if r['has_annotation'] and not r['point_complete']]
    # 完全没有标注的样本
    no_annotation = [r for r in results if not r['has_annotation']]
    
    # 按缺失数量排序
    poly_incomplete.sort(key=lambda x: len(x['missing_poly']), reverse=True)
    point_incomplete.sort(key=lambda x: len(x['missing_point']), reverse=True)
    
    # 统计每个标签的缺失情况
    poly_missing_count = defaultdict(int)
    point_missing_count = defaultdict(int)
    
    for r in results:
        for lab in r['missing_poly']:
            poly_missing_count[lab] += 1
        for lab in r['missing_point']:
            point_missing_count[lab] += 1
    
    return {
        'total_folders': total_folders,
        'total_images': total_images,
        'total_with_annotation': total_with_annotation,
        'total_without_annotation': total_without_annotation,
        'poly_incomplete_count': len(poly_incomplete),
        'point_incomplete_count': len(point_incomplete),
        'poly_incomplete': poly_incomplete,
        'point_incomplete': point_incomplete,
        'no_annotation': no_annotation,
        'poly_missing_count': dict(poly_missing_count),
        'point_missing_count': dict(point_missing_count),
        'all_results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='分析数据集标注完整性')
    parser.add_argument('--dataset', default='datasets/LAT202511', help='数据集根目录')
    parser.add_argument('--labels', default='datasets/LAT202511/label.txt', help='标签文件路径')
    parser.add_argument('--output', default='1-check_data/annotation_report.html', help='输出HTML报告路径')
    args = parser.parse_args()

    # 加载期望的标签
    all_labels = load_expected_labels(args.labels)

    # CFH是关键点标注，S1和T13不统计到分割标注完整性，其他是分割标注
    expected_poly_labels = [label for label in all_labels if label not in ['CFH', 'S1', 'T13']]
    expected_point_labels = ['CFH']

    # 分析数据集
    analysis = analyze_dataset(args.dataset, expected_poly_labels, expected_point_labels)

    # 保存中间结果为JSON
    json_output = args.output.replace('.html', '.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存到: {json_output}")

    # 生成HTML报告
    from generate_html_report import generate_html_report
    generate_html_report(analysis, expected_poly_labels, expected_point_labels, args.output)
    print(f"HTML报告已生成: {args.output}")


if __name__ == '__main__':
    main()

