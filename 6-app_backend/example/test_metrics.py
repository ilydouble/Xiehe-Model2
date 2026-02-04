#!/usr/bin/env python3
"""
测试脚本 - 调用API生成关键点JSON和指标JSON

使用方式：
python test_metrics.py --image <图像路径> --api-url http://localhost:8000
"""
import argparse
import json
import requests
from pathlib import Path


def main(image_path: str, api_url: str, output_dir: str = "."):
    """
    测试主函数

    Args:
        image_path: 图像路径
        api_url: API服务地址
        output_dir: 输出目录
    """
    print("=" * 80)
    print("🧪 侧面脊柱分析 - 调用API生成JSON")
    print("=" * 80)
    print(f"图像: {image_path}")
    print(f"API: {api_url}")
    print()

    # 检查图像
    if not Path(image_path).exists():
        print(f"❌ 图像不存在: {image_path}")
        return

    # 准备输出路径
    image_name = Path(image_path).stem
    keypoints_output = Path(output_dir) / f"{image_name}_keypoints.json"
    metrics_output = Path(output_dir) / f"{image_name}_metrics.json"

    # 步骤1: 先调用检测接口查看检测结果
    print("【步骤1】调用检测接口...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{api_url}/api/detect",
                files=files,
                timeout=60
            )

        if response.status_code != 200:
            print(f"❌ 检测失败: {response.status_code}")
            print(response.text)
            return

        detection_result = response.json()

        # 打印检测结果
        print(f"✅ 检测完成")
        print(f"   图像尺寸: {detection_result['image_width']} x {detection_result['image_height']}")
        print(f"   检测到 {len(detection_result['vertebrae'])} 个椎体:")
        vertebrae_labels = sorted([v['label'] for v in detection_result['vertebrae']])
        print(f"   {', '.join(vertebrae_labels)}")

        if detection_result.get('cfh'):
            print(f"   CFH: 是 (置信度={detection_result['cfh']['confidence']:.3f})")
        else:
            print(f"   CFH: 否")

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        print("   请确保服务已启动: python ../app.py")
        return
    except Exception as e:
        print(f"❌ 错误: {e}")
        return

    # 步骤2: 调用API获取关键点JSON
    print("\n【步骤2】调用API获取关键点...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{api_url}/api/detect_and_keypoints",
                files=files,
                timeout=60
            )

        if response.status_code != 200:
            print(f"❌ API调用失败: {response.status_code}")
            print(response.text)
            return

        keypoints_json = response.json()

        # 保存关键点JSON
        with open(keypoints_output, 'w', encoding='utf-8') as f:
            json.dump(keypoints_json, f, indent=2, ensure_ascii=False)

        print(f"✅ 关键点JSON已保存: {keypoints_output}")
        print(f"   生成了 {len(keypoints_json['measurements'])} 个指标的测量点:")
        for i, m in enumerate(keypoints_json['measurements'], 1):
            print(f"   {i:2d}. {m['type']:30s} - {len(m['points'])} 个点")

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        print("   请确保服务已启动: python ../app.py")
        return
    except Exception as e:
        print(f"❌ 错误: {e}")
        return

    # 步骤3: 计算指标JSON
    print("\n【步骤3】计算指标...")
    try:
        response = requests.post(
            f"{api_url}/api/calculate_metrics",
            json=keypoints_json,
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ 指标计算失败: {response.status_code}")
            print(response.text)
            return

        metrics_json = response.json()

        # 保存指标JSON
        with open(metrics_output, 'w', encoding='utf-8') as f:
            json.dump(metrics_json, f, indent=2, ensure_ascii=False)

        print(f"✅ 指标JSON已保存: {metrics_output}")
        print(f"   成功计算了 {len(metrics_json.get('metrics', {}))} 个指标")

        # 显示所有指标（包括未计算的）
        all_metrics = [
            ("T1_Slope", "T1倾斜角"),
            ("Cervical_Lordosis", "颈椎前凸角"),
            ("Thoracic_Kyphosis_T2_T5", "上胸椎后凸角 T2-T5"),
            ("Thoracic_Kyphosis_T5_T12", "主胸椎后凸角 T5-T12"),
            ("Lumbar_Lordosis", "腰椎前凸角"),
            ("SVA", "矢状面垂直轴"),
            ("TPA", "T1骨盆角"),
            ("PI", "骨盆入射角"),
            ("PT", "骨盆倾斜角"),
            ("SS", "骶骨倾斜角"),
        ]

        print("\n   指标详情:")
        calculated_metrics = metrics_json.get('metrics', {})
        for i, (key, name) in enumerate(all_metrics, 1):
            if key in calculated_metrics:
                value = calculated_metrics[key]
                print(f"   {i:2d}. ✅ {name:25s} ({key:30s}): {value:7.2f}°")
            else:
                print(f"   {i:2d}. ❌ {name:25s} ({key:30s}): N/A (缺少必要的椎体)")

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return
    except Exception as e:
        print(f"❌ 错误: {e}")
        return

    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print(f"输出文件:")
    print(f"  1. 关键点JSON: {keypoints_output}")
    print(f"  2. 指标JSON: {metrics_output}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='侧面脊柱分析 - 调用API生成JSON')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000',
                       help='API服务地址（默认: http://localhost:8000）')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='输出目录（默认: 当前目录）')

    args = parser.parse_args()
    main(args.image, args.api_url, args.output_dir)

