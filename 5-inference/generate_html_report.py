#!/usr/bin/env python3
"""
生成包含可视化结果的HTML报告
"""
import base64
import json
from pathlib import Path


def image_to_base64(image_path):
    """将图像转换为base64编码"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_html_report(
    grid_dir='../datasets/prediction_results',
    stats_file='../datasets/prediction_results/prediction_stats.json',
    output_file='model_analysis_report.html',
    max_images=4
):
    """生成HTML报告"""
    
    print('=' * 80)
    print('📊 生成HTML模型分析报告')
    print('=' * 80)
    
    # 读取统计信息
    stats = {}
    stats_path = Path(stats_file)
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f'✅ 读取统计信息: {stats_file}')
    else:
        print(f'⚠️  统计文件不存在: {stats_file}')
    
    # 获取网格图
    grid_path = Path(grid_dir)
    grid_files = sorted(list(grid_path.glob('grid_*.png')))[:max_images]
    
    print(f'✅ 找到 {len(grid_files)} 张网格图')
    
    # 转换图像为base64
    images_base64 = []
    for i, grid_file in enumerate(grid_files):
        print(f'  处理图像 {i+1}/{len(grid_files)}: {grid_file.name}')
        img_b64 = image_to_base64(grid_file)
        images_base64.append({
            'name': grid_file.stem.replace('grid_', ''),
            'data': img_b64
        })
    
    print(f'✅ 图像转换完成')
    print()
    
    # 生成HTML
    html_content = generate_html_content(stats, images_base64)
    
    # 保存HTML
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f'✅ HTML报告已生成: {output_path}')
    print(f'📊 文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB')
    print()
    print(f'查看报告: open {output_path}')
    
    return output_path


def generate_html_content(stats, images_base64):
    """生成HTML内容"""
    
    # 计算统计数据
    total_images = stats.get('total_images', 0)
    corner_detections = stats.get('corner_detections', 0)
    corner_keypoints = stats.get('corner_keypoints', 0)
    cfh_detections = stats.get('cfh_detections', 0)
    
    avg_vertebrae_per_image = corner_detections / total_images if total_images > 0 else 0
    avg_keypoints_per_vertebra = corner_keypoints / corner_detections if corner_detections > 0 else 0
    
    # 生成图像HTML
    images_html = ''
    for i, img in enumerate(images_base64):
        images_html += f'''
        <div class="result-image">
            <h4>样本 {i+1}: {img['name'][:50]}...</h4>
            <img src="data:image/png;base64,{img['data']}" alt="预测结果 {i+1}">
            <p class="image-caption">
                左：Corner模型（椎体+关键点） | 中：CFH模型（股骨头） | 右：组合结果
            </p>
        </div>
        '''
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>脊柱AI模型分析报告 - Corner vs CFH Detection</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .header .meta {{
            margin-top: 20px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
        }}
        
        .section-title::before {{
            content: '📊';
            margin-right: 10px;
            font-size: 1.2em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }}
        
        .stat-card .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-card .stat-desc {{
            font-size: 0.85em;
            color: #999;
            margin-top: 5px;
        }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            margin-top: 20px;
        }}

        .model-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .model-card.corner {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        }}

        .model-card.cfh {{
            background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        }}

        .model-card h3 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
        }}

        .model-card ul {{
            list-style: none;
        }}

        .model-card ul li {{
            padding: 10px;
            margin-bottom: 8px;
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
        }}

        .model-card ul li::before {{
            content: '✅';
            margin-right: 10px;
        }}

        .results-section {{
            margin-top: 40px;
        }}

        .result-image {{
            margin-bottom: 40px;
            background: #f5f7fa;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .result-image h4 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}

        .result-image img {{
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }}

        .image-caption {{
            text-align: center;
            color: #666;
            margin-top: 10px;
            font-size: 0.9em;
            font-style: italic;
        }}

        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .comparison-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        .comparison-table td {{
            padding: 15px;
            border-bottom: 1px solid #eee;
        }}

        .comparison-table tr:hover {{
            background: #f5f7fa;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}

        .badge.pose {{
            background: #fcb69f;
            color: #8b4513;
        }}

        .badge.detection {{
            background: #a1c4fd;
            color: #1e3a8a;
        }}

        .highlight-box {{
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            border-left: 5px solid #ff6b6b;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}

        .highlight-box h4 {{
            color: #8b4513;
            margin-bottom: 10px;
        }}

        .footer {{
            background: #f5f7fa;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 3px solid #667eea;
        }}

        @media (max-width: 768px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}

            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 脊柱AI模型分析报告</h1>
            <div class="subtitle">Corner关键点检测 vs CFH目标检测</div>
            <div class="meta">
                📅 生成日期: 2026-01-07 | 📊 数据集: LAT202511 | 🤖 框架: YOLO11
            </div>
        </div>

        <div class="content">
            <!-- 执行摘要 -->
            <section class="section">
                <h2 class="section-title">执行摘要</h2>
                <div class="highlight-box">
                    <h4>🎯 项目目标</h4>
                    <p>开发两个专门的YOLO11模型用于脊柱X光图像分析：</p>
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        <li><strong>Corner模型</strong>: 检测18种椎体并定位其4个角点（关键点检测）</li>
                        <li><strong>CFH模型</strong>: 检测股骨头位置（目标检测）</li>
                    </ul>
                </div>
            </section>

            <!-- 预测统计 -->
            <section class="section">
                <h2 class="section-title">预测结果统计</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">测试图像</div>
                        <div class="stat-value">{total_images}</div>
                        <div class="stat-desc">张</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">检测椎体</div>
                        <div class="stat-value">{corner_detections}</div>
                        <div class="stat-desc">个</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">关键点总数</div>
                        <div class="stat-value">{corner_keypoints}</div>
                        <div class="stat-desc">个</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">检测股骨头</div>
                        <div class="stat-value">{cfh_detections}</div>
                        <div class="stat-desc">个</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">平均椎体/图像</div>
                        <div class="stat-value">{avg_vertebrae_per_image:.1f}</div>
                        <div class="stat-desc">个/张</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">关键点/椎体</div>
                        <div class="stat-value">{avg_keypoints_per_vertebra:.1f}</div>
                        <div class="stat-desc">个/椎体</div>
                    </div>
                </div>
            </section>

            <!-- 模型对比 -->
            <section class="section">
                <h2 class="section-title">模型对比</h2>
                <div class="comparison-grid">
                    <div class="model-card corner">
                        <h3>Corner模型<span class="badge pose">Pose Detection</span></h3>
                        <ul>
                            <li>检测18种椎体（C7, T1-T12, L1-L5）</li>
                            <li>每个椎体定位4个角点</li>
                            <li>支持椎体形态分析</li>
                            <li>可计算椎体角度和形变</li>
                            <li>训练图像: 368张</li>
                            <li>标注对象: 6,610个椎体</li>
                        </ul>
                    </div>

                    <div class="model-card cfh">
                        <h3>CFH模型<span class="badge detection">Object Detection</span></h3>
                        <ul>
                            <li>检测股骨头（CFH）位置</li>
                            <li>单类别目标检测</li>
                            <li>从关键点扩展为检测框</li>
                            <li>包围盒大小基于椎体1.8倍</li>
                            <li>训练图像: 370张</li>
                            <li>标注对象: 370个CFH</li>
                        </ul>
                    </div>
                </div>
            </section>

            <!-- 详细对比表 -->
            <section class="section">
                <h2 class="section-title">技术对比</h2>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>对比维度</th>
                            <th>Corner模型</th>
                            <th>CFH模型</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>任务类型</strong></td>
                            <td>关键点检测（Pose Detection）</td>
                            <td>目标检测（Object Detection）</td>
                        </tr>
                        <tr>
                            <td><strong>YOLO模式</strong></td>
                            <td>yolo pose</td>
                            <td>yolo detect</td>
                        </tr>
                        <tr>
                            <td><strong>检测目标</strong></td>
                            <td>18种椎体</td>
                            <td>1种目标（CFH）</td>
                        </tr>
                        <tr>
                            <td><strong>输出格式</strong></td>
                            <td>边界框 + 4个关键点</td>
                            <td>边界框</td>
                        </tr>
                        <tr>
                            <td><strong>训练难度</strong></td>
                            <td>较高（多类别+关键点）</td>
                            <td>较低（单类别检测）</td>
                        </tr>
                        <tr>
                            <td><strong>推荐模型</strong></td>
                            <td>YOLO11m-pose / YOLO11l-pose</td>
                            <td>YOLO11m / YOLO11s</td>
                        </tr>
                        <tr>
                            <td><strong>预期mAP50</strong></td>
                            <td>0.85-0.92</td>
                            <td>0.90-0.95</td>
                        </tr>
                    </tbody>
                </table>
            </section>

            <!-- 可视化结果 -->
            <section class="section results-section">
                <h2 class="section-title">预测结果可视化</h2>
                <p style="margin-bottom: 20px; color: #666;">
                    以下展示了{len(images_base64)}个测试样本的预测结果。每行包含三张图像：
                    左侧为Corner模型结果（椎体+关键点），中间为CFH模型结果（股骨头），右侧为组合结果。
                </p>
                {images_html}
            </section>
        </div>

        <div class="footer">
            <p><strong>脊柱AI模型分析报告</strong></p>
            <p style="margin-top: 10px;">生成时间: 2026-01-07 | 框架: YOLO11 | 数据集: LAT202511</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Corner模型: {corner_detections}个椎体检测 | CFH模型: {cfh_detections}个股骨头检测
            </p>
        </div>
    </div>
</body>
</html>'''

    return html


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='生成HTML模型分析报告')
    parser.add_argument('--grid-dir', type=str,
                       default='../datasets/prediction_results',
                       help='网格图目录')
    parser.add_argument('--stats-file', type=str,
                       default='../datasets/prediction_results/prediction_stats.json',
                       help='统计文件路径')
    parser.add_argument('--output', type=str,
                       default='model_analysis_report.html',
                       help='输出HTML文件')
    parser.add_argument('--max-images', type=int, default=4,
                       help='最多包含的图像数量')

    args = parser.parse_args()

    generate_html_report(
        grid_dir=args.grid_dir,
        stats_file=args.stats_file,
        output_file=args.output,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()


