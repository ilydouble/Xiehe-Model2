#!/usr/bin/env python3
"""
生成标注完整性HTML报告
"""
import json
from datetime import datetime
from typing import Dict, List


def html_escape(s: str) -> str:
    """HTML转义"""
    return (s.replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def generate_summary_section(analysis: Dict) -> str:
    """生成概览部分"""
    return f"""
  <div class="summary-grid">
    <div class="summary-card">
      <div class="summary-number">{analysis['total_folders']}</div>
      <div class="summary-label">总样本数</div>
    </div>
    <div class="summary-card">
      <div class="summary-number">{analysis['total_images']}</div>
      <div class="summary-label">总图像数</div>
    </div>
    <div class="summary-card success">
      <div class="summary-number">{analysis['total_with_annotation']}</div>
      <div class="summary-label">有标注样本</div>
    </div>
    <div class="summary-card error">
      <div class="summary-number">{analysis['total_without_annotation']}</div>
      <div class="summary-label">无标注样本</div>
    </div>
    <div class="summary-card warning">
      <div class="summary-number">{analysis['poly_incomplete_count']}</div>
      <div class="summary-label">分割标注不完整</div>
    </div>
    <div class="summary-card warning">
      <div class="summary-number">{analysis['point_incomplete_count']}</div>
      <div class="summary-label">关键点标注不完整</div>
    </div>
  </div>
"""


def generate_table_rows(items: List[Dict], max_rows: int = 50, show_missing: bool = True) -> str:
    """生成表格行"""
    rows = []
    for i, item in enumerate(items[:max_rows], 1):
        folder = html_escape(item['folder_name'])
        img_count = item['image_count']
        json_count = item['json_count']
        
        if show_missing:
            missing_poly = len(item['missing_poly'])
            missing_point = len(item['missing_point'])
            missing_poly_labels = ', '.join(item['missing_poly'][:10])
            if len(item['missing_poly']) > 10:
                missing_poly_labels += '...'
            missing_point_labels = ', '.join(item['missing_point'][:10])
            if len(item['missing_point']) > 10:
                missing_point_labels += '...'
            
            rows.append(f"""
      <tr>
        <td>{i}</td>
        <td class="folder-name">{folder}</td>
        <td>{img_count}</td>
        <td>{json_count}</td>
        <td class="{'error' if missing_poly > 0 else 'success'}">{missing_poly}</td>
        <td class="{'error' if missing_point > 0 else 'success'}">{missing_point}</td>
        <td class="missing-labels">{html_escape(missing_poly_labels)}</td>
        <td class="missing-labels">{html_escape(missing_point_labels)}</td>
      </tr>
""")
        else:
            rows.append(f"""
      <tr>
        <td>{i}</td>
        <td class="folder-name">{folder}</td>
        <td>{img_count}</td>
        <td>{json_count}</td>
      </tr>
""")
    
    return ''.join(rows)


def generate_chart_data(analysis: Dict, expected_poly_labels: List[str], expected_point_labels: List[str]) -> str:
    """生成图表数据"""
    poly_counts = [analysis['poly_missing_count'].get(lab, 0) for lab in expected_poly_labels]
    point_counts = [analysis['point_missing_count'].get(lab, 0) for lab in expected_point_labels]

    return f"""
    const polyLabels = {json.dumps(expected_poly_labels, ensure_ascii=False)};
    const pointLabels = {json.dumps(expected_point_labels, ensure_ascii=False)};
    const polyMissingCounts = {json.dumps(poly_counts)};
    const pointMissingCounts = {json.dumps(point_counts)};

    const totalFolders = {analysis['total_folders']};
    const withAnnotation = {analysis['total_with_annotation']};
    const withoutAnnotation = {analysis['total_without_annotation']};
    const polyIncomplete = {analysis['poly_incomplete_count']};
    const pointIncomplete = {analysis['point_incomplete_count']};
"""


def generate_html_report(analysis: Dict, expected_poly_labels: List[str], expected_point_labels: List[str], output_path: str):
    """生成完整的HTML报告"""

    # 生成各部分内容
    summary_html = generate_summary_section(analysis)
    no_annotation_rows = generate_table_rows(analysis['no_annotation'], max_rows=50, show_missing=False)
    poly_incomplete_rows = generate_table_rows(analysis['poly_incomplete'], max_rows=50)
    point_incomplete_rows = generate_table_rows(analysis['point_incomplete'], max_rows=50)
    chart_data = generate_chart_data(analysis, expected_poly_labels, expected_point_labels)
    
    # 生成时间
    gen_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 完整的HTML内容
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>标注完整性分析报告</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: #f5f7fa;
      padding: 20px;
      color: #2c3e50;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{
      font-size: 32px;
      margin-bottom: 10px;
      color: #1a202c;
    }}
    .subtitle {{
      color: #718096;
      margin-bottom: 30px;
      font-size: 14px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }}
    .summary-card {{
      background: white;
      padding: 24px;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      text-align: center;
      border-left: 4px solid #3b82f6;
    }}
    .summary-card.success {{ border-left-color: #10b981; }}
    .summary-card.error {{ border-left-color: #ef4444; }}
    .summary-card.warning {{ border-left-color: #f59e0b; }}
    .summary-number {{
      font-size: 36px;
      font-weight: bold;
      margin-bottom: 8px;
      color: #1a202c;
    }}
    .summary-label {{
      font-size: 14px;
      color: #718096;
      font-weight: 500;
    }}
    .section {{
      background: white;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      margin-bottom: 30px;
    }}
    h2 {{
      font-size: 24px;
      margin-bottom: 20px;
      color: #1a202c;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 10px;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
      gap: 30px;
      margin-bottom: 30px;
    }}
    .chart-container {{
      background: white;
      padding: 20px;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .chart-container h3 {{
      font-size: 18px;
      margin-bottom: 15px;
      color: #1a202c;
    }}
    canvas {{
      max-height: 400px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
    }}
    th {{
      background: #f7fafc;
      font-weight: 600;
      color: #4a5568;
      position: sticky;
      top: 0;
    }}
    tr:hover {{
      background: #f7fafc;
    }}
    .folder-name {{
      font-family: 'Courier New', monospace;
      font-size: 13px;
      color: #2d3748;
    }}
    .missing-labels {{
      font-size: 12px;
      color: #718096;
      max-width: 300px;
    }}
    .success {{
      color: #10b981;
      font-weight: 600;
    }}
    .error {{
      color: #ef4444;
      font-weight: 600;
    }}
    .warning {{
      color: #f59e0b;
      font-weight: 600;
    }}
    .table-wrapper {{
      overflow-x: auto;
      max-height: 600px;
      overflow-y: auto;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 标注完整性分析报告</h1>
    <p class="subtitle">生成时间: {gen_time}</p>

    <h2>🔍 数据概览</h2>
    {summary_html}

    <div class="charts-grid">
      <div class="chart-container">
        <h3>📈 标注状态分布</h3>
        <canvas id="statusChart"></canvas>
      </div>
      <div class="chart-container">
        <h3>📊 完整性统计</h3>
        <canvas id="completenessChart"></canvas>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-container">
        <h3>🔴 分割标注缺失统计（按标签）</h3>
        <canvas id="polyMissingChart"></canvas>
      </div>
      <div class="chart-container">
        <h3>🔵 关键点标注缺失统计（按标签）</h3>
        <canvas id="pointMissingChart"></canvas>
      </div>
    </div>

    <div class="section">
      <h2>⚠️ 无标注样本列表 (共 {len(analysis['no_annotation'])} 个，显示前50)</h2>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>样本名称</th>
              <th>图像数</th>
              <th>JSON数</th>
            </tr>
          </thead>
          <tbody>
            {no_annotation_rows}
          </tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <h2>🟡 分割标注不完整样本 (共 {len(analysis['poly_incomplete'])} 个，显示前50)</h2>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>样本名称</th>
              <th>图像数</th>
              <th>JSON数</th>
              <th>缺失分割数</th>
              <th>缺失关键点数</th>
              <th>缺失分割标签</th>
              <th>缺失关键点标签</th>
            </tr>
          </thead>
          <tbody>
            {poly_incomplete_rows}
          </tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <h2>🔵 关键点标注不完整样本 (共 {len(analysis['point_incomplete'])} 个，显示前50)</h2>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>样本名称</th>
              <th>图像数</th>
              <th>JSON数</th>
              <th>缺失分割数</th>
              <th>缺失关键点数</th>
              <th>缺失分割标签</th>
              <th>缺失关键点标签</th>
            </tr>
          </thead>
          <tbody>
            {point_incomplete_rows}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    {chart_data}

    // 标注状态分布饼图
    new Chart(document.getElementById('statusChart'), {{
      type: 'doughnut',
      data: {{
        labels: ['有标注', '无标注'],
        datasets: [{{
          data: [withAnnotation, withoutAnnotation],
          backgroundColor: ['#10b981', '#ef4444'],
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
          legend: {{ position: 'bottom' }}
        }}
      }}
    }});

    // 完整性统计柱状图
    new Chart(document.getElementById('completenessChart'), {{
      type: 'bar',
      data: {{
        labels: ['分割标注不完整', '关键点标注不完整'],
        datasets: [{{
          label: '样本数量',
          data: [polyIncomplete, pointIncomplete],
          backgroundColor: ['#f59e0b', '#3b82f6'],
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
          legend: {{ display: false }}
        }},
        scales: {{
          y: {{ beginAtZero: true }}
        }}
      }}
    }});

    // 分割标注缺失统计
    new Chart(document.getElementById('polyMissingChart'), {{
      type: 'bar',
      data: {{
        labels: polyLabels,
        datasets: [{{
          label: '缺失样本数',
          data: polyMissingCounts,
          backgroundColor: '#ef4444',
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
          legend: {{ display: false }}
        }},
        scales: {{
          y: {{ beginAtZero: true }},
          x: {{ ticks: {{ maxRotation: 90, minRotation: 45 }} }}
        }}
      }}
    }});

    // 关键点标注缺失统计
    new Chart(document.getElementById('pointMissingChart'), {{
      type: 'bar',
      data: {{
        labels: pointLabels,
        datasets: [{{
          label: '缺失样本数',
          data: pointMissingCounts,
          backgroundColor: '#3b82f6',
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
          legend: {{ display: false }}
        }},
        scales: {{
          y: {{ beginAtZero: true }},
          x: {{ ticks: {{ maxRotation: 90, minRotation: 45 }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""

    # 保存HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML报告已生成: {output_path}")


if __name__ == '__main__':
    # 这个文件主要被 analyze_dataset.py 调用
    pass

