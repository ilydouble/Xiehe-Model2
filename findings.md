# Findings & Decisions

## Requirements
- 分析 `/Volumes/E/spine_data/20260903-侧面数据第二批` 中的侧面数据。
- 数据标注格式为 LabelMe。
- 默认分析范围：规模、配对完整性、类别与 shape 分布、图像/标注几何统计、异常检查、抽样视觉质检。
- 不改动原始数据。

## Research Findings
- 数据目录可访问，主体位于 `labelme-export/P202607076308/`。
- 排除 macOS `._*` AppleDouble 辅助文件后，共有 707 个 JSON、706 个 PNG、1 个 XLSX，合计约 2.206 GB。
- 同时存在 1,415 个 `._*` 辅助文件；它们不应进入训练集或标注解析流程。
- 首个抽样 JSON 是 LabelMe `2025.7.4.0` 格式，`shapes` 为椎体类别（如 C2-C7）的 polygon 标注；需要继续全量核查标签集合与几何合法性。
- JSON 比 PNG 多 1 个，可能由目录中的 XLSX 对应同名 JSON、孤立 JSON 或其他命名情况导致，需按 stem 精确核查。
- 仓库已有的 `1-check_data/analyze_dataset.py` 面向“每个样本一个子目录”的旧数据组织，并依赖预设 `label.txt`；本批次主体是单层平铺的 PNG/JSON 配对，不能直接套用旧脚本。
- 本次将新增一个通用的平铺 LabelMe 扫描脚本，输出机器可读报告，重点覆盖配对、JSON 字段、图像尺寸、标签/shape 分布、边界、退化几何、重复文件及命名风险。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 使用 Python 标准库/Pillow 做全量只读扫描 | 便于复现并避免依赖 LabelMe GUI |
| 按类别与标注数量分层抽样 | 比纯随机抽样更容易覆盖少数类和异常样本 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
- 数据目录：`/Volumes/E/spine_data/20260903-侧面数据第二批`
- 主体目录：`/Volumes/E/spine_data/20260903-侧面数据第二批/labelme-export/P202607076308`

## Visual/Browser Findings
- 待抽样查看后补充。
