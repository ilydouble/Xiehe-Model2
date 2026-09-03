# Findings & Decisions

## Requirements
- 分析 `/Volumes/E/spine_data/20260903-侧面数据第二批` 中的侧面数据。
- 数据标注格式为 LabelMe。
- 默认分析范围：规模、配对完整性、类别与 shape 分布、图像/标注几何统计、异常检查、抽样视觉质检。
- 不改动原始数据。

## Research Findings
- 待数据扫描后补充。

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

## Visual/Browser Findings
- 待抽样查看后补充。
