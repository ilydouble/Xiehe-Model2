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
- 环境未安装 Pillow；本批次图像是 PNG，因此脚本将直接读取 PNG IHDR 获取尺寸/位深/颜色类型，避免为只读统计新增依赖。视觉抽样可使用系统 `sips`/FFmpeg 生成预览。
- 现有 `datasets/LAT202511/label.txt` 只有 21 个标签，含 C7、T1-T13、L1-L5、S1、CFH，但不含抽样 JSON 已出现的 C2-C6，不能作为本批次的完整标签真值。
- `measurements.xlsx` 含两个工作表（XML 体量约 425 KB 和 455 KB），看起来是批量测量结果；先完成 LabelMe 本体统计，再决定是否用于交叉核验。
- 首轮全量扫描确认主体目录内是 706 对完整配对的 PNG/JSON，706 个 JSON 均可解析，图像头均有效，且没有完全重复的图像文件。
- 初步标签结构为 24 个椎体/骶骨标签（C2-C7、T1-T12、L1-L5、S1）加股骨头标记；股骨头存在两套互斥体系：239 例使用单点 `CFH`，459 例使用双点 `FH-1`/`FH-2`，另有 7 例没有股骨头标记。
- 首轮脚本把合法的 point/line 零面积误计为退化/微小 shape，且明细上限会让计数失真；需要修复后重跑，当前的“200 个退化/微小”不能作为数据问题结论。
- 修复后全量审计结果：706 张 PNG 与 706 个 JSON 按同目录同 stem 100% 配对；706 个 JSON 全部可解析，706 张 PNG 头均有效；没有空标注、空标签、非法点、点数不足、退化面、重复标签、尺寸不一致、`imagePath` 不一致或完全重复图像。
- 共 18,058 个 shapes：16,199 个 polygon（椎体）、1,157 个 point（股骨头相关）、702 个 line（S1）；每图 15-26 个标注，中位数 26，均值 25.58。
- 几何扫描发现 1 个明确越界标注：`FQSY2505P001483_LAT_141847.json` 的 C2 有两个顶点 y=-2.33 与 -0.30；另有 2 个需要视觉复核的上下顺序异常（T5/T6 轻微交叠、T1/T2 明显倒置）。
- 发现 5 例解剖序列内部缺层：分别缺 T12、L5（2 例）、T1-T3、L4；其中缺层可能是视野截断，也可能是漏标，需结合图像判断。
- 图像尺寸共有 292 种，384/706（54.4%）为 1536×4352；PNG 颜色格式混合：391 张 8-bit 灰度（color type 0），315 张 8-bit RGBA（color type 6）。训练前需统一解码通道数，并采用保持纵横比的 resize/letterbox。
- 文件名来源前缀共 17 类，最大三类为 LFPY 164、THBZ 141、LWSY 80；拆分训练/验证/测试时应按患者/来源分组，避免同源泄漏与站点偏差。
- 顶层第 707 个 JSON 是 `export-summary.json`，并非图像标注；主体样本目录实际为严格的 706 PNG + 706 LabelMe JSON 配对。
- 当前 Python 环境也没有 OpenCV、Matplotlib、scikit-image、CairoSVG 或 Wand；视觉质检将使用现有 FFmpeg/系统工具生成只读预览，不额外安装包。
- 官方导出汇总 `export-summary.json` 与独立扫描一致：requested=706、succeeded=706，空标注/未找到/覆盖重复/对象缺失/失败均为 0。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 使用 Python 标准库/Pillow 做全量只读扫描 | 便于复现并避免依赖 LabelMe GUI |
| 按类别与标注数量分层抽样 | 比纯随机抽样更容易覆盖少数类和异常样本 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Python 环境没有 Pillow | 对 PNG 用标准库解析 IHDR；视觉预览使用系统图像工具 |
| 初版几何规则误报 point/line 为零面积异常，且异常列表封顶影响计数 | 仅对面状 shape 检查面积，并保留完整问题计数后重跑 |
| FFmpeg 能识别 SVG 容器但没有 SVG 解码器 | 不走 SVG 栅格化，改用 FFmpeg 图像解码 + 标准库绘线 |

## Resources
- 数据目录：`/Volumes/E/spine_data/20260903-侧面数据第二批`
- 主体目录：`/Volumes/E/spine_data/20260903-侧面数据第二批/labelme-export/P202607076308`

## Visual/Browser Findings
- 待抽样查看后补充。
