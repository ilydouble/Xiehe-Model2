# Task Plan: 侧面 LabelMe 数据集分析

## Goal
对 `/Volumes/E/spine_data/20260903-侧面数据第二批` 做只读的数据完整性、标注分布与质量分析，并向用户交付可核查的结论。

## Current Phase
Phase 2

## Phases

### Phase 1: 数据发现与口径确认
- [x] 确认目录可访问及文件组织方式
- [x] 识别图片、LabelMe JSON 与其他文件
- [x] 记录数据分析口径
- **Status:** complete

### Phase 2: 完整性与统计分析
- [ ] 检查图像/JSON 配对及 JSON 可解析性
- [ ] 统计类别、shape 类型、图像尺寸和标注数量
- [ ] 检查坐标越界、退化、多边形/矩形异常等问题
- **Status:** in_progress

### Phase 3: 抽样视觉质检
- [ ] 生成或查看代表性标注样本
- [ ] 记录可疑标注与典型模式
- **Status:** pending

### Phase 4: 复核与交付
- [ ] 复核统计结果
- [ ] 输出问题清单和后续建议
- **Status:** pending

## Key Questions
1. 数据集规模、类别、标注类型和图像尺寸分布是什么？
2. 是否存在缺失配对、损坏文件、坐标越界、空标注、重复数据或其他质量风险？
3. 从抽样图像看，侧面标注的一致性和可训练性如何？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 原始数据只读 | 避免分析过程改变数据集 |
| 先全量自动统计，再做分层抽样视觉检查 | 同时覆盖规模性问题与肉眼可见的标注质量问题 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `.git/index.lock` 无法创建（沙箱只读） | 1 | 使用受控权限执行仅针对本任务文件的提交 |
| `ModuleNotFoundError: PIL` | 1 | 不安装依赖，改用标准库解析 PNG 元数据 |
| 初版审计将合法 point/line 当成退化和微小 shape | 1 | 限定面状 shape 的面积规则并取消影响计数的明细截断 |
| FFmpeg 无 SVG decoder | 1 | 改用 PPM 中间图与标准库绘制标注预览 |

## Notes
- 不纳入或覆盖工作区已有的无关修改。
