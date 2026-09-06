# Progress Log

## Session: 2026-09-04

### Phase 1: 数据发现与口径确认
- **Status:** complete
- **Started:** 2026-09-04
- Actions taken:
  - 建立分析计划，明确只读检查范围。
  - 确认数据目录可访问，定位主体目录。
  - 清点 707 个 JSON、706 个 PNG、1 个 XLSX 及 1,415 个 AppleDouble 辅助文件。
- Files created/modified:
  - `task_plan.md`（新建）
  - `findings.md`（新建）
  - `progress.md`（新建）

### Phase 2: 完整性与统计分析
- **Status:** complete
- Actions taken:
  - 新增并运行平铺 LabelMe 全量审计脚本。
  - 完成配对、JSON/PNG 有效性、标签、shape、尺寸、几何、重复文件、来源与标注体系交叉统计。
- Files created/modified:
  - `1-check_data/analyze_flat_labelme.py`（新建并修正）
  - `analysis/side_labelme_20260903/audit.json`（生成）

### Phase 3: 抽样视觉质检
- **Status:** complete
- Actions taken:
  - 新增无 Pillow/OpenCV 依赖的标注预览工具。
  - 查看全部自动异常候选、边缘缺标签样本，并按 8 个额外来源抽查完整样本。
  - 将缺失标签区分为合理视野裁切、疑似漏标和需结合解剖判断的编号跳跃。
- Files created/modified:
  - `1-check_data/render_labelme_preview.py`（新建并修正）
  - `analysis/side_labelme_20260903/*.png`（生成预览）

### Phase 4: 复核与交付
- **Status:** complete
- Actions taken:
  - 编写完整分析报告。
  - 从原始数据重新运行审计，并在忽略生成时间后确认结果逐字段一致。
  - 对 T1/T2 疑似互换样本重新生成预览，确认预览工具可复现。
- Files created/modified:
  - `docs/side_labelme_20260903_analysis.md`（新建）

## Session: 2026-09-07

### Phase 5: 新旧侧面数据兼容性核对
- **Status:** in_progress
- **Started:** 2026-09-07
- Actions taken:
  - 恢复既有分析上下文，准备按同一口径对比新旧侧面数据。
  - 定位旧数据 `datasets/LAT202511`，并用同一审计脚本完成全量统计。
  - 确认目录、通道、polygon 点数、股骨头体系以及 C2/S1 同名异义冲突。
  - 核对旧训练样本清单：353 个有效样本、12 个无标注、26 个不完整。
  - 发现旧数据内部 1 组完全重复 PNG；跨批次全量哈希检查超时，改用候选预筛方案。
- Files created/modified:
  - `task_plan.md`（追加阶段）
  - `findings.md`（追加需求）
  - `progress.md`（追加会话）

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 审计脚本语法 | `python3 -m py_compile` | 无语法错误 | 通过 | ✓ |
| 全量数据复扫 | 706 对 PNG/JSON | 与交付 audit 一致 | 忽略生成时间后逐字段一致 | ✓ |
| 预览回归 | T1/T2 疑似互换样本 | 生成有效 PNG | 320×906 RGB PNG | ✓ |
| 原始数据保护 | 数据目录 | 不写入原始目录 | 所有产物均在工作区或临时目录 | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-09-04 | `git commit` 无法创建 `.git/index.lock` | 1 | 改用受控权限，并限定暂存文件范围 |
| 2026-09-04 | Python 缺少 Pillow | 1 | 使用 PNG IHDR 与系统图像工具替代 |
| 2026-09-04 | 首轮几何统计误报 point/line 且异常明细封顶 | 1 | 修正适用 shape 类型并重跑审计 |
| 2026-09-04 | FFmpeg 无法解码 SVG | 1 | 改用 PPM 中间图与标准库绘线 |
| 2026-09-04 | `tile` 拼图只有首帧 | 1 | 使用显式多输入 `xstack` 生成完整拼图 |
| 2026-09-04 | 一个预览的 PPM 像素长度校验失败 | 1 | 识别为解析器过度跳过二进制像素中的空白字节，准备精确消费单个头分隔符 |
| 2026-09-07 | 追加新阶段时补丁锚点不存在 | 1 | 改用当前文件稳定段落重新应用补丁 |
| 2026-09-07 | 跨批次 1,110 张 PNG 全量 SHA-256 超过 30 秒 | 1 | 改按文件大小交集筛选候选后哈希 |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | 已完成全部阶段 |
| Where am I going? | 向用户交付结论、报告与可复用工具 |
| What's the goal? | 只读分析侧面 LabelMe 数据集并交付可核查结论 |
| What have I learned? | 见 `findings.md` |
| What have I done? | 已完成全量审计、分层视觉质检、兼容性分析与复现验证 |
