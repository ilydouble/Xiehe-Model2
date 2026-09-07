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
- **Status:** complete
- **Started:** 2026-09-07
- Actions taken:
  - 恢复既有分析上下文，准备按同一口径对比新旧侧面数据。
  - 定位旧数据 `datasets/LAT202511`，并用同一审计脚本完成全量统计。
  - 确认目录、通道、polygon 点数、股骨头体系以及 C2/S1 同名异义冲突。
  - 核对旧训练样本清单：353 个有效样本、12 个无标注、26 个不完整。
  - 发现旧数据内部 1 组完全重复 PNG；跨批次全量哈希检查超时，改用候选预筛方案。
  - 用文件大小预筛确认新旧批次无字节级重复 PNG。
  - 按共同 18 类统计，新批 700/706 可直接进入完整类别候选池；旧 353 个有效患者样本对应现有 368 张 YOLO 图像。
  - 识别旧转换器“首个 JSON 应用于目录内所有 PNG”的多图目录错配风险。
  - 审计现有 YOLO split，确认 5 个患者跨 train/val/test，需在合并时整体重拆。
  - 用旧数据 18 例 C1/C2 circle 与 CFH 的共标注验证“两股骨头中心中点≈CFH”，中位误差 5.88 px。
  - 确认新批 CFH 与 FH 双点中点的归一化位置分布一致，并识别出 705 个患者 ID，其中 1 个患者有两张片。
  - 对旧有效患者逐图严格配对，确认可直接使用的共同 18 类唯一图像为 356 张；现有 YOLO 中有 10 张疑似套错标注的无同 stem 图像。
- Files created/modified:
  - `findings.md`（追加兼容性结论）
  - `progress.md`（追加核对过程）

### Phase 6: 合并方案与风险结论
- **Status:** complete
- Actions taken:
  - 形成推荐的 18 类合并口径、数据规范、转换流程和风险控制方案。
  - 复跑旧批严格配对、新批共同类别、跨批精确重复、患者 split 泄漏和股骨头中点验证。
  - 确认保守结构完整候选规模为 1,056 张，已知 T1/T2 互换修正后可使用全部候选。
- Files created/modified:
  - `docs/lateral_dataset_merge_assessment.md`（新建）
- Files created/modified:
  - `task_plan.md`（追加阶段）
  - `findings.md`（追加需求）
  - `progress.md`（追加会话）

### Phase 7: 第二批转换规范与实现
- **Status:** complete
- **Started:** 2026-09-07
- Actions taken:
  - 确认用户要求在 `datasets/` 下实际生成第二批 YOLO 训练集，并提供 AutoDL 训练脚本。
  - 固定本轮标签范围为 23 类椎体 Pose（C2-C7、T1-T12、L1-L5），S1/CFH 分离，T13 等待新增有效病例后扩类。
  - 核对第二批 C2 polygon 697 例、C7 polygon 705 例，确认可先独立训练完整椎体模型。
  - 检查现有训练入口和 `.gitignore`：生成数据已被 Git 排除；新模型需要独立的 23 类配置与训练入口，并处理水平翻转时的关键点索引交换。
  - 新增依赖标准库的 LabelMe→YOLO Pose 转换器、8例默认复核隔离表和4项单元测试。
  - 转换器实现严格配对、患者级确定性拆分、四角点规范排序、越界裁剪、原图无损复制、manifest 与转换报告输出。

### Phase 8: 生成第二批 YOLO 数据集
- **Status:** complete
- Actions taken:
  - 运行转换器，将706对源文件构建为 `datasets/yolo_lateral_20260903_23cls`。
  - 隔离8例人工复核样本，最终纳入698张图像、697名患者；按患者拆为558 train、70 val、70 test。
  - 全量验证698对图像/标签、16,032行标注、23类ID、17字段格式、归一化坐标与关键点可见性；未发现错误或患者泄漏。

### Phase 9: AutoDL 训练入口
- **Status:** complete
- Actions taken:
  - 确认本地旧模型位于 `6-app_backend/models/corner_model.pt`，可选作迁移训练初始化权重。
  - 确认数据集不会随 Git 上传，训练说明需明确单独传输约2.0 GiB的数据目录。
  - 新增23类AutoDL训练脚本，支持官方或旧模型初始化、分辨率/batch/device等参数、断点续训与dry-run。
  - 新增训练脚本测试和完整服务器操作说明；真实数据dry-run验证通过。

### Phase 10: 完整验证与交付
- **Status:** complete
- Actions taken:
  - 最终复跑7项转换/训练测试，全部通过。
  - 最终复跑真实数据dry-run与交付文件检查，确认698张图像、697名患者和8例隔离记录完整。
  - 执行Git空白错误检查并确认未触碰用户已有的 `4-model_training_CFH/train_cfh_detection.sh` 修改及未跟踪分析产物。

### Phase 11: 联合骨盆数据规范与转换器
- **Status:** in_progress
- **Started:** 2026-09-07
- Actions taken:
  - 用户提供第一批新位置 `/Volumes/E/spine_data/LAT202511`，已确认目录可访问。
  - 复核旧批335例和新批694例具备统一三关键点所需的完整标注，原始候选合计1,029例。
  - 固定联合目标为单个pelvis对象与 `CFH、S1图像左端、S1图像右端` 三关键点。
  - 核对两批严格同stem配对和代表性几何，确认旧384对、新706对，目标point/line结构可统一。
  - 新增联合骨盆LabelMe→YOLO Pose转换器及覆盖直接CFH、FH中点、S1端点排序、去重和端到端转换的测试。
  - 首轮测试暴露Python 3.13动态导入dataclass模块需先注册`sys.modules`，已修复测试夹具并保留错误记录。
  - 首次真实生成发现旧批只纳入329例；追查确认5例同时含合法S1 line与辅助S1 circle，规则已修正为采用唯一合法line，预计最终总量由1,023增至1,028。

### Phase 12: 生成联合骨盆数据集
- **Status:** complete
- Actions taken:
  - 从旧批384对和新批706对LabelMe源文件生成 `datasets/yolo_pelvis_3kpt_all`。
  - 纳入旧批334张、新批694张，共1,028张图像、1,023个患者组；按患者拆为824 train、102 val、102 test。
  - 保留569个直接CFH和459个FH-1/FH-2中点CFH，移除1张旧批完全重复图像。
  - 独立全量检查图片/标签配对、14字段格式、类别、归一化坐标、3点可见性和患者split，错误数为0。

### Phase 13: 骨盆模型AutoDL训练入口
- **Status:** complete
- Actions taken:
  - 新增1类3关键点YOLO11 Pose训练入口，默认1280、batch 4、200 epochs，使用保守医学影像增强和正确水平翻转。
  - 支持官方 `yolo11m-pose.pt` 或上一阶段23类侧面模型 `best.pt` 初始化，以及断点续训、设备、batch、imgsz等参数。
  - 训练前全量校验14字段标签、三点可见性、图片配对、manifest计数及患者split。
  - 新增4项训练脚本测试和AutoDL说明；真实数据dry-run正确识别1,028张图像、1,023个患者组。

### Phase 14: 最终验证与交付
- **Status:** complete
- Actions taken:
  - 复跑第二批脊柱和联合骨盆的转换/训练测试共17项，全部通过。
  - 复跑联合骨盆真实数据dry-run，确认824/102/102、1,028对象、1,023患者组。
  - 检查Python语法、Git空白错误和数据忽略规则，均通过；3.7 GiB数据目录不会进入Git。
  - 确认用户已有CFH训练脚本修改和未跟踪 `analysis/` 产物未被触碰或提交。
  - 用户指出训练目录职责后，将骨盆三关键点训练入口从脊柱目录移至 `4-model_training_CFH`，同步修正测试、说明和独立依赖文件；脊柱23类入口继续保留在 `3-model_training`。
  - 进一步参考原CFH训练入口，新增骨盆专用shell启动器，提供quick/standard/best预设、脊柱权重迁移、断点续训和dry-run，同时使用脚本绝对路径避免工作目录差异。
  - 同步为第二批23类脊柱模型新增同风格shell启动器，提供三档预设、旧18类corner权重迁移、续训和dry-run，使两个训练目录的使用方式一致。

### Phase 15: 第二批图像规格与黑边统计实现
- **Status:** complete
- **Started:** 2026-09-07
- Actions taken:
  - 用户要求进一步全量统计第二批图像的分辨率、纵横比和黑边情况。
  - 决定在既有LabelMe结构审计基础上增加像素外观审计，原始数据保持只读。
  - 计划同时输出逐图CSV和汇总JSON，并用极端样本视觉复核自动黑边判定。
  - 检查本机图像能力：无Pillow/OpenCV/NumPy，但有FFmpeg；决定用FFmpeg原生解码并缩小为灰度PGM，再用标准库统计边缘像素。
  - 首次阈值校准误选AppleDouble伪PNG并被FFmpeg拒绝，已将`._*`过滤加入正式实现要求。
  - 校准12个来源代表样本后，固定近黑阈值为灰度≤5、每行/列近黑像素≥98%；单侧占比≥1%定义为明显黑边。
  - 新增全量图像外观统计脚本和4项测试，覆盖PNG头、PGM首像素边界、黑边计算、分位数与汇总逻辑，全部通过。

### Phase 16: 全量运行与视觉复核
- **Status:** complete
- Actions taken:
  - 用4个并行FFmpeg解码任务完成706/706张PNG扫描，输出逐图CSV和汇总JSON，无解码失败。
  - 得到292种分辨率、宽高比和17个来源的分层统计，并区分任何黑线、明显黑边和严重黑边。
  - 视觉查看最大上下黑边LFDS、明显上下黑边LWSY、唯一左侧黑边XTZY和无边框FQSY，自动检测与肉眼一致。
  - 补充分辨率主规格占比和黑边严重度分层；对全部315张RGBA抽取alpha通道，确认全部完全不透明。

### Phase 17: 分析报告与训练建议
- **Status:** complete
- Actions taken:
  - 编写第二批图像规格与黑边正式报告，覆盖分辨率、宽高比、通道、来源分布、黑边严重度和训练影响。
  - 建议基线训练保持原图和letterbox/rect，不拉伸；后续按来源误差决定是否开展自动裁黑边并同步关键点坐标的对照实验。
  - 在新临时目录复扫706张，CSV与JSON均和首次结果逐字节一致；复跑4项单元测试和语法检查全部通过。

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 审计脚本语法 | `python3 -m py_compile` | 无语法错误 | 通过 | ✓ |
| 全量数据复扫 | 706 对 PNG/JSON | 与交付 audit 一致 | 忽略生成时间后逐字段一致 | ✓ |
| 预览回归 | T1/T2 疑似互换样本 | 生成有效 PNG | 320×906 RGB PNG | ✓ |
| 原始数据保护 | 数据目录 | 不写入原始目录 | 所有产物均在工作区或临时目录 | ✓ |
| 旧批严格配对 | 353 个已选患者目录 | 每图同 stem JSON、18 类完整、去重 | 356 张唯一完整图像 | ✓ |
| 新批共同标签 | 706 张图像 | 统计共同 18 类完整性 | 700 完整、6 不完整 | ✓ |
| 跨批精确重复 | 404 旧 PNG + 706 新 PNG | 无重复或列出重复 | 0 组跨批字节级重复 | ✓ |
| 患者 split | 现有 368 张 YOLO 图像 | 无患者跨集合 | 发现 5 个患者跨集合，需重拆 | ⚠ |
| 股骨头中点 | 旧 C1/C2/CFH 共标 18 例 | 中点接近 CFH | 中位 5.88 px、最大 10.87 px | ✓ |
| 新转换器单元测试 | 角点排序、裁剪、患者拆分、端到端转换 | 全部通过 | 4/4通过 | ✓ |
| 第二批 YOLO 全量校验 | 698张图像与16,032个对象 | 配对、字段、ID、坐标、患者拆分合法 | 0错误、0患者泄漏 | ✓ |
| AutoDL训练dry-run | 生成的23类数据集 | 校验通过且不启动GPU | 识别698张、16,032对象 | ✓ |

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
| 2026-09-07 | split 重叠检查破坏 zsh PATH 且 awk 语法失败 | 1 | 改用临时 Python 脚本，避免 shell 特殊变量和保留名 |
| 2026-09-07 | 股骨头分布脚本误读 AppleDouble JSON | 1 | 加入 `not name.startswith("._")` 过滤 |
| 2026-09-07 | `jq` 查询审计标签时字段路径错误返回 null | 1 | 检查顶层结构后使用 `annotations.label_stats` |
| 2026-09-07 | `xargs jq` 抽样管道被末端 `head` 提前关闭并报告SIGPIPE | 1 | 数据已正确取得；后续不再用会提前关流的同类管道 |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | 已完成新旧侧面数据合并评估 |
| Where am I going? | 向用户交付合并结论；如获指示再实际构建大数据集 |
| What's the goal? | 只读分析侧面 LabelMe 数据集并交付可核查结论 |
| What have I learned? | 见 `findings.md` |
| What have I done? | 已完成两批数据同口径对比、严格候选计数、去重、split 泄漏检查及合并方案 |
