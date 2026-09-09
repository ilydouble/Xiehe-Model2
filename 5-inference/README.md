# 侧面脊柱联合推理

`batch_predict.py`在同一张侧位片上独立运行两个最终 Pose 模型：

- 20类整脊柱：`C2、C7、T1-T13、L1-L5`，每类4个角点；
- pelvis三关键点：`CFH、S1_left、S1_right`。

脊柱预测按类别分组，每类只保留置信度最高的一个实例；pelvis同样只保留最高置信度实例。

## 快速运行

```bash
cd 5-inference
./run_batch_predict.sh
```

默认读取20类侧面数据集的完整test图目录，并使用：

- `3-model_training/runs/pose/yolo11l_lateral_20cls_scratch_best/weights/best.pt`
- `4-model_training_CFH/runs/yolo11l_pelvis_3kpt_roi_mixed_best/weights/best.pt`

只测试前5张：

```bash
./run_batch_predict.sh --limit 5 --output-dir ../datasets/lateral_combined_predictions_smoke
```

推理任意图片目录：

```bash
./run_batch_predict.sh \
  --image-dir /path/to/images \
  --output-dir /path/to/output \
  --device 0
```

可用参数见：

```bash
./run_batch_predict.sh --help
```

启动脚本优先使用当前环境的`python3`；若本机默认Python没有Ultralytics，会自动尝试`/opt/miniconda3/bin/python3`。也可通过`INFERENCE_PYTHON=/path/to/python`显式指定模型环境。

旧参数名`--corner-model`、`--cfh-model`、`--num-samples`和`--conf`仍可使用，分别映射到新的脊柱模型、pelvis模型、处理上限和置信度参数。

## 输出

默认输出目录为`datasets/lateral_combined_predictions`：

- `combined/`：20类脊柱与三点的联合叠加图；
- `spine_only/`：仅脊柱四角点；
- `pelvis_only/`：仅pelvis框和三个关键点；
- `json/`：每张图一份结构化坐标；
- `predictions.jsonl`：全部图像的汇总坐标；
- `prediction_stats.json`：模型哈希、参数和检测完整度；
- `index.html`：无需脚本即可打开的联合结果画廊。

三点JSON同时保留原始关键点数组和`CFH`、`S1_left`、`S1_right`命名映射。所有坐标均为原图像素坐标。
