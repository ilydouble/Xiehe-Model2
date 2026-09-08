# yolo_corner旧18类标注来源全量审计

## 结论

对 `datasets/yolo_corner` 的368张图逐一反向重放旧转换器，并把现有YOLO Pose标签与患者目录内每一份原始LabelMe JSON的转换结果比较：

- 353张标签精确来自本图同stem JSON。
- 15张标签精确来自另一张图的JSON。
- 上述15张中，1张目标PNG与标签来源PNG的SHA-256完全相同，是重复拷贝，应去重而非重新标注。
- 其余14张不是同一图，需要重新补全全部23类：C2-C7、T1-T12、L1-L5。
- 14张分布为train 13张、val 1张，涉及11位患者。
- 14张中10张没有同stem JSON；另4张虽然存在同stem JSON，旧转换器仍选用了患者目录里的另一份JSON。
- 368张数据集PNG与映射到的原始PNG逐字节一致，排除了复制、缩放或裁剪数据集图像造成的错配。

当前E盘C2-C6复核包仅用“是否存在同stem JSON”筛出10张，因此低估了4张有同stem JSON但旧标签实际来自其他JSON的情况。该包不能作为最终23类重建依据，需在下一版中把以下14张改为“补全23类”人工复核。

## 需要补全23类的14张图

1. `images/train/CUI_YI_NING.DX.DX_ALL-SPINE_AP_LAT_MOSAIC.3738.0104.2025.06.12.19.11.56.265625.98916781.png`
2. `images/train/DE_JI_BAI_MA.DX.DX_ALL-SPINE_AP_LAT_MOSAIC.2476.0104.2025.06.09.01.25.58.984375.95342191.png`
3. `images/train/FU_YI_YAN.CR.DX_ALL-SPINE_AP_LAT_MOSAIC.0002.0001.2025.06.12.19.22.12.359375.99073147.png`
4. `images/train/HUANG_SHI_TING.DX.DX_ALL-SPINE_AP_LAT_MOSAIC.7458.0107.2025.06.08.19.52.01.171875.90833340.png`
5. `images/train/LIANG_TIAN.CR.DX_ALL-SPINE_AP_LAT_MOSAIC.0001.0001.2025.06.13.03.14.29.984375.106057416.png`
6. `images/train/LI_CHEN_XIA.DX.DX_ALL-SPINE_LR_BENDING_LATERAL_FLEXION.1530.0101.2025.06.12.22.19.50.312500.101319966.png`
7. `images/train/WANG_YANG_DI.DX.DX_ALL-SPINE_AP_LAT_MOSAIC.3358.0104.2025.06.08.14.11.49.484375.86976230.png`
8. `images/train/WANG_ZHUO_YAN.CR.DX_ALL-SPINE_AP_LAT_MOSAIC.0001.0001.2025.06.08.17.51.21.328125.89497846.png`
9. `images/train/WANG_ZHUO_YAN.DX.DX_ALL-SPINE_LAT_MOSAIC.9962.0104.2025.06.08.17.51.21.328125.89526016.png`
10. `images/train/WANG_ZHUO_YAN.DX.DX_ALL-SPINE_LAT_MOSAIC.9962.0104.2025.06.08.17.51.21.328125.89526082.png`
11. `images/train/ZHAO_YI_XUAN.DX.DX_CHEST_AP_LAT.5852.0101.2025.06.08.19.36.44.375000.90648583.png`
12. `images/train/ZHAO_YI_XUAN.DX.DX_CHEST_AP_LAT.5852.0104.2025.06.08.19.36.44.375000.90648704.png`
13. `images/train/ZHU_YU_JIE.DX.DX_ALL-SPINE_AP_LAT_MOSAIC.2060.0104.2025.06.08.15.52.15.906250.88147858.png`
14. `images/val/GUO_ZI_YAN.DX.DX_ALL-SPINE_AP_LAT_MOSAIC.2316.0110.2025.06.12.20.45.26.937500.99762046.png`

## 应去重而非重标的1张图

`images/train/WANG_YANG.CR.DX_ALL-SPINE_AP_LAT.0002.0001.2025.06.08.14.40.46.937500.87242504.png`

它与标签来源 `WANG_YANG.CR.DX_ALL-SPINE_AP_LAT.0002.0001.2025.06.08.14.49.36.671875.87242504.png` 的SHA-256完全相同。最终数据集只保留一份。

## 复现

```bash
/opt/miniconda3/bin/python3 1-check_data/audit_yolo_corner_label_sources.py
```

审计产物位于 `analysis/yolo_corner_label_source_audit/`：

- `summary.json`
- `per_image_audit.csv`
- `full_23cls_reannotation_list.md`
