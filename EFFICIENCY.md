# Track 5 基线效率统计

本基线的推理系统包括冻结的 XLSR-53、OpenSMILE eGeMAPSv02 特征提取、StandardScaler，以及 XGBoost、LightGBM、逻辑回归三个分类器。不能只把分类器的 `joblib` 文件大小当作整套系统的参数量，也不能把树节点比较误记成 MAC。

可运行：

```bash
python scripts/complexity_hyb.py \
  --checkpoint models \
  --output "$RUN_ROOT/complexity.json"
```

脚本统计 XLSR 参数量、两类树模型的树数、逻辑回归系数数目和分类器权重文件大小。若要得到 PyTorch profiler 可统计到的 XLSR 算子 FLOP 下界，可额外加 `--profile --device cuda`。该值按八条录音放大，**不是**完整的 GMAC/subject：它不覆盖 OpenSMILE 特征计算，也可能漏掉 profiler 不支持的算子。课程效率计分所需的完整计算量，应另用能覆盖整条推理链的分析方法估算，并在报告中注明统计范围与假设。

对课程重训权重实测：XLSR-53 含 315,437,696 个参数；XGBoost 和 LightGBM 各含 10,000 棵多分类树；逻辑回归系数 1,685 个；`models/final_model.joblib` 为 57,110,157 字节。树数和文件大小不等同于参数量或 GMAC。

参考比较统一使用 batch size 1、一名受试者八条原始 8 kHz 录音、每条最长 5 秒。多模型权重和冻结的 XLSR 都应计入参数与存储开销。
