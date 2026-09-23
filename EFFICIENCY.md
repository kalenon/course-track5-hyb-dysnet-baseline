# Track 5 基线效率口径

此基线的分类器是随机森林，主要运算是阈值比较。把树推理写成 `0 GMAC/subject` 会遗漏频谱特征提取和树节点比较，不能直接用于课程效率名次。请同时报告树节点数、平均每位受试者访问的树节点数、权重文件字节数，以及特征提取的运行时间或经过统一口径估计的 FFT 运算量。

```bash
python scripts/complexity.py \
  --checkpoint runs/baseline/final_model.joblib \
  --features-cache runs/baseline/features.joblib \
  --output runs/baseline/complexity.json
```

课程组在对随机森林与神经模型进行效率排名前，应先统一“阈值比较、FFT、模型加载、预训练特征提取”如何计入计算量。当前脚本输出可审计的原始量，不把树比较伪装为乘加运算。
