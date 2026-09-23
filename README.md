# 语音信号处理大作业 Track 5：构音障碍严重程度基线

本仓库按八条 8 kHz 录音预测一个受试者级五分类结果。每条录音提取固定的对数频谱统计量、能量和过零率，串接八条录音后用带类别平衡权重的随机森林分类。它只依赖公开训练数据，不需要外部预训练权重。

## 安装

```bash
conda create -n ssp-track5 python=3.10 -y
conda activate ssp-track5
pip install -r requirements.txt
```

## 下载与放置数据

从课程平台下载训练数据包并解压；把 `train.csv` 和 `train/` 放在任意数据目录。`train.csv` 至少含 `ID,CLASS`，每个 `ID` 在 `train/phonationA/` 等八个目录各有一个 `<ID>_<任务名>.wav`。数据版权与授权按课程平台说明执行。盲测包稍后发布，只需将 `test.csv`、`test/` 按同样结构放置，无需标签。

如使用本地课程准备好的数据，可把下列 `DATA_ROOT` 换成自己的路径：

```bash
DATA_ROOT=/path/to/course_track5_data
python scripts/train.py \
  --audio-root "$DATA_ROOT/train" \
  --labels "$DATA_ROOT/train.csv" \
  --output-dir runs/baseline
```

训练脚本以固定随机种子对训练受试者作分层 80/20 划分，仅用其中 175 人拟合验证模型，在 44 人上计算准确率、五类 Macro-F1、各类别 F1 和混淆矩阵，然后使用全部 219 人训练最终模型。特征缓存、两个模型、验证预测和报告保存在 `runs/baseline/`。

## 基线验证成绩

在课程 219 人训练集的 44 人分层内部验证划分上，随机种子 42：

| 指标 | 实测值 |
|---|---:|
| Accuracy | 0.4545 |
| Macro-F1 | 0.4798 |

这两个数字属于内部验证集，不是后期 53 人盲测集的成绩。可检查 `runs/baseline/validation_metrics.json` 中的每类 F1 与混淆矩阵。最终模型 `final_model.joblib` 使用全部 219 名训练受试者。

## 验证与盲测接口

内部验证成绩由 `scripts/train.py` 自动生成。拿到无标签盲测包后运行：

```bash
python scripts/infer.py \
  --audio-root "$DATA_ROOT/test" \
  --metadata "$DATA_ROOT/test.csv" \
  --checkpoint runs/baseline/final_model.joblib \
  --output runs/baseline/predictions.csv
python scripts/validate_submission.py \
  --metadata "$DATA_ROOT/test.csv" \
  --predictions runs/baseline/predictions.csv
```

教师持有标签时，可用 `python scripts/evaluate.py --labels /private/labels.csv --predictions runs/baseline/predictions.csv --output runs/baseline/metrics.json` 计算受试者级指标。盲测阶段学生只需调用推理和格式检查脚本。

## 文件与模型资源

`requirements.txt` 固定主要运行依赖；训练前不需要下载模型权重。仓库不收录原始音频、受试者映射和后期盲测标签。模型权重由上面的训练命令生成。
