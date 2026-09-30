# 语音信号处理大作业 Track 5：构音障碍严重程度评估基线

本项目提供 Hyb-DysNet 结构的参考系统、**使用课程训练数据重新训练的基线权重**、复现脚本，以及验证与盲测提交接口。赛题规则见 [大作业 Track 5](docs/大作业%20track%205.md)。这是课程中的语音算法练习，不用于医疗诊断。

## 方法概览

对每名受试者的八条录音分别提取 OpenSMILE eGeMAPSv02 和冻结的 XLSR-53 语音表征；经标准化后，用 XGBoost、LightGBM 和逻辑回归做加权软投票。八条录音各产生一个类别，再以多数投票得到受试者级的 1–5 类预测。模型使用 16 kHz、最长 5 秒的音频进行深度特征提取；课程提供的 8 kHz WAV 由脚本自动重采样。

随仓库提供的课程重训权重使用 337 维输入：88 维 OpenSMILE 特征和 249 维 XLSR 时序特征。特征实现与该权重保持一致。年龄、性别只用于保留元数据接口，不参与此参考系统。

## 平台资源与环境

课程平台提供训练集和公开验证集供挂载使用；GPU 编号及数据挂载路径取决于运行环境。若平台已有兼容环境，可直接使用。以下命令在仓库根目录执行：

```bash
conda create -n ssp-track5 python=3.10 -y
conda activate ssp-track5
pip install -r requirements.txt
```

下列示例路径分别对应平台提供的数据挂载点和可写工作目录，使用时按实际路径设置；数据挂载可以只读，`RUN_ROOT` 必须可写：

```bash
export GPU_ID=0
export DATA_ROOT=/path/to/platform-mounted/track5_data
export RUN_ROOT=/path/to/your-writable-workspace/track5
mkdir -p "$RUN_ROOT"
export TORCH_HOME="$RUN_ROOT/torch-cache"    # XLSR-53 预训练权重缓存
```

`CUDA_VISIBLE_DEVICES="$GPU_ID"` 会把所选 GPU 映射为程序内的 `cuda:0`，无需修改脚本；CPU 环境下可使用 `--device cpu` 并减小 batch size。

目录结构：

```text
$DATA_ROOT/
  train.csv              # ID,Age,Sex,Class；全部有标签训练受试者
  baseline_train.csv     # 固定基线拟合划分
  validation.csv         # 固定公开验证划分
  train/phonationA/S0001_phonationA.wav
  train/phonationE/S0001_phonationE.wav
  ...                    # 每人 8 条 WAV，另有 rhythmPA/TA/KA
  test.csv               # 后续发布；无 Class 字段
  test/phonationA/...    # 后续发布
```

数据按受试者划分，同一名受试者的录音不得分入不同划分。受试者 ID 只用于定位录音与对齐结果，不是模型特征。

## 准备 XLSR-53 并使用课程基线权重

`models/final_model.joblib` 是本课程使用全部 219 名有标签训练受试者重训得到的最终基线，已包含在本仓库，无需另行下载。冻结的 XLSR-53 特征提取器还依赖 torchaudio 预训练权重（约 1.2 GB），由使用者下载或指定平台预缓存，可运行：

```bash
python scripts/prepare_xlsr.py --torch-home "$TORCH_HOME"
```

脚本通过 torchaudio 官方模型包下载到指定 PyTorch 缓存。

可用课程录音检查基线权重的推理流程：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/infer_hyb.py \
  --audio-root "$DATA_ROOT/train" \
  --metadata "$DATA_ROOT/validation.csv" \
  --checkpoint models \
  --features-cache "$RUN_ROOT/public_validation_features.pkl" \
  --batch-size 4 --device cuda \
  --output "$RUN_ROOT/public_validation_predictions.csv"
```

`final_model.joblib` 使用了全部 219 名训练受试者，其中包含公开验证划分，因此**其在 `validation.csv` 上的分数不能作为课程基线成绩**。公平比较应使用下一节的固定 175/44 划分重训结果，或参考 [基线结果说明](BASELINE_RESULTS.md)。CPU 环境下可将 `--device cuda` 改为 `--device cpu`，并调小 `--batch-size`。第一次提取特征最耗时；重复调用会读取 `--features-cache`。缓存只能用于同一份音频和同一组受试者，替换音频后需使用新的缓存路径。

## 在课程训练集上重新训练

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train_hyb.py \
  --audio-root "$DATA_ROOT/train" \
  --metadata "$DATA_ROOT/train.csv" \
  --baseline-train "$DATA_ROOT/baseline_train.csv" \
  --validation "$DATA_ROOT/validation.csv" \
  --output-dir "$RUN_ROOT/retrained" \
  --batch-size 4 --device cuda --threads 4
```

脚本先用固定划分的训练受试者训练并在公开验证受试者上评估，再用全部有标签训练受试者训练最终模型。训练预处理包括训练集内 SMOTE、StandardScaler 和加权软投票集成；验证受试者不参与重采样或模型拟合。输出包括 `validation_model.joblib`、`validation_predictions.csv`、`validation_metrics.json` 和 `final_model.joblib`。首次运行会缓存全部训练音频的特征到 `train_features.pkl`，再次训练可复用；换数据时应删除旧缓存或指定新的输出目录。CPU 训练也可运行，但特征提取和集成训练会较慢。

技术报告应记录实际训练配置。验证成绩位于 `$RUN_ROOT/retrained/validation_metrics.json`，其中评估的是只使用 `baseline_train.csv` 拟合的验证模型；`final_model.joblib` 使用全部 219 名训练受试者，不用于报告公开验证分数。XLSR 特征提取使用所选 GPU；此脚本的 XGBoost 和 LightGBM 拟合在 CPU 上进行，`--threads` 控制其线程数。

## 盲测接口

无标签盲测包将在后续由课程平台提供。盲测数据不需要 `Class` 字段，也不需要访问标签；数据挂载后可运行以下命令。

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/infer_hyb.py \
  --audio-root "$DATA_ROOT/test" \
  --metadata "$DATA_ROOT/test.csv" \
  --checkpoint models \
  --features-cache "$RUN_ROOT/test_features.pkl" \
  --batch-size 4 --device cuda \
  --output "$RUN_ROOT/test_predictions.csv"
python scripts/validate_submission.py \
  --metadata "$DATA_ROOT/test.csv" \
  --predictions "$RUN_ROOT/test_predictions.csv"
```

使用重新训练的最终权重时，`--checkpoint models` 可改为 `--checkpoint "$RUN_ROOT/retrained/final_model.joblib"`。预测文件格式为 `ID,Class`，每名受试者恰好一行，类别为 1–5。`scripts/evaluate.py` 仅用于有标签的公开验证集；无标签盲测只运行推理和格式检查。

## 结果与效率

[基线结果说明](BASELINE_RESULTS.md) 和 [效率统计说明](EFFICIENCY.md) 提供比较口径。公开验证成绩以相同划分、相同评测脚本的运行结果为准。报告中应记录全部预处理的版本与资源开销。