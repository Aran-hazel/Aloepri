# Qwen CPU 本机运行流程

本文档记录在 Windows PowerShell 中，从零运行本项目 Qwen 部分的完整命令流程。

## 1. 进入项目目录

```powershell
cd E:\Aloepri
```

## 2. 创建或启用环境

如果还没有创建环境：

```powershell
conda env create -f environment.qwen-transformers.yml
```

如果环境已经创建过：

```powershell
conda activate qwen-transformers
```

如果 PowerShell 找不到 `conda`，优先打开 `Anaconda Prompt` 或 `Miniconda Prompt`，再执行：

```bat
cd /d E:\Aloepri
conda activate qwen-transformers
```

确认 Python 依赖可用：

```powershell
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__)"
```

## 3. 下载 Qwen 模型

新版 Hugging Face CLI 使用 `hf`，不要再用废弃的 `huggingface-cli`。

```powershell
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir model/Qwen2.5-0.5B-Instruct
```

如果提示没有登录，可以先运行：

```powershell
hf auth login
```

下载后检查模型目录：

```powershell
Get-ChildItem model/Qwen2.5-0.5B-Instruct
```

## 4. 先运行明文 Qwen

这一步只验证原始 Qwen 模型、环境和 Transformers 是否能正常工作，不涉及 AloePri 混淆工件。

```powershell
python infer_qwen.py --model-dir model/Qwen2.5-0.5B-Instruct --prompt "请用一句话介绍你自己。" --max-new-tokens 16
```

如果这里能输出文本，说明基础环境已经跑通。

## 5. 导出 AloePri Qwen 工件

按顺序执行以下命令。CPU 下第一步通常最慢。

### 5.1 导出 Stage H pretrained-like 工件

```powershell
python scripts/export_stage_h_pretrained.py --model-dir model/Qwen2.5-0.5B-Instruct
```

成功后应生成：

```text
artifacts/stage_h_pretrained/
```

### 5.2 导出 Stage J redesign 工件

```powershell
python scripts/export_stage_j_redesign_checkpoint.py --materialize
```

成功后应生成：

```text
artifacts/stage_j_qwen_redesign/
```

### 5.3 导出 Stage J paper-consistent 工件

```powershell
python scripts/export_stage_j_paper_consistent_checkpoint.py --materialize
```

成功后应生成：

```text
artifacts/stage_j_qwen_paper_consistent/
```

### 5.4 导出 Stage K release

```powershell
python scripts/export_stage_k_release.py --materialize
```

成功后应生成：

```text
artifacts/stage_k_release/
```

`--materialize` 表示复制目录，而不是创建符号链接。Windows 上推荐加这个参数，避免 symlink 权限问题。

## 6. 运行最终 Qwen Stage K 推理

```powershell
python scripts/infer_stage_k_release.py --release-dir artifacts/stage_k_release --profile default --prompt "请用一句话介绍你自己。" --max-new-tokens 8
```

也可以测试 `reference` profile：

```powershell
python scripts/infer_stage_k_release.py --release-dir artifacts/stage_k_release --profile reference --prompt "请解释一下注意力机制。" --max-new-tokens 16
```

## 7. 可选：运行 correctness 检查

```powershell
python scripts/run_stage_k_release_correctness.py --release-dir artifacts/stage_k_release
```

成功后应生成：

```text
outputs/stage_k_release/correctness/default.json
outputs/stage_k_release/correctness/reference.json
outputs/stage_k_release/correctness_summary.json
```

## 8. 最小成功标准

至少满足这两点：

1. 明文 Qwen 能跑：

```powershell
python infer_qwen.py --model-dir model/Qwen2.5-0.5B-Instruct --prompt "请用一句话介绍你自己。" --max-new-tokens 16
```

2. AloePri Stage K release 能跑：

```powershell
python scripts/infer_stage_k_release.py --release-dir artifacts/stage_k_release --profile default --prompt "请用一句话介绍你自己。" --max-new-tokens 8
```

如果第二步能输出文本，说明 Qwen 主线已经在本机 CPU 跑通。

## 9. 常见问题


### huggingface-cli 提示废弃

使用新命令：

```powershell
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir model/Qwen2.5-0.5B-Instruct
```

### 没有 artifacts 目录

这是正常的。`artifacts/` 默认被 `.gitignore` 忽略，需要通过 Stage H/J/K 导出命令生成。

### CPU 运行很慢

正常。尤其是：

```powershell
python scripts/export_stage_h_pretrained.py --model-dir model/Qwen2.5-0.5B-Instruct
```

这一阶段会加载模型、构建混淆模型、定标并保存工件，CPU 下耗时最长。
