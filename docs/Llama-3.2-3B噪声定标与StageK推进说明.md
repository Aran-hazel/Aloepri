> Canonical note: 本文档只回答当前 `Llama-3.2-3B-Instruct` 的噪声工作点继承与 `Stage K` 推进状态，不承担全局主线说明。Llama 唯一主线入口见 [docs/Llama-3.2-3B最终部署主线.md](Llama-3.2-3B最终部署主线.md)。

# Llama-3.2-3B-Instruct 噪声工作点与 Stage K 推进说明

## 1. 当前语义变化

旧 Llama 主线使用：

- `stable_reference`
- `tiny_a`

作为 release profile。现在这两个名称不再是活跃 release profile，只保留为历史噪声定标语义。

当前活跃主线改为：

- 唯一 Stage J 候选：`artifacts/stage_j_llama_instruct_paper_consistent`
- 唯一 Stage K release：`artifacts/stage_k_llama_release`
- 活跃 profile：`default` / `reference`

## 2. 当前继承的噪声工作点

Llama 当前 paper-consistent 工件默认继承旧 `tiny_a` 的非零噪声工作点：

```text
alpha_e = 0.02
alpha_h = 0.01
```

原因是历史真实 3B 验证显示该工作点在 generation correctness 上稳定，而论文默认的 `paper_like` 噪声点对当前 Llama standard-shape 路线过强。

这并不表示 Llama 已经完成论文默认参数同态复现；它只是当前 `Llama-3.2-3B-Instruct` release 的可部署工作点。

## 3. 当前导出入口

导出 Stage J：

```bash
python scripts/export_stage_j_llama_paper_consistent_checkpoint.py \
  --model-dir model/Llama-3.2-3B-Instruct \
  --export-dir artifacts/stage_j_llama_instruct_paper_consistent \
  --device cpu \
  --dtype bfloat16 \
  --alpha-e 0.02 \
  --alpha-h 0.01
```

导出 Stage K：

```bash
python scripts/export_stage_k_llama_release.py \
  --export-dir artifacts/stage_k_llama_release \
  --materialize
```

运行 correctness：

```bash
python scripts/run_stage_k_llama_release_correctness.py \
  --baseline-model-dir model/Llama-3.2-3B-Instruct \
  --release-dir artifacts/stage_k_llama_release \
  --profiles default reference \
  --device cuda \
  --dtype bfloat16
```

## 4. 当前 Stage K catalog 语义

`artifacts/stage_k_llama_release/catalog.json` 应包含：

- `stage_lineage = paper_consistent_stage_j`
- `recommended_profile = "default"`
- `reference_profile = "reference"`
- `profiles = ["default", "reference"]`

两个 profile 都指向：

```text
artifacts/stage_j_llama_instruct_paper_consistent
```

## 5. 历史噪声材料如何理解

以下对象只用于说明旧路线如何选择工作点，不再代表当前 release 主线：

- `outputs/stage_j_llama/real_noise_calibration.json`
- `outputs/stage_j_llama/real_tiny_a_remote_validation.json`
- `artifacts/stage_j_llama_real_full_square`
- `artifacts/stage_j_llama_real_full_square_tiny_a`

当前 correctness 只认 Stage K 自身路径：

```text
outputs/stage_k_llama_release/correctness/default.json
outputs/stage_k_llama_release/correctness/reference.json
outputs/stage_k_llama_release/correctness_summary.json
```

## 6. 当前结论

Llama 已经从旧的 `stable_reference / tiny_a` release profile 语义切换到 `Llama-3.2-3B-Instruct paper_consistent` release 语义。

当前已经对齐 Qwen 的部分：

- 唯一 Stage J 源工件
- 唯一 Stage K release 面
- `default` / `reference` profile
- release-surface correctness 证据入口

尚未对齐 Qwen 的部分：

- `VMA / IMA / ISA` 安全评测闭环
- 更完整的论文复杂扰动恢复
- 论文默认参数和 public corpus 规模同态复跑
