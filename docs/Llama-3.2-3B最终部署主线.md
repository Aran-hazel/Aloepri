# Llama-3.2-3B-Instruct 论文一致最终部署主线

## 1. 唯一目标

当前 Llama 文档只保留一条主线：

> **Llama-3.2-3B-Instruct 的论文一致最终可交付部署线**

这条主线的最终目标与 Qwen 当前部署线保持同构：

- 标准 Transformer 运行图
- 标准 `model.* / lm_head.*` 键布局
- 尽量保留可部署的 embedding / head / attention / FFN / norm 参数扰动表达
- 形成最终可交付的独立 `Stage K` release

当前唯一活跃模型根为：

- `model/Llama-3.2-3B-Instruct`

## 2. 当前状态总览

当前 Llama 不再把 `stable_reference / tiny_a` 作为活跃主线 profile 语义。

它们现在只保留为历史噪声定标和旧交付线证据。当前唯一主线已经收口到：

- `Stage J` 唯一候选：`artifacts/stage_j_llama_instruct_paper_consistent`
- `Stage K` 唯一 release 面：`artifacts/stage_k_llama_release`
- 活跃 profile：`default` / `reference`

这次改造的含义是：

- Llama 默认模型从 base `Llama-3.2-3B` 切换为 `Llama-3.2-3B-Instruct`
- Llama release 组织方式与 Qwen 的 `paper_consistent` 部署线对齐
- release-surface correctness 不再借 Stage J 的 `real_tiny_a` 结果代指，而是使用 `Stage K` 自身路径
- 安全评测仍未达到 Qwen 的完整闭环强度

当前活跃支撑文档为：

- [docs/Llama-3.2-3B标准形状恢复报告.md](Llama-3.2-3B标准形状恢复报告.md)
- [docs/Llama-3.2-3B噪声定标与StageK推进说明.md](Llama-3.2-3B噪声定标与StageK推进说明.md)
- [docs/Llama-3.2-3B客户端与Server使用说明.md](Llama-3.2-3B客户端与Server使用说明.md)

## 3. 当前 `Stage H / I / J / K` 含义

### Stage H

沿用论文部署适配原则，定义哪些混淆表达仍可被吸收到标准 Transformer 参数中。

### Stage I

验证 Llama Instruct 工件是否仍可作为标准 Hugging Face checkpoint 加载，并保持 client/server token mapping 契约。

### Stage J

把 Llama Instruct 目标收束为唯一论文一致候选：

- `artifacts/stage_j_llama_instruct_paper_consistent`

当前导出入口为：

```bash
python scripts/export_stage_j_llama_paper_consistent_checkpoint.py \
  --model-dir model/Llama-3.2-3B-Instruct \
  --export-dir artifacts/stage_j_llama_instruct_paper_consistent \
  --device cpu \
  --dtype bfloat16 \
  --alpha-e 0.02 \
  --alpha-h 0.01
```

### Stage K

把最终确认后的 `Stage J` 产物整理成唯一 release 面：

- `artifacts/stage_k_llama_release`

当前活跃 profile 为：

- `default`
- `reference`

两个 profile 当前都指向同一个 `paper_consistent` Llama Instruct 源工件，但承担不同入口语义：

- `default`：默认交付入口
- `reference`：审计与证据入口

## 4. 当前证据入口

当前直接证据入口为：

- `Stage K` release surface：`artifacts/stage_k_llama_release`
- `Stage K` catalog：`artifacts/stage_k_llama_release/catalog.json`
- `Stage K` correctness evidence：
  - `outputs/stage_k_llama_release/correctness/default.json`
  - `outputs/stage_k_llama_release/correctness/reference.json`
  - `outputs/stage_k_llama_release/correctness_summary.json`

对应 correctness 入口：

```bash
python scripts/run_stage_k_llama_release_correctness.py \
  --baseline-model-dir model/Llama-3.2-3B-Instruct \
  --release-dir artifacts/stage_k_llama_release \
  --profiles default reference \
  --device cuda \
  --dtype bfloat16
```

## 5. 云端部署顺序

在云端 GPU 机器上，推荐顺序为：

1. 下载或准备 `model/Llama-3.2-3B-Instruct`
2. 运行 baseline smoke：

```bash
python scripts/run_llama_baseline_smoke.py \
  --model-dir model/Llama-3.2-3B-Instruct \
  --device cuda \
  --dtype bfloat16
```

3. 导出 Stage J 论文一致候选：

```bash
python scripts/export_stage_j_llama_paper_consistent_checkpoint.py \
  --model-dir model/Llama-3.2-3B-Instruct \
  --export-dir artifacts/stage_j_llama_instruct_paper_consistent \
  --device cpu \
  --dtype bfloat16
```

4. 导出 Stage K release：

```bash
python scripts/export_stage_k_llama_release.py \
  --export-dir artifacts/stage_k_llama_release \
  --materialize
```

5. 运行 release-surface correctness：

```bash
python scripts/run_stage_k_llama_release_correctness.py \
  --baseline-model-dir model/Llama-3.2-3B-Instruct \
  --release-dir artifacts/stage_k_llama_release \
  --profiles default reference \
  --device cuda \
  --dtype bfloat16
```

6. 运行最终 smoke：

```bash
python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile default \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

或者直接使用：

```bash
bash scripts/run_llama_3b_stagek_pipeline.sh
```

## 6. 与 Qwen 的当前差异

Llama 当前已经在 Stage J/K 组织方式上与 Qwen 对齐：

- 都有唯一 `paper_consistent` Stage J 源工件
- 都有唯一 Stage K release 面
- 都使用 `default` / `reference` profile
- 都要求 release-surface correctness 证据落在 Stage K 自身路径

但 Llama 仍弱于 Qwen：

- 尚未补齐与 Qwen 同强度的 `VMA / IMA / ISA` 安全评测闭环
- 当前 Llama 的 paper-consistent 工件仍使用 standard-shape square transform 路线，尚未完整恢复论文中所有 attention / FFN / norm 复杂扰动表达
- 仍需在真实云端环境复跑并落盘新的 Stage K correctness 结果

## 7. 历史语义说明

以下名称现在只保留历史证据价值：

- `stable_reference`
- `tiny_a`
- `artifacts/stage_j_llama_real_full_square`
- `artifacts/stage_j_llama_real_full_square_tiny_a`
- `outputs/stage_j_llama/real_remote_validation.json`
- `outputs/stage_j_llama/real_tiny_a_remote_validation.json`

它们不再代表当前活跃 Llama release profile 语义。

历史使用说明与旧计划保留在：

- `docs/history/llama/Llama-3.2-3B快速使用说明.md`
- `docs/history/llama/Llama-3.2-3B云端验证说明.md`
- `docs/history/llama/Llama-3.2-3B本机改造与云验证计划.md`

## 8. 当前仍未完成的关键项

- 在云端真实 `Llama-3.2-3B-Instruct` 上重新落盘 Stage K correctness
- 以当前 `Stage K` release 为口径补齐更完整安全评测
- 继续向论文部署适配机制靠拢，恢复更多 attention / FFN / norm 参数层复杂扰动

## 9. 一句话结论

当前 Llama 主线已经从旧的 `stable_reference / tiny_a` 交付线改为 `Llama-3.2-3B-Instruct paper_consistent` 部署线；代码和 release 组织方式已经与 Qwen 当前部署线对齐，但安全评测和复杂扰动恢复程度仍需要后续继续补齐。
