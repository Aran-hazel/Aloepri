# Llama-Instruct 论文一致部署线说明

本文档整理当前仓库中 `Llama-3.2-3B-Instruct` 的实现状态、工件结构、运行方式，以及它与旧 `Llama-3.2-3B` 交付线的区别。

## 1. 当前主线是什么

当前 Llama 主线已经从旧的：

```text
stable_reference / tiny_a
```

切换为与 Qwen 同构的：

```text
paper_consistent
```

当前唯一活跃主线是：

- 模型根：`model/Llama-3.2-3B-Instruct`
- Stage J 候选：`artifacts/stage_j_llama_instruct_paper_consistent`
- Stage K release：`artifacts/stage_k_llama_release`
- 活跃 profile：`default` / `reference`

这条线的目标是：

- 标准 Transformer 运行图
- 标准 `model.* / lm_head.*` 键布局
- 保持 client/server 分离部署
- 用唯一 Stage J 源工件生成唯一 Stage K release 面

## 2. 和旧 Llama 线的区别

旧线的语义是：

- `stable_reference`：零噪声 correctness 基线
- `tiny_a`：推荐非零噪声工作点

它的问题不是不能用，而是组织方式更像“噪声 profile 交付线”，不是“论文一致主线”。

现在的新线改成：

- 不再把 `stable_reference / tiny_a` 当作当前 release profile
- 不再把 Stage J 的旧 validation 结果当作 Stage K correctness 证据
- 把 Llama 的主线组织方式对齐到 Qwen：
  - 唯一 Stage J 源工件
  - 唯一 Stage K release 面
  - `default / reference` profile
  - Stage K 自身 correctness 结果

旧对象现在只保留历史参考价值：

- `artifacts/stage_j_llama_real_full_square`
- `artifacts/stage_j_llama_real_full_square_tiny_a`
- `outputs/stage_j_llama/real_remote_validation.json`
- `outputs/stage_j_llama/real_tiny_a_remote_validation.json`

## 3. 当前关键代码

当前这条线的核心入口有这些：

- Llama Stage J 论文一致导出：
  - `src/stage_j_llama_paper_consistent.py`
  - `scripts/export_stage_j_llama_paper_consistent_checkpoint.py`

- Llama Stage K release：
  - `src/stage_k_llama_release.py`
  - `scripts/export_stage_k_llama_release.py`

- Llama Stage K correctness：
  - `src/stage_k_llama_correctness.py`
  - `scripts/run_stage_k_llama_release_correctness.py`

- 统一推理入口：
  - `scripts/infer_stage_k_release.py`

- client/server 辅助脚本：
  - `scripts/llama_client_prepare_request.py`
  - `scripts/llama_client_restore_ids.py`

## 4. 当前工件结构

### 4.1 Stage J 候选工件

当前唯一 Stage J 候选目录：

```text
artifacts/stage_j_llama_instruct_paper_consistent/
├── server/
├── client/
├── manifest.json
└── paper_consistent_target.json
```

其中：

- `server/` 是标准 HF checkpoint
- `client/client_secret.pt` 保存 `perm_vocab` 与 `inv_perm_vocab`
- `manifest.json` 记录这是一条 `paper_consistent_candidate`

### 4.2 Stage K release

当前唯一 release 目录：

```text
artifacts/stage_k_llama_release/
├── catalog.json
├── deployment_contract.json
├── README.md
└── profiles/
    ├── default/
    │   ├── server/
    │   └── client/
    └── reference/
        ├── server/
        └── client/
```

两个 profile 当前都指向同一个 Stage J 源工件，但语义不同：

- `default`：默认交付入口
- `reference`：审计和证据入口

## 5. 当前 correctness 证据

Llama 当前已经不再把 Stage J 的 `real_tiny_a` 或 `real_remote_validation` 当作活跃 release 证据。

当前只认 Stage K 自身路径：

```text
outputs/stage_k_llama_release/correctness/default.json
outputs/stage_k_llama_release/correctness/reference.json
outputs/stage_k_llama_release/correctness_summary.json
```

这和 Qwen 当前主线是一致的。

## 6. 云端部署顺序

如果要在云端使用当前 Llama-Instruct 主线，推荐顺序如下。

### 6.1 准备模型

```bash
hf auth login
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir model/Llama-3.2-3B-Instruct
```

### 6.2 跑明文 smoke

```bash
python scripts/run_llama_baseline_smoke.py \
  --model-dir model/Llama-3.2-3B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

### 6.3 导出 Stage J 论文一致候选

```bash
python scripts/export_stage_j_llama_paper_consistent_checkpoint.py \
  --model-dir model/Llama-3.2-3B-Instruct \
  --export-dir artifacts/stage_j_llama_instruct_paper_consistent \
  --device cpu \
  --dtype bfloat16 \
  --alpha-e 0.02 \
  --alpha-h 0.01
```

### 6.4 导出 Stage K release

```bash
python scripts/export_stage_k_llama_release.py \
  --export-dir artifacts/stage_k_llama_release \
  --materialize
```

### 6.5 跑 Stage K correctness

```bash
python scripts/run_stage_k_llama_release_correctness.py \
  --baseline-model-dir model/Llama-3.2-3B-Instruct \
  --release-dir artifacts/stage_k_llama_release \
  --profiles default reference \
  --device cuda \
  --dtype bfloat16
```

### 6.6 跑最终推理 smoke

```bash
python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile default \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

## 7. client/server 怎么用

### 7.1 server 侧

推荐 server 工件：

```text
artifacts/stage_k_llama_release/profiles/default/server
```

server 可以像普通 HF 模型一样加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

server_dir = "artifacts/stage_k_llama_release/profiles/default/server"
tokenizer = AutoTokenizer.from_pretrained(server_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    server_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).eval().cuda()
```

### 7.2 client 侧

client 必须持有：

```text
artifacts/stage_k_llama_release/profiles/default/client/client_secret.pt
```

client 负责：

- 输入 token id 映射
- 输出 token id 恢复
- 如有需要，输出 logits 恢复

## 8. 当前实现边界

当前这条 `Llama-3.2-3B-Instruct paper_consistent` 主线，已经和 Qwen 在工程组织上对齐了，但还没有在论文闭环强度上完全对齐。

已经完成的部分：

- 默认模型切到 `Llama-3.2-3B-Instruct`
- 唯一 Stage J 源工件
- 唯一 Stage K release 面
- `default / reference` profile
- release-surface correctness 入口
- client/server 分离使用方式

仍然没完成的部分：

- `VMA / IMA / ISA` 的完整安全闭环
- 更接近论文原始 attention / FFN / norm 复杂扰动的恢复
- 更严格的论文参数和公共语料同态复跑

## 9. 一句话结论

当前 Llama-Instruct 主线已经从“旧 profile 交付线”改造成“与 Qwen 同构的 paper-consistent 部署线”。现在它更像一条真正的论文一致 release 主线，但安全评测和复杂扰动恢复程度仍需要继续补齐。
