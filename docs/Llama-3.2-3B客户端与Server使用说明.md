> Canonical note: 本文档只回答当前 `Llama-3.2-3B-Instruct` 的 client/server 使用方式，不承担全局主线说明。Llama 唯一主线入口见 [docs/Llama-3.2-3B最终部署主线.md](Llama-3.2-3B最终部署主线.md)。

# Llama-3.2-3B-Instruct 客户端与 Server 使用说明

当前 Llama release 已切到与 Qwen 同构的 `paper_consistent` 语义：

- release：`artifacts/stage_k_llama_release`
- 推荐 profile：`default`
- 审计 profile：`reference`
- Stage J 源工件：`artifacts/stage_j_llama_instruct_paper_consistent`

旧的 `tiny_a / stable_reference` 只保留为历史噪声定标证据，不再是当前活跃 profile。

---

## 1. Server 侧工件

推荐使用：

```text
artifacts/stage_k_llama_release/profiles/default/server
```

这是标准 HF 目录，包含：

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`

server 侧可以像普通 HF 模型一样加载：

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

## 2. Client 侧 secret

client 必须持有：

```text
artifacts/stage_k_llama_release/profiles/default/client/client_secret.pt
```

其中保存：

- `perm_vocab`
- `inv_perm_vocab`

client 负责：

- 输入 token id 映射
- 输出 token id 或 logits 恢复

---

## 3. 本地一体化推理

最简单的 smoke 命令：

```bash
python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile default \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

`reference` profile 也可用于审计：

```bash
python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile reference \
  --prompt "请解释一下注意力机制。" \
  --max-new-tokens 8
```

---

## 4. 正式 client/server 分离

### 4.1 client 准备输入

```bash
python scripts/llama_client_prepare_request.py \
  --server-dir artifacts/stage_k_llama_release/profiles/default/server \
  --client-secret artifacts/stage_k_llama_release/profiles/default/client/client_secret.pt \
  --prompt "请用一句话介绍你自己。" \
  --output-path outputs/llama_client_request.json
```

生成的 `outputs/llama_client_request.json` 包含：

- `input_ids`
- `mapped_input_ids`
- `attention_mask`

真正发送给 server 的应该是：

- `mapped_input_ids`
- `attention_mask`

### 4.2 server 正常推理

```python
import json
import torch
from transformers import AutoModelForCausalLM

server_dir = "artifacts/stage_k_llama_release/profiles/default/server"
request_path = "outputs/llama_client_request.json"

model = AutoModelForCausalLM.from_pretrained(
    server_dir,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).eval().cuda()

payload = json.load(open(request_path, "r", encoding="utf-8"))
input_ids = torch.tensor([payload["mapped_input_ids"]], device="cuda")
attention_mask = torch.tensor([payload["attention_mask"]], device="cuda")

generated = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=8,
    do_sample=False,
)

generated_token_ids = generated[0, input_ids.shape[1]:].tolist()
print({"generated_token_ids": generated_token_ids})
```

### 4.3 client 恢复输出

如果 server 返回：

```json
{"generated_token_ids": [123, 456, 789]}
```

client 执行：

```bash
python scripts/llama_client_restore_ids.py \
  --server-dir artifacts/stage_k_llama_release/profiles/default/server \
  --client-secret artifacts/stage_k_llama_release/profiles/default/client/client_secret.pt \
  --mapped-token-ids "123,456,789" \
  --output-path outputs/llama_client_restored.json
```

输出中会包含：

- `restored_token_ids`
- `decoded_text`

---

## 5. correctness 验收

当前 Llama release-surface correctness 入口为：

```bash
python scripts/run_stage_k_llama_release_correctness.py \
  --baseline-model-dir model/Llama-3.2-3B-Instruct \
  --release-dir artifacts/stage_k_llama_release \
  --profiles default reference \
  --device cuda \
  --dtype bfloat16
```

成功后生成：

```text
outputs/stage_k_llama_release/correctness/default.json
outputs/stage_k_llama_release/correctness/reference.json
outputs/stage_k_llama_release/correctness_summary.json
```
