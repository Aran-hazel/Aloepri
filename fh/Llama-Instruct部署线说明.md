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
- 服务器外部模型根：`/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B-Instruct`
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

### 6.0 新建服务器环境

以下命令假设服务器是 Linux + CUDA GPU，并且已经进入仓库根目录：

```bash
cd /path/to/Aloepri
```

新建并激活 Conda 环境：

```bash
conda create -n aloepri python=3.11 -y
conda activate aloepri
```

安装 PyTorch CUDA 版和项目运行依赖：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -U transformers tokenizers accelerate safetensors sentencepiece tiktoken protobuf huggingface_hub
```

如果 tokenizer 或模型加载阶段提示缺少 `blobfile`，再补装：

```bash
pip install blobfile
```

### 6.1 准备模型

如果模型已经下载在服务器外部目录，推荐直接使用该目录，不必再复制到仓库内：

```bash
export MODEL_DIR=/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B-Instruct
ls -lh "$MODEL_DIR"
```

如果还没有下载，也可以下载到仓库内默认位置：

```bash
hf auth login
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir model/Llama-3.2-3B-Instruct
```

注意：新版 Hugging Face CLI 使用 `hf`，不要再使用已经废弃的 `huggingface-cli download`。

如果下载时报：

```text
Access denied. This repository requires approval.
```

说明当前 Hugging Face 账号尚未获得 `meta-llama/Llama-3.2-3B-Instruct` 访问权限。需要先在浏览器打开模型页面，同意 Meta Llama 协议或申请权限：

```text
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

权限通过后，在服务器重新登录并下载：

```bash
hf auth login
hf auth whoami
hf download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir model/Llama-3.2-3B-Instruct
```

如果之前下载失败导致目录里存在残缺文件，可以先清理：

```bash
rm -rf model/Llama-3.2-3B-Instruct
mkdir -p model/Llama-3.2-3B-Instruct
hf download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir model/Llama-3.2-3B-Instruct
```

下载完成后检查目录：

```bash
ls -lh model/Llama-3.2-3B-Instruct
```

正常应包含 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`generation_config.json`、`model-*.safetensors` 或 `model.safetensors.index.json` 等文件。

也可以先单独测试 tokenizer：

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("model/Llama-3.2-3B-Instruct", trust_remote_code=True)
print(type(tok))
print(tok("hello")["input_ids"][:10])
PY
```

### 6.2 跑明文 smoke

```bash
python scripts/run_llama_baseline_smoke.py \
  --model-dir "$MODEL_DIR" \
  --device cuda \
  --dtype bfloat16 \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

### 6.3 导出 Stage J 论文一致候选

```bash
python scripts/export_stage_j_llama_paper_consistent_checkpoint.py \
  --model-dir "$MODEL_DIR" \
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
  --baseline-model-dir "$MODEL_DIR" \
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

### 6.7 一键 Stage-K 流程

如果环境和模型已经准备好，可以直接使用仓库脚本串完整流程：

```bash
REPO_DIR=/home/nss-d/Aloepri \
MODEL_DIR=/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B-Instruct \
CONDA_ENV=aloepri \
DTYPE=bfloat16 \
INFER_DEVICE=cuda \
bash scripts/run_llama_3b_stagek_pipeline.sh
```

当前仓库脚本已经把上述服务器路径作为默认值；如果服务器路径不变，也可以在仓库根目录直接运行：

```bash
bash scripts/run_llama_3b_stagek_pipeline.sh
```

完成后，默认服务端模型目录为：

```text
artifacts/stage_k_llama_release/profiles/default/server
```

客户端密钥目录为：

```text
artifacts/stage_k_llama_release/profiles/default/client/client_secret.pt
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

## 8. 封装成可远程调用的问答接口

如果目标是让其他机器通过 HTTP 调用当前混淆后的 Llama 模型，可以在 Stage-K release 生成后包一层 FastAPI 服务。

需要注意：严格的 client/server 分离语义下，`client_secret.pt` 应放在 client 或 gateway 侧；真正 server 只接收已经映射过的 token ids。为了先打通外部问答接口，也可以把映射和模型推理放在同一个 FastAPI wrapper 内。

安装 API 依赖：

```bash
pip install fastapi uvicorn pydantic
```

在仓库根目录新建 `serve_llama_stagek.py`：

```python
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.transforms import map_input_ids, unmap_output_ids


SERVER_DIR = "artifacts/stage_k_llama_release/profiles/default/server"
CLIENT_SECRET = "artifacts/stage_k_llama_release/profiles/default/client/client_secret.pt"

DEVICE = "cuda"
DTYPE = torch.bfloat16

app = FastAPI(title="AloePri Llama Stage-K API")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    do_sample: bool = True


class GenerateResponse(BaseModel):
    answer: str


tokenizer = AutoTokenizer.from_pretrained(SERVER_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    SERVER_DIR,
    trust_remote_code=True,
    torch_dtype=DTYPE,
).eval().to(DEVICE)

secret = torch.load(CLIENT_SECRET, map_location="cpu")
perm_vocab = torch.as_tensor(secret["perm_vocab"], dtype=torch.long)
inv_perm_vocab = torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
@torch.inference_mode()
def generate(req: GenerateRequest):
    encoded = tokenizer(req.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"].to(DEVICE)

    mapped_input_ids = map_input_ids(input_ids, perm_vocab).to(DEVICE)

    output_ids = model.generate(
        input_ids=mapped_input_ids,
        attention_mask=attention_mask,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        temperature=req.temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    mapped_new_ids = output_ids[0, mapped_input_ids.shape[1]:].detach().cpu()
    restored_new_ids = unmap_output_ids(mapped_new_ids, inv_perm_vocab)
    answer = tokenizer.decode(restored_new_ids, skip_special_tokens=True)
    return GenerateResponse(answer=answer)
```

启动服务：

```bash
uvicorn serve_llama_stagek:app --host 0.0.0.0 --port 8000
```

其他机器调用：

```bash
curl -X POST http://服务器IP:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请用一句话介绍你自己。",
    "max_new_tokens": 64,
    "temperature": 0.7
  }'
```

返回格式：

```json
{
  "answer": "..."
}
```

如果服务器只想暴露模型推理，不想持有 `client_secret.pt`，则应把 `scripts/llama_client_prepare_request.py` 和 `scripts/llama_client_restore_ids.py` 放在调用方或 gateway 侧使用：

```bash
python scripts/llama_client_prepare_request.py \
  --server-dir artifacts/stage_k_llama_release/profiles/default/server \
  --client-secret artifacts/stage_k_llama_release/profiles/default/client/client_secret.pt \
  --prompt "请用一句话介绍你自己。"

python scripts/llama_client_restore_ids.py \
  --server-dir artifacts/stage_k_llama_release/profiles/default/server \
  --client-secret artifacts/stage_k_llama_release/profiles/default/client/client_secret.pt \
  --mapped-token-ids "..."
```

## 9. 当前实现边界

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

## 10. 一句话结论

当前 Llama-Instruct 主线已经从“旧 profile 交付线”改造成“与 Qwen 同构的 paper-consistent 部署线”。现在它更像一条真正的论文一致 release 主线，但安全评测和复杂扰动恢复程度仍需要继续补齐。
