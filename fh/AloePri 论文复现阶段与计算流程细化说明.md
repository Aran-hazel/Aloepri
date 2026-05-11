# AloePri 论文复现阶段与计算流程细化说明

本文档回答两个问题：

1. 项目中的 `Stage A -> K` 分别对应论文的哪个部分。
2. 如果要完整复现 AloePri 论文，每一步实际需要做哪些计算、产生什么工件、如何判断成功。

注意：项目中的 `Stage A -> K` 不是论文原文的章节名，而是本仓库为了逐步复现论文方法而设计的工程阶段。

---

## 1. 论文主线和项目阶段的总对应关系

论文主线可以压缩成五部分：

| 论文内容 | 核心问题 | 项目阶段 |
| --- | --- | --- |
| Section 4：Covariant Obfuscation 理论 | 为什么可以同时混淆输入、模型参数和输出，并仍然保持推理近似正确 | `Stage B-D` 的协变恢复验证，`Stage D` 的多层组合验证 |
| Section 5.1：AloePri 总流程 | 离线模型混淆 + 在线 prompt 混淆 + response 反混淆 | `Stage A`、`Stage K` |
| Section 5.2.1：Key Matrix Generation | 如何生成 hidden-space 的秘密坐标变换 | `Stage F`、`Stage G` |
| Section 5.2.2-5.2.5：Embedding / Head / Attention / FFN / Norm 混淆 | 如何分别改写模型各组件权重 | `Stage A`、`Stage C`、`Stage E`、`Stage F`、`Stage G`、`Stage H` |
| Section 5.3-5.5 与 Section 6：在线推理、正确性、安全性、实验 | 如何运行、如何验证 correctness、如何评估攻击风险 | `Stage I`、`Stage J`、`Stage K`、`security_qwen` |

更细的阶段对应如下：

| 项目阶段 | 项目名称 | 对应论文部分 | 阶段目的 |
| --- | --- | --- | --- |
| `Stage A` | 词表空间闭环 | Section 5.1, 5.2.2, 5.3 | 复现 token permutation、embedding/head 行置换、输入输出映射 |
| `Stage B` | hidden-space 入口与 block0 attention | Section 4.2, 5.2.3 | 引入 hidden transform，验证 attention 子层能在混淆空间闭环 |
| `Stage C` | block0 完整恢复 | Section 5.2.3, 5.2.4, Norm 部分 | 在一个 block 内复现 norm + attention + FFN 的协变恢复 |
| `Stage D` | 多层 block 传播 | Section 4.2 组合定理 | 验证单层协变恢复是否可以推广到多层 |
| `Stage E` | 复杂 attention 结构 | Section 5.2.3 Attention Obfuscation | 接入 head/group permutation、RoPE 侧变换、block permutation 等复杂 attention 混淆 |
| `Stage F` | KeyMat 原生体系接入 | Section 5.2.1 Algorithm 1 | 接入 Key Matrix / inverse Key Matrix |
| `Stage G` | KeyMat 融合化 / 去 bridge 化 | Section 5.2.1-5.2.5 | 尽量把 runtime bridge 吸收到权重表达里 |
| `Stage H` | 可部署混淆表达整理 | Section 5.2 全部组件 | 判断哪些论文混淆表达可以保持标准 Transformer 推理图 |
| `Stage I` | 标准 HF/vLLM 部署边界验证 | 论文工业兼容约束，Section 5.3 | 验证工件是否能被标准推理框架加载和调用 |
| `Stage J` | 标准 shape / 论文一致候选 | Section 5.2, 5.4 | 形成标准 shape、标准权重键、尽量保留论文扰动表达的候选 checkpoint |
| `Stage K` | release 包装与最终推理入口 | Section 5.3, Section 6 | 形成可交付 release、profile、client/server contract、correctness 证据 |

---

## 2. 论文方法的完整计算视角

AloePri 的核心不是单纯加密，也不是只替换 prompt 里的敏感词，而是做一套协变混淆：

```text
明文输入 x
明文模型参数 θ
明文输出 y

经过混淆后变成：

混淆输入 φX(x)
混淆模型参数 φΘ(θ)
混淆输出 φY(y)

服务端在混淆空间中运行：

f_tilde(φX(x), φΘ(θ)) ≈ φY(f(x, θ))

客户端最后用：

ψY(φY(y)) = y
```

直观解释：

```text
客户端把 prompt 变成混淆 token
服务端只看到混淆 prompt 和混淆模型
服务端输出混淆 token/logits
客户端再恢复成明文 token
```

所以完整复现必须同时处理：

1. 输入 token 如何混淆。
2. embedding 如何跟着 token permutation 改写。
3. hidden state 如何进入秘密坐标系。
4. attention / FFN / norm 如何在秘密坐标系里保持近似正确。
5. lm head 如何输出混淆 logits。
6. 客户端如何恢复 logits 或 token id。
7. 如何证明导出工件没有破坏模型行为。
8. 如何评估 VMA / IMA / ISA 等攻击是否还能恢复明文信息。

---

## 3. Stage A：词表空间闭环

### 对应论文部分

- Section 5.1：AloePri overview
- Section 5.2.2：Embedding and Model Head Obfuscation
- Section 5.3：Online inference

### 目标

复现最基础的 token-level permutation：

```text
plain token id -> obfuscated token id
obfuscated output id/logits -> plain output id/logits
```

### 需要计算什么

1. 读取 tokenizer，得到词表大小 `n`。
2. 识别不能移动的 token，例如 special tokens。
3. 生成一个 secret permutation：

```text
perm_vocab[i] = τ(i)
inv_perm_vocab[τ(i)] = i
```

4. 改写 embedding：

```text
W_embed_tilde[τ(i)] = W_embed[i]
```

5. 改写 lm head：

```text
W_head_tilde[τ(i)] = W_head[i]
```

6. 在线输入映射：

```python
mapped_ids = perm_vocab[input_ids]
```

7. 输出 logits 恢复：

```python
restored_logits = logits_perm.index_select(dim=-1, index=perm_vocab)
```

### 对应代码

- `src/key_manager.py`
- `src/transforms.py`
- `src/obfuscate_embed_head.py`
- `src/stage_b.py::prepare_stage_a_model`
- `scripts/run_permuted_eval.py`

### 成功标准

如果只做 Stage A，模型结构不应改变，核心判断是：

```text
明文模型在 plain ids 上的输出
≈
置换模型在 mapped ids 上输出、再 restore 后的输出
```

也就是：

```text
baseline logits ≈ restore_logits(permuted logits)
生成 token/text 尽量一致
```

---

## 4. Stage B：hidden-space 入口与 block0 attention

### 对应论文部分

- Section 4.2：composition theorems
- Section 5.2.3：Attention Obfuscation

### 目标

Stage A 只混 token，Stage B 开始混 hidden state。

也就是引入：

```text
h_obf = h P
h_restore = h_obf Q
```

其中 `P` 是 hidden transform，`Q` 是 inverse transform。

### 需要计算什么

1. 生成 hidden transform：

```text
perm_hidden
scale_hidden
inverse_hidden
```

2. 对 block0 attention 的输入做 hidden transform。
3. 改写 attention 的 `q_proj / k_proj / v_proj / o_proj`，使其适配混淆 hidden。
4. 记录中间状态，用 recorder 对比：

```text
baseline block0 attention output
restored obfuscated block0 attention output
```

### 对应代码

- `src/hidden_keys.py`
- `src/stage_b.py`
- `scripts/run_stage_b_hidden_only.py`
- `scripts/run_stage_b_block0_attn_wrapper.py`
- `scripts/run_stage_b_block0_attn_fused.py`

### 成功标准

block0 attention 子层输出在恢复后应和明文 attention 输出接近。

---

## 5. Stage C：block0 完整恢复

### 对应论文部分

- Section 5.2.3：Attention Obfuscation
- Section 5.2.4：FFN Obfuscation
- RMSNorm / LayerNorm 混淆部分

### 目标

不再只验证 attention，而是在一个完整 Transformer block 内验证：

```text
RMSNorm
Attention
Residual
RMSNorm
FFN
Residual
```

这些都能在混淆空间中正确传播。

### 需要计算什么

1. 对 input RMSNorm 做混淆适配。
2. 对 attention 做混淆适配。
3. 对 post-attention RMSNorm 做混淆适配。
4. 对 FFN 的 `gate_proj / up_proj / down_proj` 做 permutation + scaling。
5. 对 block0 的最终输出做恢复比较。

### FFN 的关键计算

原始 FFN：

```text
gate = x Wgate
up = x Wup
hidden = act(gate) * up
out = hidden Wdown
```

混淆时不能随便做任意矩阵，因为 `act` 和逐元素乘法要求中间维度对齐。

所以复现中使用：

```text
中间维度 permutation
中间维度 scaling
```

同步改写：

```text
Wgate
Wup
Wdown
```

让：

```text
restored FFN output ≈ baseline FFN output
```

### 对应代码

- `src/obfuscate_rmsnorm.py`
- `src/obfuscate_ffn.py`
- `src/stage_c.py`
- `scripts/run_stage_c_block0_full.py`

### 成功标准

block0 完整输出在恢复后接近 baseline。

---

## 6. Stage D：多层 block 传播

### 对应论文部分

- Section 4.2：Sequential Composition Theorem
- Section 4.2：Summation Composition Theorem

### 目标

验证论文中的组合定理能否在工程里落地：

```text
单个 block 可恢复
-> 多个 block 串联后仍可恢复
```

### 需要计算什么

1. 选择前 `k` 层，例如 2、4、8、full。
2. 对每一层重复 Stage C 的 norm / attention / FFN 混淆。
3. 确保相邻层的 hidden-space 边界一致。
4. 比较最终 logits 或各层 hidden state。

### 对应代码

- `src/stage_d.py`
- `scripts/run_stage_d_layers.py`
- `scripts/run_stage_d_layers_2.py`
- `scripts/run_stage_d_layers_4.py`
- `scripts/run_stage_d_layers_8.py`
- `scripts/run_stage_d_layers_full.py`

### 成功标准

多层传播误差不应爆炸，生成行为应尽量保持。

---

## 7. Stage E：复杂 attention 结构

### 对应论文部分

- Section 5.2.3：Attention Obfuscation
- Algorithm 2：Intra-head attention obfuscation

### 目标

接入论文里更复杂的 attention 混淆，而不是只做简单 hidden transform。

### 需要计算什么

论文 attention 混淆大致包含两类：

1. Intra-head transformation
2. Inter-head permutation

项目中具体表达为：

```text
R_hat_qk
H_hat_qk
Z_hat_block
tau_kv
tau_group
RoPE side rotation / scaling
```

对于 Qwen 这类 GQA 模型，还要处理：

```text
num_attention_heads
num_key_value_heads
query heads 与 kv heads 的分组关系
```

### Attention 原始计算

```text
q = x Wq
k = x Wk
v = x Wv
score = softmax(q k^T / sqrt(d))
out = score v Wo
```

### Attention 混淆计算

目标是让混淆后的：

```text
q_tilde
k_tilde
v_tilde
out_tilde
```

在经过恢复后满足：

```text
restore(out_tilde) ≈ out
```

同时让攻击者难以从：

```text
Wq_tilde, Wk_tilde, Wv_tilde, Wo_tilde
```

恢复原始 head 对齐和 token permutation。

### 对应代码

- `src/attention_keys.py`
- `src/gqa_layout.py`
- `src/obfuscate_attention_complex.py`
- `src/stage_e.py`
- `scripts/run_stage_e_block0_attention_complex.py`
- `scripts/run_stage_e_ablation.py`
- `scripts/run_stage_e_head_trace_check.py`

### 成功标准

复杂 attention 模块在 block0 或 prefix layers 上能恢复输出，并且 RoPE / GQA 布局没有错位。

---

## 8. Stage F：KeyMat 原生体系接入

### 对应论文部分

- Section 5.2.1：Key Matrix Generation
- Algorithm 1：Key Matrix Generation

### 目标

接入论文里的 Key Matrix / inverse Key Matrix。

### 需要计算什么

论文 Algorithm 1 生成：

```text
P_hat
Q_hat
```

理想关系：

```text
P_hat Q_hat ≈ I
```

它们用于把 hidden state 搬到秘密坐标系：

```text
h_tilde = h P_hat
```

并在相邻模块之间抵消：

```text
h ≈ h_tilde Q_hat
```

### 复现中的具体计算

1. 根据 hidden size `d`、expansion size `h`、系数 `lambda` 生成 KeyMat。
2. 对 embedding/head 接入 KeyMat side transform。
3. 对 norm 估计 kappa。
4. 对 attention/FFN 接入 KeyMat 变换。
5. 在 block0、prefix layers、full layers 上逐步回归。

### 对应代码

- `src/keymat.py`
- `src/keymat_embed_head.py`
- `src/keymat_norm.py`
- `src/keymat_ffn.py`
- `src/keymat_attention_bridge.py`
- `src/stage_f.py`
- `scripts/run_stage_f_keymat_unit.py`
- `scripts/run_stage_f_embed_head.py`
- `scripts/run_stage_f_block0.py`
- `scripts/run_stage_f_prefix_layers.py`
- `scripts/run_stage_f_full_layers.py`

### 成功标准

KeyMat 本身应满足维度和近似逆关系；接入后模型输出不应严重偏离 baseline。

---

## 9. Stage G：KeyMat 融合化 / 去 bridge 化

### 对应论文部分

- Section 5.2.1-5.2.5：各组件权重改写
- Section 5.3：保持标准在线推理

### 目标

Stage F 可能依赖 runtime bridge，也就是运行时显式做：

```text
P / Q 转换
```

但论文希望在线推理尽量像普通模型一样运行。

Stage G 的目标是把这些 bridge 尽量吸收到权重中：

```text
runtime transform
-> offline weight rewrite
```

### 需要计算什么

1. 对 norm 做融合。
2. 对 FFN 做融合。
3. 对 attention 做融合。
4. 导出更像普通模型的 obfuscated artifact。

### 对应代码

- `src/stage_g_norm.py`
- `src/stage_g_ffn.py`
- `src/stage_g_attention.py`
- `src/stage_g.py`
- `src/stage_g_artifact.py`
- `scripts/run_stage_g_regression.py`
- `scripts/export_stage_g_model.py`
- `scripts/infer_stage_g_model.py`

### 成功标准

减少 runtime custom logic，同时保持 correctness。

---

## 10. Stage H：可部署混淆表达整理

### 对应论文部分

- Section 5.2：全部组件混淆
- 论文中的工业化约束：兼容现有推理框架

### 目标

不是所有数学上可写的混淆都能直接部署。

Stage H 要回答：

```text
哪些混淆表达可以在不改变标准 Transformer 运行图的情况下保留？
```

### 当前保留表达

项目文档中保留的可部署表达包括：

```text
embedding/head:
  token permutation
  key-matrix side transform
  noise terms

attention:
  block permutation
  head/group diversity
  RoPE side rotation and scaling profile

FFN:
  component-specific transform
  per-layer diversity

norm:
  kappa correction
  offline fusion preference
```

### 对应代码

- `src/stage_h.py`
- `src/stage_h_attention_static.py`
- `src/stage_h_noise.py`
- `src/stage_h_artifact.py`
- `src/stage_h_pretrained.py`
- `scripts/export_stage_h_pretrained.py`

### 成功标准

形成可供 Stage I/J/K 继续使用的可部署表达清单和工件。

---

## 11. Stage I：标准 HF/vLLM 部署边界验证

### 对应论文部分

- Section 5.3：Online inference
- 论文对软件兼容性的要求：vLLM / SGLang / 标准推理框架

### 目标

验证混淆模型是否还能被标准推理框架加载。

关键约束：

```text
标准 Transformer graph
标准 HF checkpoint layout
不引入在线 custom operator
client 只做轻量 token mapping
```

### 需要计算什么

1. 导出 server 侧文件：

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

2. 导出 client 侧 secret：

```text
client_secret.pt
  perm_vocab
  inv_perm_vocab
```

3. 验证 server 目录能被：

```python
AutoTokenizer.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
```

正常加载。

### 对应代码

- `src/stage_i_vllm.py`
- `src/stage_i_square.py`
- `scripts/export_stage_i_vllm_checkpoint.py`
- `scripts/run_stage_i_hf_regression.py`
- `scripts/run_stage_i_vllm_regression.py`

### 成功标准

标准 HF 加载器能加载 server 目录，client secret 能恢复 token/logits。

---

## 12. Stage J：标准 shape / 论文一致候选

### 对应论文部分

- Section 5.2：组件混淆
- Section 5.4：correctness analysis

### 目标

形成真正可交付的标准 checkpoint 候选。

对于 Qwen 当前主线，目标是：

```text
artifacts/stage_j_qwen_paper_consistent
```

它应满足：

```text
标准 Transformer 运行图
标准 model.* / lm_head.* 权重键
尽量保留论文 attention / FFN / norm / embedding / head 扰动表达
```

### 需要计算什么

1. 从 Stage H 或 redesign artifact 生成标准可见候选。
2. 检查权重键是否是标准 HF layout。
3. 检查 attention / FFN / norm 的表达是否在导出 manifest 中可见。
4. 跑 bridge / correctness regression。
5. 写出 evidence bundle：

```text
outputs/stage_j/paper_consistent/standard_weight_proof.json
outputs/stage_j/paper_consistent/attention_export_visible_proof.json
outputs/stage_j/paper_consistent/ffn_export_visible_proof.json
outputs/stage_j/paper_consistent/norm_export_visible_proof.json
outputs/stage_j/paper_consistent/correctness_regression.json
outputs/stage_j/paper_consistent/completion_summary.json
```

### 对应代码

- `src/stage_j_paper_consistent.py`
- `src/stage_j_standard_bridge.py`
- `src/stage_j_standard_weight_proof.py`
- `scripts/export_stage_j_paper_consistent_checkpoint.py`
- `scripts/run_stage_j_paper_consistent_completion.py`

### 成功标准

`completion_summary.json` 中达到：

```text
completion_status = export_visible_complete
```

---

## 13. Stage K：release 包装与最终推理入口

### 对应论文部分

- Section 5.3：在线推理流程
- Section 6：实验评测入口

### 目标

把 Stage J 候选工件包装成最终可运行 release。

Qwen 当前 release：

```text
artifacts/stage_k_release
```

Llama 当前 release：

```text
artifacts/stage_k_llama_release
```

### 需要计算什么

1. 建立 release catalog：

```text
catalog.json
```

2. 建立 deployment contract：

```text
deployment_contract.json
```

3. 建立 profile：

Qwen：

```text
default
reference
```

Llama：

```text
stable_reference
tiny_a
```

4. 保留 client/server 分离结构：

```text
profiles/<profile>/server
profiles/<profile>/client/client_secret.pt
```

5. 提供统一推理入口：

```text
scripts/infer_stage_k_release.py
```

### 对应代码

- `src/stage_k_release.py`
- `src/stage_k_llama_release.py`
- `scripts/export_stage_k_release.py`
- `scripts/export_stage_k_llama_release.py`
- `scripts/infer_stage_k_release.py`

### 成功标准

能执行：

```powershell
python scripts/infer_stage_k_release.py --release-dir artifacts/stage_k_release --profile default --prompt "请用一句话介绍你自己。" --max-new-tokens 8
```

并输出文本。

---

## 14. 在线推理时的完整计算

以 Stage K release 为例，推理循环如下：

### 14.1 client 编码 prompt

```text
prompt -> tokenizer -> plain input_ids
```

### 14.2 client 映射 input ids

```python
mapped_ids = map_input_ids(input_ids, perm_vocab)
```

等价于：

```text
mapped_ids[i] = τ(input_ids[i])
```

### 14.3 server 侧普通 forward

server 不需要知道明文 token：

```python
logits_perm = model(input_ids=mapped_ids, attention_mask=attention_mask).logits
```

### 14.4 client 恢复 logits

```python
restored_logits = restore_logits(logits_perm, perm_vocab)
```

### 14.5 client 选择下一个明文 token

```python
next_token = argmax(restored_logits[-1])
```

### 14.6 自回归循环

```text
plain ids append next_token
再次 map_input_ids
再次 server forward
再次 restore logits
```

最终得到明文输出。

---

## 15. correctness 检查实际在验证什么

correctness 不是评测模型聪不聪明，而是验证：

```text
导出的 release 工件是否保持了上游参考工件的行为
```

主要计算：

```text
baseline / source logits
stage_k logits
restore stage_k logits
比较误差
```

指标包括：

```text
full_logits_max_abs_error
full_logits_mean_abs_error
last_token_logits_max_abs_error
last_token_logits_mean_abs_error
greedy_first_token_match
generated_ids_exact_match
generated_text_exact_match
NaN / Inf 检查
```

对应代码：

```text
src/stage_k_correctness.py
scripts/run_stage_k_release_correctness.py
```

---

## 16. 安全评测对应论文哪部分

### 对应论文部分

- Section 5.5：security analysis
- Section 6：privacy evaluation
- Appendix attack descriptions

### 项目中的攻击

| 攻击 | 论文含义 | 项目目标 |
| --- | --- | --- |
| `VMA` | Vocabulary-Matching Attack | 从明文/混淆权重关系恢复 token permutation |
| `IMA` | Inversion Model Attack | 训练反演模型，从混淆 embedding/hidden 恢复 token |
| `ISA` | Internal State Attack | 利用 hidden state 或 attention score 反推输入 |
| `IA` | Invariant Attack | 利用权重统计不变量恢复映射 |
| `TFMA / SDA` | 频率或统计侧攻击 | 检查 deterministic token mapping 带来的泄露 |

### 对应代码

```text
src/security_qwen/
scripts/security_qwen/
```

### Qwen 当前主线结论

当前文档记录的 Stage K release 面结果是：

```text
VMA: low risk
IMA paper_like: low risk
ISA hidden_state: low risk
ISA attention_score: low risk
```

但这不等于完全复现论文所有大规模实验，因为当前本地 public corpus、模型规模、混淆参数和论文默认实验仍有差异。

---

## 17. 如果要完整复现论文，推荐执行顺序

### 第一步：跑明文模型

目的：确认模型、tokenizer、环境可用。

```text
infer_qwen.py
run_llama_baseline_smoke.py
```

### 第二步：跑 Stage A

目的：确认 token permutation + embedding/head 置换闭环。

### 第三步：跑 Stage B-D

目的：确认 hidden transform、block0、多层传播的 correctness。

### 第四步：跑 Stage E

目的：确认复杂 attention 表达没有维度错位、RoPE 错位、GQA 分组错位。

### 第五步：跑 Stage F

目的：接入 Algorithm 1 KeyMat，并验证 KeyMat 与各组件组合。

### 第六步：跑 Stage G-H

目的：把 KeyMat / attention / FFN / norm 表达整理成可部署路线。

### 第七步：跑 Stage I

目的：导出标准 HF checkpoint，验证 server/client 分离。

### 第八步：跑 Stage J

目的：形成标准 shape、标准键布局、论文一致候选工件。

### 第九步：跑 Stage K

目的：形成 release 目录和统一推理入口。

### 第十步：跑 correctness

目的：证明 release 工件没有在包装过程中破坏行为。

### 第十一步：跑 security

目的：复现论文隐私评测方向，包括 VMA / IMA / ISA。

---

## 18. Qwen 和 Llama 当前进展差异

### Qwen

Qwen 当前是最完整主线：

```text
Stage J 已到 paper_consistent
Stage K 已有唯一 release surface
correctness 已补齐
VMA / IMA / ISA 已在 Stage K 面上补齐
```

但仍需注意：

```text
尚未完全同态复现论文默认 public corpus、论文规模模型和全部推荐参数
```

### Llama

Llama 当前是可交付部署线：

```text
LlamaArchitectureAdapter 已接入
Stage I/J 标准形状导出链路成立
stable_reference / tiny_a profile 已形成
Stage K Llama release 已成立
```

但 Llama 还没有达到 Qwen 那种 paper_consistent 论文一致闭环：

```text
安全攻击评测未完整补齐
论文同口径叙事弱于 Qwen
```

---

## 19. 一句话总括

本项目的复现路线可以理解为：

```text
先证明 token permutation 能闭环
再证明 hidden-space 协变变换能在 block 内闭环
再把 attention / FFN / norm / KeyMat 全部接入
再判断哪些表达能标准部署
最后导出标准 HF release，并用 correctness 与 security 证据验收
```

也就是把论文中的：

```text
协变混淆理论
组件级权重混淆
在线私有推理
正确性证明
隐私攻击评测
```

逐步落成：

```text
可运行代码
可导出工件
可推理 release
可验证 JSON 证据
```
