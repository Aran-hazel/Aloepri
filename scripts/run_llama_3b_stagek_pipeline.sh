#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/nss-d/sf/Aloepri}"
MODEL_DIR="${MODEL_DIR:-/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B-Instruct}"
CONDA_ENV="${CONDA_ENV:-qwen-transformers}"
DTYPE="${DTYPE:-bfloat16}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
SEED="${SEED:-20260323}"

cd "$REPO_DIR"

echo "[1/5] run Llama Instruct baseline smoke"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_llama_baseline_smoke.py \
  --model-dir "$MODEL_DIR" \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/llama_instruct_baseline_smoke.json

echo "[2/5] export Llama Instruct paper-consistent Stage-J candidate"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/export_stage_j_llama_paper_consistent_checkpoint.py \
  --model-dir "$MODEL_DIR" \
  --export-dir artifacts/stage_j_llama_instruct_paper_consistent \
  --dtype "$DTYPE" \
  --device cpu \
  --seed "$SEED" \
  --alpha-e 0.02 \
  --alpha-h 0.01

echo "[3/5] export Llama Stage-K release"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/export_stage_k_llama_release.py \
  --export-dir artifacts/stage_k_llama_release \
  --materialize

echo "[4/5] validate Stage-K release surface"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_stage_k_llama_release_correctness.py \
  --baseline-model-dir "$MODEL_DIR" \
  --release-dir artifacts/stage_k_llama_release \
  --profiles default reference \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-dir outputs/stage_k_llama_release/correctness

echo "[5/5] smoke Stage-K default profile"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile default \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8

echo "Done. Outputs are under:"
echo "  $REPO_DIR/outputs/stage_k_llama_release/"
echo "Artifacts are under:"
echo "  $REPO_DIR/artifacts/stage_j_llama_instruct_paper_consistent"
echo "  $REPO_DIR/artifacts/stage_k_llama_release"
