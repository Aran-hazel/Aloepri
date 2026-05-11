#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/nss-d/Aloepri}"
MODEL_DIR="${MODEL_DIR:-/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B-Instruct}"
CONDA_ENV="${CONDA_ENV:-aloepri}"
DTYPE="${DTYPE:-bfloat16}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
INFER_DEVICE="${INFER_DEVICE:-cuda}"
SEED="${SEED:-20260323}"

cd "$REPO_DIR"

echo "[1/6] Llama Instruct baseline smoke"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_llama_baseline_smoke.py \
  --model-dir "$MODEL_DIR" \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --output-path outputs/llama_instruct_baseline_smoke.json

echo "[2/6] export Stage I Llama Instruct checkpoint"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/export_stage_i_llama_real_checkpoint.py \
  --model-dir "$MODEL_DIR" \
  --export-dir artifacts/stage_i_llama_instruct \
  --dtype "$DTYPE" \
  --device "$EXPORT_DEVICE" \
  --seed "$SEED"

echo "[3/6] Stage I Llama Instruct artifact sanity"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_stage_i_artifact_sanity.py \
  --model-dir "$MODEL_DIR" \
  --server-dir artifacts/stage_i_llama_instruct/server \
  --client-secret artifacts/stage_i_llama_instruct/client/client_secret.pt \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/stage_i_llama/instruct_artifact_sanity.json

echo "[4/6] Stage I Llama Instruct remote validation"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_llama_remote_validation.py \
  --baseline-model-dir "$MODEL_DIR" \
  --server-dir artifacts/stage_i_llama_instruct/server \
  --client-secret artifacts/stage_i_llama_instruct/client/client_secret.pt \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/stage_i_llama/instruct_remote_validation.json

echo "[5/6] export Stage J Llama Instruct paper-consistent checkpoint"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/export_stage_j_llama_paper_consistent_checkpoint.py \
  --model-dir "$MODEL_DIR" \
  --export-dir artifacts/stage_j_llama_instruct_paper_consistent \
  --dtype "$DTYPE" \
  --device "$EXPORT_DEVICE" \
  --seed "$SEED" \
  --alpha-e 0.02 \
  --alpha-h 0.01

echo "[6/6] Stage J Llama Instruct remote validation"
conda run --no-capture-output -n "$CONDA_ENV" python scripts/run_llama_remote_validation.py \
  --baseline-model-dir "$MODEL_DIR" \
  --server-dir artifacts/stage_j_llama_instruct_paper_consistent/server \
  --client-secret artifacts/stage_j_llama_instruct_paper_consistent/client/client_secret.pt \
  --device "$INFER_DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED" \
  --output-path outputs/stage_j_llama/instruct_remote_validation.json

echo "Done. Outputs are under:"
echo "  $REPO_DIR/outputs/"
echo "Artifacts are under:"
echo "  $REPO_DIR/artifacts/stage_i_llama_instruct"
echo "  $REPO_DIR/artifacts/stage_j_llama_instruct_paper_consistent"
