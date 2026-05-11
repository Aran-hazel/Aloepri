from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.defaults import DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_i_vllm import export_stage_i_vllm_checkpoint, summarize_token_partitions
from src.stage_j_block0 import build_stage_j_square_model
from src.stage_j_standard_weight_proof import build_stage_j_standard_weight_proof


DEFAULT_LLAMA_INSTRUCT_MODEL_DIR = "model/Llama-3.2-3B-Instruct"
DEFAULT_LLAMA_PAPER_CONSISTENT_DIR = "artifacts/stage_j_llama_instruct_paper_consistent"


def build_stage_j_llama_paper_consistent_target() -> dict[str, Any]:
    return {
        "stage": "J",
        "model_line": "llama_3_2_3b_instruct",
        "goal": "paper_consistent_standard_deployable_obfuscated_checkpoint",
        "standard_graph_required": True,
        "standard_visible_keys_required": True,
        "canonical_candidate_dir": DEFAULT_LLAMA_PAPER_CONSISTENT_DIR,
        "release_surface": "artifacts/stage_k_llama_release",
        "active_profiles": ["default", "reference"],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_stage_j_llama_paper_consistent_checkpoint(
    export_dir: str | Path = DEFAULT_LLAMA_PAPER_CONSISTENT_DIR,
    *,
    model_dir: str | Path = DEFAULT_LLAMA_INSTRUCT_MODEL_DIR,
    seed: int = DEFAULT_SEED,
    dtype: str = "bfloat16",
    device: str = "cpu",
    alpha_e: float = 0.02,
    alpha_h: float = 0.01,
) -> dict[str, Path]:
    """
    Export the active Llama-3.2-3B-Instruct Stage-J candidate.

    This follows the Qwen paper-consistent line shape: one canonical Stage-J
    source artifact, standard HF key layout, and downstream Stage-K
    default/reference profiles that both resolve to this candidate. The current
    Llama implementation keeps the deployment-safe square standard-shape
    transform and records the paper-consistent target explicitly in the manifest.
    """
    export_dir = Path(export_dir)
    model_dir = Path(model_dir)
    set_global_seed(seed)

    tokenizer, baseline_model = load_model_and_tokenizer(str(model_dir), device=device, dtype=dtype)
    model_type = str(getattr(baseline_model.config, "model_type", "")).lower()
    if model_type != "llama":
        raise ValueError(f"Expected a Llama-family checkpoint with model_type='llama', got {model_type!r}")

    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=adapted_layers,
        seed=seed,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
    )
    metadata = {
        "stage": "J",
        "track": "paper_consistent_candidate",
        "model_line": "llama_3_2_3b_instruct",
        "variant": "llama_instruct_paper_consistent_standard_shape_full",
        "goal": "paper_consistent_standard_deployable_obfuscated_checkpoint",
        "model_dir": str(model_dir),
        "seed": seed,
        "dtype": dtype,
        "device": device,
        "alpha_e": alpha_e,
        "alpha_h": alpha_h,
        "adapted_layers": adapted_layers,
        "transform_family": "square_monomial_standard_shape",
        "transform_dim": int(transform.dim),
        "movable_token_count": int(ordinary_token_ids(tokenizer).numel()),
        **summarize_token_partitions(
            tokenizer=tokenizer,
            model_vocab_size=stage_model.get_input_embeddings().weight.shape[0],
            perm_vocab=perm_vocab,
        ),
    }
    paths = export_stage_i_vllm_checkpoint(
        export_dir,
        tokenizer=tokenizer,
        stage_a_model=stage_model,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        metadata=metadata,
    )

    manifest_path = export_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest.update(
        {
            **build_stage_j_llama_paper_consistent_target(),
            "materialized": True,
            "server_dir": "server",
            "client_dir": "client",
            "model_dir": str(model_dir),
            "track": "paper_consistent_candidate",
            "candidate_role": "canonical_stage_j_acceptance_target",
            "standard_weight_proof": build_stage_j_standard_weight_proof(export_dir / "server"),
            "export_visible_components": {
                "embedding_head": {
                    "token_permutation": True,
                    "noise_terms_present": True,
                    "alpha_e": alpha_e,
                    "alpha_h": alpha_h,
                },
                "attention": {
                    "standard_graph_preserved": True,
                    "transform_family": "square_monomial_standard_shape",
                    "adapted_layers_count": len(adapted_layers),
                },
                "ffn": {
                    "standard_graph_preserved": True,
                    "transform_family": "square_monomial_standard_shape",
                    "adapted_layers_count": len(adapted_layers),
                },
                "norm": {
                    "standard_rmsnorm_shape_preserved": True,
                    "strategy": "square_monomial_weight_permutation",
                },
            },
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(export_dir / "paper_consistent_target.json", build_stage_j_llama_paper_consistent_target())
    return paths
