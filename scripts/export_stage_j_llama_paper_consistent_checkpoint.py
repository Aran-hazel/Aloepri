from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage_j_llama_paper_consistent import (
    DEFAULT_LLAMA_INSTRUCT_MODEL_DIR,
    DEFAULT_LLAMA_PAPER_CONSISTENT_DIR,
    export_stage_j_llama_paper_consistent_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the active Llama-3.2-3B-Instruct Stage-J paper-consistent candidate."
    )
    parser.add_argument("--model-dir", default=DEFAULT_LLAMA_INSTRUCT_MODEL_DIR)
    parser.add_argument("--export-dir", default=DEFAULT_LLAMA_PAPER_CONSISTENT_DIR)
    parser.add_argument("--seed", type=int, default=20260323)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha-e", type=float, default=0.02)
    parser.add_argument("--alpha-h", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = export_stage_j_llama_paper_consistent_checkpoint(
        args.export_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
    )
    print(f"Exported Llama paper-consistent Stage-J candidate to {paths['export_dir']}")
    print(f"Server: {paths['server_dir']}")
    print(f"Client secret: {paths['client_secret_path']}")


if __name__ == "__main__":
    main()
