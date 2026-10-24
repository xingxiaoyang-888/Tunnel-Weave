"""Portable loading of the pinned official backbones and trainable project modules."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .contracts import require


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def load_model(config_path, arm="M_full", device="cuda:0"):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from sam2.build_sam import build_sam2
    from .backbone import TunnelScopeModel, attach_qwen_lora
    from .ablations import PaperModel

    require(arm in {"M_full", "Q_no_semantic_guidance", "I_no_instruction_supervision"}, "Unknown ablation arm")
    require(torch.cuda.is_available() and torch.device(device).type == "cuda", "Full model execution requires CUDA")
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    configure_determinism()
    seed_everything(cfg["seed"])
    qwen_path = Path(os.environ.get("TUNNELWEAVE_QWEN_PATH", cfg["qwen_path"]))
    sam_path = Path(os.environ.get("TUNNELWEAVE_SAM_CHECKPOINT", cfg["sam_checkpoint"]))
    require(qwen_path.is_dir(), "Download the pinned Qwen3.5-9B snapshot first")
    require(file_sha256(sam_path) == cfg["sam_checkpoint_sha256"], "SAM2.1 checkpoint SHA256 mismatch")
    processor = AutoProcessor.from_pretrained(qwen_path, local_files_only=True)
    processor.image_processor.size = {**processor.image_processor.size, "longest_edge": cfg["qwen_max_image_pixels"]}
    tokenizer = processor.tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": cfg["special_tokens"]})
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(qwen_path, local_files_only=True,
                                                         torch_dtype=torch.bfloat16, device_map=device)
    qwen.resize_token_embeddings(len(tokenizer))
    qwen, _ = attach_qwen_lora(qwen, **cfg["lora"])
    sam = build_sam2(cfg["sam_config"], ckpt_path=str(sam_path), device=device,
                     mode="train", apply_postprocessing=False)
    require(sam.image_size == 1024, "Expected the official SAM2.1 1024-pixel encoder")
    token_ids = [tokenizer.convert_tokens_to_ids(x) for x in cfg["special_tokens"]]
    require(len(set(token_ids)) == 3 and tokenizer.unk_token_id not in token_ids, "Invalid special tokens")
    cross = cfg["cross_attention"]
    base = TunnelScopeModel(qwen, sam, *token_ids, cross["inner_dim"], cross["heads"],
                            cross["gate_initial"], fusion_mode="sequence_concat")
    for module in [base.fusion, base.main_projection, base.aux_projection]:
        module.to(device=device, dtype=torch.float32)
    base.token_rows.to(device=device)
    model = PaperModel(base, arm)
    model.set_training_stage("B")
    model.train()
    qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.validate_placement("cuda")
    identity = {"qwen_repo": cfg["qwen_repo"], "qwen_revision": cfg["qwen_revision"],
                "qwen_config_sha256": file_sha256(qwen_path / "config.json"),
                "sam_checkpoint_sha256": cfg["sam_checkpoint_sha256"],
                "special_token_ids": token_ids, "lora": cfg["lora"],
                "fusion_mode": "sequence_concat", "arm": arm}
    return model, processor, identity
