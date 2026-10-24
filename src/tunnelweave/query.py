"""One fixed image-only query interface, identical in training and inference."""
from __future__ import annotations
import hashlib
import json
import torch
from .contracts import CLASSES, ContractError
from .processing import prepare_processor_inputs

VERSION = "c1_mainline_fixed13_v1"
QUERY_PROMPT = (
    "Inspect the entire image for the following thirteen defect classes: " + ", ".join(CLASSES) +
    ". Evaluate each class independently from visible image evidence. The fixed class slots "
    "are queries, not assertions that defects are present. Do not use annotations or a reference description."
)
QUERY_SLOTS = "[NOR][ANO]\n" + "\n".join(c + " [SEG]" for c in CLASSES) + "\nEND"


def query_contract():
    payload = {"version": VERSION, "classes": list(CLASSES), "prompt": QUERY_PROMPT,
               "slots": QUERY_SLOTS, "gt_in_model_input": False,
               "class_loss": "known_label_masked_BCE", "mask_loss_gated_by_presence": False}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def prepare_fixed_queries(processor, image, device=None):
    # No labels, source, ROI or row are arguments. The assistant suffix is constant
    # for all images, including negative and partially labelled images.
    inputs, ignored_labels = prepare_processor_inputs(processor, image, QUERY_PROMPT, QUERY_SLOTS)
    if ignored_labels is None:
        raise ContractError("fixed slots require a complete, untruncated processor sequence")
    token_ids = {s: processor.tokenizer.convert_tokens_to_ids(s) for s in ("[SEG]", "[NOR]", "[ANO]")}
    if len(set(token_ids.values())) != 3 or processor.tokenizer.unk_token_id in token_ids.values():
        raise ContractError("invalid fixed query special token mapping")
    ids = inputs["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ContractError("fixed13 requires one image per forward")
    for s, count in (("[SEG]", 13), ("[NOR]", 1), ("[ANO]", 1)):
        if int((ids == token_ids[s]).sum()) != count:
            raise ContractError("fixed query slots truncated or special tokens appeared in prompt: " + s)
    # Validate the actual assistant token sequence, not just the count of SEG.
    assistant = ids[ignored_labels != -100].detach().cpu().tolist()
    expected = processor.tokenizer.encode(QUERY_SLOTS, add_special_tokens=False)
    if assistant[:len(expected)] != expected:
        raise ContractError("processor altered fixed class order or query suffix")
    moved = {k: v.to(device) if device is not None and isinstance(v, torch.Tensor) else v
             for k, v in inputs.items()}
    return moved, {"query_contract": query_contract(), "input_tokens": int(ids.shape[1]),
                   "query_SEG_count": 13, "language_targets_for_fixed_slots": 0}
