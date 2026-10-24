"""Official Qwen chat/image preprocessing; assistant-only language labels."""
from __future__ import annotations
from typing import Any
import torch
from PIL import Image
from .contracts import ContractError

def _messages(image: Image.Image, user_text: str, answer: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "image", "image": image},
                                                           {"type": "text", "text": user_text}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def prepare_processor_inputs(processor, image: Image.Image, user_text: str,
                             answer: str | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Official Qwen processor image path; labels cover assistant content only."""
    prefix = processor.apply_chat_template(_messages(image, user_text), tokenize=True,
                                           add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False)
    _check_qwen_pixel_budget(prefix, processor)
    if answer is None:
        return prefix, None
    full = processor.apply_chat_template(_messages(image, user_text, answer), tokenize=True,
                                         add_generation_prompt=False, return_dict=True,
                                         return_tensors="pt", enable_thinking=False)
    _check_qwen_pixel_budget(full, processor)
    prefix_ids = prefix["input_ids"][0]
    full_ids = full["input_ids"][0]
    if not torch.equal(full_ids[:len(prefix_ids)], prefix_ids):
        raise ContractError("Qwen chat prefix differs between input and teacher-forced sequence")
    labels = full["input_ids"].clone()
    labels[:, :len(prefix_ids)] = -100
    return full, labels


def _check_qwen_pixel_budget(inputs, processor) -> None:
    budget = int(processor.image_processor.size["longest_edge"])
    patch = int(processor.image_processor.patch_size)
    for _, h, w in inputs["image_grid_thw"].tolist():
        if h * w * patch * patch > budget + 4 * patch * patch:
            raise ContractError("Qwen processor exceeded locked image pixel budget")
