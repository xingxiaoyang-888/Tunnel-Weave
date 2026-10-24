"""Qwen visual-sequence concatenation used by the semantic-spatial model.

Retain original visual tokens and append equally sized aligned tokens immediately
after that visual block. Aligned tokens reuse the corresponding visual 3D rotary
coordinates. Causal order remains sequence order. The decoder must run without a
KV cache, and gather_original must restore original positions before language
loss or SEG extraction. This is the TunnelWeave sequence adaptation.
"""
from dataclasses import dataclass

import torch


@dataclass
class SplicedVisualSequence:
    embeddings: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    original_to_expanded: torch.Tensor
    aligned_positions: torch.Tensor

    def gather_original(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[:2] != self.embeddings.shape[:2]:
            raise ValueError("expanded hidden states do not match the spliced sequence")
        return hidden.index_select(1, self.original_to_expanded)


def append_aligned_visual_tokens(
    embeddings: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_mask: torch.Tensor,
    aligned_tokens: torch.Tensor,
    *,
    use_cache: bool = False,
) -> SplicedVisualSequence:
    """Splice one unpadded image's features without changing original token IDs.

    The original Qwen placeholder check and 3D-position calculation must already
    have succeeded. This helper handles neither raw images nor ground-truth masks.
    """
    if use_cache:
        raise ValueError("cached decoding is not implemented for this position policy")
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise ValueError("one image/sample per call is required")
    _, length, dim = embeddings.shape
    if position_ids.shape != (3, 1, length) or position_ids.dtype != torch.long:
        raise ValueError("expected Qwen 3D long position IDs")
    if attention_mask.shape != (1, length) or not bool((attention_mask == 1).all()):
        raise ValueError("only unpadded full-prefix attention is supported")
    if image_token_mask.shape != (1, length) or image_token_mask.dtype != torch.bool:
        raise ValueError("image-token mask must be Boolean [1,L]")
    tensors = (position_ids, attention_mask, image_token_mask, aligned_tokens)
    if any(t.device != embeddings.device for t in tensors):
        raise ValueError("all splice tensors must share a device")
    positions = image_token_mask[0].nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise ValueError("image placeholder block is empty")
    begin, end = int(positions[0]), int(positions[-1]) + 1
    count = positions.numel()
    if end - begin != count:
        raise ValueError("a single contiguous image placeholder block is required")
    if aligned_tokens.shape != (1, count, dim):
        raise ValueError("aligned tokens must match original visual token count/dimension")
    if not bool(torch.isfinite(aligned_tokens).all()):
        raise ValueError("aligned features contain nonfinite values")
    expanded = torch.cat((embeddings[:, :end], aligned_tokens.to(embeddings.dtype),
                          embeddings[:, end:]), dim=1)
    expanded_positions = torch.cat((position_ids[:, :, :end],
                                    position_ids.index_select(2, positions),
                                    position_ids[:, :, end:]), dim=2)
    expanded_attention = torch.ones((1, length + count), device=embeddings.device,
                                    dtype=attention_mask.dtype)
    original_to_expanded = torch.arange(length, device=embeddings.device)
    original_to_expanded[end:] += count
    aligned_positions = torch.arange(end, end + count, device=embeddings.device)
    return SplicedVisualSequence(expanded, expanded_positions, expanded_attention,
                                  original_to_expanded, aligned_positions)
