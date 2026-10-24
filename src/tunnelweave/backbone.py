"""Differentiable Qwen visual injection and SAM2.1 interaction components.

No weights are loaded on import. The production route uses sequence concatenation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import ContractError
from .sequence_concat import append_aligned_visual_tokens


FUSION_MODES = ("residual", "sequence_concat")


class RowWiseSpecialTokens(nn.Module):
    """Shared trainable input/output rows; original vocabulary remains frozen."""
    def __init__(self, base_embedding: nn.Embedding, special_ids: list[int]):
        super().__init__()
        if len(set(special_ids)) != len(special_ids):
            raise ContractError("duplicate special token ids")
        self.register_buffer("ids", torch.tensor(special_ids, dtype=torch.long))
        self.rows = nn.Parameter(base_embedding.weight.detach()[special_ids].clone())

    def embed(self, input_ids: torch.Tensor, frozen_embedding: nn.Embedding) -> torch.Tensor:
        result = frozen_embedding(input_ids)
        for i, token_id in enumerate(self.ids.tolist()):
            result = torch.where((input_ids == token_id)[..., None], self.rows[i].to(result.dtype), result)
        return result

    def logits(self, hidden: torch.Tensor, frozen_head: nn.Module) -> torch.Tensor:
        logits = frozen_head(hidden)
        replacement = F.linear(hidden, self.rows.to(hidden.dtype))
        return logits.scatter(-1, self.ids.expand(*logits.shape[:-1], -1), replacement)


class SpatialCrossAttention(nn.Module):
    def __init__(self, language_dim: int, spatial_dim: int, inner_dim: int = 256, heads: int = 4):
        super().__init__()
        self.q = nn.Linear(language_dim + 2, inner_dim)
        self.kv = nn.Linear(spatial_dim + 2, inner_dim)
        self.attn = nn.MultiheadAttention(inner_dim, heads, batch_first=True)
        self.out = nn.Linear(inner_dim, language_dim)
        self.gate = nn.Parameter(torch.tensor(.001))

    def forward(self, visual_tokens: torch.Tensor, sam_map: torch.Tensor,
                q_grid_hw: tuple[int, int], orig_hw: tuple[int, int],
                mode: str = "residual") -> torch.Tensor:
        """Return Fs + gate*Delta for residual, or pure Delta for concatenation."""
        if mode not in FUSION_MODES:
            raise ContractError(f"unsupported fusion mode: {mode}")
        if visual_tokens.ndim != 3 or sam_map.ndim != 4:
            raise ContractError("expected visual [B,N,D] and SAM [B,C,H,W]")
        h, w = q_grid_hw
        if min(*orig_hw) <= 0:
            raise ContractError("invalid original image geometry")
        if h * w != visual_tokens.shape[1]:
            raise ContractError("Qwen grid does not match visual token count")
        dtype = self.q.weight.dtype
        pooled = F.adaptive_avg_pool2d(sam_map.to(dtype), (16, 16))
        centers = (torch.arange(16, device=pooled.device) + .5) / 16
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        xy = torch.stack((xx, yy), dim=0).expand(pooled.shape[0], -1, -1, -1)
        spatial = torch.cat((pooled, xy), dim=1).flatten(2).transpose(1, 2)
        qy, qx = torch.meshgrid((torch.arange(h, device=visual_tokens.device) + .5) / h,
                                (torch.arange(w, device=visual_tokens.device) + .5) / w, indexing="ij")
        # Both processors consume the same uncropped original frame. Normalize
        # pixel centers through original coordinates before branch resizing.
        q_original_xy = torch.stack((qx * orig_hw[1], qy * orig_hw[0]), dim=-1)
        q_normalized = q_original_xy / torch.tensor([orig_hw[1], orig_hw[0]], device=visual_tokens.device)
        q_normalized = q_normalized.view(1, h * w, 2).expand(visual_tokens.shape[0], -1, -1)
        query = self.q(torch.cat((visual_tokens.to(dtype), q_normalized.to(dtype)), dim=-1))
        key_value = self.kv(spatial)
        fused, _ = self.attn(query, key_value, key_value, need_weights=False)
        delta = self.out(fused)
        if mode == "residual":
            return visual_tokens + (self.gate * delta).to(visual_tokens.dtype)
        return delta.to(visual_tokens.dtype)


def image_token_hidden(model, token_rows: RowWiseSpecialTokens, input_ids: torch.Tensor,
                       pixel_values: torch.Tensor, image_grid_thw: torch.Tensor,
                       spatial_map: torch.Tensor, fusion: SpatialCrossAttention,
                       attention_mask: torch.Tensor | None = None,
                       vision_tokens: torch.Tensor | None = None,
                       orig_hw: tuple[int, int] | None = None,
                       use_interaction: bool = True,
                       fusion_mode: str = "residual",
                       logit_mode: str = "all") -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use Qwen3.5's actual visual features and 3D position IDs, one image per call."""
    if fusion_mode not in FUSION_MODES:
        raise ContractError(f"unsupported fusion mode: {fusion_mode}")
    if logit_mode not in {"all", "last", "none"}:
        raise ContractError("unknown vocabulary projection mode")
    if fusion_mode == "sequence_concat" and not use_interaction:
        raise ContractError("sequence_concat requires enabled visual interaction")
    if image_grid_thw.shape != (1, 3) or input_ids.shape[0] != 1:
        raise ContractError("current implementation requires one image per sample")
    base = model.model
    if vision_tokens is None:
        with torch.no_grad():
            vision = base.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output[0]
    else:
        vision = vision_tokens
    merge = base.visual.spatial_merge_size
    _, gh, gw = image_grid_thw[0].tolist()
    if gh % merge or gw % merge:
        raise ContractError("Qwen grid incompatible with spatial merge")
    if orig_hw is None:
        raise ContractError("original image size required for two-branch coordinate mapping")
    aligned = fusion(vision[None], spatial_map, (gh // merge, gw // merge),
                     orig_hw, mode=fusion_mode)[0] if use_interaction else vision
    embeddings = token_rows.embed(input_ids, model.get_input_embeddings())
    original_visual = vision if fusion_mode == "sequence_concat" else aligned
    image_mask, _ = base.get_placeholder_mask(input_ids, embeddings, image_features=original_visual)
    embeddings = embeddings.masked_scatter(image_mask, original_visual.to(embeddings.dtype))
    positions = base.compute_3d_position_ids(input_ids=input_ids, image_grid_thw=image_grid_thw,
                                              inputs_embeds=embeddings, attention_mask=attention_mask)
    splice = None
    if fusion_mode == "sequence_concat":
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        splice = append_aligned_visual_tokens(
            embeddings, positions, attention_mask,
            input_ids == base.config.image_token_id, aligned[None], use_cache=False)
        embeddings, positions, attention_mask = (splice.embeddings, splice.position_ids,
                                                  splice.attention_mask)
    outputs = base.language_model(input_ids=None, inputs_embeds=embeddings, position_ids=positions,
                                  attention_mask=attention_mask, use_cache=False, return_dict=True)
    hidden = splice.gather_original(outputs.last_hidden_state) if splice is not None else outputs.last_hidden_state
    # Image/prompt positions carry no language targets. Avoid allocating their
    # full vocabulary logits; hidden states and all attention stay unchanged.
    logits = None if logit_mode == "none" else token_rows.logits(
        hidden[:, -1:] if logit_mode == "last" else hidden, model.lm_head)
    return hidden, logits


def own_token_state(hidden: torch.Tensor, input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    """Read the generated token's own post-transformer state, never its predecessor."""
    if hidden.shape[:2] != input_ids.shape:
        raise ContractError("hidden/input sequence mismatch")
    rows = []
    for b in range(input_ids.shape[0]):
        positions = (input_ids[b] == token_id).nonzero(as_tuple=True)[0]
        rows.append(hidden[b, positions])
    if len({r.shape[0] for r in rows}) != 1:
        raise ContractError("batch has differing numbers of special tokens")
    return torch.stack(rows)


def decode_ordered_entries(parsed: dict[str, Any], positions: torch.Tensor,
                           hidden: torch.Tensor, decode_state) -> dict[str, torch.Tensor]:
    """Keep raw SEG order; merge duplicate class probabilities by max."""
    if len(positions) != parsed["seg_count"]:
        raise ContractError("SEG_class_alignment_failed")
    masks: dict[str, torch.Tensor] = {}
    for entry in parsed["entries"]:
        disease = entry["disease"]
        probability = decode_state(hidden[positions[entry["seg_index"]]])
        masks[disease] = torch.maximum(masks[disease], probability) if disease in masks else probability
    return masks


class PromptProjection(nn.Module):
    def __init__(self, language_dim: int, prompt_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(language_dim, language_dim), nn.GELU(),
                                    nn.Linear(language_dim, prompt_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden.to(self.layers[0].weight.dtype))


@dataclass
class SamFeatures:
    image_embed: torch.Tensor
    high_res_feats: list[torch.Tensor]
    orig_hw: tuple[int, int]
    input_hw: tuple[int, int]


class Sam21Interactive:
    """Official SAM2ImagePredictor encoder; differentiable official mask decoder."""
    def __init__(self, sam_model):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(sam_model)
        self.model = sam_model

    def encode_once(self, image) -> SamFeatures:
        # Official transform uses model.image_size (SAM2.1 Hiera Large: 1024), not SAM3's 1008.
        from PIL import Image
        import numpy as np
        if isinstance(image, Image.Image):
            orig_hw = (image.height, image.width)
        elif isinstance(image, np.ndarray) and image.ndim == 3:
            orig_hw = image.shape[:2]
        else:
            raise ContractError("SAM2.1 expects a PIL RGB image or HWC RGB array")
        transformed = self.predictor._transforms(image)[None].to(next(self.model.parameters()).device)
        with torch.no_grad():
            backbone = self.model.image_encoder(transformed)
        if self.model.use_high_res_features_in_sam:
            # These are decoder parameters; keep their gradient path.
            backbone["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(backbone["backbone_fpn"][0])
            backbone["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(backbone["backbone_fpn"][1])
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        sizes = self.predictor._bb_feat_sizes
        maps = [feat.permute(1, 2, 0).view(1, -1, *shape)
                for feat, shape in zip(vision_feats[::-1], sizes[::-1])][::-1]
        return SamFeatures(maps[-1], maps[:-1], orig_hw,
                           (self.model.image_size, self.model.image_size))

    def decode(self, features: SamFeatures, prompt_embedding: torch.Tensor,
               decoder: nn.Module | None = None) -> torch.Tensor:
        if prompt_embedding.ndim != 3 or prompt_embedding.shape[1] != 1:
            raise ContractError("prompt embedding must be [B,1,C]")
        prompt_encoder = self.model.sam_prompt_encoder
        _, dense = prompt_encoder(points=None, boxes=None, masks=None)
        mask_decoder = decoder if decoder is not None else self.model.sam_mask_decoder
        low_res, _, _, _ = mask_decoder(
            image_embeddings=features.image_embed,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=prompt_embedding,
            dense_prompt_embeddings=dense.expand(prompt_embedding.shape[0], -1, -1, -1),
            multimask_output=False, repeat_image=False,
            high_res_features=features.high_res_feats)
        return F.interpolate(low_res, size=features.orig_hw, mode="bilinear", align_corners=False)


class TunnelScopeModel(nn.Module):
    """One image, one frozen image encoding per branch, differentiable mask prompts."""
    def __init__(self, qwen, sam_model, seg_id: int, nor_id: int, ano_id: int,
                 cross_inner_dim: int = 256, cross_heads: int = 4,
                 cross_gate_initial: float = .001,
                 enable_interaction: bool = True, enable_auxiliary: bool = True,
                 fusion_mode: str = "residual"):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ContractError(f"unsupported fusion mode: {fusion_mode}")
        if fusion_mode == "sequence_concat" and not enable_interaction:
            raise ContractError("sequence_concat requires enabled visual interaction")
        self.qwen = qwen
        self.sam_model = sam_model
        self.sam = Sam21Interactive(sam_model)
        self.token_rows = RowWiseSpecialTokens(qwen.get_input_embeddings(), [seg_id, nor_id, ano_id])
        language_dim = qwen.config.text_config.hidden_size
        spatial_dim = sam_model.hidden_dim
        prompt_dim = sam_model.sam_prompt_embed_dim
        self.fusion = SpatialCrossAttention(language_dim, spatial_dim,
                                            cross_inner_dim, cross_heads)
        with torch.no_grad():
            self.fusion.gate.fill_(cross_gate_initial)
        self.main_projection = PromptProjection(language_dim, prompt_dim)
        self.aux_projection = PromptProjection(language_dim, prompt_dim)
        import copy
        self.aux_decoder = copy.deepcopy(sam_model.sam_mask_decoder)
        self.seg_id, self.nor_id, self.ano_id = seg_id, nor_id, ano_id
        self.enable_interaction = enable_interaction
        self.enable_auxiliary = enable_auxiliary
        self.fusion_mode = fusion_mode
        for p in sam_model.parameters():
            p.requires_grad_(False)
        for p in qwen.model.visual.parameters():
            p.requires_grad_(False)
        self.set_training_stage("A")

    def forward(self, image, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                image_grid_thw: torch.Tensor, attention_mask: torch.Tensor | None = None,
                decode_masks: bool = True, decode_aux: bool = True,
                logit_mode: str = "all") -> dict[str, Any]:
        features = self.sam.encode_once(image)
        hidden, logits = image_token_hidden(self.qwen, self.token_rows, input_ids, pixel_values,
                                            image_grid_thw, features.image_embed, self.fusion,
                                            attention_mask, orig_hw=features.orig_hw,
                                            use_interaction=self.enable_interaction,
                                            fusion_mode=self.fusion_mode, logit_mode=logit_mode)
        result: dict[str, Any] = {"hidden": hidden, "logits": logits, "main_masks": [], "aux_logits": None,
                                  "geometry": {"original_hw": features.orig_hw,
                                               "sam_input_hw": features.input_hw,
                                               "qwen_grid_thw": image_grid_thw[0].tolist(),
                                               "coordinate_convention": "full_frame_normalized_pixel_centers"}}
        if decode_masks:
            seg_states = own_token_state(hidden, input_ids, self.seg_id)
            for state in seg_states[0]:
                prompt = self.main_projection(state)[None, None]
                result["main_masks"].append(self.sam.decode(features, prompt))
            if decode_aux and self.enable_auxiliary:
                nor = own_token_state(hidden, input_ids, self.nor_id)
                ano = own_token_state(hidden, input_ids, self.ano_id)
                if nor.shape[1] != 1 or ano.shape[1] != 1:
                    raise ContractError("exactly one [NOR] and [ANO] anchor required")
                aux = []
                for state in (nor[0, 0], ano[0, 0]):
                    prompt = self.aux_projection(state)[None, None]
                    aux.append(self.sam.decode(features, prompt, self.aux_decoder))
                result["aux_logits"] = torch.cat(aux, dim=1)
        return result

    @torch.inference_mode()
    def generate(self, image, prefix_ids: torch.Tensor, pixel_values: torch.Tensor,
                 image_grid_thw: torch.Tensor, tokenizer, active_classes: list[str],
                 max_new_tokens: int = 128,
                 return_aux_probabilities: bool = False) -> dict[str, Any]:
        """Deterministic full-prefix generation; caches are deliberately unused.

        This is slower than incremental caching but keeps Qwen3.5 attention/DeltaNet
        state and [SEG] post-token position consistent. Never reads coverage or GT.
        """
        from .parsing import parse_generated
        if prefix_ids.shape[0] != 1:
            raise ContractError("generate supports one image")
        features = self.sam.encode_once(image)
        with torch.no_grad():
            vision = self.qwen.model.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output[0]
        prefix_length = prefix_ids.shape[1]
        ids = prefix_ids
        generated = []
        eos = tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            attention_mask = torch.ones_like(ids, dtype=torch.long)
            hidden, logits = image_token_hidden(self.qwen, self.token_rows, ids, pixel_values,
                                                image_grid_thw, features.image_embed, self.fusion,
                                                attention_mask=attention_mask,
                                                vision_tokens=vision, orig_hw=features.orig_hw,
                                                use_interaction=self.enable_interaction,
                                                fusion_mode=self.fusion_mode, logit_mode="last")
            next_id = int(logits[0, -1].argmax())
            generated.append(next_id)
            ids = torch.cat((ids, torch.tensor([[next_id]], device=ids.device)), dim=1)
            if next_id == eos:
                break
            import re
            if re.search(r"(?:^|\n)END\s*$", tokenizer.decode(generated, skip_special_tokens=False)):
                break
        hidden, _ = image_token_hidden(self.qwen, self.token_rows, ids, pixel_values,
                                       image_grid_thw, features.image_embed, self.fusion,
                                       attention_mask=torch.ones_like(ids, dtype=torch.long),
                                       vision_tokens=vision, orig_hw=features.orig_hw,
                                       use_interaction=self.enable_interaction,
                                       fusion_mode=self.fusion_mode, logit_mode="none")
        display_ids = generated[:-1] if generated and generated[-1] == eos else generated
        raw = tokenizer.decode(display_ids, skip_special_tokens=False)
        parsed = parse_generated(raw, active_classes)
        # The user prefix is never a generated class entry, even if malformed
        # external input contains a literal [SEG] token.
        positions = (ids[0, prefix_length:] == self.seg_id).nonzero(as_tuple=True)[0] + prefix_length
        if self.enable_auxiliary and ((ids[0, prefix_length:] == self.nor_id).any() or
                                      (ids[0, prefix_length:] == self.ano_id).any()):
            parsed["errors"].append("generated_aux_anchor")
            parsed["status"] = "partial_or_failed"
        def decode_state(state):
            prompt = self.main_projection(state)[None, None]
            return self.sam.decode(features, prompt).sigmoid()[0, 0]
        try:
            masks = decode_ordered_entries(parsed, positions, hidden[0], decode_state)
        except ContractError:
            masks = {}
            parsed["errors"].append("SEG_class_alignment_failed")
            parsed["status"] = "partial_or_failed"
        result = {**parsed, "masks": masks, "generated_ids": generated}
        if return_aux_probabilities:
            probability = None
            if self.enable_auxiliary:
                nor_positions = (prefix_ids[0] == self.nor_id).nonzero(as_tuple=True)[0]
                ano_positions = (prefix_ids[0] == self.ano_id).nonzero(as_tuple=True)[0]
                if len(nor_positions) != 1 or len(ano_positions) != 1:
                    raise ContractError("exactly one prefixed [NOR] and [ANO] anchor required for auxiliary inference")
                aux_logits = torch.cat([
                    self.sam.decode(features, self.aux_projection(state)[None, None], self.aux_decoder)
                    for state in (hidden[0, nor_positions[0]], hidden[0, ano_positions[0]])], dim=1)
                probability = aux_logits.float().softmax(dim=1)[0, 1]
            result["aux_abnormal_probability"] = probability
        return result

    def set_training_stage(self, stage: str) -> dict[str, int]:
        if stage not in {"A", "B"}:
            raise ContractError("stage must be A or B")
        for name, p in self.qwen.named_parameters():
            p.requires_grad_(stage == "B" and "lora_" in name)
        for p in self.sam_model.parameters():
            p.requires_grad_(False)
        for name, p in self.sam_model.sam_mask_decoder.named_parameters():
            if not any(x in name for x in ("iou_prediction_head", "pred_obj_score_head", "obj_score_token", "iou_token")):
                p.requires_grad_(True)
        for name, p in self.aux_decoder.named_parameters():
            p.requires_grad_(self.enable_auxiliary and not any(x in name for x in ("iou_prediction_head", "pred_obj_score_head",
                                                       "obj_score_token", "iou_token", "conv_s0", "conv_s1")))
        if not self.enable_auxiliary:
            for p in self.aux_projection.parameters():
                p.requires_grad_(False)
        for p in self.fusion.parameters():
            p.requires_grad_(self.enable_interaction)
        # Keep the shared gate in both state dicts, but concat never uses it.
        self.fusion.gate.requires_grad_(self.enable_interaction and self.fusion_mode == "residual")
        counts = {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                  "total": sum(p.numel() for p in self.parameters())}
        if counts["trainable"] == 0:
            raise ContractError("no trainable parameters in selected stage")
        return counts

    def validate_placement(self, expected_device: str = "cuda") -> dict[str, str]:
        modules = {"qwen_embedding": self.qwen.get_input_embeddings().weight,
                   "sam_image_encoder": next(self.sam_model.image_encoder.parameters()),
                   "fusion": self.fusion.q.weight,
                   "main_projection": self.main_projection.layers[0].weight,
                   "aux_projection": self.aux_projection.layers[0].weight,
                   "token_rows": self.token_rows.rows}
        locations = {name: f"{tensor.device}:{tensor.dtype}" for name, tensor in modules.items()}
        if any(tensor.device.type != expected_device for tensor in modules.values()):
            raise ContractError(f"model components on mixed devices: {locations}")
        return locations


def qwen_lora_targets(qwen) -> list[str]:
    """Exact Qwen3.5 text linear modules, including DeltaNet paths."""
    names = []
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
               "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"}
    for name, module in qwen.named_modules():
        if name.startswith("model.language_model.layers.") and name.rsplit(".", 1)[-1] in allowed and isinstance(module, nn.Linear):
            names.append(name)
    if not names or not any("linear_attn" in n for n in names) or not any("self_attn" in n for n in names):
        raise ContractError("Qwen3.5 language attention module inventory incomplete")
    return names


def attach_qwen_lora(qwen, rank: int = 16, alpha: int = 32,
                     dropout: float = .05):
    from peft import LoraConfig, inject_adapter_in_model
    targets = qwen_lora_targets(qwen)
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, lora_dropout=dropout,
                        bias="none")
    # In-place injection keeps the official Qwen3.5 multimodal API accessible.
    inject_adapter_in_model(config, qwen)
    return qwen, targets
