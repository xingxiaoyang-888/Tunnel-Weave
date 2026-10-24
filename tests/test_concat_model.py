"""Tiny CPU contracts: no SAM/Qwen weights, not a model-quality validation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from tunnelweave.backbone import (SamFeatures, SpatialCrossAttention, TunnelScopeModel,
                               image_token_hidden)
from tunnelweave.contracts import ContractError


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None
        self.last_positions = None
        self.last_attention = None

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
                attention_mask=None, use_cache=None, return_dict=None):
        assert input_ids is None and use_cache is False and return_dict is True
        self.last_input = inputs_embeds
        self.last_positions = position_ids
        self.last_attention = attention_mask
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(dim=1))


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = SimpleNamespace(spatial_merge_size=1)
        self.config = SimpleNamespace(image_token_id=9)
        self.language_model = TinyLM()

    def get_placeholder_mask(self, input_ids, embeddings, image_features):
        mask = (input_ids == 9)[..., None].expand_as(embeddings)
        assert int(mask.sum()) == image_features.numel()
        return mask, None

    def compute_3d_position_ids(self, input_ids, image_grid_thw, inputs_embeds,
                                attention_mask=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        return positions.view(1, 1, -1).repeat(3, 1, 1)


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBase()
        self.embedding = nn.Embedding(20, 4)
        self.lm_head = nn.Linear(4, 20, bias=False)
        self.lora_probe = nn.Parameter(torch.ones(1))

    def get_input_embeddings(self):
        return self.embedding


class TinyRows:
    def embed(self, ids, embedding):
        return embedding(ids)

    def logits(self, hidden, lm_head):
        return lm_head(hidden)


class TinySam:
    def encode_once(self, image):
        return SamFeatures(torch.ones(1, 2, 4, 4), [], (4, 4), (4, 4))

    def decode(self, features, prompt_embedding, decoder=None):
        return prompt_embedding[0, 0, 0].expand(1, 1, 4, 4)


class Tokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        assert ids == [5]
        return "NONE\nEND"


class ConcatModelContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.qwen = TinyQwen()
        self.rows = TinyRows()
        self.fusion = SpatialCrossAttention(4, 2, inner_dim=8, heads=2)
        with torch.no_grad():
            self.fusion.gate.fill_(0.5)
        self.ids = torch.tensor([[1, 2, 9, 9, 9, 9, 3, 4]])
        self.grid = torch.tensor([[1, 2, 2]])
        self.vision = torch.randn(4, 4)
        self.spatial = torch.randn(1, 2, 4, 4)

    def run_hidden(self, mode):
        return image_token_hidden(self.qwen, self.rows, self.ids, torch.empty(0),
                                  self.grid, self.spatial, self.fusion,
                                  attention_mask=torch.ones_like(self.ids),
                                  vision_tokens=self.vision, orig_hw=(8, 8),
                                  fusion_mode=mode)

    def test_concat_keeps_fs_appends_pure_delta_and_gathers_suffix(self):
        hidden, logits = self.run_hidden("sequence_concat")
        expanded = self.qwen.model.language_model.last_input
        self.assertEqual(expanded.shape, (1, 12, 4))
        self.assertEqual(hidden.shape, (1, 8, 4))
        self.assertEqual(logits.shape, (1, 8, 20))
        self.assertTrue(torch.allclose(expanded[0, 2:6], self.vision))
        self.assertFalse(torch.equal(expanded[0, 6:10], self.vision))
        delta = self.fusion(self.vision[None], self.spatial, (2, 2), (8, 8),
                            mode="sequence_concat")[0]
        self.assertTrue(torch.allclose(expanded[0, 6:10], delta))
        self.assertFalse(torch.allclose(expanded[0, 6:10],
                                        self.vision + self.fusion.gate * delta))
        self.assertTrue(torch.equal(self.qwen.model.language_model.last_positions[:, :, 6:10],
                                    self.qwen.model.language_model.last_positions[:, :, 2:6]))
        self.assertTrue(torch.all(self.qwen.model.language_model.last_attention == 1))
        hidden[0, -1].sum().backward()
        self.assertIsNone(self.fusion.gate.grad)
        self.assertIsNotNone(self.fusion.out.weight.grad)
        self.assertTrue(torch.isfinite(self.fusion.out.weight.grad).all())
        self.assertGreater(float(self.fusion.out.weight.grad.abs().sum()), 0)

    def test_gate_changes_residual_but_not_concat(self):
        with torch.no_grad():
            self.fusion.gate.fill_(0.0)
        self.run_hidden("sequence_concat")
        concat_zero = self.qwen.model.language_model.last_input.detach().clone()
        self.run_hidden("residual")
        residual_zero = self.qwen.model.language_model.last_input.detach().clone()
        with torch.no_grad():
            self.fusion.gate.fill_(0.8)
        self.run_hidden("sequence_concat")
        concat_nonzero = self.qwen.model.language_model.last_input.detach().clone()
        self.run_hidden("residual")
        residual_nonzero = self.qwen.model.language_model.last_input.detach().clone()
        self.assertTrue(torch.allclose(concat_zero, concat_nonzero))
        self.assertFalse(torch.allclose(residual_zero, residual_nonzero))

    def test_selected_vocabulary_projection_preserves_hidden_and_last_logits(self):
        hidden, logits = self.run_hidden("sequence_concat")
        for mode in ("last", "none"):
            h, z = image_token_hidden(self.qwen, self.rows, self.ids, torch.empty(0),
                self.grid, self.spatial, self.fusion, attention_mask=torch.ones_like(self.ids),
                vision_tokens=self.vision, orig_hw=(8, 8), fusion_mode="sequence_concat", logit_mode=mode)
            torch.testing.assert_close(h, hidden)
            if mode == "none":
                self.assertIsNone(z)
            else:
                torch.testing.assert_close(z, logits[:, -1:])

    def test_residual_uses_original_sequence_length_and_same_fusion_parameters(self):
        hidden, logits = self.run_hidden("residual")
        self.assertEqual(self.qwen.model.language_model.last_input.shape, (1, 8, 4))
        self.assertEqual(hidden.shape, (1, 8, 4))
        self.assertEqual(logits.shape, (1, 8, 20))
        self.assertFalse(torch.equal(self.qwen.model.language_model.last_input[0, 2:6], self.vision))
        self.assertTrue(self.fusion.gate.requires_grad)

    def test_invalid_mode_and_concat_without_interaction_fail(self):
        with self.assertRaises(ContractError):
            self.run_hidden("unknown")
        with self.assertRaisesRegex(ContractError, "requires enabled"):
            image_token_hidden(self.qwen, self.rows, self.ids, torch.empty(0), self.grid,
                               self.spatial, self.fusion, vision_tokens=self.vision,
                               orig_hw=(8, 8), use_interaction=False,
                               fusion_mode="sequence_concat")

    def test_stage_switch_preserves_fusion_gate_policy(self):
        model = TunnelScopeModel.__new__(TunnelScopeModel)
        nn.Module.__init__(model)
        model.qwen = self.qwen
        sam = nn.Module()
        sam.sam_mask_decoder = nn.Linear(4, 4)
        model.sam_model = sam
        model.aux_decoder = nn.Linear(4, 4)
        model.aux_projection = nn.Linear(4, 4)
        model.fusion = self.fusion
        model.enable_interaction = True
        model.enable_auxiliary = False
        model.fusion_mode = "sequence_concat"
        model.set_training_stage("A")
        self.assertFalse(self.fusion.gate.requires_grad)
        self.assertTrue(self.fusion.out.weight.requires_grad)
        self.assertFalse(self.qwen.lora_probe.requires_grad)
        self.assertFalse(any(p.requires_grad for p in model.aux_decoder.parameters()))
        model.set_training_stage("B")
        self.assertTrue(self.qwen.lora_probe.requires_grad)
        self.assertFalse(self.fusion.gate.requires_grad)
        model.fusion_mode = "residual"
        model.set_training_stage("A")
        self.assertTrue(self.fusion.gate.requires_grad)
        model.set_training_stage("B")
        self.assertTrue(self.fusion.gate.requires_grad)

    def test_full_prefix_generate_routes_mode_and_optional_aux_probability(self):
        model = TunnelScopeModel.__new__(TunnelScopeModel)
        nn.Module.__init__(model)
        model.qwen = self.qwen
        self.qwen.model.get_image_features = lambda *a, **kw: SimpleNamespace(pooler_output=[self.vision])
        model.sam = TinySam()
        model.fusion = self.fusion
        model.token_rows = nn.Identity()
        model.main_projection = nn.Identity()
        model.aux_projection = nn.Identity()
        model.aux_decoder = nn.Identity()
        model.seg_id, model.nor_id, model.ano_id = 6, 2, 3
        model.enable_interaction = True
        model.enable_auxiliary = True
        model.fusion_mode = "sequence_concat"
        modes = []

        def stub_hidden(qwen, rows, ids, pixels, grid, spatial, fusion, **kwargs):
            modes.append(kwargs["fusion_mode"])
            hidden = ids.float()[..., None].expand(-1, -1, 4).contiguous()
            logits = torch.zeros(1, ids.shape[1], 10)
            logits[0, -1, 5] = 10
            return hidden, logits

        with patch("tunnelweave.backbone.image_token_hidden", side_effect=stub_hidden):
            default = model.generate(None, torch.tensor([[1, 2, 3, 4]]), torch.empty(0),
                                     self.grid, Tokenizer(), ["crack"], 1)
            result = model.generate(None, torch.tensor([[1, 2, 3, 4]]), torch.empty(0),
                                    self.grid, Tokenizer(), ["crack"], 1,
                                    return_aux_probabilities=True)
        self.assertEqual(modes, ["sequence_concat"] * 4)
        self.assertNotIn("aux_abnormal_probability", default)
        self.assertEqual(result["classes"], [])
        self.assertEqual(result["masks"], {})
        self.assertEqual(result["aux_abnormal_probability"].shape, (4, 4))
        self.assertTrue(torch.all(result["aux_abnormal_probability"] > 0.5))

    def test_generate_ignores_prefix_seg_and_rejects_repeated_aux_anchor_without_crash(self):
        model = TunnelScopeModel.__new__(TunnelScopeModel)
        nn.Module.__init__(model)
        model.qwen = self.qwen
        self.qwen.model.get_image_features = lambda *a, **kw: SimpleNamespace(pooler_output=[self.vision])
        model.sam = TinySam()
        model.fusion = self.fusion
        model.token_rows = nn.Identity()
        model.main_projection = nn.Identity()
        model.aux_projection = nn.Identity()
        model.aux_decoder = nn.Identity()
        model.seg_id, model.nor_id, model.ano_id = 6, 2, 3
        model.enable_interaction = True
        model.enable_auxiliary = True
        model.fusion_mode = "sequence_concat"

        class ScriptedTokenizer:
            eos_token_id = 0

            def decode(self, ids, skip_special_tokens=False):
                mapping = {2: "[NOR]", 7: "crack ", 6: "[SEG]", 8: "\nEND"}
                return "".join(mapping[i] for i in ids)

        def stub_hidden(qwen, rows, ids, pixels, grid, spatial, fusion, **kwargs):
            hidden = ids.float()[..., None].expand(-1, -1, 4).contiguous()
            logits = torch.zeros(1, ids.shape[1], 10)
            generated = ids.shape[1] - 5
            logits[0, -1, [7, 6, 8][generated] if generated < 3 else 2] = 10
            return hidden, logits

        # Prefix deliberately includes a user-side [SEG], then exactly one
        # auxiliary pair. Only generated [SEG] may drive main mask decoding.
        prefix = torch.tensor([[1, 6, 2, 3, 4]])
        with patch("tunnelweave.backbone.image_token_hidden", side_effect=stub_hidden):
            good = model.generate(None, prefix, torch.empty(0), self.grid,
                                  ScriptedTokenizer(), ["crack"], 3,
                                  return_aux_probabilities=True)
        self.assertEqual(good["status"], "ok")
        self.assertEqual(good["classes"], ["crack"])
        self.assertEqual(good["masks"]["crack"].shape, (4, 4))
        self.assertNotIn("SEG_class_alignment_failed", good["errors"])

        def repeated_anchor(qwen, rows, ids, pixels, grid, spatial, fusion, **kwargs):
            hidden = ids.float()[..., None].expand(-1, -1, 4).contiguous()
            logits = torch.zeros(1, ids.shape[1], 10)
            logits[0, -1, 2] = 10
            return hidden, logits

        with patch("tunnelweave.backbone.image_token_hidden", side_effect=repeated_anchor):
            bad = model.generate(None, prefix, torch.empty(0), self.grid,
                                 ScriptedTokenizer(), ["crack"], 1,
                                 return_aux_probabilities=True)
        self.assertEqual(bad["status"], "partial_or_failed")
        self.assertIn("generated_aux_anchor", bad["errors"])
        self.assertNotIn("SEG_class_alignment_failed", bad["errors"])
        self.assertEqual(bad["aux_abnormal_probability"].shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
