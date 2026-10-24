import unittest

import torch

from tunnelweave.sequence_concat import append_aligned_visual_tokens


class SequenceConcatContract(unittest.TestCase):
    def inputs(self, suffix=3):
        length = 2 + 4 + suffix
        emb = torch.arange(length * 3, dtype=torch.float32).reshape(1, length, 3)
        emb.requires_grad_()
        positions = torch.arange(length).reshape(1, 1, length).repeat(3, 1, 1)
        positions[:, :, 2:6] = torch.tensor([[[2, 2, 2, 2]],
                                            [[2, 2, 3, 3]],
                                            [[2, 3, 2, 3]]])
        attention = torch.ones((1, length), dtype=torch.long)
        mask = torch.zeros((1, length), dtype=torch.bool)
        mask[:, 2:6] = True
        aligned = torch.ones((1, 4, 3), requires_grad=True)
        return emb, positions, attention, mask, aligned

    def test_original_tokens_and_positions_are_preserved(self):
        args = self.inputs()
        result = append_aligned_visual_tokens(*args)
        self.assertEqual(result.embeddings.shape, (1, 13, 3))
        self.assertTrue(torch.equal(result.gather_original(result.embeddings), args[0]))
        self.assertTrue(torch.equal(result.position_ids.index_select(2, result.original_to_expanded), args[1]))
        self.assertTrue(torch.equal(result.position_ids[:, :, 6:10], args[1][:, :, 2:6]))
        self.assertEqual(result.original_to_expanded[-1].item(), 12)

    def test_text_loss_gradient_reaches_aligned_features(self):
        args = self.inputs()
        result = append_aligned_visual_tokens(*args)
        # Minimal causal mixing surrogate: later text must depend on appended
        # features. This tests index/gradient plumbing, not Qwen model behavior.
        hidden = result.embeddings.cumsum(dim=1)
        result.gather_original(hidden)[:, -1].sum().backward()
        self.assertTrue(torch.equal(args[-1].grad, torch.ones_like(args[-1])))

    def test_full_prefix_extension_preserves_visual_layout(self):
        short = append_aligned_visual_tokens(*self.inputs(suffix=1))
        long = append_aligned_visual_tokens(*self.inputs(suffix=4))
        self.assertTrue(torch.equal(short.embeddings[:, :10], long.embeddings[:, :10]))
        self.assertTrue(torch.equal(short.position_ids[:, :, :10], long.position_ids[:, :, :10]))
        self.assertEqual(long.original_to_expanded[-1].item(), 13)

    def test_reject_discontinuous_visual_tokens(self):
        args = list(self.inputs())
        args[3][0, 3] = False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            append_aligned_visual_tokens(*args)

    def test_reject_cache_padding_and_wrong_count(self):
        with self.assertRaisesRegex(ValueError, "cached"):
            append_aligned_visual_tokens(*self.inputs(), use_cache=True)
        args = list(self.inputs())
        args[2][0, -1] = 0
        with self.assertRaisesRegex(ValueError, "unpadded"):
            append_aligned_visual_tokens(*args)
        args = list(self.inputs())
        args[-1] = args[-1][:, :-1]
        with self.assertRaisesRegex(ValueError, "count"):
            append_aligned_visual_tokens(*args)


if __name__ == "__main__":
    unittest.main()
