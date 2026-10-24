"""Behavioral CPU checks for the public data, loss, and checkpoint boundaries."""
import copy
import io
import json
import random

import numpy as np
from PIL import Image
import pytest
import torch

from tunnelweave.contracts import CLASSES, ContractError
from tunnelweave.data import prepare_dataset, read_jsonl, load_targets, load_image
from tunnelweave.losses import mask_class_losses, masked_presence_loss, compute_loss
from tunnelweave.checkpoint import save_checkpoint, load_checkpoint, capture_rng, adapter_parameters
from tunnelweave.metrics import score_sample, aggregate
from tunnelweave.predictions import assemble_predictions, save_predictions
from tunnelweave.train import build_schedule, validate_schedule, description_loss
from tunnelweave.evaluate import evaluate
from tunnelweave.model import MainlineModel
from test_model import make_base, inputs


def png(array):
    import hashlib
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    data = stream.getvalue()
    return {'bytes': data, 'path': None}, hashlib.sha256(data).hexdigest()


def source_row(roi=False):
    image, sha = png(np.full((4, 5, 3), 100, np.uint8))
    foreground = np.zeros((4, 5), np.uint8)
    foreground[0, 0] = 255
    mask, masksha = png(foreground)
    region = np.zeros((4, 5), np.uint8)
    region[:2] = 255
    valid, validsha = png(region)
    return {'sample_id': 'sample1', 'image': image, 'image_sha256': sha,
            'source_dataset': 'synthetic_test_fixture', 'split': 'train', 'group_id': 'g1',
            'is_synthetic': False, 'height': 4, 'width': 5, 'positive_classes': ['crack'],
            'full_image_coverage_classes': [] if roi else CLASSES.copy(),
            'region_coverage_classes': ['crack', 'surface_wetness'] if roi else [],
            'presence': [1] + [-1 if roi else 0] * 12, 'full_normal_for_13': False,
            'valid_region': valid if roi else None, 'valid_region_sha256': validsha if roi else None,
            'mask_parts': [{'class_name': 'crack', 'part_index': 0, 'mask': mask, 'sha256': masksha, 'provenance': 'unit test'}]}


def text_row(row, sid=None):
    return {'record_id': 'reference1', 'sample_id': sid or row['sample_id'], 'image': row['image'],
            'image_sha256': row['image_sha256'], 'category': 'crack', 'instruction': 'Describe visible evidence.',
            'response': 'A crack is visible.', 'split': 'train'}


def prepared(tmp_path, roi=False):
    original = source_row(roi)
    root = tmp_path / 'data'
    prepare_dataset(root, [original], [text_row(original)])
    row = read_jsonl(root / 'segmentation.jsonl')[0]
    return root, row, load_targets(root, row)


def test_native_roi_unknowns_and_auxiliary_validity(tmp_path):
    root, row, bundle = prepared(tmp_path, True)
    assert load_image(root, row).size == (5, 4)
    assert bundle['valid'].sum() == 20
    assert bundle['presence'].tolist() == [1] + [-1] * 12
    assert bundle['aux_valid'].sum() == 1
    assert bundle['target'].sum() == 1


def test_asset_tamper_is_rejected(tmp_path):
    row = source_row()
    row['image_sha256'] = '0' * 64
    with pytest.raises(ContractError, match='SHA256'):
        prepare_dataset(tmp_path / 'bad', [row], [])


def test_unpaired_instruction_retained_but_not_given_fake_mask(tmp_path):
    row = source_row()
    result = prepare_dataset(tmp_path / 'd', [row], [text_row(row, 'unpaired')])
    assert result['instruction_records'] == 1 and result['joint_training_instruction_records'] == 0
    assert result['unpaired_instruction_record_ids'] == ['reference1']


def test_mixed_schedule_preserves_each_rank_ratio_and_all_pixel_images(tmp_path):
    root, row, _ = prepared(tmp_path)
    rows = [{**row, 'sample_id': f's{i}'} for i in range(6)]
    texts = [{**read_jsonl(root / 'instructions_paired.jsonl')[0], 'sample_id': 's0'}]
    events = build_schedule(rows, texts, 2)
    validate_schedule(events, rows, texts, 2)
    assert len({x['sample_id'] for x in events[:8] if x['task'] == 'pixel'}) == 6
    for start in range(0, 16, 4):
        assert sum(x['task'] == 'instruction' for x in events[start:start + 4]) == 1
    assert events == build_schedule(rows, texts, 2)
    rows[0]['split'] = 'validation'
    with pytest.raises(ContractError, match='training image'):
        validate_schedule(events, rows, texts, 2)


def test_unknown_nan_mask_and_presence_are_disconnected():
    x = torch.tensor([[[[0., float('nan')]]]], requires_grad=True)
    y = torch.tensor([[[[1., float('nan')]]]])
    v = torch.tensor([[[[True, False]]]])
    value = mask_class_losses(x, y, v).sum()
    value.backward()
    assert torch.isfinite(value) and x.grad[0, 0, 0, 1] == 0
    p = torch.full((1, 13), float('nan'), requires_grad=True)
    truth = torch.full((1, 13), -1)
    loss = masked_presence_loss(p, truth)
    loss.backward()
    assert loss == 0 and torch.equal(p.grad, torch.zeros_like(p))


def test_negative_target_keeps_full_bce_without_dice_or_half_weight():
    loss = mask_class_losses(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2, dtype=torch.bool))
    assert float(loss[0]) == pytest.approx(np.log(2))


def test_b32_coefficients_and_mask_gradient_ignore_presence_gate(tmp_path):
    _, _, bundle = prepared(tmp_path)
    masks = torch.zeros(1, 13, 4, 5, requires_grad=True)
    output = {'presence_logits': torch.full((1, 13), -100., requires_grad=True),
              'mask_logits': masks, 'aux_logits': torch.zeros(1, 2, 4, 5, requires_grad=True)}
    total, report = compute_loss(output, bundle, ['crack'])
    total.backward()
    assert report['mask_coefficients'][0] == 1
    assert report['mask_coefficients'][1] == pytest.approx(1 / 13)
    assert masks.grad[0, 0].abs().sum() > 0


def test_gated_miss_is_zero_and_roi_fp_ignores_unknown_pixels(tmp_path):
    root, row, bundle = prepared(tmp_path, True)
    masks = [torch.full((1, 1, 4, 5), -100.) for _ in CLASSES]
    masks[0][0, 0, 0, 0] = 100
    masks[1][0, 0, 3, 0] = 100  # outside valid ROI
    presence = torch.full((1, 13), -100.)
    presence[0, 1] = 100
    result = assemble_predictions({'presence_logits': presence, 'main_masks': masks})
    score = aggregate(score_sample(result, bundle, row))
    assert score['positive_iou_miss_zero'] == 0
    assert score['raw_query_positive_iou'] == 1
    assert score['ROI_zero_false_positive_pixels'] == 0
    row['split'] = 'validation'
    manifest = root / 'heldout.jsonl'
    manifest.write_text(json.dumps(row) + '\n')
    saved = tmp_path / 'predictions'
    save_predictions(saved / '000000', result)
    (saved / 'index.jsonl').write_text(json.dumps({'sample_id': row['sample_id'], 'directory': '000000'}) + '\n')
    scored = evaluate(root, manifest, saved, tmp_path / 'evaluation')
    assert scored['all']['positive_iou_miss_zero'] == 0


def test_checkpoint_exact_optimizer_rng_and_mutation_guard(tmp_path):
    model = MainlineModel(make_base())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    out = model(**inputs())
    sum(x.square().mean() for x in out['main_masks']).backward()
    opt.step()
    rng = capture_rng('cpu')
    identity, run = {'model': 'tiny_cpu_test'}, {'schedule': 'fixed'}
    path = tmp_path / 'checkpoint'
    save_checkpoint(path, model, opt, 1, identity, [rng], run)
    expected = {n: p.detach().clone() for n, p in adapter_parameters(model).items()}
    expected_random = torch.rand(3)
    with torch.no_grad():
        model.presence_head.weight.add_(9)
    assert load_checkpoint(path, model, identity, opt, 0, 1, run) == 1
    assert torch.equal(torch.rand(3), expected_random)
    assert all(torch.equal(adapter_parameters(model)[n], v) for n, v in expected.items())
    assert opt.state
    before = model.presence_head.weight.detach().clone()
    with pytest.raises(ContractError, match='identity'):
        load_checkpoint(path, model, {'model': 'wrong'})
    assert torch.equal(before, model.presence_head.weight)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, opt, 1, identity, [rng], run)


def test_teacher_ce_includes_only_assistant_shifted_labels():
    from tunnelweave.processing import prepare_processor_inputs
    class Processor:
        image_processor = type('IP', (), {'size': {'longest_edge': 64}, 'patch_size': 1})()
        def apply_chat_template(self, messages, **kwargs):
            ids = [5, 6, 7] if len(messages) == 1 else [5, 6, 7, 8, 9, 10]
            return {'input_ids': torch.tensor([ids]), 'image_grid_thw': torch.tensor([[1, 2, 2]])}
    _, labels = prepare_processor_inputs(Processor(), Image.new('RGB', (2, 2)), 'instruction', 'answer')
    assert labels.tolist() == [[-100, -100, -100, 8, 9, 10]]
