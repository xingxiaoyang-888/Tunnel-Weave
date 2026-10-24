"""One- or two-GPU training with explicit global-batch SUM reduction."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .contracts import CLASSES, require
from .data import covered, load_image, load_targets, read_jsonl, write_json
from .inference import forward_image
from .losses import compute_loss
from .processing import prepare_processor_inputs
from .runtime import load_model, seed_everything, file_sha256
from .checkpoint import capture_rng, save_checkpoint, load_checkpoint


def build_schedule(rows, texts, steps, seed=17):
    """Public-data recipe; one shuffled coverage cycle precedes pixel repeats."""
    require(rows and texts and steps > 0, 'Nonempty images, instructions, and positive steps required')
    rng = random.Random(seed)
    pixels = list(range(len(rows)))
    instructions = list(range(len(texts)))
    rng.shuffle(pixels)
    rng.shuffle(instructions)
    p = t = 0
    events = []
    for step in range(steps):
        for slot in range(8):
            if slot in (0, 4):
                text = texts[instructions[t % len(texts)]]
                events.append({'step': step + 1, 'slot': slot, 'task': 'instruction',
                               'sample_id': text['sample_id'], 'record_id': text['record_id'],
                               'primary_classes': [] if text['category'] == 'normal' else [text['category']]})
                t += 1
            else:
                row = rows[pixels[p % len(pixels)]]
                primary = list(row['positive_classes']) if row['full_image_coverage_classes'] else [covered(row)[(p // len(pixels)) % len(covered(row))]]
                events.append({'step': step + 1, 'slot': slot, 'task': 'pixel',
                               'sample_id': row['sample_id'], 'primary_classes': primary})
                p += 1
    return events


def validate_schedule(events, rows, texts, steps):
    by_id = {x['sample_id']: x for x in rows}
    by_text = {x['record_id']: x for x in texts}
    require(len(events) == steps * 8 and len(by_id) == len(rows) and len(by_text) == len(texts), 'Schedule size or duplicate IDs')
    for i, e in enumerate(events):
        require(e['step'] == i // 8 + 1 and e['slot'] == i % 8, 'Schedule ordering mismatch')
        require(e['task'] == ('instruction' if i % 8 in (0, 4) else 'pixel'), 'Each rank receives one instruction and three pixel events')
        require(e['sample_id'] in by_id and by_id[e['sample_id']]['split'] == 'train', 'Training requires an explicit training image')
        row = by_id[e['sample_id']]
        primary = e['primary_classes']
        require(len(primary) == len(set(primary)) and set(primary) <= set(covered(row)), 'Invalid primary class')
        if e['task'] == 'instruction':
            text = by_text[e['record_id']]
            require(text['sample_id'] == e['sample_id'] and text['image_sha256'] == row['image_sha256'], 'Instruction/image mismatch')
            require(primary == ([] if text['category'] == 'normal' else [text['category']]), 'Instruction primary identity changed')
            require(text['category'] in row['positive_classes'] or (text['category'] == 'normal' and row['full_normal_for_13']), 'Instruction is not a reliable positive/normal label')
    return by_id, by_text


def description_loss(model, processor, image, text):
    inputs, labels = prepare_processor_inputs(processor, image, text['instruction'], text['response'])
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    labels = labels.to(device)
    valid = labels[:, 1:] != -100
    tokens = int(valid.sum())
    require(0 < tokens <= 512, 'Instruction response exceeds 512 supervised tokens; no truncation')
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        output = model.base(image, inputs['input_ids'], inputs['pixel_values'], inputs['image_grid_thw'],
                            inputs.get('attention_mask'), decode_masks=False, decode_aux=False, logit_mode='none')
        logits = model.base.token_rows.logits(output['hidden'][:, :-1][valid], model.base.qwen.lm_head)
        loss = F.cross_entropy(logits.float(), labels[:, 1:][valid])
    return loss, tokens


def reduce_gradients(parameters, world_size):
    """SUM already-scaled event gradients; never divide by rank count again."""
    if world_size == 1:
        return
    flags = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(flags, op=dist.ReduceOp.SUM)
    for p, active in zip(parameters, flags.tolist()):
        if not active:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def train(args):
    cfg = json.loads(Path(args.train_config).read_text())
    require(cfg['batch_size'] == 8 and cfg['instruction_per_batch'] == 2, 'Recipe requires global batch8, instruction:pixel=2:6')
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    require(world in (1, 2), 'Supported training rank counts are 1 and 2')
    require(torch.cuda.is_available(), 'Training requires CUDA')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    try:
        root = Path(args.data)
        rows = read_jsonl(root / 'segmentation.jsonl')
        texts = read_jsonl(root / 'instructions_paired.jsonl')
        events = read_jsonl(args.schedule) if args.schedule else build_schedule(rows, texts, cfg['steps'], cfg['seed'])
        by_id, by_text = validate_schedule(events, rows, texts, cfg['steps'])
        output = Path(args.output)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            with (output / 'schedule.jsonl').open('w', encoding='utf-8') as stream:
                for event in events:
                    stream.write(json.dumps(event, sort_keys=True) + '\n')
            write_json(output / 'train_config.json', cfg)
        if world > 1:
            dist.barrier()
        contract = {'config': cfg, 'segmentation_sha256': file_sha256(root / 'segmentation.jsonl'),
                    'instructions_sha256': file_sha256(root / 'instructions_paired.jsonl'),
                    'schedule_sha256': file_sha256(output / 'schedule.jsonl')}
        model, processor, identity = load_model(args.model_config, args.arm or cfg['arm'], str(device))
        groups, _ = model.parameter_groups(cfg['learning_rates'], cfg['weight_decay'])
        optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
        parameters = [p for p in model.parameters() if p.requires_grad]
        # Common initialization, distinct deterministic dropout streams afterward.
        seed_everything(cfg['seed'] + rank)
        start = 0
        if args.resume:
            start = load_checkpoint(args.resume, model, identity, optimizer, rank, world, contract)
        require(start < cfg['steps'], 'Checkpoint already reached the requested endpoint')
        model.train()

        def checkpoint(step):
            state = capture_rng(device)
            states = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(state, states, dst=0)
            else:
                states = [state]
            if rank == 0:
                save_checkpoint(output / f'checkpoint-{step:06d}', model, optimizer, step, identity, states, contract)
            if world > 1:
                dist.barrier()

        if not args.resume:
            checkpoint(0)
        log_path = output / f'events_rank{rank}.jsonl'
        with log_path.open('x', encoding='utf-8', buffering=1) as log:
            for step in range(start, cfg['steps']):
                optimizer.zero_grad(set_to_none=True)
                receipts = []
                batch = events[step * 8:(step + 1) * 8]
                local_batch = batch if world == 1 else batch[rank * 4:(rank + 1) * 4]
                for event in local_batch:
                    row = by_id[event['sample_id']]
                    image, bundle = load_image(root, row), load_targets(root, row)
                    prediction, _ = forward_image(model, processor, image)
                    ce, tokens = None, 0
                    if event['task'] == 'instruction' and identity['arm'] != 'I_no_instruction_supervision':
                        ce, tokens = description_loss(model, processor, image, by_text[event['record_id']])
                    total, losses = compute_loss(prediction, bundle, event['primary_classes'], ce)
                    (total / 8).backward()
                    receipts.append({**event, 'losses': losses, 'supervised_tokens': tokens, 'batch_scale': .125})
                    del prediction, total, ce, bundle
                reduce_gradients(parameters, world)
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                log.write(json.dumps({'step': step + 1, 'completed': True, 'rank': rank,
                                     'pre_clip_norm': float(norm), 'events': receipts}) + '\n')
                if (step + 1) % cfg['checkpoint_every'] == 0 or step + 1 == cfg['steps']:
                    checkpoint(step + 1)
                if rank == 0 and (step + 1) % 10 == 0:
                    print(json.dumps({'step': step + 1, 'target': cfg['steps'], 'pre_clip_norm': float(norm)}), flush=True)
        if rank == 0:
            write_json(output / 'completion.json', {'steps': cfg['steps'], 'events': len(events), 'arm': identity['arm']})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
