"""Portable adapter checkpoints with optimizer, per-rank RNG, and file hashes."""
from pathlib import Path
import random

import numpy as np
import torch

from .contracts import require
from .data import write_json
from .runtime import file_sha256


def adapter_parameters(model):
    prefixes = ('base.token_rows.', 'base.fusion.', 'base.main_projection.', 'base.aux_projection.',
                'base.sam_model.sam_mask_decoder.', 'base.aux_decoder.', 'presence_head.', 'visual_queries.')
    return {n: p for n, p in model.named_parameters() if n.startswith(prefixes) or 'lora_' in n}


def optimizer_names(model, optimizer):
    if optimizer is None:
        return None
    names = {id(p): n for n, p in model.named_parameters()}
    groups = [[names[id(p)] for p in g['params']] for g in optimizer.param_groups]
    flattened = [x for group in groups for x in group]
    require(len(flattened) == len(set(flattened)), 'Duplicate optimizer parameters')
    require(set(flattened) == {n for n, p in model.named_parameters() if p.requires_grad}, 'Optimizer must contain all trainable parameters')
    return groups


def capture_rng(device):
    np_state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': [np_state[0], np_state[1].tolist(), *np_state[2:]],
            'cuda': torch.cuda.get_rng_state(device).cpu() if torch.device(device).type == 'cuda' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    if torch.device(device).type == 'cuda':
        require(state['cuda'] is not None, 'CUDA RNG missing')
        torch.cuda.set_rng_state(state['cuda'], device)


def _cpu_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_tree(v) for v in obj)
    return obj


def save_checkpoint(path, model, optimizer, step, identity, rng_states, run_contract):
    path = Path(path)
    params = adapter_parameters(model)
    require(params and all(bool(torch.isfinite(p).all()) for p in params.values()), 'Cannot save missing or nonfinite adapters')
    state = {'adapters': _cpu_tree(params), 'buffers': _cpu_tree(dict(model.named_buffers())),
             'optimizer': _cpu_tree(optimizer.state_dict()) if optimizer else None,
             'rng_states': rng_states, 'step': step}
    config = {'schema': 'tunnelweave.checkpoint.v1', 'step': step, 'identity': identity,
              'run_contract': run_contract, 'optimizer_names': optimizer_names(model, optimizer),
              'world_size': len(rng_states)}
    path.mkdir(parents=True, exist_ok=False)
    torch.save(state, path / 'state.pt')
    write_json(path / 'config.json', config)
    # Read back actual serialized values before publishing the completion manifest.
    back = torch.load(path / 'state.pt', map_location='cpu', weights_only=True)
    require(set(back['adapters']) == set(params), 'Checkpoint adapter inventory changed')
    for n, p in params.items():
        require(torch.equal(back['adapters'][n], p.detach().cpu()), 'Checkpoint tensor readback failed: ' + n)
    write_json(path / 'SHA256.json', {n: file_sha256(path / n) for n in ['state.pt', 'config.json']})


def load_checkpoint(path, model, identity, optimizer=None, rank=0, world_size=1, run_contract=None):
    import json
    path = Path(path)
    hashes = json.loads((path / 'SHA256.json').read_text())
    require(set(hashes) == {'state.pt', 'config.json'}, 'Incomplete checkpoint')
    require(all(file_sha256(path / n) == h for n, h in hashes.items()), 'Checkpoint SHA256 mismatch')
    cfg = json.loads((path / 'config.json').read_text())
    require(cfg['schema'] == 'tunnelweave.checkpoint.v1' and cfg['identity'] == identity, 'Checkpoint backbone/arm identity mismatch')
    if optimizer is not None:
        require(cfg['world_size'] == world_size, 'Exact resume requires the original rank count')
        require(cfg['optimizer_names'] == optimizer_names(model, optimizer), 'Optimizer ordering differs')
        require(cfg['run_contract'] == run_contract, 'Resume dataset/schedule/training configuration changed')
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=True)
    require(state['step'] == cfg['step'], 'Checkpoint step mismatch')
    for key, current in [('adapters', adapter_parameters(model)), ('buffers', dict(model.named_buffers()))]:
        saved = state[key]
        require(set(saved) == set(current), 'Checkpoint key inventory mismatch: ' + key)
        for n, value in current.items():
            other = saved[n]
            require(value.shape == other.shape and value.dtype == other.dtype, 'Tensor shape/dtype mismatch: ' + n)
            require(bool(torch.isfinite(other).all()), 'Nonfinite checkpoint: ' + n)
    if optimizer is not None:
        require(state['optimizer'] is not None and len(state['rng_states']) == world_size, 'Resume state missing')
    with torch.no_grad():
        for key, current in [('adapters', adapter_parameters(model)), ('buffers', dict(model.named_buffers()))]:
            for n, value in current.items():
                value.copy_(state[key][n].to(value.device))
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng_states'][rank], next(model.parameters()).device)
    return cfg['step']
