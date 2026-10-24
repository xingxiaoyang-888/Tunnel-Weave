"""Native-pixel BCE/Dice, unknown masking, and B32 class weights."""
import numpy as np
import torch
import torch.nn.functional as F
from .contracts import CLASSES, ContractError

def auxiliary_loss(aux_logits: torch.Tensor, abnormal: torch.Tensor,
                   valid: torch.Tensor) -> torch.Tensor:
    """Two logits [normal, abnormal]; ignore invalid pixels exactly."""
    if aux_logits.ndim != 4 or aux_logits.shape[1] != 2 or abnormal.shape != valid.shape or abnormal.shape != aux_logits.shape[:1] + aux_logits.shape[2:]:
        raise ContractError("invalid auxiliary shapes")
    weights = valid.float()
    if weights.sum() == 0:
        return aux_logits.sum() * 0
    ce = F.cross_entropy(aux_logits, abnormal.long(), reduction="none")
    ce = (ce * weights).sum() / weights.sum()
    probability = aux_logits.softmax(dim=1)[:, 1]
    target = abnormal.float()
    intersection = (probability * target * weights).sum(dim=(-1, -2))
    total = (probability * weights).sum(dim=(-1, -2)) + (target * weights).sum(dim=(-1, -2))
    has_positive = (target * weights).sum(dim=(-1, -2)) > 0
    dice = 1 - (2 * intersection + 1) / (total + 1)
    return ce + (dice * has_positive).sum() / has_positive.sum().clamp_min(1) if has_positive.any() else ce


def mask_class_losses(logits, target, valid):
    if logits.ndim != 4 or logits.shape != target.shape or logits.shape != valid.shape or logits.shape[0] != 1:
        raise ContractError('one-image class loss requires equal [1,C,H,W] shapes')
    valid = valid.bool()
    if not torch.isfinite(logits[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ContractError('nonfinite supervised mask value')
    if not ((target[valid] == 0) | (target[valid] == 1)).all():
        raise ContractError('supervised masks must be binary')
    # Multiplying NaN by zero is unsafe. where disconnects unknown values first.
    x = torch.where(valid, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    y = torch.where(valid, target.float(), torch.zeros_like(target, dtype=torch.float32))
    m = valid.float()
    pixels = m.sum((-1, -2))
    bce = (F.binary_cross_entropy_with_logits(x, y, reduction='none') * m).sum((-1, -2)) / pixels.clamp_min(1)
    p = x.sigmoid()
    fg = (y*m).sum((-1, -2))
    dice = 1 - (2*(p*y*m).sum((-1, -2)) + 1) / ((p*m).sum((-1, -2)) + fg + 1)
    return ((bce + dice*(fg > 0)) * (pixels > 0))[0]


def safe_auxiliary_loss(logits, target, valid):
    if logits.ndim != 4 or logits.shape[1] != 2 or target.shape != valid.shape or target.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ContractError('invalid aux shapes')
    valid = valid.bool()
    if not torch.isfinite(logits.masked_select(valid[:, None].expand_as(logits))).all():
        raise ContractError('nonfinite supervised auxiliary logits')
    if not torch.isfinite(target[valid]).all() or not ((target[valid] == 0) | (target[valid] == 1)).all():
        raise ContractError('invalid supervised auxiliary target')
    x = torch.where(valid[:, None], logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    y = torch.where(valid, target, torch.zeros_like(target))
    return auxiliary_loss(x, y.long(), valid)


def masked_presence_loss(logits, presence):
    if logits.shape != (1,13) or presence.shape != (1,13):
        raise ContractError('presence logits and labels require [1,13]')
    if not bool(((presence == -1) | (presence == 0) | (presence == 1)).all()):
        raise ContractError('presence supports only unknown/absent/present')
    known = presence != -1
    if not bool(torch.isfinite(logits[known]).all()):
        raise ContractError('nonfinite known presence logit')
    # Disconnect unknowns before arithmetic, including NaN/Inf placeholders.
    x = torch.where(known, logits.float(), torch.zeros_like(logits, dtype=torch.float32))
    y = torch.where(known, presence.float(), torch.zeros_like(logits, dtype=torch.float32))
    values = F.binary_cross_entropy_with_logits(x, y, reduction='none')
    return (values * known).sum() / known.sum().clamp_min(1)


def _mask_logits(output):
    x = output.get('mask_logits')
    parts = output.get('main_masks')
    if parts is not None:
        if not isinstance(parts,(tuple,list)) or len(parts)!=13:
            raise ContractError('main_masks must contain all13 raw class masks')
        if any(v.ndim != 4 or v.shape[:2] != (1,1) for v in parts):
            raise ContractError('each main_masks entry requires [1,1,H,W]')
        cat = torch.cat(parts, dim=1)
        if x is None:
            x = cat
        elif x.shape != cat.shape or not torch.allclose(x.detach(),cat.detach(),rtol=0,atol=0,equal_nan=True):
            raise ContractError('mask_logits/main_masks disagree')
    if not isinstance(x,torch.Tensor):
        raise ContractError('raw ungated mask_logits or all13 main_masks required')
    return x


def compute_loss(output, bundle, primary_classes, description_loss=None):
    """One event; the caller divides by the global batch size exactly once."""
    masks = _mask_logits(output)
    device = masks.device
    target = torch.as_tensor(bundle['target'], device=device, dtype=torch.float32)[None]
    valid = torch.as_tensor(bundle['valid'], device=device, dtype=torch.bool)[None]
    presence = torch.as_tensor(bundle['presence'], device=device, dtype=torch.int8)[None]
    if masks.shape != target.shape:
        raise ContractError('Prediction and native annotation geometry mismatch')
    covered = valid.any((-1, -2))[0]
    K, M = int(covered.sum()), len(primary_classes)
    if K <= 0 or M != len(set(primary_classes)) or any(c not in CLASSES or not covered[CLASSES.index(c)] for c in primary_classes):
        raise ContractError('Invalid reliable/primary class denominators')
    coefficients = torch.zeros(13, device=device)
    for i, c in enumerate(CLASSES):
        if covered[i]:
            coefficients[i] = 1 / M if c in primary_classes and presence[0, i] == 1 else 1 / K
    p = masked_presence_loss(output['presence_logits'], presence)
    m = (mask_class_losses(masks, target, valid) * coefficients).sum()
    a = safe_auxiliary_loss(output['aux_logits'],
                           torch.as_tensor(bundle['aux_target'], device=device)[None],
                           torch.as_tensor(bundle['aux_valid'], device=device)[None])
    d = p * 0 if description_loss is None else description_loss
    total = p + m + .2 * a + .2 * d
    if any(not bool(torch.isfinite(x)) for x in [p, m, a, d, total]):
        raise ContractError('Nonfinite supervised loss')
    return total, {'presence': float(p.detach()), 'mask': float(m.detach()),
                   'auxiliary': float(a.detach()), 'description': float(d.detach()),
                   'total': float(total.detach()), 'mask_coefficients': coefficients.tolist()}
