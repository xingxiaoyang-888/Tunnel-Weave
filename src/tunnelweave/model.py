"""Shared class states for presence prediction and native-resolution masks."""
from itertools import chain
from pathlib import Path
import hashlib
from .contracts import CLASSES
import torch
from torch import nn
from .contracts import ContractError
from .backbone import own_token_state
from .optimization import parameter_groups as base_parameter_groups
from .architecture import validate_base_architecture, actual_architecture_identity

class MainlineModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        validate_base_architecture(base)
        self.base = base
        dim = base.qwen.config.text_config.hidden_size
        reference = base.main_projection.layers[0].weight
        self.presence_head = nn.Linear(dim, 1, device=reference.device, dtype=torch.float32)
        self.presence_head_training_status = 'new_random_untrained'
        self._stage = 'B'
        self.set_training_stage('B')
        self.train(base.training)

    def forward(self, image, input_ids, pixel_values, image_grid_thw, attention_mask=None):
        validate_base_architecture(self.base)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ContractError('mainline forward requires one unpadded image sequence')
        for token, count, label in ((self.base.seg_id,13,'SEG'),(self.base.nor_id,1,'NOR'),(self.base.ano_id,1,'ANO')):
            if int((input_ids == token).sum()) != count:
                raise ContractError(f'mainline requires exactly {count} {label} tokens')
        out = self.base(image, input_ids, pixel_values, image_grid_thw, attention_mask,
                        decode_masks=True, decode_aux=True, logit_mode='none')
        if len(out['main_masks']) != 13:
            raise ContractError('all 13 class mask routes must be returned independent of presence predictions')
        states = own_token_state(out['hidden'], input_ids, self.base.seg_id)
        if states.shape[:2] != (1,13):
            raise ContractError('presence states must align with all 13 own SEG positions')
        logits = self.presence_head(states.to(self.presence_head.weight.dtype)).squeeze(-1)
        return {**out, 'presence_logits':logits}

    def set_training_stage(self, stage='B'):
        if stage not in {'B','frozen'}:
            raise ContractError('mainline supports stage B or frozen only; stage A is not the frozen plan')
        self._stage = stage
        if stage == 'frozen':
            for p in self.parameters(): p.requires_grad_(False)
            self.eval()
        else:
            self.base.set_training_stage('B')
            for p in self.presence_head.parameters(): p.requires_grad_(True)
        return {'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'total':sum(p.numel() for p in self.parameters())}

    def train(self, mode=True):
        if mode and getattr(self,'_stage',None) == 'frozen':
            raise ContractError('frozen model cannot enter training without an explicit stage B transition')
        super().train(mode)
        self.base.qwen.model.visual.eval()
        self.base.sam_model.image_encoder.eval()
        return self

    def parameter_groups(self, learning_rates, weight_decay=.01):
        if self._stage != 'B':
            raise ContractError('optimizer groups require explicit stage B')
        groups, counts = base_parameter_groups(self.base, learning_rates, weight_decay)
        for no_decay in (False,True):
            params=[p for n,p in self.presence_head.named_parameters()
                    if p.requires_grad and (n.endswith('bias') or p.ndim<2)==no_decay]
            if params:
                groups.append({'params':params,'lr':learning_rates['new'],
                               'weight_decay':0.0 if no_decay else weight_decay,
                               'family':'new','no_decay':no_decay})
        counts['new'] = counts.get('new',0)+sum(p.numel() for p in self.presence_head.parameters() if p.requires_grad)
        selected=list(chain.from_iterable(g['params'] for g in groups))
        expected=[p for p in self.parameters() if p.requires_grad]
        if len({id(p) for p in selected}) != len(selected) or {id(p) for p in selected} != {id(p) for p in expected}:
            raise ContractError('optimizer omitted or duplicated trainable parameters')
        return groups,counts

    def architecture_identity(self):
        return {**actual_architecture_identity(self.base),
                'wrapper_class':type(self).__module__+'.'+type(self).__qualname__,
                'wrapper_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'class_order':list(CLASSES),
                'presence_head':{'kind':'shared Linear(D,1)','hidden_dim':self.presence_head.in_features,
                    'slots':13,'training_status':self.presence_head_training_status,
                    'mask_loss_gated_by_presence':False}}


TunnelWeave = MainlineModel
