"""Fail-closed identity for the real C1 implementation, never config echoing."""
from pathlib import Path
import hashlib
import inspect
from .contracts import ContractError
from .backbone import TunnelScopeModel, SpatialCrossAttention

RUNTIME = Path(__file__).resolve().parent

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_base_architecture(base):
    expected = (RUNTIME / 'backbone.py').resolve()
    if type(base) is not TunnelScopeModel:
        raise ContractError('mainline requires the exact audited TunnelScopeModel class')
    actual = Path(inspect.getsourcefile(type(base))).resolve()
    if actual != expected:
        raise ContractError('C1 implementation source is outside candidate runtime; legacy root tree rejected')
    if getattr(base, 'fusion_mode', None) != 'sequence_concat':
        raise ContractError('C1 mainline requires actual sequence_concat fusion')
    if base.enable_interaction is not True or base.enable_auxiliary is not True:
        raise ContractError('C1 mainline requires interaction and auxiliary branches')
    if type(base.fusion) is not SpatialCrossAttention:
        raise ContractError('unexpected C1 fusion implementation')
    if len({base.seg_id, base.nor_id, base.ano_id}) != 3:
        raise ContractError('special token IDs must be distinct')
    return True

def actual_architecture_identity(base):
    validate_base_architecture(base)
    paths = [RUNTIME/'backbone.py', RUNTIME/'sequence_concat.py', Path(__file__)]
    return {'schema':'c1_mainline_actual_architecture.v1',
            'base_class':type(base).__module__+'.'+type(base).__qualname__,
            'fusion_class':type(base.fusion).__module__+'.'+type(base.fusion).__qualname__,
            'fusion_mode':base.fusion_mode, 'interaction':base.enable_interaction,
            'auxiliary':base.enable_auxiliary, 'special_token_ids':[base.seg_id,base.nor_id,base.ano_id],
            'source_sha256':{str(p.resolve()):_sha(p) for p in paths},
            'mask_route':'13 own-SEG states -> main_projection -> original SAM high_res decoder; no presence gating'}
