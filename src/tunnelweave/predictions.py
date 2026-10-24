"""Preserve scores, raw masks, gated masks and descriptions as different evidence."""
from __future__ import annotations
import math
import numpy as np
import torch
from .contracts import CLASSES, ContractError


def prediction_geometry(mask):
    m = np.asarray(mask)
    if m.ndim != 2 or m.dtype != np.bool_:
        raise ContractError("geometry requires an original-coordinate boolean prediction mask")
    ys, xs = np.where(m)
    return {"height": int(m.shape[0]), "width": int(m.shape[1]), "area_pixels": int(m.sum()),
            "area_fraction": float(m.mean()),
            "bbox_xyxy_exclusive": None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
            "origin": "predicted_binary_mask_not_GT"}


def assemble_predictions(output, *, presence_threshold=.5, mask_threshold=.5):
    for value in (presence_threshold, mask_threshold):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value < 1:
            raise ContractError("fixed thresholds must be finite and within (0,1)")
    logits = output["presence_logits"]
    if logits.shape != (1,13) or not torch.isfinite(logits).all():
        raise ContractError("invalid presence logits")
    masks = output["main_masks"]
    if len(masks) != 13:
        raise ContractError("all thirteen ungated mask channels must be preserved")
    shape = masks[0].shape
    if len(shape)!=4 or shape[:2]!=(1,1):
        raise ContractError("mask must have shape [1,1,H,W]")
    expected_hw = output.get("geometry", {}).get("original_hw")
    if expected_hw is not None and tuple(shape[-2:]) != tuple(expected_hw):
        raise ContractError("mask is not in original image coordinates")
    result=[];known=[]
    for i,(c,m) in enumerate(zip(CLASSES,masks)):
        if m.shape != shape or not torch.isfinite(m).all():
            raise ContractError("mismatched/nonfinite original mask")
        prob=m.detach().float().sigmoid()[0,0].cpu().numpy()
        raw=prob>=mask_threshold
        score=float(logits[0,i].detach().float().sigmoid())
        present=score>=presence_threshold
        gated=raw.copy() if present else np.zeros_like(raw)
        pid=f"class_{i:02d}"
        record={"prediction_id":pid,"class_name":c,"presence_logit":float(logits[0,i].detach()),
                "presence_score":score,"score_calibrated":False,"predicted_present":present,
                "raw_mask_probability":prob,"raw_mask":raw,"gated_mask":gated,
                "raw_geometry":prediction_geometry(raw),"gated_geometry":prediction_geometry(gated),
                "raw_mask_nonempty":bool(raw.any()),"gated_mask_nonempty":bool(gated.any()),
                "localization_success":None}
        result.append(record)
        if present:
            known.append({"prediction_id":pid,"class_name":c,"mask":gated})
    return {"schema":"c1_mainline_predictions.v1","classes":result,"known_predictions":known,
            "presence_threshold":presence_threshold,"mask_threshold":mask_threshold,
            "presence_calibration":"not_validated","unknown_detected":None,
            "unknown_policy":"independent_open_observation_only_not_low_scores_or_empty_masks",
            "localization_success_requires_GT_scoring":True}


def save_predictions(output_dir, result):
    """No overwrite. Raw probability and both binary masks are saved separately."""
    from pathlib import Path
    import json, hashlib
    from PIL import Image
    p=Path(output_dir);p.mkdir(parents=True,exist_ok=False)
    rows=[]
    for record in result['classes']:
        row={k:v for k,v in record.items() if k not in {'raw_mask_probability','raw_mask','gated_mask'}}
        stem=record['prediction_id']
        np.save(p/(stem+'_probability.npy'),record['raw_mask_probability'],allow_pickle=False)
        Image.fromarray(record['raw_mask'].astype('uint8')*255).save(p/(stem+'_raw.png'))
        Image.fromarray(record['gated_mask'].astype('uint8')*255).save(p/(stem+'_gated.png'))
        row['files']={'probability':stem+'_probability.npy','raw':stem+'_raw.png','gated':stem+'_gated.png'}
        rows.append(row)
    summary={k:v for k,v in result.items() if k not in {'classes','known_predictions'}}
    summary['classes']=rows
    (p/'predictions.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    sha={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in p.iterdir() if q.is_file()}
    (p/'SHA256.json').write_text(json.dumps(sha,indent=2),encoding='utf-8')
    return {'path':str(p),'files':sha}
