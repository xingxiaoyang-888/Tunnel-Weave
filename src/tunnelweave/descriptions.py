"""Structured known-region descriptions and independent open observations."""
from __future__ import annotations
import json
import re
from collections import Counter
from typing import Any
import numpy as np
from .contracts import CLASSES
from .contracts import ContractError

SCHEMA_VERSION = "c1_evidence_description.v1"
CONTROL_TOKENS = ("[SEG]", "[NOR]", "[ANO]", "<|im_start|>", "<|im_end|>")
OBS_KEYS = {"id", "classification_status", "class_name", "region_id",
            "visible_evidence", "uncertainty", "location_text"}
DEICTIC_RE = re.compile(r"\b(marked|highlighted|outlined|crop|cropped|reference)\b", re.I)
BASE_RULES = (
    "Describe only observable evidence in the original image. Do not invent physical "
    "dimensions, causes, structural safety, severity, repairs, or hidden material state. "
    "Do not refer to annotation colors, marked regions, file names, dataset names, or "
    "reference descriptions. A prediction and its geometry may be wrong: inspect the "
    "image, state uncertainty, and do not treat a supplied class as verified truth. "
    "Return exactly one JSON object, without markdown or special control tokens. "
    "Use schema_version c1_evidence_description.v1 and an observations array. Each item "
    "must have id, classification_status, class_name, region_id, visible_evidence, "
    "uncertainty, and location_text. Evidence, uncertainty, and location_text must be "
    "nonempty strings. Write visible_evidence, uncertainty, and location_text in Simplified Chinese "
    "(use Chinese sentences); keep the JSON keys, IDs, schema_version, and canonical class_name "
    "values exactly as specified. An empty observations array is permitted."
)

def _require(value, message):
    if not value:
        raise ContractError(message)

def prediction_bindings(predictions=(), image_hw=None):
    """Derive all displayed geometry from predicted Boolean masks, never annotations."""
    result = {}
    for p in predictions:
        _require(isinstance(p, dict) and set(p) == {"prediction_id", "class_name", "mask"},
                 "prediction must contain only prediction_id/class_name/mask")
        pid, cls = p["prediction_id"], p["class_name"]
        _require(isinstance(pid, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", pid),
                 "invalid prediction_id")
        _require(pid not in result, "duplicate prediction_id")
        _require(cls in CLASSES, "prediction class outside fixed vocabulary")
        mask = p["mask"]
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        _require(mask.ndim == 2 and mask.dtype == np.bool_, "prediction mask must be Boolean H,W")
        _require(min(mask.shape) > 0, "empty mask geometry")
        if image_hw is not None:
            _require(tuple(mask.shape) == tuple(image_hw), "prediction/original image geometry mismatch")
        ys, xs = np.nonzero(mask)
        geometry = None if not len(xs) else {
            "bbox_xyxy_exclusive": [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
            "area_pixels": int(len(xs)), "area_fraction": float(len(xs)/mask.size),
            "centroid_xy": [float(xs.mean()), float(ys.mean())],
        }
        result[pid] = {"prediction_id": pid, "class_name": cls,
                       "original_hw": list(mask.shape), "has_nonempty_mask": bool(len(xs)),
                       "geometry": geometry}
    return result

def build_description_prompt(predictions=(), *, mode="known", image_hw=None):
    _require(mode in {"known", "open_observation"}, "unsupported description mode")
    if mode == "open_observation":
        _require(not predictions, "open observation does not receive predicted or true classes/regions")
        return (BASE_RULES + " Independently inspect the whole original image. The fixed known "
                "vocabulary is: " + ", ".join(CLASSES) + ". This vocabulary is the same for every image; "
                "it is not a list of ground-truth or predicted classes present in this image. "
                "Report only visually suspicious findings that cannot be reliably assigned to any "
                "of these thirteen known classes. Omit findings that clearly match a known class "
                "from this unclassified-candidate branch, even if an independent detector might "
                "miss them. Do not turn all visible defects into unknown candidates. For each "
                "retained candidate, uncertainty must explain the visible ambiguity or limitation "
                "that prevents reliable assignment to the known vocabulary. If there is no such "
                "supported finding, return an empty observations array. Each retained observation "
                "must use classification_status unclassified_candidate, class_name null, region_id "
                "null, and a unique candidate id. It is a human-review candidate, not proof of a "
                "novel disease or reliable open-set detection. Do not infer novelty from low scores, "
                "absent known-class predictions, NONE, an empty mask, or a parsing failure.")
    bindings = prediction_bindings(predictions, image_hw)
    return (BASE_RULES + " Describe only supplied predicted regions that you can support visually. "
            "For a known observation, id and region_id must both equal its prediction_id, "
            "classification_status must be known, and class_name must match that prediction. "
            "You may omit a region when evidence is insufficient; empty predicted masks have no "
            "localized geometry and must not receive a known region description. Do not add other IDs. "
            "These are model predictions, not ground truth: "
            + json.dumps(list(bindings.values()), ensure_ascii=False, sort_keys=True))

def _no_duplicate_keys(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate JSON key: "+k)
        result[k] = v
    return result

def validate_description_object(obj, *, predictions=(), mode="known", image_hw=None):
    """Schema/identity validation only. Does not validate visual truth of any sentence."""
    errors = []
    bindings = prediction_bindings(predictions, image_hw) if mode == "known" else {}
    if mode not in {"known", "open_observation"}:
        return ["unsupported_mode"]
    if mode == "open_observation" and predictions:
        return ["open_observation_received_predictions"]
    if not isinstance(obj, dict) or set(obj) != {"schema_version", "observations"}:
        return ["top_level_schema"]
    if obj["schema_version"] != SCHEMA_VERSION:
        errors.append("schema_version")
    observations = obj["observations"]
    if not isinstance(observations, list):
        return errors + ["observations_not_array"]
    seen = set()
    for index, o in enumerate(observations):
        prefix = f"observation_{index}:"
        if not isinstance(o, dict) or set(o) != OBS_KEYS:
            errors.append(prefix+"fields")
            continue
        oid = o["id"]
        if not isinstance(oid, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", oid):
            errors.append(prefix+"invalid_id")
        elif oid in seen:
            errors.append(prefix+"duplicate_id")
        else:
            seen.add(oid)
        for field in ["visible_evidence", "uncertainty", "location_text"]:
            if not isinstance(o[field], str) or not o[field].strip():
                errors.append(prefix+"empty_"+field)
            elif not re.search(r"[\u3400-\u9fff]", o[field]):
                errors.append(prefix+"Chinese_language_required_"+field)
        if mode == "known":
            b = bindings.get(o["region_id"]) if isinstance(o["region_id"], str) else None
            if o["classification_status"] != "known":
                errors.append(prefix+"known_status_required")
            if b is None or oid != o["region_id"]:
                errors.append(prefix+"unbound_prediction_id")
            else:
                if o["class_name"] != b["class_name"]:
                    errors.append(prefix+"class_conflicts_with_prediction")
                if not b["has_nonempty_mask"]:
                    errors.append(prefix+"empty_prediction_has_no_localized_region")
        else:
            if o["classification_status"] != "unclassified_candidate":
                errors.append(prefix+"candidate_status_required")
            if o["class_name"] is not None or o["region_id"] is not None:
                errors.append(prefix+"candidate_must_not_claim_class_or_mask")
    return errors

def parse_description(raw_text, *, predictions=(), mode="known", image_hw=None, truncated=False):
    """Preserve raw failures; two JSON documents, controls and duplicate IDs fail closed."""
    errors = []
    obj = None
    if not isinstance(raw_text, str):
        errors.append("raw_text_not_string")
    else:
        if any(t in raw_text for t in CONTROL_TOKENS):
            errors.append("special_control_token_in_description")
        try:
            obj = json.loads(raw_text, object_pairs_hook=_no_duplicate_keys)
            # Check decoded strings as well: JSON Unicode escapes must not hide controls.
            decoded = json.dumps(obj, ensure_ascii=False)
            if any(t in decoded for t in CONTROL_TOKENS) and "special_control_token_in_description" not in errors:
                errors.append("special_control_token_in_description")
            errors.extend(validate_description_object(
                obj, predictions=predictions, mode=mode, image_hw=image_hw))
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            errors.append("invalid_single_JSON:"+str(e))
    if truncated:
        errors.append("generation_truncated")
    bindings = prediction_bindings(predictions, image_hw) if mode == "known" else {}
    valid = not errors
    observations = obj.get("observations", []) if valid else []
    described = {o["region_id"] for o in observations} if mode == "known" else set()
    return {
        "raw_text": raw_text, "parsed": obj if valid else None,
        "errors": errors, "truncated": bool(truncated),
        "status": "schema_valid_unverified" if valid else "parse_failed_unverified",
        "fact_verification": "not_verified", "uncalibrated": True,
        "unknown_detection_claim": False,
        "omitted_prediction_ids": sorted(set(bindings)-described) if mode == "known" else [],
        "description_mode": mode,
    }

def _generate_text(base, processor, image, prompt, max_new_tokens):
    """Full-prefix greedy decoding on real C1; no mask parser and no caption SEG decode."""
    import torch,time
    from .processing import prepare_processor_inputs
    from .backbone import image_token_hidden
    _require(isinstance(max_new_tokens, int) and max_new_tokens > 0, "invalid language token budget")
    _require(not base.training, "description inference requires eval mode")
    device = next(base.parameters()).device
    def synchronize_clock():
        if device.type=='cuda':torch.cuda.synchronize(device)
    synchronize_clock();fixed_start=time.perf_counter()
    decode_step_seconds=[];decode_prefix_tokens=[]
    with torch.inference_mode():
        inputs, _ = prepare_processor_inputs(processor, image, prompt)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        ids = inputs["input_ids"]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features = base.sam.encode_once(image)
            vision = base.qwen.model.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True).pooler_output[0]
        synchronize_clock();fixed_encode_seconds=time.perf_counter()-fixed_start
        prompt_tokens=int(ids.shape[1])
        generated = []
        eos = processor.tokenizer.eos_token_id
        stopped = False
        for _ in range(max_new_tokens):
            synchronize_clock();decode_start=time.perf_counter()
            decode_prefix_tokens.append(int(ids.shape[1]))
            # CUDA autocast matches the C1 bf16 path; CPU mock tests need no CUDA runtime.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, logits = image_token_hidden(
                    base.qwen, base.token_rows, ids, inputs["pixel_values"],
                    inputs["image_grid_thw"], features.image_embed, base.fusion,
                    attention_mask=torch.ones_like(ids), vision_tokens=vision,
                    orig_hw=features.orig_hw, use_interaction=base.enable_interaction,
                    fusion_mode=base.fusion_mode, logit_mode="last")
            nid = int(logits[0, -1].argmax())
            synchronize_clock();decode_step_seconds.append(time.perf_counter()-decode_start)
            generated.append(nid)
            if eos is not None and nid == eos:
                stopped = True
                break
            ids = torch.cat((ids, torch.tensor([[nid]], device=device)), dim=1)
        display = generated[:-1] if stopped else generated
        return {"raw_text": processor.tokenizer.decode(display, skip_special_tokens=False),
                "generated_ids": generated, "truncated": not stopped,
                "stop_reason": "eos" if stopped else "token_limit",
                "max_new_tokens": max_new_tokens,
                "optimizer_updates": 0, "backward_calls": 0,
                "generation_timing": {
                    "fixed_encode_seconds": fixed_encode_seconds,
                    "prompt_tokens": prompt_tokens,
                    "decode_step_seconds": decode_step_seconds,
                    "decode_prefix_tokens": decode_prefix_tokens,
                    "max_decode_step_seconds": max(decode_step_seconds),
                    "max_measured_prefix_tokens": max(decode_prefix_tokens),
                    "greedy_rules_unchanged": True,
                    "method": "synchronized actual image encode and full-prefix token timings; no cache or decoding algorithm change"}}

def describe_image(base, processor, image, predictions=(), *, mode="known", max_new_tokens=512):
    """Public input boundary excludes true classes, ROI, source and reference descriptions."""
    image_hw = (image.height, image.width)
    prompt = build_description_prompt(predictions, mode=mode, image_hw=image_hw)
    generation = _generate_text(base, processor, image, prompt, max_new_tokens)
    parsed = parse_description(generation["raw_text"], predictions=predictions,
                               mode=mode, image_hw=image_hw, truncated=generation["truncated"])
    return {**generation, **parsed, "prompt": prompt,
            "prediction_bindings": prediction_bindings(predictions, image_hw) if mode == "known" else {},
            "image_input": "original_rgb_only", "GT_ROI_source_reference_inputs": False}
