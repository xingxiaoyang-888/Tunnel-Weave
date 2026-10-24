"""Portable native-resolution annotations and Hugging Face export preparation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import CLASSES, ContractError, require


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def local_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Dataset path escapes its root")
    return path


def binary_image(path, hw):
    with Image.open(path) as image:
        require((image.height, image.width) == tuple(hw), "Native annotation geometry mismatch")
        return np.asarray(image.convert("L")) > 0


def load_image(root, row):
    path = local_file(root, row["image_path"])
    with Image.open(path) as im:
        require((im.height, im.width) == (row["height"], row["width"]), "Original image geometry mismatch")
        return im.convert("RGB")


def covered(row):
    full, region = row["full_image_coverage_classes"], row["region_coverage_classes"]
    require(bool(full) != bool(region), "Exactly one full-image or regional coverage vocabulary is required")
    require(bool(row.get("valid_region_path")) == bool(region), "ROI coverage needs a valid-region mask")
    values = full or region
    require(len(values) == len(set(values)) and set(values) <= set(CLASSES), "Invalid coverage classes")
    return [c for c in CLASSES if c in values]


def label_contract(row):
    scope = covered(row)
    positive = row["positive_classes"]
    require(len(positive) == len(set(positive)) and set(positive) <= set(scope), "Invalid positive classes")
    require({t["disease"] for t in row["targets"]} == set(positive), "Target/positive classes disagree")
    full = set(row["full_image_coverage_classes"])
    if row.get("full_normal_for_13"):
        require(full == set(CLASSES) and not positive, "Normal requires full coverage of all 13 classes")
    presence = [1 if c in positive else 0 if c in full else -1 for c in CLASSES]
    require(row.get("presence", presence) == presence, "Stored presence conflicts with native annotation coverage")
    return {"presence": presence, "pixel_covered": [c in scope for c in CLASSES]}


def validate_bundle(bundle, row):
    contract = label_contract(row)
    target, valid = bundle["target"], bundle["valid"]
    shape = (13, row["height"], row["width"])
    require(target.shape == valid.shape == shape, "Native [13,H,W] arrays required")
    require(target.dtype == valid.dtype == np.bool_, "Boolean masks required")
    require(not (target & ~valid).any(), "Foreground outside reliable pixels")
    require(np.array_equal(bundle["presence"], contract["presence"]), "Presence mismatch")
    require(np.array_equal(valid.any((-1, -2)), contract["pixel_covered"]), "Pixel coverage mismatch")
    require(np.array_equal(target.any((-1, -2)), bundle["presence"] == 1), "Positive class has no reliable foreground")
    if row["full_image_coverage_classes"]:
        require(valid[valid.any((-1, -2))].all(), "Full-image coverage must span the original image")
    abnormal = target.any(0)
    require(np.array_equal(bundle["aux_target"], abnormal), "Auxiliary foreground must be the reliable union")
    require(np.array_equal(bundle["aux_valid"], abnormal | valid.all(0)), "Unknown background must not supervise the auxiliary head")
    return contract


def load_targets(root, row):
    contract = label_contract(row)
    hw = (row["height"], row["width"])
    spatial = binary_image(local_file(root, row["valid_region_path"]), hw) if row.get("valid_region_path") else np.ones(hw, bool)
    target = np.zeros((13, *hw), bool)
    valid = np.zeros_like(target)
    for c in covered(row):
        valid[CLASSES.index(c)] = spatial
    for part in row["mask_parts"]:
        c = part["class_name"]
        require(c in row["positive_classes"], "Mask part is outside the positive vocabulary")
        target[CLASSES.index(c)] |= binary_image(local_file(root, part["path"]), hw) & spatial
    abnormal = target.any(0)
    bundle = {"classes": CLASSES, "target": target, "valid": valid,
              "presence": np.array(contract["presence"], dtype=np.int8),
              "aux_target": abnormal, "aux_valid": abnormal | valid.all(0)}
    validate_bundle(bundle, row)
    return bundle


def _payload(value):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return bytes(value["bytes"])
        return Path(value["path"]).read_bytes()
    if isinstance(value, Image.Image):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return buffer.getvalue()
    raise ContractError("Unsupported embedded image value")


def _store(root, value, expected_sha=None):
    payload = _payload(value)
    digest = hashlib.sha256(payload).hexdigest()
    require(not expected_sha or expected_sha == digest, "Embedded asset SHA256 mismatch")
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    relative = f"assets/{digest[:2]}/{digest}.img"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(payload)
    else:
        require(hashlib.sha256(destination.read_bytes()).hexdigest() == digest, "Existing asset changed")
    return relative


def iter_public_rows(kind, local_dir=None, token=True):
    """Parquet bytes are retained so the published original SHA256 stays meaningful."""
    if local_dir:
        import pyarrow.parquet as pq
        for path in sorted(Path(local_dir).glob("data*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
                yield from batch.to_pylist()
    else:
        from datasets import load_dataset, Image as HFImage
        repo = "augustusper/" + ("TunnelWeave-Segmentation" if kind == "segmentation" else "TunnelInstruct")
        dataset = load_dataset(repo, "all", split="train", token=token)
        # Nested image fields retain raw bytes when decoding is disabled globally.
        dataset = dataset.with_format("arrow")
        for table in dataset.iter(batch_size=1):
            yield from table.to_pylist()


def prepare_dataset(output, segmentation_rows, instruction_rows):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    segmentation = []
    seen = set()
    for original in segmentation_rows:
        row = {k: v for k, v in original.items() if k not in {"image", "valid_region", "mask_parts"}}
        require(row["sample_id"] not in seen, "Duplicate segmentation sample")
        seen.add(row["sample_id"])
        row["image_path"] = _store(root, original["image"], row["image_sha256"])
        row["valid_region_path"] = _store(root, original["valid_region"], row.get("valid_region_sha256")) if original.get("valid_region") else None
        row["mask_parts"] = [{**{k: v for k, v in p.items() if k != "mask"},
                              "path": _store(root, p["mask"], p["sha256"])} for p in original["mask_parts"]]
        row["targets"] = [{"disease": c} for c in row["positive_classes"]]
        load_image(root, row)
        load_targets(root, row)
        segmentation.append(row)
    instructions, paired, unpaired = [], [], []
    for original in instruction_rows:
        row = {k: v for k, v in original.items() if k != "image"}
        row["image_path"] = _store(root, original["image"], row["image_sha256"])
        instructions.append(row)
        (paired if row["sample_id"] in seen else unpaired).append(row)
    for name, records in [("segmentation.jsonl", segmentation), ("instructions.jsonl", instructions),
                           ("instructions_paired.jsonl", paired)]:
        with (root / name).open("w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    receipt = {"segmentation_images": len(segmentation), "instruction_records": len(instructions),
               "joint_training_instruction_records": len(paired),
               "unpaired_instruction_record_ids": [x["record_id"] for x in unpaired],
               "classes": CLASSES, "original_assets_sha256_verified": True}
    write_json(root / "preparation.json", receipt)
    return receipt
