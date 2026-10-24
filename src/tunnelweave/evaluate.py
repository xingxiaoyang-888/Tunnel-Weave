"""Saved-prediction scoring with original GT/valid pixels and misses retained."""
import json
from pathlib import Path

from .contracts import CLASSES, require
from .data import read_jsonl, local_file, binary_image, load_targets, write_json
from .metrics import score_sample, aggregate
from .runtime import file_sha256


def evaluate(data_root, manifest, predictions_root, output, allow_training=False):
    source = read_jsonl(manifest)
    require(source, 'Evaluation manifest is empty')
    splits = sorted({r['split'] for r in source})
    require(allow_training or 'train' not in splits, 'Training data must not be presented as validation; use --allow-training-split for diagnostics')
    root = Path(predictions_root)
    index_rows = read_jsonl(root / 'index.jsonl')
    index = {r['sample_id']: r['directory'] for r in index_rows}
    require(len(index) == len(index_rows) and set(index) == {r['sample_id'] for r in source}, 'Predictions must cover the exact evaluation manifest')
    all_rows = []
    for row in source:
        directory = local_file(root, index[row['sample_id']])
        hashes = json.loads((directory / 'SHA256.json').read_text())
        require(all(file_sha256(local_file(directory, n)) == h for n, h in hashes.items()), 'Saved prediction SHA256 mismatch')
        prediction = json.loads((directory / 'predictions.json').read_text())
        require([r['class_name'] for r in prediction['classes']] == CLASSES, 'Prediction class order mismatch')
        for record in prediction['classes']:
            for field, key in [('raw_mask', 'raw'), ('gated_mask', 'gated')]:
                record[field] = binary_image(local_file(directory, record['files'][key]), (row['height'], row['width']))
        all_rows.extend(score_sample(prediction, load_targets(data_root, row), row))
    result = {'splits': splits, 'training_diagnostic': 'train' in splits,
              'all': aggregate(all_rows),
              'by_source': {s: aggregate([r for r in all_rows if r['source'] == s]) for s in sorted({r['source'] for r in all_rows})},
              'by_class': {c: aggregate([r for r in all_rows if r['class_name'] == c]) for c in CLASSES},
              'by_registered_group': {g: aggregate([r for r in all_rows if r['parent_group'] == g]) for g in sorted({r['parent_group'] for r in all_rows})},
              'notes': ['Empty predictions remain in positive-pair IoU/Dice denominators.',
                        'ROI false positives use gated masks intersected with valid pixels.',
                        'Registered groups do not certify independent scenes.',
                        'Language factuality and unknown-defect reliability require separate evaluation.']}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'per_image_class.jsonl').open('w', encoding='utf-8') as stream:
        for row in all_rows:
            stream.write(json.dumps(row) + '\n')
    write_json(out / 'summary.json', result)
    write_json(out / 'SHA256.json', {p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    return result
