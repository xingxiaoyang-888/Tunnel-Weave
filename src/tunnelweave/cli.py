"""Public command-line entry points."""
import argparse
import json
from pathlib import Path

from .contracts import require


def main():
    parser = argparse.ArgumentParser(prog='tunnelweave', description='TunnelWeave training, inference, and native-pixel evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Prepare the two gated Hugging Face datasets')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--segmentation-dir', help='Optional local directory of exported Parquet shards')
    prepare.add_argument('--instruction-dir', help='Optional local directory of exported Parquet shards')

    train = sub.add_parser('train', help='Train one model with the public joint-supervision recipe')
    train.add_argument('--data', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--model-config', default='configs/model.json')
    train.add_argument('--train-config', default='configs/train.json')
    train.add_argument('--schedule', help='Explicit JSONL event schedule instead of the public default')
    train.add_argument('--resume', help='Exact checkpoint resume into a new output directory')
    train.add_argument('--arm', choices=['M_full', 'Q_no_semantic_guidance', 'I_no_instruction_supervision'])

    for name in ['infer', 'infer-dataset']:
        p = sub.add_parser(name, help='Save original-resolution predictions from a trained checkpoint')
        p.add_argument('--model-config', default='configs/model.json')
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--descriptions', action='store_true', help='Generate known-region and open-observation descriptions')
        if name == 'infer':
            p.add_argument('--image', required=True)
        else:
            p.add_argument('--data', required=True)
            p.add_argument('--manifest', required=True)

    evaluation = sub.add_parser('evaluate', help='CPU evaluation of saved masks on annotated images')
    evaluation.add_argument('--data', required=True)
    evaluation.add_argument('--manifest', required=True)
    evaluation.add_argument('--predictions', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--allow-training-split', action='store_true', help='Explicitly score a training diagnostic, not validation')
    args = parser.parse_args()

    if args.command == 'prepare':
        from .data import iter_public_rows, prepare_dataset
        result = prepare_dataset(args.output,
            iter_public_rows('segmentation', args.segmentation_dir),
            iter_public_rows('instructions', args.instruction_dir))
        print(json.dumps(result, indent=2))
    elif args.command == 'train':
        from .train import train as run_training
        run_training(args)
    elif args.command in ('infer', 'infer-dataset'):
        from PIL import Image
        from .runtime import load_model
        from .checkpoint import load_checkpoint
        from .data import read_jsonl, load_image
        from .inference import predict
        from .predictions import save_predictions
        cfg = json.loads((Path(args.checkpoint) / 'config.json').read_text())
        model, processor, identity = load_model(args.model_config, cfg['identity']['arm'])
        load_checkpoint(args.checkpoint, model, identity)
        model.eval()
        if args.command == 'infer':
            with Image.open(args.image) as source:
                result = predict(model, processor, source.convert('RGB'), descriptions=args.descriptions)
            save_predictions(args.output, result)
        else:
            rows = read_jsonl(args.manifest)
            root = Path(args.output)
            root.mkdir(parents=True, exist_ok=False)
            with (root / 'index.jsonl').open('x', encoding='utf-8') as index:
                for i, row in enumerate(rows):
                    result = predict(model, processor, load_image(args.data, row), descriptions=args.descriptions)
                    directory = f'{i:06d}'
                    save_predictions(root / directory, result)
                    index.write(json.dumps({'sample_id': row['sample_id'], 'directory': directory}) + '\n')
    else:
        from .evaluate import evaluate
        result = evaluate(args.data, args.manifest, args.predictions, args.output, args.allow_training_split)
        print(json.dumps(result['all'], indent=2))


if __name__ == '__main__':
    main()
