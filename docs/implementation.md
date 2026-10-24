# Implementation and data contracts

The numerical core is extracted from the frozen TunnelWeave implementation used in the experiments. The public package replaces machine-specific launchers with portable commands. Internal class names are retained where they help preserve tensor names. No scheduler submission, private server paths, training logs, or dataset pixels are bundled in Git.

## Model

- Qwen3.5-9B uses the pinned revision in `configs/model.json`; SAM 2.1 uses Hiera-L and the checked original checkpoint SHA256.
- Original Qwen visual tokens and coordinate-aligned SAM features are concatenated **before** language-model processing. Original token indices are restored before extracting task states or computing language loss.
- Thirteen fixed class queries feed a shared presence head and the main SAM decoder. The model input does not accept ground-truth classes, annotation coverage, source identifiers, or reference descriptions.
- Presence gates saved inference masks only. It does not gate the training mask loss. The auxiliary decoder predicts reliable normal/abnormal pixels.
- The visual encoders are frozen. Qwen language layers use LoRA; coupling, task-token rows, projections, presence, and the selected SAM decoder parameters are trainable. Generation uses deterministic full-prefix decoding with a 512-token limit.
- The optional structured-description prompts preserve the experiment's Chinese prose fields and English JSON schema. The public instruction corpus also contains English reference responses. Open observations are unclassified review candidates, not a calibrated unknown-defect detector.

## Annotations

`prepare` verifies embedded SHA256 values, retains native geometry, unions mask parts by class, and writes portable relative paths. Presence is `1` for a known positive, `0` for a confirmed full-image negative, and `-1` for unknown. A zero target inside an ROI does not establish global absence. Unknown pixels never become background targets.

The preparation output contains all 1,740 instruction records plus a separate joint-training list matched to segmentation images. Two instruction records do not belong to the admitted 8,889-image segmentation set; they remain in the full instruction export and are listed explicitly in `preparation.json` rather than silently receiving fabricated masks.

The prepared JSONL schema is also the interface for user-supplied validation data. Each segmentation row retains `sample_id`, `split`, `source_dataset`, `group_id`, `image_path`, `image_sha256`, `height`, `width`, `positive_classes`, `full_image_coverage_classes`, `region_coverage_classes`, `valid_region_path`, `presence`, `targets`, and `mask_parts`. Each mask part has `class_name` and a relative `path`. `targets` contains one `disease` entry per positive class. Never split related images across training and evaluation merely by shuffling rows.

## Training recipe and ablations

The public recipe starts from the official backbones and newly initialized project modules. It uses global batch 8, with 2 instruction and 6 pixel events, BF16, AdamW, clipping at 1, and learning rates of 1e-5 / 2e-5 / 1e-6 for LoRA / new modules / decoders. Each event loss is divided by 8 before backward; distributed gradients are summed without another division by rank count. Background BCE is unattenuated; Dice is applied only to positive targets. A primary reliable positive receives weight 1/M; the remaining reliable class terms receive 1/K.

The default public schedule traverses shuffled segmentation images before repeating them and cycles through the paired instruction records. It is a usable fresh-training recipe, **not** a claim to reproduce the paper's historical curriculum or 192-reference matched-ablation schedule. Supply `--schedule` to use an explicitly exported event schedule. The exact ordered schedule, configuration, event losses, optimizer, and per-rank RNG are saved with each public run. Resumption requires identical data, schedule, arm, configuration, and rank count, and writes into a new output directory.

`M_full` is the complete model. `Q_no_semantic_guidance` replaces the visual prediction route's Qwen states with 15 image-conditioned learned queries while retaining the language branch and its CE. It is not a parameter-matched removal of the entire language model. `I_no_instruction_supervision` disables description CE but preserves instruction-image recognition, masks, and auxiliary supervision. All arms use the same event schedule when given the same inputs and seed.

## Checkpoints and evaluation

The released checkpoint format stores the project parameters, relevant frozen decoder parameters, registered buffers, optimizer, and per-rank RNG, with SHA256 verification and exact key/shape/dtype checks. Official backbone weights are downloaded separately. Historical internal experiment checkpoints use a different wrapper format and require explicit conversion; they cannot be silently resumed with this CLI.

`evaluate` recomputes scores from saved original-resolution raw and presence-gated masks. Recognition F1 is computed only for classes with supported full-image positive labels. Positive-pair IoU/Dice retain missed detections as zero, and full-image and ROI results are separated. Empty-region false positives are counted inside reliable pixels. Training diagnostics require an explicit flag. JSON validity is not scored as description factuality, and these pixel metrics do not validate unknown-defect recognition.

The public datasets contain training data. The manuscript's held-out evaluation images and trained adapter checkpoint are not bundled in this source release. Do not label evaluation on the public training corpus as the paper's validation result.

## Release checks

The portable package passed 33 CPU tests, including two-process gradient reduction, and a wheel build. Thirty-five retained numerical definitions match the frozen implementation after package-import relocation. A native-data adapter check covered 15 original images across all seven sources and thirteen positive classes. These checks do not constitute a new end-to-end GPU training run of the public wrapper.
