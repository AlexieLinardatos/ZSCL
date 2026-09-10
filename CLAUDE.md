# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis research project on **Continual Learning with Vision-Language Models (CLIP)**. It contains two distinct research tracks:

- **CIL** (Class-Incremental Learning): Uses Hydra configuration system
- **MTIL** (Multi-Task Incremental Learning): Uses argparse CLI arguments with gradient analysis capabilities

Both tracks build on OpenAI's CLIP model (lightly modified copies in `cil/clip/` and `mtil/clip/`).

## Common Commands

### CIL Track

```bash
# Setup environment
cd cil
bash setup_environment.sh

# Run experiment (from cil directory)
python main.py \
    --config-path configs/class \
    --config-name imagenet100_10-10.yaml \
    dataset_root="../datasets/" \
    class_order="class_orders/imagenet100.yaml"
```

### MTIL Track

```bash
# Training (local)
python -m src.main \
    --method ZSCL \
    --train-mode whole \
    --train-dataset DTD \
    --iterations 1000 \
    --lr 1e-5 \
    --save ckpt/exp_name

# Evaluation only
python -m src.main \
    --load ckpt/model.pth \
    --eval-only

# Single image evaluation
python -m src.main \
    --load ckpt/model.pth \
    --eval-single path/to/image.jpg \
    --class-names data/text_classes/imagenet_classes.txt \
    --eval-only

# SLURM submission
sbatch train.sh
```

## Architecture

### CIL (`cil/`)
- `main.py`: Entry point using Hydra
- `continual_clip/models.py`: ClassIncremental model with CLIP backbone
- `continual_clip/datasets.py`: Dataset loading (CIFAR100, ImageNet, TinyImageNet)
- `continual_clip/dynamic_dataset.py`: Memory management for iCaRL exemplars
- `configs/class/`: YAML configs defining dataset, method (LwF, iCaRL, ZSCL), and hyperparameters
- `class_orders/`: YAML files defining class orderings per task

### MTIL (`mtil/src/`)
- `main.py`: Entry point with argparse
- `args.py`: All CLI arguments (50+ parameters)
- `models/modeling.py`: ImageEncoder, ClassificationHead, ImageClassifier
- `models/finetune.py`: Training pipeline with gradient tracking
- `models/evaluation.py`: Evaluation metrics and zero-shot classification
- `models/helpers.py`: WiSE-FT, L2 loss, distillation utilities
- `datasets/`: Comprehensive dataset implementations (collections.py for small datasets, imagenet*.py for variants)
- `templates/`: Prompt templates for zero-shot classification

### Training Methods
- `finetune`: Full model adaptation
- `lwf`: Learning without Forgetting (distillation-based)
- `icarl`: Incremental Classifier and Representation Learning (exemplar memory)
- `ZSCL`: Zero-Shot Continual Learning

### Train Modes (MTIL)
- `whole`: Train entire model
- `text`: Text encoder only
- `image`: Image encoder only
- `image-fc`: Image encoder + classification head
- `fc`: Classification head only

## Dataset Preparation

Large datasets require manual download. See `mtil/datasets.md` for instructions covering:
- Conceptual Captions (reference text embeddings)
- ImageNet variants (ImageNet-A, ImageNet-R, ImageNet-Sketch, ImageNet-V2)
- WILDS datasets (FMOW, IWildCam)
- ObjectNet, YTBB-Robust

Small datasets (CIFAR, DTD, MNIST, etc.) download automatically via torchvision.

## Key Configuration

### CIL Config Example (`configs/class/imagenet100_10-10-ZSCL.yaml`)
Defines: model architecture, dataset path, number of classes per task, learning rate, optimizer settings, method-specific parameters.

### MTIL Key Arguments
```
--method          # finetune, lwf, ZSCL, icarl
--train-mode      # whole, text, image, image-fc, fc
--train-dataset   # DTD, CIFAR100, ImageNet, etc.
--ref-dataset     # Reference dataset for ZSCL regularization
--ref-sentences   # Reference text (e.g., conceptual_captions)
--iterations      # Training iterations
--save/--load     # Checkpoint paths
```

## Active Work: Feature Replay (as of 2026-09-10)

Next extension being implemented. Store CLIP's final 512-d image embedding (fp16, ~1 KB) in the
replay buffer instead of the preprocessed image tensor (224x224x3 fp32, 602 KB). At the 11k budget
this is 6.6 GB -> ~11 MB.

**Rationale (verified in code).** `L_zscl` (`src/models/training.py:548`) and `L_RD`
(`phase3/losses_phase3.py`) compute teacher text embeddings under `no_grad`, so both send gradient
to the **image encoder only**. `compute_replay_loss` (`training.py:765`) recomputes class-name
embeddings with the live model, making it the **only term that updates the text encoder for
previous-task classes**. Replay's image-side gradient is redundant with two distillation terms; its
text-side gradient is unique. Feature replay drops the former and keeps the latter.

Note: the `--text_loss` flag transposes the logits (changes softmax direction). It does **not**
route gradient into the text encoder.

**Two variants.**
- **A (pure):** buffer holds embeddings only. Replay CE becomes
  `logits = scale * stored_feat @ text_emb.T`; only `text_emb` carries grad.
- **B (hybrid):** small pixel buffer (feeds `L_RD` + full-gradient CE) plus large feature buffer
  (text-side CE only). Safer, and contains A as its zero-pixel endpoint.

**Required side-effect.** `L_RD` needs real pixels. Under variant A it must switch to current-task
images; rebuttal control C1b measured this as free (66.66 vs 66.72 Transfer).

**Do this first.** Drift gate — re-encode the v4 buffer with the task-0 and final checkpoints,
compare per-task cosine similarity against a Seq-FT control. One day, existing checkpoints, no GPU.
Cosine > ~0.95 licenses the direction; ~0.7 means drop it.

Full write-up: `replay_storage_proposal.md`. Note `replay_extensions_slides.md` is stale (still
lists class-name and VQ-token replay, both dropped).

## Web Application

A web UI for this model is available at: https://github.com/JuicedCooky/zscl_ui

## be efficient at using tokens 