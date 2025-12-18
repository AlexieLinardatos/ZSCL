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

## Web Application

A web UI for this model is available at: https://github.com/JuicedCooky/zscl_ui
