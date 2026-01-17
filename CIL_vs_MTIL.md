# CIL vs MTIL: Continual Learning Approaches with CLIP

This document describes the two research tracks in this thesis project and the continual learning methods implemented in each.

## Overview

| Aspect | CIL | MTIL |
|--------|-----|------|
| **Full Name** | Class-Incremental Learning | Multi-Task Incremental Learning |
| **Task Definition** | Classes split into sequential tasks | Different datasets as sequential tasks |
| **Configuration** | Hydra (YAML files) | argparse (CLI arguments) |
| **Entry Point** | `cil/main.py` | `mtil/src/main.py` |
| **Primary Use Case** | Single dataset, class-incremental | Multiple datasets, task-agnostic |

---

## CIL (Class-Incremental Learning)

### Concept

CIL divides a single dataset's classes into sequential tasks. For example, CIFAR-100 with 100 classes can be split into 10 tasks of 10 classes each. The model learns new classes incrementally while retaining knowledge of previously learned classes.

**Key Characteristics:**
- Uses the `continuum` library for scenario management
- Task boundaries are defined by class groupings
- Evaluation is cumulative (all seen classes)
- Class order is configurable via YAML files

### Architecture

```
cil/
├── main.py                    # Hydra entry point
├── continual_clip/
│   ├── models.py              # ClassIncremental model
│   ├── datasets.py            # Dataset loaders (CIFAR100, ImageNet, TinyImageNet)
│   ├── dynamic_dataset.py     # Exemplar memory for iCaRL
│   └── cc.py                  # Conceptual Captions reference dataset
├── configs/class/             # Hydra YAML configurations
└── class_orders/              # Class ordering definitions
```

### Training Loop

```
For each task t = 1, 2, ..., T:
    1. Add new class names to vocabulary
    2. Update exemplar memory (iCaRL only)
    3. Train on current task data with distillation regularization
    4. Evaluate on all classes seen so far
```

### Methods in CIL

#### 1. ZSCL (Zero-Shot Continual Learning)

**Concept:** Uses a frozen zero-shot CLIP model as the reference for distillation, combined with an external reference dataset (Conceptual Captions).

**How it works:**
- Reference model: Fresh CLIP (never updated)
- Reference data: Conceptual Captions images and text
- Loss: CE on current task + distillation to zero-shot predictions

**Rationale:** The zero-shot model provides stable, generalizable representations that prevent catastrophic forgetting of semantic knowledge.

```
L = L_CE(current_task) + λ * L_distill(ref_data, zero_shot_model)
```

#### 2. LwF (Learning without Forgetting)

**Concept:** Uses the model from the previous task as the reference for distillation on current task data.

**How it works:**
- Reference model: Model after task t-1
- Reference data: Current task data
- Loss: CE on current task + distillation to old model's predictions

**Rationale:** Preserves decision boundaries learned in previous tasks by encouraging similar output distributions.

```
L = L_CE(current_task) + λ * L_distill(current_data, old_model)
```

#### 3. LwF-vr (LwF with Virtual Random Text)

**Concept:** Variant of LwF that uses random token sequences as "anti-semantic" text references.

**How it works:**
- Same as LwF but with randomly generated text tokens
- Provides regularization without semantic bias

**Rationale:** Tests whether the semantic content of reference text matters for continual learning.

#### 4. iCaRL (Incremental Classifier and Representation Learning)

**Concept:** Maintains an exemplar memory of representative samples from previous tasks.

**How it works:**
- Reference model: Previous task model
- Reference data: Exemplar memory (fixed budget, e.g., 5000 samples)
- Exemplar selection: Samples closest to class centroids
- Loss: CE on current + exemplars + distillation

**Rationale:** Rehearsal with carefully selected exemplars prevents forgetting while managing memory constraints.

```
L = L_CE(current_task ∪ exemplars) + λ * L_distill(exemplars, old_model)
```

#### 5. Zeroshot (Baseline)

**Concept:** No training; uses CLIP's zero-shot classification directly.

**How it works:**
- No adaptation to task data
- Classification via cosine similarity between image and text embeddings

**Rationale:** Establishes baseline performance without any continual learning.

---

## MTIL (Multi-Task Incremental Learning)

### Concept

MTIL treats different datasets as sequential tasks. For example, first train on DTD (textures), then on CIFAR-100, then on ImageNet. The model learns to perform well across diverse visual domains.

**Key Characteristics:**
- Task-agnostic: no explicit task IDs at inference
- Supports component-level training (text encoder, image encoder, or both)
- Includes gradient analysis tools for research
- Supports 50+ datasets

### Architecture

```
mtil/src/
├── main.py                    # argparse entry point
├── args.py                    # 50+ CLI arguments
├── models/
│   ├── modeling.py            # ImageEncoder, ClassificationHead, ImageClassifier
│   ├── finetune.py            # Main training pipeline
│   ├── evaluation.py          # Evaluation utilities
│   └── helpers.py             # WiSE-FT, distillation, L2 loss
├── datasets/                  # Dataset implementations
│   ├── collections.py         # Small datasets (CIFAR, DTD, etc.)
│   └── imagenet*.py           # ImageNet variants
└── templates/                 # Prompt templates for zero-shot
```

### Training Modes

| Mode | Description | Trainable Components |
|------|-------------|---------------------|
| `whole` | Full model fine-tuning | Image encoder + text encoder |
| `image` | Image encoder only | Image encoder |
| `text` | Text encoder only | Text encoder |
| `image-fc` | Image encoder + head | Image encoder + classification head |
| `fc` | Linear probe | Classification head only |

### Methods in MTIL

#### 1. finetune (Standard Fine-Tuning)

**Concept:** Direct fine-tuning on each dataset with cross-entropy loss.

**How it works:**
- Train on current dataset
- Optional: L2 regularization to prevent drift from initial weights
- Optional: Weight Exponential Moving Average for stability

```
L = L_CE(current_dataset) + λ_L2 * ||θ - θ_0||²
```

#### 2. ZSCL (Zero-Shot Continual Learning)

**Concept:** Same principle as CIL-ZSCL but applied to multi-task setting.

**How it works:**
- Reference model: Configurable (zero-shot, WiSE-merged, or custom checkpoint)
- Reference data: Conceptual Captions or other external dataset
- Distillation in both image and text embedding spaces

**Additional features in MTIL:**
- `--ref-model`: Choose reference model type
- `--ref-wise-alpha`: WiSE merge ratio for reference
- `--T`: Temperature for distillation

```
L = L_CE(current) + λ * [L_distill_img(ref) + L_distill_text(ref)]
```

#### 3. LwF (Learning without Forgetting)

**Concept:** Embedded within finetune pipeline with distillation flag.

**How it works:**
- Reference model: Model from previous dataset
- Distillation on current dataset batches
- Integrated with training mode flexibility

#### 4. iCaRL (Incremental Classifier and Representation Learning)

**Concept:** Adapted for multi-dataset setting with shared exemplar memory.

**How it works:**
- Global exemplar database across all datasets
- Per-dataset exemplar budget allocation
- Centroid-based selection adapted for domain diversity

**File:** `mtil/src/models/icarl.py`

### MTIL-Specific Features

#### Orthogonal Gradient Decomposition (OGD)

**Concept:** Projects gradients to be orthogonal to previous tasks' gradient subspaces.

**How it works:**
1. Track gradients during training for each task
2. Compute SVD basis of gradient space (97% energy retention)
3. Project new task gradients orthogonal to this basis
4. Reduces task interference

**Usage:**
```bash
--orthogonal-gradients --orthogonal-gradients-path ./grad_basis/
```

#### WiSE-FT (Weight-Space Ensembling for Fine-Tuning)

**Concept:** Test-time model merging between zero-shot and fine-tuned models.

**How it works:**
```
θ_final = α * θ_finetuned + (1-α) * θ_zeroshot
```

**Rationale:** Balances task-specific performance with zero-shot generalization.

**Usage:**
```bash
--wise-ft --wise-ft-alpha 0.5
```

---

## Loss Functions

### Cross-Entropy Loss

Standard classification loss on current task/dataset classes.

```python
L_CE = -Σ y_i * log(softmax(logits)_i)
```

### Distillation Loss (Knowledge Distillation)

Soft target matching between student (current) and teacher (reference) models.

```python
L_distill = KL(softmax(z_teacher/T) || softmax(z_student/T)) * T²
```

Where T is the temperature (typically T=2).

### L2 Regularization

Weight-space regularization to initial model.

```python
L_L2 = ||θ - θ_0||²
```

---

## Comparison Summary

| Feature | CIL | MTIL |
|---------|-----|------|
| **Task Type** | Class splits | Dataset sequences |
| **Methods** | ZSCL, LwF, LwF-vr, iCaRL | ZSCL, LwF, iCaRL, finetune |
| **Training Granularity** | Full model | Component-level |
| **Gradient Analysis** | Not supported | OGD supported |
| **WiSE-FT** | Not supported | Supported |
| **Exemplar Memory** | DynamicDataset | DynamicDataset |
| **Reference Model Options** | Method-dependent | Fully configurable |
| **Distillation Spaces** | Image + Text | Image + Text (optional) |

---

## When to Use Which

**Use CIL when:**
- Working with a single dataset split by classes
- Need standard class-incremental benchmarks (CIFAR-100, ImageNet)
- Want Hydra-based experiment management

**Use MTIL when:**
- Learning across multiple diverse datasets
- Need fine-grained control over trainable components
- Want to analyze gradient dynamics
- Need WiSE-FT model merging
