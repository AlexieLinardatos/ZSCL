# Continual Learning Methods Comparison

This document compares the continual learning methods implemented in this repository for Vision-Language Models (CLIP).

## Overview

| Method | Core Strategy | Memory Required | External Data |
|--------|--------------|-----------------|---------------|
| **Finetune** | Direct training on new tasks | None | No |
| **LwF** | Distillation from previous model | None | No |
| **iCaRL** | Exemplar replay + distillation | Fixed exemplar buffer | No |
| **ZSCL** | Distillation from zero-shot CLIP | None | Yes (Conceptual Captions) |

---

## Method Details

### 1. Finetune (Baseline)

Standard end-to-end finetuning without explicit anti-forgetting mechanisms.

**Loss Function:**
```
L = CrossEntropy(logits, labels)
```

**Characteristics:**
- Simplest approach with minimal compute overhead
- Suffers from catastrophic forgetting on previous tasks
- Serves as performance baseline for other methods

**When to use:** Baseline comparison or when forgetting is acceptable.

---

### 2. LwF (Learning without Forgetting)

Uses knowledge distillation from the previous task's model to regularize learning.

**Reference Model:** Previous model checkpoint (after last task)

**Reference Data:** Current task training data

**Loss Function:**
```
L = CE(logits, labels) + λ * Distillation(logits_prev, logits_curr)
```

Where distillation uses soft targets with temperature T:
```python
p = softmax(logits_prev / T)
L_distill = CrossEntropy(logits_curr / T, p) * T²
```

**Variants:**
- **LwF**: Uses previous task class names as reference text
- **LwF-vR**: Uses random virtual vocabulary for more diverse regularization

**Characteristics:**
- No external dataset required
- Lightweight memory footprint
- Moderate anti-forgetting capability
- Regularization limited to knowledge from previous task only

---

### 3. iCaRL (Incremental Classifier and Representation Learning)

Maintains an exemplar memory of representative samples from past tasks using herding selection.

**Reference Model:** Previous model checkpoint

**Reference Data:** Exemplar memory (selected past samples)

**Exemplar Selection (Herding):**
```python
# Select examples closest to class prototype
prototype = mean(class_features)
for k in range(budget):
    selected = argmin(||prototype - sample||²)
    exemplars.append(selected)
    # Update running mean
```

**Memory Management:**
```python
# Equal budget per task
per_task_budget = total_memory / num_tasks
# Reduce old exemplars when new task arrives
```

**Loss Function:**
```
L = CE(logits, labels) + λ₁ * Distill_image + λ₂ * Distill_text
```

**Characteristics:**
- Explicit rehearsal through exemplar replay
- Strong anti-forgetting through direct access to past data
- Fixed memory budget limits scalability
- Herding ensures representative exemplar selection

---

### 4. ZSCL (Zero-Shot Continual Learning)

Leverages CLIP's zero-shot knowledge as a stable anchor for regularization using external reference data.

**Reference Model:** Pristine zero-shot CLIP (or WiSE-FT weighted model)

**Reference Data:** Conceptual Captions (large-scale image-caption pairs)

**Reference Text:** Conceptual Captions captions or random token sequences

**Loss Function:**
```
L = CE(logits, labels) + L_image_distill + L_text_distill
```

Where both distillation losses compare current vs. zero-shot model outputs:
```python
# Image-space distillation
logits_curr = scale * features_curr @ text_embeddings.T
logits_ref = scale * features_ref @ text_embeddings.T
L_image = Distillation(logits_ref, logits_curr, T=2)

# Text-space distillation (transpose)
L_text = Distillation(logits_ref.T, logits_curr.T, T=2)
```

**Additional Techniques:**
- **Weight Averaging (WE):** Maintains exponential moving average of weights
- **WiSE-FT:** Interpolates between finetuned and zero-shot weights: `α * θ_ft + (1-α) * θ_zs`
- **L2 Regularization:** Optional penalty on weight deviation

**Characteristics:**
- No exemplar memory required
- Zero-shot CLIP provides strong, task-agnostic regularization
- Scales well with large reference datasets
- Requires external dataset (Conceptual Captions)
- Strongest anti-forgetting among methods tested

---

## Detailed Comparison

### Knowledge Source

| Method | Source of Regularization |
|--------|-------------------------|
| Finetune | None (implicit in weights) |
| LwF | Previous task model outputs |
| iCaRL | Stored exemplars from past tasks |
| ZSCL | Zero-shot CLIP + external dataset |

### Computational Trade-offs

| Method | Training Cost | Storage Cost | Inference Cost |
|--------|--------------|--------------|----------------|
| Finetune | Low | Minimal | Single forward pass |
| LwF | Low | Minimal (prev model) | Single forward pass |
| iCaRL | Medium | Fixed buffer | Single forward pass |
| ZSCL | Medium-High | Minimal | Single forward pass |

### Anti-Forgetting Strength

```
Finetune < LwF < iCaRL < ZSCL
(weakest)                (strongest)
```

---

## Loss Components Summary

All distillation-based methods share the same core distillation function:

```python
def distillation(teacher_logits, student_logits, T=2):
    """
    Soft-target distillation with temperature scaling.

    Args:
        teacher_logits: Reference model outputs
        student_logits: Current model outputs
        T: Temperature (higher = softer targets)
    """
    soft_targets = softmax(teacher_logits / T)
    loss = CrossEntropy(student_logits / T, soft_targets) * (T ** 2)
    return loss
```

**Key Differences:**
1. **Reference model source** (zero-shot vs. previous checkpoint)
2. **Reference data source** (external dataset vs. exemplars vs. current data)
3. **Loss weighting** coefficients
4. **Distillation dimensions** (ZSCL applies both image and text space)

---

## Configuration Examples

### ZSCL (CIL)
```yaml
method: "ZSCL"
lr: 7e-6
ls: 0.2              # label smoothing
we: 1                # weight averaging
avg_freq: 10
ref_dataset: "conceptual_captions"
ref_sentences: "conceptual_captions"
```

### iCaRL (CIL)
```yaml
method: "iCaRL"
lr: 7e-6
ls: 0
memory_size: 5000    # exemplars to keep
weight_decay: 0.1
```

### LwF (CIL)
```yaml
method: "lwfvr"      # lwf or lwfvr
lr: 7e-6
ls: 0
ref_sentences: "random"  # for LwF-vR variant
```

### MTIL Command Examples
```bash
# ZSCL
python -m src.main --method ZSCL \
    --ref-dataset conceptual_captions \
    --ref-sentences conceptual_captions \
    --image_loss --text_loss

# iCaRL
python -m src.main --method icarl \
    --memory_size 10000 \
    --image_loss --text_loss

# LwF
python -m src.main --method lwf \
    --ref_sentences random

# Finetune
python -m src.main --method finetune
```

---

## When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| Maximum accuracy retention | ZSCL |
| No external data available | iCaRL |
| Memory-constrained environment | LwF |
| Baseline/comparison only | Finetune |
| Large number of tasks | ZSCL (no memory growth) |
| Privacy constraints (no exemplars) | LwF or ZSCL |

---

## Key Insights

1. **ZSCL's advantage** comes from using zero-shot CLIP as a stable, task-agnostic anchor rather than a drifting previous model.

2. **iCaRL's herding** strategy ensures exemplars are representative of class distributions, not random samples.

3. **LwF-vR's random vocabulary** provides more diverse regularization than using actual class names, improving generalization.

4. **Weight averaging** and **WiSE-FT** are orthogonal techniques that can enhance any method by smoothing the optimization trajectory.

5. **All methods** benefit from the strong pretrained representations in CLIP, making continual learning more tractable than training from scratch.
