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

## CVPR 2027 Run State (as of 2026-11-05)

**Start here after a reset.** Initial single-seed numbers are due Sunday 2026-11-08; other seeds run after that.
Check status on Nibi from `~/projects/def-fqureshi/alexie/ZSCL/mtil` (Nibi must be on `cvpr-2027`, `bca5b3a` or later):
`sacct -u alexie -S 2026-11-02 -X --format=JobID,JobName%22,State,Elapsed,ExitCode`.
Fetch CSVs from local WSL with `bash mtil/scripts/fetch_results.sh`. Claude cannot reach Nibi; the user runs cluster commands.

**Done (P1, 5/5, seed 42, v4 schedule).** Transfer is the paper's per-task upper-triangle mean, not the 55-cell value printed by `compute_metrics.py`.

| Run | Dir under `ckpt/11task/` | Transfer | Avg | Last | Diag |
|---|---|---|---|---|---|
| WE + L2 only (not true fine-tune) | `cvpr/no_anchor_no_replay_v4` | 67.05 | 74.35 | 83.39 | 87.08 |
| Distillation only | `cvpr/zscl_only_v4` | 68.66 | 76.02 | 84.45 | 87.36 |
| Replay + ZSCL | `robustness/buf_11k_nord` | 68.93 | 77.51 | 86.87 | 87.11 |
| Replay + ZSCL + RD (v4 control) | `cvpr/imgsel_random_b11000` | 69.02 | 77.19 | 86.16 | 87.11 |

The control reproduces v4 (86.17 Last). Zero-shot eval also completed: `/scratch/alexie/ckpt/rebuttal/zeroshot/evaluate_all_results.csv` (numbers not read yet).

**Queued for Sunday (P2, all PENDING at last check).** Results land in `ckpt/11task/featrep_v0*` and `ckpt/11task/remind_l*`.

| Job | Run |
|---|---|
| 26253970 | Embeddings fp16, `BUDGET=2000000` (every image) |
| 26253972 | REMIND tap 2, m=128, 65k tokens (~1.66 GB, same-MB) |
| 26253973 / 74 / 75 / 76 | Embedding precision fp16 (control) / fp32 / int8 / int4 |
| 26253980 | Unquantized tokens, tap 6 (`--remind_no_pq`) |

These are enough for Sunday. If any is not finished in time, cut its row; never fill in a number.

**After Sunday, not submitted yet:**
- P3 and P4: `bash scripts/11task/cvpr/submit_cvpr.sh P3` (9 jobs), then `P4` (7 jobs). P3's 11k runs replace Tab. 5's 11k column, which may come from old 13k runs.
- True fine-tune and Replay + ZSCL seeds 1, 2: generate with `sed` from `cvpr/no_anchor_no_replay_v4.sh` (delete `--we`, `--avg_freq 50`, `--l2 1`; rename job and save dir to `true_ft_v4`) and `robustness/buf_11k_nord.sh` (add `--seed N`; rename to `buf11k_nord_sN` and `buf_11k_nord_seedN`), piped to `sbatch`. First run with `--we` off in this trainer, so check it first if it fails early.
- 2 more seeds for the best storage form, 1 Order II run without RD, and Replay only + Replay + RD on the v4 schedule.
- **P3 selection fix (2026-11-06, must be on Nibi before P3):** `ReplayBuffer(ranked=True)` stores non-random selections best first and shrinks by prefix; `exemplar_sel.sh` passes `--replay_encode_share`. Before this, later tasks shrank each selection randomly, so old herding/EL2N/GCR/coverage results and the "selection is null" conclusion are confounded. Feature buffers already kept a ranked prefix, so P4 is not affected.
- Do not copy scripts from the Windows checkout to Nibi: they have CRLF line endings. Local `M` entries under `mtil/` are line endings only.

**Paper plan.** Goal: answer Faisal's 7 questions. Drafts: `cvpr2027_tables.tex`, `figs/q2a_budget.pdf`, `figs/q3_memory.pdf` (`mtil/scripts/plot_budget_curve.py`, `plot_memory_curve.py`). `cvpr2027_main.tex` not edited yet. Cells marked `\running` include jobs not submitted (P3, P4); herding/EL2N/GCR old runs used 13k, so they are not in the 11k column.
Table 1 replay + distillation with prior-work rows (Q1); Fig. 1 budget; Table 2 selection, images and embeddings; Table 3 storage form and precision; Fig. 2 Last vs. memory. RD framing: redundant with ZSCL, substitutes for it without reference images (still uses CC text anchors). Order II goes to the supplement.
Cut Tab. 3 (duplicate, old metric). Move Tabs. 5, 6, 8, 9, 10, 12, 13 and Appendices A-C to the supplement. Selection gets 1-2 sentences.
Text fixes: Order II 68.6 Transfer claim contradicts Tab. 1 (63.57), delete it; replace ImageNet-column claims (68.37, 68.00, +0.37); remove duplicate Metrics paragraph, "ExRD", and "RD is our key addition". Claim Last and memory cost, not overall SOTA: Transfer trails GIFT (69.3) and LoRA-Loop (69.8). 86.87 is single-seed.

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