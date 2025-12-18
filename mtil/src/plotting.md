# Plotting Utilities

This document describes the plotting utilities available in `src/plot.py` for visualizing CLIP model embeddings, training metrics, and model comparisons.

## Quick Reference

| Mode | Command |
|------|---------|
| t-SNE from dataset | `--tsne --dataset CIFAR100` |
| t-SNE with text embeddings | `--tsne --dataset DTD --include-text` |
| t-SNE from embeddings | `--tsne --embeddings file.npy` |
| t-SNE from image directory | `--tsne --image-dir ./images` |
| Training metrics | `--single --path ./ckpt/exp` |
| Sequential results | `--all --path ./ckpt/exp` |
| Model comparison | `--result-paths model1/results.csv model2/results.csv` |

---

## t-SNE Visualization

Generate t-SNE plots to visualize high-dimensional embeddings in 2D space.

### From Dataset (Recommended)

Use the existing dataset loaders to extract embeddings with proper preprocessing and class labels.

```bash
# Basic usage with pretrained CLIP
python -m src.plot --tsne --dataset CIFAR100 --save-path tsne.png

# With finetuned model
python -m src.plot --tsne --dataset DTD --model-path ckpt/model.pth --save-path tsne.png

# Include text embeddings from class names (uses dataset templates)
python -m src.plot --tsne --dataset Flowers --include-text --save-path tsne.png

# Use train split instead of test
python -m src.plot --tsne --dataset ImageNet --split train --save-path tsne.png

# Limit number of samples (for large datasets)
python -m src.plot --tsne --dataset ImageNet --max-samples 5000 --save-path tsne.png

# Custom data location
python -m src.plot --tsne --dataset CIFAR100 --data-location /path/to/data --save-path tsne.png

# CPU inference
python -m src.plot --tsne --dataset MNIST --device cpu --save-path tsne.png
```

#### Available Datasets

**Small datasets** (auto-download):
- `CIFAR10`, `CIFAR100`, `MNIST`
- `DTD`, `EuroSAT`, `Flowers`, `Food`
- `Aircraft`, `Caltech101`, `OxfordPet`, `StanfordCars`, `SUN397`

**ImageNet variants** (require manual download):
- `ImageNet`, `ImageNetA`, `ImageNetR`, `ImageNetSketch`, `ImageNetV2`
- `ImageNetSM`, `ImageNetSUB`, `ImageNetSC`

**Other** (require manual download):
- `FMOW`, `IWildCam`, `ObjectNet`, `YTBBRobust`

### From Pre-computed Embeddings

```bash
# Basic usage with numpy file
python -m src.plot --tsne --embeddings embeddings.npy --save-path tsne.png

# With labels for coloring
python -m src.plot --tsne --embeddings embeddings.npy --labels labels.npy --save-path tsne.png

# With PyTorch tensor files
python -m src.plot --tsne --embeddings embeddings.pt --labels labels.pt --save-path tsne.png

# Custom title and perplexity
python -m src.plot --tsne --embeddings embeddings.npy --save-path tsne.png \
    --title "My Embeddings" --perplexity 50
```

### From Image Directory

Extract embeddings directly from images in a directory.

```bash
# Using pretrained CLIP
python -m src.plot --tsne --image-dir ./data/images --save-path tsne.png

# Using a finetuned model
python -m src.plot --tsne --model-path ckpt/model.pth --image-dir ./data/images --save-path tsne.png

# Limit number of images
python -m src.plot --tsne --image-dir ./data/images --max-images 1000 --save-path tsne.png

# Specific image paths
python -m src.plot --tsne --image-paths img1.jpg img2.jpg img3.jpg --save-path tsne.png
```

#### Directory Structure for Automatic Labels

If your image directory has subdirectories, each subdirectory is treated as a class:

```
images/
├── cat/
│   ├── cat1.jpg
│   └── cat2.jpg
├── dog/
│   ├── dog1.jpg
│   └── dog2.jpg
└── bird/
    └── bird1.jpg
```

### From Text

```bash
# Embed text strings
python -m src.plot --tsne --texts "a photo of a cat" "a photo of a dog" --save-path tsne.png

# Embed texts from a file (one per line)
python -m src.plot --tsne --text-file class_names.txt --save-path tsne.png
```

### Combined Image and Text

```bash
# Visualize both image and text embeddings together
python -m src.plot --tsne --image-dir ./data/images --texts "cat" "dog" "bird" --save-path tsne.png
```

### t-SNE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--perplexity` | 30 | t-SNE perplexity (typically 5-50) |
| `--batch-size` | 32 | Batch size for embedding extraction |
| `--max-samples` | None | Maximum samples from dataset |
| `--max-images` | None | Maximum images from directory |
| `--device` | cuda | Device for model inference |
| `--model-name` | ViT-B/32 | CLIP architecture |
| `--split` | test | Dataset split (train/test) |
| `--include-text` | False | Include text embeddings from class names |

---

## Training Metrics

Plot accuracy curves over training iterations.

### Single Experiment

```bash
python -m src.plot --single --path ./ckpt/experiment/

# Custom save path and title
python -m src.plot --single --path ./ckpt/experiment/ \
    --save-path metrics.png --title "Training Progress"
```

**Expected CSV format** (`*metrics*.csv` files in the directory):

```csv
iteration,top1,top5
100,45.2,72.1
200,52.3,78.4
300,58.1,82.3
```

### Sequential Training Results

Plot accuracy across multiple training stages (nested directories).

```bash
python -m src.plot --all --path ./ckpt/experiment/

# Show accuracy values as text labels
python -m src.plot --all --path ./ckpt/experiment/ --text
```

**Expected directory structure**:

```
experiment/
├── results.csv
└── stage1/
    ├── results.csv
    └── stage2/
        └── results.csv
```

**Expected CSV format** (`*results*.csv`):

```csv
dataset,top1,top5
CIFAR10,85.2,97.1
CIFAR100,62.3,85.4
ImageNet,58.1,82.0
```

---

## Model Comparison

Create bar charts comparing multiple models across datasets.

### Basic Comparison

```bash
python -m src.plot --result-paths \
    model1/results.csv \
    model2/results.csv \
    model3/results.csv \
    --save-path comparison.png
```

### Compare Against Baseline

Show accuracy differences relative to a baseline:

```bash
python -m src.plot --result-paths \
    finetuned/results.csv \
    lwf/results.csv \
    zscl/results.csv \
    --compare-original pretrained/results.csv \
    --save-path comparison_diff.png
```

### Use Top-5 Accuracy

```bash
python -m src.plot --result-paths model1/results.csv model2/results.csv \
    --save-path comparison.png --top5
```

**Expected CSV format**:

```csv
dataset,top1,top5
CIFAR10,85.2,97.1
CIFAR100,62.3,85.4
DTD,44.5,71.2
```

---

## Python API

Use the plotting functions directly in Python:

### t-SNE from Dataset

```python
from src.plot import plot_tsne_from_dataset, extract_embeddings_from_dataset, load_clip_model

# All-in-one: load dataset, extract embeddings, and plot
embeddings, tsne_coords = plot_tsne_from_dataset(
    dataset_name="CIFAR100",
    model_path="ckpt/model.pth",  # or None for pretrained CLIP
    data_location="./data",
    split="test",
    max_samples=5000,
    include_text=True,           # Include text embeddings from class names
    save_path="tsne.png",
    title="CIFAR-100 Embeddings",
    perplexity=30,
)

# Or extract embeddings separately
model, preprocess = load_clip_model(model_path="ckpt/model.pth")
embeddings, labels, class_names = extract_embeddings_from_dataset(
    model=model,
    dataset_name="DTD",
    data_location="./data",
    split="test",
    include_text=True,
)

# Then plot
from src.plot import plot_tsne
plot_tsne(embeddings, labels=labels, class_names=class_names, save_path="tsne.png")
```

### t-SNE from Embeddings

```python
from src.plot import plot_tsne
import numpy as np

# Basic usage
embeddings = np.random.randn(100, 512)
labels = np.repeat(range(10), 10)
tsne_coords = plot_tsne(embeddings, labels=labels, save_path="tsne.png")

# Full customization
tsne_coords = plot_tsne(
    embeddings,                          # numpy array or torch tensor
    labels=labels,                       # optional integer labels
    class_names=["cat", "dog", "bird"],  # optional class name mapping
    title="My Embeddings",
    save_path="output.png",
    perplexity=30,
    n_iter=1000,
    figsize=(10, 8),
    alpha=0.7,
    cmap="tab10",
    show_legend=True,
)
```

### t-SNE from Model and Images

```python
from src.plot import plot_tsne_from_model, load_clip_model, extract_embeddings_from_model

# All-in-one
embeddings, tsne_coords = plot_tsne_from_model(
    model_path="ckpt/model.pth",
    image_dir="./data/images",
    save_path="tsne.png",
    title="CLIP Embeddings",
    perplexity=30,
    batch_size=32,
    max_images=1000,
)

# Or extract embeddings separately
model, preprocess = load_clip_model(model_path="ckpt/model.pth")
embeddings, labels, class_names = extract_embeddings_from_model(
    model, preprocess,
    image_dir="./data/images",
    device="cuda",
    batch_size=32,
)
```

### Training Metrics

```python
from src.plot import plot_metrics, plot_sequential_results, compare_models

# Single experiment
plot_metrics("./ckpt/experiment/", save_path="metrics.png", title="Training Progress")

# Sequential results
plot_sequential_results("./ckpt/experiment/", save_path="stages.png", show_text=True)

# Model comparison
compare_models(
    result_paths=["model1/results.csv", "model2/results.csv"],
    save_path="comparison.png",
    baseline_path="pretrained/results.csv",
    top5=False,
    title="Method Comparison",
)
```

---

## Complete CLI Reference

```
usage: python -m src.plot [OPTIONS]

Plotting utilities for CLIP model analysis

General arguments:
  --path PATH           Path to directory for metrics/results
  --save-path PATH      Path to save output plot
  --title TITLE         Plot title

Mode selection:
  --single              Plot training metrics from a single directory
  --all                 Plot sequential training results
  --text                Show text labels on plot

Model comparison:
  --result-paths PATH [PATH ...]
                        Paths to results CSV files for comparison
  --compare-original PATH
                        Baseline results CSV for computing differences
  --top5                Use top-5 accuracy instead of top-1

t-SNE arguments:
  --tsne                Generate t-SNE plot
  --embeddings PATH     Path to embeddings file (.npy or .pt)
  --labels PATH         Path to labels file (.npy or .pt)
  --perplexity N        t-SNE perplexity (default: 30)

t-SNE from model:
  --model-path PATH     Path to model checkpoint for embedding extraction
  --model-name NAME     CLIP architecture (default: ViT-B/32)
  --image-dir DIR       Directory containing images for t-SNE
  --image-paths PATH [PATH ...]
                        List of image paths for t-SNE
  --texts TEXT [TEXT ...]
                        List of texts to embed for t-SNE
  --text-file PATH      File with texts (one per line) for t-SNE
  --batch-size N        Batch size for embedding extraction (default: 32)
  --max-images N        Maximum images to process
  --device DEVICE       Device for model inference (default: cuda)

t-SNE from dataset:
  --dataset NAME        Dataset name (e.g., CIFAR100, DTD, ImageNet)
  --data-location PATH  Root directory for datasets (default: ./data)
  --split {train,test}  Dataset split (default: test)
  --max-samples N       Maximum samples to process from dataset
  --include-text        Include text embeddings from class names
```

---

## Examples

### Visualizing Dataset Embeddings with Text Alignment

```bash
# See how well text embeddings align with image clusters
python -m src.plot --tsne \
    --dataset Flowers \
    --include-text \
    --model-path ckpt/flowers_finetuned.pth \
    --save-path flowers_alignment.png \
    --title "Flowers: Image-Text Alignment"
```

### Comparing Pretrained vs Finetuned Representations

```bash
# Pretrained CLIP
python -m src.plot --tsne --dataset DTD --save-path dtd_pretrained.png

# Finetuned on DTD
python -m src.plot --tsne --dataset DTD --model-path ckpt/dtd.pth --save-path dtd_finetuned.png
```

### Comparing Continual Learning Methods

```bash
python -m src.plot \
    --result-paths \
        ckpt/finetune/results.csv \
        ckpt/lwf/results.csv \
        ckpt/zscl/results.csv \
    --save-path method_comparison.png \
    --compare-original ckpt/pretrained/results.csv
```

### Batch Processing Multiple Datasets

```bash
for dataset in CIFAR10 CIFAR100 DTD Flowers; do
    python -m src.plot --tsne \
        --dataset $dataset \
        --model-path ckpt/model.pth \
        --save-path plots/${dataset}_tsne.png \
        --max-samples 2000
done
```

### Large Dataset Visualization

```bash
# ImageNet with sampling
python -m src.plot --tsne \
    --dataset ImageNet \
    --max-samples 10000 \
    --include-text \
    --perplexity 50 \
    --save-path imagenet_tsne.png
```
