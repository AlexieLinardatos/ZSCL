# Plotting Utilities

This document describes the plotting utilities available in `src/plot.py` for visualizing CLIP model embeddings, training metrics, and model comparisons.

## Quick Reference

| Mode | Command |
|------|---------|
| t-SNE from embeddings | `--tsne --embeddings file.npy` |
| t-SNE from images | `--tsne --image-dir ./images` |
| t-SNE from model | `--tsne --model-path model.pth --image-dir ./images` |
| Training metrics | `--single --path ./ckpt/exp` |
| Sequential results | `--all --path ./ckpt/exp` |
| Model comparison | `--result-paths model1/results.csv model2/results.csv` |

---

## t-SNE Visualization

Generate t-SNE plots to visualize high-dimensional embeddings in 2D space.

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

### From Images (Model-based Extraction)

Extract embeddings directly from images using CLIP.

```bash
# Using pretrained CLIP (ViT-B/32)
python -m src.plot --tsne --image-dir ./data/images --save-path tsne.png

# Using a finetuned model checkpoint
python -m src.plot --tsne --model-path ckpt/model.pth --image-dir ./data/images --save-path tsne.png

# Using a different CLIP architecture
python -m src.plot --tsne --model-name ViT-B/16 --image-dir ./data/images --save-path tsne.png

# Limit number of images (useful for large datasets)
python -m src.plot --tsne --image-dir ./data/images --max-images 1000 --save-path tsne.png

# Specific image paths
python -m src.plot --tsne --image-paths img1.jpg img2.jpg img3.jpg --save-path tsne.png

# CPU inference
python -m src.plot --tsne --image-dir ./data/images --device cpu --save-path tsne.png
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

This will automatically assign labels and use subdirectory names as class names in the legend.

### From Text

```bash
# Embed text strings
python -m src.plot --tsne --texts "a photo of a cat" "a photo of a dog" "a photo of a bird" --save-path tsne.png

# Embed texts from a file (one per line)
python -m src.plot --tsne --text-file class_names.txt --save-path tsne.png
```

### Combined Image and Text

```bash
# Visualize both image and text embeddings together
python -m src.plot --tsne --image-dir ./data/images --texts "cat" "dog" "bird" --save-path tsne.png

# With text file
python -m src.plot --tsne --image-dir ./data/images --text-file prompts.txt --save-path tsne.png
```

### t-SNE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--perplexity` | 30 | t-SNE perplexity (typically 5-50) |
| `--batch-size` | 32 | Batch size for embedding extraction |
| `--max-images` | None | Maximum images to process |
| `--device` | cuda | Device for model inference |
| `--model-name` | ViT-B/32 | CLIP architecture |

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

### Custom Title

```bash
python -m src.plot --result-paths model1/results.csv model2/results.csv \
    --save-path comparison.png --title "Model Comparison on ImageNet"
```

**Expected CSV format**:

```csv
dataset,top1,top5
CIFAR10,85.2,97.1
CIFAR100,62.3,85.4
DTD,44.5,71.2
```

Model names are automatically extracted from the parent directory of each CSV file.

---

## Python API

You can also use the plotting functions directly in Python:

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
    embeddings,                          # numpy array or torch tensor (n_samples, n_features)
    labels=labels,                       # optional integer labels (n_samples,)
    class_names=["cat", "dog", "bird"],  # optional class name mapping
    title="My Embeddings",
    save_path="output.png",              # None to display instead of save
    perplexity=30,                       # t-SNE perplexity (5-50)
    n_iter=1000,                         # optimization iterations
    random_state=42,                     # for reproducibility
    figsize=(10, 8),                     # figure dimensions
    alpha=0.7,                           # point transparency
    cmap="tab10",                        # colormap for classes
    show_legend=True                     # show legend when labels provided
)
```

### t-SNE from Model

```python
from src.plot import plot_tsne_from_model, load_clip_model, extract_embeddings_from_model

# All-in-one: load model, extract embeddings, and plot
embeddings, tsne_coords = plot_tsne_from_model(
    model_path="ckpt/model.pth",         # or None for pretrained CLIP
    model_name="ViT-B/32",               # CLIP architecture
    image_dir="./data/images",           # directory with images
    save_path="tsne.png",
    title="CLIP Embeddings",
    perplexity=30,
    batch_size=32,
    max_images=1000,
    device="cuda",
)

# Or extract embeddings separately
model, preprocess = load_clip_model(model_path="ckpt/model.pth")
embeddings, labels, class_names = extract_embeddings_from_model(
    model, preprocess,
    image_dir="./data/images",
    device="cuda",
    batch_size=32,
)

# Then plot
from src.plot import plot_tsne
plot_tsne(embeddings, labels=labels, class_names=class_names, save_path="tsne.png")
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
```

---

## Examples

### Visualizing CLIP Embeddings from a Dataset

```python
import torch
from src.plot import plot_tsne
from src.models.modeling import ImageEncoder

# Load model and extract embeddings
encoder = ImageEncoder.load("ckpt/model.pth")
embeddings = []
labels = []

for images, targets in dataloader:
    with torch.no_grad():
        emb = encoder(images)
    embeddings.append(emb)
    labels.append(targets)

embeddings = torch.cat(embeddings)
labels = torch.cat(labels)

# Create t-SNE visualization
plot_tsne(
    embeddings,
    labels=labels,
    class_names=dataset.classes,
    title="CLIP Embeddings by Class",
    save_path="clip_tsne.png",
    perplexity=30
)
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

### Batch Processing Multiple Experiments

```bash
# Generate t-SNE for multiple checkpoints
for ckpt in ckpt/*/; do
    name=$(basename $ckpt)
    python -m src.plot --tsne \
        --model-path ${ckpt}model.pth \
        --image-dir ./data/test_images \
        --save-path plots/${name}_tsne.png \
        --title "$name Embeddings" \
        --max-images 500
done
```

### Comparing Image and Text Embeddings

```bash
# Visualize how well text prompts align with images
python -m src.plot --tsne \
    --model-path ckpt/finetuned.pth \
    --image-dir ./data/images \
    --texts "a photo of a cat" "a photo of a dog" "a photo of a bird" \
    --save-path image_text_alignment.png \
    --title "Image-Text Embedding Alignment"
```
