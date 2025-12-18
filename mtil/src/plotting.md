# Plotting Utilities

This document describes the plotting utilities available in `src/plot.py`.

## Command-Line Usage

```bash
python -m src.plot [OPTIONS]
```

## Available Functions

### 1. t-SNE Visualization

Generate t-SNE plots to visualize high-dimensional embeddings in 2D.

#### Command-Line

```bash
# Basic usage
python -m src.plot --tsne \
    --embeddings path/to/embeddings.npy \
    --save-path output.png

# With labels for colored clusters
python -m src.plot --tsne \
    --embeddings embeddings.npy \
    --labels labels.npy \
    --save-path tsne_plot.png

# With custom parameters
python -m src.plot --tsne \
    --embeddings embeddings.pt \
    --labels labels.pt \
    --save-path tsne_plot.png \
    --perplexity 50 \
    --title "CLIP Image Embeddings"
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tsne` | flag | - | Enable t-SNE plotting mode |
| `--embeddings` | str | required | Path to embeddings file (`.npy` or `.pt`) |
| `--labels` | str | None | Path to labels file (`.npy` or `.pt`) |
| `--save-path` | str | required | Output path for the plot |
| `--perplexity` | int | 30 | t-SNE perplexity (typically 5-50) |
| `--title` | str | "t-SNE Visualization" | Plot title |

#### Python API

```python
from src.plot import plot_tsne

# Basic usage
tsne_results = plot_tsne(embeddings, save_path="tsne.png")

# With labels
tsne_results = plot_tsne(
    embeddings,
    labels=labels,
    save_path="tsne.png"
)

# Full customization
tsne_results = plot_tsne(
    embeddings,              # numpy array or torch tensor (n_samples, n_features)
    labels=labels,           # optional integer labels (n_samples,)
    class_names=["cat", "dog", "bird"],  # optional class name mapping
    title="My Embeddings",
    save_path="output.png",  # None to display instead of save
    perplexity=30,           # t-SNE perplexity (5-50)
    n_iter=1000,             # optimization iterations
    random_state=42,         # for reproducibility
    figsize=(10, 8),         # figure dimensions
    alpha=0.7,               # point transparency
    cmap="tab10",            # colormap for classes
    show_legend=True         # show legend when labels provided
)
```

#### Returns

- `tsne_results`: 2D numpy array of shape `(n_samples, 2)` containing the reduced coordinates

---

### 2. Training Metrics Plot

Plot accuracy metrics over training iterations from CSV files.

#### Command-Line

```bash
python -m src.plot --single --path ./ckpt/experiment/
```

#### Expected CSV Format

Files must be named `*metrics*.csv` with columns:
- `iteration`: training iteration number
- `top1`: top-1 accuracy
- `top5`: top-5 accuracy

#### Output

Saves `output.png` in the specified path directory.

---

### 3. All Versions Plot

Plot accuracy progression across multiple training versions/stages.

#### Command-Line

```bash
# Basic
python -m src.plot --all --path ./ckpt/experiment/

# With text labels on points
python -m src.plot --all --path ./ckpt/experiment/ --text
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--all` | flag | Enable all-versions plotting mode |
| `--path` | str | Root path containing nested experiment folders |
| `--text` | flag | Display accuracy values as text labels on plot |

#### Expected Structure

```
experiment/
├── results.csv
└── stage1/
    ├── results.csv
    └── stage2/
        └── results.csv
```

#### Output

Saves `output_all.png` in the deepest directory.

---

### 4. Compare Models

Compare accuracy across multiple models using bar charts.

#### Command-Line

```bash
# Compare top-1 accuracy
python -m src.plot \
    --result-paths model1/results.csv model2/results.csv model3/results.csv \
    --save-path comparison.png

# Compare top-5 accuracy
python -m src.plot \
    --result-paths model1/results.csv model2/results.csv \
    --save-path comparison_top5.png \
    --top5

# Compare against baseline (shows difference)
python -m src.plot \
    --result-paths model1/results.csv model2/results.csv \
    --save-path comparison_diff.png \
    --compare-original baseline/results.csv
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--result-paths` | str[] | Paths to result CSV files (space-separated) |
| `--save-path` | str | Output path for the comparison plot |
| `--top5` | flag | Use top-5 accuracy instead of top-1 |
| `--compare-original` | str | Baseline results CSV (plots difference from baseline) |

#### Expected CSV Format

```csv
dataset,top1,top5
CIFAR100,85.2,96.1
ImageNet,72.4,91.3
```

---

## Examples

### Visualizing CLIP Embeddings

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
for ckpt in ckpt/*/embeddings.npy; do
    name=$(dirname $ckpt | xargs basename)
    python -m src.plot --tsne \
        --embeddings $ckpt \
        --save-path plots/${name}_tsne.png \
        --title "$name Embeddings"
done
```
