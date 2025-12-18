"""
Plotting utilities for CLIP model analysis and training metrics visualization.

This module provides functions for:
- t-SNE visualization of embeddings (from files or extracted from models)
- Training metrics plotting (accuracy over iterations)
- Model comparison bar charts
"""

import matplotlib.pyplot as plt
import csv
import os
from matplotlib.ticker import FormatStrFormatter
import argparse
import numpy as np
from sklearn.manifold import TSNE
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm


# =============================================================================
# t-SNE Visualization Functions
# =============================================================================

def plot_tsne(
    embeddings,
    labels=None,
    class_names=None,
    title="t-SNE Visualization",
    save_path=None,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    figsize=(10, 8),
    alpha=0.7,
    cmap="tab10",
    show_legend=True,
):
    """
    Create a t-SNE plot from embeddings.

    Args:
        embeddings: numpy array or torch tensor of shape (n_samples, n_features)
        labels: optional array of integer labels for coloring points (n_samples,)
        class_names: optional list of class names corresponding to label indices
        title: plot title
        save_path: path to save the figure (if None, displays the plot)
        perplexity: t-SNE perplexity parameter (typically 5-50)
        n_iter: number of iterations for t-SNE optimization
        random_state: random seed for reproducibility
        figsize: figure size as (width, height)
        alpha: point transparency
        cmap: colormap name for coloring different classes
        show_legend: whether to show legend when labels are provided

    Returns:
        tsne_results: 2D numpy array of shape (n_samples, 2)
    """
    # Convert torch tensor to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Ensure embeddings are 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(embeddings) - 1),
        n_iter=n_iter,
        random_state=random_state,
    )
    tsne_results = tsne.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap(cmap, len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = class_names[label] if class_names is not None else f"Class {label}"
            ax.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                c=[colors(i)],
                label=label_name,
                alpha=alpha,
                s=50,
            )

        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            alpha=alpha,
            s=50,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

    return tsne_results


def load_clip_model(model_path=None, model_name="ViT-B/32", device="cuda"):
    """
    Load a CLIP model for extracting embeddings.

    Args:
        model_path: Path to model checkpoint, or None to use pretrained CLIP
        model_name: CLIP architecture name (e.g., "ViT-B/32", "ViT-B/16")
        device: Device to load the model on

    Returns:
        model: The loaded CLIP model
        preprocess: The preprocessing transform for images
    """
    import clip.clip as clip

    if model_path is not None:
        model, _, preprocess = clip.load(model_path, device, jit=False)
    else:
        model, _, preprocess = clip.load(model_name, device, jit=False)

    model.eval()
    return model, preprocess


def get_image_paths_from_directory(directory, extensions=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")):
    """
    Get all image paths from a directory.

    Args:
        directory: Path to directory containing images
        extensions: Tuple of glob patterns for image extensions

    Returns:
        image_paths: List of image file paths
        labels: List of integer labels (based on subdirectory structure, or None if flat)
        class_names: List of class names (subdirectory names)
    """
    image_paths = []
    labels = []
    class_names = []

    # Check if directory has subdirectories (class-based structure)
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    if subdirs:
        # Class-based directory structure
        subdirs = sorted(subdirs)
        class_names = subdirs
        for class_idx, class_name in enumerate(subdirs):
            class_dir = os.path.join(directory, class_name)
            for ext in extensions:
                class_images = glob(os.path.join(class_dir, ext))
                class_images.extend(glob(os.path.join(class_dir, ext.upper())))
                image_paths.extend(class_images)
                labels.extend([class_idx] * len(class_images))
    else:
        # Flat directory structure
        for ext in extensions:
            image_paths.extend(glob(os.path.join(directory, ext)))
            image_paths.extend(glob(os.path.join(directory, ext.upper())))
        labels = None
        class_names = None

    return sorted(image_paths), labels, class_names


def extract_embeddings_from_model(
    model,
    preprocess,
    image_dir=None,
    image_paths=None,
    texts=None,
    text_file=None,
    device="cuda",
    batch_size=32,
    max_images=None,
):
    """
    Extract embeddings from images and/or text using a CLIP model.

    Args:
        model: CLIP model (with encode_image and encode_text methods)
        preprocess: Preprocessing transform for images
        image_dir: Directory containing images (with optional class subdirectories)
        image_paths: List of specific image paths (alternative to image_dir)
        texts: List of text strings to embed
        text_file: Path to file with texts (one per line)
        device: Computation device
        batch_size: Batch size for embedding extraction
        max_images: Maximum number of images to process (for large datasets)

    Returns:
        embeddings: numpy array of embeddings
        labels: numpy array of labels (or None)
        class_names: list of class names (or None)
    """
    import clip.clip as clip

    embeddings = None
    labels = None
    class_names = None

    # Extract image embeddings
    if image_dir is not None:
        paths, labels, class_names = get_image_paths_from_directory(image_dir)
        if max_images is not None and len(paths) > max_images:
            # Sample images while maintaining class distribution if possible
            if labels is not None:
                indices = []
                unique_labels = np.unique(labels)
                per_class = max_images // len(unique_labels)
                for lbl in unique_labels:
                    lbl_indices = [i for i, l in enumerate(labels) if l == lbl]
                    if len(lbl_indices) > per_class:
                        lbl_indices = np.random.choice(lbl_indices, per_class, replace=False).tolist()
                    indices.extend(lbl_indices)
                paths = [paths[i] for i in indices]
                labels = [labels[i] for i in indices]
            else:
                indices = np.random.choice(len(paths), max_images, replace=False)
                paths = [paths[i] for i in indices]
        image_paths = paths

    if image_paths is not None:
        print(f"Extracting embeddings from {len(image_paths)} images...")
        all_embeddings = []
        valid_indices = []

        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                batch_indices = []

                for idx, path in enumerate(batch_paths):
                    try:
                        img = Image.open(path).convert("RGB")
                        img_tensor = preprocess(img)
                        batch_images.append(img_tensor)
                        batch_indices.append(i + idx)
                    except Exception as e:
                        print(f"Failed to load image {path}: {e}")

                if batch_images:
                    batch_tensor = torch.stack(batch_images).to(device)
                    batch_embeddings = model.encode_image(batch_tensor)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                    all_embeddings.append(batch_embeddings.cpu().numpy())
                    valid_indices.extend(batch_indices)

        if all_embeddings:
            embeddings = np.concatenate(all_embeddings, axis=0)
            if labels is not None:
                labels = np.array([labels[i] for i in valid_indices])

    # Extract text embeddings
    if texts is not None or text_file is not None:
        if text_file is not None:
            with open(text_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]

        print(f"Extracting embeddings from {len(texts)} texts...")
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(device)
            text_embeddings = model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.cpu().numpy()

        if embeddings is not None:
            embeddings = np.concatenate([embeddings, text_embeddings], axis=0)
            if labels is not None:
                text_label = max(labels) + 1 if len(labels) > 0 else 0
                labels = np.concatenate([labels, np.full(len(texts), text_label)])
                if class_names is not None:
                    class_names = list(class_names) + ["text"]
        else:
            embeddings = text_embeddings
            class_names = [t[:20] + "..." if len(t) > 20 else t for t in texts]
            labels = np.arange(len(texts))

    return embeddings, labels, class_names


def extract_embeddings_from_dataset(
    model,
    dataset_name,
    data_location="./data",
    device="cuda",
    split="test",
    max_samples=None,
    include_text=False,
    batch_size=32,
):
    """
    Extract embeddings from a dataset using the existing dataset loaders.

    Args:
        model: CLIP model (with encode_image and encode_text methods)
        dataset_name: Name of the dataset (e.g., "CIFAR100", "DTD", "ImageNet")
        data_location: Root directory for datasets
        device: Computation device
        split: "train" or "test" split
        max_samples: Maximum number of samples to process (None for all)
        include_text: Whether to also extract text embeddings from class names
        batch_size: Batch size for dataloader

    Returns:
        embeddings: numpy array of embeddings
        labels: numpy array of labels
        class_names: list of class names
    """
    import clip.clip as clip
    from . import datasets as dataset_module

    # Get the dataset class
    if not hasattr(dataset_module, dataset_name):
        available = [name for name in dir(dataset_module) if not name.startswith('_')]
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available[:20]}...")

    dataset_class = getattr(dataset_module, dataset_name)

    # Get preprocessing from model
    _, _, preprocess = clip.load("ViT-B/32", device, jit=False)

    # Create dataset instance
    dataset = dataset_class(
        preprocess=preprocess,
        location=data_location,
        batch_size=batch_size,
    )

    # Get dataloader
    dataloader = dataset.train_loader if split == "train" else dataset.test_loader
    class_names = dataset.classnames

    print(f"Extracting embeddings from {dataset_name} ({split} split)...")
    print(f"Classes: {len(class_names)}, Samples: {len(dataloader.dataset)}")

    all_embeddings = []
    all_labels = []
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch['images']
                labels = batch['labels']
            elif isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

            total_samples += len(images)
            if max_samples is not None and total_samples >= max_samples:
                break

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Trim to max_samples if needed
    if max_samples is not None and len(embeddings) > max_samples:
        embeddings = embeddings[:max_samples]
        labels = labels[:max_samples]

    print(f"Extracted {len(embeddings)} image embeddings")

    # Optionally add text embeddings
    if include_text:
        print(f"Extracting text embeddings for {len(class_names)} classes...")

        # Use dataset templates if available
        if hasattr(dataset, 'templates') and dataset.templates:
            template = dataset.templates[0]
            texts = [template(name) for name in class_names]
        else:
            texts = [f"a photo of a {name}" for name in class_names]

        text_tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            text_embeddings = model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.cpu().numpy()

        # Combine image and text embeddings
        embeddings = np.concatenate([embeddings, text_embeddings], axis=0)
        # Add text labels (use negative indices or max_label + class_idx)
        text_labels = np.arange(len(class_names)) + labels.max() + 1
        labels = np.concatenate([labels, text_labels])
        # Update class names to distinguish text
        class_names = list(class_names) + [f"[text] {name}" for name in class_names]

        print(f"Total embeddings: {len(embeddings)} (images + text)")

    return embeddings, labels, class_names


def plot_tsne_from_dataset(
    dataset_name,
    model_path=None,
    model_name="ViT-B/32",
    data_location="./data",
    device="cuda",
    split="test",
    max_samples=None,
    include_text=False,
    save_path=None,
    title=None,
    perplexity=30,
    n_iter=1000,
    batch_size=32,
    **tsne_kwargs
):
    """
    Create t-SNE plot from a dataset using existing dataset loaders.

    Args:
        dataset_name: Name of the dataset (e.g., "CIFAR100", "DTD", "ImageNet")
        model_path: Path to model checkpoint, or None for pretrained CLIP
        model_name: CLIP architecture name
        data_location: Root directory for datasets
        device: Computation device
        split: "train" or "test" split
        max_samples: Maximum samples to process
        include_text: Whether to include text embeddings from class names
        save_path: Path to save the plot
        title: Plot title
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
        batch_size: Batch size for dataloader
        **tsne_kwargs: Additional arguments for plot_tsne

    Returns:
        embeddings: The extracted embeddings
        tsne_results: The 2D t-SNE coordinates
    """
    model, _ = load_clip_model(model_path, model_name, device)

    embeddings, labels, class_names = extract_embeddings_from_dataset(
        model=model,
        dataset_name=dataset_name,
        data_location=data_location,
        device=device,
        split=split,
        max_samples=max_samples,
        include_text=include_text,
        batch_size=batch_size,
    )

    plot_title = title or f"t-SNE: {dataset_name} ({split})"

    tsne_results = plot_tsne(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        title=plot_title,
        save_path=save_path,
        perplexity=perplexity,
        n_iter=n_iter,
        **tsne_kwargs
    )

    return embeddings, tsne_results


def plot_tsne_from_model(
    model_path=None,
    model_name="ViT-B/32",
    image_dir=None,
    image_paths=None,
    texts=None,
    text_file=None,
    device="cuda",
    save_path=None,
    title="t-SNE Visualization",
    perplexity=30,
    n_iter=1000,
    batch_size=32,
    max_images=None,
    **tsne_kwargs
):
    """
    Load a model and create t-SNE plot from images or text.

    Args:
        model_path: Path to model checkpoint, or None for pretrained CLIP
        model_name: CLIP architecture name
        image_dir: Directory containing images (with optional class subdirectories)
        image_paths: List of specific image paths (alternative to image_dir)
        texts: List of text strings to embed
        text_file: Path to file with texts (one per line)
        device: Computation device
        save_path: Path to save the plot
        title: Plot title
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
        batch_size: Batch size for embedding extraction
        max_images: Maximum number of images to process (for large datasets)
        **tsne_kwargs: Additional arguments for plot_tsne

    Returns:
        embeddings: The extracted embeddings
        tsne_results: The 2D t-SNE coordinates
    """
    model, preprocess = load_clip_model(model_path, model_name, device)

    embeddings, labels, class_names = extract_embeddings_from_model(
        model=model,
        preprocess=preprocess,
        image_dir=image_dir,
        image_paths=image_paths,
        texts=texts,
        text_file=text_file,
        device=device,
        batch_size=batch_size,
        max_images=max_images,
    )

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings extracted. Provide image_dir, image_paths, texts, or text_file.")

    print(f"Total embeddings: {len(embeddings)}, shape: {embeddings.shape}")

    tsne_results = plot_tsne(
        embeddings=embeddings,
        labels=labels,
        class_names=class_names,
        title=title,
        save_path=save_path,
        perplexity=perplexity,
        n_iter=n_iter,
        **tsne_kwargs
    )

    return embeddings, tsne_results


# =============================================================================
# Training Metrics Plotting Functions
# =============================================================================

def plot_metrics(path, save_path=None, title=None):
    """
    Plot accuracy metrics over training iterations from CSV files.

    Reads all *metrics*.csv files in the given directory and plots
    top-1 accuracy curves for each dataset.

    Args:
        path: Directory containing metrics CSV files
        save_path: Path to save the plot (default: path/output.png)
        title: Plot title (default: "Accuracy over iterations")

    Expected CSV format:
        iteration,top1,top5
        100,45.2,72.1
        200,52.3,78.4
        ...
    """
    dataset_names = []
    iterations = []
    accuracies_top1 = []

    for csv_file in os.listdir(path):
        if csv_file.endswith(".csv") and "metrics" in csv_file:
            dataset_name = csv_file.replace(".csv", "").split("_")[-1]
            dataset_names.append(dataset_name)

            with open(os.path.join(path, csv_file), newline="") as f:
                reader = list(csv.DictReader(f))
                iterations.append([int(row["iteration"]) for row in reader])
                accuracies_top1.append([float(row["top1"]) for row in reader])

    if not dataset_names:
        print(f"No metrics CSV files found in {path}")
        return

    plt.figure(figsize=(10, 6))
    for i, dataset in enumerate(dataset_names):
        plt.plot(iterations[i], accuracies_top1[i], label=dataset, marker='o', markersize=3)

    plt.title(title or "Accuracy over iterations", fontsize=12)
    plt.legend()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Iterations")
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    output_path = save_path or os.path.join(path, "output.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_sequential_results(path, save_path=None, title=None, show_text=False):
    """
    Plot accuracy across sequential training stages.

    Traverses nested directories to find results CSV files and plots
    how accuracy changes across training stages.

    Args:
        path: Root directory containing nested result folders
        save_path: Path to save the plot
        title: Plot title
        show_text: Whether to show accuracy values as text labels

    Expected directory structure:
        path/
            results.csv
            stage1/
                results.csv
                stage2/
                    results.csv
    """
    stages = []
    datasets = []
    accuracies_top1 = []

    current_path = path
    while True:
        subdirs = [f for f in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, f))]

        for csv_file in os.listdir(current_path):
            if csv_file.endswith(".csv") and "results" in csv_file:
                stage_name = os.path.basename(current_path)
                stages.append(stage_name)

                with open(os.path.join(current_path, csv_file), newline="") as f:
                    reader = list(csv.DictReader(f))
                    if not datasets:
                        datasets = [row["dataset"] for row in reader]
                    accuracies_top1.append([float(row["top1"]) for row in reader])

        if not subdirs:
            break
        current_path = os.path.join(current_path, subdirs[0])

    if not stages:
        print(f"No results CSV files found in {path}")
        return

    plt.figure(figsize=(12, 6))
    for i, dataset in enumerate(datasets):
        accs = [accuracies_top1[j][i] for j in range(len(stages))]
        plt.plot(stages, accs, label=dataset, marker="o")

        if show_text:
            for j, acc in enumerate(accs):
                plt.text(j, acc + 1, f"{acc:.1f}%", ha='center', fontsize=6)

    plt.title(title or "Accuracy across training stages", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Training Stage")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = save_path or os.path.join(path, "output_all.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


# =============================================================================
# Model Comparison Functions
# =============================================================================

def compare_models(result_paths, save_path, baseline_path=None, top5=False, title=None):
    """
    Create a bar chart comparing multiple models across datasets.

    Args:
        result_paths: List of paths to results CSV files
        save_path: Path to save the comparison plot
        baseline_path: Optional path to baseline results for computing differences
        top5: If True, use top-5 accuracy instead of top-1
        title: Plot title

    Expected CSV format:
        dataset,top1,top5
        CIFAR10,85.2,97.1
        CIFAR100,62.3,85.4
        ...
    """
    model_names = []
    accuracies = [[] for _ in range(len(result_paths))]
    datasets = []
    acc_column = 2 if top5 else 1

    # Load baseline if provided
    baseline_accuracies = None
    if baseline_path is not None:
        baseline_accuracies = []
        with open(baseline_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                baseline_accuracies.append(float(row[acc_column]))

    # Load results from each model
    for i, path in enumerate(result_paths):
        model_name = os.path.basename(os.path.dirname(path))
        model_names.append(model_name)

        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if i == 0:
                    datasets.append(row[0])
                accuracies[i].append(float(row[acc_column]))

    # Create bar chart
    x = np.arange(len(datasets))
    num_models = len(model_names)
    width = 0.8 / num_models

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_accs in enumerate(accuracies):
        if baseline_accuracies is not None:
            model_accs = np.array(model_accs) - np.array(baseline_accuracies)
            metric = "top5" if top5 else "top1"
            print(f"{model_names[i]}: average {metric} difference = {np.mean(model_accs):.2f}%")

        ax.bar(x + i * width, model_accs, width=width, label=model_names[i])

    ax.set_xticks(x + width * (num_models - 1) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=min(3, num_models))

    ylabel = "Accuracy Difference (%)" if baseline_accuracies else "Accuracy (%)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plotting utilities for CLIP model analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # t-SNE from a dataset (recommended)
  python -m src.plot --tsne --dataset CIFAR100 --save-path tsne.png
  python -m src.plot --tsne --dataset DTD --include-text --save-path tsne.png

  # t-SNE from a dataset with finetuned model
  python -m src.plot --tsne --dataset ImageNet --model-path ckpt/model.pth --save-path tsne.png

  # t-SNE from pre-computed embeddings
  python -m src.plot --tsne --embeddings emb.npy --save-path tsne.png

  # t-SNE from images directory
  python -m src.plot --tsne --image-dir ./data/images --save-path tsne.png

  # Plot training metrics
  python -m src.plot --single --path ./ckpt/experiment/

  # Compare multiple models
  python -m src.plot --result-paths model1/results.csv model2/results.csv --save-path compare.png

Available datasets:
  CIFAR10, CIFAR100, DTD, MNIST, SUN397, Aircraft, Caltech101, EuroSAT,
  Flowers, Food, OxfordPet, StanfordCars, ImageNet, ImageNetA, ImageNetR,
  ImageNetSketch, ImageNetV2, and more.
        """
    )

    # General arguments
    parser.add_argument("--path", type=str, help="Path to directory for metrics/results")
    parser.add_argument("--save-path", type=str, help="Path to save output plot")
    parser.add_argument("--title", type=str, default=None, help="Plot title")

    # Mode selection
    parser.add_argument("--single", action="store_true", help="Plot training metrics from a single directory")
    parser.add_argument("--all", action="store_true", help="Plot sequential training results")
    parser.add_argument("--text", action="store_true", help="Show text labels on plot")

    # Model comparison
    parser.add_argument("--result-paths", nargs="+", type=str, help="Paths to results CSV files for comparison")
    parser.add_argument("--compare-original", type=str, help="Baseline results CSV for computing differences")
    parser.add_argument("--top5", action="store_true", help="Use top-5 accuracy instead of top-1")

    # t-SNE arguments
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE plot")
    parser.add_argument("--embeddings", type=str, help="Path to embeddings file (.npy or .pt)")
    parser.add_argument("--labels", type=str, help="Path to labels file (.npy or .pt)")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity (default: 30)")

    # t-SNE from model arguments
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint for embedding extraction")
    parser.add_argument("--model-name", type=str, default="ViT-B/32", help="CLIP architecture (default: ViT-B/32)")
    parser.add_argument("--image-dir", type=str, help="Directory containing images for t-SNE")
    parser.add_argument("--image-paths", nargs="+", type=str, help="List of image paths for t-SNE")
    parser.add_argument("--texts", nargs="+", type=str, help="List of texts to embed for t-SNE")
    parser.add_argument("--text-file", type=str, help="File with texts (one per line) for t-SNE")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model inference (default: cuda)")

    # t-SNE from dataset arguments
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., CIFAR100, DTD, ImageNet)")
    parser.add_argument("--data-location", type=str, default="./data", help="Root directory for datasets")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to process from dataset")
    parser.add_argument("--include-text", action="store_true", help="Include text embeddings from class names")

    args = parser.parse_args()

    if args.tsne:
        assert args.save_path is not None, "Must provide --save-path for t-SNE"

        # Priority: dataset > image_dir/image_paths/texts > embeddings
        if args.dataset is not None:
            # Use dataset loader (recommended)
            plot_tsne_from_dataset(
                dataset_name=args.dataset,
                model_path=args.model_path,
                model_name=args.model_name,
                data_location=args.data_location,
                device=args.device,
                split=args.split,
                max_samples=args.max_samples,
                include_text=args.include_text,
                save_path=args.save_path,
                title=args.title,
                perplexity=args.perplexity,
                batch_size=args.batch_size,
            )
        elif any([args.image_dir, args.image_paths, args.texts, args.text_file]):
            # Use image directory or paths
            plot_tsne_from_model(
                model_path=args.model_path,
                model_name=args.model_name,
                image_dir=args.image_dir,
                image_paths=args.image_paths,
                texts=args.texts,
                text_file=args.text_file,
                device=args.device,
                save_path=args.save_path,
                title=args.title or "t-SNE Visualization",
                perplexity=args.perplexity,
                batch_size=args.batch_size,
                max_images=args.max_images,
            )
        elif args.embeddings is not None:
            # Use pre-computed embeddings
            if args.embeddings.endswith(".npy"):
                embeddings = np.load(args.embeddings)
            elif args.embeddings.endswith(".pt"):
                embeddings = torch.load(args.embeddings)
            else:
                raise ValueError("Embeddings must be .npy or .pt file")

            labels = None
            if args.labels is not None:
                if args.labels.endswith(".npy"):
                    labels = np.load(args.labels)
                elif args.labels.endswith(".pt"):
                    labels = torch.load(args.labels)

            plot_tsne(
                embeddings=embeddings,
                labels=labels,
                title=args.title or "t-SNE Visualization",
                save_path=args.save_path,
                perplexity=args.perplexity,
            )
        else:
            raise ValueError(
                "Must provide one of: --dataset, --image-dir, --image-paths, --texts, --text-file, or --embeddings"
            )

    elif args.single:
        assert args.path is not None, "Must provide --path for single mode"
        plot_metrics(args.path, save_path=args.save_path, title=args.title)

    elif args.all:
        assert args.path is not None, "Must provide --path for all mode"
        plot_sequential_results(args.path, save_path=args.save_path, title=args.title, show_text=args.text)

    elif args.result_paths is not None:
        assert args.save_path is not None, "Must provide --save-path for model comparison"
        compare_models(
            args.result_paths,
            args.save_path,
            baseline_path=args.compare_original,
            top5=args.top5,
            title=args.title,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
