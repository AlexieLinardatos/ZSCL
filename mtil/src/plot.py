import matplotlib.pyplot as plt
import csv
import os
from matplotlib.ticker import FormatStrFormatter
import argparse
import numpy as np
from sklearn.manifold import TSNE
import torch


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


def plot_metrics(path):
    x_label = []
    iterations = []
    accuracies_top1 = []
    accuracies_top5 = []
    

    for csv_file in os.listdir(path):
        if csv_file.endswith(".csv") and csv_file.find("metrics")!=-1:
            x_label.append(csv_file.split(".csv")[0].split("_")[-1])

            with open(path +"/"+ csv_file, newline="") as f:
                reader = list(csv.DictReader(f))

                iterations.append([int(row["iteration"]) for row in reader])
                accuracies_top1.append([float(row["top1"]) for row in reader])
                accuracies_top5.append([float(row["top5"]) for row in reader])
                # ZSCL_losses = [row["ZSCL"] for row in reader]
    
    # print(x_label)
    # print(iterations)
    # print(accuracies_top1)
    # print(accuracies_top5)
    # print(ZSCL_losses)

    index = 0
    for dataset in x_label:
        plt.plot(iterations[index], accuracies_top1[index], label=dataset)
        
        index+=1

    # plt.ylim(0,100)
    plt.title("accuracy over iterations\n \
            Trained on: DTD",
            fontsize=12)
    plt.legend()
    plt.ylabel("accuracy(%)")
    plt.xlabel("iterations")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    plt.savefig(path +"/"+ "output.png")

    # plt.plot(iterations, )

def plot_all(path, text = False):
    versions = []
    datasets = []
    accuracies_top1 = []
    accuracies_top5 = []

    index = 1
    while True:
        print(f"os.listdir {os.listdir(path)}")
        folder = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
        print(f"FOLDER:{folder}")
        if folder:
            folder = folder[0]
            print(f"folder:{folder}")
        
        for csv_file in os.listdir(path):
            if csv_file.endswith(".csv") and csv_file.find("results")!=-1:
                print(csv_file)
                versions.append(path.split("/")[-1])

                with open(path +"/"+ csv_file, newline="") as f:
                    reader = list(csv.DictReader(f))

                    datasets = [(row["dataset"]) for row in reader]
                    accuracies_top1.append([float(row["top1"]) for row in reader])
                    accuracies_top5.append([float(row["top5"]) for row in reader])
        index+=1
        if not folder:
            break
        path = os.path.join(path,folder)
    
    print(versions)
    versions[0] = "DTD"
    print(datasets)
    print(accuracies_top1)    
    print(versions)

    index = 0
    for dataset in datasets:
        plt.plot(versions, [accuracies_top1[i][index] for i in range(len(versions))], label=dataset, marker="o")
        
        if text:
            for i, v in enumerate(accuracies_top1[i][index] for i in range(len(versions))):
                plt.text(versions[i], v + 1, f"{v:.1f}%", ha='center', fontsize=6)
        index+=1

    # plt.ylim(0,100)
    plt.title("accuracy over iterations\n \
            Trained on: DTD",
            fontsize=12)
    plt.legend()
    plt.ylabel("accuracy(%)")
    plt.xlabel("versions")
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    plt.savefig(path +"/"+ "output_all.png")

def compare_models(result_paths, save_path, top5=False):
    model_names = []
    top1_accuracies = [[] for _ in range(len(result_paths))]
    datasets = []
    has_added_datasets = False 
    index = 0

    if args.compare_original is not None:
        original_top1_accuracies = []
        with open(args.compare_original, newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if top5:
                    original_top1_accuracies.append(float(row[2]))
                else:
                    original_top1_accuracies.append(float(row[1]))

    for path in result_paths:
        model_names.append(os.path.basename(os.path.dirname(path)))
        with open(path, newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if not has_added_datasets:
                    datasets.append(row[0])
                if top5:
                    top1_accuracies[index].append(float(row[2]))
                else:
                    top1_accuracies[index].append(float(row[1]))
            has_added_datasets = True
        index+=1


    x = np.arange(len(datasets)) 
    
    num_models = len(model_names)
    width = 0.8 / num_models 


    fig, ax = plt.subplots()

    for index, accuracies in enumerate(top1_accuracies):
        if args.compare_original is not None:
            accuracies = np.array(accuracies) - np.array(original_top1_accuracies)

            top_prompt = "top1"            
            if top5:
                top_prompt = "top5"
            print(f"{model_names[index]}, average {top_prompt} accuracy: {np.mean(accuracies)}")
        ax.bar(x + index * (width), accuracies, width=width, label=model_names[index])

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05))

    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_xticks(x + width * (num_models - 1) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha='right')

    # plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(save_path)

    print(top1_accuracies)
    # print(datasets)

                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--all", action="store_true", default=None)
    parser.add_argument("--single", action="store_true", default=None)
    parser.add_argument("--result-paths", nargs="+", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--compare-original", type=str, default=None)
    parser.add_argument("--top5", action="store_true", default=None)
    parser.add_argument("--text", action="store_true", default=None)

    # t-SNE arguments
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE plot")
    parser.add_argument("--embeddings", type=str, help="Path to embeddings file (.npy or .pt)")
    parser.add_argument("--labels", type=str, help="Path to labels file (.npy or .pt)")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--title", type=str, default="t-SNE Visualization", help="Plot title")

    args = parser.parse_args()

    # path = "./ckpt/DTD_finetune/"
    # path = "./ckpt/DTD_finetune/MNIST_finetune/EuroSAT_finetune/Aircraft_finetune"
    if args.tsne:
        assert args.embeddings is not None, "Must provide --embeddings for t-SNE"
        assert args.save_path is not None, "Must provide --save-path for t-SNE"

        # Load embeddings
        if args.embeddings.endswith(".npy"):
            embeddings = np.load(args.embeddings)
        elif args.embeddings.endswith(".pt"):
            embeddings = torch.load(args.embeddings)
        else:
            raise ValueError("Embeddings must be .npy or .pt file")

        # Load labels if provided
        labels = None
        if args.labels is not None:
            if args.labels.endswith(".npy"):
                labels = np.load(args.labels)
            elif args.labels.endswith(".pt"):
                labels = torch.load(args.labels)

        plot_tsne(
            embeddings=embeddings,
            labels=labels,
            title=args.title,
            save_path=args.save_path,
            perplexity=args.perplexity,
        )
    elif args.single:
        assert args.path is not None
        plot_metrics(args.path)
    elif args.all:
        assert args.path is not None
        if args.text is not None:
            plot_all(args.path, True)
        else:
            plot_all(args.path)
    elif args.result_paths is not None:
        assert args.save_path is not None
        if args.top5:
            compare_models(args.result_paths, args.save_path, True)
        else:
            compare_models(args.result_paths, args.save_path)