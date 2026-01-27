import clip

import os
import csv
import torch
import gc
from tqdm import tqdm
import tkinter

from .. import datasets
from ..datasets.common import get_dataloader, maybe_dictionarize
from PIL import Image
from .probes import ProbeLayer, EncoderProbes




def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def load_probes_from_checkpoint(checkpoint_path, model):
    """Load probe layers from a checkpoint if they exist."""
    if checkpoint_path is None:
        return None

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if "probes_state_dict" not in checkpoint:
        return None

    # Get embedding dimension from model
    if hasattr(model, "module"):
        m = model.module
    else:
        m = model

    if hasattr(m.visual, 'output_dim'):
        embed_dim = m.visual.output_dim
    else:
        embed_dim = m.visual.proj.shape[1]

    probes = EncoderProbes(embed_dim)
    probes.load_state_dict(checkpoint["probes_state_dict"])
    probes = probes.cuda()
    probes.eval()
    print(f"[Evaluation] Loaded probe layers with embedding dim: {embed_dim}")
    return probes


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model, probes=None):
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    for classname in classnames:
        texts = [template(classname) for template in templates]  # format with class
        texts = clip.tokenize(texts).cuda()  # tokenize
        class_embeddings = model.encode_text(texts)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

        # Apply text probe if available
        if probes is not None:
            class_embeddings = probes.forward_text(class_embeddings)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

        del class_embeddings, class_embedding
        torch.cuda.empty_cache()

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights, args=None, probes=None):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):
        if args is not None:
            if args.max_evaluation_size is not None:
                if i >= args.max_evaluation_size:
                    break
        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Apply image probe if available
        if probes is not None:
            image_features = probes.forward_image(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = 100.0 * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier, dataset, args, probes=None):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()

    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model, probes=probes
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights, probes=probes)

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

    return top1, top5


def evaluate(image_classifier, args, val_preprocess):
    if args.eval_datasets is None:
        return

    # Load probes if they exist in checkpoint
    probes = None
    if args.added_layer and args.load is not None:
        probes = load_probes_from_checkpoint(args.load, image_classifier)

    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        top1, top5 = eval_single_dataset(image_classifier, dataset, args, probes=probes)

        path = os.path.dirname(args.load) + "/" + "evaluate_all_results.csv"


        row = {
                "dataset": dataset_name,
                "top1": top1,
                "top5": top5,
            }

        with open(path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset","top1","top5"])
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)



def eval_single_image(image_classifier, args, val_preprocess):
    if args.eval_single is None:
        image_path = input("Enter a path to a image: ")
    else:
        image_path = args.eval_single

    if args.class_names is None:
        return

    #load model
    model = image_classifier
    model.eval()

    # Load probes if they exist in checkpoint
    probes = None
    if args.added_layer and args.load is not None:
        probes = load_probes_from_checkpoint(args.load, model)

    #loading image
    image = val_preprocess(Image.open(image_path)).unsqueeze(0).to("cuda")

    #loading class names
    with open(args.class_names, "r") as file:
        lines = file.readlines()

    class_names = [line.strip() for line in lines]

    prompts = ""
    if args.prompt is not None:
        prompts = [f"{args.prompt} {name}" for name in class_names]
    else:
        prompts = [f"a photo of a {name}" for name in class_names]

    text = clip.tokenize(prompts).to("cuda")

    with torch.no_grad():
        image_feat = model.encode_image(image)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        text_feat = model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # Apply probes if available
        if probes is not None:
            image_feat = probes.forward_image(image_feat)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = probes.forward_text(text_feat)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # Compute logits manually with probed features
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_feat @ text_feat.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print(probs.tolist()[0])
    result = dict(zip(prompts, probs.tolist()[0]))
    result = dict(sorted(result.items(), key=lambda item: item[1]))

    if args.save:
        with open(os.path.join(args.save, os.path.splitext(os.path.basename(args.eval_single))[0]) + ".txt", "w") as file:
            for key, value in result.items():
                file.write(f"{key}: {value:.6f}\n")
    else:
        print(f"label probabilities: {result}")
    


def eval_single_dataset_2(image_classifier, dataset, args, probes=None):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()

    if hasattr(model, "module"):
        model = model.module

    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model, probes=probes
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights, args=args, probes=probes)
    print(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-5 accuracy: {top5:.2f}")
    del dataloader, zeroshot_weights
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "top1": top1,
        "top5": top5
        }

def evaluate_2(image_classifier, args, val_preprocess):
    result = []

    if args.eval_datasets is None:
        return

    # Load probes if they exist in checkpoint
    probes = None
    if args.added_layer and args.load is not None:
        probes = load_probes_from_checkpoint(args.load, image_classifier)

    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        tops = eval_single_dataset_2(image_classifier, dataset, args, probes=probes)
        dict = {"dataset_name": dataset_name, "metrics": tops}

        result.append(dict)

        del dataset, tops, dataset_class
        torch.cuda.empty_cache()
        gc.collect()
    return result
