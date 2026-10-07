"""
Where does a run forget: in the image tower or the text tower?

For each old task t, a 2x2 swap between the checkpoint saved right after
training t (the snapshot) and the final checkpoint:

                          text = snapshot      text = final
    image = snapshot      diag (sanity)        image kept, text drifted
    image = final         text kept, image     last row (sanity)
                          drifted

Two cells reproduce numbers already in task_summary.csv, which checks the
pipeline. The other two attribute the forgetting:

    text part  = acc(final image, snapshot text) - acc(final, final)
    image part = acc(snapshot image, final text) - acc(final, final)

A large text part means old class-name embeddings drifted, which only L_rep
constrains today, and points at distillation that reaches the text tower. A
large image part points at the image encoder instead. The parts need not sum to
the total: the towers can drift together in a way neither swap undoes.

Eval only, no training. Classifier weights come from the repo's own
zeroshot_classifier (same templates and averaging as training-time eval).

Run:  python -m src.diag_text_drift --ckpt-dir ckpt/11task/remind_l2_m128_rdnone
"""

import argparse
import csv
import os

import torch

import clip.clip as clip
from src import datasets, utils
from src.models.evaluation import zeroshot_classifier
from src.datasets.common import maybe_dictionarize

ORDER = ["Aircraft", "Caltech101", "CIFAR100", "DTD", "EuroSAT", "Flowers",
         "Food", "MNIST", "OxfordPet", "StanfordCars", "SUN397"]


def load(model_name, path):
    model, _, val_preprocess = clip.load(model_name, jit=False)
    utils.torch_load(model, path)
    return model.cuda().eval(), val_preprocess


@torch.no_grad()
def image_features(model, loader):
    feats, labels = [], []
    for data in loader:
        data = maybe_dictionarize(data)
        f = model.encode_image(data["images"].cuda())
        feats.append(f / f.norm(dim=-1, keepdim=True))
        labels.append(data["labels"].cuda())
    return torch.cat(feats), torch.cat(labels)


def top1(feats, labels, weights):
    pred = (100.0 * feats @ weights).argmax(dim=1)
    return (pred == labels).float().mean().item() * 100


def read_summary(path):
    """task_summary.csv -> {row task name: {column: acc}}, or {} if missing."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["task_name"]: r for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--model", default="ViT-B/16")
    p.add_argument("--data-location", default="/scratch/alexie/data")
    p.add_argument("--batch-size-eval", type=int, default=64)
    args = p.parse_args()

    final_task = ORDER[-1]
    final, val_preprocess = load(
        args.model, os.path.join(args.ckpt_dir, f"{final_task}.pth"))
    summary = read_summary(os.path.join(args.ckpt_dir, "task_summary.csv"))

    rows = []
    for task in ORDER[:-1]:
        snap, _ = load(args.model, os.path.join(args.ckpt_dir, f"{task}.pth"))
        ds = getattr(datasets, task)(
            val_preprocess, location=args.data_location,
            batch_size=args.batch_size_eval, batch_size_eval=args.batch_size_eval,
        )
        w = {"snap": zeroshot_classifier(ds.classnames, ds.templates, snap),
             "final": zeroshot_classifier(ds.classnames, ds.templates, final)}
        f = {"snap": image_features(snap, ds.test_loader),
             "final": image_features(final, ds.test_loader)}

        acc = {(i, t): top1(f[i][0], f[i][1], w[t])
               for i in ("snap", "final") for t in ("snap", "final")}
        row = {
            "task": task,
            "diag": acc["snap", "snap"],
            "last": acc["final", "final"],
            "final_img_snap_text": acc["final", "snap"],
            "snap_img_final_text": acc["snap", "final"],
            "csv_diag": float(summary.get(task, {}).get(task, "nan")),
            "csv_last": float(summary.get(final_task, {}).get(task, "nan")),
        }
        row["forgetting"] = row["diag"] - row["last"]
        row["text_part"] = row["final_img_snap_text"] - row["last"]
        row["image_part"] = row["snap_img_final_text"] - row["last"]
        rows.append(row)
        print(f"[diag] {task:<13} diag={row['diag']:.2f} (csv {row['csv_diag']:.2f})  "
              f"last={row['last']:.2f} (csv {row['csv_last']:.2f})  "
              f"forgot={row['forgetting']:+.2f}  text={row['text_part']:+.2f}  "
              f"image={row['image_part']:+.2f}")
        del snap, f, w
        torch.cuda.empty_cache()

    out = os.path.join(args.ckpt_dir, "diag_text_drift.csv")
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mean = lambda k: sum(r[k] for r in rows) / len(rows)
    print(f"\n[diag] mean over {len(rows)} old tasks: forgetting={mean('forgetting'):+.2f}  "
          f"text part={mean('text_part'):+.2f}  image part={mean('image_part'):+.2f}")
    print("[diag] sanity: diag/last should match the csv columns to ~0.1; a "
          "mismatch means the checkpoint is not the evaluated model.")
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()
