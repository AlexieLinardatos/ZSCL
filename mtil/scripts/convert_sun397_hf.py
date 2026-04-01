"""
Convert HuggingFace SUN397 (1aurent/SUN397) Parquet download
into the ImageFolder structure that torchvision.datasets.SUN397 expects:

    {out_dir}/SUN397/<classname>/image_0001.jpg
    {out_dir}/SUN397/<classname>/image_0002.jpg
    ...

Usage:
    python convert_sun397_hf.py \
        --hf_dir  C:/Users/100746621/Documents/ZSCL/data/SUN397_hf \
        --out_dir C:/Users/100746621/Documents/ZSCL/data
"""

import argparse
import os
from pathlib import Path

from datasets import load_from_disk, load_dataset


def convert(hf_dir: str, out_dir: str):
    print(f"Loading dataset from {hf_dir} ...")
    try:
        ds = load_from_disk(hf_dir)
    except Exception:
        # fallback: load as parquet directly
        ds = load_dataset("parquet", data_dir=hf_dir, split="train")

    print(f"  {len(ds)} total samples")
    print(f"  columns: {ds.column_names}")

    # detect image and label columns
    img_col = next(c for c in ds.column_names if "image" in c.lower())
    lbl_col = next(c for c in ds.column_names if "label" in c.lower() and "name" not in c.lower())

    # get class names
    if hasattr(ds.features[lbl_col], "names"):
        class_names = ds.features[lbl_col].names
    else:
        class_names = sorted(set(ds[lbl_col]))

    print(f"  {len(class_names)} classes, image col='{img_col}', label col='{lbl_col}'")

    root = Path(out_dir) / "SUN397"
    root.mkdir(parents=True, exist_ok=True)

    counters = {}
    for i, sample in enumerate(ds):
        label_idx = sample[lbl_col]
        class_name = class_names[label_idx] if isinstance(label_idx, int) else label_idx
        # sanitise class name for filesystem
        class_name_safe = class_name.replace("/", "_").strip("_")
        class_dir = root / class_name_safe
        class_dir.mkdir(exist_ok=True)

        count = counters.get(class_name_safe, 0)
        counters[class_name_safe] = count + 1
        out_path = class_dir / f"image_{count:05d}.jpg"

        img = sample[img_col]
        if hasattr(img, "save"):
            img.save(out_path, "JPEG")
        else:
            # bytes
            with open(out_path, "wb") as f:
                f.write(img)

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(ds)} converted ...")

    print(f"\nDone. Images saved to: {root}")
    print(f"Total classes written: {len(counters)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dir",  required=True, help="Path to SUN397_hf download")
    parser.add_argument("--out_dir", required=True, help="Output root (SUN397/ folder created here)")
    args = parser.parse_args()
    convert(args.hf_dir, args.out_dir)
