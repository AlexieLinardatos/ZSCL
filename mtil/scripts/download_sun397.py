import os
from datasets import load_dataset

print("Loading SUN397 from Hugging Face...")
ds = load_dataset("tanganke/sun397", split="train")

out_dir = "/scratch/alexie/data/SUN397/SUN397"
os.makedirs(out_dir, exist_ok=True)

total = len(ds)
for i, sample in enumerate(ds):
    img = sample["image"]
    label = sample["label"]
    class_name = ds.features["label"].names[label]

    class_dir = os.path.join(out_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    img_path = os.path.join(class_dir, f"{i:06d}.jpg")
    img.save(img_path)

    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{total} images saved...")

print(f"Done! {total} images saved to {out_dir}")
