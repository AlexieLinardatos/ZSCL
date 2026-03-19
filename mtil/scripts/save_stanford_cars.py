"""
Converts HuggingFace tanganke/stanford_cars cache to ImageFolder format.
Run locally in PowerShell:
  python scripts/save_stanford_cars.py
"""
import os
from datasets import load_dataset

CACHE_DIR = "C:/Users/100746621/Downloads/stanford_cars"
OUT_DIR = "C:/Users/100746621/Downloads/stanford_cars_images"

print("Loading from local cache...")
train_ds = load_dataset("tanganke/stanford_cars", split="train", cache_dir=CACHE_DIR)
test_ds  = load_dataset("tanganke/stanford_cars", split="test",  cache_dir=CACHE_DIR)

label_names = train_ds.features["label"].names

for split_name, ds in [("train", train_ds), ("test", test_ds)]:
    print(f"Saving {split_name} ({len(ds)} images)...")
    for i, sample in enumerate(ds):
        img   = sample["image"]
        label = sample["label"]
        cls   = label_names[label].replace("/", "_").replace(" ", "_")
        folder = os.path.join(OUT_DIR, split_name, cls)
        os.makedirs(folder, exist_ok=True)
        img.save(os.path.join(folder, f"{i:06d}.jpg"))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(ds)}")

print(f"Done! Saved to {OUT_DIR}")
print(f"Train: {len(train_ds)} images | Test: {len(test_ds)} images | Classes: {len(label_names)}")
