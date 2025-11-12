import os
import shutil

# Path to your flat folder of images
src_dir = "./data/ImageNet/val"

# Where you want to store the organized dataset
dst_dir = "./data/ImageNet/val"

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if not filename.lower().endswith((".jpeg", ".jpg", ".png")):
        continue  # skip non-image files

    # Get class name from prefix before first "_"
    class_name = filename.split("_")[2]   # e.g. 'n01440764'

    # Make class folder
    class_dir = os.path.join(dst_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Move image into class folder
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(class_dir, filename)

    shutil.move(src_path, dst_path)

print("✅ Done! Images organized by class.")
