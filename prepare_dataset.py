import os
import random
import shutil

source = "dataset"

os.makedirs("datasets/images/train", exist_ok=True)
os.makedirs("datasets/images/val", exist_ok=True)
os.makedirs("datasets/labels/train", exist_ok=True)
os.makedirs("datasets/labels/val", exist_ok=True)

images = [f for f in os.listdir(source) if f.endswith(".jpeg")]

random.shuffle(images)

split = int(len(images) * 0.8)

train_images = images[:split]
val_images = images[split:]

for img in train_images:
    txt = img.replace(".jpeg", ".txt")

    shutil.copy(os.path.join(source, img),
                os.path.join("datasets/images/train", img))

    shutil.copy(os.path.join(source, txt),
                os.path.join("datasets/labels/train", txt))

for img in val_images:
    txt = img.replace(".jpeg", ".txt")

    shutil.copy(os.path.join(source, img),
                os.path.join("datasets/images/val", img))

    shutil.copy(os.path.join(source, txt),
                os.path.join("datasets/labels/val", txt))

print("Dataset prepared successfully!")