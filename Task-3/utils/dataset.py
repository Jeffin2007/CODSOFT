"""
dataset.py
PyTorch Dataset for Flickr8k (and compatible Flickr30k).

Flickr8k structure expected on disk:
    data/
        Images/          <- all .jpg images
        captions.txt     <- format: "image_name.jpg,caption text"
                            (first line is header: image,caption)

The dataset returns (image_tensor, caption_tensor) pairs.
Each image has 5 reference captions in Flickr8k; we treat each as an
independent training sample (so 8k images → ~40k training pairs).
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from utils.vocabulary import Vocabulary


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transform(split: str = "train") -> transforms.Compose:
    """Return image transforms appropriate for training or inference.

    ResNet50 was pre-trained on ImageNet with images normalised to these
    mean/std values, so we must apply the same normalisation.
    Train split adds random horizontal flip and crop for data augmentation.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:   # val / test / inference
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Pad caption sequences to the length of the longest in the batch.

    PyTorch's DataLoader requires all tensors in a batch to have the same
    shape. Captions have variable length, so we pad shorter ones with
    <PAD> (index 0) on the right.

    Returns
    -------
    images   : (B, 3, 224, 224) float tensor
    captions : (B, max_seq_len) long tensor  — padded
    lengths  : (B,) list of actual caption lengths (before padding)
    """
    images, captions = zip(*batch)

    images = torch.stack(images, dim=0)                           # (B, C, H, W)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, (cap, length) in enumerate(zip(captions, lengths)):
        padded[i, :length] = cap

    return images, padded, lengths


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class Flickr8kDataset(Dataset):
    """Flickr8k image-caption dataset.

    Parameters
    ----------
    root_dir      : path to the data/ folder containing Images/ and captions.txt
    captions_file : filename of the CSV captions file (default: captions.txt)
    transform     : torchvision transform pipeline for images
    vocab         : pre-built Vocabulary instance (pass None to build fresh)
    freq_threshold: only used when vocab=None; passed to Vocabulary.__init__
    split         : "train" | "val" | "test"
    split_ratios  : (train_frac, val_frac) — test gets the remainder
    """

    def __init__(
        self,
        root_dir: str,
        captions_file: str = "captions.txt",
        transform=None,
        vocab: Vocabulary = None,
        freq_threshold: int = 5,
        split: str = "train",
        split_ratios: tuple = (0.80, 0.10),
    ):
        self.root_dir  = root_dir
        self.img_dir   = os.path.join(root_dir, "Images")
        self.transform = transform or get_transform(split)
        self.split     = split

        # ── Load and parse captions file ────────────────────────────────────
        df = pd.read_csv(
            os.path.join(root_dir, captions_file),
            header=0,
        )
        # Normalise column names robustly
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={df.columns[0]: "image", df.columns[1]: "caption"})
        df["image"]   = df["image"].str.strip()
        df["caption"] = df["caption"].str.strip()

        # ── Train / val / test split (by unique image, not by row) ──────────
        unique_images = df["image"].unique()
        n = len(unique_images)
        n_train = int(n * split_ratios[0])
        n_val   = int(n * split_ratios[1])

        if split == "train":
            selected = set(unique_images[:n_train])
        elif split == "val":
            selected = set(unique_images[n_train : n_train + n_val])
        else:
            selected = set(unique_images[n_train + n_val :])

        self.df = df[df["image"].isin(selected)].reset_index(drop=True)

        # ── Build or adopt vocabulary ────────────────────────────────────────
        if vocab is None:
            # Always build from the FULL dataset so val/test can decode properly
            all_captions = df["caption"].tolist()
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build(all_captions)
        else:
            self.vocab = vocab

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        row     = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Encode caption: [SOS, w1, w2, ..., EOS]
        caption = torch.tensor(self.vocab.encode(row["caption"]), dtype=torch.long)

        return image, caption

    # ------------------------------------------------------------------
    def get_all_captions_for_image(self, image_name: str) -> list:
        """Return all 5 reference captions for a given image filename."""
        rows = self.df[self.df["image"] == image_name]
        return rows["caption"].tolist()
