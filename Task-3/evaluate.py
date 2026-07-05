"""
evaluate.py
BLEU score computation and evaluation utilities.

BLEU (Bilingual Evaluation Understudy) measures the n-gram overlap between
a generated caption and a set of reference captions. Despite being designed
for machine translation, it's the standard metric for image captioning.

BLEU-1 measures unigram precision (individual word matches).
BLEU-4 measures 4-gram precision (short phrase matches) — the main metric.

We use nltk's sentence_bleu with smoothing to handle cases where higher-order
n-gram counts are zero (common for short or imperfect captions).

Usage (standalone):
    python evaluate.py --checkpoint checkpoints/best_model.pth \
                       --data_dir data --split test
"""

import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import (
    sentence_bleu,
    corpus_bleu,
    SmoothingFunction,
)

from utils.vocabulary import Vocabulary
from utils.dataset    import Flickr8kDataset, get_transform, collate_fn
from models.encoder   import EncoderCNN
from models.decoder   import DecoderLSTM


# ─────────────────────────────────────────────────────────────────────────────
# BLEU HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(
    encoder,
    decoder,
    dataset: Flickr8kDataset,
    device: torch.device,
    beam_width: int = 5,
    max_samples: int = 500,
) -> dict:
    """Compute corpus BLEU-1 through BLEU-4 on a dataset split.

    Parameters
    ----------
    encoder     : trained EncoderCNN
    decoder     : trained DecoderLSTM
    dataset     : Flickr8kDataset (val or test split)
    device      : torch device
    beam_width  : beam search width (1 = greedy)
    max_samples : cap evaluation to this many images (for speed during training)

    Returns
    -------
    scores : dict with keys "bleu1", "bleu2", "bleu3", "bleu4"
    """
    encoder.eval()
    decoder.eval()

    smoothing = SmoothingFunction().method1

    # For corpus BLEU: lists of (references, hypothesis) per image
    all_references  = []   # list of list-of-list-of-tokens
    all_hypotheses  = []   # list of list-of-tokens

    # We iterate over unique images so each image's 5 captions are all
    # used as references simultaneously (the correct way to evaluate)
    seen_images = {}
    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        img_name = row["image"]
        if img_name in seen_images:
            continue
        seen_images[img_name] = idx
        if len(seen_images) >= max_samples:
            break

    transform = get_transform("val")

    from PIL import Image as PILImage
    import os

    for img_name, idx in seen_images.items():
        img_path = os.path.join(dataset.img_dir, img_name)
        image    = PILImage.open(img_path).convert("RGB")
        image    = transform(image).unsqueeze(0).to(device)   # (1, 3, 224, 224)

        with torch.no_grad():
            feat = encoder(image)                              # (1, embed_dim)
            if beam_width == 1:
                pred_ids = decoder.generate_greedy(
                    feat, sos_idx=Vocabulary.SOS_IDX, eos_idx=Vocabulary.EOS_IDX
                )
            else:
                pred_ids = decoder.generate_beam(
                    feat, beam_width=beam_width,
                    sos_idx=Vocabulary.SOS_IDX, eos_idx=Vocabulary.EOS_IDX
                )

        # Hypothesis: decoded token list (strings)
        hyp_tokens = dataset.vocab.decode(pred_ids).split()

        # References: all 5 ground-truth captions for this image
        ref_captions = dataset.get_all_captions_for_image(img_name)
        ref_tokens   = [
            Vocabulary.tokenize(cap)
            for cap in ref_captions
        ]

        all_hypotheses.append(hyp_tokens)
        all_references.append(ref_tokens)

    # Compute corpus BLEU for each n-gram order
    weights_map = {
        "bleu1": (1, 0, 0, 0),
        "bleu2": (0.5, 0.5, 0, 0),
        "bleu3": (0.33, 0.33, 0.33, 0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }

    scores = {}
    for name, weights in weights_map.items():
        scores[name] = corpus_bleu(
            all_references,
            all_hypotheses,
            weights         = weights,
            smoothing_function = smoothing,
        )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE EVALUATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--vocab_path", default="checkpoints/vocab.pkl")
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--split",      default="test", choices=["train","val","test"])
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--max_samples",type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    embed_dim  = ckpt.get("embed_dim",  256)
    hidden_dim = ckpt.get("hidden_dim", 512)
    num_layers = ckpt.get("num_layers", 1)

    encoder = EncoderCNN(embed_dim=embed_dim).to(device)
    decoder = DecoderLSTM(
        vocab_size  = len(vocab),
        embed_dim   = embed_dim,
        hidden_dim  = hidden_dim,
        num_layers  = num_layers,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Dataset
    dataset = Flickr8kDataset(
        root_dir  = args.data_dir,
        transform = get_transform("val"),
        vocab     = vocab,
        split     = args.split,
    )

    print(f"\nEvaluating on {args.split} split "
          f"(up to {args.max_samples} images, beam={args.beam_width})...")

    scores = compute_bleu(
        encoder, decoder, dataset, device,
        beam_width  = args.beam_width,
        max_samples = args.max_samples,
    )

    print("\n── BLEU Scores ──────────────────────")
    for name, val in scores.items():
        print(f"  {name.upper()}: {val:.4f}  ({val*100:.2f}%)")
    print("─────────────────────────────────────")


if __name__ == "__main__":
    main()
