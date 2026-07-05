"""
caption.py
End-to-end image captioning inference — the main entry point for users.

Usage:
    # Caption a single image (greedy decoding)
    python caption.py --image path/to/photo.jpg

    # Caption with beam search (better quality, slightly slower)
    python caption.py --image path/to/photo.jpg --beam_width 5

    # Caption all images in a folder
    python caption.py --image_dir path/to/images/ --beam_width 5

    # Show both greedy and beam results side by side
    python caption.py --image photo.jpg --compare

Prerequisites:
    Run train.py first to generate checkpoints/best_model.pth and
    checkpoints/vocab.pkl, or download a pre-trained checkpoint.
"""

import os
import sys
import argparse
import pickle
import glob

import torch
from PIL import Image

from utils.vocabulary import Vocabulary
from utils.dataset    import get_transform
from models.encoder   import EncoderCNN
from models.decoder   import DecoderLSTM


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, vocab_path: str, device: torch.device):
    """Load encoder, decoder, and vocabulary from disk.

    Returns
    -------
    encoder, decoder, vocab
    """
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at {checkpoint_path}")
        print("Run train.py first to generate a checkpoint.")
        sys.exit(1)

    if not os.path.exists(vocab_path):
        print(f"ERROR: vocabulary not found at {vocab_path}")
        sys.exit(1)

    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab: Vocabulary = pickle.load(f)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    embed_dim  = ckpt.get("embed_dim",  256)
    hidden_dim = ckpt.get("hidden_dim", 512)
    num_layers = ckpt.get("num_layers", 1)

    encoder = EncoderCNN(embed_dim=embed_dim, fine_tune=False).to(device)
    decoder = DecoderLSTM(
        vocab_size  = len(vocab),
        embed_dim   = embed_dim,
        hidden_dim  = hidden_dim,
        num_layers  = num_layers,
        dropout     = 0.0,   # no dropout at inference time
    ).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    print(f"Model loaded — epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}, "
          f"vocab={len(vocab)}")
    return encoder, decoder, vocab


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE IMAGE CAPTIONING
# ─────────────────────────────────────────────────────────────────────────────

def caption_image(
    image_path: str,
    encoder,
    decoder,
    vocab: Vocabulary,
    device: torch.device,
    beam_width: int = 5,
) -> dict:
    """Generate a caption for a single image file.

    Parameters
    ----------
    image_path : path to a JPEG/PNG image
    beam_width : 1 = greedy decoding, >1 = beam search

    Returns
    -------
    dict with keys:
        "greedy" : greedy-decoded caption string
        "beam"   : beam-search caption string (only if beam_width > 1)
        "image"  : image path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = get_transform("val")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)    # (1, 3, 224, 224)

    result = {"image": image_path}

    with torch.no_grad():
        features = encoder(image_tensor)                        # (1, embed_dim)

        # Greedy decoding — fast, reasonable quality
        greedy_ids = decoder.generate_greedy(
            features,
            sos_idx = Vocabulary.SOS_IDX,
            eos_idx = Vocabulary.EOS_IDX,
        )
        result["greedy"] = vocab.decode(greedy_ids)

        # Beam search — better quality (skip if beam_width == 1)
        if beam_width > 1:
            beam_ids = decoder.generate_beam(
                features,
                beam_width = beam_width,
                sos_idx    = Vocabulary.SOS_IDX,
                eos_idx    = Vocabulary.EOS_IDX,
            )
            result["beam"] = vocab.decode(beam_ids)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate captions for images")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image",     help="Path to a single image file")
    g.add_argument("--image_dir", help="Path to a folder of images")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--vocab_path", default="checkpoints/vocab.pkl")
    p.add_argument("--beam_width", type=int, default=5,
                   help="Beam search width. 1 = greedy decoding.")
    p.add_argument("--compare", action="store_true",
                   help="Show greedy vs beam captions side by side")
    return p.parse_args()


def print_result(result: dict, compare: bool = False) -> None:
    """Pretty-print a captioning result."""
    print(f"\nImage : {result['image']}")
    print("-" * 50)
    if compare and "beam" in result:
        print(f"Greedy: {result['greedy']}")
        print(f"Beam  : {result['beam']}")
    elif "beam" in result:
        print(f"Caption: {result['beam']}")
    else:
        print(f"Caption: {result['greedy']}")
    print("-" * 50)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    encoder, decoder, vocab = load_model(
        args.checkpoint, args.vocab_path, device
    )

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths = sorted(image_paths)
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            sys.exit(1)
        print(f"Found {len(image_paths)} images in {args.image_dir}")

    # Caption each image
    for path in image_paths:
        try:
            result = caption_image(
                path, encoder, decoder, vocab, device,
                beam_width = args.beam_width,
            )
            print_result(result, compare=args.compare)
        except Exception as e:
            print(f"  ERROR on {path}: {e}")


if __name__ == "__main__":
    main()
