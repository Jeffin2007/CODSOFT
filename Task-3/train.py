"""
train.py
Training loop for the image captioning model.

Usage:
    python train.py

The script will:
1. Load the Flickr8k dataset from data/
2. Build the vocabulary
3. Initialise the ResNet50 encoder + LSTM decoder
4. Train for --epochs epochs, saving the best checkpoint by validation loss
5. Print BLEU-4 on the validation set after each epoch
"""

import os
import sys
import math
import time
import argparse
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.vocabulary import Vocabulary
from utils.dataset    import Flickr8kDataset, get_transform, collate_fn
from models.encoder   import EncoderCNN
from models.decoder   import DecoderLSTM
from evaluate         import compute_bleu


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train image captioning model")
    p.add_argument("--data_dir",      default="data",           help="Path to Flickr8k data dir")
    p.add_argument("--checkpoint_dir",default="checkpoints",    help="Where to save model weights")
    p.add_argument("--embed_dim",     type=int, default=256,    help="Embedding dimension")
    p.add_argument("--hidden_dim",    type=int, default=512,    help="LSTM hidden size")
    p.add_argument("--num_layers",    type=int, default=1,      help="LSTM layers")
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--freq_threshold",type=int, default=5,      help="Vocab frequency cutoff")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--epochs",        type=int, default=25)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--fine_tune_after",type=int, default=10,
                   help="Enable CNN fine-tuning after this many epochs (0=never)")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  [checkpoint saved → {path}]")


def log(msg: str) -> None:
    print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    encoder, decoder, loader, criterion, optimizer, device, vocab_size
) -> float:
    """Run one full training epoch. Returns average loss."""
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (images, captions, lengths) in enumerate(loader):
        images   = images.to(device)
        captions = captions.to(device)

        # ── Forward pass ────────────────────────────────────────────────────
        img_features = encoder(images)                          # (B, embed_dim)

        # Input to decoder: all tokens except the last (we predict each next token)
        decoder_input  = captions[:, :-1]                       # (B, seq_len-1)
        # Target: all tokens except the first (shift left by 1)
        decoder_target = captions[:, 1:]                        # (B, seq_len-1)

        logits = decoder(img_features, decoder_input)           # (B, seq_len-1, V)

        # ── Loss: cross-entropy, ignoring <PAD> tokens ───────────────────────
        # Reshape for nn.CrossEntropyLoss: (B*T, V) and (B*T,)
        B, T, V = logits.shape
        loss = criterion(
            logits.reshape(B * T, V),
            decoder_target.reshape(B * T),
        )

        # ── Backward pass ────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients in RNNs
        nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        # Count non-padding tokens for accurate average loss
        non_pad = (decoder_target != 0).sum().item()
        total_loss   += loss.item() * non_pad
        total_tokens += non_pad

        if (batch_idx + 1) % 50 == 0:
            log(f"    batch {batch_idx+1}/{len(loader)}  "
                f"loss={loss.item():.4f}")

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def val_epoch(
    encoder, decoder, loader, criterion, device
) -> float:
    """Run validation. Returns average loss (no gradient updates)."""
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for images, captions, lengths in loader:
            images   = images.to(device)
            captions = captions.to(device)

            img_features   = encoder(images)
            decoder_input  = captions[:, :-1]
            decoder_target = captions[:, 1:]
            logits         = decoder(img_features, decoder_input)

            B, T, V = logits.shape
            loss = criterion(
                logits.reshape(B * T, V),
                decoder_target.reshape(B * T),
            )

            non_pad = (decoder_target != 0).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\nDevice: {device}")
    log(f"PyTorch: {torch.__version__}\n")

    # ── Dataset ─────────────────────────────────────────────────────────────
    log("Loading datasets...")
    train_dataset = Flickr8kDataset(
        root_dir        = args.data_dir,
        transform       = get_transform("train"),
        freq_threshold  = args.freq_threshold,
        split           = "train",
    )
    vocab = train_dataset.vocab
    log(f"  Vocabulary size : {len(vocab)}")
    log(f"  Training samples: {len(train_dataset)}")

    # Save vocabulary so caption.py can reload it without the dataset
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    val_dataset = Flickr8kDataset(
        root_dir       = args.data_dir,
        transform      = get_transform("val"),
        vocab          = vocab,
        split          = "val",
    )
    log(f"  Validation samples: {len(val_dataset)}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = device.type == "cuda",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    log("Building model...")
    encoder = EncoderCNN(embed_dim=args.embed_dim, fine_tune=False).to(device)
    decoder = DecoderLSTM(
        vocab_size  = len(vocab),
        embed_dim   = args.embed_dim,
        hidden_dim  = args.hidden_dim,
        num_layers  = args.num_layers,
        dropout     = args.dropout,
    ).to(device)

    encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    log(f"  Encoder trainable params: {encoder_params:,}")
    log(f"  Decoder trainable params: {decoder_params:,}\n")

    # ── Loss & Optimiser ─────────────────────────────────────────────────────
    # ignore_index=0 excludes <PAD> tokens from the loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX)

    # Optimise only the trainable parameters
    params = (
        list(filter(lambda p: p.requires_grad, encoder.parameters()))
        + list(decoder.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log("=" * 55)
    log("Starting training...")
    log("=" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        log(f"\nEpoch {epoch}/{args.epochs}")

        # Enable CNN fine-tuning after warm-up phase
        if args.fine_tune_after > 0 and epoch == args.fine_tune_after + 1:
            log("  [Enabling CNN fine-tuning (layer4)]")
            encoder.enable_fine_tuning()
            # Re-build optimizer to include newly unfrozen params
            params = (
                list(filter(lambda p: p.requires_grad, encoder.parameters()))
                + list(decoder.parameters())
            )
            optimizer = torch.optim.Adam(params, lr=args.lr * 0.1)

        train_loss = train_epoch(
            encoder, decoder, train_loader, criterion, optimizer, device, len(vocab)
        )
        val_loss = val_epoch(encoder, decoder, val_loader, criterion, device)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        log(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"perplexity={math.exp(min(val_loss, 10)):.2f}  "
            f"time={elapsed:.1f}s")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch"      : epoch,
                    "encoder"    : encoder.state_dict(),
                    "decoder"    : decoder.state_dict(),
                    "optimizer"  : optimizer.state_dict(),
                    "val_loss"   : val_loss,
                    "vocab_size" : len(vocab),
                    "embed_dim"  : args.embed_dim,
                    "hidden_dim" : args.hidden_dim,
                    "num_layers" : args.num_layers,
                },
                os.path.join(args.checkpoint_dir, "best_model.pth"),
            )

    log("\nTraining complete.")
    log(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
