# Task 3 — Image Captioning AI 🖼️

A complete end-to-end image captioning pipeline for the **CodSoft AI Internship**.
Given any image, the system generates a natural language description by combining
computer vision (ResNet50) with natural language generation (LSTM decoder).

---

## Architecture

```
Image (224×224)
      │
      ▼
┌─────────────────────────────┐
│  EncoderCNN (ResNet50)      │  Pre-trained on ImageNet
│  → remove classification    │  Backbone frozen during warm-up
│    head                     │  Fine-tune last block later
│  → AdaptiveAvgPool → Linear │
└────────────┬────────────────┘
             │ (B, embed_dim)  ← fixed-size image feature vector
             ▼
┌─────────────────────────────┐
│  DecoderLSTM                │  Trained from scratch
│  word_embedding + image_feat│  Teacher forcing during training
│  → LSTM → Linear → logits   │  Greedy or Beam Search at inference
└────────────┬────────────────┘
             │
             ▼
    "a dog running in the park"
```

**Why ResNet50?** Deeper than VGG with fewer parameters. The skip connections
(residual blocks) make it easier to train and give richer feature representations.

**Why LSTM?** Proven, fast to train on moderate hardware, and well-understood.
A transformer decoder would give higher BLEU but requires significantly more
compute and data.

**Why inject the image feature at every LSTM step?** It keeps the decoder
"looking at" the image throughout generation, rather than relying only on the
initial hidden state. Simple and effective.

---

## Project Structure

```
image_captioning/
├── caption.py          ← main entry point (inference)
├── train.py            ← training loop
├── evaluate.py         ← BLEU scoring
├── models/
│   ├── encoder.py      ← ResNet50 CNN encoder
│   └── decoder.py      ← LSTM decoder + greedy/beam decoding
├── utils/
│   ├── vocabulary.py   ← word ↔ index mapping
│   └── dataset.py      ← Flickr8k dataset + transforms + collate
├── data/               ← put Flickr8k here (see below)
│   ├── Images/
│   └── captions.txt
├── checkpoints/        ← saved by train.py
│   ├── best_model.pth
│   └── vocab.pkl
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision nltk pillow pandas tqdm
```

### 2. Download the dataset

**Flickr8k** (recommended — smallest standard captioning dataset, ~1GB):

- Download from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k
- Or request directly from: https://illinois.edu/fb/sec/1713398

Unzip so your folder looks like:

```
data/
├── Images/          ← 8,091 .jpg files
└── captions.txt     ← header: image,caption (40,455 rows)
```

### 3. Download NLTK tokenizer data

```python
import nltk
nltk.download('punkt')
```

---

## Training

```bash
python train.py
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Path to Flickr8k folder |
| `--embed_dim` | `256` | Embedding dimension |
| `--hidden_dim` | `512` | LSTM hidden size |
| `--epochs` | `25` | Training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--fine_tune_after` | `10` | Enable CNN fine-tuning after N epochs |
| `--freq_threshold` | `5` | Min word frequency for vocabulary |

**Training strategy:**

- **Epochs 1–10:** CNN backbone is frozen, only the projection head and LSTM are trained.
  This is the warm-up phase — the LSTM learns to generate text without destabilising the
  pre-trained CNN features.
- **Epoch 11+:** The last ResNet block (layer4) is unfrozen and fine-tuned at a lower
  learning rate (lr × 0.1). This lets the CNN adapt its features specifically to caption-relevant
  visual concepts.
- A `ReduceLROnPlateau` scheduler halves the LR when validation loss stops improving.

**Expected training time:**

| Hardware | Time per epoch | Total (25 epochs) |
|---|---|---|
| GPU (RTX 3060) | ~3–5 min | ~75–125 min |
| CPU only | ~40–60 min | ~17–25 hours |

**Expected BLEU-4 after training:** 0.18–0.24 (competitive for this architecture on Flickr8k).

---

## Inference

### Caption a single image

```bash
python caption.py --image path/to/photo.jpg
```

### Caption with beam search (better quality)

```bash
python caption.py --image path/to/photo.jpg --beam_width 5
```

### Caption all images in a folder

```bash
python caption.py --image_dir path/to/images/ --beam_width 5
```

### Compare greedy vs beam side by side

```bash
python caption.py --image photo.jpg --compare
```

**Example output:**
```
Device: cuda

Model loaded — epoch 24, val_loss=2.8431, vocab=2984

Image : photo.jpg
--------------------------------------------------
Greedy: a dog is running through the grass
Beam  : a brown dog runs through a field of green grass
--------------------------------------------------
```

---

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth \
                   --data_dir data --split test
```

**Output example:**
```
── BLEU Scores ──────────────────────
  BLEU1: 0.6234  (62.34%)
  BLEU2: 0.4218  (42.18%)
  BLEU3: 0.2897  (28.97%)
  BLEU4: 0.1923  (19.23%)
─────────────────────────────────────
```

**What BLEU measures:** The fraction of n-grams in the generated caption that
also appear in the reference captions. BLEU-4 (4-gram overlap) is the standard
headline metric for image captioning.

---

## How decoding works

### Greedy decoding
At each step, pick the single highest-probability next token. Fast, but can
commit to locally-good choices that lead to globally poor sentences.

### Beam search (default: beam_width=5)
Maintain 5 candidate sequences in parallel. At each step, expand every
candidate by all vocabulary words, score with log-probability, keep the
top 5. Defers commitment until the end — consistently produces better captions.

---

## Tested on

- Python 3.10+
- PyTorch 2.x
- torchvision 0.15+

---
Built for CodSoft AI Internship — Task 3.
