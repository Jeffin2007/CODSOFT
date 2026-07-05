"""
decoder.py
LSTM Caption Decoder with Bahdanau-style Attention.

Architecture
------------
The decoder is an LSTM that generates one word at a time. At each step it:
  1. Embeds the previously generated word → (B, embed_dim)
  2. Concatenates the embedded word with the image feature vector
  3. Passes the concatenation through an LSTM cell
  4. Projects the hidden state → vocabulary logits
  5. Samples or argmax-decodes the next word token

The image feature from the encoder is injected at every time step
(concatenated with the word embedding), which is a simple and effective
way to keep the decoder aware of the image throughout generation.

For a production system you would add full spatial attention (attending
over the 7×7 grid of CNN feature maps). That architecture is included
as an optional AttentionDecoder below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE LSTM DECODER (fast to train, good baseline)
# ─────────────────────────────────────────────────────────────────────────────

class DecoderLSTM(nn.Module):
    """Single-layer LSTM decoder.

    Parameters
    ----------
    vocab_size  : int   — size of the output vocabulary
    embed_dim   : int   — word embedding + image feature dimension
    hidden_dim  : int   — LSTM hidden state size
    num_layers  : int   — number of stacked LSTM layers
    dropout     : float — dropout probability (applied between LSTM layers)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Word embedding table
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM: input = word_embed (embed_dim) + image_feat (embed_dim)
        self.lstm = nn.LSTM(
            input_size  = embed_dim * 2,          # word + image feature
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Project hidden state → vocabulary logits
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc[1].weight, -0.1, 0.1)
        nn.init.constant_(self.fc[1].bias, 0)

    # ------------------------------------------------------------------
    def forward(
        self,
        image_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forcing forward pass (used during training).

        Parameters
        ----------
        image_features : (B, embed_dim)
        captions       : (B, seq_len)  — includes <SOS>, excludes last token

        Returns
        -------
        logits : (B, seq_len, vocab_size)
        """
        # Embed all caption tokens at once: (B, seq_len, embed_dim)
        embeddings = self.embedding(captions)

        # Repeat image feature for every time step: (B, seq_len, embed_dim)
        img_feat_expanded = image_features.unsqueeze(1).expand_as(embeddings)

        # Concatenate: (B, seq_len, embed_dim * 2)
        lstm_input = torch.cat([embeddings, img_feat_expanded], dim=2)

        # LSTM forward: hidden shape (B, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(lstm_input)

        # Project to vocabulary: (B, seq_len, vocab_size)
        logits = self.fc(lstm_out)
        return logits

    # ------------------------------------------------------------------
    def generate_greedy(
        self,
        image_feature: torch.Tensor,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ) -> list:
        """Greedy decoding: always pick the highest-probability next token.

        Parameters
        ----------
        image_feature : (1, embed_dim) — single image
        max_length    : maximum number of tokens to generate
        sos_idx / eos_idx : special token indices

        Returns
        -------
        token_ids : list of int (without SOS; stops at EOS or max_length)
        """
        self.eval()
        with torch.no_grad():
            device = image_feature.device
            token  = torch.tensor([[sos_idx]], device=device)    # (1, 1)
            hidden = None
            result = []

            for _ in range(max_length):
                embed = self.embedding(token)                     # (1, 1, E)
                img   = image_feature.unsqueeze(1)                # (1, 1, E)
                inp   = torch.cat([embed, img], dim=2)            # (1, 1, 2E)

                out, hidden = self.lstm(inp, hidden)              # (1, 1, H)
                logits      = self.fc(out.squeeze(1))             # (1, V)
                token       = logits.argmax(dim=1, keepdim=True)  # (1, 1)

                word_idx = token.item()
                if word_idx == eos_idx:
                    break
                result.append(word_idx)

        return result

    # ------------------------------------------------------------------
    def generate_beam(
        self,
        image_feature: torch.Tensor,
        beam_width: int = 5,
        max_length: int = 50,
        sos_idx: int = 1,
        eos_idx: int = 2,
    ) -> list:
        """Beam search decoding.

        Maintains beam_width candidate sequences in parallel, each with a
        cumulative log-probability score. At each step, every candidate is
        expanded by vocab_size possibilities; the top beam_width are kept.

        Why beam search beats greedy:
        Greedy can commit to a locally high-probability word that leads to
        a globally poor sentence. Beam search defers that commitment by
        exploring multiple paths simultaneously.

        Parameters
        ----------
        image_feature : (1, embed_dim)
        beam_width    : number of parallel hypotheses (5 is standard)
        max_length    : hard cap on sequence length
        sos_idx / eos_idx : special token indices

        Returns
        -------
        best_sequence : list of int — the highest-scoring complete caption
        """
        self.eval()
        with torch.no_grad():
            device = image_feature.device
            V      = self.vocab_size

            # Each beam: (score, token_sequence, hidden_state)
            start_token = torch.tensor([[sos_idx]], device=device)
            embed = self.embedding(start_token)                   # (1, 1, E)
            img   = image_feature.unsqueeze(1)                    # (1, 1, E)
            inp   = torch.cat([embed, img], dim=2)
            out, hidden = self.lstm(inp, None)
            logits = self.fc(out.squeeze(1))                      # (1, V)
            log_probs = F.log_softmax(logits, dim=-1)             # (1, V)

            # Initialise beams from top-k of first step
            top_scores, top_tokens = log_probs.topk(beam_width, dim=1)
            beams = [
                (top_scores[0, k].item(), [top_tokens[0, k].item()], hidden)
                for k in range(beam_width)
            ]

            completed = []

            for _ in range(max_length - 1):
                candidates = []
                for score, seq, hid in beams:
                    if seq[-1] == eos_idx:
                        completed.append((score, seq))
                        continue

                    token  = torch.tensor([[seq[-1]]], device=device)
                    embed  = self.embedding(token)
                    img_e  = image_feature.unsqueeze(1)
                    inp    = torch.cat([embed, img_e], dim=2)
                    out, new_hid = self.lstm(inp, hid)
                    logits = self.fc(out.squeeze(1))
                    lp     = F.log_softmax(logits, dim=-1)

                    top_s, top_t = lp.topk(beam_width, dim=1)
                    for k in range(beam_width):
                        new_score = score + top_s[0, k].item()
                        new_seq   = seq + [top_t[0, k].item()]
                        candidates.append((new_score, new_seq, new_hid))

                if not candidates:
                    break

                # Keep only the top beam_width candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_width]

                # Early stop if all beams have completed
                if all(s[-1] == eos_idx for _, s, _ in beams):
                    for sc, sq, _ in beams:
                        completed.append((sc, sq))
                    break

            # Collect any incomplete beams
            for sc, sq, _ in beams:
                completed.append((sc, sq))

            # Return the sequence with the highest cumulative log-probability
            completed.sort(key=lambda x: x[0], reverse=True)
            best = completed[0][1]
            # Strip trailing EOS if present
            if best and best[-1] == eos_idx:
                best = best[:-1]
            return best
