"""
vocabulary.py
Builds and manages the word-to-index mapping for the caption vocabulary.

Usage:
    vocab = Vocabulary(freq_threshold=5)
    vocab.build(captions_list)
    idx = vocab.word2idx["dog"]
    word = vocab.idx2word[idx]
"""

from collections import Counter
import re


class Vocabulary:
    """Maps tokens <-> integer indices.

    Special tokens
    --------------
    <PAD>  idx 0  — padding to make batches uniform length
    <SOS>  idx 1  — start-of-sequence marker
    <EOS>  idx 2  — end-of-sequence marker
    <UNK>  idx 3  — rare words below freq_threshold
    """

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.word2idx: dict = {}
        self.idx2word: dict = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        specials = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for idx, token in enumerate(specials):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @staticmethod
    def tokenize(text: str) -> list:
        """Lowercase and split on non-alphanumeric characters."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def build(self, captions: list) -> None:
        """Build vocabulary from a list of raw caption strings."""
        counter = Counter()
        for caption in captions:
            counter.update(self.tokenize(caption))

        next_idx = len(self.word2idx)
        for word, freq in counter.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = next_idx
                self.idx2word[next_idx] = word
                next_idx += 1

    def encode(self, caption: str) -> list:
        """Caption string → list of integer indices (with SOS/EOS)."""
        tokens = self.tokenize(caption)
        indices = [self.SOS_IDX]
        indices += [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        indices.append(self.EOS_IDX)
        return indices

    def decode(self, indices: list, skip_special: bool = True) -> str:
        """List of integer indices → caption string."""
        skip = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX} if skip_special else set()
        words = [
            self.idx2word.get(i, self.UNK_TOKEN)
            for i in indices
            if i not in skip
        ]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, freq_threshold={self.freq_threshold})"
