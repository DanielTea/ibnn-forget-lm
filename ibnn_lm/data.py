# Copyright 2026. Apache License 2.0.
#
# Data fetching + a character-level tokenizer for the IBNN-LM harness.
#
# Everything here is deliberately dependency-light (stdlib urllib only) so a training run is
# fully self-contained and local: call `prepare(name)` to download/cache a corpus, build a
# `CharTokenizer` from it, and hand the encoded ids to the trainer. The tokenizer's vocabulary
# is serialized into the model checkpoint, so generation never needs the dataset again.

import os
import urllib.request
from dataclasses import dataclass

import torch

# Project-root-relative cache for downloaded corpora.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# name -> (url, description). All are small, public, char-level-friendly plain-text corpora.
DATASETS = {
    "tinyshakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt",
        "~1.1MB of Shakespeare, the classic char-level LM benchmark.",
    ),
    "tinystories": (
        "https://raw.githubusercontent.com/karpathy/llama2.c/master/data/"
        "tinystories_sample.txt",
        "A small sample of synthetic children's stories (simple vocabulary).",
    ),
    "shakespeare_full": (
        "https://www.gutenberg.org/files/100/100-0.txt",
        "~5.4MB: the complete works of Shakespeare (Gutenberg) - 5x tinyshakespeare.",
    ),
    "enwik8": (
        "http://mattmahoney.net/dc/enwik8.zip",
        "100MB of English Wikipedia (Hutter Prize) - the standard 'big' char-LM benchmark. "
        "Subset it with max_mb/--max_mb to keep local training tractable.",
    ),
}

# Registry entries whose URL is a .zip; value is the file name inside the archive.
_ZIP_MEMBER = {"enwik8": "enwik8"}

# Built-in offline fallback so the harness runs with zero network access.
SYNTH = (
    "the quick brown fox jumps over the lazy dog. " * 400
    + "pack my box with five dozen liquor jugs. " * 400
    + "how vexingly quick daft zebras jump! " * 400
    + "the five boxing wizards jump quickly. " * 400
)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url}\n       -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "ibnn-lm/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    with open(dest, "wb") as f:
        f.write(body)
    print(f"saved {len(body):,} bytes")


def prepare(name_or_path: str) -> str:
    """Return a path to a local plain-text corpus.

    `name_or_path` may be (1) a path to an existing .txt file, used as-is; (2) one of the
    keys in DATASETS, downloaded and cached under data/<name>/input.txt; or (3) the literal
    string "synth", which writes the built-in synthetic corpus to disk (no network).
    """
    if name_or_path != "synth" and os.path.isfile(name_or_path):
        return name_or_path

    if name_or_path == "synth":
        dest = os.path.join(DATA_DIR, "synth", "input.txt")
        if not os.path.isfile(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                f.write(SYNTH)
        return dest

    if name_or_path not in DATASETS:
        raise ValueError(
            f"unknown dataset '{name_or_path}'. Known: {list(DATASETS)}, 'synth', "
            f"or a path to a .txt file."
        )

    url, _ = DATASETS[name_or_path]

    # Zip-archived corpora (e.g. enwik8): download the archive once, extract the member.
    if name_or_path in _ZIP_MEMBER:
        member = _ZIP_MEMBER[name_or_path]
        dest_dir = os.path.join(DATA_DIR, name_or_path)
        dest = os.path.join(dest_dir, member)
        if not os.path.isfile(dest):
            import zipfile
            zip_path = os.path.join(dest_dir, member + ".zip")
            try:
                if not os.path.isfile(zip_path):
                    _download(url, zip_path)
                with zipfile.ZipFile(zip_path) as z:
                    z.extract(member, dest_dir)
            except Exception as e:  # noqa: BLE001
                print(f"download/extract failed ({e}); falling back to synthetic corpus.")
                return prepare("synth")
        return dest

    dest = os.path.join(DATA_DIR, name_or_path, "input.txt")
    if not os.path.isfile(dest):
        try:
            _download(url, dest)
        except Exception as e:  # noqa: BLE001 - any network failure falls back to synth
            print(f"download failed ({e}); falling back to built-in synthetic corpus.")
            return prepare("synth")
    return dest


class CharTokenizer:
    """A minimal, reversible character-level tokenizer.

    Vocabulary is the sorted set of characters seen in the training text. It round-trips
    through a plain dict so it can be stored inside (and restored from) a checkpoint.
    """

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str):
        # Unknown characters (only possible at generation time) are dropped.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def to_dict(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])


@dataclass
class Dataset:
    train: torch.Tensor       # 1-D long tensor of token ids
    val: torch.Tensor         # 1-D long tensor of token ids
    tokenizer: CharTokenizer


def load(name_or_path: str, train_frac: float = 1.0, val_split: float = 0.1,
         max_mb: float = 0.0, byte_level: bool = False) -> Dataset:
    """Download/prepare a corpus, tokenize it, and split into train/val id tensors.

    train_frac (in (0, 1]) optionally shrinks the *training* split only, for probing the
    data-efficiency claim. val_split is the fraction of the corpus held out for validation.
    max_mb (>0) truncates the corpus to the first max_mb megabytes BEFORE tokenizing, so the
    vocabulary and splits are consistent - useful to subset a large corpus like enwik8.
    byte_level reads the file as raw bytes (latin-1, a byte<->codepoint bijection) so the vocab
    is the <=256 distinct bytes. This is the standard setup for enwik8, whose char-level vocab
    would otherwise be thousands of rare Unicode codepoints.
    """
    path = prepare(name_or_path)
    n_bytes = int(max_mb * 1_000_000) if max_mb and max_mb > 0 else -1
    encoding = "latin-1" if byte_level else "utf-8"
    with open(path, "r", encoding=encoding, errors="replace") as f:
        text = f.read(n_bytes) if n_bytes > 0 else f.read()
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    n_train = int(len(ids) * (1.0 - val_split))
    train, val = ids[:n_train], ids[n_train:]
    if train_frac < 1.0:
        train = train[: max(1, int(len(train) * train_frac))]
    return Dataset(train=train, val=val, tokenizer=tok)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample a random (x, y) batch of next-token-prediction windows."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    if device == "cuda":
        # async host->device copy; pin for throughput
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
